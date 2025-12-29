#legacy.image_manager.py
# --- required imports for this module ---
import os
import time
import gzip
from io import BytesIO
from typing import Optional, Dict
import datetime
from datetime import timezone
import numpy as np
from PIL import Image
import tifffile as tiff

# add this near your other optional imports
from astropy.io import fits
try:
    from astropy.io.fits.verify import VerifyError
except Exception:
    # Fallback for older Astropy – we'll just treat it as a generic Exception
    class VerifyError(Exception):
        pass

def _drop_invalid_cards(header: fits.Header) -> fits.Header:
    """
    Return a copy of the FITS header with any cards that raise VerifyError removed.
    This prevents 'Unparsable card (FOO)' from blowing up later on .value access.
    """
    if not isinstance(header, fits.Header):
        return header

    hdr = header.copy()
    bad_keys = []
    for card in list(hdr.cards):
        try:
            # Accessing .value is what triggers VerifyError for bad cards
            _ = card.value
        except VerifyError as e:
            print(f"[ImageManager] Dropping invalid FITS card {card.keyword!r}: {e}")
            bad_keys.append(card.keyword)

    for key in bad_keys:
        try:
            del hdr[key]
        except Exception:
            pass

    return hdr



try:
    import rawpy

except Exception:
    rawpy = None  # optional; RAW loading will raise if it's None

from setiastro.saspro.xisf import XISF

from PyQt6.QtCore import QObject, pyqtSignal

def _looks_like_xisf_header(hdr) -> bool:
    try:
        if isinstance(hdr, (fits.Header, dict)):
            for k in hdr.keys():
                if isinstance(k, str) and k.startswith("XISF:"):
                    return True
    except Exception:
        pass
    return False

def _iter_header_items(hdr):
    """Yield (key, value) safely from fits.Header or dict; else yield nothing."""
    if isinstance(hdr, fits.Header):
        # .items() is supported and yields (key, value)
        for kv in hdr.items():
            yield kv
    elif isinstance(hdr, dict):
        for kv in hdr.items():
            yield kv

class ImageManager(QObject):
    """
    Manages multiple image slots with associated metadata and supports undo/redo operations for each slot.
    Emits a signal whenever an image or its metadata changes.
    """
    
    # Signal emitted when an image or its metadata changes.
    # Parameters:
    # - slot (int): The slot number.
    # - image (np.ndarray): The new image data.
    # - metadata (dict): Associated metadata for the image.
    image_changed = pyqtSignal(int, np.ndarray, dict)
    current_slot_changed = pyqtSignal(int)    
    # Keys we always carry forward unless caller explicitly supplies a non-empty replacement
    PRESERVE_META_KEYS = ("file_path", "FILE", "path", "fits_header", "header")


    def __init__(self, max_slots=5, parent=None):
        """
        Initializes the ImageManager with a specified number of slots.
        
        :param max_slots: Maximum number of image slots to manage.
        """
        super().__init__()
        self.parent = parent
        self.max_slots = max_slots
        self._images = {i: None for i in range(max_slots)}
        self._metadata = {i: {} for i in range(max_slots)}
        self._undo_stacks = {i: [] for i in range(max_slots)}
        self._redo_stacks = {i: [] for i in range(max_slots)}
        self.current_slot = 0  # Default to the first slot
        self.active_previews = {}  # Track active preview windows by slot
        self.mask_manager = MaskManager(max_slots)  # Add a MaskManager

    def _looks_like_path(self, v: object) -> bool:
        if not isinstance(v, str):
            return False
        # treat as path if it has a separator or a known extension
        ext_ok = v.lower().endswith((".fits", ".fit", ".fts", ".fz", ".fits.fz"))
        return (os.path.sep in v) or ext_ok

    def _attach_step_name(self, merged_meta: Dict, step_name: Optional[str]) -> Dict:
        if step_name is not None and str(step_name).strip():
            merged_meta["step_name"] = step_name.strip()
        return merged_meta

    def _merge_metadata(self, base: Optional[Dict], updates: Optional[Dict]) -> Dict:
        out = (base or {}).copy()
        if not updates:
            return out
        for k, v in updates.items():
            if k in ("file_path", "FILE", "path"):
                # Only accept if it looks like a real path; ignore labels like "Cropped Image"
                if not self._looks_like_path(v):
                    continue
            if k in ("fits_header", "header"):
                # Don’t replace with None/blank
                if v is None or (isinstance(v, str) and not v.strip()):
                    continue
            out[k] = v
        return out

    def _emit_change(self, slot: int):
        """Centralized emitter to avoid passing None metadata to listeners."""
        img = self._images[slot]
        meta = self._metadata[slot]
        self.image_changed.emit(slot, img, meta)
        if self.parent and hasattr(self.parent, "update_undo_redo_action_labels"):
            self.parent.update_undo_redo_action_labels()
        if self.parent and hasattr(self.parent, "update_slot_toolbar_highlight"):
            self.parent.update_slot_toolbar_highlight()


    def get_current_image_and_metadata(self):
        slot = self.current_slot
        return self._images[slot], self._metadata[slot]

    def rename_slot(self, slot: int, new_name: str):
        """Store a custom slot_name in metadata and emit an update."""
        if 0 <= slot < self.max_slots:
            self._metadata[slot]['slot_name'] = new_name

            # explicitly check for None, avoid ambiguous truth-check on ndarray
            existing = self._images[slot]
            if existing is None:
                img = np.zeros((1,1), dtype=np.uint8)
            else:
                img = existing

            # re-emit image_changed so UI labels (menus/toolbars) can refresh
            self.image_changed.emit(slot, img, self._metadata[slot])
        else:
            print(f"ImageManager: cannot rename slot {slot}, out of range")

    def get_mask(self, slot=None):
        """
        Retrieves the mask for the current or specified slot.
        :param slot: Slot number. If None, uses current slot.
        :return: Mask as numpy array or None.
        """
        if slot is None:
            slot = self.current_slot
        return self.mask_manager.get_mask(slot)

    def set_mask(self, mask, slot=None):
        """
        Sets a mask for the current or specified slot.
        :param mask: Numpy array representing the mask.
        :param slot: Slot number. If None, uses current slot.
        """
        if slot is None:
            slot = self.current_slot
        self.mask_manager.set_mask(slot, mask)

    def clear_mask(self, slot=None):
        """
        Clears the mask for the current or specified slot.
        :param slot: Slot number. If None, uses current slot.
        """
        if slot is None:
            slot = self.current_slot
        self.mask_manager.clear_mask(slot)        

    def set_current_slot(self, slot):
        if 0 <= slot < self.max_slots:
            self.current_slot = slot
            self.current_slot_changed.emit(slot)
            # Use a non-empty placeholder if the slot is empty
            image_to_emit = self._images[slot] if self._images[slot] is not None and self._images[slot].size > 0 else np.zeros((1, 1), dtype=np.uint8)
            self.image_changed.emit(slot, image_to_emit, self._metadata[slot])
            print(f"ImageManager: Current slot set to {slot}.")
            if self.parent and hasattr(self.parent, "update_slot_toolbar_highlight"):
                self.parent.update_slot_toolbar_highlight()            
        else:
            print(f"ImageManager: Slot {slot} is out of range.")


    def add_image(self, slot, image, metadata):
        """
        Adds an image and its metadata to a specified slot.
        
        :param slot: The slot number where the image will be added.
        :param image: The image data (numpy array).
        :param metadata: A dictionary containing metadata for the image.
        """
        if 0 <= slot < self.max_slots:
            self._images[slot] = image
            self._metadata[slot] = metadata
            # Clear undo/redo stacks when a new image is added
            self._undo_stacks[slot].clear()
            self._redo_stacks[slot].clear()
            self.current_slot = slot
            self.image_changed.emit(slot, image, metadata)
            print(f"ImageManager: Image added to slot {slot} with metadata.")
        else:
            print(f"ImageManager: Slot {slot} is out of range. Max slots: {self.max_slots}")
        if metadata is None:
            metadata = {}
        metadata.setdefault("step_name", "Loaded")


    def set_image(self, new_image, metadata, step_name=None):
        slot = self.current_slot
        if self._images[slot] is not None:
            self._undo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy(), step_name or "Unnamed Step")
            )
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous image in slot {slot} pushed to undo stack.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to push to undo stack.")

        merged = self._merge_metadata(self._metadata[slot], metadata)
        merged = self._attach_step_name(merged, step_name)  # <-- add this
        self._images[slot] = new_image
        self._metadata[slot] = merged
        self._emit_change(slot)
        print(f"ImageManager: Image set for slot {slot} with merged metadata.")


    def set_image_for_slot(self, slot, new_image, metadata, step_name=None):
        if slot < 0 or slot >= self.max_slots:
            print(f"ImageManager: Slot {slot} is out of range. Max slots={self.max_slots}")
            return

        if self._images[slot] is not None:
            self._undo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy(), step_name or "Unnamed Step")
            )
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous image in slot {slot} pushed to undo stack.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to push to undo stack.")

        merged = self._merge_metadata(self._metadata[slot], metadata)
        merged = self._attach_step_name(merged, step_name)
        self._images[slot] = new_image
        self._metadata[slot] = merged
        self.current_slot = slot
        self._emit_change(slot)
        print(f"ImageManager: Image set for slot {slot} with merged metadata.")


    @property
    def image(self):
        return self._images[self.current_slot]

    @image.setter
    def image(self, new_image):
        """
        Default image setter that stores undo as an unnamed step.
        """
        self.set_image_with_step_name(new_image, self._metadata[self.current_slot], step_name="Unnamed Step")

    def set_image_with_step_name(self, new_image, metadata, step_name="Unnamed Step"):
        slot = self.current_slot
        if self._images[slot] is not None:
            self._undo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy(), step_name)
            )
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous image in slot {slot} pushed to undo stack (step: {step_name})")
        else:
            print(f"ImageManager: No existing image in slot {slot} to push to undo stack.")

        merged = self._merge_metadata(self._metadata[slot], metadata)
        merged = self._attach_step_name(merged, step_name)
        self._images[slot] = new_image
        self._metadata[slot] = merged
        self._emit_change(slot)
        print(f"ImageManager: Image set for slot {slot} via set_image_with_step_name (merged).")


    def get_slot_name(self, slot):
        """
        Returns the display name for a given slot.
        If a slot has been renamed (stored under "slot_name" in metadata), that name is returned.
        Otherwise, it returns "Slot X" (using 1-indexed numbering for display).
        """
        metadata = self._metadata.get(slot, {})
        if 'slot_name' in metadata:
            return metadata['slot_name']
        else:
            return f"Slot {slot}"


    def set_metadata(self, metadata):
        slot = self.current_slot
        if self._images[slot] is not None:
            self._undo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy())
            )
            self._redo_stacks[slot].clear()
            print(f"ImageManager: Previous metadata in slot {slot} pushed to undo stack.")
        else:
            print(f"ImageManager: No existing image in slot {slot} to set metadata.")

        merged = self._merge_metadata(self._metadata[slot], metadata)
        self._metadata[slot] = merged
        self._emit_change(slot)
        print(f"ImageManager: Metadata set for slot {slot} (merged).")

    def update_image(self, updated_image, metadata=None, slot=None):
        if slot is None:
            slot = self.current_slot

        self._images[slot] = updated_image
        if metadata is not None:
            merged = self._merge_metadata(self._metadata[slot], metadata)
            self._metadata[slot] = merged

        self._emit_change(slot)

    def can_undo(self, slot=None):
        """
        Determines if there are actions available to undo for the specified slot.
        
        :param slot: (Optional) The slot number to check. If None, uses current_slot.
        :return: True if undo is possible, False otherwise.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            return len(self._undo_stacks[slot]) > 0
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot check can_undo.")
            return False

    def can_redo(self, slot=None):
        """
        Determines if there are actions available to redo for the specified slot.
        
        :param slot: (Optional) The slot number to check. If None, uses current_slot.
        :return: True if redo is possible, False otherwise.
        """
        if slot is None:
            slot = self.current_slot
        if 0 <= slot < self.max_slots:
            return len(self._redo_stacks[slot]) > 0
        else:
            print(f"ImageManager: Slot {slot} is out of range. Cannot check can_redo.")
            return False

    def undo(self, slot=None):
        if slot is None:
            slot = self.current_slot

        if 0 <= slot < self.max_slots and self.can_undo(slot):
            self._redo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy(), "Redo of Previous Step")
            )

            popped = self._undo_stacks[slot].pop()
            if len(popped) == 3:
                prev_img, prev_meta, step_name = popped
            else:
                prev_img, prev_meta = popped
                step_name = "Unnamed Undo Step"

            self._images[slot] = prev_img
            self._metadata[slot] = prev_meta
            self.image_changed.emit(slot, prev_img, prev_meta)

            print(f"ImageManager: Undo performed on slot {slot}: {step_name}")
            return step_name
        else:
            print(f"ImageManager: Cannot perform undo on slot {slot}.")
            return None



    def redo(self, slot=None):
        if slot is None:
            slot = self.current_slot

        if 0 <= slot < self.max_slots and self.can_redo(slot):
            self._undo_stacks[slot].append(
                (self._images[slot].copy(), self._metadata[slot].copy(), "Undo of Redone Step")
            )

            popped = self._redo_stacks[slot].pop()
            if len(popped) == 3:
                redo_img, redo_meta, step_name = popped
            else:
                redo_img, redo_meta = popped
                step_name = "Unnamed Redo Step"

            self._images[slot] = redo_img
            self._metadata[slot] = redo_meta
            self.image_changed.emit(slot, redo_img, redo_meta)

            print(f"ImageManager: Redo performed on slot {slot}: {step_name}")
            return step_name
        else:
            print(f"ImageManager: Cannot perform redo on slot {slot}.")
            return None

    def get_history_image(self, slot: int, index: int):
        """
        Get a specific image from the undo stack (not applied, just for preview).
        :param slot: Slot number.
        :param index: Index from the bottom (0 = oldest).
        """
        if 0 <= slot < self.max_slots:
            stack = self._undo_stacks[slot]
            if 0 <= index < len(stack):
                img, meta, _ = stack[index] if len(stack[index]) == 3 else (*stack[index], "Unnamed")
                return img.copy(), meta.copy()
        return None, None

    def get_image_for_slot(self, slot: int) -> Optional[np.ndarray]:
        """Return the image stored in slot, or None if empty."""
        return self._images.get(slot)

class MaskManager(QObject):
    """
    Manages masks and tracks whether a mask is applied to the image.
    """
    mask_changed = pyqtSignal(int, np.ndarray)  # Signal to notify mask changes (slot, mask)
    applied_mask_changed = pyqtSignal(int, np.ndarray)  # Signal for applied mask updates

    def __init__(self, max_slots=5):
        super().__init__()
        self.max_slots = max_slots
        self._masks = {i: None for i in range(max_slots)}  # Store masks for each slot
        self.applied_mask_slot = None  # Slot from which the mask is applied
        self.applied_mask = None  # Currently applied mask (numpy array)

    def set_mask(self, slot, mask):
        """
        Sets the mask for a specific slot.
        """
        if 0 <= slot < self.max_slots:
            self._masks[slot] = mask
            self.mask_changed.emit(slot, mask)

    def get_mask(self, slot):
        """
        Retrieves the mask from a specific slot.
        """
        return self._masks.get(slot, None)

    def clear_applied_mask(self):
        """
        Clears the currently applied mask and emits an empty mask.
        """
        self.applied_mask_slot = None
        self.applied_mask = None

        # Emit an empty mask instead of None
        empty_mask = np.zeros((1, 1), dtype=np.uint8)  
        self.applied_mask_changed.emit(-1, empty_mask)  # Signal that no mask is applied

        print("Applied mask cleared.")



    def apply_mask_from_slot(self, slot):
        """
        Applies the mask from the specified slot.
        """
        if slot in self._masks and self._masks[slot] is not None:
            self.applied_mask_slot = slot
            self.applied_mask = self._masks[slot]
            self.applied_mask_changed.emit(slot, self.applied_mask)
            print(f"Mask from slot {slot} applied.")
        else:
            print(f"Mask from slot {slot} cannot be applied (empty).")

    def get_applied_mask(self):
        """
        Retrieves the currently applied mask.
        """
        return self.applied_mask

    def get_applied_mask_slot(self):
        """
        Retrieves the slot from which the currently applied mask originated.
        """
        return self.applied_mask_slot

def _finalize_loaded_image(arr: np.ndarray) -> np.ndarray:
    """Ensure float32 [finite], C-contiguous for downstream Qt/Numba."""
    if arr is None:
        return None
    # Replace NaN/Inf (can appear after BSCALE/BZERO math)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    # Force float32 + C-order (copies if needed; detaches from memmap)
    return np.asarray(arr, dtype=np.float32, order="C")

def list_fits_extensions(path: str) -> dict:
    """
    Return a dict {extname_or_index: {"index": i, "shape": shape, "dtype": dtype}} for all IMAGE HDUs.
    extname_or_index prefers the HDU name (uppercased) when present, otherwise the numeric index.
    """
    if path.lower().endswith(('.fits.gz', '.fit.gz')):
        with gzip.open(path, 'rb') as f:
            buf = BytesIO(f.read())
        hdul = fits.open(buf, memmap=False)
    else:
        hdul = fits.open(path, memmap=False)

    info = {}
    with hdul as hdul:
        for i, hdu in enumerate(hdul):
            if getattr(hdu, 'data', None) is None:
                continue
            if not hasattr(hdu, 'data'):
                continue
            key = (hdu.name or str(i)).upper()
            try:
                shp = tuple(hdu.data.shape)
                dt  = hdu.data.dtype
                info[key] = {"index": i, "shape": shp, "dtype": str(dt)}
            except Exception:
                pass
    return info


def load_fits_extension(path: str, key: str | int):
    """
    Load a single IMAGE HDU (by extname or index) as float32 in [0..1] (like load_image does).
    Returns (image: np.ndarray, header: fits.Header, bit_depth: str, is_mono: bool).
    """
    if path.lower().endswith(('.fits.gz', '.fit.gz')):
        with gzip.open(path, 'rb') as f:
            buf = BytesIO(f.read())
        hdul = fits.open(buf, memmap=False)
    else:
        hdul = fits.open(path, memmap=False)

    with hdul as hdul:
        # resolve key
        if isinstance(key, str):
            # find first matching extname (case-insensitive)
            idx = None
            for i, hdu in enumerate(hdul):
                if (hdu.name or '').upper() == key.upper():
                    idx = i; break
            if idx is None:
                raise KeyError(f"Extension '{key}' not found in {path}")
        else:
            idx = int(key)

        hdu = hdul[idx]
        data = hdu.data
        if data is None:
            raise ValueError(f"HDU {key} has no image data")

        # normalize like your load_image
        import numpy as np
        if data.dtype == np.uint8:
            bit_depth = "8-bit";  img = data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            bit_depth = "16-bit"; img = data.astype(np.float32) / 65535.0
        elif data.dtype == np.uint32:
            bit_depth = "32-bit unsigned"; 
            bzero  = hdu.header.get('BZERO', 0); bscale = hdu.header.get('BSCALE', 1)
            img = data.astype(np.float32) * bscale + bzero
        elif data.dtype == np.int32:
            bit_depth = "32-bit signed";
            bzero  = hdu.header.get('BZERO', 0); bscale = hdu.header.get('BSCALE', 1)
            img = data.astype(np.float32) * bscale + bzero
        elif data.dtype == np.float32:
            bit_depth = "32-bit floating point"; img = np.array(data, dtype=np.float32, copy=True, order="C")
        else:
            raise ValueError(f"Unsupported FITS extension dtype: {data.dtype}")

        img = np.squeeze(img)
        if img.dtype == np.float32 and img.size and img.max() > 1.0:
            img = img / float(img.max())

        if img.ndim == 2:
            is_mono = True
        elif img.ndim == 3 and img.shape[0] == 3 and img.shape[1] > 1 and img.shape[2] > 1:
            img = np.transpose(img, (1, 2, 0)); is_mono = False
        elif img.ndim == 3 and img.shape[-1] == 3:
            is_mono = False
        else:
            raise ValueError(f"Unsupported FITS ext dimensions: {img.shape}")

        from .image_manager import _finalize_loaded_image  # or adjust import if needed
        img = _finalize_loaded_image(img)
        return img, hdu.header, bit_depth, is_mono


def _normalize_to_float(image_u16: np.ndarray) -> tuple[np.ndarray, str, bool]:
    """Normalize uint16/uint8 arrays to float32 [0,1] and detect mono."""
    if image_u16.dtype == np.uint16:
        bit_depth = "16-bit"
        img = image_u16.astype(np.float32) / 65535.0
    elif image_u16.dtype == np.uint8:
        bit_depth = "8-bit"
        img = image_u16.astype(np.float32) / 255.0
    else:
        bit_depth = str(image_u16.dtype)
        img = image_u16.astype(np.float32)
        mx = float(img.max()) if img.size else 1.0
        if mx > 0:
            img /= mx
    is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return img, bit_depth, is_mono


def _try_load_raw_with_rawpy(filename: str, allow_thumb_preview: bool = True, debug_thumb: bool = True):
    """
    Open RAW with rawpy/LibRaw and return a normalized [0,1] Bayer mosaic (mono=True).
    Fallbacks:
      1) raw.raw_image_visible
      2) raw.raw_image
      3) raw.postprocess(...) → linear 16-bit RGB (no auto-bright), normalized to [0,1]
      4) Embedded JPEG preview (8-bit)
    Returns: (image, header, bit_depth, is_mono)
    """
    if rawpy is None:
        raise RuntimeError("rawpy not installed")

    def _normalize_bayer(arr: np.ndarray, raw) -> tuple[np.ndarray, fits.Header, str, bool]:
        arr = arr.astype(np.float32, copy=False)
        blk = float(np.mean(getattr(raw, "black_level_per_channel", [0, 0, 0, 0])))
        wht = float(getattr(raw, "white_level", max(1.0, float(arr.max()))))
        arr = np.clip(arr - blk, 0, None)
        scale = max(1.0, wht - blk)
        arr /= scale

        hdr = fits.Header()
        # Fill from raw.metadata first
        hdr = _fill_hdr_from_raw_metadata(raw, hdr)

        # Optional extra bits you already had:
        try:
            if getattr(raw, "camera_whitebalance", None) is not None:
                hdr["CAMWB0"] = float(raw.camera_whitebalance[0])
        except Exception:
            pass

        for key, attr in (("EXPTIME", "shutter"),
                          ("ISO", "iso_speed"),
                          ("FOCAL", "focal_len"),
                          ("TIMESTAMP", "timestamp")):
            if hasattr(raw, attr) and key not in hdr:
                hdr[key] = getattr(raw, attr)

        try:
            cfa = getattr(raw, "raw_colors_visible", None)
            if cfa is not None:
                mapping = {0: "R", 1: "G", 2: "B"}
                desc = "".join(mapping.get(int(v), "?") for v in cfa.flatten()[:4])
                hdr["CFA"] = desc
        except Exception:
            pass

        return arr, hdr, "16-bit", True  # Bayer mosaic → mono=True

    # Attempt 1: visible mosaic
    try:
        with rawpy.imread(filename) as raw:
            bayer = raw.raw_image_visible
            if bayer is None:
                raise RuntimeError("raw_image_visible is None")
            return _normalize_bayer(bayer, raw)
    except Exception as e1:
        print(f"[rawpy] full decode (visible) failed: {e1}")

    # Attempt 2: full raw mosaic (no explicit unpack)
    try:
        with rawpy.imread(filename) as raw:
            bayer = getattr(raw, "raw_image", None)
            if bayer is None:
                raise RuntimeError("raw_image is None")
            return _normalize_bayer(bayer, raw)
    except Exception as e2:
        print(f"[rawpy] second pass (raw_image) failed: {e2}")

    # Attempt 3: safe demosaic (linear, no auto-bright) → RGB float32 [0,1]
    try:
        with rawpy.imread(filename) as raw:
            rgb16 = raw.postprocess(
                output_bps=16,
                gamma=(1, 1),              # keep linear
                no_auto_bright=True,       # avoid LibRaw “lift”
                use_camera_wb=False,       # neutral; you can set True if desired
                output_color=rawpy.ColorSpace.raw,
                user_flip=0,
            )
            img = rgb16.astype(np.float32) / 65535.0  # HxWx3

            hdr = fits.Header()
            hdr = _fill_hdr_from_raw_metadata(raw, hdr)
            hdr["RAW_DEM"] = (True, "LibRaw postprocess; linear, no auto-bright, RAW color")

            return img, hdr, "16-bit demosaiced", False
    except Exception as e3:
        print(f"[rawpy] postprocess fallback failed: {e3}")

    # Attempt 4: embedded JPEG preview
    if allow_thumb_preview:
        try:
            with rawpy.imread(filename) as raw2:
                th = raw2.extract_thumb()
                if debug_thumb:
                    kind = getattr(th.format, "name", str(th.format))
                    print(f"[rawpy] extract_thumb: kind={kind}, bytes={len(th.data)}")
                from io import BytesIO as _BytesIO
                pil = Image.open(_BytesIO(th.data))
                if pil.mode not in ("RGB", "L"):
                    pil = pil.convert("RGB")
                img = np.array(pil, dtype=np.float32) / 255.0
                is_mono = (img.ndim == 2)

                hdr = fits.Header()
                hdr = _fill_hdr_from_raw_metadata(raw2, hdr)
                hdr["RAW_PREV"] = (True, "Embedded JPEG preview (no linear RAW data)")

                return img, hdr, "8-bit preview (JPEG from RAW)", is_mono
        except Exception as e4:
            print(f"[rawpy] extract_thumb failed: {e4}")


    raise RuntimeError("RAW decode failed (rawpy).")

import os
import datetime

import exifread

RAW_EXTS = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')


def _is_raw_file(path: str) -> bool:
    return path.lower().endswith(RAW_EXTS)


def _parse_fraction_or_float(val) -> float | None:
    """
    Accepts things like '1/125', '0.008', 8, or exifread Ratio objects.
    Returns float seconds or None.
    """
    s = str(val).strip()
    if not s:
        return None
    try:
        # exifread often gives a single Ratio or list of one Ratio
        if hasattr(val, "num") and hasattr(val, "den"):
            return float(val.num) / float(val.den)
        if isinstance(val, (list, tuple)) and val and hasattr(val[0], "num"):
            r = val[0]
            return float(r.num) / float(r.den)

        if '/' in s:
            num, den = s.split('/', 1)
            return float(num) / float(den)
        return float(s)
    except Exception:
        return None


def _parse_exif_datetime(dt_str: str) -> str | None:
    """
    EXIF typically: 'YYYY:MM:DD HH:MM:SS'.
    Returns ISO-like 'YYYY-MM-DDTHH:MM:SS' or None.
    """
    s = str(dt_str).strip()
    if not s:
        return None

    # exifread sometimes formats as "YYYY:MM:DD HH:MM:SS"
    try:
        date_part, time_part = s.split(' ', 1)
        y, m, d = date_part.split(':', 2)
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{time_part}"
    except Exception:
        return None


def _ensure_minimal_header(header, file_path: str) -> fits.Header:
    """
    Guarantee we have a FITS Header. For non-FITS sources (TIFF/PNG/JPG/etc),
    synthesize a basic header and fill DATE-OBS from file mtime if missing.
    """
    if header is None:
        header = fits.Header()
        header["SIMPLE"]  = True
        header["BITPIX"]  = 16
        header["CREATOR"] = "SetiAstroSuite"

    # Try to provide DATE-OBS if not present
    if "DATE-OBS" not in header:
        try:
            ts = os.path.getmtime(file_path)
            dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
            header["DATE-OBS"] = (
                dt.isoformat(timespec="seconds"),
                "File modification time (UTC) used as DATE-OBS"
            )
        except Exception:
            pass

    return header


def _enrich_header_from_exif(header: fits.Header, file_path: str) -> fits.Header:
    """
    Merge EXIF metadata from a RAW file into an existing header without
    blowing away other keys. Only fills keys that are missing.
    """
    header = header.copy() if header is not None else fits.Header()
    header.setdefault("SIMPLE", True)
    header.setdefault("BITPIX", 16)
    header.setdefault("CREATOR", "SetiAstroSuite")

    try:
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        # Can't read EXIF → just return what we have
        return header

    def get_tag(*names):
        for n in names:
            t = tags.get(n)
            if t is not None:
                return t
        return None

    # Exposure time
    exptime_tag = get_tag("EXIF ExposureTime", "EXIF ShutterSpeedValue")
    if exptime_tag and "EXPTIME" not in header:
        val = _parse_fraction_or_float(exptime_tag.values)
        if val is not None:
            header["EXPTIME"] = (float(val), "Exposure time (s) from EXIF")

    # ISO
    iso_tag = get_tag("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
    if iso_tag and "ISO" not in header:
        try:
            header["ISO"] = (int(str(iso_tag.values)), "ISO from EXIF")
        except Exception:
            header["ISO"] = (str(iso_tag.values), "ISO from EXIF")

    # Date/time
    date_tag = get_tag(
        "EXIF DateTimeOriginal",
        "EXIF DateTimeDigitized",
        "Image DateTime",
    )
    if date_tag and "DATE-OBS" not in header:
        dt = _parse_exif_datetime(date_tag.values)
        if dt:
            header["DATE-OBS"] = (dt, "Start of exposure (camera local time)")

    # Aperture
    fnum_tag = get_tag("EXIF FNumber")
    if fnum_tag and "FNUMBER" not in header:
        val = _parse_fraction_or_float(fnum_tag.values)
        if val is not None:
            header["FNUMBER"] = (float(val), "F-number (aperture)")

    # Focal length
    fl_tag = get_tag("EXIF FocalLength")
    if fl_tag and "FOCALLEN" not in header:
        val = _parse_fraction_or_float(fl_tag.values)
        if val is not None:
            header["FOCALLEN"] = (float(val), "Focal length (mm)")

    # Camera make/model
    make_tag  = get_tag("Image Make")
    model_tag = get_tag("Image Model")
    cam_parts = []
    if make_tag:
        cam_parts.append(str(make_tag.values).strip())
    if model_tag:
        cam_parts.append(str(model_tag.values).strip())
    camera_str = " ".join(p for p in cam_parts if p)
    if camera_str:
        header.setdefault("INSTRUME", camera_str)  # instrument / camera
        header.setdefault("CAMERA", camera_str)    # custom keyword

    return header

def _fill_hdr_from_raw_metadata(raw, hdr: fits.Header | None = None) -> fits.Header:
    """
    Merge LibRaw/rawpy metadata into hdr (EXPTIME, ISO, FNUMBER, FOCALLEN, camera, DATE-OBS).
    Does NOT overwrite existing keys.
    """
    if hdr is None:
        hdr = fits.Header()

    try:
        m = raw.metadata
    except Exception:
        return hdr

    # Exposure time (seconds)
    if hasattr(m, "exposure") and m.exposure is not None and "EXPTIME" not in hdr:
        try:
            hdr["EXPTIME"] = (float(m.exposure), "Exposure time (s) from RAW metadata")
        except Exception:
            pass

    # ISO
    if hasattr(m, "iso") and m.iso is not None and "ISO" not in hdr:
        try:
            hdr["ISO"] = (int(m.iso), "ISO from RAW metadata")
        except Exception:
            hdr["ISO"] = (str(m.iso), "ISO from RAW metadata")

    # Aperture
    if hasattr(m, "aperture") and m.aperture is not None and "FNUMBER" not in hdr:
        try:
            hdr["FNUMBER"] = (float(m.aperture), "F-number (aperture) from RAW metadata")
        except Exception:
            pass

    # Focal length (mm)
    if hasattr(m, "focal_len") and m.focal_len is not None and "FOCALLEN" not in hdr:
        try:
            hdr["FOCALLEN"] = (float(m.focal_len), "Focal length (mm) from RAW metadata")
        except Exception:
            pass

    # Camera make/model
    make  = getattr(m, "make", None)
    model = getattr(m, "model", None)
    cam_parts = []
    if make:
        cam_parts.append(str(make).strip())
    if model:
        cam_parts.append(str(model).strip())
    camera_str = " ".join(p for p in cam_parts if p)
    if camera_str:
        hdr.setdefault("INSTRUME", camera_str)
        hdr.setdefault("CAMERA",   camera_str)

    # Timestamp → DATE-OBS in UTC
    if hasattr(m, "timestamp") and m.timestamp and "DATE-OBS" not in hdr:
        try:
            dt = datetime.datetime.fromtimestamp(m.timestamp, tz=datetime.timezone.utc)
            hdr["DATE-OBS"] = (dt.isoformat(timespec="seconds"), "RAW timestamp (UTC)")
        except Exception:
            pass

    return hdr

from astropy.wcs import WCS

import ast

def _coerce_fits_value(v):
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip()

    # PixInsight T/F
    if s in ("T", "TRUE", "True", "true"):
        return True
    if s in ("F", "FALSE", "False", "false"):
        return False

    # int?
    try:
        if s.isdigit() or (s.startswith(("+", "-")) and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    # float? (handles 8.9669e+03 etc)
    try:
        return float(s)
    except Exception:
        pass

    # strip quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s


def xisf_fits_header_from_meta(image_meta: dict, file_meta: dict | None = None) -> fits.Header:
    """
    Robustly extract FITSKeywords from XISF wrappers matching your real structure.

    Handles:
      - image_meta["FITSKeywords"]
      - image_meta["xisf_meta"]["FITSKeywords"]
      - image_meta["xisf_meta"] as a stringified dict containing FITSKeywords
      - file_meta FITSKeywords (only fills missing keys)
    """
    hdr = fits.Header()

    def _get_kw_dict(meta: dict):
        if not isinstance(meta, dict):
            return None

        # direct
        kw = meta.get("FITSKeywords")
        if isinstance(kw, dict):
            return kw

        # nested dict
        xm = meta.get("xisf_meta")
        if isinstance(xm, dict):
            kw = xm.get("FITSKeywords")
            if isinstance(kw, dict):
                return kw

        # stringified dict
        if isinstance(xm, str) and "FITSKeywords" in xm:
            try:
                xm2 = ast.literal_eval(xm)
                if isinstance(xm2, dict) and isinstance(xm2.get("FITSKeywords"), dict):
                    return xm2["FITSKeywords"]
            except Exception:
                pass

        return None

    def _apply_kw_dict(kw: dict, only_missing: bool):
        for key, entries in kw.items():
            try:
                k = str(key).strip()
                if not k:
                    continue
                if only_missing and (k in hdr):
                    continue

                # your structure: KEY: [ {"value": "...", "comment": "..."} ]
                val = None
                com = None
                if isinstance(entries, list) and entries:
                    e0 = entries[0]
                    if isinstance(e0, dict):
                        val = _coerce_fits_value(e0.get("value"))
                        com = e0.get("comment")
                    else:
                        val = _coerce_fits_value(e0)
                elif isinstance(entries, dict):
                    val = _coerce_fits_value(entries.get("value"))
                    com = entries.get("comment")
                else:
                    val = _coerce_fits_value(entries)

                if com is not None:
                    hdr[k] = (val, str(com))
                else:
                    hdr[k] = val
            except Exception:
                pass

    # First: image-level FITSKeywords (authoritative)
    kw_img = _get_kw_dict(image_meta) or {}
    if isinstance(kw_img, dict):
        _apply_kw_dict(kw_img, only_missing=False)

    # Then: file-level FITSKeywords (fill gaps only)
    kw_file = _get_kw_dict(file_meta or {}) or {}
    if isinstance(kw_file, dict):
        _apply_kw_dict(kw_file, only_missing=True)

    return hdr


def attach_wcs_to_metadata(meta: dict, hdr: fits.Header | dict | None) -> dict:
    """
    If hdr contains WCS, create an astropy.wcs.WCS and stash in metadata.
    """
    if not hdr or meta is None:
        return meta or {}

    if meta.get("wcs") is not None:
        return meta  # already present

    try:
        fhdr = hdr if isinstance(hdr, fits.Header) else fits.Header(hdr)

        # 🔹 Drop problematic long-string cards that upset astropy.wcs
        # FILE_PATH is the one we saw erroring, but you can add more here if needed.
        if "FILE_PATH" in fhdr:
            val = str(fhdr["FILE_PATH"])
            if len(val) > 68:  # FITS cards max 80 chars, ~68 for value
                print(f"⚠️ Dropping FILE_PATH from WCS header build (too long: {len(val)} chars)")
                del fhdr["FILE_PATH"]

        # Optional: also run through our invalid-card stripper
        fhdr = _drop_invalid_cards(fhdr)

        # --- Quick sanity: no basic WCS → bail quietly ---
        core_keys = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2")
        if not all(k in fhdr for k in core_keys):
            return meta

        # --- Attempt 1: basic WCS ---
        try:
            w = WCS(fhdr, relax=True)
        except Exception as e1:
            print(f"⚠️ WCS(fhdr, relax=True) failed: {e1}")
            print("⚠️ Retrying WCS with naxis=2 (ignore extra axis).")
            try:
                w = WCS(fhdr, relax=True, naxis=2)
            except Exception as e2:
                print(f"⚠️ WCS(..., naxis=2) failed: {e2}")
                print("⚠️ Retrying WCS with naxis=2 after stripping SIP terms.")
                try:
                    fhdr2 = fhdr.copy()
                    for k in list(fhdr2.keys()):
                        if k.startswith(("A_", "B_", "AP_", "BP_", "A_ORDER", "B_ORDER")):
                            del fhdr2[k]
                    w = WCS(fhdr2, relax=True, naxis=2)
                except Exception as e3:
                    print(f"⚠️ WCS(..., naxis=2) after SIP-strip failed: {e3}")
                    raise e1  # re-raise original

        if getattr(w, "has_celestial", False):
            meta["wcs"] = w
            meta["wcs_header"] = w.to_header(relax=True)
            meta["wcsaxes"] = int(getattr(w, "naxis", getattr(w.wcs, "naxis", 2)))
            print(f"🔷 Attached astropy WCS into metadata (naxis={meta['wcsaxes']})")
        else:
            print("⚠️ WCS parsed but has no celestial axes; not attaching.")

    except Exception as e:
        print(f"⚠️ Failed to build WCS from header: {e}")

    return meta


def load_image(filename, max_retries=3, wait_seconds=3, return_metadata: bool = False):
    """
    Loads an image from the specified filename with support for various formats.
    If a "buffer is too small for requested array" error occurs, it retries loading after waiting.

    Parameters:
        filename (str): Path to the image file.
        max_retries (int): Number of times to retry on specific buffer error.
        wait_seconds (int): Seconds to wait before retrying.

    Returns:
        tuple: (image, original_header, bit_depth, is_mono) or (None, None, None, None) on failure.
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            image = None  # Ensure 'image' is explicitly declared
            bit_depth = None
            is_mono = False
            original_header = None

            # --- Unified FITS handling ---
            if filename.lower().endswith(('.fits', '.fit', '.fits.gz', '.fit.gz', '.fz', '.fz')):
                # Use get_valid_header to retrieve the header and extension index.
                original_header, ext_index = get_valid_header(filename)

                
                # Open the file appropriately.
                if filename.lower().endswith(('.fits.gz', '.fit.gz')):
                    print(f"Loading compressed FITS file: {filename}")
                    with gzip.open(filename, 'rb') as f:
                        file_content = f.read()
                    hdul = fits.open(BytesIO(file_content))
                else:
                    if filename.lower().endswith(('.fz', '.fz')):
                        print(f"Loading Rice-compressed FITS file: {filename}")
                    else:
                        print(f"Loading FITS file: {filename}")
                    hdul = fits.open(filename)

                with hdul as hdul:
                    # Retrieve image data from the extension indicated by get_valid_header.
                    image_data = hdul[ext_index].data
                    if image_data is None:
                        raise ValueError(f"No image data found in FITS file in extension {ext_index}.")

                    # Ensure native byte order
                    if image_data.dtype.byteorder not in ('=', '|'):
                        image_data = image_data.astype(image_data.dtype.newbyteorder('='))

                    # ---------------------------------------------------------------------
                    # 1) Detect bit depth and convert to float32
                    # ---------------------------------------------------------------------
                    if image_data.dtype == np.uint8:
                        bit_depth = "8-bit"
                        print("Identified 8-bit FITS image.")
                        image = image_data.astype(np.float32) / 255.0

                    elif image_data.dtype == np.uint16:
                        bit_depth = "16-bit"
                        print("Identified 16-bit FITS image.")
                        image = image_data.astype(np.float32) / 65535.0

                    elif image_data.dtype == np.int16:
                        bit_depth = "16-bit signed"
                        print("Identified 16-bit signed FITS image.")
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        data = image_data.astype(np.float32) * float(bscale) + float(bzero)

                        if bzero != 0 or bscale != 1:
                            image = np.clip(data / 65535.0, 0.0, 1.0)
                        else:
                            dmin = float(data.min())
                            dmax = float(data.max())
                            if dmax > dmin:
                                image = (data - dmin) / (dmax - dmin)
                            else:
                                image = np.zeros_like(data, dtype=np.float32)

                    elif image_data.dtype == np.int8:
                        bit_depth = "8-bit signed"
                        print("Identified 8-bit signed FITS image.")
                        # Use BSCALE/BZERO if present, else generic normalize
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        data = image_data.astype(np.float32) * float(bscale) + float(bzero)
                        dmin = float(data.min())
                        dmax = float(data.max())
                        if dmax > dmin:
                            image = (data - dmin) / (dmax - dmin)
                        else:
                            image = np.zeros_like(data, dtype=np.float32)

                    elif image_data.dtype == np.int32:
                        bit_depth = "32-bit signed"
                        print("Identified 32-bit signed FITS image.")
                        bzero  = float(original_header.get('BZERO', 0))
                        bscale = float(original_header.get('BSCALE', 1))

                        # Rebuild physical values
                        data = image_data.astype(np.float32) * bscale + bzero

                        # Normalize to [0,1] for the viewer / pipeline
                        dmin = float(data.min())
                        dmax = float(data.max())
                        if dmax > dmin:
                            image = (data - dmin) / (dmax - dmin)
                        else:
                            image = np.zeros_like(data, dtype=np.float32)


                    elif image_data.dtype == np.uint32:
                        bit_depth = "32-bit unsigned"
                        print("Identified 32-bit unsigned FITS image.")

                        bzero  = float(original_header.get('BZERO', 0))
                        bscale = float(original_header.get('BSCALE', 1))

                        if bzero == 0.0 and bscale == 1.0:
                            # Literal 0..2^32-1 data → map directly to [0,1]
                            image = image_data.astype(np.float32) / 4294967295.0
                        else:
                            # Non-trivial BSCALE/BZERO: reconstruct physical values, then normalize
                            data = image_data.astype(np.float32) * bscale + bzero
                            dmin = float(data.min())
                            dmax = float(data.max())
                            if dmax > dmin:
                                image = (data - dmin) / (dmax - dmin)
                            else:
                                image = np.zeros_like(data, dtype=np.float32)


                    elif image_data.dtype == np.float32:
                        bit_depth = "32-bit floating point"
                        print("Identified 32-bit floating point FITS image.")
                        image = np.array(image_data, dtype=np.float32, copy=True, order="C")

                    elif image_data.dtype == np.float64:
                        bit_depth = "64-bit floating point"
                        print("Identified 64-bit floating point FITS image.")
                        # Keep dynamic range as-is, just cast down to float32
                        image = image_data.astype(np.float32, copy=True)

                    else:
                        raise ValueError(f"Unsupported FITS data type: {image_data.dtype}")


                    # ---------------------------------------------------------------------
                    # 2) Squeeze out any singleton dimensions (fix weird NAXIS combos)
                    # ---------------------------------------------------------------------
                    image = np.squeeze(image)

                    #if image.dtype == np.float32:
                    #    max_val = image.max()
                    #    if max_val > 1.0:
                    #        print(f"Detected float image with max value {max_val:.3f} > 1.0; rescales to [0,1]")
                    #        image = image / max_val
                    # ---------------------------------------------------------------------
                    # 3) Interpret final shape to decide if mono or color
                    # ---------------------------------------------------------------------
                    if image.ndim == 2:
                        is_mono = True
                    elif image.ndim == 3:
                        if image.shape[0] == 3 and image.shape[1] > 1 and image.shape[2] > 1:
                            image = np.transpose(image, (1, 2, 0))
                            is_mono = False
                        elif image.shape[-1] == 3:
                            is_mono = False
                        else:
                            raise ValueError(f"Unsupported 3D shape after squeeze: {image.shape}")
                    else:
                        raise ValueError(f"Unsupported FITS dimensions after squeeze: {image.shape}")

                    print(f"Loaded FITS image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
                    image = _finalize_loaded_image(image)

                    # NEW: build metadata + attach WCS
                    meta = {
                        "file_path": filename,
                        "fits_header": original_header,
                        "bit_depth": bit_depth,
                        "mono": is_mono,
                    }
                    meta = attach_wcs_to_metadata(meta, original_header)

                    if return_metadata:
                        return image, original_header, bit_depth, is_mono, meta
                    return image, original_header, bit_depth, is_mono


            elif filename.lower().endswith(('.tiff', '.tif')):
                print(f"Loading TIFF file: {filename}")
                image_data = tiff.imread(filename)
                print(f"Loaded TIFF image with dtype: {image_data.dtype}")

                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    image = image_data.astype(np.float32) / 255.0

                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    image = image_data.astype(np.float32) / 65535.0

                elif image_data.dtype == np.int16:
                    bit_depth = "16-bit signed"
                    print("Detected 16-bit signed TIFF image.")
                    data = image_data.astype(np.float32)
                    dmin = float(data.min())
                    dmax = float(data.max())
                    if dmax > dmin:
                        image = (data - dmin) / (dmax - dmin)
                    else:
                        image = np.zeros_like(data, dtype=np.float32)

                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    image = image_data.astype(np.float32) / 4294967295.0

                elif image_data.dtype == np.int32:
                    bit_depth = "32-bit signed"
                    print("Detected 32-bit signed TIFF image.")
                    data = image_data.astype(np.float32)
                    dmin = float(data.min())
                    dmax = float(data.max())
                    if dmax > dmin:
                        image = (data - dmin) / (dmax - dmin)
                    else:
                        image = np.zeros_like(data, dtype=np.float32)

                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    image = image_data.astype(np.float32)

                elif image_data.dtype == np.float64:
                    bit_depth = "64-bit floating point"
                    image = image_data.astype(np.float32)

                elif np.issubdtype(image_data.dtype, np.integer):
                    # Generic integer fallback (int8, etc.)
                    info = np.iinfo(image_data.dtype)
                    bit_depth = f"{info.bits}-bit signed"
                    print(f"Generic int TIFF; normalizing by [{info.min}, {info.max}]")
                    data = image_data.astype(np.float32)
                    # shift to [0, max-min] then normalize
                    data -= info.min
                    image = data / float(info.max - info.min)

                else:
                    raise ValueError("Unsupported TIFF format!")


                #if image.dtype == np.float32:
                #    max_val = image.max()
                #    if max_val > 1.0:
                #        print(f"Detected float image with max value {max_val:.3f} > 1.0; rescales to [0,1]")
                #        image = image / max_val

                # Handle mono or RGB TIFFs
                if image_data.ndim == 2:  # Mono
                    is_mono = True
                elif image_data.ndim == 3 and image_data.shape[2] == 3:  # RGB
                    is_mono = False
                else:
                    raise ValueError("Unsupported TIFF image dimensions!")

            elif filename.lower().endswith('.xisf'):
                print(f"Loading XISF file: {filename}")
                xisf = XISF(filename)

                # Read image data (assuming the first image in the XISF file)
                image_data = xisf.read_image(0)  # Adjust the index if multiple images are present

                # Retrieve metadata
                image_meta = xisf.get_images_metadata()[0]  # Assuming single image
                file_meta = xisf.get_file_metadata()


                # Here we check the maximum pixel value to determine bit depth
                # --- Detect the bit depth by dtype ---
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    print("Debug: Detected 8-bit dtype. Normalizing by 255.")
                    image = image_data.astype(np.float32) / 255.0

                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    print("Debug: Detected 16-bit dtype. Normalizing by 65535.")
                    image = image_data.astype(np.float32) / 65535.0

                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    print("Debug: Detected 32-bit unsigned dtype. Normalizing by 4294967295.")
                    image = image_data.astype(np.float32) / 4294967295.0

                elif image_data.dtype == np.float32 or image_data.dtype == np.float64:
                    bit_depth = "32-bit floating point"
                    print("Debug: Detected float dtype. Casting to float32 (no normalization).")
                    image = image_data.astype(np.float32)

                else:
                    raise ValueError(f"Unsupported XISF data type: {image_data.dtype}")

                # Handle mono or RGB XISF
                if image_data.ndim == 2:
                    # We know it's mono. Already normalized in `image`.
                    is_mono = True
                    # If you really want to store it in an RGB shape:
                    #image = np.stack([image] * 3, axis=-1)

                elif image_data.ndim == 3 and image_data.shape[2] == 1:
                    # It's mono with shape (H, W, 1)
                    is_mono = True
                    # Squeeze the normalized image, not the original image_data
                    image = np.squeeze(image, axis=2)
                    # If you want an RGB shape, you can do:
                    #image = np.stack([image] * 3, axis=-1)

                elif image_data.ndim == 3 and image_data.shape[2] == 3:
                    is_mono = False
                    # We already stored the normalized float32 data in `image`.
                    # So no change needed if it’s already shape (H, W, 3).

                else:
                    raise ValueError("Unsupported XISF image dimensions!")

                # ─── Build FITS header from PixInsight XISFProperties ─────────────────
                # ─── Build FITS header from XISFProperties, then fallback to FITSKeywords & Pixel‐Scale ─────────────────

                def _dump_astrometric_keys(props, image_meta, file_meta):
                    print("🔎 [XISF] XISFProperties AstrometricSolution-related keys:")
                    for k in sorted(props.keys()):
                        if "AstrometricSolution" in k or "SplineWorldTransformation" in k or "SIP" in k:
                            print("   ", k)

                    def _dump_fk(meta, tag):
                        fk = meta.get("FITSKeywords", {})
                        if not fk:
                            print(f"🔎 [XISF] No FITSKeywords in {tag}")
                            return
                        sip_keys = [k for k in fk.keys() if k.startswith(("A_", "B_", "AP_", "BP_", "A_ORDER", "B_ORDER"))]
                        print(f"🔎 [XISF] FITSKeywords SIP-ish keys in {tag}: {sorted(sip_keys)}")

                    _dump_fk(image_meta, "image_meta")
                    _dump_fk(file_meta, "file_meta")          
                # Build base header from FITSKeywords (typed) first
                hdr = xisf_fits_header_from_meta(image_meta, file_meta)   # your new helper
                _filled = set(hdr.keys())

                # Now get XISFProperties (for PI grids + fallback)
                props = (image_meta.get("XISFProperties", {}) or
                        file_meta.get("XISFProperties", {}) or {})
                #_filled = set()

                # 1) PixInsight astrometric solution (fallback only)
                # 1) PixInsight astrometric solution (fallback only)
                try:
                    if not all(k in hdr for k in ("CRPIX1","CRPIX2","CRVAL1","CRVAL2")):
                        ref_img = props['PCL:AstrometricSolution:ReferenceImageCoordinates']['value']
                        ref_sky = props['PCL:AstrometricSolution:ReferenceCelestialCoordinates']['value']

                        # Some files store extra values; only first two are CRPIX/CRVAL
                        im0, im1 = float(ref_img[0]), float(ref_img[1])
                        w0,  w1  = float(ref_sky[0]), float(ref_sky[1])

                        hdr['CRPIX1'], hdr['CRPIX2'] = im0, im1
                        hdr['CRVAL1'], hdr['CRVAL2'] = w0, w1
                        hdr.setdefault('CTYPE1', 'RA---TAN-SIP')
                        hdr.setdefault('CTYPE2', 'DEC--TAN-SIP')
                        _filled |= {'CRPIX1','CRPIX2','CRVAL1','CRVAL2','CTYPE1','CTYPE2'}
                        print("🔷 Injected CRPIX/CRVAL from XISFProperties (fallback)")
                except KeyError:
                    pass
                except Exception as e:
                    print(f"⚠️ XISFProperties CRPIX/CRVAL parse failed; skipping. Reason: {e}")

                # 2) CD matrix (fallback only)
                try:
                    if not all(k in hdr for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
                        lin = np.asarray(props['PCL:AstrometricSolution:LinearTransformationMatrix']['value'], float)
                        hdr['CD1_1'], hdr['CD1_2'] = float(lin[0,0]), float(lin[0,1])
                        hdr['CD2_1'], hdr['CD2_2'] = float(lin[1,0]), float(lin[1,1])
                        _filled |= {'CD1_1','CD1_2','CD2_1','CD2_2'}
                        print("🔷 Injected CD matrix from XISFProperties (fallback)")
                except KeyError:
                    pass

                # 3) SIP polynomial fitting  (CORRECTED for PI ImageToNative grids)
                def _try_inject_sip_from_fitskeywords(hdr, image_meta, file_meta):
                    """If PI already wrote SIP in FITSKeywords, pull it in verbatim."""
                    def _lookup_kw(key):
                        for meta in (image_meta, file_meta):
                            fk = meta.get("FITSKeywords", {})
                            if key in fk and fk[key]:
                                return fk[key][0].get("value")
                        return None

                    a_order = _lookup_kw("A_ORDER")
                    b_order = _lookup_kw("B_ORDER")
                    if a_order is None or b_order is None:
                        return False

                    try:
                        a_order = int(a_order); b_order = int(b_order)
                    except Exception:
                        return False

                    hdr["A_ORDER"] = a_order
                    hdr["B_ORDER"] = b_order

                    # pull all A_i_j / B_i_j that exist in FITSKeywords
                    for order_key, prefix in (("A_ORDER", "A_"), ("B_ORDER", "B_")):
                        o = int(hdr[order_key])
                        for i in range(o + 1):
                            for j in range(o + 1 - i):
                                if i == 0 and j == 0:
                                    continue
                                k = f"{prefix}{i}_{j}"
                                v = _lookup_kw(k)
                                if v is not None:
                                    try:
                                        hdr[k] = float(v)
                                    except Exception:
                                        pass

                    # if CTYPE isn't SIP already, make it SIP
                    hdr.setdefault("CTYPE1", "RA---TAN-SIP")
                    hdr.setdefault("CTYPE2", "DEC--TAN-SIP")

                    print(f"🔷 Injected SIP directly from FITSKeywords (A/B order {a_order})")
                    return True
                # 3a) First try to import SIP directly if PI already gave it to us
                if _try_inject_sip_from_fitskeywords(hdr, image_meta, file_meta):
                    _filled |= {"A_ORDER", "B_ORDER"} | {k for k in hdr.keys() if k.startswith(("A_", "B_"))}
                else:
                    try:
                        def _find_image_to_native_grid(props):
                            """
                            Return a dict-like pg with keys GridX/GridY/Delta/Rect in the same shape
                            your SIP fitter expects.

                            PI can store this either as:
                            A) one nested property:
                                ...:PointGridInterpolation:ImageToNative  -> dict with GridX/GridY/etc
                            B) separate leaf properties:
                                ...:ImageToNative:GridX, :GridY, :Delta, :Rect
                            """
                            base = "PCL:AstrometricSolution:SplineWorldTransformation:PointGridInterpolation:ImageToNative"

                            # Case A: full nested block exists
                            if base in props:
                                return props[base]

                            # Case B: leaf keys exist — rebuild a pseudo-block
                            gx_key = base + ":GridX"
                            gy_key = base + ":GridY"
                            if gx_key in props and gy_key in props:
                                pg = {
                                    "GridX": props[gx_key],
                                    "GridY": props[gy_key],
                                    "Delta": props.get(base + ":Delta", {"value": 1.0}),
                                    "Rect":  props.get(base + ":Rect",  {"value": None}),
                                }
                                return pg

                            return None
                       
                        pg = _find_image_to_native_grid(props)
                        if pg is None:
                            raise KeyError("No ImageToNative grid found")
                        gx = np.asarray(pg['GridX']['value'], dtype=float)
                        gy = np.asarray(pg['GridY']['value'], dtype=float)
                        delta = float(pg.get('Delta', {}).get('value', 1.0))
                        rect  = np.asarray(pg.get('Rect', {}).get('value', [0,0,gx.shape[1]*delta, gx.shape[0]*delta]), dtype=float)
                        x0, y0 = rect[0], rect[1]

                        # grid gives native-plane coords (deg) at sampled pixels
                        # build pixel coord for each grid sample
                        rows, cols = gx.shape
                        xs = x0 + np.arange(cols, dtype=float) * delta
                        ys = y0 + np.arange(rows, dtype=float) * delta
                        Xs, Ys = np.meshgrid(xs, ys)

                        # u,v relative to CRPIX for SIP basis
                        crpix1, crpix2 = float(hdr['CRPIX1']), float(hdr['CRPIX2'])
                        u = (Xs - crpix1).ravel()
                        v = (Ys - crpix2).ravel()

                        # linear native-plane coords from CD
                        CD = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                                    [hdr['CD2_1'], hdr['CD2_2']]], dtype=float)
                        duv = np.vstack([u, v])  # 2×N
                        native_lin = CD @ duv                 # deg residuals predicted by linear model
                        native_true = np.vstack([gx.ravel(), gy.ravel()])  # deg native coords from PI grids

                        # residual in native plane (deg)
                        d_native = native_true - native_lin   # 2×N in degrees

                        # convert residual degrees back to pixel residuals (dp) using inv(CD)
                        try:
                            invCD = np.linalg.inv(CD)
                        except np.linalg.LinAlgError:
                            invCD = np.linalg.pinv(CD)
                        d_pix = invCD @ d_native              # 2×N in pixels
                        dx_pix = d_pix[0]
                        dy_pix = d_pix[1]

                        # robust mask to avoid NaNs/infs
                        m = np.isfinite(u) & np.isfinite(v) & np.isfinite(dx_pix) & np.isfinite(dy_pix)
                        u = u[m]; v = v[m]; dx_pix = dx_pix[m]; dy_pix = dy_pix[m]

                        def fit_sip_pixels(u, v, dx, dy, order):
                            terms = [(i,j) for i in range(order+1) for j in range(order+1-i) if (i,j)!=(0,0)]
                            M = np.vstack([(u**i)*(v**j) for (i,j) in terms]).T
                            a, *_ = np.linalg.lstsq(M, dx, rcond=None)
                            b, *_ = np.linalg.lstsq(M, dy, rcond=None)
                            rms = np.hypot(dx - M.dot(a), dy - M.dot(b)).std()
                            return a, b, terms, rms

                        # cap order hard to avoid overfit; PI splines can be complex
                        best = {'order':None, 'rms':np.inf}

                        for order in (2,3,4):  # <=4 is plenty for real optics
                            a, b, terms, rms = fit_sip_pixels(u, v, dx_pix, dy_pix, order)
                            if rms < best['rms']:
                                best.update(order=order, a=a, b=b, terms=terms, rms=rms)

                        o = best['order']
                        hdr['A_ORDER'] = o; hdr['B_ORDER'] = o
                        _filled |= {'A_ORDER','B_ORDER'}

                        for (i,j), coef in zip(best['terms'], best['a']):
                            hdr[f'A_{i}_{j}'] = float(coef); _filled.add(f'A_{i}_{j}')
                        for (i,j), coef in zip(best['terms'], best['b']):
                            hdr[f'B_{i}_{j}'] = float(coef); _filled.add(f'B_{i}_{j}')

                        print(f"🔷 Injected SIP order {o} (from PI native grids), rms={best['rms']:.4g}px")

                    except KeyError:
                        print("⚠️ No PI ImageToNative grid; skipping SIP")
                    except Exception as e:
                        print(f"⚠️ SIP fit failed; skipping SIP. Reason: {e}")



                # Helper: look in FITSKeywords dicts
                def _lookup_kw(key):
                    for meta in (image_meta, file_meta):
                        fk = meta.get('FITSKeywords',{})
                        if key in fk and fk[key]:
                            return fk[key][0]['value']
                    return None

                # 4) Fallback WCS/CD from FITSKeywords
                for key in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CTYPE1','CTYPE2',
                            'CD1_1','CD1_2','CD2_1','CD2_2'):
                    if key not in hdr:
                        v = _lookup_kw(key)
                        if v is not None:
                            hdr[key] = v
                            _filled.add(key)
                            print(f"🔷 Injected {key} from FITSKeywords")

                # 5) Generic RA/DEC fallback
                if 'CRVAL1' not in hdr or 'CRVAL2' not in hdr:
                    for ra_kw, dec_kw in (('RA','DEC'),('OBJCTRA','OBJCTDEC')):
                        ra = _lookup_kw(ra_kw); dec = _lookup_kw(dec_kw)
                        if ra and dec:
                            try:
                                ra_deg = float(ra); dec_deg = float(dec)
                            except ValueError:
                                from astropy.coordinates import Angle
                                ra_deg  = Angle(str(ra), unit='hourangle').degree
                                dec_deg = Angle(str(dec), unit='deg').degree
                            hdr['CRVAL1'], hdr['CRVAL2'] = ra_deg, dec_deg
                            hdr.setdefault('CTYPE1','RA---TAN'); hdr.setdefault('CTYPE2','DEC--TAN')
                            print(f"🔷 Fallback CRVAL from {ra_kw}/{dec_kw}")
                            break

                # 6) Pixel‐scale fallback → inject CDELT if no CD or CDELT
                if not any(k in hdr for k in ('CD1_1','CDELT1')):
                    pix_arcsec = None
                    for kw in ('PIXSCALE','SCALE'):
                        val = _lookup_kw(kw)
                        if val:
                            pix_arcsec = float(val); break
                    if pix_arcsec is None:
                        xpsz = _lookup_kw('XPIXSZ'); foc = _lookup_kw('FOCALLEN')
                        if xpsz and foc:
                            pix_arcsec = float(xpsz)*1e-3/float(foc)*206265
                    if pix_arcsec:
                        degpix = pix_arcsec / 3600.0
                        hdr['CDELT1'], hdr['CDELT2'] = -degpix, degpix
                        print(f"🔷 Injected pixel scale {pix_arcsec:.3f}\"/px → CDELT={degpix:.6f}°")

                # 7) Copy any remaining simple FITSKeywords
                for kw, vals in file_meta.get('FITSKeywords',{}).items():
                    if kw in hdr: continue
                    v = vals[0].get('value')
                    if isinstance(v, (int,float,str)):
                        hdr[kw] = v

                # 8) Binning
                bx = int(_lookup_kw('XBINNING') or 1)
                by = int(_lookup_kw('YBINNING') or bx)
                if bx!=by: print(f"⚠️ Unequal binning {bx}×{by}, averaging")
                hdr['XBINNING'], hdr['YBINNING'] = bx, by

                original_header = hdr
                print(f"Loaded XISF header with keys: {_filled}")
                image = _finalize_loaded_image(image)

                # NEW: build metadata + attach WCS
                meta = {
                    "file_path": filename,
                    "fits_header": original_header,   # your synthesized FITS header
                    "bit_depth": bit_depth,
                    "mono": is_mono,
                    "xisf_meta": image_meta,          # optional, handy for debugging later
                }
                meta = attach_wcs_to_metadata(meta, original_header)

                if return_metadata:
                    return image, original_header, bit_depth, is_mono, meta
                return image, original_header, bit_depth, is_mono

            elif filename.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
                print(f"Loading RAW file: {filename}")

                try:
                    image, original_header, bit_depth, is_mono = _try_load_raw_with_rawpy(
                        filename,
                        allow_thumb_preview=True,   # keep your current behavior
                        debug_thumb=True
                    )

                    if original_header is None:
                        original_header = fits.Header()

                    # 🔹 Fold in EXIF, but only for keys missing from raw metadata
                    original_header = _enrich_header_from_exif(original_header, filename)

                    # If preview path returned a minimal header, that's fine—upstream UI will message it
                    if "preview" in str(bit_depth).lower():
                        print("RAW decode failed; using embedded JPEG preview (non-linear, 8-bit).")

                    image = _finalize_loaded_image(image)
                    return image, original_header, bit_depth, is_mono

                except Exception as e_raw:
                    print(f"rawpy failed: {e_raw}")
                    raise



            elif filename.lower().endswith('.png'):
                print(f"Loading PNG file: {filename}")
                img = Image.open(filename)

                # Convert unsupported modes to RGB
                if img.mode not in ('L', 'RGB'):
                    print(f"Unsupported PNG mode: {img.mode}, converting to RGB")
                    img = img.convert("RGB")

                # Convert image to numpy array and normalize pixel values to [0, 1]
                image = np.array(img, dtype=np.float32) / 255.0
                bit_depth = "8-bit"

                # Determine if the image is grayscale or RGB
                if len(image.shape) == 2:  # Grayscale image
                    is_mono = True
                elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
                    is_mono = False
                else:
                    raise ValueError(f"Unsupported PNG dimensions: {image.shape}")

                print(f"Loaded PNG image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")

            elif filename.lower().endswith(('.jpg', '.jpeg')):
                print(f"Loading JPG file: {filename}")
                img = Image.open(filename)
                if img.mode == 'L':  # Grayscale
                    is_mono = True
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                elif img.mode == 'RGB':  # RGB
                    is_mono = False
                    image = np.array(img, dtype=np.float32) / 255.0
                    bit_depth = "8-bit"
                else:
                    raise ValueError("Unsupported JPG format!")            

            else:
                raise ValueError("Unsupported file format!")

            print(f"Loaded image: shape={image.shape}, bit depth={bit_depth}, mono={is_mono}")
            image = _finalize_loaded_image(image)
            return image, original_header, bit_depth, is_mono

        except Exception as e:
            error_message = str(e)
            if "buffer is too small for requested array" in error_message.lower():
                if attempt < max_retries:
                    attempt += 1
                    print(f"Error reading image {filename}: {e}")
                    print(f"Retrying in {wait_seconds} seconds... (Attempt {attempt}/{max_retries})")
                    time.sleep(wait_seconds)
                    continue  # Retry loading the image
                else:
                    print(f"Error reading image {filename} after {max_retries} retries: {e}")
            else:
                print(f"Error reading image {filename}: {e}")
            return None, None, None, None

def get_valid_header(file_path):
    """
    Opens the FITS file (handling compressed files as needed), finds the first HDU
    with image data, and then searches through all HDUs for additional keywords (e.g. BAYERPAT).
    Returns a composite header (a copy of the image HDU header updated with extra keywords)
    and the extension index of the image data.
    """
    # Open file appropriately for compressed files
    if file_path.lower().endswith(('.fits.gz', '.fit.gz')):
        
        with gzip.open(file_path, 'rb') as f:
            file_content = f.read()
        hdul = fits.open(BytesIO(file_content))
    else:
        
        hdul = fits.open(file_path)

    with hdul as hdul:
        image_hdu = None
        image_index = None
        # First, find the HDU that contains image data
        for i, hdu in enumerate(hdul):
            
            if hdu.data is not None:
                image_hdu = hdu
                image_index = i
                
                break
        if image_hdu is None:
            raise ValueError("No image data found in FITS file.")

        # Start with a copy of the image HDU header
        composite_header = image_hdu.header.copy()
        # Drop any cards that will raise VerifyError later (e.g. broken TELESCOP)
        composite_header = _drop_invalid_cards(composite_header)


        # Now search all HDUs for extra keywords (e.g. BAYERPAT)
        for i, hdu in enumerate(hdul):
            if 'BAYERPAT' in hdu.header:
                composite_header['BAYERPAT'] = hdu.header['BAYERPAT']

                break

    return composite_header, image_index

def get_bayer_header(file_path):
    """
    Iterates through all HDUs in the FITS file (handling compressed files if needed)
    to find a header that contains the 'BAYERPAT' keyword.
    Returns the header if found, otherwise None.
    """


    try:
        # Check for compressed files first.
        if file_path.lower().endswith(('.fits.gz', '.fit.gz')):
            with gzip.open(file_path, 'rb') as f:
                file_content = f.read()
            hdul = fits.open(BytesIO(file_content))
        else:
            hdul = fits.open(file_path)
        with hdul as hdul:
            for hdu in hdul:
                if 'BAYERPAT' in hdu.header:
                    return hdu.header
    except Exception as e:
        print(f"Error in get_bayer_header: {e}")
    return None


_BIT_DEPTH_STRS = {
    "8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"
}

def _normalize_format(fmt: str) -> str:
    """Normalize an input format/extension (with or without leading dot)."""
    f = (fmt or "").lower().lstrip(".")
    if f == "jpeg": f = "jpg"
    if f == "tiff": f = "tif"
    return f

def _is_header_obj(h) -> bool:
    """True if h looks like a FITS header-ish object."""
    return isinstance(h, (fits.Header, dict))

def _looks_like_xisf_header(hdr) -> bool:
    """Detects XISF-origin metadata safely without assuming .keys() exists."""
    try:
        if isinstance(hdr, fits.Header):
            # fits.Header supports .keys() and iteration
            for k in hdr.keys():
                if isinstance(k, str) and k.startswith("XISF:"):
                    return True
        elif isinstance(hdr, dict):
            for k in hdr.keys():
                if isinstance(k, str) and k.startswith("XISF:"):
                    return True
    except Exception:
        pass
    return False

def _has_xisf_props(meta) -> bool:
    """True if meta appears to contain XISFProperties (dict or list-of-dicts)."""
    try:
        if isinstance(meta, dict):
            return "XISFProperties" in meta
        if isinstance(meta, list) and meta and isinstance(meta[0], dict):
            return "XISFProperties" in meta[0]
    except Exception:
        pass
    return False

import logging

log = logging.getLogger(__name__)

def save_image(img_array,
               filename,
               original_format,
               bit_depth=None,
               original_header=None,
               is_mono=False,
               image_meta=None,
               file_meta=None,
               wcs_header=None):   # 🔥 NEW
    """
    Save an image array to a file in the specified format and bit depth.
    - Robust to mis-ordered positional args (header/bit_depth swap).
    - Never calls .keys() on a non-mapping.
    - FITS always written as float32; header is sanitized or synthesized.
    """
    # 🔊 Debug what we got
    if isinstance(original_header, fits.Header):
        log.debug(
            "[legacy_save_image] original_header: fits.Header with %d cards, first few:",
            len(original_header)
        )
        for i, card in enumerate(original_header.cards):
            if i >= 20:
                log.debug("[legacy_save_image]   ... (truncated)")
                break
            log.debug("[legacy_save_image]   %-10s = %r", card.keyword, card.value)
    else:
        log.debug(
            "[legacy_save_image] original_header is %r, wcs_header is %r",
            type(original_header), type(wcs_header),
        )

    # --- Fix for accidental positional arg swap: (header <-> bit_depth) -----
    if isinstance(original_header, str) and original_header in _BIT_DEPTH_STRS and _is_header_obj(bit_depth):
        original_header, bit_depth = bit_depth, original_header

    # Normalize format and extension
    fmt = _normalize_format(original_format)
    base, _ = os.path.splitext(filename)
    out_ext = "jpg" if fmt == "jpg" else ("tif" if fmt == "tif" else fmt)
    if not filename.lower().endswith(f".{out_ext}"):
        filename = f"{base}.{out_ext}"

    # Ensure correct byte order for numpy data
    img_array = ensure_native_byte_order(img_array)

    # Detect XISF origin (safely)
    is_xisf = _looks_like_xisf_header(original_header) or _has_xisf_props(image_meta)

    try:
        # ---------------------------------------------------------------------
        # PNG/JPG — always write 8-bit preview-style data
        # ---------------------------------------------------------------------
        if fmt == "png":
            img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
            img.save(filename)
            print(f"Saved 8-bit PNG image to: {filename}")
            return

        if fmt == "jpg":
            img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
            # You can pass quality=95, subsampling=0 if you want
            img.save(filename)
            print(f"Saved 8-bit JPG image to: {filename}")
            return

        # ---------------------------------------------------------------------
        # TIFF — honor bit depth (fallback to 32-bit floating point)
        # ---------------------------------------------------------------------
        if fmt in ("tif",):
            bd = bit_depth or "32-bit floating point"
            if bd == "8-bit":
                tiff.imwrite(filename, (np.clip(img_array, 0, 1) * 255).astype(np.uint8))
            elif bd == "16-bit":
                tiff.imwrite(filename, (np.clip(img_array, 0, 1) * 65535).astype(np.uint16))
            elif bd == "32-bit unsigned":
                tiff.imwrite(filename, (np.clip(img_array, 0, 1) * 4294967295).astype(np.uint32))
            elif bd == "32-bit floating point":
                tiff.imwrite(filename, img_array.astype(np.float32))
            else:
                raise ValueError(f"Unsupported bit depth for TIFF: {bd}")
            print(f"Saved {bd} TIFF image to: {filename}")
            return

        # ---------------------------------------------------------------------
        # FITS — honor bit_depth like TIFF (8/16/32U/32f)
        # ---------------------------------------------------------------------
        if fmt in ("fit", "fits"):
            # Helper to build minimal valid header
            def _minimal_fits_header(h: int, w: int, is_rgb: bool) -> fits.Header:
                hdr = fits.Header()
                hdr["SIMPLE"] = True
                hdr["BITPIX"] = -32  # will be overridden below if needed
                hdr["NAXIS"]  = 3 if is_rgb else 2
                hdr["NAXIS1"] = w
                hdr["NAXIS2"] = h
                if is_rgb:
                    hdr["NAXIS3"] = 3
                hdr["BSCALE"] = 1.0
                hdr["BZERO"]  = 0.0
                hdr["CREATOR"] = "Seti Astro Suite Pro"
                hdr.add_history("Written by Seti Astro Suite Pro")
                return hdr

            h, w = img_array.shape[:2]
            is_rgb = (img_array.ndim == 3 and img_array.shape[2] == 3)

            # Build base header (same as before)
            if is_xisf:
                fits_header = fits.Header()
                props = None
                if isinstance(image_meta, dict):
                    props = image_meta.get("XISFProperties")
                elif isinstance(image_meta, list) and image_meta and isinstance(image_meta[0], dict):
                    props = image_meta[0].get("XISFProperties")
                if isinstance(props, dict):
                    try:
                        if "PCL:AstrometricSolution:ReferenceCoordinates" in props:
                            ra, dec = props["PCL:AstrometricSolution:ReferenceCoordinates"]["value"]
                            fits_header["CRVAL1"] = ra
                            fits_header["CRVAL2"] = dec
                        if "PCL:AstrometricSolution:ReferenceLocation" in props:
                            cx, cy = props["PCL:AstrometricSolution:ReferenceLocation"]["value"]
                            fits_header["CRPIX1"] = cx
                            fits_header["CRPIX2"] = cy
                        if "PCL:AstrometricSolution:PixelSize" in props:
                            px = props["PCL:AstrometricSolution:PixelSize"]["value"]
                            fits_header["CDELT1"] = -px / 3600.0
                            fits_header["CDELT2"] =  px / 3600.0
                        if "PCL:AstrometricSolution:LinearTransformationMatrix" in props:
                            m = props["PCL:AstrometricSolution:LinearTransformationMatrix"]["value"]
                            fits_header["CD1_1"] = m[0][0]; fits_header["CD1_2"] = m[0][1]
                            fits_header["CD2_1"] = m[1][0]; fits_header["CD2_2"] = m[1][1]
                    except Exception:
                        pass
                fits_header.setdefault("CTYPE1", "RA---TAN")
                fits_header.setdefault("CTYPE2", "DEC--TAN")

            elif _is_header_obj(original_header):
                # Clean up invalid cards
                if isinstance(original_header, fits.Header):
                    safe_header = _drop_invalid_cards(original_header)
                    src_items = safe_header.items()
                else:
                    safe_header = original_header
                    src_items = safe_header.items()

                fits_header = fits.Header()
                for key, value in src_items:
                    if isinstance(key, str) and key.startswith("XISF:"):
                        continue
                    if key in ("RANGE_LOW", "RANGE_HIGH"):
                        continue
                    if isinstance(value, dict) and "value" in value:
                        value = value["value"]
                    try:
                        fits_header[key] = value
                    except Exception:
                        pass
            else:
                fits_header = _minimal_fits_header(h, w, is_rgb)

            # 🔥 Merge explicit WCS header from metadata, if present
            from astropy.io import fits as _fits_mod
            if isinstance(wcs_header, _fits_mod.Header):
                for key, value in wcs_header.items():
                    if key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                               "NAXIS3", "BSCALE", "BZERO", "EXTEND", "END"):
                        continue
                    try:
                        fits_header[key] = value
                    except Exception:
                        pass

            # --- Shape + base data (float), then quantize based on bit_depth ---
            if is_rgb:
                base_data = np.transpose(img_array, (2, 0, 1))  # (3, H, W)
                fits_header["NAXIS"]  = 3
                fits_header["NAXIS1"] = w
                fits_header["NAXIS2"] = h
                fits_header["NAXIS3"] = 3
            else:
                if img_array.ndim == 3 and img_array.shape[2] == 1:
                    base_data = img_array[:, :, 0]
                else:
                    base_data = img_array
                fits_header["NAXIS"]  = 2
                fits_header["NAXIS1"] = w
                fits_header["NAXIS2"] = h
                fits_header.pop("NAXIS3", None)

            bd = (bit_depth or "32-bit floating point").lower()

            if bd == "8-bit":
                data_to_write = (np.clip(base_data, 0, 1) * 255).astype(np.uint8)
                fits_header["BITPIX"] = 8
            elif bd == "16-bit":
                data_to_write = (np.clip(base_data, 0, 1) * 65535).astype(np.uint16)
                fits_header["BITPIX"] = 16
            elif bd == "32-bit unsigned":
                data_to_write = (np.clip(base_data, 0, 1) * 4294967295).astype(np.uint32)
                fits_header["BITPIX"] = 32
            else:
                # default / 32-bit float
                data_to_write = base_data.astype(np.float32)
                fits_header["BITPIX"] = -32

            # Linear scaling for all these
            fits_header["BSCALE"] = 1.0
            fits_header["BZERO"]  = 0.0

            # --- Write with the same robust path you already had ---
            hdu = fits.PrimaryHDU(data_to_write, header=fits_header)

            try:
                hdu.writeto(filename, overwrite=True)
            except VerifyError as ve:
                print(f"FITS header verify error while saving {filename}: {ve}")
                print("Attempting header auto-fix via hdu.verify('fix') and manual cleanup...")
                try:
                    hdu.verify('fix')
                except Exception as ve2:
                    print(f"hdu.verify('fix') raised: {ve2}")

                bad_keys = []
                for card in list(hdu.header.cards):
                    try:
                        _ = str(card)
                    except Exception:
                        bad_keys.append(card.keyword)
                for key in bad_keys:
                    try:
                        del hdu.header[key]
                        print(f"Dropped invalid FITS header card {key!r}")
                    except Exception:
                        pass

                try:
                    hdu.writeto(filename, overwrite=True)
                except VerifyError as ve3:
                    print(f"Still failing after cleanup: {ve3}")
                    print("Falling back to minimal FITS header (dropping all original cards).")
                    clean_header = _minimal_fits_header(h, w, is_rgb)
                    hdu2 = fits.PrimaryHDU(data_to_write.astype(np.float32), header=clean_header)
                    hdu2.writeto(filename, overwrite=True)

            print(f"Saved FITS image to: {filename}")
            return
        # ---------------------------------------------------------------------
        # RAW inputs — not writable; convert to FITS (float32)
        # ---------------------------------------------------------------------
        if fmt in ("cr2", "nef", "arw", "dng", "orf", "rw2", "pef"):
            print("RAW formats are not writable. Saving as FITS instead.")
            filename = f"{base}.fits"

            fits_header = fits.Header()
            if _is_header_obj(original_header):
                src_items = (original_header.items()
                             if isinstance(original_header, fits.Header)
                             else original_header.items())
                for k, v in src_items:
                    try:
                        fits_header[k] = v
                    except Exception:
                        pass

            fits_header["BSCALE"] = 1.0
            fits_header["BZERO"]  = 0.0
            fits_header["BITPIX"] = -32

            if is_mono:
                data = (img_array[:, :, 0] if (img_array.ndim == 3 and img_array.shape[2] == 1) else img_array)
                img_array_fits = data.astype(np.float32)
                fits_header["NAXIS"]  = 2
                fits_header["NAXIS1"] = img_array.shape[1]
                fits_header["NAXIS2"] = img_array.shape[0]
                fits_header.pop("NAXIS3", None)
            else:
                img_array_transposed = np.transpose(img_array, (2, 0, 1))  # (C,H,W)
                img_array_fits = img_array_transposed.astype(np.float32)
                fits_header["NAXIS"]  = 3
                fits_header["NAXIS1"] = img_array_transposed.shape[2]
                fits_header["NAXIS2"] = img_array_transposed.shape[1]
                fits_header["NAXIS3"] = img_array_transposed.shape[0]

            hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
            hdu.writeto(filename, overwrite=True)
            print(f"RAW processed and saved as FITS to: {filename}")
            return

        # ---------------------------------------------------------------------
        # XISF — use XISF.write; manage metadata shapes
        # ---------------------------------------------------------------------
        if fmt == "xisf":
            print(f"Original image shape: {img_array.shape}, dtype: {img_array.dtype}")
            print(f"Bit depth: {bit_depth}")

            bd = bit_depth or "32-bit floating point"
            if bd == "16-bit":
                processed_image = (np.clip(img_array, 0, 1) * 65535).astype(np.uint16)
            elif bd == "32-bit unsigned":
                processed_image = (np.clip(img_array, 0, 1) * 4294967295).astype(np.uint32)
            else:
                processed_image = img_array.astype(np.float32)

            # Normalize metadata shape hints
            if is_mono:
                if processed_image.ndim == 3 and processed_image.shape[2] > 1:
                    processed_image = processed_image[:, :, 0]
                if processed_image.ndim == 2:
                    processed_image = processed_image[:, :, np.newaxis]  # H, W, 1

                if not isinstance(image_meta, list):
                    image_meta = [{}]
                image_meta[0].setdefault("geometry", (processed_image.shape[1], processed_image.shape[0], 1))
                image_meta[0]["colorSpace"] = "Gray"
            else:
                if not isinstance(image_meta, list):
                    image_meta = [{}]
                ch = processed_image.shape[2] if processed_image.ndim == 3 else 1
                image_meta[0].setdefault("geometry", (processed_image.shape[1], processed_image.shape[0], ch))
                image_meta[0]["colorSpace"] = "RGB" if ch >= 3 else "Gray"

            if file_meta is None:
                file_meta = {}

            print(f"Processed image shape for XISF: {processed_image.shape}, dtype: {processed_image.dtype}")

            XISF.write(
                filename,
                processed_image,
                creator_app="Seti Astro Cosmic Clarity",
                image_metadata=image_meta[0],
                xisf_metadata=file_meta,
                shuffle=True
            )
            print(f"Saved {bd} XISF image to: {filename}")
            return

        # ---------------------------------------------------------------------
        # Unknown format
        # ---------------------------------------------------------------------
        raise ValueError(f"Unsupported file format: {original_format!r}")

    except Exception as e:
        print(f"Error saving image to {filename}: {e}")
        raise


def ensure_native_byte_order(array):
    """
    Ensures that the array is in the native byte order.
    If the array is in a non-native byte order, it will convert it.
    """
    if array.dtype.byteorder == '=':  # Already in native byte order
        return array
    elif array.dtype.byteorder in ('<', '>'):  # Non-native byte order
        return array.byteswap().view(array.dtype.newbyteorder('='))
    return array
