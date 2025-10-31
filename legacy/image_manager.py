#legacy.image_manager.py
# --- required imports for this module ---
import os, time, gzip
from io import BytesIO
from typing import Optional, Dict

import numpy as np
from PIL import Image
import tifffile as tiff
from astropy.io import fits
# add this near your other optional imports


try:
    import rawpy

except Exception:
    rawpy = None  # optional; RAW loading will raise if it's None

from xisf import XISF

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
        try:
            if getattr(raw, "camera_whitebalance", None) is not None:
                hdr["CAMWB0"] = float(raw.camera_whitebalance[0])
        except Exception:
            pass
        for key, attr in (("EXPTIME", "shutter"),
                          ("ISO", "iso_speed"),
                          ("FOCAL", "focal_len"),
                          ("TIMESTAMP", "timestamp")):
            if hasattr(raw, attr):
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
                output_color=rawpy.ColorSpace.raw,  # raw color space (no matrix)
                user_flip=0,               # no rotation
            )
            img = rgb16.astype(np.float32) / 65535.0  # HxWx3
            hdr = fits.Header()
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
                hdr["RAW_PREV"] = (True, "Embedded JPEG preview (no linear RAW data)")
                return img, hdr, "8-bit preview (JPEG from RAW)", is_mono
        except Exception as e4:
            print(f"[rawpy] extract_thumb failed: {e4}")

    raise RuntimeError("RAW decode failed (rawpy).")



def load_image(filename, max_retries=3, wait_seconds=3):
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

                    elif image_data.dtype == np.int32:
                        bit_depth = "32-bit signed"
                        print("Identified 32-bit signed FITS image.")
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image_data.astype(np.float32) * bscale + bzero

                    elif image_data.dtype == np.uint32:
                        bit_depth = "32-bit unsigned"
                        print("Identified 32-bit unsigned FITS image.")
                        bzero  = original_header.get('BZERO', 0)
                        bscale = original_header.get('BSCALE', 1)
                        image = image_data.astype(np.float32) * bscale + bzero

                    elif image_data.dtype == np.float32:
                        bit_depth = "32-bit floating point"
                        print("Identified 32-bit floating point FITS image.")
                        image = np.array(image_data, dtype=np.float32, copy=True, order="C")
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
                    return image, original_header, bit_depth, is_mono



            elif filename.lower().endswith(('.tiff', '.tif')):
                print(f"Loading TIFF file: {filename}")
                image_data = tiff.imread(filename)
                print(f"Loaded TIFF image with dtype: {image_data.dtype}")

                # Determine bit depth and normalize
                if image_data.dtype == np.uint8:
                    bit_depth = "8-bit"
                    image = image_data.astype(np.float32) / 255.0
                elif image_data.dtype == np.uint16:
                    bit_depth = "16-bit"
                    image = image_data.astype(np.float32) / 65535.0
                elif image_data.dtype == np.uint32:
                    bit_depth = "32-bit unsigned"
                    image = image_data.astype(np.float32) / 4294967295.0
                elif image_data.dtype == np.float32:
                    bit_depth = "32-bit floating point"
                    image = image_data
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
                props = image_meta.get('XISFProperties', {})
                hdr   = fits.Header()
                _filled = set()

                # 1) PixInsight astrometric solution
                try:
                    im0, im1 = props['PCL:AstrometricSolution:ReferenceImageCoordinates']['value']
                    w0,  w1  = props['PCL:AstrometricSolution:ReferenceCelestialCoordinates']['value']
                    hdr['CRPIX1'], hdr['CRPIX2'] = float(im0), float(im1)
                    hdr['CRVAL1'], hdr['CRVAL2'] = float(w0), float(w1)
                    hdr['CTYPE1'], hdr['CTYPE2'] = 'RA---TAN-SIP','DEC--TAN-SIP'
                    _filled |= {'CRPIX1','CRPIX2','CRVAL1','CRVAL2','CTYPE1','CTYPE2'}
                    print("🔷 Injected CRPIX/CRVAL from XISFProperties")
                except KeyError:
                    print("⚠️ Missing reference coords in XISFProperties")

                # 2) CD matrix
                try:
                    lin = np.asarray(props['PCL:AstrometricSolution:LinearTransformationMatrix']['value'], float)
                    hdr['CD1_1'], hdr['CD1_2'] = lin[0,0], lin[0,1]
                    hdr['CD2_1'], hdr['CD2_2'] = lin[1,0], lin[1,1]
                    _filled |= {'CD1_1','CD1_2','CD2_1','CD2_2'}
                    print("🔷 Injected CD matrix from XISFProperties")
                except KeyError:
                    print("⚠️ Missing CD matrix in XISFProperties")

                # 3) SIP polynomial fitting
                try:
                    gx = np.array(props['PCL:AstrometricSolution:SplineWorldTransformation:'
                                        'PointGridInterpolation:ImageToNative:GridX']['value'], dtype=float)
                    gy = np.array(props['PCL:AstrometricSolution:SplineWorldTransformation:'
                                        'PointGridInterpolation:ImageToNative:GridY']['value'], dtype=float)
                    grid = np.stack([gx, gy], axis=-1)
                    crpix = (hdr['CRPIX1'], hdr['CRPIX2'])
                    def fit_sip(grid, cr, order):
                        rows, cols, _ = grid.shape
                        u = np.repeat(np.arange(cols), rows) - cr[0]
                        v = np.tile(np.arange(rows), cols)   - cr[1]
                        dx = grid[:,:,0].ravel(); dy = grid[:,:,1].ravel()
                        terms = [(i,j) for i in range(order+1) for j in range(order+1-i) if (i,j)!=(0,0)]
                        M = np.vstack([(u**i)*(v**j) for (i,j) in terms]).T
                        a, *_ = np.linalg.lstsq(M, dx, rcond=None)
                        b, *_ = np.linalg.lstsq(M, dy, rcond=None)
                        rms = np.hypot(dx - M.dot(a), dy - M.dot(b)).std()
                        return a, b, terms, rms

                    best = {'order':None, 'rms':np.inf}
                    for order in range(2,7):
                        a, b, terms, rms = fit_sip(grid, crpix, order)
                        if rms < best['rms']:
                            best.update(order=order, a=a, b=b, terms=terms, rms=rms)
                    o = best['order']
                    hdr['A_ORDER'] = o; hdr['B_ORDER'] = o
                    _filled |= {'A_ORDER','B_ORDER'}
                    for (i,j), coef in zip(best['terms'], best['a']):
                        hdr[f'A_{i}_{j}'] = float(coef)
                        _filled.add(f'A_{i}_{j}')
                    for (i,j), coef in zip(best['terms'], best['b']):
                        hdr[f'B_{i}_{j}'] = float(coef)
                        _filled.add(f'B_{i}_{j}')
                    print(f"🔷 Injected SIP order {o}")
                except KeyError:
                    print("⚠️ No SIP grid in XISFProperties; skipping SIP")

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
                return image, original_header, bit_depth, is_mono

            elif filename.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
                print(f"Loading RAW file: {filename}")

                try:
                    image, original_header, bit_depth, is_mono = _try_load_raw_with_rawpy(
                        filename,
                        allow_thumb_preview=True,   # set False if you *never* want JPEG preview
                        debug_thumb=True
                    )
                    image = _finalize_loaded_image(image)

                    if original_header is None:
                        original_header = fits.Header()
                    # If preview path returned a minimal header, that's fine—upstream UI will message it

                    # Message what happened
                    if "preview" in str(bit_depth).lower():
                        print("RAW decode failed; using embedded JPEG preview (non-linear, 8-bit).")
                    else:
                        pass

                    return image, original_header, bit_depth, is_mono

                except Exception as e_raw:
                    print(f"rawpy failed: {e_raw}")
                    # No other in-process fallback; bail out
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

def save_image(img_array,
               filename,
               original_format,
               bit_depth=None,
               original_header=None,
               is_mono=False,
               image_meta=None,
               file_meta=None):
    """
    Save an image array to a file in the specified format and bit depth.
    - Robust to mis-ordered positional args (header/bit_depth swap).
    - Never calls .keys() on a non-mapping.
    - FITS always written as float32; header is sanitized or synthesized.
    """
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
        # FITS — always float32, sanitize or synthesize header
        # ---------------------------------------------------------------------
        if fmt in ("fit", "fits"):
            # Helper to build minimal valid header
            def _minimal_fits_header(h: int, w: int, is_rgb: bool) -> fits.Header:
                hdr = fits.Header()
                hdr["SIMPLE"] = True
                hdr["BITPIX"] = -32
                hdr["NAXIS"] = 3 if is_rgb else 2
                hdr["NAXIS1"] = w
                hdr["NAXIS2"] = h
                if is_rgb:
                    hdr["NAXIS3"] = 3
                hdr["BSCALE"] = 1.0
                hdr["BZERO"] = 0.0
                hdr["CREATOR"] = "Seti Astro Suite Pro"
                hdr.add_history("Written by Seti Astro Suite Pro")
                return hdr

            h, w = img_array.shape[:2]
            is_rgb = (img_array.ndim == 3 and img_array.shape[2] == 3)

            # Build header
            if is_xisf:
                fits_header = fits.Header()
                # read props from dict or list-of-dicts
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
                # Sanitize provided header (FITS.Header or dict)
                src_items = (original_header.items()
                             if isinstance(original_header, fits.Header)
                             else original_header.items())
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
                        # skip keys astropy can't serialize
                        pass
            else:
                fits_header = _minimal_fits_header(h, w, is_rgb)

            # Ensure dimensional + datatype keywords match what we write
            fits_header["BSCALE"] = 1.0
            fits_header["BZERO"]  = 0.0
            fits_header["BITPIX"] = -32

            if is_rgb:
                data_to_write = np.transpose(img_array, (2, 0, 1)).astype(np.float32)  # (3,H,W)
                fits_header["NAXIS"]  = 3
                fits_header["NAXIS1"] = w
                fits_header["NAXIS2"] = h
                fits_header["NAXIS3"] = 3
            else:
                if img_array.ndim == 3 and img_array.shape[2] == 1:
                    data = img_array[:, :, 0]
                else:
                    data = img_array
                data_to_write = data.astype(np.float32)
                fits_header["NAXIS"]  = 2
                fits_header["NAXIS1"] = w
                fits_header["NAXIS2"] = h
                if "NAXIS3" in fits_header:
                    del fits_header["NAXIS3"]

            hdu = fits.PrimaryHDU(data_to_write, header=fits_header)
            hdu.writeto(filename, overwrite=True)
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
