# --- required imports for this module ---
import os, time, gzip
from io import BytesIO
from typing import Optional, Dict

import numpy as np
from PIL import Image
import tifffile as tiff
from astropy.io import fits
try:
    import rawpy
except Exception:
    rawpy = None  # optional; RAW loading will raise if it's None

from xisf import XISF

from PyQt6.QtCore import QObject, pyqtSignal

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
                        image = image_data
                    else:
                        raise ValueError(f"Unsupported FITS data type: {image_data.dtype}")

                    # ---------------------------------------------------------------------
                    # 2) Squeeze out any singleton dimensions (fix weird NAXIS combos)
                    # ---------------------------------------------------------------------
                    image = np.squeeze(image)

                    if image.dtype == np.float32:
                        max_val = image.max()
                        if max_val > 1.0:
                            print(f"Detected float image with max value {max_val:.3f} > 1.0; rescales to [0,1]")
                            image = image / max_val
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

                if image.dtype == np.float32:
                    max_val = image.max()
                    if max_val > 1.0:
                        print(f"Detected float image with max value {max_val:.3f} > 1.0; rescales to [0,1]")
                        image = image / max_val

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
                return image, original_header, bit_depth, is_mono

            elif filename.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
                print(f"Loading RAW file: {filename}")
                with rawpy.imread(filename) as raw:
                    # 1) Read the raw Bayer data (no demosaic)
                    bayer_image = raw.raw_image_visible.astype(np.float32)
                    print(f"Raw Bayer image dtype: {bayer_image.dtype}, "
                        f"min: {bayer_image.min():.2f}, max: {bayer_image.max():.2f}")

                    # 2) Get camera black/white levels
                    black_levels = raw.black_level_per_channel  # e.g. [512, 512, 512, 512]
                    white_level  = raw.white_level              # e.g. 16383 for 14-bit
                    avg_black = float(np.mean(black_levels))    # Simple average

                    # 3) Subtract black level, clip negatives to 0
                    bayer_image -= avg_black
                    bayer_image = np.clip(bayer_image, 0, None)

                    # 4) Divide by (white_level - black_level) to normalize to [0..1]
                    scale = float(white_level - avg_black)
                    if scale <= 0:
                        # Safety check if black >= white
                        scale = 1.0
                    bayer_image /= scale

                    # Now dark frames should hover near 0.0 instead of ~0.7

                    # 5) Check shape to decide if mono vs. color mosaic
                    #    Usually it's 2D for a raw Bayer pattern
                    if bayer_image.ndim == 2:
                        image = bayer_image
                        is_mono = True
                    elif bayer_image.ndim == 3 and bayer_image.shape[2] == 3:
                        # Rare case if raw.raw_image_visible is already color
                        image = bayer_image
                        is_mono = False
                    else:
                        raise ValueError(f"Unexpected RAW Bayer image shape: {bayer_image.shape}")

                    # 6) Assume 16-bit raw data (typical for DSLRs)
                    bit_depth = "16-bit"

                    # 7) Build a minimal header from raw metadata
                    original_header_dict = {
                        'CAMERA': raw.camera_whitebalance[0] if raw.camera_whitebalance else 'Unknown',
                        'EXPTIME': raw.shutter if hasattr(raw, 'shutter') else 0.0,
                        'ISO': raw.iso_speed if hasattr(raw, 'iso_speed') else 0,
                        'FOCAL': raw.focal_len if hasattr(raw, 'focal_len') else 0.0,
                        'DATE': raw.timestamp if hasattr(raw, 'timestamp') else 'Unknown',
                    }

                    # 8) Extract the CFA pattern
                    cfa_pattern = raw.raw_colors_visible  # 2D array of 0/1/2
                    cfa_mapping = {0: 'R', 1: 'G', 2: 'B'}
                    cfa_description = ''.join([cfa_mapping.get(color, '?')
                                            for color in cfa_pattern.flatten()[:4]])
                    original_header_dict['CFA'] = (cfa_description, 'Color Filter Array pattern')

                    # 9) Convert dict → FITS Header
                    original_header = fits.Header()
                    for key, value in original_header_dict.items():
                        original_header[key] = value

                    print(f"RAW file loaded with CFA pattern: {cfa_description}, "
                        f"dark frames ~0, bright frames ~1 now.")
                    return image, original_header, bit_depth, is_mono

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

def save_image(img_array, filename, original_format, bit_depth=None, original_header=None, is_mono=False, image_meta=None, file_meta=None):
 
    """
    Save an image array to a file in the specified format and bit depth.
    """
    img_array = ensure_native_byte_order(img_array)  # Ensure correct byte order
    is_xisf = False  # Flag to determine if the original file was XISF

    # **🔹 Detect If Original File Was XISF**
    if original_header:
        for key in original_header.keys():
            if key.startswith("XISF:"):
                is_xisf = True
                break

    if image_meta and "XISFProperties" in image_meta:
        is_xisf = True  # Confirm XISF metadata exists

    try:
        if original_format == 'png':
            img = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert to 8-bit and save as PNG
            img.save(filename)
            print(f"Saved 8-bit PNG image to: {filename}")
        elif original_format in ['jpg', 'jpeg']:
            img = Image.fromarray((img_array * 255).astype(np.uint8))  # Convert to 8-bit and save as PNG
            img.save(filename)
            print(f"Saved 8-bit JPG image to: {filename}")        
        elif original_format in ['tiff', 'tif']:
            # Save TIFF files based on bit depth
            if bit_depth == "8-bit":
                tiff.imwrite(filename, (img_array * 255).astype(np.uint8))  # Save as 8-bit TIFF
            elif bit_depth == "16-bit":
                tiff.imwrite(filename, (img_array * 65535).astype(np.uint16))  # Save as 16-bit TIFF
            elif bit_depth == "32-bit unsigned":
                tiff.imwrite(filename, (img_array * 4294967295).astype(np.uint32))  # Save as 32-bit unsigned TIFF
            elif bit_depth == "32-bit floating point":
                tiff.imwrite(filename, img_array.astype(np.float32))  # Save as 32-bit floating point TIFF
            else:
                raise ValueError("Unsupported bit depth for TIFF!")
            print(f"Saved {bit_depth} TIFF image to: {filename}")

        elif original_format in ['fits', 'fit']:
            # Preserve the original extension
            if not filename.lower().endswith(f".{original_format}"):
                filename = filename.rsplit('.', 1)[0] + f".{original_format}"

            # **📌 CASE 1: ORIGINAL FILE WAS XISF → CONVERT TO FITS HEADER**
            if is_xisf:
                print("Detected XISF metadata. Converting to FITS header...")
                fits_header = fits.Header()

                if 'XISFProperties' in image_meta:
                    xisf_props = image_meta['XISFProperties']

                    # Extract WCS parameters
                    if 'PCL:AstrometricSolution:ReferenceCoordinates' in xisf_props:
                        ref_coords = xisf_props['PCL:AstrometricSolution:ReferenceCoordinates']['value']
                        fits_header['CRVAL1'] = ref_coords[0]
                        fits_header['CRVAL2'] = ref_coords[1]

                    if 'PCL:AstrometricSolution:ReferenceLocation' in xisf_props:
                        ref_pixel = xisf_props['PCL:AstrometricSolution:ReferenceLocation']['value']
                        fits_header['CRPIX1'] = ref_pixel[0]
                        fits_header['CRPIX2'] = ref_pixel[1]

                    if 'PCL:AstrometricSolution:PixelSize' in xisf_props:
                        pixel_size = xisf_props['PCL:AstrometricSolution:PixelSize']['value']
                        fits_header['CDELT1'] = -pixel_size / 3600.0
                        fits_header['CDELT2'] = pixel_size / 3600.0

                    if 'PCL:AstrometricSolution:LinearTransformationMatrix' in xisf_props:
                        linear_transform = xisf_props['PCL:AstrometricSolution:LinearTransformationMatrix']['value']
                        fits_header['CD1_1'] = linear_transform[0][0]
                        fits_header['CD1_2'] = linear_transform[0][1]
                        fits_header['CD2_1'] = linear_transform[1][0]
                        fits_header['CD2_2'] = linear_transform[1][1]

                # Ensure essential WCS headers exist
                fits_header.setdefault('CTYPE1', 'RA---TAN')
                fits_header.setdefault('CTYPE2', 'DEC--TAN')

                print("Converted XISF metadata to FITS header.")

            # **📌 CASE 2: ORIGINAL FILE WAS FITS → PRESERVE HEADER**
            elif original_header is not None:
                print("Detected FITS format. Preserving original FITS header.")
                fits_header = fits.Header()
                for key, value in original_header.items():
                    if key.startswith("XISF:"):
                        continue  # Skip XISF metadata

                    if key in ["RANGE_LOW", "RANGE_HIGH"]:
                        print(f"Removing {key} from header to prevent overflow.")
                        continue  # Skip adding RANGE_LOW and RANGE_HIGH

                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']

                    try:
                        fits_header[key] = value
                    except Exception as e:
                        print(f"Skipping problematic key {key} due to error: {e}")
            else:
                raise ValueError("Original header is required for FITS format!")

            # **📌 Image Processing for FITS**
            fits_header['BSCALE'] = 1.0
            fits_header['BZERO'] = 0.0

            if is_mono or img_array.ndim == 2:
                img_array_fits = img_array[:, :, 0] if len(img_array.shape) == 3 else img_array
                fits_header['NAXIS'] = 2
            else:
                img_array_fits = np.transpose(img_array, (2, 0, 1))
                fits_header['NAXIS'] = 3
                fits_header['NAXIS3'] = 3

            fits_header['NAXIS1'] = img_array.shape[1]
            fits_header['NAXIS2'] = img_array.shape[0]

            # force 32-bit floats and update header
            img_array_fits = img_array_fits.astype(np.float32)
            fits_header['BITPIX'] = -32

            # **💾 Save the FITS File**
            hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
            hdu.writeto(filename, overwrite=True)
            print(f"Saved FITS image to: {filename}")
            return



        elif original_format in ['.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef']:
            # Save as FITS file with metadata
            print("RAW formats are not writable. Saving as FITS instead.")
            filename = filename.rsplit('.', 1)[0] + ".fits"

            if original_header is not None:
                # Convert original_header (dictionary) to astropy Header object
                fits_header = fits.Header()
                for key, value in original_header.items():
                    fits_header[key] = value
                fits_header['BSCALE'] = 1.0  # Scaling factor
                fits_header['BZERO'] = 0.0   # Offset for brightness    

                if is_mono:  # Grayscale FITS
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array[:, :, 0] * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = (img_array[:, :, 0].astype(np.float32) * bscale + bzero).astype(np.uint32)
                    else:  # 32-bit float
                        img_array_fits = img_array[:, :, 0].astype(np.float32)

                    # Update header for a 2D (grayscale) image
                    fits_header['NAXIS'] = 2
                    fits_header['NAXIS1'] = img_array.shape[1]  # Width
                    fits_header['NAXIS2'] = img_array.shape[0]  # Height
                    fits_header.pop('NAXIS3', None)  # Remove if present

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)
                else:  # RGB FITS
                    img_array_transposed = np.transpose(img_array, (2, 0, 1))  # Channels, Height, Width
                    if bit_depth == "16-bit":
                        img_array_fits = (img_array_transposed * 65535).astype(np.uint16)
                    elif bit_depth == "32-bit unsigned":
                        bzero = fits_header.get('BZERO', 0)
                        bscale = fits_header.get('BSCALE', 1)
                        img_array_fits = img_array_transposed.astype(np.float32) * bscale + bzero
                        fits_header['BITPIX'] = -32
                    else:  # Default to 32-bit float
                        img_array_fits = img_array_transposed.astype(np.float32)

                    # Update header for a 3D (RGB) image
                    fits_header['NAXIS'] = 3
                    fits_header['NAXIS1'] = img_array_transposed.shape[2]  # Width
                    fits_header['NAXIS2'] = img_array_transposed.shape[1]  # Height
                    fits_header['NAXIS3'] = img_array_transposed.shape[0]  # Channels

                    hdu = fits.PrimaryHDU(img_array_fits, header=fits_header)

                # Write the FITS file
                try:
                    hdu.writeto(filename, overwrite=True)
                    print(f"RAW processed and saved as FITS to: {filename}")
                except Exception as e:
                    print(f"Error saving FITS file: {e}")
            else:
                raise ValueError("Original header is required for FITS format!")

        elif original_format == 'xisf':
            try:
                print(f"Original image shape: {img_array.shape}, dtype: {img_array.dtype}")
                print(f"Bit depth: {bit_depth}")

                # Adjust bit depth for saving
                if bit_depth == "16-bit":
                    processed_image = (img_array * 65535).astype(np.uint16)
                elif bit_depth == "32-bit unsigned":
                    processed_image = (img_array * 4294967295).astype(np.uint32)
                else:  # Default to 32-bit float
                    processed_image = img_array.astype(np.float32)

                # Handle mono images explicitly
                if is_mono:
                    print("Detected mono image. Preparing for XISF...")
                    if processed_image.ndim == 3 and processed_image.shape[2] > 1:
                        processed_image = processed_image[:, :, 0]  # Extract single channel
                    processed_image = processed_image[:, :, np.newaxis]  # Add back channel dimension

                    # Update metadata for mono images
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], 1)
                        image_meta[0]['colorSpace'] = 'Gray'
                    else:
                        # Create default metadata for mono images
                        image_meta = [{
                            'geometry': (processed_image.shape[1], processed_image.shape[0], 1),
                            'colorSpace': 'Gray'
                        }]

                # Handle RGB images
                else:
                    if image_meta and isinstance(image_meta, list):
                        image_meta[0]['geometry'] = (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2])
                        image_meta[0]['colorSpace'] = 'RGB'
                    else:
                        # Create default metadata for RGB images
                        image_meta = [{
                            'geometry': (processed_image.shape[1], processed_image.shape[0], processed_image.shape[2]),
                            'colorSpace': 'RGB'
                        }]

                # Ensure fallback for `image_meta` and `file_meta`
                if image_meta is None or not isinstance(image_meta, list):
                    image_meta = [{
                        'geometry': (processed_image.shape[1], processed_image.shape[0], 1 if is_mono else 3),
                        'colorSpace': 'Gray' if is_mono else 'RGB'
                    }]
                if file_meta is None:
                    file_meta = {}

                # Debug: Print processed image details and metadata
                print(f"Processed image shape for XISF: {processed_image.shape}, dtype: {processed_image.dtype}")

                # Save the image using XISF.write
                XISF.write(
                    filename,                    # Output path
                    processed_image,             # Final processed image
                    creator_app="Seti Astro Cosmic Clarity",
                    image_metadata=image_meta[0],  # First block of image metadata
                    xisf_metadata=file_meta,       # File-level metadata
                    shuffle=True
                )

                print(f"Saved {bit_depth} XISF image to: {filename}")

            except Exception as e:
                print(f"Error saving XISF file: {e}")
                raise


        else:
            raise ValueError("Unsupported file format!")

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
