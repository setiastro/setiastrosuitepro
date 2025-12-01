# pro/gui/mixins/mask_mixin.py
"""
Mask management mixin for AstroSuiteProMainWindow.

This mixin contains all functionality for creating, managing, and 
manipulating masks on document images.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    pass


class MaskMixin:
    """
    Mixin for mask management.
    
    Provides methods for creating, applying, inverting, and removing masks
    from document images.
    """

    def _show_mask_overlay(self):
        """Show the mask overlay on the active view."""
        vw = self._active_view()
        if not vw:
            return
        # require an active mask on this doc
        doc = getattr(vw, "document", None)
        has_mask = bool(doc and getattr(doc, "active_mask_id", None))
        if not has_mask:
            QMessageBox.information(self, "Mask Overlay", "No active mask on this image.")
            return
        vw.show_mask_overlay = True
        # ensure visuals are up-to-date immediately
        try:
            vw._set_mask_highlight(True)
        except Exception:
            pass
        vw._render(rebuild=True)
        self._refresh_mask_action_states()

    def _hide_mask_overlay(self):
        """Hide the mask overlay on the active view."""
        vw = self._active_view()
        if not vw:
            return
        vw.show_mask_overlay = False
        vw._render(rebuild=True)
        self._refresh_mask_action_states()

    def _invert_mask(self):
        """Invert the active mask on the current document."""
        doc = self._active_doc()
        if not doc:
            return
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return
        layer = (getattr(doc, "masks", {}) or {}).get(mid)
        if layer is None or getattr(layer, "data", None) is None:
            return

        m = np.asarray(layer.data)
        if m.size == 0:
            return

        # invert (preserve dtype)
        if m.dtype.kind in "ui":
            maxv = np.iinfo(m.dtype).max
            layer.data = (maxv - m).astype(m.dtype, copy=False)
        else:
            layer.data = (1.0 - m.astype(np.float32, copy=False)).clip(0.0, 1.0)

        # notify listeners (triggers ImageSubWindow.render via your existing hookup)
        if hasattr(doc, "changed"):
            doc.changed.emit()

        # and explicitly refresh the active view overlay right now
        vw = self._active_view()
        if vw and hasattr(vw, "refresh_mask_overlay"):
            vw.refresh_mask_overlay()

        # keep menu states tidy
        if hasattr(self, "_refresh_mask_action_states"):
            self._refresh_mask_action_states()

    def _action_create_mask(self):
        """Create a new mask from the current document."""
        from pro.masks_core import create_mask_and_attach
        
        doc = self._current_document()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        created = create_mask_and_attach(self, doc)
        # Optional toast/log
        if created and hasattr(self, "_log"):
            self._log("Mask created and set active.")

    def _list_candidate_mask_sources(self, exclude_doc=None):
        """Return list of open documents that can serve as mask sources."""
        return [d for d in self._list_open_docs() if d is not exclude_doc]

    def _prepare_mask_array(self, src_img, target_hw, invert=False, feather_px=0.0):
        """
        Prepare a mask array from source image.
        
        Args:
            src_img: Source image array
            target_hw: Target (height, width) tuple
            invert: Whether to invert the mask
            feather_px: Feather radius in pixels
            
        Returns:
            Prepared mask as float32 array
        """
        a = np.asarray(src_img)
        if a.ndim == 3:
            a = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2])
        elif a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        a = a.astype(np.float32, copy=False)
        if a.dtype.kind in "ui":
            a /= float(np.iinfo(a.dtype).max)
        else:
            mx = float(a.max()) if a.size else 1.0
            if mx > 1.0:
                a /= mx
        a = np.clip(a, 0.0, 1.0)

        th, tw = target_hw
        sh, sw = a.shape[:2]
        if (sh, sw) != (th, tw):
            yi = (np.linspace(0, sh - 1, th)).astype(np.int32)
            xi = (np.linspace(0, sw - 1, tw)).astype(np.int32)
            a = a[yi][:, xi]
        if invert:
            a = 1.0 - a
        if feather_px and feather_px > 0.5:
            k = max(1, min(int(round(feather_px)), 64))
            w = np.ones((k,), dtype=np.float32) / float(k)
            a = np.apply_along_axis(lambda r: np.convolve(r, w, mode='same'), 1, a)
            a = np.apply_along_axis(lambda c: np.convolve(c, w, mode='same'), 0, a)
            a = np.clip(a, 0.0, 1.0)
        return a.astype(np.float32, copy=False)

    def _attach_mask_to_document(self, target_doc, mask_doc, *, name="Mask", mode="replace", invert=False, feather=0.0):
        """
        Attach a mask from mask_doc to target_doc.
        
        Args:
            target_doc: Document to attach mask to
            mask_doc: Document to use as mask source
            name: Name for the mask layer
            mode: Mask blend mode
            invert: Whether to invert the mask
            feather: Feather radius in pixels
            
        Returns:
            True if successful, False otherwise
        """
        if getattr(target_doc, "image", None) is None or getattr(mask_doc, "image", None) is None:
            return False
        th, tw = target_doc.image.shape[:2]
        mask_arr = self._prepare_mask_array(mask_doc.image, (th, tw), invert=invert, feather_px=feather)

        try:
            from pro.masks_core import MaskLayer
        except Exception:
            from uuid import uuid4

            class MaskLayer:
                def __init__(self, name, data, mode="replace", opacity=1.0):
                    self.id = f"mask-{uuid4().hex[:8]}"
                    self.name = name
                    self.data = data
                    self.mode = mode
                    self.opacity = opacity

        layer = MaskLayer(id=name, name=name, data=mask_arr, mode=mode, opacity=1.0)
        try:
            target_doc.add_mask(layer, make_active=True)
        except Exception:
            if not hasattr(target_doc, "masks"):
                target_doc.masks = {}
            target_doc.masks[layer.id] = layer
            target_doc.active_mask_id = layer.id
            target_doc.changed.emit()

        md = target_doc.metadata.setdefault("masks_meta", {})
        md[layer.id] = {"name": name, "mode": mode, "invert": bool(invert), "feather": float(feather)}
        target_doc.changed.emit()
        return True

    def _apply_mask_menu(self):
        """Show dialog to apply a mask from another document."""
        target_doc = self._active_doc()
        if not target_doc:
            QMessageBox.information(self, "Mask", "No active document.")
            return

        candidates = self._list_candidate_mask_sources(exclude_doc=target_doc)
        if not candidates:
            QMessageBox.information(self, "Mask", "Open another image to use as a mask.")
            return

        # If there are multiple, ask which one to use
        mask_doc = None
        if len(candidates) == 1:
            mask_doc = candidates[0]
        else:
            from PyQt6.QtWidgets import QInputDialog
            names = [f"{i + 1}. {d.display_name()}" for i, d in enumerate(candidates)]
            choice, ok = QInputDialog.getItem(self, "Choose Mask Image",
                                              "Use this image as mask:", names, 0, False)
            if not ok:
                return
            idx = names.index(choice)
            mask_doc = candidates[idx]

        name = mask_doc.display_name() or "Mask"
        ok = self._attach_mask_to_document(target_doc, mask_doc,
                                           name=name, mode="replace",
                                           invert=False, feather=0.0)
        if ok and hasattr(self, "_log"):
            self._log(f"Mask '{name}' applied to '{target_doc.display_name()}'")

        # Force views to update title/overlay immediately
        if ok:
            try:
                target_doc.changed.emit()
            except Exception:
                pass

        self._refresh_mask_action_states()

    def _resolve_mask_source_doc_from_payload(self, payload: dict):
        """
        Robustly resolve the source document for a mask drop using any of:
        - doc_ptr / mask_doc_ptr (legacy pointer)
        - doc_uid / base_doc_uid
        - file_path
        """
        # 1) Try pointer first, using the existing helper if present
        ptr = payload.get("doc_ptr") or payload.get("mask_doc_ptr")
        if ptr and hasattr(self, "_doc_by_ptr"):
            try:
                doc = self._doc_by_ptr(ptr)
                if doc is not None:
                    return doc
            except Exception:
                pass

        # 2) Fall back to uid-based matching
        uid = payload.get("doc_uid") or payload.get("base_doc_uid")
        file_path = payload.get("file_path")

        try:
            open_docs = self._list_open_docs()
        except Exception:
            open_docs = []

        for d in open_docs:
            # A) uid match
            if uid and getattr(d, "uid", None) == uid:
                return d

        if file_path:
            # B) file_path match as last resort
            for d in open_docs:
                meta = getattr(d, "metadata", {}) or {}
                if meta.get("file_path") == file_path:
                    return d

        return None


    def _remove_mask_menu(self):
        """Remove the active mask from the current document."""
        doc = self._active_doc()
        if not doc:
            return
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            QMessageBox.information(self, "Mask", "No active mask to remove.")
            return
        try:
            doc.remove_mask(mid)
            doc.changed.emit()
            if hasattr(self, "_log"):
                self._log(f"Removed active mask from '{doc.display_name()}'")
        except Exception:
            ...
        # If overlay was on, hide it now
        vw = self._active_view()
        if vw and getattr(vw, "show_mask_overlay", False):
            vw.show_mask_overlay = False
            vw._render(rebuild=True)

        self._refresh_mask_action_states()

    def _handle_mask_drop(self, payload: dict, target_sw):
        print("[MainWindow] _handle_mask_drop payload:", payload)
        """
        Handle mask drag-and-drop from one document to another.
        
        Args:
            payload: Dict with source document info
            target_sw: Target QMdiSubWindow
        """
        # --- Resolve the source mask doc robustly ---
        src_doc = self._resolve_mask_source_doc_from_payload(payload)
        if not src_doc:
            print("[MainWindow] _handle_mask_drop: could not resolve src_doc from payload")
            return

        # --- Resolve the target view/doc ---
        target_view = None
        if target_sw is not None:
            try:
                root = target_sw.widget()
            except Exception:
                root = None

            if root is not None:
                # Try direct: is the root itself an ImageSubWindow?
                try:
                    from pro.subwindow import ImageSubWindow  # local import to avoid cycles
                    if isinstance(root, ImageSubWindow):
                        target_view = root
                    else:
                        # Search inside the container for the first ImageSubWindow
                        target_view = root.findChild(ImageSubWindow)
                except Exception:
                    # Fallback: just use root if it has a .document
                    if hasattr(root, "document"):
                        target_view = root

        # Last-resort: fall back to the current active view
        if target_view is None:
            try:
                target_view = self._active_view()
            except Exception:
                target_view = None

        target_doc = getattr(target_view, "document", None) if target_view else None

        # Don’t apply mask if we still don’t have a valid target doc, or if it’s the same as src
        if not target_doc or target_doc is src_doc:
            print(
                "[MainWindow] _handle_mask_drop: no valid target_doc "
                f"(target_view={target_view}, target_doc={target_doc})"
            )
            return

        # --- Apply source as mask onto target ---
        name = src_doc.display_name() or "Dropped Mask"
        ok = self._attach_mask_to_document(
            target_doc, src_doc,
            name=name, mode="replace",
            invert=False, feather=0.0,
        )

        if ok and hasattr(self, "_log"):
            self._log(f"Mask '{name}' dropped onto '{target_doc.display_name()}'")

        if ok:
            try:
                target_doc.changed.emit()
            except Exception:
                pass

        self._refresh_mask_action_states()
