# pro/gui/mixins/geometry_mixin.py
"""
Geometry operations mixin for AstroSuiteProMainWindow.

This mixin contains all geometry-related functionality: invert, flip,
rotate, rescale, and WCS transformation handling.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox, QInputDialog, QDialog

# Import numba-accelerated functions
try:
    from setiastro.saspro.legacy.numba_utils import (
        invert_image_numba,
        flip_horizontal_numba,
        flip_vertical_numba,
        rotate_90_clockwise_numba,
        rotate_90_counterclockwise_numba,
        rotate_180_numba,
        rescale_image_numba,
    )
except ImportError:
    # Fallback stubs if numba_utils is not available
    def invert_image_numba(arr):
        return 1.0 - arr

    def flip_horizontal_numba(arr):
        return arr[:, ::-1].copy()

    def flip_vertical_numba(arr):
        return arr[::-1, :].copy()

    def rotate_90_clockwise_numba(arr):
        return np.rot90(arr, k=-1)

    def rotate_90_counterclockwise_numba(arr):
        return np.rot90(arr, k=1)

    def rotate_180_numba(arr):
        return np.rot90(arr, k=2)

    def rescale_image_numba(arr, factor):
        import cv2
        h, w = arr.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


from setiastro.saspro.wcs_update import update_wcs_after_crop

import cv2
import math

if TYPE_CHECKING:
    pass


class GeometryMixin:
    """
    Mixin for geometry operations.
    
    Provides methods for inverting, flipping, rotating, and rescaling images,
    with automatic WCS (World Coordinate System) updates when applicable.
    """

    def _apply_geom_with_wcs(self, doc, out_image: np.ndarray,
                              M_src_to_dst: np.ndarray | None,
                              step_name: str):
        """
        Apply a geometry transform to `doc` and update WCS (if present)
        using the same machinery as crop (update_wcs_after_crop).
        
        Args:
            doc: Document to apply transform to
            out_image: Transformed image array
            M_src_to_dst: 3x3 transformation matrix (source to destination)
            step_name: Name of the operation for history
        """
        out_h, out_w = out_image.shape[:2]
        meta = dict(getattr(doc, "metadata", {}) or {})

        if update_wcs_after_crop is not None and M_src_to_dst is not None:
            try:
                meta = update_wcs_after_crop(
                    meta,
                    M_src_to_dst=M_src_to_dst,
                    out_w=out_w,
                    out_h=out_h,
                )
            except Exception as e:
                print(f"[WCS-GEOM] WCS update failed for {step_name}: {e}")

        # Push the image + updated metadata back into the document
        if hasattr(doc, "apply_edit"):
            doc.apply_edit(
                out_image,
                metadata={**meta, "step_name": step_name},
                step_name=step_name,
            )
        else:
            doc.image = out_image
            try:
                setattr(doc, "metadata", {**meta, "step_name": step_name})
            except Exception:
                pass
            if hasattr(doc, "changed"):
                try:
                    doc.changed.emit()
                except Exception:
                    pass

        # If WCS was successfully refit, update_wcs_after_crop
        # will have stashed a '__wcs_debug__' payload in metadata.
        dbg = meta.get("__wcs_debug__")
        if isinstance(dbg, dict):
            try:
                self._show_wcs_update_popup(dbg, step_name=step_name)
            except Exception as e:
                print(f"[WCS-GEOM] Failed to show WCS popup for {step_name}: {e}")

    def _exec_geom_invert(self):
        """Execute invert operation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Invert"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_invert_to_doc(doc)
            self._log("Invert applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Invert", str(e))

    def _exec_geom_flip_h(self):
        """Execute horizontal flip on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Flip Horizontal"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_flip_h_to_doc(doc)
            self._log("Flip Horizontal applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Horizontal", str(e))

    def _exec_geom_flip_v(self):
        """Execute vertical flip on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Flip Vertical"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_flip_v_to_doc(doc)
            self._log("Flip Vertical applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Vertical", str(e))

    def _exec_geom_rot_cw(self):
        """Execute 90 degree clockwise rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 90° CW"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_cw_to_doc(doc)
            self._log("Rotate 90° CW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CW", str(e))

    def _exec_geom_rot_ccw(self):
        """Execute 90 degree counterclockwise rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 90° CCW"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_ccw_to_doc(doc)
            self._log("Rotate 90° CCW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CCW", str(e))

    def _exec_geom_rot_180(self):
        """Execute 180 degree rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 180°"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_180_to_doc(doc)
            self._log("Rotate 180° applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 180°", str(e))

    def _exec_geom_rot_any(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate..."), self.tr("Active view has no image."))
            return

        if cv2 is None:
            QMessageBox.warning(self, self.tr("Rotate..."), self.tr("OpenCV (cv2) is required for arbitrary rotation."))
            return

        dlg = QInputDialog(self)
        dlg.setWindowTitle(self.tr("Rotate..."))
        dlg.setLabelText(self.tr("Angle in degrees (positive = CCW):"))
        dlg.setInputMode(QInputDialog.InputMode.DoubleInput)
        dlg.setDoubleRange(-360.0, 360.0)
        dlg.setDoubleDecimals(2)
        dlg.setDoubleValue(0.0)
        dlg.setWindowFlag(Qt.WindowType.Window, True)

        try:
            from setiastro.saspro.resources import rotatearbitrary_path
            dlg.setWindowIcon(QIcon(rotatearbitrary_path))
        except Exception:
            pass

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        angle = float(dlg.doubleValue())
        try:
            self._apply_geom_rot_any_to_doc(doc, angle_deg=angle)
            self._log(f"Rotate ({angle:g}°) applied to active view")
        except Exception as e:
            QMessageBox.critical(self, self.tr("Rotate..."), str(e))


    def _exec_geom_rescale(self):
        """Execute rescale operation on active view with dialog."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rescale Image"), self.tr("Active view has no image."))
            return

        # remember last value
        if not hasattr(self, "_last_rescale_factor"):
            self._last_rescale_factor = 1.0

        dlg = QInputDialog(self)
        dlg.setWindowTitle(self.tr("Rescale Image"))
        dlg.setLabelText(self.tr("Enter scaling factor (e.g., 0.5 for 50%, 2 for 200%):"))
        dlg.setInputMode(QInputDialog.InputMode.DoubleInput)
        dlg.setDoubleRange(0.1, 10.0)
        dlg.setDoubleDecimals(2)
        dlg.setDoubleValue(self._last_rescale_factor)

        # make sure it's a true window so the icon shows on all platforms
        dlg.setWindowFlag(Qt.WindowType.Window, True)

        # set the icon from rescale_path if available
        try:
            from setiastro.saspro.resources import rescale_path
            dlg.setWindowIcon(QIcon(rescale_path))
        except Exception:
            pass

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        factor = dlg.doubleValue()

        try:
            self._apply_geom_rescale_to_doc(doc, factor=factor)
            self._last_rescale_factor = factor
            self._log(f"Rescale ({factor:g}×) applied to active view")
        except Exception as e:
            QMessageBox.critical(self, self.tr("Rescale Image"), str(e))

    # --- Geometry: headless apply-to-doc helpers ---

    def _apply_geom_invert_to_doc(self, doc):
        """Apply invert to document."""
        arr = np.asarray(doc.image, dtype=np.float32)
        out = invert_image_numba(arr)
        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name="Invert")
        else:
            doc.image = out

    def _apply_geom_flip_h_to_doc(self, doc):
        """Apply horizontal flip to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = flip_horizontal_numba(arr)

        M = np.array([
            [-1.0, 0.0, w - 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Flip Horizontal")

    def _apply_geom_flip_v_to_doc(self, doc):
        """Apply vertical flip to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = flip_vertical_numba(arr)

        M = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, h - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Flip Vertical")

    def _apply_geom_rot_cw_to_doc(self, doc):
        """Apply 90° clockwise rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_90_clockwise_numba(arr)  # out shape: (w, h)

        M = np.array([
            [0.0, -1.0, h - 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 90° Clockwise")

    def _apply_geom_rot_ccw_to_doc(self, doc):
        """Apply 90° counterclockwise rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_90_counterclockwise_numba(arr)  # out shape: (w, h)

        M = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, w - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 90° Counterclockwise")

    def _apply_geom_rot_180_to_doc(self, doc):
        """Apply 180° rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_180_numba(arr)  # out shape: (h, w)

        # 180° rotation around the image center:
        # (x, y) -> (w-1 - x, h-1 - y)
        M = np.array([
            [-1.0, 0.0, w - 1.0],
            [0.0, -1.0, h - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 180°")

    def _apply_geom_rot_any_to_doc(self, doc, *, angle_deg: float):
        if cv2 is None:
            raise RuntimeError("cv2 is required for arbitrary rotation")

        src = np.asarray(doc.image, dtype=np.float32, order="C")
        h, w = src.shape[:2]

        # Rotation about center
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5

        # OpenCV uses CCW degrees
        A2 = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)  # 2x3

        # Convert to 3x3
        M = np.array([
            [A2[0,0], A2[0,1], A2[0,2]],
            [A2[1,0], A2[1,1], A2[1,2]],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float32)

        # Compute output bounds by rotating the four corners
        corners = np.array([
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
            [0.0, h - 1.0, 1.0],
        ], dtype=np.float32).T  # 3x4

        rc = (M @ corners)  # 3x4
        xs = rc[0, :]
        ys = rc[1, :]

        min_x = float(xs.min())
        max_x = float(xs.max())
        min_y = float(ys.min())
        max_y = float(ys.max())

        out_w = int(math.ceil(max_x - min_x + 1.0))
        out_h = int(math.ceil(max_y - min_y + 1.0))
        if out_w <= 0 or out_h <= 0:
            raise RuntimeError("Invalid output size after rotation")

        # Shift so that min corner maps to (0,0)
        T = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        M = (T @ M).astype(np.float32)  # final src->dst 3x3

        # Warp
        # cv2.warpPerspective expects (W,H)
        flags = cv2.INTER_LANCZOS4
        if src.ndim == 2:
            out = cv2.warpPerspective(src, M, (out_w, out_h), flags=flags)
        else:
            # warpPerspective works on multi-channel too
            out = cv2.warpPerspective(src, M, (out_w, out_h), flags=flags)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name=f"Rotate ({angle_deg:g}°)")


    def _apply_geom_rescale_to_doc(self, doc, *, factor: float):
        """Apply rescale to document with WCS update."""
        factor = float(max(0.1, min(10.0, factor)))
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rescale_image_numba(arr, factor)

        M = np.array([
            [factor, 0.0, 0.0],
            [0.0, factor, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M,
                                  step_name=f"Rescale ({factor:g}×)")

    def _apply_geom_rescale_preset_to_doc(self, doc, preset):
        """
        Accepts flexible presets:
        - dict with 'factor' or 'scale'
        - a lone float/int
        - a '0.5x'/'2x' string
        - (factor, ...) tuple/list
        Falls back to 1.0 if unparsable.
        """
        factor = None
        try:
            if isinstance(preset, dict):
                factor = preset.get("factor", preset.get("scale", None))
            elif isinstance(preset, (float, int)):
                factor = float(preset)
            elif isinstance(preset, str):
                s = preset.strip().lower().replace("×", "x")
                if s.endswith("x"):
                    s = s[:-1]
                factor = float(s)
            elif isinstance(preset, (list, tuple)) and preset:
                factor = float(preset[0])
        except Exception:
            factor = None

        if factor is None:
            factor = getattr(self, "_last_rescale_factor", 1.0) or 1.0

        self._apply_geom_rescale_to_doc(doc, factor=factor)

    def _apply_rescale_preset_to_doc(self, doc, preset: dict):
        """
        Headless rescale for drag-and-drop / shortcut preset application.
        Expects preset like {"factor": 1.25}.
        """
        factor = float(preset.get("factor", 1.0))
        if not (0.10 <= factor <= 10.0):
            raise ValueError("Rescale factor must be between 0.10 and 10.0")

        if getattr(doc, "image", None) is None:
            raise RuntimeError("Target document has no image")

        src = np.asarray(doc.image, dtype=np.float32, order="C")
        out = rescale_image_numba(src, factor)

        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name=f"Rescale ×{factor:.2f}")
        elif hasattr(doc, "apply_numpy"):
            doc.apply_numpy(out, step_name=f"Rescale ×{factor:.2f}")
        else:
            doc.image = out
