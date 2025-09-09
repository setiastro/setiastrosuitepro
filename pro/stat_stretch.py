# pro/stat_stretch.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QSlider
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QCursor
import numpy as np
from PyQt6 import sip

from .doc_manager import ImageDocument
# use your existing stretch code
from imageops.stretch import stretch_mono_image, stretch_color_image

class StatisticalStretchDialog(QDialog):
    """
    Non-destructive preview; Apply commits to the document image.
    """
    def __init__(self, parent, document: ImageDocument):
        super().__init__(parent)
        self.setWindowTitle("Statistical Stretch")
        self.setModal(True)
        self.doc = document
        self._last_preview = None    # np array last computed
        self._panning = False
        self._pan_last = None  # QPoint
        self._preview_scale = 1.0       # NEW: zoom factor for preview
        self._preview_qimg = None       # NEW: store unscaled QImage for clean scaling

        # --- Controls ---
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0.01, 0.99)
        self.spin_target.setSingleStep(0.01)
        self.spin_target.setValue(0.25)
        self.spin_target.setDecimals(3)

        self.chk_linked = QCheckBox("Linked channels")
        self.chk_linked.setChecked(False)

        self.chk_normalize = QCheckBox("Normalize to [0..1]")
        self.chk_normalize.setChecked(False)

        # NEW: Curves boost
        self.chk_curves = QCheckBox("Curves boost")
        self.chk_curves.setChecked(False)

        self.curves_row = QWidget()
        cr_lay = QHBoxLayout(self.curves_row); cr_lay.setContentsMargins(0,0,0,0)
        cr_lay.setSpacing(8)
        cr_lay.addWidget(QLabel("Strength:"))
        self.sld_curves = QSlider(Qt.Orientation.Horizontal)
        self.sld_curves.setRange(0, 100)           # 0.00 … 1.00 mapped to 0…100
        self.sld_curves.setSingleStep(1)
        self.sld_curves.setPageStep(5)
        self.sld_curves.setValue(20)               # default 0.20
        self.lbl_curves_val = QLabel("0.20")
        self.sld_curves.valueChanged.connect(lambda v: self.lbl_curves_val.setText(f"{v/100:.2f}"))
        cr_lay.addWidget(self.sld_curves, 1)
        cr_lay.addWidget(self.lbl_curves_val)
        self.curves_row.setVisible(False)          # hidden until checkbox is ticked
        self.chk_curves.toggled.connect(self.curves_row.setVisible)

        # Preview area
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(320, 240))
        self.preview_label.setScaledContents(False) 
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)           # <- was True; we manage size
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Buttons
        self.btn_preview = QPushButton("Preview")
        self.btn_apply   = QPushButton("Apply")
        self.btn_close   = QPushButton("Close")

        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_close.clicked.connect(self.close)

        # --- Layout ---
        form = QFormLayout()
        form.addRow("Target median:", self.spin_target)
        form.addRow("", self.chk_linked)
        form.addRow("", self.chk_normalize)
        form.addRow("", self.chk_curves)     # NEW
        form.addRow("", self.curves_row)     # NEW


        left = QVBoxLayout()
        left.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_apply)
        row.addStretch(1)
        left.addLayout(row)
        left.addStretch(1)

        main = QHBoxLayout(self)
        main.addLayout(left, 0)
        main.addWidget(self.preview_scroll, 1)

        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_label.installEventFilter(self)

        self._populate_initial_preview()

    # ----- helpers -----
    def _get_source_float(self) -> np.ndarray:
        """
        Return a float32 array scaled into ~[0..1] for stretching.
        """
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None

        if np.issubdtype(src.dtype, np.integer):
            # Assume 16-bit astro sources by default; adjust if you prefer
            scale = 65535.0 if src.dtype.itemsize >= 2 else 255.0
            return (src.astype(np.float32) / scale).clip(0, 1)
        else:
            f = src.astype(np.float32)
            # If values are way above 1 (linear calibrated data), compress softly
            mx = float(f.max()) if f.size else 1.0
            if mx > 5.0:
                f = f / mx
            return f

    # --- MASK helpers ----------------------------------------------------
    def _active_mask_array(self) -> np.ndarray | None:
        """Return active mask as float32 [H,W] in 0..1, resized to doc image."""
        try:
            mid = getattr(self.doc, "active_mask_id", None)
            if not mid:
                return None
            layer = getattr(self.doc, "masks", {}).get(mid)
            if layer is None:
                return None

            m = np.asarray(getattr(layer, "data", None))
            if m is None or m.size == 0:
                return None

            # squeeze to 2D
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[..., 0]
            elif m.ndim == 3:  # RGB/whatever → luminance
                m = (0.2126*m[...,0] + 0.7152*m[...,1] + 0.0722*m[...,2])

            m = m.astype(np.float32, copy=False)
            # normalize if integer / out-of-range
            if m.dtype.kind in "ui":
                m /= float(np.iinfo(m.dtype).max)
            m = np.clip(m, 0.0, 1.0)

            th, tw = self.doc.image.shape[:2]
            sh, sw = m.shape[:2]
            if (sh, sw) != (th, tw):
                yi = (np.linspace(0, sh-1, th)).astype(np.int32)
                xi = (np.linspace(0, sw-1, tw)).astype(np.int32)
                m = m[yi][:, xi]

            # honor opacity if present
            opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
            if opacity < 1.0:
                m *= opacity
            return m
        except Exception:
            return None

    def _blend_with_mask(self, base: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """base/out can be mono or 3ch; mask is [H,W] in 0..1."""
        if out.ndim == 3 and out.shape[2] == 3:
            m = mask[..., None]
        else:
            m = mask
        return base * (1.0 - m) + out * m


    def _run_stretch(self) -> np.ndarray | None:
        imgf = self._get_source_float()
        if imgf is None:
            return None

        target = float(self.spin_target.value())
        linked = bool(self.chk_linked.isChecked())
        normalize = bool(self.chk_normalize.isChecked())
        apply_curves = bool(self.chk_curves.isChecked())
        curves_boost = float(self.sld_curves.value()) / 100.0

        if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
            out = stretch_mono_image(
                imgf.squeeze(),
                target_median=target,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
            )
        else:
            out = stretch_color_image(
                imgf,
                target_median=target,
                linked=linked,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
            )

        # ✅ If a mask is active, blend stretched result with original
        m = self._active_mask_array()
        if m is not None:
            base = imgf.astype(np.float32, copy=False)
            out = self._blend_with_mask(base, out, m)

        return out            


    def _set_preview_pixmap(self, arr: np.ndarray):
        vis = arr
        if vis is None or vis.size == 0:
            self.preview_label.clear()
            return

        # Ensure 3 channels for display
        if vis.ndim == 2:
            vis3 = np.stack([vis] * 3, axis=-1)
        elif vis.ndim == 3 and vis.shape[2] == 1:
            vis3 = np.repeat(vis, 3, axis=2)
        else:
            vis3 = vis

        # Convert to 8-bit RGB
        if vis3.dtype == np.uint8:
            buf8 = vis3
        elif vis3.dtype == np.uint16:
            buf8 = (vis3.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            buf8 = (np.clip(vis3, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Must be C-contiguous for QImage
        buf8 = np.ascontiguousarray(buf8)
        h, w, _ = buf8.shape
        bytes_per_line = buf8.strides[0]

        # Build QImage from raw pointer; keep references alive
        self._last_preview = buf8  # keep backing store alive
        ptr = sip.voidptr(self._last_preview.ctypes.data)
        qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self._preview_qimg = qimg
        self._update_preview_scaled()

    # ----- slots -----
    def _populate_initial_preview(self):
        # show the current (unstretched) image as baseline
        src = self._get_source_float()
        if src is not None:
            self._set_preview_pixmap(np.clip(src, 0, 1))

    def _do_preview(self):
        try:
            out = self._run_stretch()
            if out is None:
                QMessageBox.information(self, "No image", "No image is loaded in the active document.")
                return
            self._set_preview_pixmap(out)
        except Exception as e:
            QMessageBox.warning(self, "Preview failed", str(e))

    def _do_apply(self):
        try:
            out = self._run_stretch()
            if out is None:
                QMessageBox.information(self, "No image", "No image is loaded in the active document.")
                return

            # Preserve mono vs color shape (same logic you had)
            if out.ndim == 3 and out.shape[2] == 3 and (self.doc.image.ndim == 2 or self.doc.image.shape[-1] == 1):
                out = out[..., 0]

            # Build a nice step name
            target = float(self.spin_target.value())
            linked = bool(self.chk_linked.isChecked())
            normalize = bool(self.chk_normalize.isChecked())
            apply_curves = getattr(self, "chk_curves", None) and self.chk_curves.isChecked()
            curves_boost = getattr(self, "sld_curves", None) and (self.sld_curves.value() / 100.0)
            parts = [f"target={target:.2f}", "linked" if linked else "unlinked"]
            if normalize: parts.append("norm")
            if apply_curves: parts.append(f"curves={curves_boost:.2f}")
            if self._active_mask_array() is not None:
                parts.append("masked")
            step_name = f"Statistical Stretch ({', '.join(parts)})"

            # ✅ Push to undo & apply
            self.doc.apply_edit(out.astype(np.float32, copy=False), step_name=step_name)

            # Turn off display stretch to avoid double-stretch
            mw = self.parent()
            if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
                view = mw.mdi.activeSubWindow().widget()
                if getattr(view, "autostretch_enabled", False):
                    view.set_autostretch(False)

            if hasattr(mw, "_log"):
                tgt = float(self.spin_target.value())
                linked = bool(self.chk_linked.isChecked())
                norm = bool(self.chk_normalize.isChecked())
                curves_on = getattr(self, "chk_curves", None) and self.chk_curves.isChecked()
                boost = getattr(self, "slider_boost", None)
                boost_val = (boost.value()/100.0) if boost else 0.0
                mw._log(f"Applied Statistical Stretch (target={tgt:.3f}, linked={linked}, normalize={norm}, "
                        f"curves={'ON' if curves_on else 'OFF'}{', boost='+str(round(boost_val,2)) if curves_on else ''}, "
                        f"mask={'ON' if self._active_mask_array() is not None else 'OFF'})")


            # Close the dialog after applying
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Apply failed", str(e))


    def _update_preview_scaled(self):
        if self._preview_qimg is None:
            self.preview_label.clear()
            return
        sw = max(1, int(self._preview_qimg.width()  * self._preview_scale))
        sh = max(1, int(self._preview_qimg.height() * self._preview_scale))
        scaled = self._preview_qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))
        self.preview_label.resize(scaled.size())            # <- crucial for scrollbars

    def eventFilter(self, obj, ev):
        # Ctrl+wheel zoom
        if ev.type() == QEvent.Type.Wheel and (obj is self.preview_scroll.viewport() or obj is self.preview_label):
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
                self._preview_scale = max(0.05, min(self._preview_scale * factor, 8.0))
                self._update_preview_scaled()
                return True
            return False  # let the scroll area do normal scrolling

        # Click+drag pan (left or middle mouse)
        if obj is self.preview_scroll.viewport() or obj is self.preview_label:
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton):
                    self._panning = True
                    self._pan_last = ev.position().toPoint()
                    # show a "grab" cursor where the drag begins
                    if obj is self.preview_label:
                        self.preview_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    else:
                        self.preview_scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif ev.type() == QEvent.Type.MouseMove and self._panning:
                pos = ev.position().toPoint()
                delta = pos - self._pan_last
                self._pan_last = pos

                hsb = self.preview_scroll.horizontalScrollBar()
                vsb = self.preview_scroll.verticalScrollBar()
                hsb.setValue(hsb.value() - delta.x())
                vsb.setValue(vsb.value() - delta.y())
                return True

            elif ev.type() == QEvent.Type.MouseButtonRelease and self._panning:
                self._panning = False
                self._pan_last = None
                # restore cursor
                self.preview_label.unsetCursor()
                self.preview_scroll.viewport().unsetCursor()
                return True

        return super().eventFilter(obj, ev)
     