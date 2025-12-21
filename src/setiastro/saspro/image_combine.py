# pro/image_combine.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QRect, QEvent
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QSlider,
    QCheckBox, QScrollArea, QPushButton, QDialogButtonBox, QApplication, QMessageBox
)

# NEW: optional cv2 for fast gray/resize
try:
    import cv2
except Exception:
    cv2 = None

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# ---------- helpers ----------
def _doc_name(d) -> str:
    try: return d.display_name()
    except Exception: return "Untitled"

def _rgb_to_luma(img: np.ndarray) -> np.ndarray:
    f = _to_float01(img)
    if f.ndim == 2: return f
    if f.ndim == 3 and f.shape[2] == 1: return f[..., 0]
    if f.ndim == 3 and f.shape[2] == 3:
        w = _LUMA_WEIGHTS
        return f[..., 0]*w[0] + f[..., 1]*w[1] + f[..., 2]*w[2]
    raise ValueError(f"Unsupported image shape: {img.shape}")

def _recombine_luma_into_rgb(Y: np.ndarray, RGB: np.ndarray) -> np.ndarray:
    rgb = _to_float01(RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Recombine requires RGB target.")
    w = _LUMA_WEIGHTS
    orig_Y = rgb[..., 0]*w[0] + rgb[..., 1]*w[1] + rgb[..., 2]*w[2]
    chroma = rgb / (orig_Y[..., None] + 1e-6)
    return np.clip(chroma * Y[..., None], 0.0, 1.0)

def _blend_dispatch(A: np.ndarray, B: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    A = _to_float01(A); B = _to_float01(B)
    if A.ndim == 2: A = A[..., None]
    if B.ndim == 2: B = B[..., None]
    if A.shape != B.shape:
        raise ValueError("Images must have same size/channels.")

    if mode == "Average": return np.clip(0.5*(A+B), 0.0, 1.0)
    if mode == "Blend":   return np.clip(A*(1-alpha) + B*alpha, 0.0, 1.0)
    def mix(x): return np.clip(A*(1-alpha) + x*alpha, 0.0, 1.0)

    eps = 1e-6
    if mode == "Add":        return mix(np.clip(A+B, 0.0, 1.0))
    if mode == "Subtract":   return mix(np.clip(A-B, 0.0, 1.0))
    if mode == "Multiply":   return mix(A*B)
    if mode == "Divide":     return mix(np.clip(A/(B+eps), 0.0, 1.0))
    if mode == "Screen":     return mix(1.0 - (1.0-A)*(1.0-B))
    if mode == "Overlay":    return mix(np.clip(np.where(A<=0.5, 2*A*B, 1-2*(1-A)*(1-B)), 0.0, 1.0))
    if mode == "Difference": return mix(np.abs(A-B))
    return np.clip(A*(1-alpha) + B*alpha, 0.0, 1.0)

# ---------- mask helpers ----------
def _resize_mask_nearest(m: np.ndarray, shape_hw: tuple[int,int]) -> np.ndarray:
    """Resize mask to (H,W) with nearest neighbor."""
    h, w = shape_hw
    if m.shape == (h, w):
        return m
    if cv2 is not None:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    # fallback NN without cv2
    yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
    xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
    return m[yi][:, xi].astype(np.float32, copy=False)

# ---------- dialog ----------
class ImageCombineDialog(QDialog):
    """
    Views-based Image Combine with realtime preview, zoom/pan, luma-only, and mask overlay.
    Output: replace A or create new view.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle("Image Combine")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.mw = main_window
        self.dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
        self.zoom = 1.0
        self._pan_origin = None
        self._hstart = 0; self._vstart = 0
        self._pix = None  # last preview QPixmap

        # --- UI ---
        root = QVBoxLayout(self)

        frm = QFormLayout()
        self.cbA = QComboBox(); self.cbB = QComboBox()
        frm.addRow("Source A:", self.cbA)
        frm.addRow("Source B:", self.cbB)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.cbMode = QComboBox()
        self.cbMode.addItems(["Average","Add","Subtract","Blend","Multiply","Divide","Screen","Overlay","Difference"])
        row.addWidget(self.cbMode, 1)
        row.addWidget(QLabel("Opacity:"))
        self.slAlpha = QSlider(Qt.Orientation.Horizontal); self.slAlpha.setRange(0,100); self.slAlpha.setValue(100)
        row.addWidget(self.slAlpha, 2)
        frm.addRow(row)

        # luma-only
        self.chkLuma = QCheckBox("Combine luminance only (keep A’s color)")
        frm.addRow(self.chkLuma)

        # mask overlay
        mrow = QHBoxLayout()
        self.chkOverlay = QCheckBox("Show mask overlay")
        self.chkInvert  = QCheckBox("Invert mask")
        mrow.addWidget(self.chkOverlay)
        mrow.addWidget(self.chkInvert)
        mrow.addWidget(QLabel("Overlay opacity:"))
        self.slOverlay = QSlider(Qt.Orientation.Horizontal); self.slOverlay.setRange(5,95); self.slOverlay.setValue(40)
        mrow.addWidget(self.slOverlay, 1)
        frm.addRow(mrow)
        root.addLayout(frm)

        # preview
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.lbl = QLabel(""); self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.lbl)
        root.addWidget(self.scroll, 1)

        # zoom (themed)
        zrow = QHBoxLayout()

        btnOut = themed_toolbtn("zoom-out", "Zoom Out")
        btnFit = themed_toolbtn("zoom-fit-best", "Fit to Preview")
        btnIn  = themed_toolbtn("zoom-in", "Zoom In")

        btnOut.clicked.connect(self._zoom_out)
        btnIn .clicked.connect(self._zoom_in)
        btnFit.clicked.connect(self._fit)

        zrow.addWidget(btnOut)
        zrow.addWidget(btnFit)
        zrow.addWidget(btnIn)
        root.addLayout(zrow)

        # buttons
        btns = QDialogButtonBox()
        self.btnApply = btns.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btnClose = btns.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.btnClose.clicked.connect(self.reject)
        self.btnApply.clicked.connect(self._commit)
        root.addWidget(btns)

        # hooks
        for w in (self.cbA, self.cbB, self.cbMode):
            w.currentIndexChanged.connect(self._update_preview)
        self.slAlpha.valueChanged.connect(self._update_preview)
        self.chkLuma.toggled.connect(self._update_preview)
        self.chkOverlay.toggled.connect(self._update_preview)
        self.chkInvert.toggled.connect(self._update_preview)
        self.slOverlay.valueChanged.connect(self._update_preview)
        self.scroll.viewport().installEventFilter(self)

        self._populate_docs()
        self._update_preview()

    # ---------- doc utilities ----------
    def _open_docs(self) -> list:
        if not self.dm: return []
        docs = list(getattr(self.dm, "_docs", []) or [])
        return [d for d in docs if getattr(d, "image", None) is not None]

    def _active_doc(self):
        if self.dm and hasattr(self.dm, "get_active_document"):
            return self.dm.get_active_document()
        return None

    def _populate_docs(self):
        docs = self._open_docs()
        self.cbA.blockSignals(True); self.cbB.blockSignals(True)
        self.cbA.clear(); self.cbB.clear()
        for d in docs:
            self.cbA.addItem(_doc_name(d), userData=d)
            self.cbB.addItem(_doc_name(d), userData=d)
        self.cbA.blockSignals(False); self.cbB.blockSignals(False)
        if docs:
            act = self._active_doc()
            if act in docs:
                self.cbA.setCurrentIndex(docs.index(act))
                # B defaults to “other”
                j = 0 if len(docs) < 2 else (1 if docs[0] is act else 0)
                self.cbB.setCurrentIndex(j)

    # ---------- mask helpers ----------
    def _mask01_for_doc(self, doc, *, shape_hw: tuple[int,int], channels: int | None, invert_flag: bool):
        """
        Return mask for the given doc resized to (H,W).
        If channels is 3 and mask is 2D, expand with np.repeat.
        """
        m = _active_mask_array_from_doc(doc)
        if m is None:
            # last-resort fallback to global mask manager (in case user applied a global mask)
            mm = getattr(getattr(self.mw, "image_manager", None), "mask_manager", None)
            if mm and hasattr(mm, "get_applied_mask"):
                try:
                    mg = mm.get_applied_mask()
                    if mg is not None:
                        mg = np.asarray(mg).astype(np.float32)
                        if mg.ndim == 3:
                            mg = mg.mean(axis=2)
                        if mg.max() > 1.0:
                            mg /= 255.0
                        m = np.clip(mg, 0.0, 1.0)
                except Exception:
                    m = None
        if m is None:
            return None

        m = _resize_mask_nearest(m, shape_hw)
        if invert_flag:
            m = 1.0 - m
        m = np.clip(m, 0.0, 1.0)
        if channels and channels > 1 and m.ndim == 2:
            m = np.repeat(m[:, :, None], channels, axis=2)
        return m

    def _apply_overlay(self, img, mask, opacity):
        # show protected region (A) as red wash: vis = 1 - m
        vis = 1.0 - np.clip(mask, 0.0, 1.0)
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        else:
            rgb = img
        overlay = np.zeros_like(rgb, dtype=np.float32); overlay[..., 0] = 1.0
        if vis.ndim == 2: vis = vis[..., None]
        return np.clip(rgb*(1.0 - vis*opacity) + overlay*(vis*opacity), 0.0, 1.0)

    # ---------- preview ----------
    def _update_preview(self, *_):
        A = self.cbA.currentData(); B = self.cbB.currentData()
        if not (A and B): return
        imgA = getattr(A, "image", None); imgB = getattr(B, "image", None)
        if imgA is None or imgB is None: return
        if imgA.shape[:2] != imgB.shape[:2]:
            self.lbl.setText("Images must be the same size.")
            return

        alpha = self.slAlpha.value()/100.0
        mode  = self.cbMode.currentText()

        try:
            if self.chkLuma.isChecked():
                if imgA.ndim != 3 or imgA.shape[2] != 3:
                    self.lbl.setText("Luminance mode requires RGB A."); return
                YA = _rgb_to_luma(imgA)
                YB = _rgb_to_luma(imgB)
                Ymix = _blend_dispatch(YA[...,None], YB[...,None], mode, alpha)[...,0]

                # mask from destination doc (A)
                m = self._mask01_for_doc(A, shape_hw=Ymix.shape[:2], channels=None,
                                         invert_flag=self.chkInvert.isChecked())
                if m is not None:
                    Ymix = Ymix*m + YA*(1.0 - m)

                blended = _recombine_luma_into_rgb(Ymix, imgA)

            else:
                A3 = imgA if imgA.ndim == 3 else imgA[..., None]
                B3 = imgB if imgB.ndim == 3 else imgB[..., None]
                blended = _blend_dispatch(A3, B3, mode, alpha)
                if imgA.ndim == 2:
                    blended = blended[...,0]

                # mask from destination doc (A)
                m = self._mask01_for_doc(
                    A, shape_hw=blended.shape[:2],
                    channels=(blended.shape[2] if blended.ndim == 3 else 1),
                    invert_flag=self.chkInvert.isChecked()
                )
                if m is not None:
                    blended = np.clip(blended*m + _to_float01(imgA)*(1.0 - m), 0.0, 1.0)

            # optional red overlay
            if self.chkOverlay.isChecked():
                m = self._mask01_for_doc(
                    A, shape_hw=blended.shape[:2],
                    channels=(blended.shape[2] if blended.ndim == 3 else 1),
                    invert_flag=self.chkInvert.isChecked()
                )
                if m is not None:
                    blended = self._apply_overlay(_to_float01(blended), m, self.slOverlay.value()/100.0)

            # to pixmap
            f = _to_float01(blended); h, w = f.shape[:2]
            if f.ndim == 2:
                buf = (f*255).astype(np.uint8); q = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8)
            else:
                buf = (f*255).astype(np.uint8); q = QImage(buf.data, w, h, 3*w, QImage.Format.Format_RGB888)
            self._pix = QPixmap.fromImage(q)
            self._apply_zoom()
        except Exception as e:
            self.lbl.setText(f"Error: {e}")

    # ---------- apply ----------
    def _commit(self):
        A = self.cbA.currentData(); B = self.cbB.currentData()
        if not (A and B): return
        imgA = getattr(A, "image", None); imgB = getattr(B, "image", None)
        if imgA is None or imgB is None: return
        if imgA.shape[:2] != imgB.shape[:2]:
            QMessageBox.warning(self, "Image Combine", "Image sizes must match."); return

        alpha = self.slAlpha.value()/100.0
        mode  = self.cbMode.currentText()

        try:
            if self.chkLuma.isChecked():
                YA = _rgb_to_luma(imgA); YB = _rgb_to_luma(imgB)
                Ymix = _blend_dispatch(YA[...,None], YB[...,None], mode, alpha)[...,0]

                m = self._mask01_for_doc(A, shape_hw=Ymix.shape[:2], channels=None,
                                         invert_flag=self.chkInvert.isChecked())
                if m is not None:
                    Ymix = Ymix*m + YA*(1.0 - m)

                result = _recombine_luma_into_rgb(Ymix, imgA)
                step = f"Luminance {mode}"
            else:
                A3 = imgA if imgA.ndim == 3 else imgA[..., None]
                B3 = imgB if imgB.ndim == 3 else imgB[..., None]
                result = _blend_dispatch(A3, B3, mode, alpha)
                if imgA.ndim == 2: result = result[...,0]

                m = self._mask01_for_doc(
                    A, shape_hw=result.shape[:2],
                    channels=(result.shape[2] if result.ndim == 3 else 1),
                    invert_flag=self.chkInvert.isChecked()
                )
                if m is not None:
                    result = np.clip(result*m + _to_float01(imgA)*(1.0 - m), 0.0, 1.0)
                step = f"{mode} Combine"

            result = _to_float01(result)

            # Replace A (overwrite active view) or create new?
            replace = True
            if replace:
                if hasattr(A, "set_image"):
                    A.set_image(result, step_name=f"Image Combine: {step}")
                else:
                    A.image = result
                try: self.mw._log(f"Image Combine → replaced '{_doc_name(A)}' ({step})")
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            else:
                newdoc = self.dm.create_document(result, metadata={
                    "display_name": f"Combined ({step})",
                    "bit_depth": "32-bit floating point",
                    "is_mono": (result.ndim == 2),
                    "source": f"Combine: {step}",
                }, name=f"Combined ({step})")
                self.mw._spawn_subwindow_for(newdoc)
                try: self.mw._log(f"Image Combine → new view '{_doc_name(newdoc)}' ({step})")
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        except Exception as e:
            QMessageBox.critical(self, "Image Combine", f"Failed:\n{e}")
            
    # ---------- zoom/pan ----------
    def _apply_zoom(self):
        if self._pix is None: return
        scaled = self._pix.scaled(self._pix.size()*self.zoom, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(scaled)

    def _zoom_in(self):  self.zoom *= 1.25; self._apply_zoom()
    def _zoom_out(self): self.zoom /= 1.25; self._apply_zoom()
    def _fit(self):
        if self._pix is None: return
        area = self.scroll.viewport().size(); pix = self._pix.size()
        sx = area.width()/max(1,pix.width()); sy = area.height()/max(1,pix.height())
        self.zoom = min(sx, sy, 1.0); self._apply_zoom()

    def eventFilter(self, src, ev):
        if src is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._pan_origin = ev.pos()
                self._hstart = self.scroll.horizontalScrollBar().value()
                self._vstart = self.scroll.verticalScrollBar().value()
                return True
            if ev.type() == QEvent.Type.MouseMove and self._pan_origin is not None:
                d = ev.pos() - self._pan_origin
                self.scroll.horizontalScrollBar().setValue(self._hstart - d.x())
                self.scroll.verticalScrollBar().setValue(self._vstart - d.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._pan_origin = None; return True
            return False
        return super().eventFilter(src, ev)

