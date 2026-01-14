# pro/texture_clarity.py
from __future__ import annotations
import numpy as np
import os

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSlider, QHBoxLayout,
    QPushButton, QMessageBox, QCheckBox, QScrollArea, QWidget
)
from PyQt6.QtGui import QPixmap, QImage, QMovie

try:
    import cv2
except Exception:
    cv2 = None

# ---------- utils ----------
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)

def _as_qimage_rgb8(float01: np.ndarray) -> QImage:
    f = np.asarray(float01, dtype=np.float32)

    # Ensure 3-channel RGB for preview
    if f.ndim == 2:
        f = np.stack([f]*3, axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)

    # [0,1] -> uint8 and force C-contiguous
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    buf8 = np.ascontiguousarray(buf8)
    h, w, _ = buf8.shape
    bpl = int(buf8.strides[0])
    
    # Detach
    data = buf8.tobytes()
    qimg = QImage(data, w, h, bpl, QImage.Format.Format_RGB888)
    return qimg.copy()

def _ensure_rgb(arr: np.ndarray) -> np.ndarray | None:
    a = _to_float01(arr)
    if a is None: return None
    if a.ndim == 2: return a
    if a.ndim == 3 and a.shape[2] == 1: return a
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3].astype(np.float32, copy=False)
    return None

def _midtone_mask(image: np.ndarray) -> np.ndarray:
    """
    Generate a midtone mask where midtones (0.5) are 1.0 and shadows/highlights are 0.0.
    """
    return np.clip(1.0 - 4.0 * (image - 0.5)**2, 0.0, 1.0)

def _apply_texture(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """
    TEXTURE: Enhances 'Texture' frequency band (Difference of Gaussians).
    - Method: DoG (Band-pass). Isolate frequencies between Radius and 2*Radius.
    """
    if abs(amount) < 0.001: return image
    
    # Ensure input is valid float32 and contiguous
    img = np.ascontiguousarray(image, dtype=np.float32)
    if np.any(np.isnan(img)):
        img = np.nan_to_num(img)
    
    sigma1 = radius
    sigma2 = radius * 2.0
    
    ksize1 = int(2 * round(3 * sigma1) + 1); ksize1 += 1 if ksize1 % 2 == 0 else 0
    ksize2 = int(2 * round(3 * sigma2) + 1); ksize2 += 1 if ksize2 % 2 == 0 else 0
    
    if cv2 is not None:
        try:
            b1 = cv2.GaussianBlur(img, (ksize1, ksize1), sigma1)
            b2 = cv2.GaussianBlur(img, (ksize2, ksize2), sigma2)
        except Exception:
            # Fallback if CV2 fails
            return image
    else:
        return image
        
    texture_band = b1 - b2
    boost = 2.0 * amount 
    enhanced = img + texture_band * boost
    return np.clip(enhanced, 0.0, 1.0)

def _apply_clarity(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """
    CLARITY: Local Contrast with Edge Preservation (Bilateral).
    - Method: Original + Amount * (Original - Bilateral_Base).
    - Optimization: Uses Downscale-Process-Upscale for large radii.
      This allows effective large-radius filtering without using massive kernels that crash OpenCV.
    - Safety: Kernel diameter 'd' is kept small relative to the processed image.
    """
    if abs(amount) < 0.001: return image
    
    # Target Sigma Space
    sigma_space_target = radius * 10.0
    sigma_color = 0.1 
    
    img_f32 = np.ascontiguousarray(image, dtype=np.float32)
    if np.any(np.isnan(img_f32)):
        img_f32 = np.nan_to_num(img_f32)
        
    base = img_f32
    
    if cv2 is not None:
        try:
            # Multi-scale Logic:
            # If sigma_space is large (e.g. > 10.0), downscale the image.
            # This makes the "pixels" larger, so a small kernel covers more area.
            
            scale = 1.0
            if sigma_space_target > 10.0:
                # Calculate scale factor
                # We want the effective sigma on the downscaled image to be manageable, say ~5-10.
                # scaled_sigma = sigma_target * scale
                # scale = 5.0 / sigma_target
                scale = 5.0 / sigma_space_target
                scale = max(0.1, min(scale, 1.0)) # Limit minimum scale to 10%
            
            # If downscaling is significant
            if scale < 0.95:
                h, w = img_f32.shape[:2]
                small_w = int(w * scale)
                small_h = int(h * scale)
                
                # Resize down
                small_img = cv2.resize(img_f32, (small_w, small_h), interpolation=cv2.INTER_AREA)
                
                # Adjust sigma for the small scale
                sigma_small = sigma_space_target * scale
                
                # A safe 'd' for the small image. 
                # Since we successfully shrunk the problem, d=9 is now effectively d=9/scale in original pixels.
                # e.g with scale 0.2, d=9 covers 45 original pixels.
                d_safe = 9
                
                small_base = cv2.bilateralFilter(small_img, d=d_safe, sigmaColor=sigma_color, sigmaSpace=sigma_small)
                
                # Resize up (using linear/cubic to smooth)
                base = cv2.resize(small_base, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # Standard processing for small radii
                d_safe = 9
                base = cv2.bilateralFilter(img_f32, d=d_safe, sigmaColor=sigma_color, sigmaSpace=sigma_space_target)
                
        except Exception as e:
            print(f"Bilateral Filter failed: {e}")
            try:
                base = cv2.GaussianBlur(img_f32, (0, 0), sigma_space_target)
            except:
                return image
    else:
        return image
        
    detail = img_f32 - base
    mask = _midtone_mask(img_f32)
    enhanced = img_f32 + amount * detail * mask
    
    return np.clip(enhanced, 0.0, 1.0)

def _compute_texture_clarity(image: np.ndarray, texture_amt: float, texture_rad: float, clarity_amt: float, clarity_rad: float) -> np.ndarray:
    # 1. Texture (DoG Band)
    out = _apply_texture(image, texture_amt, texture_rad)
    
    # 2. Clarity (Bilateral Base)
    out = _apply_clarity(out, clarity_amt, clarity_rad)
    
    return out

# ---------- headless core ----------
def texture_clarity_headless(
    doc,
    texture_amount: float = 0.0,
    texture_radius: float = 1.0,
    clarity_amount: float = 0.0,
    clarity_radius: float = 1.0,
):
    if doc is None or getattr(doc, "image", None) is None:
        return

    src = np.asarray(doc.image)
    f_src = _to_float01(src)
    if f_src is None:
        return

    is_rgb = (f_src.ndim == 3 and f_src.shape[2] >= 3)
    
    if is_rgb:
        R, G, B = f_src[..., 0], f_src[..., 1], f_src[..., 2]
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        
        L_new = _compute_texture_clarity(L, texture_amount, texture_radius, clarity_amount, clarity_radius)
        
        eps = 1e-7
        ratio = L_new / (L + eps)
        out = f_src[..., :3] * ratio[..., None]
        out = np.clip(out, 0.0, 1.0)
    else:
        if f_src.ndim == 3: f_src = f_src.squeeze()
        out = _compute_texture_clarity(f_src, texture_amount, texture_radius, clarity_amount, clarity_radius)
        if src.ndim == 3: out = out[..., None]

    # mask-aware blend
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        h, w = out.shape[:2]
        if m.shape != (h, w):
            if cv2 is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
                xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
                m = m[yi][:, xi]
        
        if out.ndim == 3 and m.ndim == 2:
            m = np.repeat(m[:, :, None], out.shape[2], axis=2)
            
        src_f = _to_float01(src)
        out = np.clip(src_f * (1.0 - m) + out * m, 0.0, 1.0)

    meta = {
        "step_name": "Texture and Clarity",
        "texture_clarity": {
            "texture_amount": texture_amount,
            "texture_radius": texture_radius,
            "clarity_amount": clarity_amount,
            "clarity_radius": clarity_radius
        }
    }
    doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name="Texture and Clarity")


# ---------- Worker ----------

class TextureClarityWorker(QThread):
    preview_ready = pyqtSignal(object)  # np.ndarray [0..1]

    def __init__(self, image: np.ndarray, params: dict):
        super().__init__()
        self.image = image
        self.params = params

    def run(self):
        src = _to_float01(self.image)
        if src is None: return

        # Re-implement core logic efficiently for preview
        texture_amt = self.params.get("t_amt", 0.0)
        texture_rad = self.params.get("t_rad", 1.0)
        clarity_amt = self.params.get("c_amt", 0.0)
        clarity_rad = self.params.get("c_rad", 1.0)

        is_rgb = (src.ndim == 3 and src.shape[2] >= 3)
        if is_rgb:
            R, G, B = src[..., 0], src[..., 1], src[..., 2]
            L = 0.2126 * R + 0.7152 * G + 0.0722 * B
            L_new = _compute_texture_clarity(L, texture_amt, texture_rad, clarity_amt, clarity_rad)
            eps = 1e-7
            ratio = L_new / (L + eps)
            out = src[..., :3] * ratio[..., None]
        else:
            if src.ndim == 3: src = src.squeeze()
            out = _compute_texture_clarity(src, texture_amt, texture_rad, clarity_amt, clarity_rad)
        
        self.preview_ready.emit(np.clip(out, 0.0, 1.0).astype(np.float32))

# ---------- Dialog ----------

class TextureClarityDialog(QDialog):
    def __init__(self, main, doc, parent=None):
        super().__init__(parent)
        self.main = main
        self.doc = doc
        self.setWindowTitle("Texture and Clarity")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._preview = None
        self._pix = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()

        # Watch for active document changes
        self._connected_doc_change = False
        if hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_doc_change = True
            except Exception:
                pass

        # Debounce timer for preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150) # 150ms debounce
        self._preview_timer.timeout.connect(self._trigger_preview)

        self._build_ui()
        # Initial preview
        self._trigger_preview()

    def _build_ui(self):
        container = QHBoxLayout(self)

        # Left Column: Controls
        left_widget = QWidget()
        left_widget.setMinimumWidth(350)
        left = QVBoxLayout(left_widget)
        
        # Texture
        left.addWidget(QLabel("Texture"))
        self.sl_tex_amt = QSlider(Qt.Orientation.Horizontal)
        self.sl_tex_amt.setRange(-100, 100); self.sl_tex_amt.setValue(0)
        self.lbl_tex_amt = QLabel("Amount: 0.00")
        self.sl_tex_amt.valueChanged.connect(lambda v: self._on_param_change(self.lbl_tex_amt, f"Amount: {v/100.0:.2f}"))
        
        self.sl_tex_rad = QSlider(Qt.Orientation.Horizontal)
        self.sl_tex_rad.setRange(1, 20); self.sl_tex_rad.setValue(10)
        self.lbl_tex_rad = QLabel("Radius: 1.0")
        self.sl_tex_rad.valueChanged.connect(lambda v: self._on_param_change(self.lbl_tex_rad, f"Radius: {v/10.0:.1f}"))

        left.addWidget(self.lbl_tex_amt); left.addWidget(self.sl_tex_amt)
        left.addWidget(self.lbl_tex_rad); left.addWidget(self.sl_tex_rad)

        left.addSpacing(20)

        # Clarity
        left.addWidget(QLabel("Clarity"))
        self.sl_clar_amt = QSlider(Qt.Orientation.Horizontal)
        self.sl_clar_amt.setRange(-100, 100); self.sl_clar_amt.setValue(0)
        self.lbl_clar_amt = QLabel("Amount: 0.00")
        self.sl_clar_amt.valueChanged.connect(lambda v: self._on_param_change(self.lbl_clar_amt, f"Amount: {v/100.0:.2f}"))

        self.sl_clar_rad = QSlider(Qt.Orientation.Horizontal)
        self.sl_clar_rad.setRange(1, 100); self.sl_clar_rad.setValue(30)
        self.lbl_clar_rad = QLabel("Radius: 3.0")
        self.sl_clar_rad.valueChanged.connect(lambda v: self._on_param_change(self.lbl_clar_rad, f"Radius: {v/10.0:.1f}"))

        left.addWidget(self.lbl_clar_amt); left.addWidget(self.sl_clar_amt)
        left.addWidget(self.lbl_clar_rad); left.addWidget(self.sl_clar_rad)

        left.addSpacing(10)

        # Toggle for Real-time Preview (Requested: below sliders)
        self.chk_realtime = QCheckBox("Real-time Preview")
        self.chk_realtime.setChecked(True)
        self.chk_realtime.toggled.connect(self._trigger_preview)
        left.addWidget(self.chk_realtime)

        left.addStretch(1)

        # Buttons
        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply"); self.btn_apply.clicked.connect(self._apply)
        
        # Reset Button (Requested)
        self.btn_reset = QPushButton("Reset"); self.btn_reset.clicked.connect(self._reset_sliders)
        
        btn_cancel= QPushButton("Cancel"); btn_cancel.clicked.connect(self.close) 
        
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_reset)
        row.addWidget(btn_cancel)
        left.addLayout(row)

        container.addWidget(left_widget, 0) # stretch 0

        # Right Column: Preview
        right = QVBoxLayout()
        
        # Zoom controls
        zoombar = QHBoxLayout()
        b_out = QPushButton("Zoom -"); b_out.clicked.connect(self._zoom_out)
        b_in  = QPushButton("Zoom +"); b_in.clicked.connect(self._zoom_in)
        b_fit = QPushButton("Fit"); b_fit.clicked.connect(self._fit)
        
        zoombar.addWidget(b_out); zoombar.addWidget(b_in); zoombar.addWidget(b_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)

        self.preview_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview_lbl)
        
        right.addWidget(self.scroll, 1) # stretch 1 (expands)

        container.addLayout(right, 1)
        self.resize(1000, 600) # Increased width slightly for wider controls

    def _on_param_change(self, lbl, text):
        lbl.setText(text)
        if self.chk_realtime.isChecked():
            self._preview_timer.start()
    
    def _reset_sliders(self):
        # Block signals to avoid 4 separate preview triggers
        self.sl_tex_amt.blockSignals(True)
        self.sl_tex_rad.blockSignals(True)
        self.sl_clar_amt.blockSignals(True)
        self.sl_clar_rad.blockSignals(True)

        self.sl_tex_amt.setValue(0)
        self.sl_tex_rad.setValue(10)
        self.sl_clar_amt.setValue(0)
        self.sl_clar_rad.setValue(30)

        self.sl_tex_amt.blockSignals(False)
        self.sl_tex_rad.blockSignals(False)
        self.sl_clar_amt.blockSignals(False)
        self.sl_clar_rad.blockSignals(False)

        # Update labels manually
        self.lbl_tex_amt.setText("Amount: 0.00")
        self.lbl_tex_rad.setText("Radius: 1.0")
        self.lbl_clar_amt.setText("Amount: 0.00")
        self.lbl_clar_rad.setText("Radius: 3.0")

        # Trigger one preview update
        if self.chk_realtime.isChecked():
            self._preview_timer.start()

    def _trigger_preview(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            return

        # Preview Toggle: If unchecked, show Original Image (Before)
        if not self.chk_realtime.isChecked():
            # Show original
            qimg = _as_qimage_rgb8(_to_float01(np.asarray(self.doc.image)))
            self._pix = QPixmap.fromImage(qimg)
            self._apply_zoom()
            return

        params = {
            "t_amt": self.sl_tex_amt.value() / 100.0,
            "t_rad": self.sl_tex_rad.value() / 10.0,
            "c_amt": self.sl_clar_amt.value() / 100.0,
            "c_rad": self.sl_clar_rad.value() / 10.0
        }

        # Kill old worker if running
        if hasattr(self, "_worker") and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()

        self._worker = TextureClarityWorker(self.doc.image, params)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.start()

    def _on_preview_ready(self, out_img):
        self._preview = out_img
        qimg = _as_qimage_rgb8(out_img)
        self._pix = QPixmap.fromImage(qimg)
        self._apply_zoom()

    def _apply(self):
        if self.doc is None: return
        t_amt = self.sl_tex_amt.value() / 100.0
        t_rad = self.sl_tex_rad.value() / 10.0
        c_amt = self.sl_clar_amt.value() / 100.0
        c_rad = self.sl_clar_rad.value() / 10.0

        texture_clarity_headless(
            self.doc, 
            texture_amount=t_amt, 
            texture_radius=t_rad,
            clarity_amount=c_amt,
            clarity_radius=c_rad
        )
        self.close()

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        if doc is not self.doc:
            self.doc = doc
            self.setWindowTitle(f"Texture and Clarity - {doc.display_name() if hasattr(doc,'display_name') else 'Image'}")
            # Reset preview
            self._trigger_preview()

    # --- Zoom / Pan ---
    def _apply_zoom(self):
        if self._pix is None: return
        scaled = self._pix.scaled(
            self._pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.resize(scaled.size())
    
    def _zoom_in(self): self._set_zoom(self._zoom * 1.25)
    def _zoom_out(self): self._set_zoom(self._zoom / 1.25)
    def _set_zoom(self, z):
        self._zoom = max(0.05, min(z, 5.0))
        self._apply_zoom()
    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        s = min(vp.width()/self._pix.width(), vp.height()/self._pix.height())
        self._set_zoom(max(0.05, s))

    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._set_zoom(self._zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True; self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                h = self.scroll.horizontalScrollBar(); v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - int(d.x())); v.setValue(v.value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True
        return super().eventFilter(obj, ev)

    def closeEvent(self, ev):
        if self._connected_doc_change and hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
            except Exception:
                pass
        super().closeEvent(ev)

def open_texture_clarity_dialog(main, doc=None, preset: dict | None = None):
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, "Texture & Clarity", "Open an image first.")
        return

    dlg = TextureClarityDialog(main, doc, parent=main)
    # If preset handling needed, add here (set sliders)
    dlg.show()
