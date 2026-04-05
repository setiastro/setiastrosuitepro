# pro/texture_clarity.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QEvent, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSlider, QHBoxLayout,
    QPushButton, QMessageBox, QCheckBox, QScrollArea, QWidget, QGroupBox
)
from PyQt6.QtGui import QPixmap, QImage

try:
    import cv2
except Exception:
    cv2 = None

from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)


# ─── image helpers ────────────────────────────────────────────────────────────

def _as_qimage_rgb8(float01: np.ndarray) -> QImage:
    f = np.asarray(float01, dtype=np.float32)
    if f.ndim == 2:
        f = np.stack([f] * 3, axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    buf8 = np.ascontiguousarray(np.clip(f, 0.0, 1.0) * 255.0, dtype=np.uint8)
    h, w, _ = buf8.shape
    return QImage(buf8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()


# ─── processing core ──────────────────────────────────────────────────────────

def _midtone_mask(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    strength=1.0  → classic midtone-only mask (peaks at 0.5, zero at 0 and 1)
    strength=0.0  → flat mask of ones (apply everywhere equally)
    """
    full = np.clip(1.0 - 4.0 * (image - 0.5) ** 2, 0.0, 1.0)
    return float(strength) * full + (1.0 - float(strength)) * np.ones_like(full)


def _apply_texture(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass sharpening."""
    if abs(amount) < 0.001 or cv2 is None:
        return image
    img = np.ascontiguousarray(np.nan_to_num(image), dtype=np.float32)
    s1, s2 = radius, radius * 2.0
    k1 = int(2 * round(3 * s1) + 1) | 1
    k2 = int(2 * round(3 * s2) + 1) | 1
    try:
        b1 = cv2.GaussianBlur(img, (k1, k1), s1)
        b2 = cv2.GaussianBlur(img, (k2, k2), s2)
    except Exception:
        return image
    return np.clip(img + (b1 - b2) * 2.0 * amount, 0.0, 1.0)


def _apply_clarity(image: np.ndarray, amount: float, radius: float,
                   mask_strength: float = 1.0) -> np.ndarray:
    """Bilateral-based local contrast with midtone mask."""
    if abs(amount) < 0.001 or cv2 is None:
        return image
    img = np.ascontiguousarray(np.nan_to_num(image), dtype=np.float32)
    sigma_space = radius * 10.0
    sigma_color = 0.1
    try:
        if sigma_space > 10.0:
            scale = max(0.1, min(5.0 / sigma_space, 1.0))
            h, w = img.shape[:2]
            small = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            small_filt = cv2.bilateralFilter(small, d=9,
                                             sigmaColor=sigma_color,
                                             sigmaSpace=sigma_space * scale)
            base = cv2.resize(small_filt, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            base = cv2.bilateralFilter(img, d=9,
                                       sigmaColor=sigma_color,
                                       sigmaSpace=sigma_space)
    except Exception:
        try:
            base = cv2.GaussianBlur(img, (0, 0), sigma_space)
        except Exception:
            return image

    mask = _midtone_mask(img, mask_strength)
    return np.clip(img + amount * (img - base) * mask, 0.0, 1.0)


def _compute(image: np.ndarray, t_amt: float, t_rad: float,
             c_amt: float, c_rad: float, mask_strength: float) -> np.ndarray:
    out = _apply_texture(image, t_amt, t_rad)
    out = _apply_clarity(out, c_amt, c_rad, mask_strength)
    return out


def _process(src_float: np.ndarray, params: dict) -> np.ndarray:
    """Apply texture+clarity, handling mono and RGB correctly."""
    t_amt        = float(params.get("t_amt", 0.0))
    t_rad        = float(params.get("t_rad", 1.0))
    c_amt        = float(params.get("c_amt", 0.0))
    c_rad        = float(params.get("c_rad", 1.0))
    mask_str     = float(params.get("mask_str", 1.0))

    if src_float.ndim == 3 and src_float.shape[2] >= 3:
        R, G, B = src_float[..., 0], src_float[..., 1], src_float[..., 2]
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        L_new = _compute(L, t_amt, t_rad, c_amt, c_rad, mask_str)
        ratio = L_new / (L + 1e-7)
        out = np.clip(src_float[..., :3] * ratio[..., None], 0.0, 1.0)
    else:
        s = src_float.squeeze() if src_float.ndim == 3 else src_float
        out = _compute(s, t_amt, t_rad, c_amt, c_rad, mask_str)
        if src_float.ndim == 3:
            out = out[..., None]

    return out.astype(np.float32, copy=False)


def _blend_mask(out: np.ndarray, src: np.ndarray,
                mask01: np.ndarray | None) -> np.ndarray:
    if mask01 is None:
        return out
    h, w = out.shape[:2]
    m = mask01
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = np.linspace(0, m.shape[0] - 1, h).astype(np.int32)
            xi = np.linspace(0, m.shape[1] - 1, w).astype(np.int32)
            m = m[yi][:, xi]
    if out.ndim == 3 and m.ndim == 2:
        m = m[:, :, None]
    src_f = src
    if out.ndim != src_f.ndim:
        if out.ndim == 2 and src_f.ndim == 3:
            src_f = src_f.squeeze()
        elif out.ndim == 3 and src_f.ndim == 2:
            src_f = np.repeat(src_f[:, :, None], out.shape[2], axis=2)
    return np.clip(src_f * (1.0 - m) + out * m, 0.0, 1.0).astype(np.float32)


# ─── headless entry point ─────────────────────────────────────────────────────

def texture_clarity_headless(doc, texture_amount=0.0, texture_radius=1.0,
                             clarity_amount=0.0, clarity_radius=1.0,
                             mask_strength=1.0):
    if doc is None or getattr(doc, "image", None) is None:
        return
    src = _to_float01(np.asarray(doc.image))
    if src is None:
        return
    params = dict(t_amt=texture_amount, t_rad=texture_radius,
                  c_amt=clarity_amount, c_rad=clarity_radius,
                  mask_str=mask_strength)
    out = _process(src, params)
    out = _blend_mask(out, src, _active_mask_array_from_doc(doc))
    doc.apply_edit(out, metadata={
        "step_name": "Texture and Clarity",
        "texture_clarity": params,
    }, step_name="Texture and Clarity")


# ─── worker ───────────────────────────────────────────────────────────────────

class TextureClarityWorker(QThread):
    preview_ready = pyqtSignal(object)   # np.ndarray

    def __init__(self, image: np.ndarray, params: dict,
                 mask01: np.ndarray | None = None):
        super().__init__()
        self.image   = image
        self.params  = params
        self.mask01  = mask01
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        src = _to_float01(self.image)
        if src is None or self._cancel:
            return
        out = _process(src, self.params)
        if self._cancel:
            return
        out = _blend_mask(out, src, self.mask01)
        if not self._cancel:
            self.preview_ready.emit(out)


# ─── dialog ───────────────────────────────────────────────────────────────────

class TextureClarityDialog(QDialog):
    def __init__(self, main, doc, parent=None):
        super().__init__(parent)
        self.main = main
        self.doc  = doc
        self.setWindowTitle("Texture and Clarity")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        # preview state
        self._pix_processed: QPixmap | None = None   # latest processed result
        self._pix_original:  QPixmap | None = None   # cached original
        self._showing_original = False
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._did_initial_fit = False
        self._worker: TextureClarityWorker | None = None

        # active-document tracking
        self._connected_doc_change = False
        if hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_doc_change = True
            except Exception:
                pass

        # debounce timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._trigger_preview)

        self._build_ui()
        self._cache_original()
        self._trigger_preview()

    # ── cache original ────────────────────────────────────────────────────────

    def _cache_original(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            self._pix_original = None
            return
        src = _to_float01(np.asarray(self.doc.image))
        if src is not None:
            self._pix_original = QPixmap.fromImage(_as_qimage_rgb8(src))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── left: controls ────────────────────────────────────────────────────
        left_w = QWidget()
        left_w.setMinimumWidth(300)
        left_w.setMaximumWidth(340)
        left = QVBoxLayout(left_w)
        left.setSpacing(6)

        def _slider_row(label_text, lo, hi, init, scale, fmt):
            grp = QGroupBox(label_text)
            gl  = QVBoxLayout(grp)
            gl.setSpacing(2)
            lbl = QLabel(fmt(init / scale))
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(lo, hi)
            sld.setValue(init)
            gl.addWidget(lbl)
            gl.addWidget(sld)
            return grp, sld, lbl, scale, fmt

        # Texture
        tex_grp = QGroupBox("Texture")
        tg = QVBoxLayout(tex_grp)
        self.lbl_t_amt = QLabel("Amount: 0.00")
        self.sl_t_amt  = QSlider(Qt.Orientation.Horizontal)
        self.sl_t_amt.setRange(-100, 100); self.sl_t_amt.setValue(0)
        self.lbl_t_rad = QLabel("Radius: 1.0")
        self.sl_t_rad  = QSlider(Qt.Orientation.Horizontal)
        # extended range: 0.1 → 10.0 (slider 1–100)
        self.sl_t_rad.setRange(1, 100); self.sl_t_rad.setValue(10)
        self.sl_t_rad.setToolTip("0.1 – 10.0 px  (fine detail → coarse texture)")
        for w in (self.lbl_t_amt, self.sl_t_amt,
                  self.lbl_t_rad, self.sl_t_rad):
            tg.addWidget(w)
        left.addWidget(tex_grp)

        # Clarity
        clar_grp = QGroupBox("Clarity")
        cg = QVBoxLayout(clar_grp)
        self.lbl_c_amt = QLabel("Amount: 0.00")
        self.sl_c_amt  = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_amt.setRange(-100, 100); self.sl_c_amt.setValue(0)
        self.lbl_c_rad = QLabel("Radius: 3.0")
        self.sl_c_rad  = QSlider(Qt.Orientation.Horizontal)
        self.sl_c_rad.setRange(1, 100); self.sl_c_rad.setValue(30)
        for w in (self.lbl_c_amt, self.sl_c_amt,
                  self.lbl_c_rad, self.sl_c_rad):
            cg.addWidget(w)
        left.addWidget(clar_grp)

        # Midtone mask strength
        mask_grp = QGroupBox("Clarity Mask")
        mg = QVBoxLayout(mask_grp)
        self.lbl_mask = QLabel("Midtone strength: 1.00  (1=midtones only, 0=everywhere)")
        self.lbl_mask.setWordWrap(True)
        self.sl_mask  = QSlider(Qt.Orientation.Horizontal)
        self.sl_mask.setRange(0, 100); self.sl_mask.setValue(100)
        self.sl_mask.setToolTip(
            "1.0 = affect midtones only (standard clarity behaviour)\n"
            "0.0 = apply clarity uniformly across shadows and highlights too"
        )
        mg.addWidget(self.lbl_mask)
        mg.addWidget(self.sl_mask)
        left.addWidget(mask_grp)

        left.addSpacing(4)

        # ── before/after toggle ───────────────────────────────────────────────
        self.btn_toggle = QPushButton("⇄  Toggle Before / After")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setToolTip(
            "Instantly switch between original and processed preview.\n"
            "No recalculation — uses cached images."
        )
        self.btn_toggle.toggled.connect(self._on_toggle)
        left.addWidget(self.btn_toggle)

        left.addStretch(1)

        # action buttons
        btn_row = QHBoxLayout()
        self.btn_apply  = QPushButton("Apply")
        self.btn_reset  = QPushButton("Reset")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_cancel.clicked.connect(self.close)
        for b in (self.btn_apply, self.btn_reset, self.btn_cancel):
            btn_row.addWidget(b)
        left.addLayout(btn_row)

        root.addWidget(left_w, 0)

        # ── right: preview ────────────────────────────────────────────────────
        right = QVBoxLayout()

        zoom_row = QHBoxLayout()
        for label, slot in (("–", self._zoom_out), ("+", self._zoom_in),
                             ("Fit", self._fit)):
            b = QPushButton(label)
            b.setFixedWidth(44)
            b.clicked.connect(slot)
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.preview_lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.preview_lbl)
        right.addWidget(self.scroll, 1)

        root.addLayout(right, 1)
        self.resize(1050, 620)

        # wire signals
        self.sl_t_amt.valueChanged.connect(
            lambda v: self._param_changed(self.lbl_t_amt, f"Amount: {v/100:.2f}"))
        self.sl_t_rad.valueChanged.connect(
            lambda v: self._param_changed(self.lbl_t_rad, f"Radius: {v/10:.1f}"))
        self.sl_c_amt.valueChanged.connect(
            lambda v: self._param_changed(self.lbl_c_amt, f"Amount: {v/100:.2f}"))
        self.sl_c_rad.valueChanged.connect(
            lambda v: self._param_changed(self.lbl_c_rad, f"Radius: {v/10:.1f}"))
        self.sl_mask.valueChanged.connect(
            lambda v: self._param_changed(self.lbl_mask,
                f"Midtone strength: {v/100:.2f}  (1=midtones only, 0=everywhere)"))

    # ── param change ──────────────────────────────────────────────────────────

    def _param_changed(self, lbl: QLabel, text: str):
        lbl.setText(text)
        # if showing original via toggle, switch back to processed view
        if self.btn_toggle.isChecked():
            self.btn_toggle.blockSignals(True)
            self.btn_toggle.setChecked(False)
            self.btn_toggle.blockSignals(False)
            self._showing_original = False
        self._preview_timer.start()

    # ── before/after toggle ───────────────────────────────────────────────────

    def _on_toggle(self, showing_before: bool):
        self._showing_original = showing_before
        if showing_before:
            # show cached original instantly — no recalculation
            if self._pix_original is not None:
                self._show_pixmap(self._pix_original)
        else:
            # show cached processed result instantly
            if self._pix_processed is not None:
                self._show_pixmap(self._pix_processed)
            else:
                self._trigger_preview()

    # ── preview ───────────────────────────────────────────────────────────────

    def _trigger_preview(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(300)

        params = {
            "t_amt":    self.sl_t_amt.value() / 100.0,
            "t_rad":    self.sl_t_rad.value() / 10.0,
            "c_amt":    self.sl_c_amt.value() / 100.0,
            "c_rad":    self.sl_c_rad.value() / 10.0,
            "mask_str": self.sl_mask.value()  / 100.0,
        }
        mask01 = _active_mask_array_from_doc(self.doc)
        self._worker = TextureClarityWorker(self.doc.image, params, mask01)
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.start()

    def _on_preview_ready(self, out: np.ndarray):
        self._pix_processed = QPixmap.fromImage(_as_qimage_rgb8(out))
        if not self._showing_original:
            self._show_pixmap(self._pix_processed)
        if not self._did_initial_fit:
            self._did_initial_fit = True
            QTimer.singleShot(0, self._fit)

    def _show_pixmap(self, pix: QPixmap):
        """Display pix at current zoom without adjusting scrollbars."""
        scaled = pix.scaled(
            pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.resize(scaled.size())

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset(self):
        for sld in (self.sl_t_amt, self.sl_t_rad,
                    self.sl_c_amt, self.sl_c_rad, self.sl_mask):
            sld.blockSignals(True)
        self.sl_t_amt.setValue(0)
        self.sl_t_rad.setValue(10)
        self.sl_c_amt.setValue(0)
        self.sl_c_rad.setValue(30)
        self.sl_mask.setValue(100)
        for sld in (self.sl_t_amt, self.sl_t_rad,
                    self.sl_c_amt, self.sl_c_rad, self.sl_mask):
            sld.blockSignals(False)
        self.lbl_t_amt.setText("Amount: 0.00")
        self.lbl_t_rad.setText("Radius: 1.0")
        self.lbl_c_amt.setText("Amount: 0.00")
        self.lbl_c_rad.setText("Radius: 3.0")
        self.lbl_mask.setText("Midtone strength: 1.00  (1=midtones only, 0=everywhere)")
        self._preview_timer.start()

    # ── apply ─────────────────────────────────────────────────────────────────

    def _apply(self):
        if self.doc is None:
            return
        texture_clarity_headless(
            self.doc,
            texture_amount = self.sl_t_amt.value() / 100.0,
            texture_radius = self.sl_t_rad.value() / 10.0,
            clarity_amount = self.sl_c_amt.value() / 100.0,
            clarity_radius = self.sl_c_rad.value() / 10.0,
            mask_strength  = self.sl_mask.value()  / 100.0,
        )
        self._save_geometry()
        self.close()

    # ── active doc change ─────────────────────────────────────────────────────

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        if doc is self.doc:
            return
        self.doc = doc
        title = doc.display_name() if hasattr(doc, "display_name") else "Image"
        self.setWindowTitle(f"Texture and Clarity — {title}")
        self._pix_processed = None
        self._showing_original = False
        self.btn_toggle.blockSignals(True)
        self.btn_toggle.setChecked(False)
        self.btn_toggle.blockSignals(False)
        self._cache_original()
        self._trigger_preview()

    # ── zoom / pan ────────────────────────────────────────────────────────────
    def _zoom_in(self):
        vp = self.scroll.viewport()
        self._set_zoom(self._zoom * 1.25,
                       anchor=QPointF(vp.width() / 2.0, vp.height() / 2.0))

    def _zoom_out(self):
        vp = self.scroll.viewport()
        self._set_zoom(self._zoom / 1.25,
                       anchor=QPointF(vp.width() / 2.0, vp.height() / 2.0))

    def _set_zoom(self, z: float, anchor: QPointF | None = None):
        """
        Zoom to z, keeping the content point under `anchor` (viewport coords)
        stationary. If anchor is None, uses viewport centre.
        """
        old_zoom = self._zoom
        self._zoom = max(0.05, min(z, 5.0))

        pix = (self._pix_original if self._showing_original
               else self._pix_processed)
        if pix is None:
            return

        scaled = pix.scaled(
            pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.resize(scaled.size())

        # ── anchor-preserving scroll adjustment ──────────────────────────────
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        if anchor is None:
            vp = self.scroll.viewport()
            anchor = QPointF(vp.width() / 2.0, vp.height() / 2.0)

        # content coords of anchor point before zoom
        cx = hbar.value() + anchor.x()
        cy = vbar.value() + anchor.y()

        # scale factor between old and new zoom
        factor = self._zoom / max(old_zoom, 1e-9)

        # restore so the same content point stays under the anchor
        hbar.setValue(int(cx * factor - anchor.x()))
        vbar.setValue(int(cy * factor - anchor.y()))

    def _fit(self):
        pix = (self._pix_original if self._showing_original
               else self._pix_processed)
        if pix is None:
            return
        vp = self.scroll.viewport().size()
        s  = min(vp.width() / pix.width(), vp.height() / pix.height())
        # fit uses centre anchor — _set_zoom defaults to centre when anchor=None
        self._set_zoom(max(0.05, s))

    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            t = ev.type()
            if (t == QEvent.Type.Wheel
                    and ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
                self._set_zoom(self._zoom * factor,
                               anchor=ev.position())   # mouse pos in viewport coords
                ev.accept(); return True
            if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if t == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                self.scroll.horizontalScrollBar().setValue(
                    self.scroll.horizontalScrollBar().value() - int(d.x()))
                self.scroll.verticalScrollBar().setValue(
                    self.scroll.verticalScrollBar().value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True
            if (t == QEvent.Type.MouseButtonRelease
                    and ev.button() == Qt.MouseButton.LeftButton
                    and self._panning):
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True
        return super().eventFilter(obj, ev)

    # ── geometry persistence ──────────────────────────────────────────────────

    def _save_geometry(self):
        try:
            QSettings().setValue("texture_clarity/window_geometry",
                                 self.saveGeometry())
        except Exception:
            pass

    def _restore_geometry(self):
        try:
            g = QSettings().value("texture_clarity/window_geometry")
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True
            QTimer.singleShot(0, lambda: (self._restore_geometry(),
                                          self._fit() if self._pix_processed else None))

    def closeEvent(self, ev):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(500)
        if self._connected_doc_change and hasattr(self.main, "currentDocumentChanged"):
            try:
                self.main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
            except Exception:
                pass
        self._save_geometry()
        super().closeEvent(ev)


# ─── public entry point ───────────────────────────────────────────────────────

def open_texture_clarity_dialog(main, doc=None, preset: dict | None = None):
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, "Texture & Clarity", "Open an image first.")
        return
    dlg = TextureClarityDialog(main, doc, parent=main)
    dlg.show()