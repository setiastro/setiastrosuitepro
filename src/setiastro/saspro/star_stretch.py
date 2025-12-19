# pro/star_stretch.py
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QPointF
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
    QPushButton, QScrollArea, QWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QMovie
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01

# --- use your Numba kernels; fall back to pure numpy SCNR if needed ----
try:
    from setiastro.saspro.legacy.numba_utils import applyPixelMath_numba, applySCNR_numba
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    # Fallback SCNR (Average Neutral) if legacy.numba_utils is unavailable
    def applySCNR_numba(image_array: np.ndarray) -> np.ndarray:
        img = image_array.astype(np.float32, copy=False)
        if img.ndim != 3 or img.shape[2] != 3:
            return img
        r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
        g2 = np.minimum(g, 0.5 * (r + b))
        out = img.copy()
        out[..., 1] = g2
        return np.clip(out, 0.0, 1.0)

# ---- small helpers --------------------------------------------------------

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

    # Prefer zero-copy via sip pointer if available; fall back to bytes
    try:
        from PyQt6 import sip
        qimg = QImage(sip.voidptr(buf8.ctypes.data), w, h, bpl, QImage.Format.Format_RGB888)
        qimg._keepalive = buf8  # keep numpy alive while qimg exists
        return qimg.copy()      # detach so Qt owns the pixels (safe for QPixmap.fromImage)
    except Exception:
        data = buf8.tobytes()
        qimg = QImage(data, w, h, bpl, QImage.Format.Format_RGB888)
        return qimg.copy()      # detach to avoid lifetime issues

def _saturation_boost(rgb01: np.ndarray, amount: float) -> np.ndarray:
    """
    Fast saturation-like boost without HSV dependency:
    C' = mean + (C - mean) * amount
    """
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        return rgb01
    mean = rgb01.mean(axis=2, keepdims=True)
    out = mean + (rgb01 - mean) * float(amount)
    return np.clip(out, 0.0, 1.0)

# ---- background thread ----------------------------------------------------

class _StarStretchWorker(QThread):
    preview_ready = pyqtSignal(object)  # np.ndarray float32 0..1

    def __init__(self, image: np.ndarray, stretch_factor: float, sat_amount: float, do_scnr: bool):
        super().__init__()
        self.image = image
        self.stretch_factor = float(stretch_factor)  # this is the "amount" for your pixel math
        self.sat_amount = float(sat_amount)
        self.do_scnr = bool(do_scnr)

    def run(self):
        imgf = _to_float01(self.image)
        if imgf is None:
            return

        # If grayscale, make it 3-channel to keep the kernels happy, then restore shape
        orig_ndim = imgf.ndim
        need_collapse = False
        if imgf.ndim == 2:
            imgf = np.stack([imgf]*3, axis=-1)
            need_collapse = True
        elif imgf.ndim == 3 and imgf.shape[2] == 1:
            imgf = np.repeat(imgf, 3, axis=2)
            need_collapse = True

        # --- Star Stretch: your Numba pixel math ---
        # amount maps to the SASv2 slider (0..8); kernel uses: f=3**amount
        out = applyPixelMath_numba(imgf.astype(np.float32, copy=False), self.stretch_factor)

        # --- Optional saturation (RGB only) ---
        if out.ndim == 3 and out.shape[2] == 3 and abs(self.sat_amount - 1.0) > 1e-6:
            out = _saturation_boost(out, self.sat_amount)

        # --- Optional SCNR (Average Neutral via your Numba kernel) ---
        if self.do_scnr and out.ndim == 3 and out.shape[2] == 3:
            out = applySCNR_numba(out.astype(np.float32, copy=False))

        # collapse back to mono if we expanded earlier
        if need_collapse:
            out = out[..., 0]

        self.preview_ready.emit(out.astype(np.float32, copy=False))

# ---- dialog ---------------------------------------------------------------

class StarStretchDialog(QDialog):
    """
    Star Stretch for SASpro.
    - Works on active ImageDocument (passed in).
    - Preview is computed in background thread.
    - 'Apply to Document' records history via doc.apply_edit(..., step_name="Star Stretch").
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Star Stretch"))
        self.doc = document
        self._preview: np.ndarray | None = None
        self._pix: QPixmap | None = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._apply_when_ready = False  

        # UI
        main = QHBoxLayout(self)

        # Left column (controls)
        left = QVBoxLayout()
        info = QLabel(
            "Instructions:\n"
            "1) Adjust stretch and options.\n"
            "2) Preview the result.\n"
            "3) Apply to the current document."
        )
        info.setWordWrap(True)
        left.addWidget(info)

        # Stretch slider (0..8.00)
        self.lbl_st = QLabel(self.tr("Stretch Amount:") + " 5.00")
        self.sld_st = QSlider(Qt.Orientation.Horizontal)
        self.sld_st.setRange(0, 800)
        self.sld_st.setValue(500)
        self.sld_st.valueChanged.connect(self._on_stretch_changed)
        left.addWidget(self.lbl_st)
        left.addWidget(self.sld_st)

        # Saturation slider (0..2.00)
        self.lbl_sat = QLabel(self.tr("Color Boost:") + " 1.00")
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 200)
        self.sld_sat.setValue(100)
        self.sld_sat.valueChanged.connect(self._on_sat_changed)
        left.addWidget(self.lbl_sat)
        left.addWidget(self.sld_sat)

        # SCNR checkbox
        self.chk_scnr = QCheckBox(self.tr("Remove Green via SCNR (Optional)"))
        left.addWidget(self.chk_scnr)

        # Buttons row
        rowb = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply = QPushButton(self.tr("Apply to Document"))
        rowb.addWidget(self.btn_preview)
        rowb.addWidget(self.btn_apply)
        left.addLayout(rowb)

        # Spinner
        self.lbl_spin = QLabel()
        self.lbl_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_spin.hide()
        spinner_gif = _guess_spinner_path()
        if spinner_gif and os.path.exists(spinner_gif):
            mv = QMovie(spinner_gif)
            self.lbl_spin.setMovie(mv)
            self._spinner = mv
        else:
            self._spinner = None
        left.addWidget(self.lbl_spin)

        left.addStretch(1)
        main.addLayout(left, 0)

        # Right column (preview with zoom/pan)
        right = QVBoxLayout()
        zoombar = QHBoxLayout()
        b_out = QPushButton(self.tr("Zoom Out"))
        b_in  = QPushButton(self.tr("Zoom In"))
        b_fit = QPushButton(self.tr("Fit to Preview"))
        b_out.clicked.connect(self._zoom_out)
        b_in.clicked.connect(self._zoom_in)
        b_fit.clicked.connect(self._fit)
        zoombar.addWidget(b_out); zoombar.addWidget(b_in); zoombar.addWidget(b_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)

        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)

        right.addWidget(self.scroll, 1)
        main.addLayout(right, 1)

        # signals
        self.btn_preview.clicked.connect(self._run_preview)
        self.btn_apply.clicked.connect(self._apply_to_doc)

        # initialize preview with current doc image
        self._update_preview_pix(self.doc.image)

    # --- UI change handlers ---
    def _on_stretch_changed(self, v: int):
        self.lbl_st.setText(f"Stretch Amount: {v/100.0:.2f}")

    def _on_sat_changed(self, v: int):
        self.lbl_sat.setText(f"Color Boost: {v/100.0:.2f}")

    # --- preview / processing ---
    def _run_preview(self):
        img = self.doc.image
        if img is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        self._show_spinner(True)
        self.btn_preview.setEnabled(False)
        self.btn_apply.setEnabled(False)

        self._thr = _StarStretchWorker(
            image=img,
            stretch_factor=self.sld_st.value()/100.0,
            sat_amount=self.sld_sat.value()/100.0,
            do_scnr=self.chk_scnr.isChecked()
        )
        self._thr.preview_ready.connect(self._on_preview_ready)
        self._thr.finished.connect(lambda: self._show_spinner(False))
        self._thr.start()

    def _on_preview_ready(self, out: np.ndarray):
        out_masked = self._blend_with_mask(out)
        self._preview = out_masked
        self.btn_preview.setEnabled(True)
        self.btn_apply.setEnabled(True)
        self._update_preview_pix(out_masked)

        mw = self._find_main_window()
        if mw and hasattr(mw, "_log"):
            mw._log("Star Stretch: preview generated.")

        # NEW: if Apply was pressed before preview completed, finish now.
        if self._apply_when_ready:
            self._apply_when_ready = False
            self._finish_apply()

    def _apply_to_doc(self):
        # If we don't have a preview yet, compute it and auto-apply when ready.
        if self._preview is None:
            if getattr(self, "_thr", None) and self._thr.isRunning():
                # already computing; just mark to apply when it lands
                self._apply_when_ready = True
                return
            self._apply_when_ready = True
            self._run_preview()
            return

        # We do have a preview → finish immediately
        self._finish_apply()  

    def _finish_apply(self):
        try:
            _marr, mid, mname = self._active_mask_layer()
            meta = {
                "step_name": "Star Stretch",
                "star_stretch": {
                    "stretch_factor": self.sld_st.value()/100.0,
                    "color_boost": self.sld_sat.value()/100.0,
                    "scnr_green": self.chk_scnr.isChecked(),
                    "numba": _HAS_NUMBA,
                },
                # ✅ mask bookkeeping
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }
            self.doc.apply_edit(self._preview.copy(), metadata=meta, step_name="Star Stretch")

            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log("Star Stretch: applied to document.")

            # 🔁 Record as last headless-style command for Replay
            try:
                if mw and hasattr(mw, "_remember_last_headless_command"):
                    preset = {
                        "stretch_factor": self.sld_st.value()/100.0,
                        "color_boost": self.sld_sat.value()/100.0,
                        "scnr_green": self.chk_scnr.isChecked(),
                    }
                    mw._remember_last_headless_command(
                        "star_stretch",
                        preset,
                        description="Star Stretch",
                    )
            except Exception:
                # Don't let replay bookkeeping break the dialog
                pass

        except Exception as e:
            QMessageBox.critical(self, "Apply failed", str(e))
            return
        self.accept()


    # --- preview rendering ---
    def _update_preview_pix(self, img: np.ndarray | None):
        if img is None:
            self.label.clear(); self._pix = None; return
        qimg = _as_qimage_rgb8(_to_float01(img))
        pm = QPixmap.fromImage(qimg)
        self._pix = pm
        self._apply_zoom()

    def _apply_zoom(self):
        if self._pix is None:
            return
        scaled = self._pix.scaled(self._pix.size()*self._zoom,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    # --- zoom/pan ---
    def _zoom_in(self):  self._set_zoom(self._zoom * 1.25)
    def _zoom_out(self): self._set_zoom(self._zoom / 1.25)
    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        if self._pix.width()==0 or self._pix.height()==0: return
        s = min(vp.width()/self._pix.width(), vp.height()/self._pix.height())
        self._set_zoom(max(0.05, s))

    def _set_zoom(self, z: float):
        self._zoom = float(max(0.05, min(z, 8.0)))
        self._apply_zoom()

    # --- spinner ---
    def _show_spinner(self, on: bool):
        if self._spinner is None:
            self.lbl_spin.setVisible(on)
            return
        if on:
            self.lbl_spin.show(); self._spinner.start()
        else:
            self._spinner.stop(); self.lbl_spin.hide()

    # --- event filter (wheel zoom + panning) ---
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

    # --- helper ---
    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    # --- mask helpers ---------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_array_float01, mask_id, mask_name) or (None, None, None)."""
        doc = self.doc
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None, None, None
        layer = getattr(doc, "masks", {}).get(mid)
        if layer is None:
            return None, None, None
        m = np.asarray(getattr(layer, "data", None), dtype=np.float32)
        if m is None or m.size == 0:
            return None, None, None
        # ensure [0..1]
        if m.dtype.kind in "ui":
            m = m / float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0:
                m = m / mx
        m = np.clip(m, 0.0, 1.0)
        return m, mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize using integer indexing (fast, dependency-free)."""
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw):
            return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask(self, stretched: np.ndarray) -> np.ndarray:
        """Blend preview/apply with original using active mask if present."""
        mask, _mid, _name = self._active_mask_layer()
        if mask is None:
            return stretched
        src = _to_float01(self.doc.image)
        out = stretched.astype(np.float32, copy=False)

        # Make sure spatial size matches mask
        th, tw = out.shape[:2]
        m = self._resample_mask_if_needed(mask, (th, tw))

        # Broadcast mask to 3ch when needed
        if out.ndim == 3 and out.shape[2] == 3:
            m = m[..., None]

        # If preview changed mono↔RGB shape, match src first
        if src.ndim == 2 and out.ndim == 3 and out.shape[2] == 3:
            src = np.stack([src]*3, axis=-1)
        elif src.ndim == 3 and src.shape[2] == 3 and out.ndim == 2:
            src = src[..., 0]  # collapse to mono

        return (m * out + (1.0 - m) * src).astype(np.float32, copy=False)


def _guess_spinner_path() -> str | None:
    here = os.path.dirname(__file__)
    cands = [
        os.path.join(here, "spinner.gif"),
        os.path.join(os.path.dirname(here), "spinner.gif"),
        os.path.join(os.getcwd(), "spinner.gif"),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None
