# pro/morphology.py
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSlider, QComboBox,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QSpinBox, QDialogButtonBox
)

# Import centralized widgets
from pro.widgets.graphics_views import ZoomableGraphicsView
from pro.widgets.image_utils import (
    extract_mask_resized as _get_active_mask_resized,
    blend_with_mask as _blend_with_mask
)

# ---------------- Core (unchanged) ----------------
def apply_morphology(image: np.ndarray, *, operation: str = "erosion",
                     kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    # ... (existing body unchanged)
    if image is None:
        raise ValueError("image is None")
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    was_hw1 = (img.ndim == 3 and img.shape[2] == 1)
    if kernel_size % 2 == 0: kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _do(u8):
        if operation == "erosion":  return cv2.erode(u8, kernel, iterations=iterations)
        if operation == "dilation": return cv2.dilate(u8, kernel, iterations=iterations)
        if operation == "opening":  return cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if operation == "closing":  return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        raise ValueError(f"Unsupported morphological operation: {operation}")

    if img.ndim == 2 or was_hw1:
        mono = (img.squeeze() * 255.0).astype(np.uint8)
        out = _do(mono).astype(np.float32) / 255.0
        out = np.clip(out, 0.0, 1.0)
        return out[..., None] if was_hw1 else out

    if img.ndim == 3 and img.shape[2] == 3:
        u8 = (img * 255.0).astype(np.uint8)
        ch = cv2.split(u8)
        ch = [_do(c) for c in ch]
        out = cv2.merge(ch).astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    raise ValueError("Input image must be mono (H,W)/(H,W,1) or RGB (H,W,3).")

def apply_morphology_to_doc(doc, preset: dict | None):
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image)
    op   = (preset or {}).get("operation", "erosion")
    ker  = int((preset or {}).get("kernel", 3))
    it   = int((preset or {}).get("iterations", 1))

    out = apply_morphology(img, operation=str(op), kernel_size=ker, iterations=it)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # Blend with active mask if present
    H, W = out.shape[:2]
    m = _get_active_mask_resized(doc, H, W)
    if m is not None:
        base = np.asarray(doc.image, dtype=np.float32)
        if base.dtype.kind in "ui":
            maxv = float(np.iinfo(base.dtype).max)
            base = base / max(1.0, maxv)
        else:
            base = np.clip(base, 0.0, 1.0)
        out = _blend_with_mask(base, out, m).astype(np.float32, copy=False)

    if hasattr(doc, "set_image"): doc.set_image(out, step_name="Morphology")
    elif hasattr(doc, "apply_numpy"): doc.apply_numpy(out, step_name="Morphology")
    else: doc.image = out

# Note: _get_active_mask_resized and _blend_with_mask imported from pro.widgets.image_utils


# ---------------- Dialog ----------------
class MorphologyDialogPro(QDialog):
    OPS = ["Erosion", "Dilation", "Opening", "Closing"]
    OP_MAP = {"Erosion":"erosion","Dilation":"dilation","Opening":"opening","Closing":"closing"}

    def __init__(self, parent, doc, icon: QIcon | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Morphological Operations")
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.doc  = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)

        disp = self.orig
        if disp.ndim == 2: disp = disp[..., None].repeat(3, axis=2)
        elif disp.ndim == 3 and disp.shape[2] == 1: disp = disp.repeat(3, axis=2)
        self._disp_base = disp

        v = QVBoxLayout(self)

        # ---- Params (unchanged) ----
        grp = QGroupBox("Morphological Parameters")
        grid = QGridLayout(grp)
        self.cb_op = QComboBox(); self.cb_op.addItems(self.OPS)
        self.sp_kernel = QSpinBox(); self.sp_kernel.setRange(1, 31); self.sp_kernel.setSingleStep(2)
        self.sp_iter   = QSpinBox(); self.sp_iter.setRange(1, 10)

        init = dict(initial or {})
        op_text = {v:k for k,v in self.OP_MAP.items()}.get(str(init.get("operation","erosion")).lower(), "Erosion")
        self.cb_op.setCurrentText(op_text)
        k = int(init.get("kernel", 3)); self.sp_kernel.setValue(k if k % 2 == 1 else k + 1)
        self.sp_iter.setValue(int(init.get("iterations", 1)))

        self.cb_op.currentTextChanged.connect(self._debounce)
        self.sp_kernel.valueChanged.connect(self._debounce)
        self.sp_iter.valueChanged.connect(self._debounce)

        grid.addWidget(QLabel("Operation:"), 0, 0); grid.addWidget(self.cb_op, 0, 1, 1, 2)
        grid.addWidget(QLabel("Kernel size:"), 1, 0); grid.addWidget(self.sp_kernel, 1, 1)
        grid.addWidget(QLabel("Iterations:"), 2, 0); grid.addWidget(self.sp_iter, 2, 1)
        v.addWidget(grp)

        # ---- Preview with zoom/pan ----
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        v.addWidget(self.view, 1)

        # ---- Zoom bar ----
        z = QHBoxLayout()
        btn_in  = QPushButton("Zoom In");  btn_in.clicked.connect(self.view.zoom_in)
        btn_out = QPushButton("Zoom Out"); btn_out.clicked.connect(self.view.zoom_out)
        btn_fit = QPushButton("Fit to Preview"); btn_fit.clicked.connect(lambda: self.view.fit_to_item(self.pix))
        z.addStretch(1); z.addWidget(btn_in); z.addWidget(btn_out); z.addWidget(btn_fit)
        v.addLayout(z)

        # ---- Buttons (unchanged) ----
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply");  btn_apply.clicked.connect(self._apply)
        btn_reset = QPushButton("Reset");  btn_reset.clicked.connect(self._reset)
        btn_cancel= QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_reset); row.addWidget(btn_cancel)
        v.addLayout(row)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        self._set_pix(self._disp_base)
        self._update_preview()
        # initial fit
        self.view.fit_to_item(self.pix)

    def _debounce(self): self._timer.start(200)

    def _set_pix(self, rgb):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.scene.setSceneRect(self.pix.boundingRect())

    def _params(self):
        op = self.OP_MAP[self.cb_op.currentText()]
        k  = int(self.sp_kernel.value())
        it = int(self.sp_iter.value())
        if k % 2 == 0: k += 1
        return op, k, it

    def _update_preview(self):
        op, k, it = self._params()
        try:
            out = apply_morphology(self._disp_base, operation=op, kernel_size=k, iterations=it)

            # Blend preview with active mask (preview is on _disp_base size)
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = self._disp_base.astype(np.float32)
                out = _blend_with_mask(base, out.astype(np.float32), m)

            self._set_pix(out)
        except Exception as e:
            QMessageBox.warning(self, "Morphology", f"Preview failed:\n{e}")

    def _apply(self):
        op, k, it = self._params()
        try:
            out = apply_morphology(self.orig, operation=op, kernel_size=k, iterations=it)
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # Blend with active mask at full resolution
            H, W = out.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
            if m is not None:
                base = np.asarray(self.doc.image, dtype=np.float32)
                if base.dtype.kind in "ui":
                    maxv = float(np.iinfo(base.dtype).max)
                    base = base / max(1.0, maxv)
                else:
                    base = np.clip(base, 0.0, 1.0)
                out = _blend_with_mask(base, out, m).astype(np.float32, copy=False)

            # Commit to document
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(out, step_name="Morphology")
            elif hasattr(self.doc, "apply_numpy"):
                self.doc.apply_numpy(out, step_name="Morphology")
            else:
                self.doc.image = out

            # ── Register as last_headless_command for replay ───────────
            try:
                main = self.parent()
                if main is not None:
                    preset = {
                        "operation": op,
                        "kernel": int(k),
                        "iterations": int(it),
                    }
                    payload = {
                        "command_id": "morphology",
                        "preset": dict(preset),
                    }
                    setattr(main, "_last_headless_command", payload)

                    # optional log
                    try:
                        if hasattr(main, "_log"):
                            main._log(
                                f"[Replay] Registered Morphology as last action "
                                f"(op={op}, kernel={k}, iter={it})"
                            )
                    except Exception:
                        pass
            except Exception:
                # never break apply if replay wiring fails
                pass
            # ────────────────────────────────────────────────────────────

            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Morphology", f"Failed to apply:\n{e}")



    def _reset(self):
        self.cb_op.setCurrentText("Erosion")
        self.sp_kernel.setValue(3)
        self.sp_iter.setValue(1)
        self._set_pix(self._disp_base)
        self.view.fit_to_item(self.pix)


# ---------------------- Preset editor (Shortcuts) ----------------------

class _MorphologyPresetDialog(QDialog):
    """
    Preset editor for Morphology shortcuts.
    Stores JSON-safe dict:
        { "operation": "erosion|dilation|opening|closing",
          "kernel": int odd,
          "iterations": int }
    """
    OPS = ["erosion", "dilation", "opening", "closing"]

    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Morphology — Preset")
        p = dict(initial or {})
        f = QFormLayout(self)

        self.cb_op = QComboBox()
        self.cb_op.addItems([op.title() for op in self.OPS])
        op0 = str(p.get("operation", "erosion")).lower()
        if op0 not in self.OPS:
            op0 = "erosion"
        self.cb_op.setCurrentText(op0.title())

        self.sp_kernel = QSpinBox()
        self.sp_kernel.setRange(1, 31)
        self.sp_kernel.setSingleStep(2)
        k = int(p.get("kernel", 3))
        if k % 2 == 0:
            k += 1
        self.sp_kernel.setValue(k)

        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(1, 10)
        self.sp_iter.setValue(int(p.get("iterations", 1)))

        f.addRow("Operation:", self.cb_op)
        f.addRow("Kernel size:", self.sp_kernel)
        f.addRow("Iterations:", self.sp_iter)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        f.addRow(btns)

    def result_dict(self) -> dict:
        k = int(self.sp_kernel.value())
        if k % 2 == 0:
            k += 1
        return {
            "operation": self.cb_op.currentText().lower(),
            "kernel": int(k),
            "iterations": int(self.sp_iter.value()),
        }


# ---------------------- Headless runner (Scripts / Presets / Replay) ----------------------

def run_morphology_via_preset(main, preset: dict | None = None, *, target_doc=None):
    """
    Headless Morphology runner.

    preset keys:
      - operation: "erosion" | "dilation" | "opening" | "closing"
      - kernel: odd int (default 3)
      - iterations: int >= 1 (default 1)
    """
    p = dict(preset or {})

    # ---- Remember for Replay ----
    try:
        remember = getattr(main, "_remember_last_headless_command", None) \
                   or getattr(main, "remember_last_headless_command", None)
        if callable(remember):
            remember("morphology", p, description="Morphology")
        else:
            setattr(main, "_last_headless_command", {
                "command_id": "morphology",
                "preset": dict(p),
            })
    except Exception:
        pass
    # ----------------------------

    dm = getattr(main, "doc_manager", None) or getattr(main, "dm", None)

    # Resolve doc
    doc = target_doc
    if doc is None:
        d = getattr(main, "_active_doc", None)
        doc = d() if callable(d) else d

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "Morphology", "Load an image first.")
        return

    try:
        apply_morphology_to_doc(doc, p)
        if hasattr(main, "_log"):
            main._log(f"✅ Morphology (headless) preset={p}")
    except Exception as e:
        QMessageBox.critical(main, "Morphology", str(e))
        if hasattr(main, "_log"):
            main._log(f"❌ Morphology failed: {e}")
