#saspro/morphology.py
from __future__ import annotations
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QFormLayout,
    QLabel, QPushButton, QSlider, QComboBox,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QSpinBox, QDialogButtonBox
)

# Import centralized widgets
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.widgets.image_utils import (
    extract_mask_resized as _get_active_mask_resized,
    blend_with_mask as _blend_with_mask
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running

# ---------------- Core (unchanged) ----------------
# The operation runs on 8-bit data, so it splits into: the 8-bit input (fixed
# for an image), the cv2 op on it, and /255 back to float. The dialog keeps the
# first and shows the second directly; that is the same picture the float
# result gave, since u8 -> /255 -> *255 -> u8 is the identity for all 256 levels.
def _morph_base_u8(img: np.ndarray) -> np.ndarray:
    """8-bit input of apply_morphology for an image already clipped to [0..1]
    float32: (H,W) for mono / (H,W,1), (H,W,3) for RGB."""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        return (img.squeeze() * 255.0).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return (img * 255.0).astype(np.uint8)
    raise ValueError("Input image must be mono (H,W)/(H,W,1) or RGB (H,W,3).")

def _morph_u8(u8: np.ndarray, operation: str, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size % 2 == 0: kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # OpenCV morphology functions handle multi-channel images natively (independent channels)
    if operation == "erosion":  return cv2.erode(u8, kernel, iterations=iterations)
    if operation == "dilation": return cv2.dilate(u8, kernel, iterations=iterations)
    if operation == "opening":  return cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if operation == "closing":  return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    raise ValueError(f"Unsupported morphological operation: {operation}")

# k -> float32(k) / 255.0, exactly what .astype(np.float32) / 255.0 gives
_U8_TO_F01 = np.arange(256, dtype=np.float32) / 255.0

def _morph_out_f01(out_u8: np.ndarray, was_hw1: bool) -> np.ndarray:
    """out_u8.astype(np.float32) / 255.0 (already inside [0, 1]) as one lookup."""
    out = cv2.LUT(np.ascontiguousarray(out_u8), _U8_TO_F01)
    return out[..., None] if was_hw1 else out

def _qimage_from_u8(u8: np.ndarray) -> QImage:
    """8-bit (H,W) or (H,W,3) -> QImage owning its pixels, already in the 32-bit
    format QPixmap.fromImage keeps as is (no conversion on the GUI thread).
    Mono goes in as Grayscale8: the same pixels as the replicated RGB."""
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape[:2]
    fmt = QImage.Format.Format_Grayscale8 if u8.ndim == 2 else QImage.Format.Format_RGB888
    return QImage(u8.data, w, h, int(u8.strides[0]), fmt).convertToFormat(QImage.Format.Format_RGB32)

def apply_morphology(image: np.ndarray, *, operation: str = "erosion",
                     kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    was_hw1 = (img.ndim == 3 and img.shape[2] == 1)
    out_u8 = _morph_u8(_morph_base_u8(img), operation, kernel_size, iterations)
    return _morph_out_f01(out_u8, was_hw1)

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

    _meta = {"step_name": "Morphology", "command_id": "morphology",
             "preset": {"operation": str(op), "kernel": int(ker), "iterations": int(it)}}
    if hasattr(doc, "set_image"): doc.set_image(out, metadata=_meta, step_name="Morphology")
    else: doc.image = out

# Note: _get_active_mask_resized and _blend_with_mask imported from setiastro.saspro.widgets.image_utils


# ---------------- Dialog ----------------
class _MorphPreviewWorker(QThread):
    """One preview off the GUI thread. cv2 and numpy only (they release the
    GIL). A QImage, unlike a QPixmap, may be built here."""
    ready = pyqtSignal(object)    # dict, see run()

    def __init__(self, orig, base_u8, op: str, k: int, it: int, mask):
        super().__init__()        # no parent: thread_keeper owns it
        self.orig = orig
        self.base_u8 = base_u8    # cached _morph_base_u8(orig), or None
        self.key = (op, int(k), int(it))
        self.mask = mask

    def run(self):
        res = {"orig": self.orig, "key": self.key}
        try:
            base = self.base_u8 if self.base_u8 is not None else _morph_base_u8(self.orig)
            out_u8 = _morph_u8(base, *self.key)
            disp = out_u8
            if self.mask is not None:
                # same blend the float preview did, then its 8-bit display
                src = self.orig.squeeze() if out_u8.ndim == 2 else self.orig
                b = _blend_with_mask(src, _morph_out_f01(out_u8, False), self.mask)
                disp = (np.clip(b, 0, 1) * 255).astype(np.uint8)
            res.update(base_u8=base, out_u8=out_u8, qimg=_qimage_from_u8(disp))
        except Exception as e:
            res["error"] = str(e)
        self.orig = self.base_u8 = self.mask = None
        self.ready.emit(res)


class MorphologyDialogPro(QDialog):
    OPS = ["Erosion", "Dilation", "Opening", "Closing"]
    OP_MAP = {"Erosion":"erosion","Dilation":"dilation","Opening":"opening","Closing":"closing"}

    def __init__(self, parent, doc, icon: QIcon | None = None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Morphological Operations"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.doc  = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)

        # Preview worker state. The 8-bit input never changes for an image, and
        # the preview's 8-bit result is what Apply needs (it is the result * 255).
        self._pv_thread = None
        self._pv_pending = False
        self._pv_fit = True       # fit the view to the first preview
        self._base_u8 = None      # (orig, _morph_base_u8(orig))
        self._pv_result = None    # (orig, (op, k, it), out_u8) of the preview on screen

        v = QVBoxLayout(self)

        # ---- Params (unchanged) ----
        grp = QGroupBox(self.tr("Morphological Parameters"))
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

        grid.addWidget(QLabel(self.tr("Operation:")), 0, 0); grid.addWidget(self.cb_op, 0, 1, 1, 2)
        grid.addWidget(QLabel(self.tr("Kernel size:")), 1, 0); grid.addWidget(self.sp_kernel, 1, 1)
        grid.addWidget(QLabel(self.tr("Iterations:")), 2, 0); grid.addWidget(self.sp_iter, 2, 1)
        v.addWidget(grp)

        # ---- Preview with zoom/pan ----
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        v.addWidget(self.view, 1)

        # ---- Zoom bar (themed) ----
        z = QHBoxLayout()
        z.addStretch(1)

        btn_in  = themed_toolbtn("zoom-in", "Zoom In")
        btn_out = themed_toolbtn("zoom-out", "Zoom Out")
        btn_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        btn_in.clicked.connect(self.view.zoom_in)
        btn_out.clicked.connect(self.view.zoom_out)
        btn_fit.clicked.connect(lambda: self.view.fit_to_item(self.pix))

        z.addWidget(btn_in)
        z.addWidget(btn_out)
        z.addWidget(btn_fit)
        v.addLayout(z) 


        # ---- Buttons (unchanged) ----
        row = QHBoxLayout()
        btn_apply = QPushButton(self.tr("Apply"));  btn_apply.clicked.connect(self._apply)
        btn_reset = QPushButton(self.tr("Reset"));  btn_reset.clicked.connect(self._reset)
        btn_cancel= QPushButton(self.tr("Cancel")); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_reset); row.addWidget(btn_cancel)
        v.addLayout(row)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import morpho_path
                _grip_icon = QIcon(morpho_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "morphology", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Morphology shortcut with these settings.\n"
                    "Drop directly on an image to apply them headlessly."),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)   # push grip hard to the LEFT edge
            self.btn_help = make_help_button("morphology", self)
            drag_row.addWidget(self.btn_help)            
            v.addLayout(drag_row)
        except Exception:
            pass

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        # First preview on the worker once the event loop runs; the view is
        # fitted to it when it lands (seed_from_preset starts it at once).
        self._timer.start(0)

    def _debounce(self): self._timer.start(200)

    def _show_u8(self, u8: np.ndarray):
        self.pix.setPixmap(QPixmap.fromImage(_qimage_from_u8(u8)))
        self.scene.setSceneRect(self.pix.boundingRect())

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

    # ── preset emit / seed ─────────────────────────────────────────────
    def get_preset(self) -> dict:
        """Emit side — byte-for-byte what _apply records / the consumer reads."""
        op, k, it = self._params()
        return {"operation": op, "kernel": int(k), "iterations": int(it)}

    def seed_from_preset(self, p: dict | None):
        """Exact inverse of get_preset."""
        p = dict(p or {})

        op = str(p.get("operation", "erosion")).lower()
        op_text = {v: k for k, v in self.OP_MAP.items()}.get(op, "Erosion")

        try:
            k = int(p.get("kernel", 3))
        except (TypeError, ValueError):
            k = 3
        if k % 2 == 0:
            k += 1
        k = max(1, min(31, k))

        try:
            it = int(p.get("iterations", 1))
        except (TypeError, ValueError):
            it = 1
        it = max(1, min(10, it))

        for w in (self.cb_op, self.sp_kernel, self.sp_iter):
            w.blockSignals(True)
        try:
            self.cb_op.setCurrentText(op_text)
            self.sp_kernel.setValue(k)
            self.sp_iter.setValue(it)
        finally:
            for w in (self.cb_op, self.sp_kernel, self.sp_iter):
                w.blockSignals(False)

        self._update_preview()

    def _update_preview(self):
        """Preview on a worker, one at a time. A change made while one runs
        re-runs it with the latest settings when it lands."""
        self._timer.stop()
        if self.orig is None:
            return
        if thread_running(self._pv_thread):
            self._pv_pending = True
            return
        op, k, it = self._params()
        try:
            # Blend preview with active mask (full size, like the image)
            H, W = self.orig.shape[:2]
            m = _get_active_mask_resized(self.doc, H, W)
        except Exception as e:
            QMessageBox.warning(self, "Morphology", f"Preview failed:\n{e}")
            return
        b = self._base_u8
        base_u8 = b[1] if (b is not None and b[0] is self.orig) else None
        thr = _MorphPreviewWorker(self.orig, base_u8, op, k, it, m)
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if getattr(self, "_memory_released", False) or res.get("orig") is not self.orig:
            return                           # closed, or the image changed meanwhile
        if "error" in res:
            QMessageBox.warning(self, "Morphology", f"Preview failed:\n{res['error']}")
            return
        self._base_u8 = (res["orig"], res["base_u8"])
        self._pv_result = (res["orig"], res["key"], res["out_u8"])
        self.pix.setPixmap(QPixmap.fromImage(res["qimg"]))
        self.scene.setSceneRect(self.pix.boundingRect())
        if self._pv_fit:
            self._pv_fit = False
            self.view.fit_to_item(self.pix)

    def _on_preview_finished(self):
        if not self._pv_pending or getattr(self, "_memory_released", False):
            return
        self._pv_pending = False
        r = self._pv_result
        if r is not None and r[0] is self.orig and r[1] == self._params():
            return                           # what is on screen already matches
        self._update_preview()

    def _compute_out(self, op: str, k: int, it: int) -> np.ndarray:
        """apply_morphology(self.orig, ...), reusing the preview work."""
        was_hw1 = (self.orig.ndim == 3 and self.orig.shape[2] == 1)
        r = self._pv_result
        if r is not None and r[0] is self.orig and r[1] == (op, int(k), int(it)):
            return _morph_out_f01(r[2], was_hw1)
        b = self._base_u8
        if b is not None and b[0] is self.orig:
            return _morph_out_f01(_morph_u8(b[1], op, k, it), was_hw1)
        return apply_morphology(self.orig, operation=op, kernel_size=k, iterations=it)

    def _apply(self):
        op, k, it = self._params()
        try:
            out = self._compute_out(op, k, it)
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
            _meta = {"step_name": "Morphology", "command_id": "morphology",
                     "preset": {"operation": op, "kernel": int(k), "iterations": int(it)}}
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(out, metadata=_meta, step_name="Morphology")
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

            # Dialog stays open so user can apply to other images
            # Refresh document reference for next operation
            self._refresh_document_from_active()
        except Exception as e:
            QMessageBox.critical(self, "Morphology", f"Failed to apply:\n{e}")

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    # Refresh preview for new document
                    self.orig = np.clip(np.asarray(new_doc.image, dtype=np.float32), 0.0, 1.0)
                    self._base_u8 = None
                    self._pv_result = None
                    self._update_preview()
        except Exception:
            pass



    def _reset(self):
        self.cb_op.setCurrentText("Erosion")
        self.sp_kernel.setValue(3)
        self.sp_iter.setValue(1)
        # show the unprocessed image (its 8-bit form is what the old float
        # display gave) until the debounced preview lands
        b = self._base_u8
        if b is None or b[0] is not self.orig:
            b = self._base_u8 = (self.orig, _morph_base_u8(self.orig))
        self._show_u8(b[1])
        self.view.fit_to_item(self.pix)

    def closeEvent(self, ev):
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(ev)

    def _release_memory(self):
        """Drop full-frame orig/display buffers and pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True
        try:
            self._timer.stop()
        except Exception:
            pass

        self.orig = None
        self._base_u8 = None
        self._pv_result = None
        self.doc = None
        try:
            self.pix.setPixmap(QPixmap())
        except Exception:
            pass
        try:
            self.scene.clear()
        except Exception:
            pass

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass


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


def open_morphology_with_preset(main_window, preset: dict | None = None):
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            try:
                doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                       else getattr(dm, "active_document", None))
            except Exception:
                doc = None
    if doc is None:
        try:
            ad = getattr(main_window, "_active_doc", None)
            if callable(ad):
                doc = ad()
        except Exception:
            doc = None
    if doc is None or getattr(doc, "image", None) is None:
        return None

    _icon = QIcon()
    try:
        from setiastro.saspro.resources import morphology_path
        _icon = QIcon(morphology_path)
    except Exception:
        _icon = QIcon()

    dlg = MorphologyDialogPro(main_window, doc, icon=_icon)   # parent FIRST, doc SECOND
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    try:
        main_window._morphology_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_morphology_dialog", None))
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg
