#saspro/halobgon.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QTransform, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox, QWidget, QRadioButton
)

# -------- Optional numba utils (LUT in-place speedups) --------------------
try:
    from setiastro.saspro.legacy.numba_utils import apply_lut_mono_inplace as _lut_mono_inplace
    from setiastro.saspro.legacy.numba_utils import apply_lut_color_inplace as _lut_color_inplace
except Exception:
    _lut_mono_inplace = None
    _lut_color_inplace = None

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running

# =============================================================================
# Helpers
# =============================================================================
def _as_rgb(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    else:
        a = a[:, :, :3]
    return a

def _qimage_from_rgb01(a: np.ndarray) -> QImage:
    a = np.clip(a, 0, 1).astype(np.float32)
    a8 = (a * 255.0).astype(np.uint8)
    h, w = a8.shape[:2]
    return QImage(a8.data, w, h, w*3, QImage.Format.Format_RGB888).copy()

def _qimage_from_u8(u8: np.ndarray) -> QImage:
    """8-bit (H,W) or (H,W,3) -> QImage owning its pixels, already in the 32-bit
    format QPixmap.fromImage keeps as is, so the GUI thread does no conversion.
    Mono goes in as Grayscale8 (same pixels as the replicated RGB). Same pixels
    as _qimage_from_rgb01 on the float result. Safe off the GUI thread."""
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape[:2]
    fmt = QImage.Format.Format_Grayscale8 if u8.ndim == 2 else QImage.Format.Format_RGB888
    return QImage(u8.data, w, h, int(u8.strides[0]), fmt).convertToFormat(QImage.Format.Format_RGB32)

# k -> float32(k) / 255.0, exactly what .astype(np.float32) / 255.0 gives
_U8_TO_F01 = np.arange(256, dtype=np.float32) / 255.0

def _u8_to_f01(u8: np.ndarray) -> np.ndarray:
    """u8.astype(np.float32) / 255.0, as one table lookup."""
    return cv2.LUT(np.ascontiguousarray(u8), _U8_TO_F01)

def _maybe_get_mask(parent, ref_img: np.ndarray) -> Optional[np.ndarray]:
    """Fetch an applied mask as float [0..1], broadcastable to ref_img."""
    try:
        mm = None
        if hasattr(parent, "mask_manager"):
            mm = parent.mask_manager
        elif hasattr(parent, "image_manager") and getattr(parent.image_manager, "mask_manager", None):
            mm = parent.image_manager.mask_manager
        if mm is None:
            return None
        m = mm.get_applied_mask()
        if m is None:
            return None
        m = np.asarray(m)
        if m.dtype.kind in "ui":
            m = m.astype(np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0)
        if ref_img.ndim == 3 and m.ndim == 2:
            m = m[..., None]
        if m.shape[:2] != ref_img.shape[:2]:
            return None
        if ref_img.ndim == 3 and m.shape[-1] == 1:
            m = np.repeat(m, ref_img.shape[2], axis=2)
        return m
    except Exception:
        return None

# -------- LUTs (curves) ---------------------------------------------------
def _curve_lut(reduction_level: int) -> np.ndarray:
    """
    SASv2 used stronger darkening as level increases:
      0→γ=1.2, 1→1.5, 2→1.8, 3→2.2
    """
    gammas = [1.2, 1.5, 1.8, 2.2]
    g = gammas[max(0, min(3, int(reduction_level)))]
    x = np.linspace(0, 1, 256, dtype=np.float32)
    y = np.power(x, g)
    lut = np.clip((y * 255.0).round(), 0, 255).astype(np.uint8)
    return lut

def _apply_curve_inplace(img01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply 8-bit LUT to an [0..1] float image. We *always* go through cv2.LUT to
    avoid any view/aliasing surprises. Returns img01 (modified in place).
    """
    u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    mapped = cv2.LUT(u8, lut).astype(np.float32) / 255.0
    np.copyto(img01, mapped)
    return img01

# =============================================================================
# Core algorithm
# =============================================================================
def compute_halo_b_gon(image: np.ndarray, reduction_level: int = 0, is_linear: bool = False) -> np.ndarray:
    """
    Exact port of SASv2 (HaloProcessingThread.applyHaloReduction):
      - operate in [0..1]
      - optional gamma-domain pre-pass for linear data (x ** 1/5)
      - lightness mask is built in 8-bit scale (divide by 255.0), then unsharp
      - enhanced_mask = (1 - unsharp) - blur(unsharp) * (level * 0.33)
      - cv2.multiply(image, enhanced_mask)
      - per-level curves (gamma) via LUT: [1.2, 1.5, 1.8, 2.2]
    Returns the SAME shape as input (2D stays 2D; RGB stays RGB; 1-chan stays 1-chan).
    """
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    work, inv, dup = _halo_prep(img, is_linear)
    out = _u8_to_f01(_halo_u8(work, inv, dup, reduction_level))

    # restore 1-channel shape if input was (H,W,1)
    if img.ndim == 3 and img.shape[2] == 1 and out.ndim == 2:
        out = out[:, :, None]

    return out

def _halo_prep(img: np.ndarray, is_linear: bool):
    """The part of compute_halo_b_gon that does not depend on the level, for an
    image already clipped to [0..1] float32: (work, inv, dup)."""
    # Work buffer (apply gamma-domain if linear)
    work = img
    if is_linear:
        with np.errstate(invalid='ignore'):
            work = np.power(np.clip(work, 0.0, 1.0), 1.0 / 5.0)

    # --- Lightness mask (SASv2 did /255.0 even though image is [0..1]) ---
    if work.ndim == 2 or (work.ndim == 3 and work.shape[2] == 1):
        # treat as grayscale
        light = work if work.ndim == 2 else work[..., 0]
        light = (light.astype(np.float32)) / 255.0
    else:
        # RGB → gray, then scale to 8-bit range
        light = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY) / 255.0

    blurred = cv2.GaussianBlur(light, (0, 0), sigmaX=2)
    unsharp = cv2.addWeighted(light, 1.66, blurred, -0.66, 0.0)

    # --- Enhanced mask (exact SASv2 order) ---
    inv = 1.0 - unsharp
    dup = cv2.GaussianBlur(unsharp, (0, 0), sigmaX=2)
    return work, inv, dup

def _halo_u8(work: np.ndarray, inv: np.ndarray, dup: np.ndarray, reduction_level: int) -> np.ndarray:
    """The level-dependent part: the 8-bit result, same shape as work.
    compute_halo_b_gon returns exactly this / 255.0."""
    scale = float(max(0, min(3, int(reduction_level)))) * 0.33
    enhanced_mask = inv - dup * scale

    # Match mask shape to image shape
    if work.ndim == 3 and work.shape[2] == 3:
        # same values as np.repeat(..., 3, axis=2), which holds the GIL ~0.3 s on 22 MP
        mask = cv2.merge((enhanced_mask, enhanced_mask, enhanced_mask)).astype(work.dtype, copy=False)
    else:
        mask = enhanced_mask.astype(work.dtype, copy=False)

    if work.shape != mask.shape:
        raise ValueError(f"Shape mismatch between image {work.shape} and enhanced_mask {mask.shape}")

    # Multiply
    masked = cv2.multiply(work, mask)

    # Curves via LUT (gamma > 1 darkens), exactly SASv2 levels
    gammas = [1.2, 1.5, 1.8, 2.2]
    g = gammas[int(max(0, min(3, reduction_level)))]
    lut = (np.clip((np.linspace(0, 1, 256, dtype=np.float32) ** g) * 255.0, 0, 255)).astype(np.uint8)

    # (clip(masked) * 255).astype(uint8), in place: masked is our own buffer
    np.clip(masked, 0.0, 1.0, out=masked)
    np.multiply(masked, 255.0, out=masked)
    u8 = masked.astype(np.uint8)
    return cv2.LUT(u8, lut)

# =============================================================================
# Headless apply
# =============================================================================
def apply_halo_b_gon_to_doc(parent, doc, preset: dict | None):
    """
    preset keys:
      - reduction: int 0..3 (0=Extra Low, 1=Low, 2=Medium, 3=High)  [default 0]
      - linear: bool (operate in gamma domain)                       [default False]
    """
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image, dtype=np.float32)
    lvl = int((preset or {}).get("reduction", 0))
    lin = bool((preset or {}).get("linear", False))

    base = _as_rgb(img) if img.ndim != 2 else img
    out = compute_halo_b_gon(img, reduction_level=lvl, is_linear=lin)

    # Blend with active mask if present
    ref = _as_rgb(img)
    m = _maybe_get_mask(parent, ref)
    if m is not None:
        out_rgb = _as_rgb(out)
        blended = np.clip(out_rgb * m + ref * (1.0 - m), 0.0, 1.0)
        # restore mono if needed
        if out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1):
            blended = np.mean(blended, axis=2, dtype=np.float32)
            if out.ndim == 3 and out.shape[2] == 1:
                blended = blended[:, :, None]
        out = blended

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    _meta = {
        "step_name": "Halo-B-Gon",
        "command_id": "halo_b_gon",
        "preset": {"reduction": int(lvl), "linear": bool(lin)},
    }
    doc.set_image(out, metadata=_meta, step_name="Halo-B-Gon")

# =============================================================================
# UI: dialog with preview, zoom/pan, fit, overwrite/new view
# =============================================================================
class _PreviewView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # ✅ Use QPainter.RenderHint.* here
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 1.0

    def wheelEvent(self, e):
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = 1.15 if e.angleDelta().y() > 0 else 1/1.15
            self.set_zoom(self._zoom * step)
            e.accept(); return
        super().wheelEvent(e)

    def set_zoom(self, z):
        z = max(0.1, min(10.0, float(z)))
        self._zoom = z
        self.setTransform(QTransform().scale(z, z))

    def zoom_in(self):  self.set_zoom(self._zoom * 1.15)
    def zoom_out(self): self.set_zoom(self._zoom / 1.15)
    def fit_to(self, rect: QRectF):
        if rect.isEmpty(): return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

class _HaloPreviewWorker(QThread):
    """One preview off the GUI thread (2-3.5 s on 22 MP, all of it on the GUI
    thread before). cv2 and numpy ufuncs only, which release the GIL. A QImage,
    unlike a QPixmap, may be built here."""
    ready = pyqtSignal(object)    # dict, see run()

    def __init__(self, orig, level: int, linear: bool, prep):
        super().__init__()        # no parent: thread_keeper owns it
        self.orig = orig
        self.level = int(level)
        self.linear = bool(linear)
        self.prep = prep          # cached (work, inv, dup) for this orig/linear, or None

    def run(self):
        res = {"orig": self.orig, "level": self.level, "linear": self.linear}
        try:
            prep = self.prep if self.prep is not None else _halo_prep(self.orig, self.linear)
            u8 = _halo_u8(*prep, self.level)
            res.update(prep=prep, u8=u8, qimg=_qimage_from_u8(u8))
        except Exception as e:
            res["error"] = str(e)
        self.orig = self.prep = None
        self.ready.emit(res)


class HaloBGonDialogPro(QDialog):
    """
    Minimal, fast UI:
      • Reduction level (0..3)
      • Linear-data checkbox
      • Live preview (debounced)
      • Apply to: Overwrite active / Create new view
    """
    def __init__(self, parent, doc, icon: Optional[QIcon] = None):
        super().__init__(parent)
        self.setWindowTitle("Halo-B-Gon")
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

        self.parent_ref = parent
        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)

        # Preview worker state. The level-independent part (_halo_prep) is kept
        # so a level change redoes only the cheap tail, and the preview's 8-bit
        # result is kept so Apply reuses it (it is the applied result * 255).
        self._pv_thread = None
        self._pv_pending = False
        self._pv_prep = None      # (orig, linear, (work, inv, dup))
        self._pv_result = None    # (orig, level, linear, u8) of the preview on screen

        # ---- UI ----
        v = QVBoxLayout(self)

        # Controls
        grp = QGroupBox(self.tr("Halo-B-Gon Parameters"))
        grid = QGridLayout(grp)

        grid.addWidget(QLabel(self.tr("Reduction:")), 0, 0)
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(0, 3); self.sl.setValue(0)
        self.lbl = QLabel(self.tr("Extra Low"))
        def _lab(v):
            self.lbl.setText([self.tr("Extra Low"),self.tr("Low"),self.tr("Medium"),self.tr("High")][int(v)])
        self.sl.valueChanged.connect(_lab); _lab(0)
        self.sl.valueChanged.connect(self._debounce)
        grid.addWidget(self.sl, 0, 1); grid.addWidget(self.lbl, 0, 2)

        self.cb_linear = QCheckBox(self.tr("Linear data")); self.cb_linear.setChecked(False)
        self.cb_linear.toggled.connect(self._debounce)
        grid.addWidget(self.cb_linear, 1, 1)

        # Apply target
        grid.addWidget(QLabel(self.tr("Apply to:")), 2, 0)
        self.cmb_target = QComboBox(); self.cmb_target.addItems([self.tr("Overwrite active view"), self.tr("Create new view")])
        grid.addWidget(self.cmb_target, 2, 1, 1, 2)

        v.addWidget(grp)

        # Preview
        self.scene = QGraphicsScene(self)
        self.view  = _PreviewView(self.scene)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        v.addWidget(self.view, 1)

        # Buttons (themed)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Zoom: Ctrl+Wheel")))

        b_minus = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        b_plus  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        b_fit   = themed_toolbtn("zoom-fit-best", self.tr("Fit to View"))

        b_minus.clicked.connect(self.view.zoom_out)
        b_plus .clicked.connect(self.view.zoom_in)
        b_fit  .clicked.connect(lambda: self.view.fit_to(self.scene.itemsBoundingRect()))

        row.addWidget(b_minus)
        row.addWidget(b_plus)
        row.addWidget(b_fit)
        row.addStretch(1)
        v.addLayout(row)


        row2 = QHBoxLayout()
        b_apply = QPushButton(self.tr("Apply")); b_apply.clicked.connect(self._apply)
        b_reset = QPushButton(self.tr("Reset")); b_reset.clicked.connect(self._reset)
        b_cancel= QPushButton(self.tr("Cancel")); b_cancel.clicked.connect(self.reject)
        row2.addStretch(1); row2.addWidget(b_apply); row2.addWidget(b_reset); row2.addWidget(b_cancel)
        v.addLayout(row2)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import halo_path
                _grip_icon = QIcon(halo_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "halo_b_gon", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Halo-B-Gon shortcut with these settings.\n"
                    "Drop directly on an image to apply them headlessly."),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)   # push grip hard to the LEFT edge
            self.btn_doc_help = make_help_button("halo_b_gon", self)
            drag_row.addWidget(self.btn_doc_help)            
            v.addLayout(drag_row)
        except Exception:
            pass

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        # First preview on the worker once the event loop runs (seed_from_preset
        # starts it at once instead).
        self._timer.start(0)
        self.resize(900, 620)

    def _debounce(self): self._timer.start(180)

    def _set_pix(self, rgb):
        q = _qimage_from_rgb01(rgb)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    def _params(self):
        return int(self.sl.value()), bool(self.cb_linear.isChecked())

    # ── preset emit / seed ─────────────────────────────────────────────
    def get_preset(self) -> dict:
        """Emit side — byte-for-byte what _apply records / the consumer reads.
        The Apply-to target is a live-session choice the headless consumer
        can't honor, so it is deliberately NOT part of the preset."""
        lvl, lin = self._params()
        return {"reduction": int(lvl), "linear": bool(lin)}

    def seed_from_preset(self, p: dict | None):
        """Exact inverse of get_preset."""
        p = dict(p or {})

        try:
            lvl = int(p.get("reduction", 0))
        except (TypeError, ValueError):
            lvl = 0
        lvl = max(0, min(3, lvl))
        lin = bool(p.get("linear", False))

        self.sl.blockSignals(True)
        self.cb_linear.blockSignals(True)
        try:
            self.sl.setValue(lvl)
            self.cb_linear.setChecked(lin)
        finally:
            self.sl.blockSignals(False)
            self.cb_linear.blockSignals(False)

        # label is driven by a valueChanged lambda we just blocked — set by hand
        self.lbl.setText([self.tr("Extra Low"), self.tr("Low"), self.tr("Medium"), self.tr("High")][lvl])
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
        lvl, lin = self._params()
        prep = self._pv_prep
        if prep is not None and (prep[0] is not self.orig or prep[1] != lin):
            self._pv_prep = prep = None      # other image / Linear toggled: free it now
        thr = _HaloPreviewWorker(self.orig, lvl, lin, prep[2] if prep is not None else None)
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if getattr(self, "_memory_released", False) or res.get("orig") is not self.orig:
            return                           # closed, or the image changed meanwhile
        if "error" in res:
            QMessageBox.warning(self, "Halo-B-Gon", self.tr("Preview failed:\n{0}").format(res['error']))
            return
        self._pv_prep = (res["orig"], res["linear"], res["prep"])
        self._pv_result = (res["orig"], res["level"], res["linear"], res["u8"])
        self.pix.setPixmap(QPixmap.fromImage(res["qimg"]))
        self.view.setSceneRect(self.pix.boundingRect())

    def _on_preview_finished(self):
        if not self._pv_pending or getattr(self, "_memory_released", False):
            return
        self._pv_pending = False
        r = self._pv_result
        if r is not None and r[0] is self.orig and (r[1], r[2]) == self._params():
            return                           # what is on screen already matches
        self._update_preview()

    def _compute_out(self, lvl: int, lin: bool) -> np.ndarray:
        """compute_halo_b_gon(self.orig, lvl, lin), reusing the preview work."""
        r = self._pv_result
        if r is not None and r[0] is self.orig and r[1] == lvl and r[2] == lin:
            return _u8_to_f01(r[3])
        prep = self._pv_prep
        if prep is not None and prep[0] is self.orig and prep[1] == lin:
            return _u8_to_f01(_halo_u8(*prep[2], lvl))
        return compute_halo_b_gon(self.orig, reduction_level=lvl, is_linear=lin)

    def _apply_overwrite(self, out: np.ndarray):
        if hasattr(self.doc, "set_image"):
            self.doc.set_image(out, step_name="Halo-B-Gon")
        else:
            self.doc.image = out

    def _apply(self):
        lvl, lin = self._params()
        try:
            out = self._compute_out(lvl, lin)

            # Mask blend (same as your preview path)
            m = _maybe_get_mask(self.parent_ref, np.asarray(self.orig, dtype=np.float32))
            if m is not None:
                if self.orig.ndim == 2 and m.ndim == 3:
                    m = m[..., 0]
                if self.orig.ndim == 3 and self.orig.shape[2] == 1 and m.ndim == 2:
                    m = m[:, :, None]
                out = np.clip(out * m + self.orig * (1.0 - m), 0.0, 1.0)

            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

            # ── Register as last_headless_command for replay ──────────
            try:
                main = self.parent()
                if main is not None:
                    preset = {
                        "reduction": int(lvl),
                        "linear": bool(lin),
                    }
                    payload = {
                        "command_id": "halo_b_gon",
                        "preset": dict(preset),
                    }
                    setattr(main, "_last_headless_command", payload)

                    # optional log
                    try:
                        if hasattr(main, "_log"):
                            main._log(
                                f"[Replay] Registered Halo-B-Gon as last action "
                                f"(level={int(lvl)}, linear={bool(lin)})"
                            )
                    except Exception:
                        pass
            except Exception:
                # don't break apply if replay wiring fails
                pass
            # ───────────────────────────────────────────────────────────

            # If user chose "Create new view", go through DocManager so the UI spawns the window.
            create_new = (self.cmb_target.currentIndex() == 1)

            if create_new:
                # Try to find a DocManager on the dialog's parent (preferred) or parent_ref.
                dm = getattr(self.parent(), "doc_manager", None)
                if dm is None:
                    dm = getattr(self.parent_ref, "doc_manager", None)

                if dm is not None:
                    # Carry forward useful metadata; let DocManager’s signal create the view.
                    title = self.doc.display_name() if hasattr(self.doc, "display_name") else "Image"
                    meta = dict(getattr(self.doc, "metadata", {}) or {})
                    # Ensure expected fields exist
                    try:
                        meta.setdefault("bit_depth", "32-bit floating point")
                        if "is_mono" not in meta:
                            meta["is_mono"] = (out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1))
                    except Exception:
                        pass

                    new_doc = dm.create_document(out.copy(), metadata=meta, name=f"{title} [Halo-B-Gon]")
                    try:
                        dm.set_active_document(new_doc)
                    except Exception:
                        pass

                    # Dialog stays open - refresh document for next operation
                    self._refresh_document_from_active()
                    return
                else:
                    # Fallback: try legacy spawner if present; else warn and overwrite.
                    spawner = getattr(self.parent(), "_spawn_new_view_from_numpy", None)
                    if spawner is None:
                        spawner = getattr(self.parent_ref, "_spawn_new_view_from_numpy", None)
                    if callable(spawner):
                        title = self.doc.display_name() if hasattr(self.doc, "display_name") else "Image"
                        spawner(out, f"{title} [Halo-B-Gon]")
                        # Dialog stays open - refresh document for next operation
                        self._refresh_document_from_active()
                        return
                    else:
                        QMessageBox.warning(
                            self, "Halo-B-Gon",
                            self.tr("Could not find DocManager or window spawner; applying to the active view instead.")
                        )
                        # fall through to overwrite

            # Overwrite current (original behavior)
            self._apply_overwrite(out)
            # Dialog stays open - refresh document for next operation
            self._refresh_document_from_active()

        except Exception as e:
            QMessageBox.critical(self, "Halo-B-Gon", self.tr("Failed to apply:\n{0}").format(e))

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
                    self._pv_prep = None
                    self._pv_result = None
                    self._update_preview()
        except Exception:
            pass


    def _reset(self):
        self.sl.setValue(0)
        self.cb_linear.setChecked(False)
        self._update_preview()

    def closeEvent(self, ev):
        self._close_teardown()
        super().closeEvent(ev)

    def done(self, r):
        # Esc / Cancel end a QDialog through done(), which never delivers
        # closeEvent: without this they skipped the release and the dialog
        # kept its full-size buffers until collected. Runs once.
        self._close_teardown()
        super().done(r)

    def _close_teardown(self):
        if getattr(self, "_close_torn_down", False):
            return
        self._close_torn_down = True
        try:
            self._release_memory()
        except Exception:
            pass

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
        self._pv_prep = None
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

def open_halo_b_gon_with_preset(main_window, preset: dict | None = None):
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
        from setiastro.saspro.resources import halo_path
        _icon = QIcon(halo_path)
    except Exception:
        _icon = QIcon()

    dlg = HaloBGonDialogPro(main_window, doc, icon=_icon)   # parent FIRST, doc SECOND
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    try:
        main_window._halo_b_gon_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_halo_b_gon_dialog", None))
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg