# pro/halobgon.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QImage, QPixmap, QIcon, QTransform, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QComboBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox, QWidget, QRadioButton
)

# -------- Optional numba utils (LUT in-place speedups) --------------------
try:
    from legacy.numba_utils import apply_lut_mono_inplace as _lut_mono_inplace
    from legacy.numba_utils import apply_lut_color_inplace as _lut_color_inplace
except Exception:
    _lut_mono_inplace = None
    _lut_color_inplace = None

from pro.widgets.themed_buttons import themed_toolbtn


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
    scale = float(max(0, min(3, int(reduction_level)))) * 0.33
    enhanced_mask = inv - dup * scale

    # Match mask shape to image shape
    if work.ndim == 3 and work.shape[2] == 3:
        mask = np.repeat(enhanced_mask[:, :, None], 3, axis=2).astype(work.dtype, copy=False)
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

    u8 = (np.clip(masked, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    mapped = cv2.LUT(u8, lut).astype(np.float32) / 255.0

    out = np.clip(mapped, 0.0, 1.0).astype(np.float32, copy=False)

    # restore 1-channel shape if input was (H,W,1)
    if img.ndim == 3 and img.shape[2] == 1 and out.ndim == 2:
        out = out[:, :, None]

    return out

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

    if hasattr(doc, "set_image"):
        doc.set_image(out, step_name="Halo-B-Gon")
    elif hasattr(doc, "apply_numpy"):
        doc.apply_numpy(out, step_name="Halo-B-Gon")
    else:
        doc.image = out

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
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.parent_ref = parent
        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)

        # Display base is RGB
        disp = _as_rgb(self.orig)
        self._disp_base = disp

        # ---- UI ----
        v = QVBoxLayout(self)

        # Controls
        grp = QGroupBox("Halo-B-Gon Parameters")
        grid = QGridLayout(grp)

        grid.addWidget(QLabel("Reduction:"), 0, 0)
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(0, 3); self.sl.setValue(0)
        self.lbl = QLabel("Extra Low")
        def _lab(v):
            self.lbl.setText(["Extra Low","Low","Medium","High"][int(v)])
        self.sl.valueChanged.connect(_lab); _lab(0)
        self.sl.valueChanged.connect(self._debounce)
        grid.addWidget(self.sl, 0, 1); grid.addWidget(self.lbl, 0, 2)

        self.cb_linear = QCheckBox("Linear data"); self.cb_linear.setChecked(False)
        self.cb_linear.toggled.connect(self._debounce)
        grid.addWidget(self.cb_linear, 1, 1)

        # Apply target
        grid.addWidget(QLabel("Apply to:"), 2, 0)
        self.cmb_target = QComboBox(); self.cmb_target.addItems(["Overwrite active view", "Create new view"])
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
        row.addWidget(QLabel("Zoom: Ctrl+Wheel"))

        b_minus = themed_toolbtn("zoom-out", "Zoom Out")
        b_plus  = themed_toolbtn("zoom-in", "Zoom In")
        b_fit   = themed_toolbtn("zoom-fit-best", "Fit to View")

        b_minus.clicked.connect(self.view.zoom_out)
        b_plus .clicked.connect(self.view.zoom_in)
        b_fit  .clicked.connect(lambda: self.view.fit_to(self.scene.itemsBoundingRect()))

        row.addWidget(b_minus)
        row.addWidget(b_plus)
        row.addWidget(b_fit)
        row.addStretch(1)
        v.addLayout(row)


        row2 = QHBoxLayout()
        b_apply = QPushButton("Apply"); b_apply.clicked.connect(self._apply)
        b_reset = QPushButton("Reset"); b_reset.clicked.connect(self._reset)
        b_cancel= QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        row2.addStretch(1); row2.addWidget(b_apply); row2.addWidget(b_reset); row2.addWidget(b_cancel)
        v.addLayout(row2)

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self._update_preview)

        self._set_pix(self._disp_base)
        self._update_preview()
        self.resize(900, 620)

    def _debounce(self): self._timer.start(180)

    def _set_pix(self, rgb):
        q = _qimage_from_rgb01(rgb)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    def _params(self):
        return int(self.sl.value()), bool(self.cb_linear.isChecked())

    def _update_preview(self):
        lvl, lin = self._params()
        try:
            out = compute_halo_b_gon(self.orig, reduction_level=lvl, is_linear=lin)
            # Display only: convert mono → RGB for the pixmap
            self._set_pix(_as_rgb(out))
        except Exception as e:
            QMessageBox.warning(self, "Halo-B-Gon", f"Preview failed:\n{e}")

    def _apply_overwrite(self, out: np.ndarray):
        if hasattr(self.doc, "set_image"):
            self.doc.set_image(out, step_name="Halo-B-Gon")
        elif hasattr(self.doc, "apply_numpy"):
            self.doc.apply_numpy(out, step_name="Halo-B-Gon")
        else:
            self.doc.image = out

    def _apply(self):
        lvl, lin = self._params()
        try:
            out = compute_halo_b_gon(self.orig, reduction_level=lvl, is_linear=lin)

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

                    self.accept()
                    return
                else:
                    # Fallback: try legacy spawner if present; else warn and overwrite.
                    spawner = getattr(self.parent(), "_spawn_new_view_from_numpy", None)
                    if spawner is None:
                        spawner = getattr(self.parent_ref, "_spawn_new_view_from_numpy", None)
                    if callable(spawner):
                        title = self.doc.display_name() if hasattr(self.doc, "display_name") else "Image"
                        spawner(out, f"{title} [Halo-B-Gon]")
                        self.accept()
                        return
                    else:
                        QMessageBox.warning(
                            self, "Halo-B-Gon",
                            "Could not find DocManager or window spawner; applying to the active view instead."
                        )
                        # fall through to overwrite

            # Overwrite current (original behavior)
            self._apply_overwrite(out)
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Halo-B-Gon", f"Failed to apply:\n{e}")



    def _reset(self):
        self.sl.setValue(0)
        self.cb_linear.setChecked(False)
        self._set_pix(self._disp_base)
        self._update_preview()
