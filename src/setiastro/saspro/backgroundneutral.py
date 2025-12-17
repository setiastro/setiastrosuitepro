# pro/backgroundneutral.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, QEvent, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QIcon, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene,
    QHBoxLayout, QPushButton, QMessageBox, QGraphicsRectItem
)

# Reuse existing helpers + autostretch
from setiastro.saspro.imageops.stretch import stretch_color_image
# Shared utilities
from setiastro.saspro.widgets.image_utils import extract_mask_from_document as _active_mask_array_from_doc
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn



# ----------------------------
# Core neutralization function
# ----------------------------
def background_neutralize_rgb(img: np.ndarray, rect_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply Background Neutralization to an RGB float32 image in [0,1],
    using an image-space rectangle (x, y, w, h) as the sample region.
    Returns a new float32 array in [0,1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Background Neutralization requires a 3-channel RGB image.")

    h, w, _ = img.shape
    x, y, rw, rh = rect_xywh
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    rw = max(1, min(int(rw), w - x))
    rh = max(1, min(int(rh), h - y))

    sample = img[y:y+rh, x:x+rw, :]
    medians = np.median(sample, axis=(0, 1)).astype(np.float32)         # (3,)
    avg_med = float(np.mean(medians))

    out = img.copy()
    eps = 1e-8
    for c in range(3):
        diff = float(medians[c] - avg_med)
        denom = 1.0 - diff
        if abs(denom) < eps:
            denom = eps if denom >= 0 else -eps
        out[..., c] = np.clip((out[..., c] - diff) / denom, 0.0, 1.0)

    return out.astype(np.float32, copy=False)


# ------------------------------------
# Auto background finder (SASv2 logic)
# ------------------------------------
def _find_best_patch_center(lum: np.ndarray) -> tuple[int, int]:
    """Port of your downhill-walk tile search (works on a luminance plane)."""
    h, w = lum.shape
    th, tw = h // 10, w // 10
    
    # Optimized: compute 10x10 tile medians using strided views where possible
    # This avoids repeated slicing and is cache-friendlier
    meds = np.zeros((10, 10), dtype=np.float32)
    
    # For tiles that fit evenly, use reshape + median (faster than loop)
    crop_h, crop_w = th * 10, tw * 10
    if crop_h <= h and crop_w <= w:
        lum_crop = lum[:crop_h, :crop_w]
        # Reshape to (10, th, 10, tw) and compute medians
        tiles = lum_crop.reshape(10, th, 10, tw).transpose(0, 2, 1, 3).reshape(10, 10, -1)
        meds = np.median(tiles, axis=2).astype(np.float32)
        
        # Handle edge tiles if image doesn't divide evenly
        if h > crop_h or w > crop_w:
            # Bottom row edge
            if h > crop_h:
                for j in range(10):
                    x0, x1 = j * tw, (j + 1) * tw if j < 9 else w
                    meds[9, j] = np.median(lum[9*th:h, x0:x1])
            # Right column edge
            if w > crop_w:
                for i in range(10):
                    y0, y1 = i * th, (i + 1) * th if i < 9 else h
                    meds[i, 9] = np.median(lum[y0:y1, 9*tw:w])
    else:
        # Fallback for very small images
        for i in range(10):
            for j in range(10):
                y0, x0 = i * th, j * tw
                y1 = (i + 1) * th if i < 9 else h
                x1 = (j + 1) * tw if j < 9 else w
                meds[i, j] = np.median(lum[y0:y1, x0:x1])

    idxs = np.argsort(meds.flatten())[:2]

    finals = []
    for idx in idxs:
        ti, tj = divmod(int(idx), 10)
        y0, x0 = ti * th, tj * tw
        y1 = (ti + 1) * th if ti < 9 else h
        x1 = (tj + 1) * tw if tj < 9 else w
        for _ in range(200):
            y = np.random.randint(y0, y1)
            x = np.random.randint(x0, x1)
            while True:
                mv, mpos = lum[y, x], (y, x)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and lum[ny, nx] < mv:
                            mv, mpos = lum[ny, nx], (ny, nx)
                if mpos == (y, x):
                    break
                y, x = mpos
            finals.append((y, x))

    best_val = np.inf
    best_pt = (h // 2, w // 2)
    for (y, x) in finals:
        y0 = max(0, y - 25); y1 = min(h, y + 25)
        x0 = max(0, x - 25); x1 = min(w, x + 25)
        m = np.median(lum[y0:y1, x0:x1])
        if m < best_val:
            best_val, best_pt = m, (y, x)
    return best_pt


def auto_rect_50x50(img_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find a robust 50×50 background rectangle (≥100 px margins) in image space.
    Returns (x, y, w, h).
    """
    h, w, ch = img_rgb.shape
    if ch != 3:
        raise ValueError("Auto background finder expects a 3-channel RGB image.")
    lum = img_rgb.mean(axis=2).astype(np.float32)

    cy, cx = _find_best_patch_center(lum)

    margin = 100
    half = 25
    min_cx, max_cx = margin + half, w - (margin + half)
    min_cy, max_cy = margin + half, h - (margin + half)
    cx = int(np.clip(cx, min_cx, max_cx))
    cy = int(np.clip(cy, min_cy, max_cy))

    # refine by ±half
    best_val = np.inf
    ty, tx = cy, cx
    for dy in (-half, 0, +half):
        for dx in (-half, 0, +half):
            y = int(np.clip(cy + dy, min_cy, max_cy))
            x = int(np.clip(cx + dx, min_cx, max_cx))
            y0, y1 = y - half, y + half
            x0, x1 = x - half, x + half
            m = np.median(lum[y0:y1, x0:x1])
            if m < best_val:
                best_val, ty, tx = m, y, x

    return (tx - half, ty - half, 50, 50)


# --------------------------------
# Headless apply (doc + preset in)
# --------------------------------
def apply_background_neutral_to_doc(doc, preset: dict | None = None):
    """
    Headless entrypoint (used by DnD shortcuts).
    Preset schema:
      {
        "mode": "auto" | "rect",
        # rect in normalized coords if mode == "rect"
        "rect_norm": [x0, y0, w, h]   # each in 0..1
      }
    Defaults to {"mode": "auto"}.
    """
    import numpy as np

    if preset is None:
        preset = {}
    mode = (preset.get("mode") or "auto").lower()

    base = np.asarray(doc.image).astype(np.float32, copy=False)
    if base.size == 0:
        raise ValueError("Empty image.")

    # Defensive normalization (should already be [0,1] in SASpro)
    maxv = float(np.nanmax(base))
    if maxv > 1.0 and np.isfinite(maxv):
        base = base / maxv

    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("Background Neutralization currently supports RGB images.")

    if mode == "rect":
        rn = preset.get("rect_norm")
        if not rn or len(rn) != 4:
            raise ValueError("rect mode requires rect_norm=[x,y,w,h] in normalized coords.")
        H, W, _ = base.shape
        x = int(np.clip(rn[0], 0, 1) * W)
        y = int(np.clip(rn[1], 0, 1) * H)
        w = int(np.clip(rn[2], 0, 1) * W)
        h = int(np.clip(rn[3], 0, 1) * H)
        rect = (x, y, max(w, 1), max(h, 1))
    else:
        rect = auto_rect_50x50(base)

    out = background_neutralize_rgb(base, rect)

    # Destination-mask blend (mask lives on the destination doc)
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        if out.ndim == 3:
            m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
        else:
            m3 = m.astype(np.float32, copy=False)
        base_for_blend = np.asarray(doc.image).astype(np.float32, copy=False)
        bmax = float(np.nanmax(base_for_blend))
        if bmax > 1.0 and np.isfinite(bmax):
            base_for_blend /= bmax
        out = base_for_blend * (1.0 - m3) + out * m3

    doc.apply_edit(
        out.astype(np.float32, copy=False),
        metadata={"step_name": "Background Neutralization", "preset": preset},
        step_name="Background Neutralization",
    )


# -------------------------
# Interactive BN dialog UI
# -------------------------
class BackgroundNeutralizationDialog(QDialog):
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.doc = doc
        if icon:
            self.setWindowIcon(icon)
        self.setWindowTitle("Background Neutralization")
        self.resize(900, 600)

        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.auto_stretch = False
        self.zoom_factor = 1.0
        self._user_zoomed = False

        # --- scene / view ---
        self.scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self)
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.graphics_view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.graphics_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # --- main layout ---
        layout = QVBoxLayout(self)
        instruction = QLabel("Draw a sample box or click ‘Find Background’ to auto-select.")
        layout.addWidget(instruction)
        layout.addWidget(self.graphics_view, 1)

        # Buttons row
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Neutralization")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_toggle_stretch = QPushButton("Enable Auto-Stretch")
        self.btn_find_bg = QPushButton("Find Background")
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_toggle_stretch)
        btn_row.addWidget(self.btn_find_bg)
        layout.addLayout(btn_row)

        # Zoom row
        # Zoom row (standardized themed toolbuttons)
        zoom_row = QHBoxLayout()

        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to View")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")

        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_fit)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addStretch(1)  # optional: keeps them left-aligned

        layout.addLayout(zoom_row)

        # Events
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_toggle_stretch.clicked.connect(self._toggle_auto_stretch)
        self.btn_find_bg.clicked.connect(self._on_find_background)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit.clicked.connect(self.fit_to_view)
        self.btn_zoom_in.clicked.connect(self.zoom_in)        

        self.graphics_view.viewport().installEventFilter(self)
        self.origin_scene = QPointF()
        self.current_rect_scene = QRectF()
        self.selection_item: QGraphicsRectItem | None = None
        self.drawing = False

        self._load_image()



    # ---------- image display ----------
    def _doc_image_normalized(self) -> np.ndarray:
        import numpy as np
        img = np.asarray(self.doc.image).astype(np.float32, copy=False)
        if img.size == 0:
            return img
        m = float(np.nanmax(img))
        if m > 1.0 and np.isfinite(m):
            img = img / m
        return img

    def _load_image(self):
        self.scene.clear()
        self.selection_item = None

        img = self._doc_image_normalized()
        if img is None or img.size == 0:
            QMessageBox.warning(self, "No Image", "Open an image first.")
            self.reject()
            return

        disp = img.copy()
        if self.auto_stretch and disp.ndim == 3 and disp.shape[2] == 3:
            disp = stretch_color_image(disp, 0.25, linked=False, normalize=False)

        # Build QImage/QPixmap
        if disp.ndim == 2:
            h, w = disp.shape
            qimg = QImage((disp * 255).astype(np.uint8).tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = disp.shape
            qimg = QImage((disp * 255).astype(np.uint8).tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        # Add to scene; force scene rect to native image pixels and place at (0,0)
        self.scene.clear()
        self.selection_item = None
        self.pixmap_item = self.scene.addPixmap(pix)
        self.pixmap_item.setPos(0, 0)
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())

        # Reset and fit (this sets initial view, later showEvent/resizeEvent will refit)
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_factor = 1.0
        self._user_zoomed = False

    def _toggle_auto_stretch(self):
        self.auto_stretch = not self.auto_stretch
        self.btn_toggle_stretch.setText("Disable Auto-Stretch" if self.auto_stretch else "Enable Auto-Stretch")
        self._load_image()

    # ---------- zoom ----------
    def eventFilter(self, source, event):
        if source is self.graphics_view.viewport():
            et = event.type()
            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.drawing = True
                self.origin_scene = self.graphics_view.mapToScene(event.pos())
                if self.selection_item:
                    self.scene.removeItem(self.selection_item)
                    self.selection_item = None
            elif et == QEvent.Type.MouseMove and self.drawing:
                cur = self.graphics_view.mapToScene(event.pos())
                self.current_rect_scene = QRectF(self.origin_scene, cur).normalized()
                if self.selection_item:
                    self.scene.removeItem(self.selection_item)
                pen = QPen(QColor(0, 255, 0), 2, Qt.PenStyle.DashLine)
                self.selection_item = self.scene.addRect(self.current_rect_scene, pen)
            elif et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self.drawing:
                self.drawing = False
                cur = self.graphics_view.mapToScene(event.pos())
                self.current_rect_scene = QRectF(self.origin_scene, cur).normalized()
                if self.selection_item:
                    self.scene.removeItem(self.selection_item)
                if self.current_rect_scene.width() < 10 or self.current_rect_scene.height() < 10:
                    QMessageBox.warning(self, "Selection Too Small", "Please draw a larger selection box.")
                    self.selection_item = None
                    self.current_rect_scene = QRectF()
                else:
                    pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.SolidLine)
                    self.selection_item = self.scene.addRect(self.current_rect_scene, pen)
        return super().eventFilter(source, event)

    def _on_find_background(self):
        img = self._doc_image_normalized()
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.warning(self, "Not RGB", "Background Neutralization supports RGB images.")
            return

        x, y, w, h = auto_rect_50x50(img)

        if self.selection_item:
            self.scene.removeItem(self.selection_item)

        pen = QPen(QColor(255, 215, 0), 2)  # gold
        rect_scene = QRectF(float(x), float(y), float(w), float(h))  # scene == image pixels now
        self.selection_item = self.scene.addRect(rect_scene, pen)
        self.current_rect_scene = rect_scene

    def _scene_rect_to_image_rect(self) -> tuple[int, int, int, int]:
        if not self.current_rect_scene or self.current_rect_scene.isNull():
            raise ValueError("No selection rectangle defined.")

        # Scene == image pixels (because we setSceneRect to pixmap bounds)
        bounds = self.pixmap_item.boundingRect()
        W = int(bounds.width())
        H = int(bounds.height())

        x = int(max(0.0, min(bounds.width(),  self.current_rect_scene.left())))
        y = int(max(0.0, min(bounds.height(), self.current_rect_scene.top())))
        w = int(max(1.0, min(bounds.width()  - x, self.current_rect_scene.width())))
        h = int(max(1.0, min(bounds.height() - y, self.current_rect_scene.height())))
        return (x, y, w, h)

    def _on_apply(self):
        try:
            rect = self._scene_rect_to_image_rect()
        except Exception as e:
            QMessageBox.warning(self, "No Selection", str(e))
            return

        img = self._doc_image_normalized()
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.warning(self, "Not RGB", "Background Neutralization supports RGB images.")
            return

        out = background_neutralize_rgb(img, rect)

        # Destination-mask blend
        m = _active_mask_array_from_doc(self.doc)
        if m is not None:
            if out.ndim == 3:
                m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
            else:
                m3 = m.astype(np.float32, copy=False)
            base_for_blend = self._doc_image_normalized()
            out = base_for_blend * (1.0 - m3) + out * m3

        # ---------- Build preset for Replay Last ----------
        preset = None
        try:
            H, W = img.shape[:2]
            x, y, w, h = rect
            if W > 0 and H > 0:
                rect_norm = [
                    float(x) / float(W),
                    float(y) / float(H),
                    float(w) / float(W),
                    float(h) / float(H),
                ]
            else:
                rect_norm = [0.0, 0.0, 1.0, 1.0]

            preset = {"mode": "rect", "rect_norm": rect_norm}

            # Walk up parent chain until we find the main window that carries
            # _last_headless_command
            main = self.parent()
            while main is not None and not hasattr(main, "_last_headless_command"):
                main = main.parent()

            if main is not None:
                try:
                    main._last_headless_command = {
                        "command_id": "background_neutral",
                        "preset": preset,
                    }
                    if hasattr(main, "_log"):
                        main._log(
                            "[Replay] Recorded background_neutral "
                            f"(mode=rect, rect_norm={rect_norm})"
                        )
                except Exception:
                    pass
        except Exception:
            # Fallback: at least record mode
            if preset is None:
                preset = {"mode": "rect"}

        # ---------- Apply edit (include preset in metadata) ----------
        meta = {
            "step_name": "Background Neutralization",
            "rect": rect,
        }
        if preset is not None:
            meta["preset"] = preset

        self.doc.apply_edit(
            out.astype(np.float32, copy=False),
            metadata=meta,
            step_name="Background Neutralization",
        )
        self.accept()


    def _zoom(self, factor: float):
        self._user_zoomed = True
        cur = self.graphics_view.transform().m11()
        new_scale = cur * factor
        if new_scale < 0.01 or new_scale > 100.0:
            return
        self.graphics_view.scale(factor, factor)

    def zoom_in(self):
        self._zoom(1.25)

    def zoom_out(self):
        self._zoom(0.8)

    def fit_to_view(self):
        self._user_zoomed = False
        self.graphics_view.resetTransform()
        # Fit the pixmap bounds (not a default huge scene)
        if hasattr(self, "pixmap_item") and self.pixmap_item is not None:
            self.graphics_view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, e):
        super().showEvent(e)
        # fit after the widget is actually visible
        QTimer.singleShot(0, self.fit_to_view)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # keep it fitted while the user hasn't manually zoomed
        if not self._user_zoomed:
            self.fit_to_view()

from setiastro.saspro.headless_utils import normalize_headless_main, unwrap_docproxy

def run_background_neutral_via_preset(main, preset=None, target_doc=None):
    from PyQt6.QtWidgets import QMessageBox
    from setiastro.saspro.backgroundneutral import apply_background_neutral_to_doc

    p = dict(preset or {})
    main, doc, _dm = normalize_headless_main(main, target_doc)

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main or None, "Background Neutralization", "Load an image first.")
        return

    apply_background_neutral_to_doc(doc, p)