# pro/wavescale_hdr.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton,
    QSlider, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
    QMessageBox, QProgressBar
)

# Import centralized widget
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# Import shared wavelet utilities
from setiastro.saspro.widgets.wavelet_utils import (
    conv_sep_reflect as _conv_sep_reflect,
    build_spaced_kernel as _build_spaced_kernel,
    atrous_decompose as _atrous_decompose,
    atrous_reconstruct as _atrous_reconstruct,
    rgb_to_lab as _rgb_to_lab,
    lab_to_rgb as _lab_to_rgb,
    B3_KERNEL as _B3,
)

# ──────────────────────────────────────────────────────────────────────────────
# Core math (shared by dialog + headless apply)
# ──────────────────────────────────────────────────────────────────────────────

def _mask_from_L(L: np.ndarray, gamma: float) -> np.ndarray:
    m = np.clip(L / 100.0, 0.0, 1.0).astype(np.float32)
    if gamma != 1.0:
        m = np.power(m, gamma, dtype=np.float32)
    return m

def _apply_dim_curve(rgb: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), gamma, dtype=np.float32)

def compute_wavescale_hdr(rgb_image: np.ndarray,
                          n_scales: int = 5,
                          compression_factor: float = 1.5,
                          mask_gamma: float = 1.0,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (transformed_rgb, luminance_mask). transformed_rgb is already
    reconstructed from modified L and gamma-dimmed.
    """
    lab = _rgb_to_lab(rgb_image)
    L0 = lab[..., 0].astype(np.float32, copy=True)
    scales = _atrous_decompose(L0, n_scales, base_kernel)

    mask = _mask_from_L(L0, mask_gamma)
    planes, residual = scales[:-1], scales[-1]

    for i, wp in enumerate(planes):
        decay = decay_rate ** i
        scale = (1.0 + (compression_factor - 1.0) * mask * decay) * 2.0
        planes[i] = wp * scale

    Lr = _atrous_reconstruct(planes + [residual])

    # midtones alignment
    med0 = float(np.median(L0))
    med1 = float(np.median(Lr)) or 1.0
    Lr = np.clip(Lr * (med0 / med1), 0.0, 100.0)

    lab[..., 0] = Lr
    rgb = _lab_to_rgb(lab)

    # gentle dimming curve to tame highlights
    rgb = _apply_dim_curve(rgb, gamma=1.0 + n_scales * 0.2)
    return rgb, mask

def compute_wavescale_hdr(rgb_image: np.ndarray,
                          n_scales: int = 5,
                          compression_factor: float = 1.5,
                          mask_gamma: float = 1.0,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5,
                          dim_gamma: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (transformed_rgb, luminance_mask).
    If dim_gamma is None, uses auto gamma = 1.0 + 0.2 * n_scales.
    """
    lab = _rgb_to_lab(rgb_image)
    L0 = lab[..., 0].astype(np.float32, copy=True)
    scales = _atrous_decompose(L0, n_scales, base_kernel)

    mask = _mask_from_L(L0, mask_gamma)
    planes, residual = scales[:-1], scales[-1]

    for i, wp in enumerate(planes):
        decay = decay_rate ** i
        scale = (1.0 + (compression_factor - 1.0) * mask * decay) * 2.0
        planes[i] = wp * scale

    Lr = _atrous_reconstruct(planes + [residual])

    # midtones alignment
    med0 = float(np.median(L0))
    med1 = float(np.median(Lr)) or 1.0
    Lr = np.clip(Lr * (med0 / med1), 0.0, 100.0)

    lab[..., 0] = Lr
    rgb = _lab_to_rgb(lab)

    # dimming curve
    g = (1.0 + n_scales * 0.2) if dim_gamma is None else float(dim_gamma)
    rgb = _apply_dim_curve(rgb, gamma=g)
    return rgb, mask


# ──────────────────────────────────────────────────────────────────────────────
# Worker (QObject in its own QThread) for the dialog
# ──────────────────────────────────────────────────────────────────────────────

class HDRWorker(QObject):
    progress_update = pyqtSignal(str, int)                # (step, percent)
    finished = pyqtSignal(np.ndarray, np.ndarray)         # (transformed_rgb, mask)

    def __init__(self, rgb_image: np.ndarray, n_scales: int, compression_factor: float,
                 mask_gamma: float, base_kernel: np.ndarray):
        super().__init__()
        self.rgb_image = rgb_image
        self.n_scales = n_scales
        self.compression_factor = compression_factor
        self.mask_gamma = mask_gamma
        self.base_kernel = base_kernel

    def run(self):
        try:
            self.progress_update.emit(self.tr("Converting to Lab color space…"), 10)
            # progress checkpoints inline here are cosmetic
            self.progress_update.emit(self.tr("Decomposing luminance with starlet…"), 20)
            # full compute
            transformed, mask = compute_wavescale_hdr(
                self.rgb_image, self.n_scales, self.compression_factor, self.mask_gamma, self.base_kernel
            )
            self.progress_update.emit(self.tr("Finalizing…"), 95)
            self.finished.emit(transformed, mask)
        except Exception as e:
            print("WaveScale HDR error:", e)
            self.finished.emit(None, None)

# ──────────────────────────────────────────────────────────────────────────────
# Simple mask window
# ──────────────────────────────────────────────────────────────────────────────

class MaskDisplayWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("HDR Mask (L-based)"))
        self.lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl.setFixedSize(400, 400)  # keep it small
        lay = QVBoxLayout(self)
        lay.addWidget(self.lbl)

    def update_mask(self, mask: np.ndarray):
        if mask is None:
            return
        m = np.clip(mask, 0, 1).astype(np.float32)
        m8 = (m * 255.0).astype(np.uint8)
        if m8.ndim == 2:
            h, w = m8.shape
            rgb = np.repeat(m8[..., None], 3, axis=2)
        else:
            h, w, _ = m8.shape
            rgb = m8
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.lbl.setPixmap(pix)

# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class WaveScaleHDRDialogPro(QDialog):
    applied_preset = pyqtSignal(object, dict)

    def __init__(self, parent, doc, icon_path: str | None = None, *, headless: bool=False, bypass_guard: bool=False):
        super().__init__(parent)
        self.setWindowTitle(self.tr("WaveScale HDR"))
        self._headless = bool(headless)
        self._bypass_guard = bool(bypass_guard)
        if self._headless:
            # Don’t show any windows; we’ll still exec() to run the event loop.
            try: self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        if icon_path:
            try: self.setWindowIcon(QIcon(icon_path))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        self.resize(980, 700)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Active document has no image.")

        # normalize to float32 [0..1] RGB for processing/preview
        img = np.asarray(base, dtype=np.float32)
        if img.ndim == 2:
            img_rgb = np.repeat(img[:, :, None], 3, axis=2)
            self._was_mono = True
            self._mono_shape = img.shape
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = np.repeat(img, 3, axis=2)
            self._was_mono = True
            self._mono_shape = img.shape
        else:
            img_rgb = img[:, :, :3]
            self._was_mono = False
            self._mono_shape = None

        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(img_rgb)) or 1.0
            img_rgb = img_rgb / max(1.0, maxv)
        img_rgb = np.clip(img_rgb, 0.0, 1.0).astype(np.float32, copy=False)

        self.original_rgb = img_rgb
        self.preview_rgb  = img_rgb.copy()

        # scene/view (⚠️ use ZoomableGraphicsView)
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        # optional: keep your scroll area wrapper
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.view)

        # controls (add zoom row)
        self.grp = QGroupBox(self.tr("HDR Controls"))
        form = QFormLayout(self.grp)

        self.s_scales = QSlider(Qt.Orientation.Horizontal); self.s_scales.setRange(2, 10); self.s_scales.setValue(5)
        self.s_comp   = QSlider(Qt.Orientation.Horizontal); self.s_comp.setRange(10, 500); self.s_comp.setValue(150)
        self.s_gamma  = QSlider(Qt.Orientation.Horizontal); self.s_gamma.setRange(10, 1000); self.s_gamma.setValue(500)

        form.addRow(self.tr("Number of Scales:"), self.s_scales)
        form.addRow(self.tr("Coarse Compression:"), self.s_comp)
        form.addRow(self.tr("Mask Gamma:"), self.s_gamma)

        row = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_toggle  = QPushButton(self.tr("Show Original")); self.btn_toggle.setCheckable(True)
        row.addWidget(self.btn_preview); row.addWidget(self.btn_toggle)
        form.addRow(row)

        # ↓ NEW: zoom controls
        zoom_row = QHBoxLayout()
        self.btn_zoom_in  = QPushButton(self.tr("Zoom In"))
        self.btn_zoom_out = QPushButton(self.tr("Zoom Out"))
        self.btn_fit      = QPushButton(self.tr("Fit to Preview"))
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_fit)
        form.addRow(zoom_row)

        # progress group (unchanged)
        self.prog_grp = QGroupBox(self.tr("Processing Progress"))
        vprog = QVBoxLayout(self.prog_grp)
        self.lbl_step = QLabel(self.tr("Idle"))
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        vprog.addWidget(self.lbl_step); vprog.addWidget(self.bar)

        # bottom buttons (unchanged)
        bot = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply to Document")); self.btn_apply.setEnabled(False)
        self.btn_reset = QPushButton(self.tr("Reset"))
        self.btn_close = QPushButton(self.tr("Close"))
        bot.addStretch(1); bot.addWidget(self.btn_apply); bot.addWidget(self.btn_reset); bot.addWidget(self.btn_close)

        # layout (unchanged)
        main = QVBoxLayout(self)
        main.addWidget(self.scroll)
        h = QHBoxLayout()
        h.addWidget(self.grp, 3)
        h.addWidget(self.prog_grp, 1)
        main.addLayout(h)
        main.addLayout(bot)

        # mask window
        self.mask_win = MaskDisplayWindow(self)
        if not self._headless:
            self.mask_win.show()


        # kernel
        self.base_kernel = _B3

        # connections
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_toggle.clicked.connect(self._toggle)

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_fit.clicked.connect(lambda: self.view.fit_item(self.pix))

        # ── Mask shown immediately ───────────────────────────────────────────
        # Precompute L from original and push initial mask to the small window
        self._lab_original = _rgb_to_lab(self.original_rgb)
        self._L_original = self._lab_original[..., 0].astype(np.float32, copy=True)
        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.timeout.connect(self._update_mask_from_gamma)
        self.s_gamma.valueChanged.connect(self._schedule_mask_refresh)

        # show initial mask right away
        self._update_mask_from_gamma()

        # initial pix
        self._set_pix(self.preview_rgb)

    def apply_preset(self, p: dict):
        # sliders are integer; map floats to their scales
        ns = int(p.get("n_scales", 5))
        comp = float(p.get("compression_factor", 1.5))
        mg  = float(p.get("mask_gamma", 5.0))  # dialog default is 5.0 (slider 500)
        # clamp safely
        ns = max(2, min(10, ns))
        comp_i = int(max(10, min(500, round(comp*100))))      # 1.0..5.0 -> 100..500
        mg_i   = int(max(10, min(1000, round(mg*100))))       # 0.1..10.0 -> 10..1000
        self.s_scales.setValue(ns)
        self.s_comp.setValue(comp_i)
        self.s_gamma.setValue(mg_i)
        # refresh mask preview (even if window is hidden)
        self._update_mask_from_gamma()

    def _headless_guard_active(self) -> bool:
        """Only guard true concurrent *headless* runs; ignore stale locks."""
        # If we are not launching headless, never block the interactive UI.
        if not self._headless:
            return False

        # Parent flags
        p = self.parent()
        if p and (getattr(p, "_wavescale_guard", False) or getattr(p, "_wavescale_headless_running", False)):
            return True

        # Settings lock with TTL
        try:
            s = QSettings()
            in_prog = bool(s.value("wavescale/headless_in_progress", False))
            started  = float(s.value("wavescale/headless_started_at", 0.0))
        except Exception:
            in_prog, started = False, 0.0

        if not in_prog:
            return False

        # consider anything older than 5 minutes stale
        import time
        if (time.time() - started) > 5 * 60:
            try:
                s.remove("wavescale/headless_in_progress")
                s.remove("wavescale/headless_started_at")
            except Exception:
                pass
            return False

        return True

    def showEvent(self, e):
        super().showEvent(e)
        if not self._bypass_guard and self._headless_guard_active():
            # Soft warning instead of rejecting the dialog
            try:
                QMessageBox.information(
                    self, self.tr("WaveScale HDR"),
                    self.tr("A headless HDR run appears to be in progress. "
                            "This window will remain open; you can still preview safely.")
                )
            except Exception:
                pass

    def exec(self) -> int:
        if not self._bypass_guard and self._headless_guard_active():
            return 0
        return super().exec()

    def _get_doc_active_mask_2d(self) -> np.ndarray | None:
        """
        Return the document's active mask as a 2-D float32 in [0..1],
        resized to the current image size. If none, return None.
        """
        doc = getattr(self, "_doc", None)
        if doc is None:
            return None

        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None

        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        if layer is None:
            return None

        # Safely pick the first non-None payload without using boolean 'or'
        data = None
        # object with attributes
        for attr in ("data", "mask", "image", "array"):
            if hasattr(layer, attr):
                val = getattr(layer, attr)
                if val is not None:
                    data = val
                    break
        # plain ndarray?
        if data is None and isinstance(layer, np.ndarray):
            data = layer
        # dict-like layer?
        if data is None and isinstance(layer, dict):
            for key in ("data", "mask", "image", "array"):
                if key in layer and layer[key] is not None:
                    data = layer[key]
                    break

        if data is None:
            return None

        m = np.asarray(data)

        # collapse RGB/alpha to gray if needed
        if m.ndim == 3:
            m = m.mean(axis=2)

        m = m.astype(np.float32, copy=False)
        # normalize to [0,1] if it looks like 0..255 or 0..65535
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0:
                m /= mx
        m = np.clip(m, 0.0, 1.0)

        # resize to current image size (nearest)
        H, W = self.original_rgb.shape[:2]
        if m.shape != (H, W):
            yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
            m = m[yi][:, xi]

        return m


    def _combine_with_doc_mask(self, hdr_mask: np.ndarray) -> np.ndarray:
        """
        Multiply the HDR luminance mask by the document active mask (if any).
        Shapes are matched to image size.
        """
        m_doc = self._get_doc_active_mask_2d()
        if m_doc is None:
            return hdr_mask
        # both are already (H, W) float32 in [0..1]
        return np.clip(hdr_mask * m_doc, 0.0, 1.0)


    def _set_pix(self, rgb: np.ndarray):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText(self.tr("Show Preview"))
            self._set_pix(self.original_rgb)
        else:
            self.btn_toggle.setText(self.tr("Show Original"))
            self._set_pix(self.preview_rgb)

    def _reset(self):
        self.s_scales.setValue(5)
        self.s_comp.setValue(150)
        self.s_gamma.setValue(500)
        self.preview_rgb = self.original_rgb.copy()
        self._set_pix(self.preview_rgb)
        self.lbl_step.setText(self.tr("Idle")); self.bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))

    def _start_preview(self):
        self.btn_preview.setEnabled(False); self.btn_apply.setEnabled(False)
        n_scales = int(self.s_scales.value())
        comp     = float(self.s_comp.value()) / 100.0
        mgamma   = float(self.s_gamma.value()) / 100.0

        self.thread = QThread(self)
        self.worker = HDRWorker(self.original_rgb, n_scales, comp, mgamma, self.base_kernel)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_progress(self, step: str, pct: int):
        self.lbl_step.setText(step); self.bar.setValue(pct)

    def _on_finished(self, transformed_rgb: np.ndarray, mask: np.ndarray):
        self.btn_preview.setEnabled(True)
        if transformed_rgb is None:
            QMessageBox.critical(self, self.tr("WaveScale HDR"), self.tr("Processing failed."))
            return

        # ← NEW: combine HDR's luminance mask with the doc's active mask (if present)
        mask_comb = self._combine_with_doc_mask(mask)

        # blend preview: original*(1-mask) + transformed*mask
        m3 = np.repeat(mask_comb[..., None], 3, axis=2)
        self.preview_rgb = self.original_rgb * (1.0 - m3) + transformed_rgb * m3
        self._set_pix(self.preview_rgb)

        # show the *combined* mask in the little window
        self.mask_win.setWindowTitle(
            self.tr("HDR Mask (L × Active Mask)") if self._get_doc_active_mask_2d() is not None else self.tr("HDR Mask (L-based)")
        )
        self.mask_win.update_mask(mask_comb)

        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))
        self.lbl_step.setText(self.tr("Preview ready")); self.bar.setValue(100)
        # Headless: apply immediately (exactly like clicking "Apply to Document")
        if self._headless:
            QTimer.singleShot(0, self._apply_to_doc)

    def _apply_to_doc(self):
        out = self.preview_rgb
        if self._was_mono:
            # collapse back to mono (keep original shape: 2D or H×W×1)
            mono = np.mean(out, axis=2, dtype=np.float32)
            if self._mono_shape and len(self._mono_shape) == 3 and self._mono_shape[2] == 1:
                mono = mono[:, :, None]
            out = mono

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out, step_name="WaveScale HDR")
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out, step_name="WaveScale HDR")
            else:
                self._doc.image = out
        except Exception as e:
            QMessageBox.critical(self, self.tr("WaveScale HDR"), self.tr("Failed to write to document:\n{0}").format(e))
            return

        # ── Build preset from current sliders ─────────────────────────
        try:
            preset = {
                "n_scales": int(self.s_scales.value()),
                "compression_factor": float(self.s_comp.value()) / 100.0,
                "mask_gamma": float(self.s_gamma.value()) / 100.0,
            }
        except Exception:
            preset = {}

        # ── Register as last_headless_command on the main window ─────
        try:
            main = self.parent()
            if main is not None:
                payload = {
                    "command_id": "wavescale_hdr",
                    "preset": dict(preset),
                }
                setattr(main, "_last_headless_command", payload)

                # Optional debug log (mirrors other tools)
                try:
                    if hasattr(main, "_log"):
                        ns   = int(preset.get("n_scales", 5))
                        comp = float(preset.get("compression_factor", 1.5))
                        mg   = float(preset.get("mask_gamma", 5.0))
                        main._log(
                            f"[Replay] Registered WaveScale HDR as last action "
                            f"(n_scales={ns}, compression={comp:.2f}, mask_gamma={mg:.2f})"
                        )
                except Exception:
                    pass
        except Exception:
            # never let replay wiring break the apply
            pass

        # ── (optional) keep emitting signal if you want it elsewhere ──
        try:
            self.applied_preset.emit(self._doc, preset)
        except Exception:
            pass

        # Dialog stays open so user can apply to other images
        # Refresh document reference for next operation
        self._refresh_document_from_active()

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self._doc:
                    self._doc = new_doc
                    # Reset L channel and refresh preview for new document
                    self._L_original = None
                    self._last_preview = None
        except Exception:
            pass


    def _schedule_mask_refresh(self, _value):
        # debounce to ~0.25s
        self._mask_timer.start(250)

    def _update_mask_from_gamma(self):
        gamma = float(self.s_gamma.value()) / 100.0
        hdr_mask = _mask_from_L(self._L_original, gamma=gamma)
        mask_comb = self._combine_with_doc_mask(hdr_mask)
        self.mask_win.setWindowTitle(
            self.tr("HDR Mask (L × Active Mask)") if self._get_doc_active_mask_2d() is not None else self.tr("HDR Mask (L-based)")
        )
        self.mask_win.update_mask(mask_comb)     
