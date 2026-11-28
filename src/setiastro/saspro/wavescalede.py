#saspro/wavescalede.py
from __future__ import annotations
import functools
import math
import numpy as np

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton,
    QSlider, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
    QMessageBox, QProgressBar, QMainWindow
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
# off-GUI-thread display helpers shared with WaveScale HDR
from setiastro.saspro.wavescale_hdr import (
    _JobThread, _apply_output, _doc_mask_finish, _mask_thumb_qimage, _rgb01, _rgb_to_qimage,
    _amount_job, blend_amount, clamp_amount,
)
# Import shared wavelet utilities
from setiastro.saspro.widgets.wavelet_utils import (
    conv_sep_reflect as _conv_sep_reflect,
    build_spaced_kernel as _build_spaced_kernel,
    atrous_decompose as _atrous_decompose,
    rgb_to_lab as _rgb_to_lab,
    lab_to_rgb as _lab_to_rgb,
    gauss_blur as _gauss_blur,
    run_row_bands as _run_row_bands,
    B3_KERNEL as _B3,
)

# ─────────────────────────────────────────────────────────────────────────────
# Core math (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _resize_mask_nn(mask2d: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    H, W = target_hw
    if mask2d.shape == (H, W):
        return mask2d.astype(np.float32, copy=False)
    yi = (np.linspace(0, mask2d.shape[0] - 1, H)).astype(np.int32)
    xi = (np.linspace(0, mask2d.shape[1] - 1, W)).astype(np.int32)
    return mask2d[yi][:, xi].astype(np.float32, copy=False)

# Darkness mask (scales 2–4, negative parts, mean → normalize → gamma → smooth → mild S-curve)
def _darkness_mask(L: np.ndarray, n_scales: int, base_k: np.ndarray, gamma: float) -> np.ndarray:
    return _darkness_mask_finish(_darkness_mask_base(L, n_scales, base_k), gamma)

def _darkness_mask_base(L: np.ndarray, n_scales: int, base_k: np.ndarray) -> np.ndarray:
    """The gamma-independent part of _darkness_mask (the dialog caches it)."""
    # Only planes[1:4] are read, and the à-trous planes below index 4 do not
    # depend on how many scales follow, so decomposing min(n, 4) scales gives
    # the same planes[1:4] (for n < 4 that slice includes the residual).
    planes = _atrous_decompose(L, min(n_scales, 4), base_k)
    # mid-scales: 1:4 (0-based → skip 0)
    sel = planes[1:4]
    if len(sel) == 0:
        return np.zeros_like(L, dtype=np.float32)
    # Elementwise steps run on row bands in parallel (numpy releases the GIL);
    # every pixel sees the same float32 operations as a whole-image call.
    combined = np.empty(sel[0].shape, dtype=np.float32)

    def _neg_mean_rows(sl):
        neg = [np.clip(-p[sl], 0, None) for p in sel]
        combined[sl] = np.mean(neg, axis=0).astype(np.float32)
    _run_row_bands(_neg_mean_rows, combined.shape[0])
    denom = float(np.max(combined) + 1e-8)

    def _norm_rows(sl):
        combined[sl] = combined[sl] / denom
    _run_row_bands(_norm_rows, combined.shape[0])
    return combined

def _darkness_mask_finish(m: np.ndarray, gamma: float) -> np.ndarray:
    """gamma → smooth → mild S-curve; never modifies m."""
    H = m.shape[0]
    if gamma != 1.0:
        p = np.empty(m.shape, dtype=np.float32)

        def _pow_rows(sl):
            p[sl] = np.power(m[sl], float(gamma), dtype=np.float32)
        _run_row_bands(_pow_rows, H)
        m = p
    m = _gauss_blur(m, sigma=3.0).astype(np.float32, copy=False)
    # gentle brighten of mids
    out = np.empty(m.shape, dtype=np.float32)

    def _curve_rows(sl):
        g = m[sl]
        out[sl] = np.clip(1.5 * g - 0.5 * (g * g), 0.0, 1.0).astype(np.float32)
    _run_row_bands(_curve_rows, H)
    return out


def _enhance_and_rebuild(planes: list, residual: np.ndarray, mask: np.ndarray,
                         boost_factor: float, decay_rate: float, hi: float) -> np.ndarray:
    """One Dark Enhancer iteration after the decomposition: for i >= 1,
    planes[i] -= clip(-planes[i], 0)·mask·(boost-1)·decay^i, then
    clip(residual + Σ planes, 0, hi). Row bands in parallel, same float32 ops
    per pixel as the whole-image version."""
    out = np.empty(residual.shape, dtype=np.float32)

    def _rows(sl):
        m = mask[sl]
        acc = residual[sl].astype(np.float32, copy=True)
        for i in range(len(planes)):
            p = planes[i][sl]
            if i != 0:      # skip highest frequency
                decay = decay_rate ** i
                neg = np.clip(-p, 0, None)
                enhancement = neg * m * (boost_factor - 1.0) * decay
                p = p - enhancement
            acc += p
        out[sl] = np.clip(acc, 0.0, hi)
    _run_row_bands(_rows, residual.shape[0])
    return out

# Main compute (mono or RGB)
def compute_wavescale_dse(image: np.ndarray,
                          n_scales: int = 6,
                          boost_factor: float = 5.0,
                          mask_gamma: float = 1.0,
                          iterations: int = 2,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5,
                          external_mask: np.ndarray | None = None  # ← NEW
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    WaveScale Dark Enhancer.
    Returns (output_image, darkness_mask_used).
    If external_mask is provided (2-D [0..1]), it will be multiplied into the darkness mask.
    """
    arr = np.asarray(image, dtype=np.float32)

    # normalize external mask now
    ext = None
    if external_mask is not None:
        m = np.asarray(external_mask)
        if m.ndim == 3:  # collapse RGB(A)
            m = m.mean(axis=2)
        m = np.clip(m.astype(np.float32), 0.0, 1.0)
        ext = _resize_mask_nn(m, arr.shape[:2])

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        L = arr.squeeze().astype(np.float32, copy=True)  # [0..1]
        mask = np.zeros_like(L, dtype=np.float32)  # define for return
        for it in range(int(iterations)):
            mask = _darkness_mask(L, n_scales, base_kernel, mask_gamma)
            if ext is not None:
                mask = np.clip(mask * ext, 0.0, 1.0)  # ← combine here

            planes = _atrous_decompose(L, n_scales, base_kernel)
            residual = planes.pop()
            L = _enhance_and_rebuild(planes, residual, mask, boost_factor, decay_rate, 1.0)

        out = L.astype(np.float32, copy=False)
        return out, mask.astype(np.float32, copy=False)

    # RGB path
    rgb = np.clip(arr[:, :, :3], 0.0, 1.0).astype(np.float32, copy=False)
    lab = _rgb_to_lab(rgb)
    L = lab[..., 0].astype(np.float32, copy=True)
    mask = np.zeros(L.shape, dtype=np.float32)  # define for return
    for it in range(int(iterations)):
        mask = _darkness_mask(np.clip(L / 100.0, 0.0, 1.0), n_scales, base_kernel, mask_gamma)
        if ext is not None:
            mask = np.clip(mask * ext, 0.0, 1.0)  # ← combine here

        planes = _atrous_decompose(L, n_scales, base_kernel)
        residual = planes.pop()
        L = _enhance_and_rebuild(planes, residual, mask, boost_factor, decay_rate, 100.0)

    lab[..., 0] = L
    out_rgb = _lab_to_rgb(lab)
    return out_rgb.astype(np.float32, copy=False), mask.astype(np.float32, copy=False)

# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
class DSEWorker(QObject):
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(object, object)  # (output, mask) | (payload, None)

    def __init__(self, image: np.ndarray, n_scales: int, boost: float, gamma: float,
                 base_kernel: np.ndarray, iterations: int,
                 external_mask: np.ndarray | None = None, *, post=None,
                 doc_mask_pre: np.ndarray | None = None):
        super().__init__()
        self.post = post              # post(out, mask, doc_mask) -> payload dict, run here
        self.doc_mask_pre = doc_mask_pre   # _doc_mask_prenorm(); clip + resize run here
        self.image = image
        self.n_scales = n_scales
        self.boost = boost
        self.gamma = gamma
        self.base_kernel = base_kernel
        self.iterations = iterations
        self.external_mask = external_mask

    def run(self):
        try:
            self.progress_update.emit(self.tr("Analyzing dark structure…"), 20)
            if self.doc_mask_pre is not None:
                self.external_mask = _doc_mask_finish(self.doc_mask_pre, *self.image.shape[:2])
            out, mask = compute_wavescale_dse(
                self.image, self.n_scales, self.boost, self.gamma,
                self.iterations, self.base_kernel,
                external_mask=self.external_mask             # ← NEW
            )
            self.progress_update.emit(self.tr("Finalizing…"), 95)
            if self.post is not None:
                self.finished.emit(self.post(out, mask, self.external_mask), None)
            else:
                self.finished.emit(out, mask)
        except Exception as e:
            print("WaveScale DSE error:", e)
            self.finished.emit(None, None)


def _dse_preview_payload(original: np.ndarray, out: np.ndarray, mask: np.ndarray,
                         doc_mask: np.ndarray | None, box_wh, want_qimage: bool,
                         amount: float = 1.0) -> dict:
    """What _on_finished used to build on the GUI thread. "full" is the result
    at Amount 1; "preview" is it blended with the original at `amount`."""
    if out.ndim == 2:
        out_rgb = np.repeat(out[:, :, None], 3, axis=2)
    else:
        out_rgb = out

    # Respect the document mask: original*(1-M) + out*M, per channel (the same
    # float32 ops as the np.repeat'ed mask, without repeat's GIL-held copy)
    if doc_mask is not None:
        M = doc_mask.astype(np.float32)
        inv = 1.0 - M
        dt = np.result_type(original.dtype, inv.dtype, out_rgb.dtype, M.dtype)
        preview = np.empty(original.shape, dtype=dt)
        for c in range(original.shape[2]):
            np.add(original[..., c] * inv, out_rgb[..., c] * M, out=preview[..., c])
    else:
        preview = out_rgb

    # show combined mask (internal darkness mask × doc mask)
    if doc_mask is not None:
        mask = np.clip(mask.astype(np.float32) * doc_mask.astype(np.float32), 0.0, 1.0)
    shown = blend_amount(original, preview, amount)
    return {
        "full": preview,
        "preview": shown,
        "amount": float(amount),
        "qimg": _rgb_to_qimage(shown) if want_qimage else None,
        "thumb": _mask_thumb_qimage(mask, *box_wh) if box_wh else None,
    }


def _dse_mask_job(original: np.ndarray, cache: dict, n_scales: int, base_k: np.ndarray,
                  gamma: float, doc_mask_pre: np.ndarray | None, box_wh, want_qimage: bool) -> dict:
    """Mask-window refresh off the GUI thread. L and the gamma-independent part
    of the darkness mask are cached; a gamma step only redoes gamma + blur."""
    out = {}
    doc_mask = None if doc_mask_pre is None else _doc_mask_finish(doc_mask_pre, *original.shape[:2])
    L01 = cache.get("L01")
    if L01 is None:
        lab = _rgb_to_lab(original)
        L = lab[..., 0] / 100.0
        L01 = np.clip(L, 0.0, 1.0)
        out["L01"] = L01
    key = min(n_scales, 4)
    base = cache.get("base") if cache.get("base_key") == key else None
    if base is None:
        base = _darkness_mask_base(L01, n_scales, base_k)
        out["base"], out["base_key"] = base, key
    algo_mask = _darkness_mask_finish(base, gamma)
    if doc_mask is not None:
        algo_mask = np.clip(algo_mask.astype(np.float32) * doc_mask.astype(np.float32), 0.0, 1.0)
    out["thumb"] = _mask_thumb_qimage(algo_mask, *box_wh)
    out["has_doc_mask"] = doc_mask is not None
    if want_qimage:
        out["qimg"] = _rgb_to_qimage(original)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Small mask window (fixed ~400×400, always shows a zoomed-out mask)
# ─────────────────────────────────────────────────────────────────────────────
class _MaskWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Dark Mask"))
        self.setMinimumSize(300, 300)
        self.resize(400, 400)
        v = QVBoxLayout(self)
        self.lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl)

    def set_mask(self, mask: np.ndarray):
        m = np.clip(mask, 0, 1).astype(np.float32)
        m8 = (m * 255.0).astype(np.uint8)
        if m8.ndim == 2:
            h, w = m8.shape
            q = QImage(m8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = m8.shape
            q = QImage(m8.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(q)
        box = self.size()
        pm2 = pm.scaled(box, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl.setPixmap(pm2)

    def set_thumb(self, qimg: QImage):
        """Show a thumbnail already built (off the GUI thread) by _mask_thumb_qimage."""
        self.lbl.setPixmap(QPixmap.fromImage(qimg))

# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────
class WaveScaleDarkEnhancerDialogPro(QDialog):
    def __init__(self, parent, doc, icon_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("WaveScale Dark Enhancer"))
        if icon_path:
            try: self.setWindowIcon(QIcon(icon_path))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        self.resize(980, 700)
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
        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Active document has no image.")

        img = np.asarray(base, dtype=np.float32)
        if img.ndim == 2:
            self._was_mono = True
            self._mono_shape = img.shape
            rgb = img                   # replicated to 3 channels by _rgb01 below
        elif img.ndim == 3 and img.shape[2] == 1:
            self._was_mono = True
            self._mono_shape = img.shape
            rgb = img
        else:
            self._was_mono = False
            self._mono_shape = None
            rgb = img[:, :, :3]
        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(rgb)) or 1.0
            rgb = rgb / max(1.0, maxv)
        rgb = _rgb01(rgb)               # == np.clip(3-channel, 0, 1).astype(float32)

        self.original = rgb
        self.preview  = rgb          # never modified in place; replaced by each preview
        self._full = None            # last preview at Amount 100% (None: no preview yet)
        self._blend_gen = 0          # bumped by each new preview / Reset; stale re-blends drop
        self._blend_busy = False
        self._blend_pending = False

        # off-GUI-thread state: L / darkness-mask cache, the two view pixmaps, mask jobs
        self._mask_cache = {}
        self._pm_original = None
        self._pm_preview = None
        self._mask_busy = False
        self._mask_pending = False
        self._mask_thr = None

        # scene/view
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.view)

        # controls
        self.grp = QGroupBox(self.tr("Dark Enhancer Controls"))
        form = QFormLayout(self.grp)

        self.s_scales = QSlider(Qt.Orientation.Horizontal); self.s_scales.setRange(2, 10);  self.s_scales.setValue(6)
        self.s_boost  = QSlider(Qt.Orientation.Horizontal); self.s_boost.setRange(10, 1000); self.s_boost.setValue(500)  # 0.10..10.00
        self.s_gamma  = QSlider(Qt.Orientation.Horizontal); self.s_gamma.setRange(10, 1000); self.s_gamma.setValue(100)  # 0.10..10.00
        self.s_iters  = QSlider(Qt.Orientation.Horizontal); self.s_iters.setRange(1, 10);   self.s_iters.setValue(2)
        # Amount: blend of the finished result with the original (no recompute)
        self.s_amount = QSlider(Qt.Orientation.Horizontal); self.s_amount.setRange(0, 100); self.s_amount.setValue(100)
        self.lbl_amount_val = QLabel(f"{int(self.s_amount.value())}%")
        self.lbl_amount_val.setMinimumWidth(42)
        self.lbl_amount_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _tip_amount = self.tr(
            "Amount (0-100%)\n\n"
            "Blends the finished result with the original image. 100% is the "
            "full effect, 0% leaves the image unchanged.\n\n"
            "Changing it after a Preview re-blends the preview without "
            "recomputing. Default: 100%."
        )
        self.s_amount.setToolTip(_tip_amount)
        self.lbl_amount_val.setToolTip(_tip_amount)
        self.s_amount.valueChanged.connect(lambda v: self.lbl_amount_val.setText(f"{int(v)}%"))
        amount_row = QHBoxLayout()
        amount_row.addWidget(self.s_amount, 1)
        amount_row.addWidget(self.lbl_amount_val)

        form.addRow(self.tr("Number of Scales:"), self.s_scales)
        form.addRow(self.tr("Boost Factor:"), self.s_boost)
        form.addRow(self.tr("Mask Gamma:"), self.s_gamma)
        form.addRow(self.tr("Iterations:"), self.s_iters)
        form.addRow(self.tr("Amount:"), amount_row)

        row = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_toggle  = QPushButton(self.tr("Show Original")); self.btn_toggle.setCheckable(True)
        row.addWidget(self.btn_preview); row.addWidget(self.btn_toggle)
        form.addRow(row)

        #sasprogress
        self.prog_grp = QGroupBox(self.tr("Progress"))
        vprog = QVBoxLayout(self.prog_grp)
        self.lbl_step = QLabel(self.tr("Idle"))
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        vprog.addWidget(self.lbl_step); vprog.addWidget(self.bar)

        # bottom
        bot = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply to Document")); self.btn_apply.setEnabled(False)
        self.btn_reset = QPushButton(self.tr("Reset"))
        self.btn_close = QPushButton(self.tr("Close"))
        bot.addStretch(1); bot.addWidget(self.btn_apply); bot.addWidget(self.btn_reset); bot.addWidget(self.btn_close)

        # layout
        main = QVBoxLayout(self)
        main.addWidget(self.scroll)

        zoom_box = QGroupBox(self.tr("Zoom Controls"))
        zr = QHBoxLayout(zoom_box)
        self.btn_zin  = QPushButton(self.tr("Zoom In"))
        self.btn_zout = QPushButton(self.tr("Zoom Out"))
        self.btn_fit  = QPushButton(self.tr("Fit to Preview"))
        zr.addWidget(self.btn_zin); zr.addWidget(self.btn_zout); zr.addWidget(self.btn_fit)
        main.addWidget(zoom_box)

        h = QHBoxLayout()
        h.addWidget(self.grp, 3)
        h.addWidget(self.prog_grp, 1)
        main.addLayout(h)
        main.addLayout(bot)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import dse_icon_path
                _grip_icon = QIcon(dse_icon_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "wavescale_dark_enhance", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a WaveScale Dark Enhancer shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            self.btn_help = make_help_button("wavescale_dark_enhance", self)
            drag_row.addWidget(self.btn_help)
            
            main.addLayout(drag_row)
        except Exception:
            pass

        # mask window (show immediately)
        self.mask_win = _MaskWindow(self); self.mask_win.show()

        # kernel
        self.base_kernel = _B3

        # connections
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_toggle.clicked.connect(self._toggle)

        self.btn_zin.clicked.connect(self.view.zoom_in)
        self.btn_zout.clicked.connect(self.view.zoom_out)
        self.btn_fit.clicked.connect(lambda: self.view.fit_item(self.pix))

        # gamma debounce → live mask updates (250ms)
        self._gamma_timer = QTimer(self)
        self._gamma_timer.setSingleShot(True)
        self._gamma_timer.timeout.connect(self._update_mask_only)
        self.s_gamma.valueChanged.connect(lambda _v: self._gamma_timer.start(250))

        # Amount re-blends the last preview (debounced, off the GUI thread)
        self._blend_timer = QTimer(self)
        self._blend_timer.setSingleShot(True)
        self._blend_timer.timeout.connect(self._update_amount_blend)
        self.s_amount.valueChanged.connect(lambda _v: self._blend_timer.start(120))

        # init preview & initial mask (built by a background job; the dialog
        # opens without waiting for it)
        self._update_mask_only()

    def _combine_with_doc_mask(self, op_mask: np.ndarray | None) -> np.ndarray | None:
        doc_m = self._get_doc_active_mask_2d()
        if doc_m is None:
            return op_mask
        if op_mask is None:
            return doc_m
        H, W = op_mask.shape[:2]
        if doc_m.shape != (H, W):
            yi = (np.linspace(0, doc_m.shape[0] - 1, H)).astype(np.int32)
            xi = (np.linspace(0, doc_m.shape[1] - 1, W)).astype(np.int32)
            doc_m = doc_m[yi][:, xi]
        return np.clip(op_mask * doc_m, 0.0, 1.0)

    def _get_doc_active_mask_2d(self) -> np.ndarray | None:
        """
        Return active document mask as 2-D float32 [0..1], resized to current image.
        """
        m = self._doc_mask_prenorm()
        if m is None:
            return None
        H, W = self.original.shape[:2]
        return _doc_mask_finish(m, H, W)

    def _doc_mask_prenorm(self) -> np.ndarray | None:
        """GUI-thread part of _get_doc_active_mask_2d: find the active mask,
        collapse to 2-D float32 and normalise (in place, as before). The clip and
        resize (_doc_mask_finish) are left to the caller / background jobs."""
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

        # pick first non-None payload without boolean 'or'
        data = None
        for attr in ("data", "mask", "image", "array"):
            if hasattr(layer, attr):
                val = getattr(layer, attr)
                if val is not None:
                    data = val
                    break
        if data is None and isinstance(layer, dict):
            for key in ("data", "mask", "image", "array"):
                if key in layer and layer[key] is not None:
                    data = layer[key]
                    break
        if data is None and isinstance(layer, np.ndarray):
            data = layer
        if data is None:
            return None

        m = np.asarray(data)
        if m.ndim == 3:
            m = m.mean(axis=2)

        m = m.astype(np.float32, copy=False)
        # normalize to [0,1] if needed
        mx = float(m.max()) if m.size else 1.0
        if mx > 1.0:
            # a new array: astype(copy=False) can hand back the document's
            # own float32 mask, which `m /= mx` divided in place
            m = m / mx
        return m


    def _combine_with_doc_mask(self, algo_mask: np.ndarray) -> np.ndarray:
        m_doc = self._get_doc_active_mask_2d()
        if m_doc is None:
            return algo_mask
        return np.clip(algo_mask.astype(np.float32) * m_doc.astype(np.float32), 0.0, 1.0)


    # --- preview pixmap ---
    def _set_pix(self, rgb: np.ndarray):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    def _show_pixmap(self, pm: QPixmap):
        self.pix.setPixmap(pm)
        self.view.setSceneRect(self.pix.boundingRect())

    def _show_original(self):
        if self._pm_original is not None:
            self._show_pixmap(self._pm_original)
        elif self.original is not None:
            self._set_pix(self.original)

    def _show_preview(self):
        if self.preview is self.original:
            self._show_original()
        elif self._pm_preview is not None:
            self._show_pixmap(self._pm_preview)
        elif self.preview is not None:
            self._set_pix(self.preview)

    # --- toggle ---
    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText(self.tr("Show Preview"))
            self._show_original()
        else:
            self.btn_toggle.setText(self.tr("Show Original"))
            self._show_preview()

    # --- reset ---
    def _reset(self):
        self._full = None
        self._blend_gen += 1                # a re-blend still running is dropped
        self.s_scales.setValue(6)
        self.s_boost.setValue(500)
        self.s_gamma.setValue(100)
        self.s_iters.setValue(2)
        self.s_amount.setValue(100)
        self.preview = self.original
        self._pm_preview = None
        self._show_original()
        self.lbl_step.setText(self.tr("Idle")); self.bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))
        self._update_mask_only()

    # -------- preset emit (grip) --------
    def get_preset(self) -> dict:
        """Emit current slider state as the DSE preset schema (inverse of seed_from_preset).
        Mirrors exactly what _apply_to_doc records."""
        return {
            "n_scales": int(self.s_scales.value()),
            "boost_factor": float(self.s_boost.value()) / 100.0,
            "mask_gamma": float(self.s_gamma.value()) / 100.0,
            "iterations": int(self.s_iters.value()),
            "amount": float(self.s_amount.value()) / 100.0,
        }

    # -------- preset seed (double-click open) --------
    def seed_from_preset(self, p: dict | None):
        """Inverse of get_preset. boost/gamma sliders are int×100; scales/iters direct.
        Clamps to each slider's range."""
        p = dict(p or {})
        try:
            ns = int(round(float(p.get("n_scales", 6))))
            self.s_scales.setValue(max(2, min(10, ns)))
        except Exception:
            pass
        try:
            bi = int(round(float(p.get("boost_factor", 5.0)) * 100.0))
            self.s_boost.setValue(max(10, min(1000, bi)))
        except Exception:
            pass
        try:
            gi = int(round(float(p.get("mask_gamma", 1.0)) * 100.0))
            self.s_gamma.setValue(max(10, min(1000, gi)))
        except Exception:
            pass
        try:
            it = int(round(float(p.get("iterations", 2))))
            self.s_iters.setValue(max(1, min(10, it)))
        except Exception:
            pass
        # presets made before Amount existed have no key: full effect, as they ran then
        self.s_amount.setValue(int(round(clamp_amount(p.get("amount", 1.0)) * 100)))
        # Refresh the mask window (depends on gamma AND n_scales).
        try:
            self._update_mask_only()
        except Exception:
            pass

    # --- live mask (no full recompute) ---
    def _mask_box(self) -> tuple[int, int]:
        sz = self.mask_win.size()
        return (sz.width(), sz.height())

    def _update_mask_only(self):
        """Refresh the mask window. The work runs in a background job; a request
        made while one runs is queued and runs next with the then-current sliders."""
        if getattr(self, "_memory_released", False):
            return
        if self._mask_busy:
            self._mask_pending = True
            return
        self._mask_pending = False
        job = functools.partial(
            _dse_mask_job, self.original, dict(self._mask_cache),
            int(self.s_scales.value()), self.base_kernel,
            float(self.s_gamma.value()) / 100.0,
            self._doc_mask_prenorm(), self._mask_box(),
            self._pm_original is None,
        )
        self._mask_busy = True
        self._mask_thr = _JobThread(job)
        self._mask_thr.done.connect(self._on_mask_job_done)
        start_owned_thread(self._mask_thr)

    def _on_mask_job_done(self, res, err):
        self._mask_busy = False
        if getattr(self, "_memory_released", False):
            return
        if err is not None:
            print("WaveScale DSE mask error:", err)
        if res is not None:
            for k in ("L01", "base", "base_key"):
                if k in res:
                    self._mask_cache[k] = res[k]
            if res.get("qimg") is not None and self._pm_original is None:
                self._pm_original = QPixmap.fromImage(res["qimg"])
                if self.btn_toggle.isChecked() or self.preview is self.original:
                    self._show_original()
            self.mask_win.setWindowTitle(
                self.tr("Dark Mask (Algo × Active Mask)") if res.get("has_doc_mask") else self.tr("Dark Mask")
            )
            self.mask_win.set_thumb(res["thumb"])
        if self._mask_pending:
            self._update_mask_only()
    # --- threaded preview ---
    def _start_preview(self):
        self.btn_preview.setEnabled(False); self.btn_apply.setEnabled(False)
        self.btn_reset.setEnabled(False)    # the running preview owns these settings
        n_scales = int(self.s_scales.value())
        boost    = float(self.s_boost.value()) / 100.0
        mgamma   = float(self.s_gamma.value()) / 100.0
        iters    = int(self.s_iters.value())
        # the doc mask's clip + resize, the blend and the display images all run
        # in the worker (the mask lookup + normalisation stays here)
        docmask_pre = self._doc_mask_prenorm()
        post = functools.partial(_dse_preview_payload, self.original,
                                 box_wh=self._mask_box(), want_qimage=True,
                                 amount=self._amount())
        # no parent: thread_keeper holds it, so closing the dialog mid-run is safe
        self.thread = QThread()
        self.worker = DSEWorker(self.original, n_scales, boost, mgamma,
                                self.base_kernel, iters,
                                post=post, doc_mask_pre=docmask_pre)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        start_owned_thread(self.thread)

    def _on_progress(self, step: str, pct: int):
        self.lbl_step.setText(step); self.bar.setValue(pct)

    def _on_finished(self, payload, _unused=None):
        if getattr(self, "_memory_released", False):
            return              # dialog closed while the worker ran
        self.btn_preview.setEnabled(True)
        self.btn_reset.setEnabled(True)
        if payload is None:
            QMessageBox.critical(self, self.tr("WaveScale Dark Enhancer"), self.tr("Processing failed."))
            return

        # doc-mask blend, combined mask and display images were done in the worker
        self._blend_gen += 1                # an older re-blend must not land on this preview
        self._full = payload["full"]
        self.preview = payload["preview"]
        qimg = payload.get("qimg")
        self._pm_preview = QPixmap.fromImage(qimg) if qimg is not None else None
        self._show_preview()
        if payload.get("thumb") is not None:
            self.mask_win.set_thumb(payload["thumb"])
        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))
        self.lbl_step.setText(self.tr("Preview ready")); self.bar.setValue(100)
        if self._amount() != payload.get("amount"):
            self._update_amount_blend()     # Amount moved while it computed

    # --- Amount re-blend ---
    def _amount(self) -> float:
        return float(self.s_amount.value()) / 100.0

    def _update_amount_blend(self):
        """Re-blend the last preview at the current Amount. Runs in a background
        job; a request made while one runs is queued with the then-current value."""
        if getattr(self, "_memory_released", False) or self._full is None:
            return
        if self._blend_busy:
            self._blend_pending = True
            return
        self._blend_pending = False
        self._blend_busy = True
        self._blend_job_gen = self._blend_gen    # one job at a time, so one slot
        job = functools.partial(_amount_job, self.original, self._full, self._amount(), True)
        self._blend_thr = _JobThread(job)
        self._blend_thr.done.connect(self._on_amount_blend_done)
        start_owned_thread(self._blend_thr)

    def _on_amount_blend_done(self, res, err):
        self._blend_busy = False
        if getattr(self, "_memory_released", False):
            return
        if err is not None:
            print("WaveScale DSE amount blend error:", err)
        if res is not None and self._blend_job_gen == self._blend_gen and self._full is not None:
            self.preview = res["preview"]
            self._pm_preview = QPixmap.fromImage(res["qimg"]) if res.get("qimg") is not None else None
            if not self.btn_toggle.isChecked():
                self._show_preview()
        if self._blend_pending:
            self._update_amount_blend()

    # --- apply back to doc ---
    def _apply_to_doc(self):
        # Amount is blended here from the full result, not read from the shown
        # preview, so a re-blend still in flight cannot make Apply use a stale value.
        rgb = self.preview if self._full is None else blend_amount(
            self.original, self._full, self._amount())
        # mono: channel mean (2-D or H×W×1 as the source was); clip to [0,1]
        out = _apply_output(rgb, self._was_mono, self._mono_shape)
        _meta = {"step_name": "WaveScale Dark Enhancer",
                 "command_id": "wavescale_dark_enhance",
                 "preset": dict(self.get_preset())}
        try:
            self._doc.set_image(out, metadata=_meta, step_name="WaveScale Dark Enhancer")
        except Exception as e:
            QMessageBox.critical(self, self.tr("WaveScale Dark Enhancer"), self.tr("Failed to write to document:\n{0}").format(e))
            return
        except Exception as e:
            QMessageBox.critical(self, self.tr("WaveScale Dark Enhancer"), self.tr("Failed to write to document:\n{0}").format(e))
            return

        # ── Build preset from current sliders ─────────────────────────
        try:
            preset = dict(self.get_preset())
        except Exception:
            preset = {}

        # ── Register as last_headless_command on the main window ─────
        try:
            main = self.parent()
            if main is not None:
                payload = {
                    "command_id": "wavescale_dark_enhance",
                    "preset": dict(preset),
                }
                setattr(main, "_last_headless_command", payload)

                # Optional debug logging similar to other tools
                try:
                    if hasattr(main, "_log"):
                        main._log(
                            "[Replay] Registered WaveScale Dark Enhancer as "
                            f"last action (n_scales={preset.get('n_scales')}, "
                            f"boost={preset.get('boost_factor')}, "
                            f"mask_gamma={preset.get('mask_gamma')}, "
                            f"iterations={preset.get('iterations')}, "
                            f"amount={preset.get('amount')})"
                        )
                except Exception:
                    pass
        except Exception:
            # Never let replay wiring break the apply
            pass

        # Applied: the tool's job is done, so it closes.
        self.accept()

    def closeEvent(self, ev):
        self._close_teardown()
        super().closeEvent(ev)

    def done(self, r):
        # Close / Esc end a QDialog through done(), which never delivers
        # closeEvent: without this they skipped the teardown and the dialog
        # kept its full-size buffers until collected. Runs once.
        self._close_teardown()
        super().done(r)

    def _close_teardown(self):
        if getattr(self, "_close_torn_down", False):
            return
        self._close_torn_down = True
        # Running jobs are held by thread_keeper and finish on their own; their
        # results are dropped once the dialog has released its buffers.
        try:
            thr = getattr(self, "thread", None)
            if thread_running(thr):
                thr.quit()
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass

    def _release_memory(self):
        """Drop full-frame RGB buffers and preview pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self.original = None
        self.preview = None
        self._full = None
        self._L_original = None
        self._mask_cache = {}
        self._pm_original = None
        self._pm_preview = None
        self._last_preview = None
        self._doc = None
        self.worker = None
        try:
            if getattr(self, "mask_win", None) is not None:
                self.mask_win.close()
                self.mask_win = None
        except Exception:
            pass
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


# ─────────────────────────────────────────────────────────────────────────────
# Installer helpers
# ─────────────────────────────────────────────────────────────────────────────
def install_wavescale_dark_enhancer(main_window: QMainWindow,
                                    dse_icon_path: str,
                                    *,
                                    command_id: str = "wavescale_dark_enhancer",
                                    menu_name: str = "Pro",
                                    toolbar_name: str = "Pro Tools"):
    """
    Creates the QAction, hooks it into menu+toolbar, and registers it
    with your ShortcutManager under `command_id`.
    Expects main_window to expose:
      • .docman.current_document() → returns doc with .image
      • ._spawn_subwindow_for(doc) (normal in your app)
      • .shortcut_manager (your ShortcutManager) — optional
    """
    # 1) QAction
    act = getattr(main_window, "act_wavescalede", None)
    if act is None:
        from PyQt6.QtGui import QAction
        from PyQt6.QtCore import QCoreApplication
        act = QAction(QIcon(dse_icon_path), QCoreApplication.translate("WaveScaleDarkEnhancerDialogPro", "WaveScale Dark Enhancer"), main_window)
        act.setObjectName(command_id)
        act.setProperty("command_id", command_id)

        def _run_dialog():
            docman = getattr(main_window, "docman", None)
            doc = None
            if docman and hasattr(docman, "current_document"):
                doc = docman.current_document()
            if doc is None or getattr(doc, "image", None) is None:
                from PyQt6.QtCore import QCoreApplication
                QMessageBox.warning(main_window, QCoreApplication.translate("WaveScaleDarkEnhancerDialogPro", "WaveScale Dark Enhancer"), QCoreApplication.translate("WaveScaleDarkEnhancerDialogPro", "No active image."))
                return
            dlg = WaveScaleDarkEnhancerDialogPro(main_window, doc, icon_path=dse_icon_path)
            dlg.exec()

        act.triggered.connect(_run_dialog)
        setattr(main_window, "act_wavescalede", act)

    # 2) Menu hookup
    menubar = main_window.menuBar()
    menu = None
    for m in menubar.findChildren(type(menubar)):
        # best-effort: ignore; we’ll just create/find by title
        pass
    menu = None
    for i in range(menubar.actions().__len__()):
        if menubar.actions()[i].text().replace("&", "") == menu_name:
            menu = menubar.actions()[i].menu()
            break
    if menu is None:
        menu = menubar.addMenu(menu_name)
    menu.addAction(act)

    # 3) Toolbar hookup
    tb = None
    for t in main_window.findChildren(type(main_window.addToolBar("tmp"))):
        # naive scan (we won't rely on this); we'll create if needed
        pass
    tb = getattr(main_window, "_tb_" + toolbar_name.replace(" ", "_").lower(), None)
    if tb is None:
        tb = main_window.addToolBar(toolbar_name)
        setattr(main_window, "_tb_" + toolbar_name.replace(" ", "_").lower(), tb)
    tb.addAction(act)

    # 4) Register with ShortcutManager (if present)
    sm = getattr(main_window, "shortcut_manager", None)
    if sm and hasattr(sm, "register_action"):
        sm.register_action(command_id, act)

    return act

def open_wavescalede_with_preset(main_window, preset: dict | None = None):
    from PyQt6.QtGui import QIcon  # noqa: F401 (icon_path is a str)

    # Resolve doc: active MDI subwindow first, then docman fallback.
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
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    if doc is None or getattr(doc, "image", None) is None:
        return None

    _icon_path = None
    try:
        from setiastro.saspro.resources import dse_icon_path
        _icon_path = dse_icon_path
    except Exception:
        _icon_path = None

    dlg = WaveScaleDarkEnhancerDialogPro(main_window, doc, icon_path=_icon_path)

    # No on-show reset of the sliders -> seed directly before show.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    try:
        main_window._wavescalede_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_wavescalede_dialog", None))
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg