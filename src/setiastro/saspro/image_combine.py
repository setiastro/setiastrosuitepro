#saspro/image_combine.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QRect, QEvent, QTimer, QSettings, QByteArray, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QSlider,
    QCheckBox, QScrollArea, QPushButton, QDialogButtonBox, QApplication, QMessageBox
)

# NEW: optional cv2 for fast gray/resize
try:
    import cv2
except Exception:
    cv2 = None

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.luminancerecombine import recombine_luminance_linear_scale as _recombine_luma_into_rgb
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)

_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# ---------- helpers ----------
def _doc_name(d) -> str:
    try: return d.display_name()
    except Exception: return "Untitled"

def _rgb_to_luma(img: np.ndarray) -> np.ndarray:
    f = _to_float01(img)
    if f.ndim == 2: return f
    if f.ndim == 3 and f.shape[2] == 1: return f[..., 0]
    if f.ndim == 3 and f.shape[2] == 3:
        w = _LUMA_WEIGHTS
        return f[..., 0]*w[0] + f[..., 1]*w[1] + f[..., 2]*w[2]
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _blend_dispatch(A: np.ndarray, B: np.ndarray, mode: str, alpha: float) -> np.ndarray:
    A = _to_float01(A); B = _to_float01(B)
    if A.ndim == 2: A = A[..., None]
    if B.ndim == 2: B = B[..., None]
    if A.shape != B.shape:
        raise ValueError("Images must have same size/channels.")

    if mode == "Average": return np.clip(0.5*(A+B), 0.0, 1.0)
    if mode == "Blend":   return np.clip(A*(1-alpha) + B*alpha, 0.0, 1.0)
    def mix(x): return np.clip(A*(1-alpha) + x*alpha, 0.0, 1.0)

    eps = 1e-6
    if mode == "Add":        return mix(np.clip(A+B, 0.0, 1.0))
    if mode == "Subtract":   return mix(np.clip(A-B, 0.0, 1.0))
    if mode == "Multiply":   return mix(A*B)
    if mode == "Divide":     return mix(np.clip(A/(B+eps), 0.0, 1.0))
    if mode == "Screen":     return mix(1.0 - (1.0-A)*(1.0-B))
    if mode == "Overlay":    return mix(np.clip(np.where(A<=0.5, 2*A*B, 1-2*(1-A)*(1-B)), 0.0, 1.0))
    if mode == "Difference": return mix(np.abs(A-B))
    return np.clip(A*(1-alpha) + B*alpha, 0.0, 1.0)

# ---------- mask helpers ----------
def _resize_mask_nearest(m: np.ndarray, shape_hw: tuple[int,int]) -> np.ndarray:
    """Resize mask to (H,W) with nearest neighbor."""
    h, w = shape_hw
    if m.shape == (h, w):
        return m
    if cv2 is not None:
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    # fallback NN without cv2
    yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
    xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
    return m[yi][:, xi].astype(np.float32, copy=False)

def _prep_mask01(m: np.ndarray, shape_hw: tuple[int,int], channels: int | None, invert_flag: bool):
    """Mask resized to (H,W) (nearest), inverted if asked, clipped; repeated
    to `channels` when that is > 1."""
    m = _resize_mask_nearest(m, shape_hw)
    if invert_flag:
        m = 1.0 - m
    m = np.clip(m, 0.0, 1.0)
    if channels and channels > 1 and m.ndim == 2:
        m = np.repeat(m[:, :, None], channels, axis=2)
    return m


# ---------- off-GUI-thread work ----------
# Every control change restarts this timer; the preview is built once the
# controls have been still that long, on a worker, one at a time.
_PREVIEW_DEBOUNCE_MS = 120


class _CombineJob(QThread):
    """Runs fn(**kwargs) off the GUI thread; its dict comes back through
    `ready` ({"error": text} if it raised). No parent: started with
    thread_keeper.start_owned_thread, so the dialog may close mid-run."""
    ready = pyqtSignal(object)

    def __init__(self, fn, kwargs):
        super().__init__(None)
        self._fn = fn
        self._kwargs = kwargs

    def run(self):
        try:
            out = self._fn(**self._kwargs)
        except Exception as e:
            out = {"error": str(e)}
        self._kwargs = None
        self.ready.emit(out)


def _proxy01(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """_to_float01(img) at (w, h): INTER_AREA when shrinking, the image itself
    at full size. Preview only."""
    f = _to_float01(img)
    if f.shape[1] == w and f.shape[0] == h:
        return f
    if cv2 is not None:
        out = cv2.resize(np.ascontiguousarray(f), (w, h), interpolation=cv2.INTER_AREA)
        if f.ndim == 3 and out.ndim == 2:          # cv2 drops a single channel
            out = out[..., None]
        return out.astype(np.float32, copy=False)
    yi = np.linspace(0, f.shape[0] - 1, h).astype(np.int32)
    xi = np.linspace(0, f.shape[1] - 1, w).astype(np.int32)
    return f[yi][:, xi]


def _preview_mask(m, w: int, h: int, channels: int | None, invert_flag: bool):
    """Display-scale counterpart of _prep_mask01 (area-averaged, broadcast)."""
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(np.ascontiguousarray(m, dtype=np.float32), (w, h), interpolation=cv2.INTER_AREA)
        else:
            m = _resize_mask_nearest(m, (h, w))
    if invert_flag:
        m = 1.0 - m
    m = np.clip(m, 0.0, 1.0)
    if channels and channels > 1 and m.ndim == 2:
        m = m[:, :, None]
    return m


# Modes that hinge on detail finer than the display (the difference of two
# averages is not the average of the differences): their preview is blended
# at full size and then reduced. Measured on a 22 MP pair at fit zoom: on
# display-size proxies Subtract / Divide / Difference were off by a mean of
# 3 / 13 / 7 grey levels; every other mode by 0.5, i.e. rounding.
_FULL_SIZE_PREVIEW_MODES = ("Subtract", "Divide", "Difference")


def _preview_work_size(mode: str, W: int, H: int, w: int, h: int) -> tuple[int, int]:
    """Size the preview is blended at: full size for _FULL_SIZE_PREVIEW_MODES,
    else the displayed size (w, h)."""
    return (W, H) if mode in _FULL_SIZE_PREVIEW_MODES else (w, h)


def _combine_preview_job(imgA, imgB, pa, pb, w, h, zoom, mode, alpha, luma,
                         mask, invert, overlay, ov_opacity):
    """The preview the dialog shows: the same steps as Apply, on float01
    proxies of A and B at the displayed size (w, h), i.e. full size x zoom
    (full size when zoom >= 1, then scaled up as before); for
    _FULL_SIZE_PREVIEW_MODES at full size, reduced to (w, h) at the end.
    Returns the proxies (for reuse) and the display QImage."""
    cw, ch = _preview_work_size(mode, imgA.shape[1], imgA.shape[0], w, h)
    if pa is None:
        pa = _proxy01(imgA, cw, ch)
    if pb is None:
        pb = _proxy01(imgB, cw, ch)

    if luma:
        YA = _rgb_to_luma(pa)
        YB = _rgb_to_luma(pb)
        Ymix = _blend_dispatch(YA[...,None], YB[...,None], mode, alpha)[...,0]
        m = None if mask is None else _preview_mask(mask, cw, ch, None, invert)
        if m is not None:
            Ymix = Ymix*m + YA*(1.0 - m)
        blended = _recombine_luma_into_rgb(pa, Ymix)
    else:
        A3 = pa if pa.ndim == 3 else pa[..., None]
        B3 = pb if pb.ndim == 3 else pb[..., None]
        blended = _blend_dispatch(A3, B3, mode, alpha)
        if imgA.ndim == 2:
            blended = blended[...,0]
        m = None if mask is None else _preview_mask(
            mask, cw, ch, (blended.shape[2] if blended.ndim == 3 else 1), invert)
        if m is not None:
            blended = np.clip(blended*m + _to_float01(pa)*(1.0 - m), 0.0, 1.0)

    if overlay and mask is not None:
        m = _preview_mask(mask, cw, ch, (blended.shape[2] if blended.ndim == 3 else 1), invert)
        blended = ImageCombineDialog._apply_overlay(_to_float01(blended), m, ov_opacity)

    f = _to_float01(blended)
    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]
    if (cw, ch) != (w, h):
        f = _proxy01(f, w, h)
    buf = np.ascontiguousarray((f*255).astype(np.uint8))
    if buf.ndim == 2:
        q = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        q = QImage(buf.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
    if zoom > 1.0:
        q = q.scaled(max(1, round(w*zoom)), max(1, round(h*zoom)), Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
    return {"imgA": imgA, "imgB": imgB, "pa": pa, "pb": pb, "pw": cw, "ph": ch,
            "zoom": zoom, "qimg": q}


def _combine_full_job(imgA, imgB, mode, alpha, luma, mask, invert):
    """Apply's full-resolution result. Unchanged math, moved off the GUI thread."""
    if luma:
        YA = _rgb_to_luma(imgA); YB = _rgb_to_luma(imgB)
        Ymix = _blend_dispatch(YA[...,None], YB[...,None], mode, alpha)[...,0]

        m = None if mask is None else _prep_mask01(mask, shape_hw=Ymix.shape[:2], channels=None,
                                                   invert_flag=invert)
        if m is not None:
            Ymix = Ymix*m + YA*(1.0 - m)

        result = _recombine_luma_into_rgb(imgA, Ymix)
        step = f"Luminance {mode}"
    else:
        A3 = imgA if imgA.ndim == 3 else imgA[..., None]
        B3 = imgB if imgB.ndim == 3 else imgB[..., None]
        result = _blend_dispatch(A3, B3, mode, alpha)
        if imgA.ndim == 2: result = result[...,0]

        m = None if mask is None else _prep_mask01(
            mask, shape_hw=result.shape[:2],
            channels=(result.shape[2] if result.ndim == 3 else 1),
            invert_flag=invert
        )
        if m is not None:
            result = np.clip(result*m + _to_float01(imgA)*(1.0 - m), 0.0, 1.0)
        step = f"{mode} Combine"

    result = _to_float01(result)
    return {"result": result, "step": step}


# ---------- dialog ----------
class ImageCombineDialog(QDialog):
    """
    Views-based Image Combine with realtime preview, zoom/pan, luma-only, and mask overlay.
    Output: replace A or create new view.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setWindowTitle(self.tr("Image Combine"))
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
        self.mw = main_window
        self.dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
        self.zoom = 1.0
        self._pan_origin = None
        self._hstart = 0; self._vstart = 0
        self._pix = None  # last preview QPixmap, at display size (_pix_zoom)
        self._pix_zoom = None
        self._full_hw = None             # A's full (H, W): zoom and fit work in it
        self._pv_timer = QTimer(self)    # debounce: restarted on every change
        self._pv_timer.setSingleShot(True)
        self._pv_timer.timeout.connect(self._launch_preview)
        self._pv_thread = None
        self._pv_pending = False         # a change arrived while a preview ran
        self._proxies = {}               # (id(img), w, h) -> (img, float01 proxy)
        self._apply_thread = None
        self._close_after_apply = None   # close requested while Apply ran
        # --- persistence ---
        self._settings = QSettings()
        self._persist_prefix = "image_combine"
        self._geom_restored = False
        self._restoring_ui = True  # guard saves while building/restoring
        # --- UI ---
        root = QVBoxLayout(self)

        frm = QFormLayout()
        self.cbA = QComboBox(); self.cbB = QComboBox()
        frm.addRow(self.tr("Source A:"), self.cbA)
        frm.addRow(self.tr("Source B:"), self.cbB)

        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Mode:")))
        self.cbMode = QComboBox()
        self.cbMode.addItems(["Average","Add","Subtract","Blend","Multiply","Divide","Screen","Overlay","Difference"])
        row.addWidget(self.cbMode, 1)
        row.addWidget(QLabel(self.tr("Opacity:")))
        self.slAlpha = QSlider(Qt.Orientation.Horizontal); self.slAlpha.setRange(0,100); self.slAlpha.setValue(100)
        row.addWidget(self.slAlpha, 2)
        frm.addRow(row)

        # luma-only
        self.chkLuma = QCheckBox(self.tr("Combine luminance only (keep A’s color)"))
        frm.addRow(self.chkLuma)

        # mask overlay
        mrow = QHBoxLayout()
        self.chkOverlay = QCheckBox(self.tr("Show mask overlay"))
        self.chkInvert  = QCheckBox(self.tr("Invert mask"))
        mrow.addWidget(self.chkOverlay)
        mrow.addWidget(self.chkInvert)
        mrow.addWidget(QLabel(self.tr("Overlay opacity:")))
        self.slOverlay = QSlider(Qt.Orientation.Horizontal); self.slOverlay.setRange(5,95); self.slOverlay.setValue(40)
        mrow.addWidget(self.slOverlay, 1)
        frm.addRow(mrow)
        root.addLayout(frm)

        # preview
        self.scroll = QScrollArea(self); self.scroll.setWidgetResizable(True)
        self.lbl = QLabel(""); self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.lbl)
        root.addWidget(self.scroll, 1)

        # zoom (themed)
        zrow = QHBoxLayout()

        btnOut = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        btnFit = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))
        btnIn  = themed_toolbtn("zoom-in", self.tr("Zoom In"))

        btnOut.clicked.connect(self._zoom_out)
        btnIn .clicked.connect(self._zoom_in)
        btnFit.clicked.connect(self._fit)

        zrow.addWidget(btnOut)
        zrow.addWidget(btnFit)
        zrow.addWidget(btnIn)


        root.addLayout(zrow)

        # buttons
        btns = QDialogButtonBox()
        self.btnApply = btns.addButton(self.tr("Apply"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.btnReset = make_reset_button(self)
        btns.addButton(self.btnReset, QDialogButtonBox.ButtonRole.ActionRole)
        self.btnReset.clicked.connect(self._reset_to_factory)
        self.btnClose = btns.addButton(self.tr("Close"), QDialogButtonBox.ButtonRole.RejectRole)
        self.btnClose.clicked.connect(self.reject)
        self.btnApply.clicked.connect(self._commit)
        root.addWidget(btns)
        # Grip + help: their own row at the bottom
        from setiastro.saspro.shortcuts import ShortcutGrip
        self.btn_help = make_help_button("image_combine", self)
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addWidget(ShortcutGrip("image_combine", self))
        grip_row.addStretch(1)
        grip_row.addWidget(self.btn_help)
        root.addLayout(grip_row)

        # hooks
        for w in (self.cbA, self.cbB, self.cbMode):
            w.currentIndexChanged.connect(self._update_preview)
        self.slAlpha.valueChanged.connect(self._update_preview)
        self.chkLuma.toggled.connect(self._update_preview)
        self.chkOverlay.toggled.connect(self._update_preview)
        self.chkInvert.toggled.connect(self._update_preview)
        self.slOverlay.valueChanged.connect(self._update_preview)
        self.scroll.viewport().installEventFilter(self)
        def _save():
            self._save_ui_state()

        for w in (self.cbA, self.cbB, self.cbMode):
            w.currentIndexChanged.connect(lambda *_: _save())
        self.slAlpha.sliderReleased.connect(_save)  # nicer than valueChanged spam
        self.chkLuma.toggled.connect(lambda *_: _save())
        self.chkOverlay.toggled.connect(lambda *_: _save())
        self.chkInvert.toggled.connect(lambda *_: _save())
        self.slOverlay.sliderReleased.connect(_save)
        self._populate_docs()
        # Factory defaults for Reset, before showEvent restores the saved
        # state. Sources A/B and the mask overlay (view only) stay.
        self._factory = snapshot_controls(self, exclude=(
            self.cbA, self.cbB, self.chkOverlay, self.slOverlay))
        self._update_preview()

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"

    def _reset_to_factory(self):
        if thread_running(self._apply_thread):
            return
        restore_controls(self._factory)
        self._save_ui_state()       # the opacity slider saves only on release

    def _save_ui_state(self):
        # Don't save while we are initializing/restoring or before first show
        if getattr(self, "_restoring_ui", False) or not getattr(self, "_geom_restored", False):
            return
        try:
            s = self._settings
            s.setValue(self._k("window_geometry"), self.saveGeometry())

            # store selections as labels (stable across sessions)
            s.setValue(self._k("A_label"), str(self.cbA.currentText()))
            s.setValue(self._k("B_label"), str(self.cbB.currentText()))

            s.setValue(self._k("mode"), str(self.cbMode.currentText()))
            s.setValue(self._k("alpha"), int(self.slAlpha.value()))

            s.setValue(self._k("luma"), bool(self.chkLuma.isChecked()))
            s.setValue(self._k("overlay"), bool(self.chkOverlay.isChecked()))
            s.setValue(self._k("invert"), bool(self.chkInvert.isChecked()))
            s.setValue(self._k("overlay_opacity"), int(self.slOverlay.value()))

            # optional: keep zoom
            s.setValue(self._k("zoom"), float(getattr(self, "zoom", 1.0)))

            try:
                s.sync()
            except Exception:
                pass
        except Exception:
            pass


    def _restore_ui_state(self):
        self._restoring_ui = True
        try:
            s = self._settings

            g = s.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)

            # restore non-combo widgets first
            mode = str(s.value(self._k("mode"), "Average"))
            alpha = int(s.value(self._k("alpha"), 100))
            luma = bool(s.value(self._k("luma"), False, type=bool))
            overlay = bool(s.value(self._k("overlay"), False, type=bool))
            invert = bool(s.value(self._k("invert"), False, type=bool))
            ovop = int(s.value(self._k("overlay_opacity"), 40))
            z = float(s.value(self._k("zoom"), 1.0))

            # block signals to avoid calling _update_preview mid-restore
            self.cbMode.blockSignals(True)
            self.slAlpha.blockSignals(True)
            self.chkLuma.blockSignals(True)
            self.chkOverlay.blockSignals(True)
            self.chkInvert.blockSignals(True)
            self.slOverlay.blockSignals(True)
            try:
                i = self.cbMode.findText(mode)
                if i >= 0:
                    self.cbMode.setCurrentIndex(i)
                self.slAlpha.setValue(max(0, min(100, alpha)))

                self.chkLuma.setChecked(bool(luma))
                self.chkOverlay.setChecked(bool(overlay))
                self.chkInvert.setChecked(bool(invert))
                self.slOverlay.setValue(max(5, min(95, ovop)))

                self.zoom = max(0.05, min(16.0, float(z)))
            finally:
                self.cbMode.blockSignals(False)
                self.slAlpha.blockSignals(False)
                self.chkLuma.blockSignals(False)
                self.chkOverlay.blockSignals(False)
                self.chkInvert.blockSignals(False)
                self.slOverlay.blockSignals(False)

            # restore A/B by label AFTER combos exist
            a_lab = str(s.value(self._k("A_label"), ""))
            b_lab = str(s.value(self._k("B_label"), ""))

            self.cbA.blockSignals(True)
            self.cbB.blockSignals(True)
            try:
                ia = self.cbA.findText(a_lab) if a_lab else -1
                ib = self.cbB.findText(b_lab) if b_lab else -1
                if ia >= 0:
                    self.cbA.setCurrentIndex(ia)
                if ib >= 0:
                    self.cbB.setCurrentIndex(ib)
            finally:
                self.cbA.blockSignals(False)
                self.cbB.blockSignals(False)

        finally:
            self._restoring_ui = False

    # ---------- doc utilities ----------
    def _open_docs(self) -> list:
        if not self.dm: return []
        docs = list(getattr(self.dm, "_docs", []) or [])
        return [d for d in docs if getattr(d, "image", None) is not None]

    def _active_doc(self):
        if self.dm and hasattr(self.dm, "get_active_document"):
            return self.dm.get_active_document()
        return None

    def _populate_docs(self):
        docs = self._open_docs()
        self.cbA.blockSignals(True); self.cbB.blockSignals(True)
        try:
            self.cbA.clear(); self.cbB.clear()
            for d in docs:
                self.cbA.addItem(_doc_name(d), userData=d)
                self.cbB.addItem(_doc_name(d), userData=d)
        finally:
            self.cbA.blockSignals(False); self.cbB.blockSignals(False)

        if docs:
            act = self._active_doc()
            if act in docs:
                self.cbA.setCurrentIndex(docs.index(act))
                j = 0 if len(docs) < 2 else (1 if docs[0] is act else 0)
                self.cbB.setCurrentIndex(j)

    # ---------- mask helpers ----------
    def _mask01_for_doc(self, doc, *, shape_hw: tuple[int,int], channels: int | None, invert_flag: bool):
        """
        Return mask for the given doc resized to (H,W).
        If channels is 3 and mask is 2D, expand with np.repeat.
        """
        m = self._mask_raw(doc)
        if m is None:
            return None
        return _prep_mask01(m, shape_hw, channels, invert_flag)

    def _mask_raw(self, doc):
        """The doc's active mask (H,W) in [0,1], else the global applied mask, else None."""
        m = _active_mask_array_from_doc(doc)
        if m is None:
            # last-resort fallback to global mask manager (in case user applied a global mask)
            mm = getattr(getattr(self.mw, "image_manager", None), "mask_manager", None)
            if mm and hasattr(mm, "get_applied_mask"):
                try:
                    mg = mm.get_applied_mask()
                    if mg is not None:
                        mg = np.asarray(mg).astype(np.float32)
                        if mg.ndim == 3:
                            mg = mg.mean(axis=2)
                        if mg.max() > 1.0:
                            mg /= 255.0
                        m = np.clip(mg, 0.0, 1.0)
                except Exception:
                    m = None
        return m

    @staticmethod
    def _apply_overlay(img, mask, opacity):
        # show protected region (A) as red wash: vis = 1 - m
        vis = 1.0 - np.clip(mask, 0.0, 1.0)
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        else:
            rgb = img
        overlay = np.zeros_like(rgb, dtype=np.float32); overlay[..., 0] = 1.0
        if vis.ndim == 2: vis = vis[..., None]
        return np.clip(rgb*(1.0 - vis*opacity) + overlay*(vis*opacity), 0.0, 1.0)

    # ---------- preview ----------
    def _update_preview(self, *_):
        """Debounced: the preview is rebuilt once the controls have been
        still for _PREVIEW_DEBOUNCE_MS (_launch_preview)."""
        self._pv_timer.start(_PREVIEW_DEBOUNCE_MS)

    def _proxy_for(self, img, w, h):
        hit = self._proxies.get((id(img), w, h))
        return hit[1] if hit is not None and hit[0] is img else None

    def _launch_preview(self):
        """Build the preview on a worker, one at a time, at the displayed
        size; a change made while one runs re-runs it when it lands."""
        self._pv_timer.stop()
        if thread_running(self._pv_thread):
            self._pv_pending = True
            return
        A = self.cbA.currentData(); B = self.cbB.currentData()
        if not (A and B): return
        imgA = getattr(A, "image", None); imgB = getattr(B, "image", None)
        if imgA is None or imgB is None: return
        if imgA.shape[:2] != imgB.shape[:2]:
            self.lbl.setText(self.tr("Images must be the same size."))
            return
        luma = self.chkLuma.isChecked()
        if luma and (imgA.ndim != 3 or imgA.shape[2] != 3):
            self.lbl.setText(self.tr("Luminance mode requires RGB A.")); return

        H, W = imgA.shape[:2]
        self._full_hw = (H, W)
        zoom = float(self.zoom)
        s = min(1.0, zoom)
        w, h = max(1, round(W*s)), max(1, round(H*s))
        mode = self.cbMode.currentText()
        cw, ch = _preview_work_size(mode, W, H, w, h)
        thr = _CombineJob(_combine_preview_job, dict(
            imgA=imgA, imgB=imgB, pa=self._proxy_for(imgA, cw, ch), pb=self._proxy_for(imgB, cw, ch),
            w=w, h=h, zoom=zoom, mode=mode, alpha=self.slAlpha.value()/100.0,
            luma=luma, mask=self._mask_raw(A), invert=self.chkInvert.isChecked(),
            overlay=self.chkOverlay.isChecked(), ov_opacity=self.slOverlay.value()/100.0))
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if "error" in res:
            self.lbl.setText(self.tr("Error: {0}").format(res['error']))
            return
        w, h = res["pw"], res["ph"]
        self._proxies = {(id(res["imgA"]), w, h): (res["imgA"], res["pa"]),
                         (id(res["imgB"]), w, h): (res["imgB"], res["pb"])}
        self._pix = QPixmap.fromImage(res["qimg"])
        self._pix_zoom = res["zoom"]
        self._apply_zoom()

    def _on_preview_finished(self):
        if self._pv_pending:
            self._pv_pending = False
            self._launch_preview()

    # ---------- apply ----------
    def _commit(self):
        if thread_running(self._apply_thread):
            return
        A = self.cbA.currentData(); B = self.cbB.currentData()
        if not (A and B): return
        imgA = getattr(A, "image", None); imgB = getattr(B, "image", None)
        if imgA is None or imgB is None: return
        if imgA.shape[:2] != imgB.shape[:2]:
            QMessageBox.warning(self, self.tr("Image Combine"), self.tr("Image sizes must match.")); return

        thr = _CombineJob(_combine_full_job, dict(
            imgA=imgA, imgB=imgB, mode=self.cbMode.currentText(),
            alpha=self.slAlpha.value()/100.0, luma=self.chkLuma.isChecked(),
            mask=self._mask_raw(A), invert=self.chkInvert.isChecked()))
        self._apply_target = A
        thr.ready.connect(self._on_commit_ready)
        self._apply_thread = thr
        self.btnApply.setEnabled(False)
        start_owned_thread(thr)

    def _on_commit_ready(self, res: dict):
        self._apply_thread = None
        A, self._apply_target = self._apply_target, None
        try:
            self.btnApply.setEnabled(True)
            self._write_commit(A, res)
        finally:
            code, self._close_after_apply = self._close_after_apply, None
            if code is not None:
                try:
                    self._save_ui_state()
                except Exception:
                    pass
                self.done(code)

    def done(self, r):
        # Close/Esc while Apply runs: hide now, close once its result is written.
        if thread_running(self._apply_thread):
            self._close_after_apply = r
            self.hide()
            return
        super().done(r)

    def _write_commit(self, A, res: dict):
        try:
            if "error" in res:
                raise RuntimeError(res["error"])
            result, step = res["result"], res["step"]

            # Replace A (overwrite active view) or create new?
            replace = True
            if replace:
                if hasattr(A, "set_image"):
                    A.set_image(result, step_name=f"Image Combine: {step}")
                else:
                    A.image = result
                try: self.mw._log(f"Image Combine → replaced '{_doc_name(A)}' ({step})")
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            else:
                newdoc = self.dm.create_document(result, metadata={
                    "display_name": f"Combined ({step})",
                    "bit_depth": "32-bit floating point",
                    "is_mono": (result.ndim == 2),
                    "source": f"Combine: {step}",
                }, name=f"Combined ({step})")
                self.mw._spawn_subwindow_for(newdoc)
                try: self.mw._log(f"Image Combine → new view '{_doc_name(newdoc)}' ({step})")
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        except Exception as e:
            QMessageBox.critical(self, self.tr("Image Combine"), self.tr("Failed:\n{0}").format(e))
            
    # ---------- zoom/pan ----------
    def _apply_zoom(self):
        if self._pix is None: return
        if self._pix_zoom == self.zoom:
            self.lbl.setPixmap(self._pix)
            return
        # Rescale what is on screen at once, and rebuild it at this zoom.
        H, W = self._full_hw
        self.lbl.setPixmap(self._pix.scaled(max(1, round(W*self.zoom)), max(1, round(H*self.zoom)),
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.FastTransformation))
        self._update_preview()

    def _zoom_in(self):  self.zoom *= 1.25; self._apply_zoom()
    def _zoom_out(self): self.zoom /= 1.25; self._apply_zoom()
    def _fit(self):
        if self._pix is None: return
        area = self.scroll.viewport().size(); H, W = self._full_hw
        sx = area.width()/max(1,W); sy = area.height()/max(1,H)
        self.zoom = min(sx, sy, 1.0); self._apply_zoom()

    def eventFilter(self, src, ev):
        if src is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._pan_origin = ev.pos()
                self._hstart = self.scroll.horizontalScrollBar().value()
                self._vstart = self.scroll.verticalScrollBar().value()
                return True
            if ev.type() == QEvent.Type.MouseMove and self._pan_origin is not None:
                d = ev.pos() - self._pan_origin
                self.scroll.horizontalScrollBar().setValue(self._hstart - d.x())
                self.scroll.verticalScrollBar().setValue(self._vstart - d.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._pan_origin = None; return True
            return False
        return super().eventFilter(src, ev)

    def showEvent(self, ev):
        super().showEvent(ev)

        # repopulate each time in case views changed
        self._populate_docs()

        if not self._geom_restored:
            self._geom_restored = True

            def _after():
                self._restore_ui_state()
                self._update_preview()
                self._apply_zoom()
            QTimer.singleShot(0, _after)


    def closeEvent(self, ev):
        if thread_running(self._apply_thread):
            # Apply still computing: close once its result is written.
            self._close_after_apply = QDialog.DialogCode.Rejected
            self.hide()
            ev.ignore()
            return
        self._pv_timer.stop()
        try:
            self._save_ui_state()
        except Exception:
            pass
        super().closeEvent(ev)