# pro/stat_stretch.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QSlider, QToolBar, QToolButton, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QCursor
import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QProgressDialog, QApplication
from .doc_manager import ImageDocument
# use your existing stretch code
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.luminancerecombine import LUMA_PROFILES

class _StretchWorker(QObject):
    finished = pyqtSignal(object, str)  # (out_array_or_None, error_message_or_empty)

    def __init__(self, dialog_ref):
        super().__init__()
        self._dlg = dialog_ref

    @pyqtSlot()
    def run(self):
        try:
            # dialog might be closing; guard
            if self._dlg is None or sip.isdeleted(self._dlg):
                self.finished.emit(None, "Dialog was closed.")
                return

            out = self._dlg._run_stretch()
            self.finished.emit(out, "")
        except Exception as e:
            self.finished.emit(None, str(e))


class StatisticalStretchDialog(QDialog):
    """
    Non-destructive preview; Apply commits to the document image.
    """
    def __init__(self, parent, document: ImageDocument):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Statistical Stretch"))

        # --- IMPORTANT: avoid “attached modal” behavior on some Linux WMs ---
        # Make this a proper top-level window (tool-style) rather than an attached sheet.
        self.setWindowFlag(Qt.WindowType.Window, True)
        # Non-modal: allow user to switch between images while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        # Don’t let the generic modal flag override the explicit modality
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
        self._main = parent
        self.doc = document
        self._last_preview = None
        self._hdr_knee_user_locked = False

        self._follow_conn = None
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                # store connection so we can cleanly disconnect
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = None
        self._panning = False
        self._pan_last = None  # QPoint
        self._preview_scale = 1.0       # NEW: zoom factor for preview
        self._preview_qimg = None       # NEW: store unscaled QImage for clean scaling
        self._suppress_replay_record = False

        # --- Controls ---
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0.01, 0.99)
        self.spin_target.setSingleStep(0.01)
        self.spin_target.setValue(0.25)
        self.spin_target.setDecimals(3)

        self.chk_linked = QCheckBox(self.tr("Linked channels"))
        self.chk_linked.setChecked(False)

        self.chk_normalize = QCheckBox(self.tr("Normalize to [0..1]"))
        self.chk_normalize.setChecked(False)

        # --- Black point sigma ---
        self.row_bp = QWidget()
        bp_lay = QHBoxLayout(self.row_bp); bp_lay.setContentsMargins(0,0,0,0); bp_lay.setSpacing(8)
        bp_lay.addWidget(QLabel(self.tr("Black point σ:")))
        self.sld_bp = QSlider(Qt.Orientation.Horizontal)
        self.sld_bp.setRange(50, 600)      # 0.50 .. 6.00
        self.sld_bp.setValue(500)          # default 2.70
        self.lbl_bp = QLabel("5.00")
        self.sld_bp.valueChanged.connect(lambda v: self.lbl_bp.setText(f"{v/100:.2f}"))
        bp_lay.addWidget(self.sld_bp, 1)
        bp_lay.addWidget(self.lbl_bp)
        bp_tip = self.tr(
            "Black point (σ) controls how aggressively the dark background is clipped.\n"
            "Higher values clip more (darker background, more contrast), but can crush faint dust.\n"
            "Lower values preserve faint background, but may leave the image gray.\n"
            "Tip: start around 2.7–5.0 depending on gradient/noise."
        )

        # Apply tooltip to everything in the row
        self.row_bp.setToolTip(bp_tip)
        self.sld_bp.setToolTip(bp_tip)
        self.lbl_bp.setToolTip(bp_tip)
        
        self.chk_no_black_clip = QCheckBox(self.tr("No black clipping"))
        self.chk_no_black_clip.setChecked(False)
        self.chk_no_black_clip.setToolTip(self.tr(
            "Disables black-point clipping.\n"
            "Uses the image minimum as the black point (preserves faint background),\n"
            "but the result may look flatter / hazier."
        ))
        # --- HDR compress ---
        self.chk_hdr = QCheckBox(self.tr("HDR highlight compress"))
        self.chk_hdr.setChecked(False)

        self.hdr_row = QWidget()
        hdr_lay = QVBoxLayout(self.hdr_row); hdr_lay.setContentsMargins(0,0,0,0); hdr_lay.setSpacing(6)

        # amount
        row_a = QHBoxLayout(); row_a.setContentsMargins(0,0,0,0); row_a.setSpacing(8)
        row_a.addWidget(QLabel(self.tr("Amount:")))
        self.sld_hdr_amt = QSlider(Qt.Orientation.Horizontal)
        self.sld_hdr_amt.setRange(0, 100)     # 0..1
        self.sld_hdr_amt.setValue(15)         # 0.15 default
        self.lbl_hdr_amt = QLabel("0.15")
        self.sld_hdr_amt.valueChanged.connect(lambda v: self.lbl_hdr_amt.setText(f"{v/100:.2f}"))
        row_a.addWidget(self.sld_hdr_amt, 1)
        row_a.addWidget(self.lbl_hdr_amt)

        # knee
        row_k = QHBoxLayout(); row_k.setContentsMargins(0,0,0,0); row_k.setSpacing(8)
        row_k.addWidget(QLabel(self.tr("Knee:")))
        self.sld_hdr_knee = QSlider(Qt.Orientation.Horizontal)
        self.sld_hdr_knee.setRange(10, 95)    # 0.10..0.95
        self.sld_hdr_knee.setValue(75)        # 0.75 default
        self.lbl_hdr_knee = QLabel("0.75")
        self.sld_hdr_knee.valueChanged.connect(lambda v: self.lbl_hdr_knee.setText(f"{v/100:.2f}"))
        row_k.addWidget(self.sld_hdr_knee, 1)
        row_k.addWidget(self.lbl_hdr_knee)

        hdr_lay.addLayout(row_a)
        hdr_lay.addLayout(row_k)
        self.sld_hdr_knee.sliderPressed.connect(lambda: setattr(self, "_hdr_knee_user_locked", True))
        self.hdr_row.setEnabled(False)
        self.chk_hdr.toggled.connect(self.hdr_row.setEnabled)
        def _suggest_hdr_knee_from_target():
            # Only auto-update if the user hasn't manually adjusted the knee yet
            if getattr(self, "_hdr_knee_user_locked", False):
                return

            t = float(self.spin_target.value())
            knee = float(np.clip(t + 0.10, 0.10, 0.95))  # or +0.15 if you prefer

            self.sld_hdr_knee.blockSignals(True)
            self.sld_hdr_knee.setValue(int(round(knee * 100)))
            self.sld_hdr_knee.blockSignals(False)
            self.lbl_hdr_knee.setText(f"{knee:.2f}")
        self.spin_target.valueChanged.connect(_suggest_hdr_knee_from_target)    
        # HDR tooltips
        self.chk_hdr.setToolTip(self.tr(
            "Compresses bright highlights after the stretch.\n"
            "Use lightly: high values can flatten nebula structure and create star ringing."
        ))

        self.sld_hdr_amt.setToolTip(self.tr(
            "Compression strength (0–1).\n"
            "Start low (0.10–0.15). Too much can flatten the image and ring stars."
        ))

        self.sld_hdr_knee.setToolTip(self.tr(
            "Where compression begins (0–1).\n"
            "Good starting point: knee ≈ target median + 0.10 to + 0.20.\n"
            "Example: target 0.25 → knee 0.35–0.45."
        ))
        self.lbl_hdr_amt.setToolTip(self.sld_hdr_amt.toolTip())
        self.lbl_hdr_knee.setToolTip(self.sld_hdr_knee.toolTip())

        # --- Luma-only row (checkbox + dropdown on one line) ---
        self.luma_row = QWidget()
        lr = QHBoxLayout(self.luma_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.setSpacing(8)

        self.chk_luma_only = QCheckBox(self.tr("Luminance-only"))
        self.chk_luma_only.setChecked(False)

        self.cmb_luma = QComboBox()
        keys = list(LUMA_PROFILES.keys())

        def _cat(k):
            return str(LUMA_PROFILES.get(k, {}).get("category", ""))

        keys.sort(key=lambda k: (_cat(k), k.lower()))
        self.cmb_luma.addItems(keys)
        self.cmb_luma.setCurrentText("rec709")

        lr.addWidget(self.chk_luma_only)
        lr.addWidget(QLabel(self.tr("Mode:")))
        lr.addWidget(self.cmb_luma, 1)

        # Start disabled until checkbox is enabled
        self.cmb_luma.setEnabled(False)
        self.chk_luma_only.toggled.connect(self.cmb_luma.setEnabled)
        # NEW: Curves boost
        self.chk_curves = QCheckBox(self.tr("Curves boost"))
        self.chk_curves.setChecked(False)

        self.curves_row = QWidget()
        cr_lay = QHBoxLayout(self.curves_row); cr_lay.setContentsMargins(0,0,0,0)
        cr_lay.setSpacing(8)
        cr_lay.addWidget(QLabel(self.tr("Strength:")))
        self.sld_curves = QSlider(Qt.Orientation.Horizontal)
        self.sld_curves.setRange(0, 100)           # 0.00 … 1.00 mapped to 0…100
        self.sld_curves.setSingleStep(1)
        self.sld_curves.setPageStep(5)
        self.sld_curves.setValue(20)               # default 0.20
        self.lbl_curves_val = QLabel("0.20")
        self.sld_curves.valueChanged.connect(lambda v: self.lbl_curves_val.setText(f"{v/100:.2f}"))
        cr_lay.addWidget(self.sld_curves, 1)
        cr_lay.addWidget(self.lbl_curves_val)
        self.curves_row.setEnabled(False)          # disabled until checkbox is ticked
        self.chk_curves.toggled.connect(self.curves_row.setEnabled)

        # Preview area
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(320, 240))
        self.preview_label.setScaledContents(False) 
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)           # <- was True; we manage size
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._fit_mode = True       # NEW: start in Fit mode

        # --- Zoom buttons row (place before the main layout or right above preview) ---
        # --- Zoom buttons row ---
        zoom_row = QHBoxLayout()

        # Use themed tool buttons (consistent with the rest of SASpro)
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")


        zoom_row.addStretch(1)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)

        # Buttons
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply   = QPushButton(self.tr("Apply"))
        self.btn_close   = QPushButton(self.tr("Close"))

        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_close.clicked.connect(self.close)

        # --- Layout ---
        form = QFormLayout()
        form.addRow(self.tr("Target median:"), self.spin_target)
        form.addRow("", self.chk_linked)
        form.addRow("", self.row_bp)
        form.addRow("", self.chk_no_black_clip)
        form.addRow("", self.chk_hdr)
        form.addRow("", self.hdr_row)
        form.addRow("", self.luma_row)      
        form.addRow("", self.chk_normalize)
        form.addRow("", self.chk_curves)
        form.addRow("", self.curves_row)

        left = QVBoxLayout()
        left.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_apply)
        row.addStretch(1)
        left.addLayout(row)
        left.addStretch(1)

        main = QHBoxLayout(self)
        main.addLayout(left, 0)

        # NEW: right column with zoom row + preview
        right = QVBoxLayout()
        right.addLayout(zoom_row)                # ← actually add the zoom controls
        right.addWidget(self.preview_scroll, 1)  # preview below the buttons
        main.addLayout(right, 1)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_100.clicked.connect(self._zoom_reset_100)
        self.btn_zoom_fit.clicked.connect(self._fit_preview)

        self.preview_scroll.viewport().installEventFilter(self)

        # Let the viewport receive all mouse events (prevents duplicate streams + jitter)
        self.preview_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        def _on_luma_only_toggled(on: bool):
            self.chk_linked.setEnabled(not on)
            # (dropdown enabling is handled by chk_luma_only.toggled -> cmb_luma.setEnabled)
        self.chk_luma_only.toggled.connect(_on_luma_only_toggled)
        _suggest_hdr_knee_from_target()
        self._populate_initial_preview()

    # ----- helpers -----
    def _show_busy(self, title: str, text: str):
        # Avoid stacking dialogs
        self._hide_busy()

        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)  # blocks only this tool window
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.setCancelButton(None)  # no cancel button (keeps it simple)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setFixedWidth(320)
        dlg.show()

        # Ensure it paints before heavy work starts
        QApplication.processEvents()
        self._busy = dlg

    def _hide_busy(self):
        try:
            if getattr(self, "_busy", None) is not None:
                self._busy.close()
                self._busy.deleteLater()
        except Exception:
            pass
        self._busy = None

    def _set_controls_enabled(self, enabled: bool):
        # Keep this minimal: disable Preview/Apply while running
        try:
            self.btn_preview.setEnabled(enabled)
            self.btn_apply.setEnabled(enabled)
        except Exception:
            pass

    def _start_stretch_job(self, mode: str):
        """
        mode: 'preview' or 'apply'
        """
        if getattr(self, "_job_running", False):
            return

        self._job_running = True
        self._job_mode = mode

        self._set_controls_enabled(False)
        self._show_busy("Statistical Stretch", "Processing…")

        self._thread = QThread(self)
        self._worker = _StretchWorker(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_stretch_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()


    def _get_source_float(self) -> np.ndarray:
        """
        Return a float32 array scaled into ~[0..1] for stretching.
        """
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None

        if np.issubdtype(src.dtype, np.integer):
            # Assume 16-bit astro sources by default; adjust if you prefer
            scale = 65535.0 if src.dtype.itemsize >= 2 else 255.0
            return (src.astype(np.float32) / scale).clip(0, 1)
        else:
            f = src.astype(np.float32)
            # If values are way above 1 (linear calibrated data), compress softly
            mx = float(f.max()) if f.size else 1.0
            if mx > 5.0:
                f = f / mx
            return f

    def _apply_current_zoom(self):
        """Apply the current zoom mode (fit or manual) to the preview image."""
        if self._preview_qimg is None:
            return
        if self._fit_mode:
            self._fit_preview()
        else:
            self._update_preview_scaled()

    def _fit_preview(self):
        """Fit the image into the visible scroll viewport."""
        if self._preview_qimg is None:
            return
        vp = self.preview_scroll.viewport().size()
        if vp.width() <= 1 or vp.height() <= 1:
            return
        iw, ih = self._preview_qimg.width(), self._preview_qimg.height()
        if iw <= 0 or ih <= 0:
            return
        # compute scale to fit
        sx = vp.width()  / iw
        sy = vp.height() / ih
        self._preview_scale = max(0.05, min(sx, sy))
        self._fit_mode = True
        self._update_preview_scaled()

    def _zoom_reset_100(self):
        """Set zoom to 100% (1:1)."""
        self._fit_mode = False
        self._preview_scale = 1.0
        self._update_preview_scaled()

    def _zoom_by(self, factor: float):
        vp = self.preview_scroll.viewport()
        center = vp.rect().center()
        self._zoom_at(factor, center)

    def _zoom_at(self, factor: float, vp_pos):
        """Zoom keeping the image point under vp_pos (viewport coords) stationary."""
        if self._preview_qimg is None:
            return

        old_scale = float(self._preview_scale)

        # Content coords (in scaled-image pixels) currently under the mouse
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        cx = hsb.value() + int(vp_pos.x())
        cy = vsb.value() + int(vp_pos.y())

        # Convert to image-space coords (unscaled)
        ix = cx / old_scale
        iy = cy / old_scale

        # Apply zoom
        self._fit_mode = False
        new_scale = max(0.05, min(old_scale * float(factor), 8.0))
        self._preview_scale = new_scale

        # Rebuild pixmap/label size
        self._update_preview_scaled()

        # New content coords for same image-space point
        ncx = int(ix * new_scale)
        ncy = int(iy * new_scale)

        # Set scrollbars so that point stays under the mouse
        hsb.setValue(ncx - int(vp_pos.x()))
        vsb.setValue(ncy - int(vp_pos.y()))


    # --- MASK helpers ----------------------------------------------------
    def _active_mask_array(self) -> np.ndarray | None:
        """Return active mask as float32 [H,W] in 0..1, resized to doc image."""
        try:
            mid = getattr(self.doc, "active_mask_id", None)
            if not mid:
                return None
            layer = getattr(self.doc, "masks", {}).get(mid)
            if layer is None:
                return None

            m = np.asarray(getattr(layer, "data", None))
            if m is None or m.size == 0:
                return None

            # squeeze to 2D
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[..., 0]
            elif m.ndim == 3:  # RGB/whatever → luminance
                m = (0.2126*m[...,0] + 0.7152*m[...,1] + 0.0722*m[...,2])

            orig = m
            # normalize if integer
            if orig.dtype.kind in "ui":
                m = orig.astype(np.float32) / float(np.iinfo(orig.dtype).max)
            else:
                m = orig.astype(np.float32, copy=False)

            m = np.clip(m, 0.0, 1.0)

            th, tw = self.doc.image.shape[:2]
            sh, sw = m.shape[:2]
            if (sh, sw) != (th, tw):
                yi = (np.linspace(0, sh-1, th)).astype(np.int32)
                xi = (np.linspace(0, sw-1, tw)).astype(np.int32)
                m = m[yi][:, xi]

            # honor opacity if present
            opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
            if opacity < 1.0:
                m *= opacity
            return m
        except Exception:
            return None

    def _blend_with_mask(self, base: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """base/out can be mono or 3ch; mask is [H,W] in 0..1."""
        if out.ndim == 3 and out.shape[2] == 3:
            m = mask[..., None]
        else:
            m = mask
        return base * (1.0 - m) + out * m


    def _run_stretch(self) -> np.ndarray | None:
        imgf = self._get_source_float()
        if imgf is None:
            return None
        blackpoint_sigma = float(self.sld_bp.value()) / 100.0
        hdr_on = bool(self.chk_hdr.isChecked())
        hdr_amount = float(self.sld_hdr_amt.value()) / 100.0
        hdr_knee = float(self.sld_hdr_knee.value()) / 100.0
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        luma_mode = str(self.cmb_luma.currentText()) if getattr(self, "cmb_luma", None) else "rec709"
        no_black_clip = bool(self.chk_no_black_clip.isChecked())

        target = float(self.spin_target.value())
        linked = bool(self.chk_linked.isChecked())
        normalize = bool(self.chk_normalize.isChecked())
        apply_curves = bool(self.chk_curves.isChecked())
        curves_boost = float(self.sld_curves.value()) / 100.0

        if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
            out = stretch_mono_image(
                imgf.squeeze(),
                target_median=target,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
                blackpoint_sigma=blackpoint_sigma,
                no_black_clip=no_black_clip,
                hdr_compress=hdr_on,
                hdr_amount=hdr_amount,
                hdr_knee=hdr_knee,
            )
        else:
            out = stretch_color_image(
                imgf,
                target_median=target,
                linked=linked,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
                blackpoint_sigma=blackpoint_sigma,
                no_black_clip=no_black_clip,
                hdr_compress=hdr_on,
                hdr_amount=hdr_amount,
                hdr_knee=hdr_knee,
                luma_only=luma_only,
                luma_mode=luma_mode,
            )

        # ✅ If a mask is active, blend stretched result with original
        m = self._active_mask_array()
        if m is not None:
            base = imgf.astype(np.float32, copy=False)
            out = self._blend_with_mask(base, out, m)

        return out            


    def _set_preview_pixmap(self, arr: np.ndarray):
        vis = arr
        if vis is None or vis.size == 0:
            self.preview_label.clear()
            return

        # Ensure 3 channels for display
        if vis.ndim == 2:
            vis3 = np.stack([vis] * 3, axis=-1)
        elif vis.ndim == 3 and vis.shape[2] == 1:
            vis3 = np.repeat(vis, 3, axis=2)
        else:
            vis3 = vis

        # Convert to 8-bit RGB
        if vis3.dtype == np.uint8:
            buf8 = vis3
        elif vis3.dtype == np.uint16:
            buf8 = (vis3.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            buf8 = (np.clip(vis3, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Must be C-contiguous for QImage
        buf8 = np.ascontiguousarray(buf8)
        h, w, _ = buf8.shape
        bytes_per_line = buf8.strides[0]

        # Build QImage from raw pointer; keep references alive
        self._last_preview = buf8  # keep backing store alive
        ptr = sip.voidptr(self._last_preview.ctypes.data)
        qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self._preview_qimg = qimg
        self._apply_current_zoom() 

    # ----- active document change -----
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._populate_initial_preview()

    # ----- slots -----
    def _populate_initial_preview(self):
        # show the current (unstretched) image as baseline
        src = self._get_source_float()
        if src is not None:
            self._set_preview_pixmap(np.clip(src, 0, 1))

    def _do_preview(self):
        self._start_stretch_job("preview")


    def _do_apply(self):
        self._start_stretch_job("apply")

    def _apply_out_to_doc(self, out: np.ndarray):
        # Preserve mono vs color shape
        if out.ndim == 3 and out.shape[2] == 3 and (self.doc.image.ndim == 2 or self.doc.image.shape[-1] == 1):
            out = out[..., 0]

        # --- Gather current UI state ------------------------------------
        target = float(self.spin_target.value())
        linked = bool(self.chk_linked.isChecked())
        normalize = bool(self.chk_normalize.isChecked())
        apply_curves = bool(getattr(self, "chk_curves", None) and self.chk_curves.isChecked())
        curves_boost = float(self.sld_curves.value()) / 100.0 if getattr(self, "sld_curves", None) is not None else 0.0
        blackpoint_sigma = float(self.sld_bp.value()) / 100.0
        hdr_on = bool(self.chk_hdr.isChecked())
        hdr_amount = float(self.sld_hdr_amt.value()) / 100.0
        hdr_knee = float(self.sld_hdr_knee.value()) / 100.0
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        luma_mode = str(self.cmb_luma.currentText()) if getattr(self, "cmb_luma", None) else "rec709"
        no_black_clip = bool(self.chk_no_black_clip.isChecked())

        parts = [f"target={target:.2f}", "linked" if linked else "unlinked"]
        if normalize:
            parts.append("norm")
        if apply_curves:
            parts.append(f"curves={curves_boost:.2f}")
        if self._active_mask_array() is not None:
            parts.append("masked")
        parts.append(f"bpσ={blackpoint_sigma:.2f}")
        if hdr_on and hdr_amount > 0:
            parts.append(f"hdr={hdr_amount:.2f}@{hdr_knee:.2f}")
        if luma_only:
            parts.append(f"luma={luma_mode}")
        if no_black_clip:
            parts.append("no_black_clip")

        step_name = f"Statistical Stretch ({', '.join(parts)})"
        self.doc.apply_edit(out.astype(np.float32, copy=False), step_name=step_name)

        # Turn off display stretch on the active view, if any
        mw = self.parent()
        if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
            view = mw.mdi.activeSubWindow().widget()
            if getattr(view, "autostretch_enabled", False):
                view.set_autostretch(False)

            # Existing logging, now using the same values as above
            if hasattr(mw, "_log"):
                curves_on = apply_curves
                boost_val = curves_boost if curves_on else 0.0
                mw._log(
                    "Applied Statistical Stretch "
                    f"(target={target:.3f}, linked={linked}, normalize={normalize}, "
                    f"bp_sigma={blackpoint_sigma:.2f}, "
                    f"hdr={'ON' if hdr_on else 'OFF'}"
                    f"{', amt='+str(round(hdr_amount,2))+' knee='+str(round(hdr_knee,2)) if hdr_on else ''}, "
                    f"luma={'ON' if luma_only else 'OFF'}{', mode='+luma_mode if luma_only else ''}, "
                    f"curves={'ON' if curves_on else 'OFF'}"
                    f"{', boost='+str(round(boost_val,2)) if curves_on else ''}, "
                    f"mask={'ON' if self._active_mask_array() is not None else 'OFF'})"
                )

            # --- Build preset for headless replay ---------------------------
            # --- Build preset for headless replay ---------------------------
            preset = {
                "target_median": target,
                "linked": linked,
                "normalize": normalize,
                "apply_curves": apply_curves,
                "curves_boost": curves_boost,
                "blackpoint_sigma": blackpoint_sigma,
                "no_black_clip": no_black_clip,
                "hdr_compress": hdr_on,
                "hdr_amount": hdr_amount,
                "hdr_knee": hdr_knee,
                "luma_only": luma_only,
                "luma_mode": luma_mode,
            }

            # ✅ Remember this as the last headless-style command
            #    (unless we are in a headless/suppressed call)
            suppress = bool(getattr(self, "_suppress_replay_record", False))
            if not suppress:
                from PyQt6.QtWidgets import QMainWindow
                try:
                    mw2 = self.parent()
                    while mw2 is not None and not isinstance(mw2, QMainWindow):
                        mw2 = mw2.parent()

                    if mw2 is not None and hasattr(mw2, "remember_last_headless_command"):
                        mw2.remember_last_headless_command(
                            command_id="stat_stretch",
                            preset=preset,
                            description="Statistical Stretch",
                        )
                        print(f"Remembered Statistical Stretch last headless command: {preset}")
                    else:
                        print("No main window with remember_last_headless_command; cannot store stat_stretch preset")
                except Exception as e:
                    print(f"Failed to remember Statistical Stretch last headless command: {e}")
            else:
                # optional debug
                print("Statistical Stretch: replay recording suppressed for this apply()")

        self.close()


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
                    # Reset preview state for new document
                    self._last_preview = None
                    self._preview_qimg = None
        except Exception:
            pass

    @pyqtSlot(object, str)
    def _on_stretch_done(self, out, err: str):
        # dialog might be closing; guard
        if sip.isdeleted(self):
            return

        self._hide_busy()
        self._set_controls_enabled(True)
        self._job_running = False

        if err:
            QMessageBox.warning(self, "Stretch failed", err)
            return

        if out is None:
            QMessageBox.information(self, "No image", "No image is loaded in the active document.")
            return

        if getattr(self, "_job_mode", "") == "preview":
            self._set_preview_pixmap(out)
            return

        # apply mode: reuse your existing apply logic, but using `out` we already computed
        self._apply_out_to_doc(out)

    def closeEvent(self, ev):
        try:
            if getattr(self, "_job_running", False):
                # let thread finish naturally; just hide popup so it doesn’t hang around
                self._hide_busy()
        except Exception:
            pass

        # existing disconnect logic...
        try:
            if self._follow_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass

        super().closeEvent(ev)

    def _update_preview_scaled(self):
        if self._preview_qimg is None:
            self.preview_label.clear()
            return
        sw = max(1, int(self._preview_qimg.width()  * self._preview_scale))
        sh = max(1, int(self._preview_qimg.height() * self._preview_scale))
        scaled = self._preview_qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))
        self.preview_label.resize(scaled.size())            # <- crucial for scrollbars

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._fit_preview()

    def eventFilter(self, obj, ev):
        # Ctrl+wheel zoom
        if ev.type() == QEvent.Type.Wheel and obj is self.preview_scroll.viewport():
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else (1/1.25)
                self._zoom_at(factor, ev.position())
                return True
            return False

        # Click+drag pan (left or middle mouse)
        if obj is self.preview_scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._panning = True
                    self._pan_last = ev.globalPosition().toPoint()
                    self.preview_scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif ev.type() == QEvent.Type.MouseMove and self._panning:
                pos = ev.globalPosition().toPoint()
                delta = pos - self._pan_last
                self._pan_last = pos

                hsb = self.preview_scroll.horizontalScrollBar()
                vsb = self.preview_scroll.verticalScrollBar()
                hsb.setValue(hsb.value() - delta.x())
                vsb.setValue(vsb.value() - delta.y())
                return True

            elif ev.type() == QEvent.Type.MouseButtonRelease and self._panning:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._panning = False
                    self._pan_last = None
                    self.preview_scroll.viewport().unsetCursor()
                    return True

        return super().eventFilter(obj, ev)
     