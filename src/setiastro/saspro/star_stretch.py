#saspro/star_stretch.py
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QPointF, QSettings, QByteArray, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
    QPushButton, QScrollArea, QWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QMovie, QIcon
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01
from setiastro.saspro.shortcuts import PresetDragHandle
from setiastro.saspro.resources import starstretch_path
from setiastro.saspro.help_support import make_help_button

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
    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]

    # [0,1] -> uint8 and force C-contiguous. Mono is converted first and its
    # bytes replicated to RGB (same bytes as stacking the floats, which built
    # a 3x float copy of a 26 MP frame: 0.66 s on the GUI thread).
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    if buf8.ndim == 2:
        try:   # same bytes as np.repeat, which holds the GIL ~0.4 s on 26 MP
            import cv2
            buf8 = cv2.cvtColor(np.ascontiguousarray(buf8), cv2.COLOR_GRAY2RGB)
        except Exception:
            buf8 = np.repeat(buf8[..., None], 3, axis=2)
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
    # (np.ndarray float32 0..1, QImage of it or None). The QImage is built
    # here when no mask will be blended in: converting a full frame to 8-bit
    # on the GUI thread froze it 0.4-0.8 s per preview. QImage (not QPixmap)
    # may be built off the GUI thread.
    preview_ready = pyqtSignal(object, object)

    def __init__(self, image: np.ndarray, stretch_factor: float, sat_amount: float, do_scnr: bool,
                 make_qimage: bool = False):
        super().__init__()
        self.image = image
        self.stretch_factor = float(stretch_factor)  # this is the "amount" for your pixel math
        self.sat_amount = float(sat_amount)
        self.do_scnr = bool(do_scnr)
        self.make_qimage = bool(make_qimage)

    def run(self):
        imgf = _to_float01(self.image)
        if imgf is None:
            return

        # Grayscale goes through the (H, W, C) kernel as one channel and is
        # collapsed back afterwards. Saturation and SCNR are RGB-only.
        need_collapse = False
        if imgf.ndim == 2:
            imgf = imgf[..., None]
            need_collapse = True
        elif imgf.ndim == 3 and imgf.shape[2] == 1:
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

        out = out.astype(np.float32, copy=False)
        qimg = _as_qimage_rgb8(_to_float01(out)) if self.make_qimage else None
        self.preview_ready.emit(out, qimg)

# ---- dialog ---------------------------------------------------------------

class StarStretchDialog(QDialog):
    """
    Star Stretch for SASpro.
    - Works on active ImageDocument (passed in).
    - Preview is computed in background thread.
    - 'Apply to Document' records history via doc.apply_edit(..., step_name="Star Stretch").

    Auto-refresh pipeline (when enabled):
      - any control change -> _auto_timer (_AUTO_REFRESH_MS) -> full-res preview
      - exactly one worker at a time; a change made while a worker is in flight
        re-arms the timer from _on_worker_finished, so edits are never dropped
        and previews never stack up
    """

    # Debounce for auto-refresh previews, in ms.  Stat Stretch uses 800 because
    # its per-channel sigma work is heavy; Star Stretch is a single pixel-math
    # pass, so it can settle faster and still feel instant.  Bump to 800 for
    # exact parity with Stat Stretch.
    _AUTO_REFRESH_MS = 250

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Star Stretch"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = parent
        self.doc = document
        self._preview: np.ndarray | None = None
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
        # Connect to active document change signal
        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False

        self._pix: QPixmap | None = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._apply_when_ready = False  
        self._headless = False  # True when constructed for a headless preset apply (no geometry save)
        # auto-refresh (instant preview) state
        self._busy = False                    # a preview worker is in flight
        self._preview_stale = True            # self._preview no longer matches the controls
        self._job_params: dict | None = None  # params the in-flight worker was started with
        self._settings_dirty = False          # user touched a control this session
        # UI
        main = QHBoxLayout(self)

        # Left column (controls)
        left = QVBoxLayout()
        info = QLabel(self.tr(
            "Instructions:\n"
            "1) Adjust stretch and options.\n"
            "2) Preview the result.\n"
            "3) Apply to the current document."
        ))
        info.setWordWrap(True)
        left.addWidget(info)

        # Stretch slider (0..8.00)
        self.lbl_st = QLabel(self.tr("Stretch Amount: {0:.2f}").format(5.0))
        self.sld_st = QSlider(Qt.Orientation.Horizontal)
        self.sld_st.setRange(0, 800)
        self.sld_st.setValue(500)
        self.sld_st.valueChanged.connect(self._on_stretch_changed)
        left.addWidget(self.lbl_st)
        left.addWidget(self.sld_st)

        # Saturation slider (0..2.00)
        self.lbl_sat = QLabel(self.tr("Color Boost: {0:.2f}").format(1.0))
        self.sld_sat = QSlider(Qt.Orientation.Horizontal)
        self.sld_sat.setRange(0, 200)
        self.sld_sat.setValue(100)
        self.sld_sat.valueChanged.connect(self._on_sat_changed)
        left.addWidget(self.lbl_sat)
        left.addWidget(self.sld_sat)

        # SCNR checkbox
        self.chk_scnr = QCheckBox(self.tr("Remove Green via SCNR (Optional)"))
        left.addWidget(self.chk_scnr)

        # Auto-refresh (instant preview) toggle - mirrors Stat Stretch
        self.chk_auto = QCheckBox(self.tr("Auto-refresh preview"))
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(self.tr(
            "When enabled, any control change updates the preview automatically\n"
            "after a short settle, so you don't have to press Preview."
        ))
        left.addWidget(self.chk_auto)

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

        # ── Drag-to-canvas grip (PI-style "new instance") ─────────────────
        drag_row = QHBoxLayout()
        drag_row.setContentsMargins(0, 0, 0, 0)
        self.preset_drag_handle = PresetDragHandle(
            "star_stretch",
            self._star_stretch_params,
            icon=QIcon(starstretch_path),
            tooltip=self.tr(
                "Drag to the canvas to create a Star Stretch shortcut\n"
                "with these exact settings.\n"
                "Drop directly on an image to apply them headlessly."
            ),
            parent=self,
        )
        drag_row.addWidget(self.preset_drag_handle)
        drag_row.addStretch(1)
        self.btn_help = make_help_button("star_stretch", self)
        drag_row.addWidget(self.btn_help)
        left.addLayout(drag_row)

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
        # NOTE: clicked(bool) would bind to _run_preview's `interactive` arg,
        # so route the button through a lambda and keep interactive=True.
        self.btn_preview.clicked.connect(lambda: self._run_preview(True))
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.chk_scnr.toggled.connect(lambda _=False: self._on_any_control_changed())
        self.chk_auto.toggled.connect(self._on_auto_refresh_toggled)

        # Auto-refresh debounce timer -> full-res preview once controls settle
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(int(self._AUTO_REFRESH_MS))
        self._auto_timer.timeout.connect(self._auto_preview)

        # Restore every user option from the last session (signals blocked
        # inside, so this cannot fire a handler or schedule a save)
        self._load_settings()

        # Persist on change, debounced: one slider drag becomes one write
        # instead of one per tick.  closeEvent flushes anything still pending.
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(400)
        self._settings_timer.timeout.connect(self._save_settings)
        self.sld_st.valueChanged.connect(self._schedule_save_settings)
        self.sld_sat.valueChanged.connect(self._schedule_save_settings)
        self.chk_scnr.toggled.connect(self._schedule_save_settings)
        self.chk_auto.toggled.connect(self._schedule_save_settings)

        # initialize preview with current doc image
        self._update_preview_pix(self.doc.image)

        # If auto-refresh was left on, show a stretched preview immediately
        if self.chk_auto.isChecked():
            self._auto_timer.start()

    # --- active document change ---
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._preview = None
        self._preview_stale = True
        self._update_preview_pix(self.doc.image)
        # Switching to another image should re-stretch it if auto is on
        chk = getattr(self, "chk_auto", None)
        if chk is not None and chk.isChecked() and not getattr(self, "_headless", False):
            self._auto_timer.start()

    # --- UI change handlers ---
    def _on_stretch_changed(self, v: int):
        self.lbl_st.setText(self.tr("Stretch Amount: {0:.2f}").format(v/100.0))
        self._on_any_control_changed()

    def _on_sat_changed(self, v: int):
        self.lbl_sat.setText(self.tr("Color Boost: {0:.2f}").format(v/100.0))
        self._on_any_control_changed()

    # --- auto-refresh (instant preview) ---
    def _on_any_control_changed(self):
        """Called by every control that changes the stretched output.

        Marks the cached preview stale (so Apply can't commit a result from
        older settings) and, when auto-refresh is on, re-arms the debounce.
        Guarded with getattr because the sliders are connected before
        chk_auto / _auto_timer exist during __init__.
        """
        self._preview_stale = True
        chk = getattr(self, "chk_auto", None)
        if chk is None or not chk.isChecked():
            return
        t = getattr(self, "_auto_timer", None)
        if t is None:
            return
        # If a worker is in flight this is a no-op worth making anyway: the
        # staleness check in _on_worker_finished re-arms once it lands.
        t.start()

    def _on_auto_refresh_toggled(self, on: bool):
        # Persistence now goes through the shared _schedule_save_settings hook,
        # which is connected to this same toggled signal.
        if on:
            if self._preview is None or self._preview_stale:
                self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _auto_preview(self):
        """Debounce timer slot."""
        if getattr(self, "_headless", False):
            return
        chk = getattr(self, "chk_auto", None)
        if chk is None or not chk.isChecked():
            return
        self._run_preview(False)

    # --- preview / processing ---
    def _run_preview(self, interactive: bool = True):
        """Start a preview worker.

        interactive=False is the auto-refresh path: it stays silent about a
        missing image instead of popping a message box behind the user.
        """
        t = getattr(self, "_auto_timer", None)
        if t is not None:
            t.stop()

        if self._busy:
            # One worker at a time.  _on_worker_finished re-arms the timer if
            # the controls have moved since this job started.
            return

        img = getattr(self.doc, "image", None) if self.doc is not None else None
        if img is None:
            if interactive:
                QMessageBox.information(self, self.tr("No image"), self.tr("Open an image first."))
            self._apply_when_ready = False
            return

        # Remember what this job is computing, so we can tell on completion
        # whether the result still matches the controls.
        self._job_params = self._star_stretch_params()

        self._busy = True
        self._show_spinner(True)
        self.btn_preview.setEnabled(False)
        self.btn_apply.setEnabled(False)

        self._thr = _StarStretchWorker(
            image=img,
            stretch_factor=self.sld_st.value()/100.0,
            sat_amount=self.sld_sat.value()/100.0,
            do_scnr=self.chk_scnr.isChecked(),
            make_qimage=(self._active_mask_layer()[0] is None),
        )
        self._thr.preview_ready.connect(self._on_preview_ready)
        self._thr.finished.connect(self._on_worker_finished)
        self._thr.start()

    def _on_preview_ready(self, out: np.ndarray, qimg=None):
        out_masked = self._blend_with_mask(out)
        self._preview = out_masked

        # Did the controls move while this was computing?
        try:
            self._preview_stale = (self._job_params != self._star_stretch_params())
        except Exception:
            self._preview_stale = False

        if qimg is not None and out_masked is out:
            # the worker already rendered exactly this array
            self._pix = QPixmap.fromImage(qimg)
            self._apply_zoom()
        else:
            self._update_preview_pix(out_masked)

        mw = self._find_main_window()
        if mw and hasattr(mw, "_log"):
            mw._log("Star Stretch: preview generated.")

        # Button re-enable and the pending-apply / pending-refresh decisions
        # moved to _on_worker_finished: preview_ready is emitted from inside
        # run(), so the thread is still alive here and _run_preview's
        # single-flight guard would reject anything started from this slot.

    def _rerun_preview_for_apply(self):
        self._run_preview(False)

    def _on_worker_finished(self):
        """Worker thread done: free the slot, then settle any pending work."""
        self._busy = False
        self._show_spinner(False)
        self.btn_preview.setEnabled(True)
        self.btn_apply.setEnabled(True)

        # Apply was pressed before the preview landed.
        if self._apply_when_ready:
            if self._preview is None:
                self._apply_when_ready = False
                return
            if self._preview_stale:
                # Controls moved mid-flight - recompute, then apply that result
                # (_apply_when_ready stays set).  Converges once the user stops
                # touching the sliders.
                # A bound method, not a lambda: PyQt drops a bound-method
                # call once the dialog is deleted; a lambda would still run.
                QTimer.singleShot(0, self._rerun_preview_for_apply)
                return
            self._apply_when_ready = False
            QTimer.singleShot(0, self._finish_apply)
            return

        # Auto-refresh: pick up changes made while the worker was running.
        chk = getattr(self, "chk_auto", None)
        if self._preview_stale and chk is not None and chk.isChecked():
            self._auto_timer.start()

    def _apply_to_doc(self):
        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass
        # If we have no preview - or the controls moved since the last one -
        # compute a fresh result and auto-apply when it lands.  Without the
        # staleness test, Preview -> nudge slider -> Apply committed the
        # previous settings.
        if self._preview is None or self._preview_stale:
            if self._busy:
                # already computing; just mark to apply when it lands
                self._apply_when_ready = True
                return
            self._apply_when_ready = True
            self._run_preview(True)
            return

        # We do have a preview → finish immediately
        self._finish_apply()  

    def _star_stretch_params(self) -> dict:
        """Canonical preset — single source of truth for apply + the drag handle."""
        return {
            "stretch_factor": self.sld_st.value() / 100.0,
            "color_boost":    self.sld_sat.value() / 100.0,
            "scnr_green":     self.chk_scnr.isChecked(),
        }

    def seed_from_preset(self, p: dict | None) -> None:
        """
        Public: load a preset dict (same schema as _star_stretch_params) into
        the live controls, so a double-clicked shortcut opens the dialog seeded.

        Divisor trap: both sliders are /100 on the way OUT in
        _star_stretch_params (stretch_factor, color_boost); seeding multiplies
        back (widget = value * 100). Labels are set explicitly since we block
        signals during the push (the valueChanged handlers that format them
        won't fire).
        """
        if not p:
            return

        widgets = [self.sld_st, self.sld_sat, self.chk_scnr]
        for w in widgets:
            try:
                w.blockSignals(True)
            except Exception:
                pass

        try:
            stretch = float(p.get("stretch_factor", 5.0))
            self.sld_st.setValue(int(round(stretch * 100)))
            self.lbl_st.setText(self.tr("Stretch Amount: {0:.2f}").format(stretch))

            boost = float(p.get("color_boost", 1.0))
            self.sld_sat.setValue(int(round(boost * 100)))
            self.lbl_sat.setText(self.tr("Color Boost: {0:.2f}").format(boost))

            self.chk_scnr.setChecked(bool(p.get("scnr_green", False)))
        finally:
            for w in widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

        # Seeded values invalidate any cached preview.  Signals were blocked
        # above, so _on_any_control_changed never fired - do its work here.
        self._preview = None
        self._preview_stale = True
        chk = getattr(self, "chk_auto", None)
        if chk is not None and chk.isChecked() and not getattr(self, "_headless", False):
            self._auto_timer.start()

    def _finish_apply(self):
        # Scheduled from _on_worker_finished; the dialog may have been closed
        # (and its image released) before this ran. Nothing to apply then --
        # going on would pop "Apply failed: 'NoneType' ..." after the close.
        if getattr(self, "_memory_released", False):
            return
        # No auto-refresh firing into a dialog that is about to close
        try:
            self._auto_timer.stop()
        except Exception:
            pass
        try:
            _marr, mid, mname = self._active_mask_layer()
            meta = {
                "step_name": "Star Stretch",
                "command_id": "star_stretch",
                "preset": self._star_stretch_params(),
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
            if mw is not None:
                display_stretch_off_for(mw, self.doc)
            if mw and hasattr(mw, "_log"):
                mw._log("Star Stretch: applied to document.")

            # 🔁 Record as last headless-style command for Replay
            try:
                if mw and hasattr(mw, "_remember_last_headless_command"):
                    preset = self._star_stretch_params()
                    mw._remember_last_headless_command(
                        "star_stretch",
                        preset,
                        description="Star Stretch",
                    )
            except Exception:
                # Don't let replay bookkeeping break the dialog
                pass

        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))
            return
        
        # Dialog stays open so user can apply to other images
        # Refresh document reference for next operation
        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass

        self.close()
        return

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self._find_main_window()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    # Reset preview for new document
                    self._preview = None
                    self._preview_stale = True
                    self._run_preview(False)
        except Exception:
            pass


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
    # --- event filter (wheel zoom + panning) ---
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                dy = ev.pixelDelta().y()

                if dy != 0:
                    abs_dy = abs(dy)
                    ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

                    if abs_dy <= 3:
                        base_factor = 1.012 if ctrl_down else 1.010
                    elif abs_dy <= 10:
                        base_factor = 1.025 if ctrl_down else 1.020
                    else:
                        base_factor = 1.040 if ctrl_down else 1.030

                    factor = base_factor if dy > 0 else 1.0 / base_factor
                else:
                    dy = ev.angleDelta().y()
                    if dy == 0:
                        ev.accept()
                        return True

                    ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl_down else 1.15
                    factor = step if dy > 0 else 1.0 / step

                self._set_zoom(self._zoom * factor)
                ev.accept()
                return True

            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept()
                return True

            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - int(d.x()))
                v.setValue(v.value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept()
                return True

            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept()
                return True

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

    def closeEvent(self, ev):
        self._close_teardown()
        super().closeEvent(ev)

    def done(self, r):
        # Esc ends a QDialog through done(), which never delivers closeEvent:
        # without this it skipped the settings flush, the worker wait and the
        # release, and the dialog kept its preview until collected. Runs once.
        self._close_teardown()
        super().done(r)

    def _close_teardown(self):
        if getattr(self, "_close_torn_down", False):
            return
        self._close_torn_down = True
        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass

        # 0a) Write out any debounced option change (e.g. a slider moved and
        #     the dialog closed inside the 400 ms settle)
        try:
            self._flush_settings()
        except Exception:
            pass

        # 0) Stop auto-refresh and make sure no preview worker outlives the
        #    dialog.  WA_DeleteOnClose + a live QThread emitting into this
        #    object is a crash; the block below used to wait on self._thread /
        #    self._worker, which this dialog never sets.  The real handle is
        #    self._thr.
        try:
            t = getattr(self, "_auto_timer", None)
            if t is not None:
                t.stop()
        except Exception:
            pass
        try:
            thr = getattr(self, "_thr", None)
            if thr is not None and thr.isRunning():
                try:
                    thr.preview_ready.disconnect()
                except Exception:
                    pass
                try:
                    thr.requestInterruption()
                except Exception:
                    pass
                thr.wait(3000)
        except Exception:
            pass

        # 1) Disconnect active-doc tracking (Fabio hook)
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False

        # 2) Stop any background preview worker/thread if you have one
        # (names may differ in your file; keep what matches your implementation)
        try:
            if getattr(self, "_worker", None) is not None:
                try:
                    self._worker.requestInterruption()
                except Exception:
                    pass
            if getattr(self, "_thread", None) is not None:
                self._thread.quit()
                self._thread.wait(500)
        except Exception:
            pass

        try:
            self._release_memory()
        except Exception:
            pass

    def _release_memory(self):
        """Drop preview buffer and pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self._preview = None
        self._pix = None
        self._thr = None
        self.doc = None
        self._main = None
        try:
            self.label.clear()
        except Exception:
            pass

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

    # --- option persistence -------------------------------------------------
    #
    # Keys live under star_stretch_ui/ to match the existing geometry key.
    # Sliders are stored x100, the same convention Stat Stretch uses, so the
    # stored value is the widget value and no rounding drift creeps in.

    _DEFAULT_STRETCH_X100 = 500   # 5.00
    _DEFAULT_BOOST_X100   = 100   # 1.00

    def _load_settings(self):
        """Restore every user-facing option from the previous session.

        Signals are blocked for the whole push, so no handler fires and no
        save is scheduled; the labels are therefore set explicitly here (the
        valueChanged handlers that normally format them won't run) - the same
        divisor trap seed_from_preset documents.
        """
        try:
            s = QSettings()
        except Exception:
            return

        widgets = [self.sld_st, self.sld_sat, self.chk_scnr, self.chk_auto]
        for w in widgets:
            try:
                w.blockSignals(True)
            except Exception:
                pass

        try:
            try:
                stretch = int(s.value("star_stretch_ui/stretch_amount_x100",
                                      self._DEFAULT_STRETCH_X100))
            except Exception:
                stretch = self._DEFAULT_STRETCH_X100
            stretch = max(self.sld_st.minimum(), min(self.sld_st.maximum(), stretch))
            self.sld_st.setValue(stretch)
            self.lbl_st.setText(self.tr("Stretch Amount: {0:.2f}").format(stretch/100.0))

            try:
                boost = int(s.value("star_stretch_ui/color_boost_x100",
                                    self._DEFAULT_BOOST_X100))
            except Exception:
                boost = self._DEFAULT_BOOST_X100
            boost = max(self.sld_sat.minimum(), min(self.sld_sat.maximum(), boost))
            self.sld_sat.setValue(boost)
            self.lbl_sat.setText(self.tr("Color Boost: {0:.2f}").format(boost/100.0))

            self.chk_scnr.setChecked(bool(s.value("star_stretch_ui/scnr_green",
                                                  False, type=bool)))
            self.chk_auto.setChecked(bool(s.value("star_stretch_ui/auto_refresh",
                                                  True, type=bool)))
        except Exception:
            pass
        finally:
            for w in widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

    def _save_settings(self, *_):
        """Persist every user-facing option."""
        if getattr(self, "_headless", False):
            return
        try:
            s = QSettings()
            s.setValue("star_stretch_ui/stretch_amount_x100", int(self.sld_st.value()))
            s.setValue("star_stretch_ui/color_boost_x100", int(self.sld_sat.value()))
            s.setValue("star_stretch_ui/scnr_green", bool(self.chk_scnr.isChecked()))
            s.setValue("star_stretch_ui/auto_refresh", bool(self.chk_auto.isChecked()))
            self._settings_dirty = False
        except Exception:
            pass

    def _schedule_save_settings(self, *_):
        """Debounce persistence: a slider drag becomes a single write.

        Only reached from a real control signal, so a preset pushed in with
        signals blocked (seed_from_preset) never marks the session dirty and
        never overwrites the user's stored defaults.
        """
        if getattr(self, "_headless", False):
            return
        self._settings_dirty = True
        t = getattr(self, "_settings_timer", None)
        if t is None:
            self._save_settings()
            return
        t.start()

    def _flush_settings(self):
        """Write a pending debounced save immediately (used on close)."""
        if getattr(self, "_headless", False):
            return
        if not getattr(self, "_settings_dirty", False):
            return
        t = getattr(self, "_settings_timer", None)
        try:
            if t is not None and t.isActive():
                t.stop()
        except Exception:
            pass
        self._save_settings()

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("star_stretch_ui/window_geometry", None)  # ✅ unique key
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("star_stretch_ui/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)

        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True

            def _after_restore_refit():
                self._restore_window_geometry()
                # viewport changed -> refit the preview so it fills correctly
                self._fit()

            QTimer.singleShot(0, _after_restore_refit)
            return

        # normal path: keep your UI consistent when re-shown
        QTimer.singleShot(0, self._fit)

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

def open_star_stretch_with_preset(main_window, preset: dict | None = None):
    """
    Open the live Star Stretch dialog seeded from a preset. Used by the
    double-click preset-open path (ShortcutManager.trigger_with_preset).
    Resolves the active document from the active MDI subwindow first —
    matching the tool's toolbar opener — so it seeds onto the visible image.
    """
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            w = sw.widget()
            doc = getattr(w, "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    if doc is None:
        return

    dlg = StarStretchDialog(main_window, doc)
    try:
        dlg.setWindowIcon(QIcon(starstretch_path))
    except Exception:
        pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()


# ─────────────────────────────────────────────────────────────────────────────
def display_stretch_off_for(main_window, doc) -> int:
    """
    After Star Stretch has changed `doc`'s pixels: turn Display-Stretch off on
    every view showing it, or the screen stretches the already-stretched data
    a second time. Statistical Stretch does the same. Returns how many views
    were switched off.
    """
    mdi = getattr(main_window, "mdi", None)
    if mdi is None or doc is None:
        return 0

    def resolved(d):
        try:
            if hasattr(d, "_target") and callable(d._target):
                return d._target()               # a _DocProxy: its current document
        except Exception:
            pass
        return d

    # The dialog is handed view.document, which is a _DocProxy; a Preview tab
    # hands it an ROI document whose _parent_doc is the image. Any of them
    # names the image, so compare by identity against all of them.
    wanted = [d for d in (doc, resolved(doc), getattr(doc, "_parent_doc", None)) if d is not None]
    n = 0
    for sw in list(mdi.subWindowList()):
        try:
            view = sw.widget()
        except RuntimeError:                     # deleted with its window
            continue
        if view is None or not getattr(view, "autostretch_enabled", False):
            continue
        shown = getattr(view, "document", None)
        mine = [d for d in (shown, resolved(shown), getattr(view, "base_document", None))
                if d is not None]
        if not any(a is b for a in mine for b in wanted):
            continue
        try:
            view.set_autostretch(False)
            n += 1
        except Exception:
            pass
    return n


def apply_star_stretch_preset_to_doc(doc, preset: dict | None = None):
    """
    Headless Star Stretch on a specific document: what a shortcut or a drop on
    a view runs, and the command-line tool. Uses the Numba kernel
    (applyPixelMath_numba); raises if it is unavailable.
    """
    import numpy as np
    preset = dict(preset or {})
    try:
        from setiastro.saspro.legacy.numba_utils import applyPixelMath_numba, applySCNR_numba
        _has_numba = True
    except Exception:
        _has_numba = False
        applyPixelMath_numba = None
        # lightweight SCNR fallback (same as in pro/star_stretch.py)
        def applySCNR_numba(image_array: np.ndarray) -> np.ndarray:
            img = image_array.astype(np.float32, copy=False)
            if img.ndim != 3 or img.shape[2] != 3:
                return img
            r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
            g2 = np.minimum(g, 0.5 * (r + b))
            out = img.copy()
            out[..., 1] = g2
            return np.clip(out, 0.0, 1.0)

    img = getattr(doc, "image", None)
    if img is None:
        raise RuntimeError("Document has no image.")

    # read preset (accept a few aliases)
    amount = float(preset.get("stretch_factor",
            preset.get("stretch_amount",
            preset.get("amount", 5.0))))  # 0..8
    sat    = float(preset.get("color_boost", preset.get("saturation", 1.0)))  # 0..2
    scnr   = bool(preset.get("scnr_green", preset.get("scnr", False)))

    # convert to float 0..1
    a = np.asarray(img)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    elif a.dtype.kind == "f":
        mx = float(a.max()) if a.size else 1.0
        a = a.astype(np.float32) / (mx if mx > 1.0 else 1.0)
    else:
        a = a.astype(np.float32)

    # mono -> 3ch temporarily
    need_collapse = False
    if a.ndim == 2:
        a = np.stack([a]*3, axis=-1); need_collapse = True
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2); need_collapse = True

    if applyPixelMath_numba is None:
        raise RuntimeError("Star Stretch requires Numba kernel (applyPixelMath_numba) for headless run.")

    out = applyPixelMath_numba(a, amount)

    # color boost
    if out.ndim == 3 and out.shape[2] == 3 and abs(sat - 1.0) > 1e-6:
        mean = out.mean(axis=2, keepdims=True)
        out = mean + (out - mean) * sat
        out = np.clip(out, 0.0, 1.0)

    # SCNR
    if scnr and out.ndim == 3 and out.shape[2] == 3:
        out = applySCNR_numba(out.astype(np.float32, copy=False))

    if need_collapse:
        out = out[..., 0]

    meta = {
        "step_name": "Star Stretch",
        "command_id": "star_stretch",
        "preset": {
            "stretch_factor": amount,
            "color_boost": sat,
            "scnr_green": scnr,
        },
        "star_stretch": {
            "stretch_factor": amount,
            "color_boost": sat,
            "scnr_green": scnr,
            "numba": _has_numba,
        }
    }
    doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name="Star Stretch")
