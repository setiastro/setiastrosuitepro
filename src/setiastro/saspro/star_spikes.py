#saspro/tools/star_spikes.py
from __future__ import annotations
import numpy as np
import math
import threading
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSplitter, QSizePolicy, QWidget,
                             QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)

from PyQt6.QtGui import QPixmap, QImage, QPainter
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)
# deps
try:
    import sep
except Exception as _e_sep:
    sep = None
try:
    import cv2
except Exception as _e_cv2:
    cv2 = None
try:
    from scipy.ndimage import gaussian_filter
    import scipy.ndimage as ndi
except Exception as _e_scipy:
    gaussian_filter = None
    ndi = None


def _tr(text: str) -> str:
    return QCoreApplication.translate("StarSpikes", text)

class PreviewView(QGraphicsView):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # drag to pan
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # zoom toward mouse
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # when the view resizes, keep the scene centered
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # nicer defaults
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

class _SpikesJob(QThread):
    """One step of a Generate (detect or render) off the GUI thread.

    No parent, started with thread_keeper.start_owned_thread, so the dialog
    may close mid-run. fn gets plain data only (no widgets) plus
    report(text) and cancelled(); its dict comes back through `result`.
    """
    result = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, fn, kwargs, cancel_evt):
        super().__init__(None)
        self._fn = fn
        self._kwargs = kwargs
        self._cancel_evt = cancel_evt

    def cancel(self):
        self._cancel_evt.set()

    def run(self):
        try:
            out = self._fn(report=self.progress.emit,
                           cancelled=self._cancel_evt.is_set, **self._kwargs)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        self._kwargs = None
        self.result.emit(out)


class StarSpikesDialogPro(QDialog):
    WARN_LIMIT = 1000
    MAX_AUTO_RETRIES = 2
    # Detection threshold budget (see _detect_stars). SEP's time grows with
    # the number of pixels over the threshold, faster than linearly, and
    # sep.extract holds the GIL for the whole call, so the app is frozen for
    # all of it. Measured 2026-11-19: 26 MP stretched Ha master, 93k px over
    # sigma 50 -> 1.0 s, 212k over 25 -> 3.7 s, 1.28M over 5 -> 90-110 s;
    # 22 MP OSC, 102k over 5 (stretched) -> 0.5 s, 199k over 5 (linear) -> 2.1 s.
    DETECT_BUDGET_PX = 200_000
    DETECT_LADDER = (200.0, 100.0, 50.0, 25.0, 12.0, 7.0)

    def __init__(self, parent=None, doc_manager=None,
                 initial_doc=None,
                 jwstpupil_path: str | None = None,
                 aperture_help_path: str | None = None,
                 spinner_path: str | None = None,
                 follow_active: bool = True):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Diffraction Spikes"))
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
        self.docman = doc_manager
        self._main = parent
        # False when a caller pinned us to one specific doc (command drop,
        # replay); then we never retarget on view switches.
        self._follow_active = bool(follow_active)
        self._follow_conn = False
        self.doc = initial_doc or self._resolve_active_doc()
        self.jwstpupil_path = jwstpupil_path
        self.aperture_help_path = aperture_help_path

        self.final_image = None
        self._img_src = None       # float32, 2D or 3D, [0..1]
        self._img_src_doc = None   # which doc _img_src was read from
        self._result_doc = None    # which doc final_image was computed from
        self._busy = False         # a Generate run is in flight (worker threads)
        self._pending_close = False
        self._rerun_pending = False  # Generate clicked mid-run: run again after
        self._run_ctx = None       # per-run state handed between the run's steps
        self._cancel_evt = None    # set on close: the worker stops between steps
        self._job = None

        # defaults (aligned to your SASv2 tool)
        self.advanced = {
            "flux_max": 300.0, "bscale_min": 10.0, "bscale_max": 30.0,
            "shrink_min": 1.0, "shrink_max": 5.0, "detect_thresh": 5.0,
        }

        self._build_ui()
        # Factory defaults, before the opener seeds a preset (apply_preset)
        self._factory = snapshot_controls(self)
        self._factory_advanced = dict(self.advanced)
        self._load_active_image()
        self._hook_active_doc_following()

    # ---------- UI ----------
    def _build_ui(self):
        # top-level splitter: controls (left) | preview (right)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        # ----- LEFT: controls panel (stacked groups) -----
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(10, 10, 10, 10)
        left_v.setSpacing(10)

        def dspin(lo, hi, step, val, decimals=2):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(decimals)
            sp.setValue(val)
            sp.setMaximumWidth(140)
            return sp

        def ispin(lo, hi, step, val):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setValue(val)
            sp.setMaximumWidth(140)
            return sp

        # --- Group: Star Detection ---
        grp_detect = QGroupBox(self.tr("Star Detection"))
        f_detect = QFormLayout(grp_detect)
        f_detect.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_detect.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.flux_min = dspin(0.0, 999999.0, 10.0, 30.0, decimals=1)
        self.detect_thresh = dspin(0.5, 50.0, 0.5, float(self.advanced.get("detect_thresh", 5.0)), decimals=2)
        self.detect_thresh.setToolTip(self.tr("σ threshold for SEP detection (higher = fewer stars)."))
        # keep self.advanced in sync if user edits
        self.detect_thresh.valueChanged.connect(lambda v: self.advanced.__setitem__("detect_thresh", float(v)))

        f_detect.addRow(self.tr("Flux Min:"), self.flux_min)
        f_detect.addRow(self.tr("Detection Threshold (σ):"), self.detect_thresh)

        # --- Group: Aperture (Geometry) ---
        grp_ap = QGroupBox(self.tr("Aperture"))
        f_ap = QFormLayout(grp_ap)
        f_ap.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_ap.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.pupil_jwst = QPushButton(self.tr("Circular"))
        self.pupil_jwst.setCheckable(True)
        self.pupil_jwst.setChecked(False)
        self.pupil_jwst.toggled.connect(lambda on: self._toggle_pupil(on))
        self.pupil_jwst.setToolTip(self.tr("Toggle between circular aperture and JWST pupil image."))
        self.pupil_jwst.setStyleSheet("""
            QPushButton { min-width: 72px; max-width: 72px; min-height: 28px; max-height: 28px;
                          border-radius: 14px; background:#ccc; border:1px solid #999;}
            QPushButton:checked { background:#66bb6a; }
        """)
        f_ap.addRow(self.tr("Aperture Type:"), self.pupil_jwst)

        self.radius      = dspin(1.0, 512.0, 1.0, 128.0, decimals=1)
        self.obstruction = dspin(0.0, 0.99, 0.01, 0.2, decimals=2)
        self.num_vanes   = ispin(2, 8, 1, 4)
        self.vane_width  = dspin(0.0, 50.0, 0.5, 4.0, decimals=2)
        self.rotation    = dspin(0.0, 360.0, 1.0, 0.0, decimals=1)

        f_ap.addRow(self.tr("Pupil Radius:"), self.radius)
        f_ap.addRow(self.tr("Obstruction:"), self.obstruction)
        f_ap.addRow(self.tr("Number of Vanes:"), self.num_vanes)
        f_ap.addRow(self.tr("Vane Width:"), self.vane_width)
        f_ap.addRow(self.tr("Rotation (deg):"), self.rotation)

        # --- Group: PSF & Synthesis ---
        grp_psf = QGroupBox(self.tr("PSF && Synthesis"))
        f_psf = QFormLayout(grp_psf)
        f_psf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_psf.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.color_boost = dspin(0.1, 10.0, 0.1, 1.5, decimals=2)
        self.blur_sigma  = dspin(0.1, 10.0, 0.1, 2.0, decimals=2)

        f_psf.addRow(self.tr("Spike Boost:"), self.color_boost)
        f_psf.addRow(self.tr("PSF Blur Sigma:"), self.blur_sigma)

        # --- Actions ---
        row_actions = QHBoxLayout()
        row_actions.setSpacing(8)
        self.btn_run = QPushButton(self.tr("Generate Spikes"))
        self.btn_run.clicked.connect(self._run)
        self.btn_apply = QPushButton(self.tr("Apply to Active Document"))
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_apply.setEnabled(False)
        self.btn_help = QPushButton(self.tr("Aperture Help"))
        self.btn_help.clicked.connect(self._show_help)
        self.btn_reset = make_reset_button(self)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        row_actions.addWidget(self.btn_run)
        row_actions.addWidget(self.btn_apply)
        row_actions.addWidget(self.btn_reset)
        row_actions.addWidget(self.btn_help)
        row_actions.addStretch(1)
        from setiastro.saspro.shortcuts import ShortcutGrip
        row_actions.addWidget(ShortcutGrip("star_spikes", self))
        row_actions.addWidget(make_help_button("star_spikes", self))

        # --- Status ---
        self.status = QLabel(self.tr("Ready"))
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)

        # assemble left panel
        left_v.addWidget(grp_detect)
        left_v.addWidget(grp_ap)
        left_v.addWidget(grp_psf)
        left_v.addLayout(row_actions)
        left_v.addWidget(self.status)
        left_v.addStretch(1)

        splitter.addWidget(left)

        # ----- RIGHT: preview panel -----
        right = QWidget()
        right_v = QVBoxLayout(right)

        # zoom toolbar
        zrow = QHBoxLayout()
        self.btn_zoom_in = QPushButton(self.tr("Zoom In"))
        self.btn_zoom_out = QPushButton(self.tr("Zoom Out"))
        self.btn_fit = QPushButton(self.tr("Fit to Preview"))
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_fit.clicked.connect(self._fit_to_preview)
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_fit)
        zrow.addStretch(1)
        right_v.addLayout(zrow)

        # graphics scene/view
        self.scene = QGraphicsScene()
        self.view  = PreviewView()
        self.view.setScene(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setMinimumSize(600, 450)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)
        right_v.addWidget(self.view, 1)

        splitter.addWidget(right)

        # make preview side bigger by default
        splitter.setStretchFactor(0, 0)  # left
        splitter.setStretchFactor(1, 1)  # right
        splitter.setSizes([360, 900])

        # top-level layout contains just the splitter
        root = QVBoxLayout(self)
        root.addWidget(splitter)

        # init pupil visibility
        self._toggle_pupil(False)

        # zoom state
        self._zoom = 1.0
        self._fit_mode = True  # start fitted

    def _toggle_pupil(self, jwst: bool):
        self.pupil_jwst.setText("JWST" if jwst else self.tr("Circular"))
        # hide circular-only params when JWST pupil is used
        for w in (self.num_vanes, self.vane_width, self.obstruction, self.radius):
            w.setVisible(not jwst)

    # ---------- active document resolution ----------
    def _resolve_active_doc(self):
        """Resolve the document behind the currently active view, ROI-aware.

        Goes through the MDI rather than DocManager's cached active document,
        which is only written on subWindowActivated and can be stale or None
        while an image is clearly on screen.
        """
        mw = getattr(self, "_main", None)
        dm = self.docman or getattr(mw, "doc_manager", None) or getattr(mw, "docman", None)

        try:
            mdi = getattr(mw, "mdi", None)
            sw = mdi.activeSubWindow() if mdi is not None else None
            view = sw.widget() if sw is not None else None
            if view is not None:
                if dm is not None and hasattr(dm, "get_document_for_view"):
                    try:
                        d = dm.get_document_for_view(view)
                        if d is not None and getattr(d, "image", None) is not None:
                            return d
                    except Exception:
                        pass
                d = getattr(view, "document", None) or getattr(view, "doc", None)
                if d is not None and getattr(d, "image", None) is not None:
                    return d
        except Exception:
            pass

        try:
            if dm is not None and hasattr(dm, "get_active_document"):
                d = dm.get_active_document()
                if d is not None and getattr(d, "image", None) is not None:
                    return d
        except Exception:
            pass

        return None

    @staticmethod
    def _doc_label(doc) -> str:
        if doc is None:
            return _tr("no image")
        try:
            n = doc.display_name()
            if n:
                return str(n)
        except Exception:
            pass
        return getattr(doc, "name", None) or _tr("image")

    def _hook_active_doc_following(self):
        if not self._follow_active:
            return
        sig = getattr(self._main, "currentDocumentChanged", None)
        if sig is None:
            return
        try:
            sig.connect(self._on_active_doc_changed)
            self._follow_conn = True
        except Exception:
            self._follow_conn = False

    def _on_active_doc_changed(self, doc):
        # Mid-run, _run_impl owns self.doc; retargeting here would reload
        # _img_src under it and clear a result it's about to write.
        if not self._follow_active or self._busy:
            return
        if doc is None or getattr(doc, "image", None) is None:
            return
        if doc is self.doc:
            return
        self.doc = doc
        self._load_active_image()
        # Previous result belongs to the previous image.
        self.final_image = None
        self._result_doc = None
        self.btn_apply.setEnabled(False)

    def closeEvent(self, ev):
        if self._busy:
            # Mid-run (the work is on a worker thread). Hide now, close for
            # real once the run unwinds; the worker stops at its next step.
            self._pending_close = True
            if self._cancel_evt is not None:
                self._cancel_evt.set()
            try:
                self.hide()
            except Exception:
                pass
            ev.ignore()
            return
        try:
            if self._follow_conn:
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = False
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(ev)

    def _release_memory(self):
        """Drop source/result buffers and preview pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self.final_image = None
        self._img_src = None
        self._img_src_doc = None
        self._result_doc = None
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

    # ---------- data/preset ----------
    def _load_active_image(self):
        # Re-resolve rather than staying permanently broken: the old code
        # returned here with self.doc == None and never tried again, so a bad
        # moment at construction killed the dialog for its whole lifetime.
        if self._follow_active and (not self.doc or getattr(self.doc, "image", None) is None):
            d = self._resolve_active_doc()
            if d is not None:
                self.doc = d

        if not self.doc or getattr(self.doc, "image", None) is None:
            self._img_src = None
            self._img_src_doc = None
            self.status.setText(self.tr("No active image."))
            return
        arr = np.asarray(self.doc.image)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        # strip alpha
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        # keep within [0..1] for the math we use
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(arr.max()) if arr.size else 1.0
            if mx > 1.0:
                arr = arr / (65535.0 if mx > 5.0 else mx)
        self._img_src = np.clip(arr, 0.0, 1.0)
        self._img_src_doc = self.doc
        self.status.setText(self.tr("Ready — {0}").format(self._doc_label(self.doc)))

    def apply_preset(self, p: dict):
        if not p:
            return
        self.flux_min.setValue(float(p.get("flux_min", self.flux_min.value())))
        self.advanced["flux_max"]   = float(p.get("flux_max", self.advanced["flux_max"]))
        self.advanced["bscale_min"] = float(p.get("bscale_min", self.advanced["bscale_min"]))
        self.advanced["bscale_max"] = float(p.get("bscale_max", self.advanced["bscale_max"]))
        self.advanced["shrink_min"] = float(p.get("shrink_min", self.advanced["shrink_min"]))
        self.advanced["shrink_max"] = float(p.get("shrink_max", self.advanced["shrink_max"]))
        self.advanced["detect_thresh"] = float(p.get("detect_thresh", self.advanced["detect_thresh"]))
        self.detect_thresh.setValue(float(self.advanced["detect_thresh"]))  # reflect in UI
        self.radius.setValue(float(p.get("radius", self.radius.value())))
        self.obstruction.setValue(float(p.get("obstruction", self.obstruction.value())))
        self.num_vanes.setValue(int(p.get("num_vanes", self.num_vanes.value())))
        self.vane_width.setValue(float(p.get("vane_width", self.vane_width.value())))
        self.rotation.setValue(float(p.get("rotation", self.rotation.value())))
        self.color_boost.setValue(float(p.get("color_boost", self.color_boost.value())))
        self.blur_sigma.setValue(float(p.get("blur_sigma", self.blur_sigma.value())))
        self.pupil_jwst.setChecked(bool(p.get("jwst", self.pupil_jwst.isChecked())))

    def _reset_to_factory(self):
        """Every spike setting back to its factory default (also the
        preset-only advanced values). The image and any result are kept."""
        if self._busy:
            return
        self.advanced = dict(self._factory_advanced)
        restore_controls(self._factory)

    # ---------- core ----------
    def _run(self):
        """Generate. The work runs on worker threads (_SpikesJob): detect,
        then (after the "too many stars" question, if it comes up) render.
        Only SEP's own calls hold the GIL, and _detect_stars keeps each
        short. A click mid-run is queued: once the run lands it runs again
        with the settings as they are then. A close mid-run hides the
        window and closes it for real when the run unwinds.
        """
        if self._busy:
            self._rerun_pending = True
            self.status.setText(self.tr("Working… will run again with the current settings."))
            return
        self._busy = True
        try:
            started = self._start_run()
        except Exception:
            self._finish_run()
            raise
        if not started:
            self._finish_run()

    def _start_run(self) -> bool:
        # Resolve on every run so the dialog follows view switches instead of
        # being frozen to whatever was active when it opened.
        if self._follow_active:
            d = self._resolve_active_doc()
            if d is not None:
                self.doc = d

        if self._img_src is None or self._img_src_doc is not self.doc:
            self._load_active_image()
        if self._img_src is None:
            QMessageBox.information(self, self.tr("Diffraction Spikes"), self.tr("No active image."))
            return False

        # deps check
        if sep is None:
            QMessageBox.critical(self, self.tr("Missing Dependency"), self.tr("python-sep is required for star detection."))
            return False
        if gaussian_filter is None or ndi is None:
            QMessageBox.critical(self, self.tr("Missing Dependency"), self.tr("scipy.ndimage is required."))
            return False

        # Capture the target for the whole run: a view switch mid-run must not
        # move the mask lookup or the Apply that follow onto another image.
        thresh = float(self.detect_thresh.value())
        self._run_ctx = {"doc": self.doc, "img": self._img_src, "thresh": thresh,
                         "flux_min": float(self.flux_min.value()), "tries": 0}
        self.status.setText(self.tr("Detecting stars…"))
        self._launch(self._detect_job, self._on_detected, img=self._img_src,
                     thresh=thresh, flux_min=self._run_ctx["flux_min"])
        return True

    def _launch(self, fn, slot, **kwargs):
        self._cancel_evt = threading.Event()
        job = _SpikesJob(fn, kwargs, self._cancel_evt)
        job.result.connect(slot)
        job.progress.connect(self.status.setText)
        self._job = job
        start_owned_thread(job)

    def _finish_run(self):
        self._busy = False
        self._run_ctx = None
        if self._pending_close:
            self._pending_close = False
            self.close()
            return
        if self._rerun_pending:
            self._rerun_pending = False
            self._run()

    def _run_dropped(self, res) -> bool:
        """True (and the run is finished) when this step's result must be
        dropped: the run was cancelled, or the dialog is closing."""
        if self._run_ctx is None or res.get("cancelled") or self._pending_close:
            self._finish_run()
            return True
        return False

    @classmethod
    def _detect_job(cls, img, thresh, flux_min, report, cancelled,
                    lin=None, base=None, prep=None):
        """Worker: un-stretch, background, SEP. lin/base/prep from an earlier
        pass of the same run are reused on a re-detect."""
        if lin is None:
            # un-stretch via midtones(0.95) for detection
            if img.ndim == 3:
                lin = img.copy()
                for c in range(3):
                    lin[..., c] = cls._midtones_m(lin[..., c], 0.95)
                base = 0.2126*lin[...,0] + 0.7152*lin[...,1] + 0.0722*lin[...,2]
            else:
                lin = cls._midtones_m(img, 0.95)
                base = lin
        if cancelled():
            return {"cancelled": True}
        if prep is None:
            prep = cls._detect_prep(base)
        if cancelled():
            return {"cancelled": True}
        out = {"lin": lin, "base": base, "prep": prep}
        try:
            out["stars"], out["sigma"] = cls._detect_stars(
                base, threshold=thresh, flux_min=flux_min, size_min=1.0, prep=prep)
        except Exception as e:
            out["stars"], out["sigma"], out["detect_error"] = [], thresh, str(e)
        return out

    def _on_detected(self, res: dict):
        if self._run_dropped(res):
            return
        if "error" in res:
            QMessageBox.critical(self, self.tr("Diffraction Spikes"), self.tr("Star detection failed:\n{0}").format(res['error']))
            self._finish_run()
            return
        ctx = self._run_ctx
        ctx.update(lin=res["lin"], base=res["base"], prep=res["prep"], sigma=float(res["sigma"]))
        err = res.get("detect_error")
        if err:
            if "internal pixel buffer full" in err:
                QMessageBox.warning(None, self.tr("Star Detection Failed"),
                                    self.tr("Star detection failed: internal pixel buffer full.\n"
                                    "Increase detection threshold or minimum flux."))
            else:
                QMessageBox.critical(None, self.tr("Star Detection Failed"), err)
        stars = res["stars"]
        thresh = ctx["thresh"]

        # interactive guardrail for dense fields
        if len(stars) > self.WARN_LIMIT and ctx["tries"] < self.MAX_AUTO_RETRIES:
            # From the sigma SEP actually ran at: on a field too dense for the
            # requested one, raising only the request would land on the same rung.
            ref = max(thresh, ctx["sigma"])
            suggested = min(50.0, max(ref + 1.0,
                                      ref * (len(stars) / float(self.WARN_LIMIT))**0.5))
            msg = QMessageBox(self)
            msg.setWindowTitle(self.tr("Too Many Stars Detected"))
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(self.tr("{0} stars detected (limit {1}).\n"
                                "Increase the detection threshold to reduce clutter?").format(len(stars), self.WARN_LIMIT))
            raise_btn = msg.addButton(self.tr("Raise to σ={0:.2f}").format(suggested), QMessageBox.ButtonRole.AcceptRole)
            cont_btn  = msg.addButton(self.tr("Continue Anyway"), QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn= msg.addButton(self.tr("Cancel"), QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(raise_btn)
            msg.exec()
            if self._pending_close:          # closed while the question was up
                self._finish_run()
                return

            clicked = msg.clickedButton()
            if clicked is raise_btn:
                ctx["thresh"] = thresh = suggested
                ctx["tries"] += 1
                self.detect_thresh.setValue(thresh)  # reflect in UI
                self.status.setText(self.tr("Re-detecting stars at σ={0:.2f}…").format(thresh))
                self._launch(self._detect_job, self._on_detected, img=ctx["img"],
                             thresh=thresh, flux_min=ctx["flux_min"],
                             lin=ctx["lin"], base=ctx["base"], prep=ctx["prep"])
                return
            elif clicked is cont_btn:
                pass
            else:  # cancel (cancel_btn, or the box was closed)
                self.status.setText(self.tr("Cancelled."))
                self._finish_run()
                return

        if len(stars) == 0:
            self.status.setText(self.tr("No stars found."))
            QMessageBox.information(self, self.tr("Diffraction Spikes"), self.tr("No stars found above flux_min."))
            self._finish_run()
            return
        self._start_render(stars)

    def _start_render(self, stars):
        ctx = self._run_ctx
        ctx["n_stars"] = len(stars)
        self.status.setText(self.tr("Building pupil/PSFs… ({0} stars)").format(len(stars)))
        jwst = self.pupil_jwst.isChecked()
        if jwst and (cv2 is None or not self.jwstpupil_path):
            QMessageBox.critical(self, self.tr("Missing JWST Pupil"),
                                 self.tr("OpenCV not available or JWST pupil image path missing."))
            self._finish_run()
            return
        p = {"jwst": jwst, "jwst_path": self.jwstpupil_path,
             "radius": self.radius.value(), "obstruction": self.obstruction.value(),
             "vane_width": self.vane_width.value(), "num_vanes": self.num_vanes.value(),
             "rotation": self.rotation.value(), "blur_sigma": self.blur_sigma.value(),
             "color_boost": self.color_boost.value(), "flux_min": self.flux_min.value(),
             **{k: self.advanced[k] for k in ("flux_max", "bscale_min", "bscale_max",
                                               "shrink_min", "shrink_max")}}
        self._launch(self._render_job, self._on_rendered, img=ctx["img"], lin=ctx["lin"],
                     stars=stars, p=p, mask=self._active_mask_array(ctx["doc"]))

    @classmethod
    def _render_job(cls, img, lin, stars, p, mask, report, cancelled):
        """Worker: PSFs, spike synthesis, re-stretch, mask, preview bytes."""
        if p["jwst"]:
            pupil = cls._load_pupil_from_png(p["jwst_path"], size=1024, rotation=p["rotation"])
        else:
            pupil = cls._make_pupil(size=1024,
                                    radius=p["radius"],
                                    obstruction=p["obstruction"],
                                    vane_width=p["vane_width"],
                                    num_vanes=p["num_vanes"],
                                    rotation=p["rotation"])

        psf_r = cls._simulate_psf(pupil, wavelength_scale=1.15, blur_sigma=p["blur_sigma"])
        psf_g = cls._simulate_psf(pupil, wavelength_scale=1.00, blur_sigma=p["blur_sigma"])
        psf_b = cls._simulate_psf(pupil, wavelength_scale=0.85, blur_sigma=p["blur_sigma"])
        if cancelled():
            return {"cancelled": True}

        report(_tr("Synthesizing spikes…"))
        H, W = img.shape[:2]
        canvas = np.zeros((H, W, 3), dtype=np.float32)

        flux_min   = p["flux_min"]
        flux_max   = p["flux_max"]
        bscale_min = p["bscale_min"]
        bscale_max = p["bscale_max"]
        shrink_min = p["shrink_min"]
        shrink_max = p["shrink_max"]
        color_boost = p["color_boost"]

        # NOTE: do *not* `import cv2` here.  A function-body import binds the
        # name as a local for the whole function, which shadows the
        # module-level cv2 (None-guarded at the top of this file).
        _HAS_CV2 = cv2 is not None

        from concurrent.futures import ThreadPoolExecutor
        def star_runner(x, y, flux, a, b):
            brightness = np.clip(np.log1p(flux)/8.0, 0.1, 3.0)
            tile_size  = int(256 + brightness*20)
            tile_size  = min(tile_size, 768)
            tile_size += tile_size % 2
            pad = tile_size // 2

            # Guard against fully out-of-bounds, but allow partial overlaps
            if not (0 <= x < W and 0 <= y < H):
                return None

            # Measure star color
            r_ratio, g_ratio, b_ratio = cls._measure_star_color(img, x, y, sampling_radius=3)

            # Extract PSF tiles
            tile_r = cls._extract_center_tile(psf_r, tile_size) * brightness * r_ratio * color_boost
            tile_g = cls._extract_center_tile(psf_g, tile_size) * brightness * g_ratio * color_boost
            tile_b = cls._extract_center_tile(psf_b, tile_size) * brightness * b_ratio * color_boost

            # Boost/Shrink
            b_scale, s_factor = cls._boost_shrink_from_flux(flux, flux_min, flux_max,
                                                            bscale_min, bscale_max, shrink_min, shrink_max)

            # --- Fast Resize (Zoom) ---
            def _fast_zoom(arr, z):
                if z == 1.0: return arr
                if _HAS_CV2:
                     h, w = arr.shape
                     nw, nh = int(round(w * z)), int(round(h * z))
                     if nw <= 0 or nh <= 0: return np.zeros((2,2), dtype=np.float32)
                     return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
                else:
                    return ndi.zoom(arr, z, order=1)

            final_r = np.clip(_fast_zoom(tile_r * b_scale, 1.0/s_factor), 0.0, 1.0)
            final_g = np.clip(_fast_zoom(tile_g * b_scale, 1.0/s_factor), 0.0, 1.0)
            final_b = np.clip(_fast_zoom(tile_b * b_scale, 1.0/s_factor), 0.0, 1.0)

            # --- Return Patch Data (y, x, patch) ---
            new_h, new_w = final_r.shape

            # Coords of the *patch top-left* relative to the image
            # The star is at (x,y), and the patch center is approx (new_w//2, new_h//2)
            # We want to center the patch on the star.
            py0 = y - (new_h // 2)
            px0 = x - (new_w // 2)

            # Combine channels
            patch = np.dstack((final_r, final_g, final_b)).astype(np.float32)
            return (int(py0), int(px0), patch)

        with ThreadPoolExecutor() as ex:
            futs = [ex.submit(star_runner, *s) for s in stars]
            # Added in star order, not completion order: float addition is not
            # associative where patches overlap, so completion order made the
            # canvas differ in the last bits from one run to the next.
            for f in futs:
                res = f.result()
                if res is None:
                    continue

                py0, px0, patch = res
                ph, pw, _ = patch.shape

                # Calculate intersection with canvas
                y_start = max(0, py0)
                y_end   = min(H, py0 + ph)
                x_start = max(0, px0)
                x_end   = min(W, px0 + pw)

                # If no overlap, skip
                if y_start >= y_end or x_start >= x_end:
                    continue

                # Offsets into the patch
                patch_y_start = y_start - py0
                patch_y_end   = patch_y_start + (y_end - y_start)
                patch_x_start = x_start - px0
                patch_x_end   = patch_x_start + (x_end - x_start)

                # Add to canvas
                canvas[y_start:y_end, x_start:x_end] += patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end]
        if cancelled():
            return {"cancelled": True}

        report(_tr("Compositing…"))
        if lin.ndim == 3:
            spiked_lin = np.clip(lin + canvas, 0, 1)
        else:
            spikes_mono = 0.2126*canvas[...,0] + 0.7152*canvas[...,1] + 0.0722*canvas[...,2]
            spiked_lin = np.clip(lin + spikes_mono, 0, 1)

        #sasprotect by active mask (document-level)
        if spiked_lin.ndim == 3:
            spiked_final = np.empty_like(spiked_lin)
            for c in range(3):
                spiked_final[..., c] = cls._midtones_m(spiked_lin[..., c], 0.05)
        else:
            spiked_final = cls._midtones_m(spiked_lin, 0.05)

        # ---- apply mask AFTER full processing ----
        m = mask
        if m is not None:
            if spiked_final.ndim == 3 and m.ndim == 2:
                m = m[..., None]

            # white = apply effect, black = protect original
            final = np.clip(spiked_final * m + img * (1.0 - m), 0.0, 1.0)
        else:
            final = spiked_final

        arr8 = np.clip(final, 0, 1)
        arr8 = (arr8 * 255.0).astype(np.uint8)
        return {"final": final, "u8": arr8}

    def _on_rendered(self, res: dict):
        if self._run_dropped(res):
            return
        if "error" in res:
            QMessageBox.critical(self, self.tr("Diffraction Spikes"), self.tr("Failed to generate spikes:\n{0}").format(res['error']))
            self._finish_run()
            return
        ctx = self._run_ctx
        run_doc = ctx["doc"]
        self.final_image = res["final"]
        self._result_doc = run_doc
        self._update_preview(res["u8"])
        self.btn_apply.setEnabled(True)
        note = ""
        if ctx["sigma"] != ctx["thresh"]:
            note = self.tr(" {0} stars, detected at σ={1:g}: this field has too "
                           "much faint signal over σ={2:g} to detect it quickly.").format(
                ctx['n_stars'], ctx['sigma'], ctx['thresh'])
        self.status.setText(self.tr("Done — {0}.{1}").format(self._doc_label(run_doc), note))
        self._finish_run()

    def _apply_to_doc(self):
        if self.final_image is None:
            QMessageBox.information(self, self.tr("Diffraction Spikes"), self.tr("Nothing to apply yet."))
            return

        # Target the document the spikes were computed from. apply_edit_to_active()
        # would land them on whatever view the user clicked into since Generate.
        doc = self._result_doc or self.doc
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, self.tr("Diffraction Spikes"),
                                self.tr("The target document is no longer available."))
            return

        out = np.asarray(self.final_image, dtype=np.float32)
        try:
            if hasattr(doc, "apply_edit"):
                doc.apply_edit(
                    out,
                    metadata={"step_name": "Diffraction Spikes",
                              "command_id": "star_spikes"},
                    step_name="Diffraction Spikes",
                )
            elif hasattr(doc, "set_image"):
                doc.set_image(out, step_name="Diffraction Spikes")
            elif self.docman is not None and hasattr(self.docman, "apply_edit_to_active"):
                self.docman.apply_edit_to_active(out, step_name="Diffraction Spikes")
            else:
                doc.image = out
        except Exception as e:
            QMessageBox.critical(self, self.tr("Diffraction Spikes"), self.tr("Failed to apply:\n{0}").format(e))
            return

        self.status.setText(self.tr("Applied to {0}.").format(self._doc_label(doc)))
        # keep dialog open so user can tweak more if desired

    # ---------- helpers ----------
    def _update_preview(self, arr8):
        """Show the uint8 preview the render worker made from final_image."""
        if arr8.ndim == 2:
            h, w = arr8.shape
            qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = arr8.shape
            qimg = QImage(arr8.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(self.pix.boundingRect())
        # keep current zoom mode
        self._apply_zoom()

    def _show_help(self):
        if not self.aperture_help_path:
            QMessageBox.information(self, self.tr("Aperture Help"), self.tr("No help image configured."))
            return
        pm = QPixmap(self.aperture_help_path)
        if pm.isNull():
            QMessageBox.critical(self, self.tr("Aperture Help"), self.tr("Failed to load help image."))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Aperture Help"))
        v = QVBoxLayout(dlg)
        lab = QLabel()
        lab.setPixmap(pm)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lab)
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.resize(480, 480)
        dlg.show()

    # ----- math from SASv2, adapted -----
    @staticmethod
    def _midtones_m(x, m):
        # The formula over the whole array, then the three special values.
        # Same float32 operations element by element as gathering the
        # "other" pixels first, so byte-identical, at half the passes.
        # (x==0, x==1 and |x-m|<eps are disjoint for m in (1e-7, 1-1e-7).)
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        den = (2.0*m - 1.0) * x
        den -= m
        den += 1e-12
        out = (m - 1.0) * x
        out /= den
        np.clip(out, 0, 1, out=out)
        out[x == 0] = 0.0
        out[x == 1] = 1.0
        out[np.abs(x - m) < 1e-7] = 0.5
        return out

    @staticmethod
    def _make_pupil(size=512, radius=100, obstruction=0.3, vane_width=2, num_vanes=4, rotation=0):
        y, x = np.indices((size, size)) - size // 2
        r = np.sqrt(x**2 + y**2)
        pupil = (r <= radius).astype(np.float32)
        pupil[r < radius * obstruction] = 0.0
        if num_vanes >= 2:
            rot = np.deg2rad(rotation)
            for angle in np.linspace(0, np.pi, num_vanes, endpoint=False) + rot:
                xp = x * np.cos(angle) + y * np.sin(angle)
                vane = np.abs(xp) < vane_width
                pupil[vane] = 0.0
        return pupil

    @staticmethod
    def _load_pupil_from_png(filepath, size=1024, rotation=0.0):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from {filepath}")
        img = img.astype(np.float32) / 255.0
        if img.shape != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        if abs(rotation) > 1e-3:
            center = (size // 2, size // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img

    @staticmethod
    def _simulate_psf(pupil, wavelength_scale=1.0, blur_sigma=1.0):
        # Module-level cv2 (None when OpenCV is unavailable).  Keep this a
        # plain lookup -- a function-body import would shadow it as a local.
        has_cv2 = cv2 is not None

        if has_cv2:
            # Gaussian blur on pupil
            # kernel size usually ~6*sigma, must be odd
            k_pupil = int(math.ceil(6 * (0.1 * wavelength_scale))) | 1
            sp = cv2.GaussianBlur(pupil, (k_pupil, k_pupil), 0.1 * wavelength_scale)
        else:
            sp = gaussian_filter(pupil, sigma=0.1 * wavelength_scale)
        
        fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sp)))
        intensity = np.abs(fft)**2
        intensity /= (intensity.max() + 1e-8)
        
        if has_cv2 and blur_sigma > 0:
            k_blur = int(math.ceil(6 * blur_sigma)) | 1
            blurred = cv2.GaussianBlur(intensity, (k_blur, k_blur), blur_sigma)
        else:
            blurred = gaussian_filter(intensity, sigma=blur_sigma)
            
        psf = blurred / max(blurred.max(), 1e-8)
        
        if wavelength_scale != 1.0:
            if has_cv2:
                h, w = psf.shape
                # Zoom uses size, NOT scale factor in resize(..., dsize=(w,h))
                # wavelength_scale > 1 => zoom in => crop middle? or simply scale?
                # The original used ndi.zoom(psf, zoom=wavelength_scale).
                # New size:
                nw, nh = int(round(w * wavelength_scale)), int(round(h * wavelength_scale))
                if nw > 0 and nh > 0:
                    scaled = cv2.resize(psf, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    # We might need to crop back to original size or pad?
                    # ndi.zoom changes the array size.
                    # The simulator seems to assume we handle whatever size comes out?
                    # Let's check _extract_center_tile usage.
                    psf = scaled
            else:
                 psf = ndi.zoom(psf, zoom=wavelength_scale, order=1)
            
            psf /= psf.max() + 1e-12
        return psf

    @staticmethod
    def _extract_center_tile(psf, tile_size):
        c = psf.shape[0]//2
        h = tile_size//2
        y0 = max(0, c-h); x0 = max(0, c-h)
        y1 = y0 + tile_size; x1 = x0 + tile_size
        cropped = psf[y0:y1, x0:x1]
        if cropped.shape != (tile_size, tile_size):
            out = np.zeros((tile_size, tile_size), dtype=np.float32)
            ph, pw = cropped.shape
            out[:ph, :pw] = cropped
            return out
        return cropped

    @staticmethod
    def _detect_prep(image):
        """Background-subtracted detection image and its global RMS."""
        data = image.astype(np.float32)
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        return data_sub, bkg.globalrms

    @classmethod
    def _affordable_sigma(cls, data_sub, err, threshold):
        """threshold when at most DETECT_BUDGET_PX pixels lie over it; else the
        lowest DETECT_LADDER rung above it that is (the top rung if none is)."""
        def over(s):
            return int(np.count_nonzero(data_sub > s * err))
        if over(threshold) <= cls.DETECT_BUDGET_PX:
            return threshold
        rungs = [s for s in cls.DETECT_LADDER if s > threshold]   # descending
        for s in reversed(rungs):
            if over(s) <= cls.DETECT_BUDGET_PX:
                return s
        return rungs[0] if rungs else threshold

    @classmethod
    def _detect_stars(cls, image, threshold=5.0, flux_min=30.0, size_min=1.0, prep=None):
        """Stars to spike, [(x, y, flux, a, b), ...], and the sigma SEP ran at.
        Raises when SEP fails. Runs on the worker thread: no Qt here.

        At the requested threshold whenever it is affordable: exactly the
        old single sep.extract call. A field with more than
        DETECT_BUDGET_PX pixels over it is detected at the lowest affordable
        DETECT_LADDER rung instead (seconds, not minutes): what that drops
        is faint signal, mostly nebula fragments, and fluxes are measured
        inside a tighter isophote, so they come out somewhat lower.
        """
        data_sub, err_val = prep if prep is not None else cls._detect_prep(image)
        sigma = cls._affordable_sigma(data_sub, err_val, threshold)
        objects = sep.extract(data_sub, sigma, err=err_val)
        stars = []
        for obj in objects:
            flux = obj['flux']; a = obj['a']; b = obj['b']
            if flux >= flux_min and max(a,b) >= size_min:
                stars.append((int(obj['x']), int(obj['y']), float(flux), float(a), float(b)))
        return stars, sigma

    # _shrink_and_boost removed (replaced by inline _fast_zoom for performance)

    @staticmethod
    def _boost_shrink_from_flux(flux, flux_min, flux_max, bmin, bmax, smin, smax):
        f = np.clip(flux, flux_min, flux_max)
        alpha = 0.0 if flux_max <= flux_min else (f - flux_min) / (flux_max - flux_min)
        bscale = bmin + alpha * (bmax - bmin)
        shrink = smax - alpha * (smax - smin)
        return float(bscale), float(shrink)

    @staticmethod
    def _measure_star_color(img_color, x, y, sampling_radius=20):
        if img_color.ndim == 2:
            return (1.0, 1.0, 1.0)
        H, W, C = img_color.shape
        if C != 3:
            return (1.0, 1.0, 1.0)
        x0 = max(0, int(x - sampling_radius)); x1 = min(W, int(x + sampling_radius + 1))
        y0 = max(0, int(y - sampling_radius)); y1 = min(H, int(y + sampling_radius + 1))
        if x1 <= x0 or y1 <= y0:
            return (1.0, 1.0, 1.0)
        patch = img_color[y0:y1, x0:x1, :]
        mean_col = np.mean(patch, axis=(0, 1))
        mx = float(np.max(mean_col))
        if mx < 1e-9:
            return (1.0, 1.0, 1.0)
        return (float(mean_col[0]/mx), float(mean_col[1]/mx), float(mean_col[2]/mx))

    @staticmethod
    def _active_mask_array(doc) -> np.ndarray | None:
        if not doc:
            return None
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            return None
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        # keep original where mask == 1.0 (protection mask semantics)
        return a

    def _apply_zoom(self):
        if self._fit_mode:
            self.view.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)
            return
        self.view.resetTransform()
        self.view.scale(self._zoom, self._zoom)

    def _zoom_in(self):
        if self.pix.pixmap().isNull():
            return
        if self._fit_mode:
            self._fit_mode = False
            self._zoom = 1.0
        self._zoom = min(self._zoom * 1.25, 20.0)
        self._apply_zoom()

    def _zoom_out(self):
        if self.pix.pixmap().isNull():
            return
        if self._fit_mode:
            self._fit_mode = False
            self._zoom = 1.0
        self._zoom = max(self._zoom / 1.25, 0.05)
        self._apply_zoom()

    def _fit_to_preview(self):
        if self.pix.pixmap().isNull():
            return
        self._fit_mode = True
        self._apply_zoom()
