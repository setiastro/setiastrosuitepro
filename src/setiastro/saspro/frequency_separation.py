# pro/frequency_separation.py
from __future__ import annotations
import os
import numpy as np

# Optional deps used by the processing threads
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pywt
except Exception:
    pywt = None

from PyQt6.QtCore import (
    Qt, QSize, QPoint, QEvent, QThread, pyqtSignal, QTimer
)
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider,
    QCheckBox, QScrollArea, QToolButton, QStyle, QFileDialog, QMessageBox
)
from PyQt6.QtGui import (
    QPixmap, QImage, QMovie, QCursor, QWheelEvent
)

from .doc_manager import ImageDocument  # add this import
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# ---------------------------- Threads ----------------------------

class FrequencySeperationThread(QThread):
    separation_done = pyqtSignal(np.ndarray, np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, image: np.ndarray, method='Gaussian', radius=10.0, tolerance=50, parent=None):
        super().__init__(parent)
        self.image = image.astype(np.float32, copy=False)
        self.method = method
        self.radius = float(radius)
        self.tolerance = int(tolerance)

    def run(self):
        try:
            if self.image.ndim == 3 and self.image.shape[2] == 3:
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for color frequency separation.")
                bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            else:
                bgr = self.image.copy()

            if self.method == 'Gaussian':
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for Gaussian blur.")
                low_bgr = cv2.GaussianBlur(bgr, (0, 0), self.radius)
            elif self.method == 'Median':
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for median blur.")
                ksize = max(1, int(self.radius) // 2 * 2 + 1)
                low_bgr = cv2.medianBlur(bgr, ksize)
            elif self.method == 'Bilateral':
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for bilateral filter.")
                sigma = 50.0 * (self.tolerance / 100.0)
                d = max(1, int(self.radius))
                low_bgr = cv2.bilateralFilter(bgr, d, sigma, sigma)
            else:
                # fallback
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for Gaussian blur.")
                low_bgr = cv2.GaussianBlur(bgr, (0, 0), self.radius)

            if low_bgr.ndim == 3 and low_bgr.shape[2] == 3:
                low_rgb = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)
            else:
                low_rgb = low_bgr

            high_rgb = self.image - low_rgb  # keep signed HF
            self.separation_done.emit(low_rgb.astype(np.float32), high_rgb.astype(np.float32))
        except Exception as e:
            self.error_signal.emit(str(e))


class HFEnhancementThread(QThread):
    enhancement_done = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        hf_image: np.ndarray,
        enable_scale=True,
        sharpen_scale=1.0,
        enable_wavelet=True,
        wavelet_level=2,
        wavelet_boost=1.2,
        wavelet_name='db2',
        enable_denoise=False,
        denoise_strength=3.0,
        parent=None
    ):
        super().__init__(parent)
        self.hf_image = hf_image.astype(np.float32, copy=False)
        self.enable_scale = bool(enable_scale)
        self.sharpen_scale = float(sharpen_scale)
        self.enable_wavelet = bool(enable_wavelet)
        self.wavelet_level = int(wavelet_level)
        self.wavelet_boost = float(wavelet_boost)
        self.wavelet_name = str(wavelet_name)
        self.enable_denoise = bool(enable_denoise)
        self.denoise_strength = float(denoise_strength)

    def run(self):
        try:
            out = self.hf_image.copy()

            if self.enable_scale:
                out *= self.sharpen_scale

            if self.enable_wavelet:
                if pywt is None:
                    raise RuntimeError("PyWavelets (pywt) is required for wavelet sharpening.")
                out = self._wavelet_sharpen(out, self.wavelet_name, self.wavelet_level, self.wavelet_boost)

            if self.enable_denoise:
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for HF denoise.")
                out = self._denoise_hf(out, self.denoise_strength)

            self.enhancement_done.emit(out.astype(np.float32))
        except Exception as e:
            self.error_signal.emit(str(e))

    def _wavelet_sharpen(self, img, wavelet='db2', level=2, boost=1.2):
        if img.ndim == 3 and img.shape[2] == 3:
            chs = []
            for c in range(3):
                chs.append(self._wavelet_sharpen_mono(img[..., c], wavelet, level, boost))
            return np.stack(chs, axis=-1)
        else:
            return self._wavelet_sharpen_mono(img, wavelet, level, boost)

    def _wavelet_sharpen_mono(self, mono, wavelet, level, boost):
        coeffs = pywt.wavedec2(mono, wavelet=wavelet, level=level, mode='periodization')
        new_coeffs = [coeffs[0]]
        for (cH, cV, cD) in coeffs[1:]:
            new_coeffs.append((cH * boost, cV * boost, cD * boost))
        rec = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='periodization')

        # shape guard
        if rec.shape != mono.shape:
            h, w = mono.shape[:2]
            rec = rec[:h, :w]
        return rec

    def _denoise_hf(self, hf, strength=3.0):
        # Shift to [0..1], denoise, shift back.
        if hf.ndim == 3 and hf.shape[2] == 3:
            bgr = hf[..., ::-1]  # RGB->BGR
            tmp = np.clip(bgr + 0.5, 0, 1)
            u8 = (tmp * 255).astype(np.uint8)
            den = cv2.fastNlMeansDenoisingColored(u8, None, strength, strength, 7, 21)
            f32 = den.astype(np.float32) / 255.0 - 0.5
            return f32[..., ::-1]  # back to RGB
        else:
            tmp = np.clip(hf + 0.5, 0, 1)
            u8 = (tmp * 255).astype(np.uint8)
            den = cv2.fastNlMeansDenoising(u8, None, strength, 7, 21)
            return den.astype(np.float32) / 255.0 - 0.5


# ---------------------------- Widget ----------------------------

class FrequencySeperationTab(QWidget):
    """
    SASpro version:
      - Side-by-side LF/HF previews with synced panning
      - Ctrl+wheel zoom-at-mouse (no wheel-scroll)
      - Gaussian / Median / Bilateral
      - Optional HF scale, wavelet sharpen, denoise
      - Push LF/HF/Combined to new views via DocManager
    """
    def __init__(self, image_manager=None, doc_manager=None, parent=None, document: ImageDocument | None = None):
        super().__init__(parent)
        self.doc_manager = doc_manager or image_manager
        self.doc: ImageDocument | None = document 

        # state
        self.image: np.ndarray | None = None
        self.low_freq_image: np.ndarray | None = None
        self.high_freq_image: np.ndarray | None = None
        self.original_header = None
        self.is_mono = False
        self.filename = None

        self.zoom_factor = 1.0
        self._dragging = False
        self._last_pos: QPoint | None = None
        self._sync_guard = False
        self._hf_history: list[np.ndarray] = []

        # parameters
        self.method = 'Gaussian'
        self.radius = 10.0
        self.tolerance = 50
        self.enable_scale = True
        self.sharpen_scale = 1.0
        self.enable_wavelet = True
        self.wavelet_level = 2
        self.wavelet_boost = 1.2
        self.enable_denoise = False
        self.denoise_strength = 3.0

        self.proc_thread: FrequencySeperationThread | None = None
        self.hf_thread: HFEnhancementThread | None = None
        self._auto_loaded = False
        self._build_ui()

        if self.doc is not None and getattr(self.doc, "image", None) is not None:
            # Preload immediately; avoids any focus/MDI ambiguity
            self.set_image_from_doc(np.asarray(self.doc.image),
                                    getattr(self.doc, "metadata", {}))
            self._auto_loaded = True        


    # ---------------- UI ----------------
    def _build_ui(self):
        main = QHBoxLayout(self)
        self.setLayout(main)

        # left controls
        left = QVBoxLayout()
        left_host = QWidget(self); left_host.setLayout(left); left_host.setFixedWidth(280)

        self.fileLabel = QLabel("", self)
        left.addWidget(self.fileLabel)

        # Method
        left.addWidget(QLabel("Method:", self))
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(['Gaussian', 'Median', 'Bilateral'])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        left.addWidget(self.method_combo)

        # Radius (0..100 mapped to 0.1..100)
        self.radius_label = QLabel("Radius: 10.00", self); left.addWidget(self.radius_label)
        self.radius_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.radius_slider.setRange(1, 100); self.radius_slider.setValue(50)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        left.addWidget(self.radius_slider)

        # Tolerance (for Bilateral only)
        self.tol_label = QLabel("Tolerance: 50%", self); left.addWidget(self.tol_label)
        self.tol_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.tol_slider.setRange(0, 100); self.tol_slider.setValue(50)
        self.tol_slider.valueChanged.connect(self._on_tol_changed)
        left.addWidget(self.tol_slider)
        self._toggle_tol_enabled(False)

        # Apply separation
        btn_apply = QPushButton("Apply – Split HF & LF", self)
        btn_apply.clicked.connect(self._apply_separation)
        left.addWidget(btn_apply)

        left.addWidget(QLabel("<b>HF Enhancements</b>", self))

        # Sharpen scale
        self.cb_scale = QCheckBox("Enable Sharpen Scale", self)
        self.cb_scale.setChecked(True); left.addWidget(self.cb_scale)
        self.scale_label = QLabel("Sharpen Scale: 1.00", self); left.addWidget(self.scale_label)
        self.scale_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scale_slider.setRange(10, 300); self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(lambda v: self._update_scale(v))
        left.addWidget(self.scale_slider)

        # Wavelet
        self.cb_wavelet = QCheckBox("Enable Wavelet Sharpening", self)
        self.cb_wavelet.setChecked(True); left.addWidget(self.cb_wavelet)
        self.wavelet_level_label = QLabel("Wavelet Level: 2", self); left.addWidget(self.wavelet_level_label)
        self.wavelet_level_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.wavelet_level_slider.setRange(1, 5); self.wavelet_level_slider.setValue(2)
        self.wavelet_level_slider.valueChanged.connect(lambda v: self._update_wavelet_level(v))
        left.addWidget(self.wavelet_level_slider)

        self.wavelet_boost_label = QLabel("Wavelet Boost: 1.20", self); left.addWidget(self.wavelet_boost_label)
        self.wavelet_boost_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.wavelet_boost_slider.setRange(50, 300); self.wavelet_boost_slider.setValue(120)
        self.wavelet_boost_slider.valueChanged.connect(lambda v: self._update_wavelet_boost(v))
        left.addWidget(self.wavelet_boost_slider)

        # Denoise
        self.cb_denoise = QCheckBox("Enable HF Denoise", self)
        self.cb_denoise.setChecked(False); left.addWidget(self.cb_denoise)
        self.denoise_label = QLabel("Denoise Strength: 3.00", self); left.addWidget(self.denoise_label)
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.denoise_slider.setRange(0, 50); self.denoise_slider.setValue(30)  # 0..5.0 (we'll /10)
        self.denoise_slider.valueChanged.connect(lambda v: self._update_denoise(v))
        left.addWidget(self.denoise_slider)

        # HF actions row
        row = QHBoxLayout()
        self.btn_apply_hf = QPushButton("Apply HF Enhancements", self)
        self.btn_apply_hf.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.btn_apply_hf.clicked.connect(self._apply_hf_enhancements)
        row.addWidget(self.btn_apply_hf)

        self.btn_undo_hf = QToolButton(self)
        self.btn_undo_hf.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.btn_undo_hf.setToolTip("Undo last HF enhancement")
        self.btn_undo_hf.setEnabled(False)
        self.btn_undo_hf.clicked.connect(self._undo_hf)
        row.addWidget(self.btn_undo_hf)
        left.addLayout(row)

        # Push buttons
        push_row = QHBoxLayout()
        self.btn_push_lf = QPushButton("Push LF", self); self.btn_push_lf.clicked.connect(lambda: self._push_array(self.low_freq_image, "LF"))
        self.btn_push_hf = QPushButton("Push HF", self); self.btn_push_hf.clicked.connect(lambda: self._push_array(self._hf_display_for_push(), "HF"))
        push_row.addWidget(self.btn_push_lf); push_row.addWidget(self.btn_push_hf)
        left.addLayout(push_row)

        #load_row = QHBoxLayout()
        #self.btn_load_hf = QPushButton("Load HF…", self)
        #self.btn_load_hf.clicked.connect(self._load_hf_from_file)
        #load_row.addWidget(self.btn_load_hf)

        #self.btn_load_lf = QPushButton("Load LF…", self)
        #self.btn_load_lf.clicked.connect(self._load_lf_from_file)
        #load_row.addWidget(self.btn_load_lf)

        #left.addLayout(load_row)

        # --- Load from VIEW (active subwindow) ---
        load_row = QHBoxLayout()
        self.btn_load_hf_view = QPushButton("Load HF (View)", self)
        self.btn_load_lf_view = QPushButton("Load LF (View)", self)
        self.btn_load_hf_view.clicked.connect(lambda: self._load_component_from_view("HF"))
        self.btn_load_lf_view.clicked.connect(lambda: self._load_component_from_view("LF"))
        load_row.addWidget(self.btn_load_lf_view)        
        load_row.addWidget(self.btn_load_hf_view)

        left.addLayout(load_row)

        btn_combine_push = QPushButton("Combine HF+LF → Push", self)
        btn_combine_push.clicked.connect(self._combine_and_push)
        left.addWidget(btn_combine_push)



        # Spinner
        self.spinnerLabel = QLabel(self); self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            # if you have a resource_path util in your project, use it; otherwise show text
            from setiastro.saspro.resources import resource_path  # adjust if your helper lives elsewhere
            mov = QMovie(resource_path("spinner.gif"))
            self.spinnerLabel.setMovie(mov)
            self._spinner = mov
        except Exception:
            self.spinnerLabel.setText("Processing…")
            self._spinner = None
        self.spinnerLabel.hide()
        left.addWidget(self.spinnerLabel)

        main.addWidget(left_host, 0)

        # right previews
        right = QVBoxLayout()
        top_row = QHBoxLayout()
        top_row.addStretch(1)

        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_at_pair(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_at_pair(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        top_row.addWidget(self.btn_zoom_in)
        top_row.addWidget(self.btn_zoom_out)
        top_row.addWidget(self.btn_fit)

        right.addLayout(top_row)


        # two scroll areas
        self.scrollHF = QScrollArea(self); self.scrollHF.setWidgetResizable(False); self.scrollHF.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scrollLF = QScrollArea(self); self.scrollLF.setWidgetResizable(False); self.scrollLF.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.labelHF = QLabel("High Frequency", self); self.labelHF.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.labelLF = QLabel("Low Frequency", self); self.labelLF.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.scrollHF.setWidget(self.labelHF)
        self.scrollLF.setWidget(self.labelLF)

        # install filters to support ctrl+wheel & drag pan, and to suppress wheel scrolling
        for w in (self.labelHF, self.scrollHF, self.scrollHF.viewport(),

                  self.labelLF, self.scrollLF, self.scrollLF.viewport(),
                ):
            w.installEventFilter(self)

        row_previews = QHBoxLayout()
        row_previews.addWidget(self.scrollHF, 1)
        row_previews.addWidget(self.scrollLF, 1)
        right.addLayout(row_previews, 1)

        right_host = QWidget(self); right_host.setLayout(right)
        main.addWidget(right_host, 1)

    def _try_autoload_active(self) -> bool:
        # 1) DocManager paths
        dm = self.doc_manager
        doc = None
        if dm is not None:
            # common names first
            for name in ("active_document", "current_document", "document"):
                doc = getattr(dm, name, None)
                if callable(doc):
                    doc = doc()
                if doc is not None:
                    break
            # sometimes the active subwindow is exposed
            if doc is None:
                sw = getattr(dm, "active_subwindow", None)
                if sw is not None:
                    doc = getattr(sw, "document", None)

        # 2) Fallback: sniff the main window’s active ImageSubWindow
        if doc is None:
            mw = self._find_main_window()
            if mw is not None:
                try:
                    from setiastro.saspro.subwindow import ImageSubWindow
                    subs = mw.findChildren(ImageSubWindow)
                    pick = None
                    for s in subs:
                        if s.isActiveWindow() or s.hasFocus():
                            pick = s; break
                    if pick is None and subs:
                        pick = subs[0]
                    if pick is not None:
                        doc = getattr(pick, "document", None)
                except Exception:
                    pass

        img = getattr(doc, "image", None) if doc is not None else None
        md  = getattr(doc, "metadata", {}) if doc is not None else {}
        if img is not None:
            self.set_image_from_doc(img, md)
            return True
        return False

    def _get_active_document(self, strict: bool = False) -> ImageDocument | None:
        """
        Try to get the currently active ImageDocument from the MDI.
        If strict=True: do NOT fall back to the most-recent DocManager doc.
        """
        # 1) MDI active subwindow
        mw = self._find_main_window()
        try:
            if mw and hasattr(mw, "mdi"):
                sub = mw.mdi.activeSubWindow()
                if sub:
                    w = sub.widget()
                    doc = getattr(w, "document", None)
                    if isinstance(doc, ImageDocument):
                        return doc
                # a softer fallback inside MDI only (still 'strict' to MDI)
                subs = getattr(mw.mdi, "subWindowList", lambda: [])()
                if subs:
                    # top of stacking order is usually most recent active
                    w = subs[0].widget()
                    doc = getattr(w, "document", None)
                    if isinstance(doc, ImageDocument):
                        return doc
        except Exception:
            pass

        if strict:
            return None  # ⬅️ don’t wander to “last-created” when strict

        # 2) Non-strict fallback: last opened doc in DocManager (as before)
        dm = self.doc_manager
        try:
            docs = getattr(dm, "_docs", None)
            if docs and len(docs) > 0 and isinstance(docs[-1], ImageDocument):
                return docs[-1]
        except Exception:
            pass
        return None


    def _use_doc_image_as(self, target: str):
        """
        Load image from *another* open view and assign to HF/LF.
        target: 'HF' or 'LF'
        """
        doc = self._get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.warning(self, "From View", "No active view found with an image.")
            return

        # If this dialog was opened for an active document, it might be the same doc.
        # That's ok—user can still use its image as HF/LF if they want.

        ref = self._ref_shape()  # shape we want to match (base image or available HF/LF)
        try:
            imgc = self._coerce_to_ref(np.asarray(doc.image), ref)
        except Exception as e:
            QMessageBox.critical(self, "From View", f"Shape/channel mismatch:\n{e}")
            return

        if self.image is None and self.low_freq_image is None and self.high_freq_image is None:
            # adopt this as the reference image (so future loads coerce to this)
            self.set_image_from_doc(imgc, getattr(doc, "metadata", {}))

        if target == "HF":
            self.high_freq_image = imgc.astype(np.float32, copy=False)
            self.labelHF.setText(f"HF ← {doc.display_name()}")
        else:
            self.low_freq_image = imgc.astype(np.float32, copy=False)
            self.labelLF.setText(f"LF ← {doc.display_name()}")

        self._update_previews()

    def _load_hf_from_view(self):
        self._use_doc_image_as("HF")

    def _load_lf_from_view(self):
        self._use_doc_image_as("LF")

    def _collect_open_documents(self) -> list[tuple[str, object]]:
        """
        Returns [(display_name, ImageDocument), ...] for all known open docs.
        Tries DocManager first; falls back to scanning MDI subwindows.
        Active view (if found) is placed first.
        """
        items: list[tuple[str, object]] = []
        active_doc = None

        # Try to get active from main window
        mw = self._find_main_window()
        if mw and hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
            try:
                active_widget = mw.mdi.activeSubWindow().widget()
                active_doc = getattr(active_widget, "document", None)
            except Exception:
                active_doc = None

        # Prefer DocManager list
        dm = self.doc_manager
        docs = []
        if dm is not None:
            for attr in ("documents", "all_documents", "_docs"):
                d = getattr(dm, attr, None)
                if d:
                    docs = list(d)
                    break

        # If no doc list, scan subwindows
        if not docs and mw is not None:
            try:
                from setiastro.saspro.subwindow import ImageSubWindow
                subs = mw.findChildren(ImageSubWindow)
                for s in subs:
                    doc = getattr(s, "document", None)
                    if doc:
                        docs.append(doc)
            except Exception:
                pass

        # Build names
        def _name_for(doc):
            name = None
            # ImageDocument has display_name(); metadata may have display_name/file_path
            for cand in ("display_name",):
                if hasattr(doc, cand) and callable(getattr(doc, cand)):
                    try:
                        name = getattr(doc, cand)()
                    except Exception:
                        name = None
            if not name:
                md = getattr(doc, "metadata", {}) or {}
                name = md.get("display_name") or md.get("file_path") or "Untitled"
                import os
                if isinstance(name, str):
                    name = os.path.basename(name)
            return name

        # Put active first
        if active_doc and active_doc in docs:
            items.append((f"★ { _name_for(active_doc) } (active)", active_doc))
            docs = [d for d in docs if d is not active_doc]

        for d in docs:
            items.append((_name_for(d), d))

        return items

    def _select_document_via_dropdown(self, which: str | None = None) -> object | None:
        items = self._collect_open_documents()
        if not items:
            QMessageBox.information(self, f"Select View for {which or ''}".strip(),
                                    "No open views/documents found.")
            return None
        dlg = SelectViewDialog(self, items, which=which)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg.selected_doc()
        return None


    def _image_from_doc(self, doc) -> np.ndarray | None:
        """
        Extract float32 image from an ImageDocument-like object.
        For integer sources, scale into [0..1] using width-based heuristic.
        """
        try:
            img = getattr(doc, "image", None)
            if img is None:
                return None
            arr = np.asarray(img)
            if arr.dtype == np.float32:
                return arr
            if np.issubdtype(arr.dtype, np.floating):
                return arr.astype(np.float32, copy=False)
            # Integer → normalize to [0..1]
            scale = 65535.0 if (arr.dtype.itemsize >= 2) else 255.0
            return (arr.astype(np.float32) / scale)
        except Exception as e:
            QMessageBox.warning(self, "Load from View", f"Could not read image from view:\n{e}")
            return None

    def _load_component_from_view(self, which: str):
        """
        which ∈ {"HF", "LF"}
        """
        doc = self._select_document_via_dropdown(which)
        if not doc:
            return
        arr = self._image_from_doc(doc)
        if arr is None:
            return

        # Assign and update preview
        if which.upper() == "HF":
            self.high_freq_image = arr.astype(np.float32, copy=False)
        else:
            self.low_freq_image = arr.astype(np.float32, copy=False)

        # Warn on dimensional mismatch (combine needs same shape)
        if (self.low_freq_image is not None and self.high_freq_image is not None and
            self.low_freq_image.shape != self.high_freq_image.shape):
            QMessageBox.warning(
                self, "Dimension Mismatch",
                "Loaded image dimensions do not match the other component.\n"
                "You can still view/edit, but Combine requires matching sizes."
            )

        self._update_previews()


    def _ref_shape(self):
        """
        Returns a reference shape to coerce incoming HF/LF to:
        - Prefer the main image's shape
        - Else prefer whichever of LF/HF exists
        - Else None (no constraint yet)
        """
        if isinstance(self.image, np.ndarray):
            return self.image.shape
        if isinstance(self.low_freq_image, np.ndarray):
            return self.low_freq_image.shape
        if isinstance(self.high_freq_image, np.ndarray):
            return self.high_freq_image.shape
        return None

    def _coerce_to_ref(self, arr: np.ndarray, ref_shape: tuple[int, ...] | None) -> np.ndarray:
        """
        Try to coerce 'arr' to match ref_shape where possible:
        - If ref is HxW and arr is HxW x3 → convert to mono (mean)
        - If ref is HxW x3 and arr is HxW → tile to 3 channels
        - H/W must match; no resize is performed (we error if they differ)
        """
        a = np.asarray(arr, dtype=np.float32)

        if ref_shape is None:
            return a  # nothing to coerce against yet

        # spatial guard
        if a.ndim == 2:
            ah, aw = a.shape
            ch = 1
        elif a.ndim == 3 and a.shape[2] in (1, 3):
            ah, aw, ch = a.shape[0], a.shape[1], a.shape[2]
        else:
            raise ValueError("Unsupported array shape for HF/LF (expect HxW or HxWx{1,3}).")

        if len(ref_shape) == 2:
            rh, rw = ref_shape
            rch = 1
        else:
            rh, rw, rch = ref_shape

        if (ah != rh) or (aw != rw):
            raise ValueError(f"Image dimensions {ah}x{aw} do not match reference {rh}x{rw}.")

        # channel reconcile
        if rch == 1 and ch == 3:
            # convert RGB→mono (luma or average; we’ll use average)
            a = a.mean(axis=2).astype(np.float32)
        elif rch == 3 and ch == 1:
            a = np.repeat(a[..., None], 3, axis=2).astype(np.float32)

        return a

    def _load_hf_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load High-Frequency Image", "",
            "Images (*.tif *.tiff *.fits *.fit *.png *.xisf);;All Files (*)"
        )
        if not path:
            return
        try:
            img, _, _, _ = legacy_load_image(path)
            if img is None:
                raise RuntimeError("Could not load image.")
            ref = self._ref_shape()
            imgc = self._coerce_to_ref(img, ref)
            self.high_freq_image = imgc.astype(np.float32, copy=False)
            self._update_previews()
            QMessageBox.information(self, "HF Loaded", os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Load HF", f"Failed to load HF:\n{e}")

    def _load_lf_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Low-Frequency Image", "",
            "Images (*.tif *.tiff *.fits *.fit *.png *.xisf);;All Files (*)"
        )
        if not path:
            return
        try:
            img, _, _, _ = legacy_load_image(path)
            if img is None:
                raise RuntimeError("Could not load image.")
            ref = self._ref_shape()
            imgc = self._coerce_to_ref(img, ref)
            self.low_freq_image = imgc.astype(np.float32, copy=False)
            self._update_previews()
            QMessageBox.information(self, "LF Loaded", os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Load LF", f"Failed to load LF:\n{e}")


    # --- NEW: autoload exactly once when the dialog shows ---
    def showEvent(self, e):
        super().showEvent(e)
        if not self._auto_loaded:
            self._auto_loaded = True
            # Strong preference order:
            #   (1) self.doc (injected at construction time)
            #   (2) active MDI doc (strict — no "last-created" fallback)
            src_doc = self.doc or self._get_active_document(strict=True)
            if src_doc is not None and getattr(src_doc, "image", None) is not None:
                try:
                    self.set_image_from_doc(np.asarray(src_doc.image),
                                            getattr(src_doc, "metadata", {}))
                    return
                except Exception:
                    pass

    # --------------- helpers ---------------
    def _toggle_tol_enabled(self, on: bool):
        self.tol_slider.setEnabled(on)
        self.tol_label.setEnabled(on)

    def _map_slider_to_radius(self, pos: int) -> float:
        if pos <= 10:
            t = pos / 10.0
            return 0.1 + t * (1.0 - 0.1)
        elif pos <= 50:
            t = (pos - 10) / 40.0
            return 1.0 + t * (10.0 - 1.0)
        else:
            t = (pos - 50) / 50.0
            return 10.0 + t * (100.0 - 10.0)

    def _update_scale(self, v: int):
        self.sharpen_scale = v / 100.0
        self.scale_label.setText(f"Sharpen Scale: {self.sharpen_scale:.2f}")

    def _update_wavelet_level(self, v: int):
        self.wavelet_level = int(v)
        self.wavelet_level_label.setText(f"Wavelet Level: {self.wavelet_level}")

    def _update_wavelet_boost(self, v: int):
        self.wavelet_boost = v / 100.0
        self.wavelet_boost_label.setText(f"Wavelet Boost: {self.wavelet_boost:.2f}")

    def _update_denoise(self, v: int):
        self.denoise_strength = v / 10.0
        self.denoise_label.setText(f"Denoise Strength: {self.denoise_strength:.2f}")

    # --------------- image I/O hooks ---------------
    def set_image_from_doc(self, image: np.ndarray, metadata: dict | None):
        """Call this from the main app when there’s an active image; or adapt to your ImageManager signal."""
        if image is None:
            return
        self.image = image.astype(np.float32, copy=False)
        md = metadata or {}
        self.filename = md.get("file_path", None)
        self.original_header = md.get("original_header", None)
        self.is_mono = bool(md.get("is_mono", False))
        self.fileLabel.setText(os.path.basename(self.filename) if self.filename else "(from view)")
        # clear outputs
        self.low_freq_image = None
        self.high_freq_image = None
        self._apply_separation()

    # --------------- controls handlers ---------------
    def _on_method_changed(self, text: str):
        self.method = text
        self._toggle_tol_enabled(self.method == 'Bilateral')

    def _on_radius_changed(self, v: int):
        self.radius = self._map_slider_to_radius(v)
        self.radius_label.setText(f"Radius: {self.radius:.2f}")

    def _on_tol_changed(self, v: int):
        self.tolerance = int(v)
        self.tol_label.setText(f"Tolerance: {self.tolerance}%")

    # --------------- processing ---------------
    def _apply_separation(self):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Load or select an image first.")
            return
        self._show_spinner(True)

        if self.proc_thread and self.proc_thread.isRunning():
            self.proc_thread.quit(); self.proc_thread.wait()

        self.proc_thread = FrequencySeperationThread(
            image=self.image, method=self.method, radius=self.radius, tolerance=self.tolerance
        )
        self.proc_thread.separation_done.connect(self._on_sep_done)
        self.proc_thread.error_signal.connect(self._on_sep_error)
        self.proc_thread.start()

    def _on_sep_done(self, lf: np.ndarray, hf: np.ndarray):
        self._show_spinner(False)
        self.low_freq_image = lf.astype(np.float32)
        self.high_freq_image = hf.astype(np.float32)
        self._update_previews()

    def _on_sep_error(self, msg: str):
        self._show_spinner(False)
        QMessageBox.critical(self, "Frequency Separation", msg)

    def _apply_hf_enhancements(self):
        if self.high_freq_image is None:
            QMessageBox.information(self, "HF", "No HF image to enhance.")
            return
        # history for undo
        self._hf_history.append(self.high_freq_image.copy())
        self.btn_undo_hf.setEnabled(True)

        self._show_spinner(True)
        if self.hf_thread and self.hf_thread.isRunning():
            self.hf_thread.quit(); self.hf_thread.wait()

        self.hf_thread = HFEnhancementThread(
            hf_image=self.high_freq_image,
            enable_scale=self.cb_scale.isChecked(),
            sharpen_scale=self.sharpen_scale,
            enable_wavelet=self.cb_wavelet.isChecked(),
            wavelet_level=self.wavelet_level,
            wavelet_boost=self.wavelet_boost,
            enable_denoise=self.cb_denoise.isChecked(),
            denoise_strength=self.denoise_strength
        )
        self.hf_thread.enhancement_done.connect(self._on_hf_done)
        self.hf_thread.error_signal.connect(self._on_hf_error)
        self.hf_thread.start()

    def _on_hf_done(self, new_hf: np.ndarray):
        self._show_spinner(False)
        self.high_freq_image = new_hf.astype(np.float32)
        self._update_previews()

    def _on_hf_error(self, msg: str):
        self._show_spinner(False)
        QMessageBox.critical(self, "HF Enhancements", msg)

    def _undo_hf(self):
        if not self._hf_history:
            return
        self.high_freq_image = self._hf_history.pop()
        self.btn_undo_hf.setEnabled(bool(self._hf_history))
        self._update_previews()

    # --------------- spinner ---------------
    def _show_spinner(self, on: bool):
        if on:
            self.spinnerLabel.show()
            if self._spinner: self._spinner.start()
        else:
            self.spinnerLabel.hide()
            if self._spinner: self._spinner.stop()

    # --------------- preview rendering ---------------
    def _numpy_to_qpix(self, arr: np.ndarray) -> QPixmap:
        a = np.clip(arr, 0, 1)
        if a.ndim == 2:
            a = np.stack([a]*3, axis=-1)
        u8 = (a * 255).astype(np.uint8)
        h, w, ch = u8.shape
        qimg = QImage(u8.data, w, h, w*ch, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _update_previews(self):
        # LF
        if self.low_freq_image is not None:
            pm = self._numpy_to_qpix(self.low_freq_image)
            scaled = pm.scaled(pm.size() * self.zoom_factor,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            self.labelLF.setPixmap(scaled)
            self.labelLF.resize(scaled.size())
        else:
            self.labelLF.setText("Low Frequency"); self.labelLF.resize(self.labelLF.sizeHint())

        # HF (offset +0.5 for view)
        if self.high_freq_image is not None:
            disp = np.clip(self.high_freq_image + 0.5, 0, 1)
            pm = self._numpy_to_qpix(disp)
            scaled = pm.scaled(pm.size() * self.zoom_factor,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            self.labelHF.setPixmap(scaled)
            self.labelHF.resize(scaled.size())
        else:
            self.labelHF.setText("High Frequency"); self.labelHF.resize(self.labelHF.sizeHint())

        # center if smaller than viewport
        QTimer.singleShot(0, self._center_if_fit)

    def _center_if_fit(self):
        for sc, lbl in ((self.scrollHF, self.labelHF), (self.scrollLF, self.labelLF)):
            if lbl.width() <= sc.viewport().width():
                sc.horizontalScrollBar().setValue((sc.horizontalScrollBar().maximum() + sc.horizontalScrollBar().minimum()) // 2)
            if lbl.height() <= sc.viewport().height():
                sc.verticalScrollBar().setValue((sc.verticalScrollBar().maximum() + sc.verticalScrollBar().minimum()) // 2)

    # --------------- zoom/pan (dual scrollareas) ---------------
    def _zoom_at_pair(self, factor: float, anchor_hf_vp: QPoint | None = None, anchor_lf_vp: QPoint | None = None):
        if self.low_freq_image is None and self.high_freq_image is None:
            return

        old = self.zoom_factor
        new = max(0.05, min(8.0, old * factor))
        ratio = new / max(1e-6, old)

        def _center(sc):
            vp = sc.viewport()
            return QPoint(vp.width() // 2, vp.height() // 2)

        if anchor_hf_vp is None: anchor_hf_vp = _center(self.scrollHF)
        if anchor_lf_vp is None: anchor_lf_vp = _center(self.scrollLF)

        HFh, HFv = self.scrollHF.horizontalScrollBar(), self.scrollHF.verticalScrollBar()
        LFh, LFv = self.scrollLF.horizontalScrollBar(), self.scrollLF.verticalScrollBar()
        hf_cx = HFh.value() + anchor_hf_vp.x()
        hf_cy = HFv.value() + anchor_hf_vp.y()
        lf_cx = LFh.value() + anchor_lf_vp.x()
        lf_cy = LFv.value() + anchor_lf_vp.y()

        self.zoom_factor = new
        self._update_previews()  # updates label sizes & scrollbar ranges

        def _restore(sc_area, anchor, cx, cy, lbl):
            hbar, vbar = sc_area.horizontalScrollBar(), sc_area.verticalScrollBar()
            vp = sc_area.viewport()
            if lbl.width() <= vp.width():
                hbar.setValue((hbar.maximum() + hbar.minimum()) // 2)
            else:
                hbar.setValue(int(cx * ratio - anchor.x()))
            if lbl.height() <= vp.height():
                vbar.setValue((vbar.maximum() + vbar.minimum()) // 2)
            else:
                vbar.setValue(int(cy * ratio - anchor.y()))

        _restore(self.scrollHF, anchor_hf_vp, hf_cx, hf_cy, self.labelHF)
        _restore(self.scrollLF, anchor_lf_vp, lf_cx, lf_cy, self.labelLF)


    def _fit_to_preview(self):
        # Fit width to the *smaller* of the two viewports; use LF size if available, else HF
        if self.image is None:
            return
        base_h, base_w = (self.low_freq_image.shape[:2]
                          if self.low_freq_image is not None else
                          (self.high_freq_image.shape[:2] if self.high_freq_image is not None else (None, None)))
        if base_w is None:
            return
        vpw = min(self.scrollHF.viewport().width(), self.scrollLF.viewport().width())
        self.zoom_factor = max(0.05, min(8.0, vpw / float(base_w)))
        self._update_previews()

    # --------------- pushing to new views ---------------
    def _push_array(self, arr: np.ndarray | None, title: str):
        if arr is None:
            QMessageBox.information(self, "Push", f"No {title} image to push.")
            return
        mw = self._find_main_window()
        dm = getattr(mw, "docman", None) or self.doc_manager
        if not dm:
            QMessageBox.critical(self, "UI", "DocManager not available.")
            return
        try:
            if hasattr(dm, "open_array"):
                doc = dm.open_array(arr, metadata={"is_mono": (arr.ndim == 2)}, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=arr, metadata={"is_mono": (arr.ndim == 2)}, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/create_document")
            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(doc)
            else:
                from setiastro.saspro.subwindow import ImageSubWindow
                sw = ImageSubWindow(doc, parent=mw); sw.setWindowTitle(title); sw.show()
        except Exception as e:
            QMessageBox.critical(self, "Push", f"Failed to open new view:\n{e}")

    def _push_to_active(self, img: np.ndarray, step_name: str, extra_md: dict | None = None):
        dm = self.doc_manager
        if dm is None:
            # try to discover from main window just in case
            mw = self.parent() or self.window()
            dm = getattr(mw, "docman", None)
        if dm is None:
            QMessageBox.critical(self, "Error", "DocManager not available; cannot apply to active view.")
            return

        # build metadata (keep what we know so history/exports are consistent)
        md = dict(extra_md or {})
        md.setdefault("original_header", getattr(self, "original_header", None))
        md.setdefault("is_mono", (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1))
        md.setdefault("bit_depth", "32-bit floating point")  # HF/LF math is float32 in this tool

        # try a few common method names so this works with your DocManager
        try:
            if hasattr(dm, "update_active_document"):
                dm.update_active_document(updated_image=img, metadata=md, step_name=step_name)
            elif hasattr(dm, "update_image"):
                dm.update_image(updated_image=img, metadata=md, step_name=step_name)
            elif hasattr(dm, "set_image"):
                # older API; many builds accept step_name here too
                dm.set_image(img, md, step_name=step_name)
            elif hasattr(dm, "apply_edit_to_active"):
                dm.apply_edit_to_active(img, step_name=step_name, metadata=md)
            else:
                raise RuntimeError("DocManager has no known update method")
        except Exception as e:
            QMessageBox.critical(self, "Apply Failed", f"Could not apply result to the active view:\n{e}")
            return


    def _hf_display_for_push(self) -> np.ndarray | None:
        # push the true HF (signed), but clamp for safety into viewable range around 0
        if self.high_freq_image is None:
            return None
        # keep signed HF; app stack supports float32 arrays
        return self.high_freq_image.astype(np.float32, copy=False)

    def _combine_and_push(self):
        if self.low_freq_image is None or self.high_freq_image is None:
            QMessageBox.information(self, "Combine", "LF or HF missing.")
            return

        combined = np.clip(self.low_freq_image + self.high_freq_image, 0.0, 1.0).astype(np.float32)
        step_name = "Frequency Separation (Combine HF+LF)"

        # ✅ Blend with active mask (if any)
        blended, mid, mname, masked = self._blend_with_active_mask(combined)

        # Build metadata
        md = {
            "bit_depth": "32-bit floating point",
            "is_mono": (blended.ndim == 2) or (blended.ndim == 3 and blended.shape[2] == 1),
            "original_header": getattr(self, "original_header", None),
        }
        if masked:
            md.update({
                "masked": True,
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            })

        # Prefer applying to the injected ImageDocument
        if isinstance(self.doc, ImageDocument):
            try:
                self.doc.apply_edit(blended, metadata=md, step_name=step_name)
            except Exception as e:
                QMessageBox.critical(self, "Apply Failed", f"Could not apply to active document:\n{e}")
            return

        # Fallback: push to active via DocManager (still pre-blended)
        self._push_to_active(blended, step_name, extra_md=md)

    # --------------- event filter (wheel + drag pan + sync) ---------------
    def eventFilter(self, obj, ev):
        # -------- Ctrl+Wheel Zoom (safe) --------
        if ev.type() == QEvent.Type.Wheel:
            targets = {self.scrollHF.viewport(), self.labelHF,
                    self.scrollLF.viewport(), self.labelLF}
            if obj in targets:
                # Only zoom when Ctrl is held; otherwise let normal scrolling work
                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    try:
                        dy = ev.pixelDelta().y()
                        if dy == 0:
                            dy = ev.angleDelta().y()
                        factor = 1.25 if dy > 0 else 0.8

                        # Anchor positions (robust mapping child→viewport)
                        if obj is self.labelHF:
                            anchor_hf = self.labelHF.mapTo(
                                self.scrollHF.viewport(), ev.position().toPoint()
                            )
                            anchor_lf = QPoint(
                                self.scrollLF.viewport().width() // 2,
                                self.scrollLF.viewport().height() // 2
                            )
                        elif obj is self.scrollHF.viewport():
                            anchor_hf = ev.position().toPoint()
                            anchor_lf = QPoint(
                                self.scrollLF.viewport().width() // 2,
                                self.scrollLF.viewport().height() // 2
                            )
                        elif obj is self.labelLF:
                            anchor_lf = self.labelLF.mapTo(
                                self.scrollLF.viewport(), ev.position().toPoint()
                            )
                            anchor_hf = QPoint(
                                self.scrollHF.viewport().width() // 2,
                                self.scrollHF.viewport().height() // 2
                            )
                        else:  # obj is self.scrollLF.viewport()
                            anchor_lf = ev.position().toPoint()
                            anchor_hf = QPoint(
                                self.scrollHF.viewport().width() // 2,
                                self.scrollHF.viewport().height() // 2
                            )

                        self._zoom_at_pair(factor, anchor_hf, anchor_lf)
                    except Exception:
                        # If anything goes weird (trackpad/gesture edge cases), center-zoom safely
                        self._zoom_at_pair(1.25 if (ev.angleDelta().y() if hasattr(ev, "angleDelta") else 1) > 0 else 0.8)
                    ev.accept()
                    return True
                # Not Ctrl: let the scroll area do normal scrolling
                return False

        # -------- Drag-pan inside each viewport (sync the other) --------
        if obj in (self.scrollHF.viewport(), self.scrollLF.viewport()):
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self._last_pos = ev.position().toPoint()
                obj.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True
            if ev.type() == QEvent.Type.MouseMove and self._dragging:
                cur = ev.position().toPoint()
                delta = cur - (self._last_pos or cur)
                self._last_pos = cur
                if obj is self.scrollHF.viewport():
                    self._move_scrolls(self.scrollHF, self.scrollLF, delta)
                else:
                    self._move_scrolls(self.scrollLF, self.scrollHF, delta)
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._dragging = False
                self._last_pos = None
                obj.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True

        return super().eventFilter(obj, ev)


    def _move_scrolls(self, src_sc: QScrollArea, dst_sc: QScrollArea, delta):
        self._sync_guard = True
        try:
            sh, sv = src_sc.horizontalScrollBar(), src_sc.verticalScrollBar()
            dh, dv = dst_sc.horizontalScrollBar(), dst_sc.verticalScrollBar()
            sh.setValue(sh.value() - delta.x()); sv.setValue(sv.value() - delta.y())
            dh.setValue(sh.value()); dv.setValue(sv.value())
        finally:
            self._sync_guard = False

    # --------------- utilities ---------------
    def _find_main_window(self):
        w = self
        from PyQt6.QtWidgets import QMainWindow, QApplication
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parentWidget()
        if w: return w
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None


    # ---- MASK HELPERS -------------------------------------------------
    def _doc_for_mask(self):
        """Prefer the dialog-injected doc; else the active MDI doc."""
        return self.doc or self._get_active_document()

    def _active_mask_array(self):
        """
        Return (mask_float01, mask_id, mask_name) or (None, None, None).
        """
        doc = self._doc_for_mask()
        if not doc:
            return None, None, None
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None, None, None
        layer = (getattr(doc, "masks", {}) or {}).get(mid)
        if layer is None:
            return None, None, None

        import numpy as np
        m = np.asarray(getattr(layer, "data", None))
        if m is None or m.size == 0:
            return None, None, None

        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0:
                m /= mx
        return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize via integer indexing."""
        import numpy as np
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw):
            return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _prepare_src_like(self, src, ref):
        """
        Convert document source image to float32 [0..1] and reconcile channels to match ref.
        """
        import numpy as np
        s = np.asarray(src)
        if s.dtype.kind in "ui":
            # assume 16-bit if >=2 bytes, else 8-bit
            scale = float(65535.0 if s.dtype.itemsize >= 2 else 255.0)
            s = s.astype(np.float32) / scale
        elif np.issubdtype(s.dtype, np.floating):
            s = s.astype(np.float32, copy=False)
            mx = float(s.max()) if s.size else 1.0
            if mx > 5.0:
                s = s / mx

        # channel reconcile
        if s.ndim == 2 and ref.ndim == 3:
            s = np.stack([s]*3, axis=-1)
        elif s.ndim == 3 and s.shape[2] == 1 and ref.ndim == 3 and ref.shape[2] == 3:
            s = np.repeat(s, 3, axis=2)
        elif s.ndim == 3 and ref.ndim == 2:
            s = s[..., 0]

        return s.astype(np.float32, copy=False)

    def _blend_with_active_mask(self, processed: np.ndarray):
        """
        Blend processed result with the *current* document image using active mask.
        Returns (blended, mask_id, mask_name, masked_bool).
        If no mask, returns (processed, None, None, False).
        """
        mask, mid, mname = self._active_mask_array()
        if mask is None:
            return processed, None, None, False

        import numpy as np
        out = np.asarray(processed, dtype=np.float32, copy=False)

        doc = self._doc_for_mask()
        src = getattr(doc, "image", None)
        if src is None:
            return processed, mid, mname, True

        srcf = self._prepare_src_like(src, out)
        m = self._resample_mask_if_needed(mask, out.shape[:2])
        if out.ndim == 3:
            m = m[..., None]

        blended = (m * out + (1.0 - m) * srcf).astype(np.float32, copy=False)
        return blended, mid, mname, True
    # ---- /MASK HELPERS ------------------------------------------------


from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QComboBox

class SelectViewDialog(QDialog):
    def __init__(self, parent, items: list[tuple[str, object]],
                 which: str | None = None, title: str | None = None):
        super().__init__(parent)
        # Use a nice default title if none provided
        if title is None:
            title = f"Select View for {which.upper()}" if which else "Select View"
        self.setWindowTitle(title)

        self._items = items
        self.combo = QComboBox(self)
        for name, _doc in items:
            self.combo.addItem(name)

        form = QFormLayout(self)
        if which:
            form.addRow(QLabel(f"Load into: {which.upper()}"))
        form.addRow("View:", self.combo)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def selected_doc(self):
        idx = self.combo.currentIndex()
        return self._items[idx][1] if 0 <= idx < len(self._items) else None
