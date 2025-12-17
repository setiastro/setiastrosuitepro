# pro/tools/star_spikes.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSplitter, QSizePolicy, QWidget, QApplication,
                             QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)

from PyQt6.QtGui import QPixmap, QImage, QPainter
from pro.widgets.themed_buttons import themed_toolbtn

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

class StarSpikesDialogPro(QDialog):
    WARN_LIMIT = 1000
    MAX_AUTO_RETRIES = 2

    def __init__(self, parent=None, doc_manager=None,
                 initial_doc=None,
                 jwstpupil_path: str | None = None,
                 aperture_help_path: str | None = None,
                 spinner_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Diffraction Spikes")
        self.docman = doc_manager
        self.doc = initial_doc or (self.docman.get_active_document() if self.docman else None)
        self.jwstpupil_path = jwstpupil_path
        self.aperture_help_path = aperture_help_path

        self.final_image = None
        self._img_src = None  # float32, 2D or 3D, [0..1]

        # defaults (aligned to your SASv2 tool)
        self.advanced = {
            "flux_max": 300.0, "bscale_min": 10.0, "bscale_max": 30.0,
            "shrink_min": 1.0, "shrink_max": 5.0, "detect_thresh": 5.0,
        }

        self._build_ui()
        self._load_active_image()

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
        grp_detect = QGroupBox("Star Detection")
        f_detect = QFormLayout(grp_detect)
        f_detect.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_detect.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.flux_min = dspin(0.0, 999999.0, 10.0, 30.0, decimals=1)
        self.detect_thresh = dspin(0.5, 50.0, 0.5, float(self.advanced.get("detect_thresh", 5.0)), decimals=2)
        self.detect_thresh.setToolTip("σ threshold for SEP detection (higher = fewer stars).")
        # keep self.advanced in sync if user edits
        self.detect_thresh.valueChanged.connect(lambda v: self.advanced.__setitem__("detect_thresh", float(v)))

        f_detect.addRow("Flux Min:", self.flux_min)
        f_detect.addRow("Detection Threshold (σ):", self.detect_thresh)

        # --- Group: Aperture (Geometry) ---
        grp_ap = QGroupBox("Aperture")
        f_ap = QFormLayout(grp_ap)
        f_ap.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_ap.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.pupil_jwst = QPushButton("Circular")
        self.pupil_jwst.setCheckable(True)
        self.pupil_jwst.setChecked(False)
        self.pupil_jwst.toggled.connect(lambda on: self._toggle_pupil(on))
        self.pupil_jwst.setToolTip("Toggle between circular aperture and JWST pupil image.")
        self.pupil_jwst.setStyleSheet("""
            QPushButton { min-width: 72px; max-width: 72px; min-height: 28px; max-height: 28px;
                          border-radius: 14px; background:#ccc; border:1px solid #999;}
            QPushButton:checked { background:#66bb6a; }
        """)
        f_ap.addRow("Aperture Type:", self.pupil_jwst)

        self.radius      = dspin(1.0, 512.0, 1.0, 128.0, decimals=1)
        self.obstruction = dspin(0.0, 0.99, 0.01, 0.2, decimals=2)
        self.num_vanes   = ispin(2, 8, 1, 4)
        self.vane_width  = dspin(0.0, 50.0, 0.5, 4.0, decimals=2)
        self.rotation    = dspin(0.0, 360.0, 1.0, 0.0, decimals=1)

        f_ap.addRow("Pupil Radius:", self.radius)
        f_ap.addRow("Obstruction:", self.obstruction)
        f_ap.addRow("Number of Vanes:", self.num_vanes)
        f_ap.addRow("Vane Width:", self.vane_width)
        f_ap.addRow("Rotation (deg):", self.rotation)

        # --- Group: PSF & Synthesis ---
        grp_psf = QGroupBox("PSF & Synthesis")
        f_psf = QFormLayout(grp_psf)
        f_psf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_psf.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.color_boost = dspin(0.1, 10.0, 0.1, 1.5, decimals=2)
        self.blur_sigma  = dspin(0.1, 10.0, 0.1, 2.0, decimals=2)

        f_psf.addRow("Spike Boost:", self.color_boost)
        f_psf.addRow("PSF Blur Sigma:", self.blur_sigma)

        # --- Actions ---
        row_actions = QHBoxLayout()
        row_actions.setSpacing(8)
        self.btn_run = QPushButton("Generate Spikes")
        self.btn_run.clicked.connect(self._run)
        self.btn_apply = QPushButton("Apply to Active Document")
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_apply.setEnabled(False)
        self.btn_help = QPushButton("Aperture Help")
        self.btn_help.clicked.connect(self._show_help)
        row_actions.addWidget(self.btn_run)
        row_actions.addWidget(self.btn_apply)
        row_actions.addWidget(self.btn_help)
        row_actions.addStretch(1)

        # --- Status ---
        self.status = QLabel("Ready")
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
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_fit.clicked.connect(self._fit_to_preview)

        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_zoom_in)
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
        self.pupil_jwst.setText("JWST" if jwst else "Circular")
        # hide circular-only params when JWST pupil is used
        for w in (self.num_vanes, self.vane_width, self.obstruction, self.radius):
            w.setVisible(not jwst)

    # ---------- data/preset ----------
    def _load_active_image(self):
        if not self.doc or getattr(self.doc, "image", None) is None:
            self.status.setText("No active image.")
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

    # ---------- core ----------
    def _run(self):
        if self._img_src is None:
            self._load_active_image()
        if self._img_src is None:
            QMessageBox.information(self, "Diffraction Spikes", "No active image.")
            return

        # deps check
        if sep is None:
            QMessageBox.critical(self, "Missing Dependency", "python-sep is required for star detection.")
            return
        if gaussian_filter is None or ndi is None:
            QMessageBox.critical(self, "Missing Dependency", "scipy.ndimage is required.")
            return

        self.status.setText("Detecting stars…")
        QApplication.processEvents()
        img = self._img_src
        # un-stretch via midtones(0.95) for detection
        if img.ndim == 3:
            lin = img.copy()
            for c in range(3):
                lin[..., c] = self._midtones_m(lin[..., c], 0.95)
            base = 0.2126*lin[...,0] + 0.7152*lin[...,1] + 0.0722*lin[...,2]
        else:
            lin = self._midtones_m(img, 0.95)
            base = lin

        # initial detection
        thresh = float(self.detect_thresh.value())
        stars = self._detect_stars(base,
                                   threshold=thresh,
                                   flux_min=self.flux_min.value(),
                                   size_min=1.0)

        # interactive guardrail for dense fields
        tries = 0
        while len(stars) > self.WARN_LIMIT and tries < self.MAX_AUTO_RETRIES:
            suggested = min(50.0, max(thresh + 1.0,
                                      thresh * (len(stars) / float(self.WARN_LIMIT))**0.5))
            msg = QMessageBox(self)
            msg.setWindowTitle("Too Many Stars Detected")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(f"{len(stars)} stars detected (limit {self.WARN_LIMIT}).\n"
                        "Increase the detection threshold to reduce clutter?")
            raise_btn = msg.addButton(f"Raise to σ={suggested:.2f}", QMessageBox.ButtonRole.AcceptRole)
            cont_btn  = msg.addButton("Continue Anyway", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn= msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(raise_btn)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked is raise_btn:
                thresh = suggested
                self.detect_thresh.setValue(thresh)  # reflect in UI
                self.status.setText(f"Re-detecting stars at σ={thresh:.2f}…")
                QApplication.processEvents()
                stars = self._detect_stars(base,
                                           threshold=thresh,
                                           flux_min=self.flux_min.value(),
                                           size_min=1.0)
                tries += 1
                continue
            elif clicked is cont_btn:
                break
            else:  # cancel
                self.status.setText("Cancelled.")
                return

        if len(stars) == 0:
            self.status.setText("No stars found.")
            QMessageBox.information(self, "Diffraction Spikes", "No stars found above flux_min.")
            return

        self.status.setText(f"Building pupil/PSFs… ({len(stars)} stars)")
        QApplication.processEvents()
        if self.pupil_jwst.isChecked():
            if cv2 is None or not self.jwstpupil_path:
                QMessageBox.critical(self, "Missing JWST Pupil",
                                     "OpenCV not available or JWST pupil image path missing.")
                return
            pupil = self._load_pupil_from_png(self.jwstpupil_path, size=1024, rotation=self.rotation.value())
        else:
            pupil = self._make_pupil(size=1024,
                                     radius=self.radius.value(),
                                     obstruction=self.obstruction.value(),
                                     vane_width=self.vane_width.value(),
                                     num_vanes=self.num_vanes.value(),
                                     rotation=self.rotation.value())

        psf_r = self._simulate_psf(pupil, wavelength_scale=1.15, blur_sigma=self.blur_sigma.value())
        psf_g = self._simulate_psf(pupil, wavelength_scale=1.00, blur_sigma=self.blur_sigma.value())
        psf_b = self._simulate_psf(pupil, wavelength_scale=0.85, blur_sigma=self.blur_sigma.value())

        self.status.setText("Synthesizing spikes…")
        QApplication.processEvents()
        H, W = img.shape[:2]
        canvas = np.zeros((H, W, 3), dtype=np.float32)

        flux_max   = self.advanced["flux_max"]
        bscale_min = self.advanced["bscale_min"]
        bscale_max = self.advanced["bscale_max"]
        shrink_min = self.advanced["shrink_min"]
        shrink_max = self.advanced["shrink_max"]
        color_boost = self.color_boost.value()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        def star_runner(x, y, flux, a, b):
            brightness = np.clip(np.log1p(flux)/8.0, 0.1, 3.0)
            tile_size  = int(256 + brightness*20)
            tile_size  = min(tile_size, 768)
            tile_size += tile_size % 2
            pad = tile_size // 2
            if not (pad <= x < W - pad and pad <= y < H - pad):
                return None

            r_ratio, g_ratio, b_ratio = self._measure_star_color(img, x, y, sampling_radius=3)
            tile_r = self._extract_center_tile(psf_r, tile_size) * brightness * r_ratio * color_boost
            tile_g = self._extract_center_tile(psf_g, tile_size) * brightness * g_ratio * color_boost
            tile_b = self._extract_center_tile(psf_b, tile_size) * brightness * b_ratio * color_boost

            b_scale, s_factor = self._boost_shrink_from_flux(flux, self.flux_min.value(), flux_max,
                                                             bscale_min, bscale_max, shrink_min, shrink_max)
            final_r = self._shrink_and_boost(tile_r, b_scale, s_factor)
            final_g = self._shrink_and_boost(tile_g, b_scale, s_factor)
            final_b = self._shrink_and_boost(tile_b, b_scale, s_factor)

            new_size = final_r.shape[0]
            pad_new  = new_size // 2
            y0, y1   = y - pad_new, y - pad_new + new_size
            x0, x1   = x - pad_new, x - pad_new + new_size
            if (y0 < 0 or y1 > H or x0 < 0 or x1 > W):
                return None

            part = np.zeros((H, W, 3), dtype=np.float32)
            part[y0:y1, x0:x1, 0] = final_r
            part[y0:y1, x0:x1, 1] = final_g
            part[y0:y1, x0:x1, 2] = final_b
            return part

        with ThreadPoolExecutor() as ex:
            futs = [ex.submit(star_runner, *s) for s in stars]
            for f in as_completed(futs):
                part = f.result()
                if part is not None:
                    canvas += part

        self.status.setText("Compositing…")
        QApplication.processEvents()
        if lin.ndim == 3:
            spiked_lin = np.clip(lin + canvas, 0, 1)
        else:
            spikes_mono = 0.2126*canvas[...,0] + 0.7152*canvas[...,1] + 0.0722*canvas[...,2]
            spiked_lin = np.clip(lin + spikes_mono, 0, 1)

        # protect by active mask (document-level)
        if spiked_lin.ndim == 3:
            spiked_final = np.empty_like(spiked_lin)
            for c in range(3):
                spiked_final[..., c] = self._midtones_m(spiked_lin[..., c], 0.05)
        else:
            spiked_final = self._midtones_m(spiked_lin, 0.05)

        # ---- apply mask AFTER full processing ----
        m = self._active_mask_array(self.doc)
        if m is not None:
            if spiked_final.ndim == 3 and m.ndim == 2:
                m = m[..., None]

            # white = apply effect, black = protect original
            final = np.clip(spiked_final * m + img * (1.0 - m), 0.0, 1.0)
        else:
            final = spiked_final

        self.final_image = final
        self._update_preview(final)
        self.btn_apply.setEnabled(True)
        self.status.setText("Done.")

    def _apply_to_doc(self):
        if self.final_image is None:
            QMessageBox.information(self, "Diffraction Spikes", "Nothing to apply yet.")
            return
        if not self.docman:
            QMessageBox.warning(self, "No DocManager", "DocManager not available.")
            return
        self.docman.apply_edit_to_active(self.final_image, step_name="Diffraction Spikes")
        self.status.setText("Applied to active document.")
        # keep dialog open so user can tweak more if desired

    # ---------- helpers ----------
    def _update_preview(self, arr):
        arr8 = np.clip(arr, 0, 1)
        arr8 = (arr8 * 255.0).astype(np.uint8)
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
            QMessageBox.information(self, "Aperture Help", "No help image configured.")
            return
        pm = QPixmap(self.aperture_help_path)
        if pm.isNull():
            QMessageBox.critical(self, "Aperture Help", "Failed to load help image.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Aperture Help")
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
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        out = np.zeros_like(x, dtype=np.float32)
        mask0 = (x == 0); out[mask0] = 0.0
        mask1 = (x == 1); out[mask1] = 1.0
        eps = 1e-7
        maskm = (np.abs(x - m) < eps); out[maskm] = 0.5
        mask_oth = ~(mask0 | mask1 | maskm)
        xm = x[mask_oth]
        num = (m - 1.0)*xm
        den = (2.0*m - 1.0)*xm - m
        out[mask_oth] = np.clip(num/(den+1e-12),0,1)
        return out

    def _make_pupil(self, size=512, radius=100, obstruction=0.3, vane_width=2, num_vanes=4, rotation=0):
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

    def _load_pupil_from_png(self, filepath, size=1024, rotation=0.0):
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

    def _simulate_psf(self, pupil, wavelength_scale=1.0, blur_sigma=1.0):
        sp = gaussian_filter(pupil, sigma=0.1 * wavelength_scale)
        fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sp)))
        intensity = np.abs(fft)**2
        intensity /= (intensity.max() + 1e-8)
        blurred = gaussian_filter(intensity, sigma=blur_sigma)
        psf = blurred / max(blurred.max(), 1e-8)
        if wavelength_scale != 1.0:
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
    def _detect_stars(image, threshold=5.0, flux_min=30.0, size_min=1.0):
        data = image.astype(np.float32)
        bkg = sep.Background(data)
        data_sub = data - bkg.back()
        err_val = bkg.globalrms
        try:
            objects = sep.extract(data_sub, threshold, err=err_val)
        except Exception as e:
            if "internal pixel buffer full" in str(e):
                QMessageBox.warning(None, "Star Detection Failed",
                                    "Star detection failed: internal pixel buffer full.\n"
                                    "Increase detection threshold or minimum flux.")
            else:
                QMessageBox.critical(None, "Star Detection Failed", str(e))
            return []
        stars = []
        for obj in objects:
            flux = obj['flux']; a = obj['a']; b = obj['b']
            if flux >= flux_min and max(a,b) >= size_min:
                stars.append((int(obj['x']), int(obj['y']), float(flux), float(a), float(b)))
        return stars

    @staticmethod
    def _shrink_and_boost(tile, brightness_scale=2.0, shrink_factor=1.5):
        tile = np.clip(tile * float(brightness_scale), 0.0, 1.0)
        in_sz = tile.shape[0]
        out_sz = int(in_sz // float(shrink_factor))
        out_sz += out_sz % 2
        if out_sz <= 0: out_sz = 2
        z = out_sz / float(in_sz)
        return np.clip(ndi.zoom(tile, z, order=1), 0.0, 1.0)

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
