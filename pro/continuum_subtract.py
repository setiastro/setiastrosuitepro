# pro/continuum_subtract.py
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
    Qt, QSize, QPoint, QEvent, QThread, pyqtSignal, QTimer, QCoreApplication
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QDialog, QInputDialog, QFileDialog,
    QMessageBox, QCheckBox, QApplication, QMainWindow
)
from PyQt6.QtGui import (
    QPixmap, QImage, QCursor, QWheelEvent
)



from .doc_manager import ImageDocument  # add this import
from legacy.image_manager import load_image as legacy_load_image
from imageops.stretch import stretch_mono_image, stretch_color_image
from imageops.starbasedwhitebalance import apply_star_based_white_balance
from legacy.numba_utils import apply_curves_numba


def apply_curves_adjustment(image, target_median, curves_boost):
    """
    Original signature unchanged, but now uses a Numba helper
    to do the pixel-by-pixel interpolation.

    'image' can be 2D (H,W) or 3D (H,W,3).
    """
    # Build the curve array as before
    curve = [
        [0.0, 0.0],
        [0.5 * target_median, 0.5 * target_median],
        [target_median, target_median],
        [
            (1/4 * (1 - target_median) + target_median),
            np.power((1/4 * (1 - target_median) + target_median), (1 - curves_boost))
        ],
        [
            (3/4 * (1 - target_median) + target_median),
            np.power(np.power((3/4 * (1 - target_median) + target_median), (1 - curves_boost)), (1 - curves_boost))
        ],
        [1.0, 1.0]
    ]
    # Convert to arrays
    xvals = np.array([p[0] for p in curve], dtype=np.float32)
    yvals = np.array([p[1] for p in curve], dtype=np.float32)

    # Ensure 'image' is float32
    image_32 = image.astype(np.float32, copy=False)

    # Now apply the piecewise linear function in Numba
    adjusted_image = apply_curves_numba(image_32, xvals, yvals)
    return adjusted_image

class ContinuumSubtractTab(QWidget):
    def __init__(self, doc_manager, document=None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.doc_manager = doc_manager
        self.initUI()
        self._threads = []
        # — initialize every loadable image to None —
        self.ha_image    = None
        self.sii_image   = None
        self.oiii_image  = None
        self.red_image   = None
        self.green_image = None
        self.osc_image   = None

        self.filename = None
        self.is_mono = True
        self.combined_image = None
        self.processing_thread = None
        self.original_header = None
        self._clickable_images = {}


    def initUI(self):
        self.spinnerLabel = QLabel("")                # starts empty
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerLabel.setStyleSheet("color:#999; font-style:italic;")
        self.spinnerLabel.hide()

        # images (starless)
        self.ha_starless_image    = None
        self.sii_starless_image   = None
        self.oiii_starless_image  = None
        self.red_starless_image   = None
        self.green_starless_image = None
        self.osc_starless_image   = None

        # status
        self.statusLabel = QLabel("")
        self.statusLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout = QVBoxLayout()           # overall vertical: columns + bottom row
        columns_layout = QHBoxLayout()        # holds the three groups

        # — NB group —
        nb_group = QGroupBox("Narrowband Filters")
        nb_l = QVBoxLayout()
        for name, attr in [("Ha","ha"), ("SII","sii"), ("OIII","oiii")]:
            # starry
            btn = QPushButton(f"Load {name}")
            lbl = QLabel(f"No {name}")
            setattr(self, f"{attr}Button", btn)
            setattr(self, f"{attr}Label", lbl)
            btn.clicked.connect(lambda _, n=name: self.loadImage(n))
            nb_l.addWidget(btn); nb_l.addWidget(lbl)

            # starless
            btn_sl = QPushButton(f"Load {name} (Starless)")
            lbl_sl = QLabel(f"No {name} (starless)")
            setattr(self, f"{attr}StarlessButton", btn_sl)
            setattr(self, f"{attr}StarlessLabel", lbl_sl)
            btn_sl.clicked.connect(lambda _, n=f"{name} (Starless)": self.loadImage(n))
            nb_l.addWidget(btn_sl); nb_l.addWidget(lbl_sl)

        # user controls
        self.linear_output_checkbox = QCheckBox("Output Linear Image Only")
        nb_l.addWidget(self.linear_output_checkbox)

        # ---- Advanced (collapsed) ----
        # defaults used elsewhere
        self.threshold_value = 5.0
        self.q_factor = 0.80
        self.summary_gamma = 0.6  # gamma < 1.0 brightens summary previews

        # header row with toggle button
        adv_hdr = QHBoxLayout()
        self.advanced_btn = QPushButton("Advanced ▸")
        self.advanced_btn.setCheckable(False)
        self.advanced_btn.setFlat(True)
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        adv_hdr.addWidget(self.advanced_btn, stretch=0)
        adv_hdr.addStretch(1)
        nb_l.addLayout(adv_hdr)

        # panel that will be shown/hidden
        self.advanced_panel = QWidget()
        adv_l = QVBoxLayout(self.advanced_panel)
        adv_l.setContentsMargins(12, 0, 0, 0)  # small indent

        # WB threshold control (UI)
        thr_row = QHBoxLayout()
        self.threshold_label = QLabel(f"WB star detect threshold: {self.threshold_value:.1f}")
        self.threshold_btn = QPushButton("Change…")
        self.threshold_btn.clicked.connect(self._change_threshold)
        thr_row.addWidget(self.threshold_label)
        thr_row.addWidget(self.threshold_btn)
        adv_l.addLayout(thr_row)

        # Q factor control (UI)
        q_row = QHBoxLayout()
        self.q_label = QLabel(f"Continuum Q factor: {self.q_factor:.2f}")
        self.q_btn = QPushButton("Change…")
        self.q_btn.clicked.connect(self._change_q)
        q_row.addWidget(self.q_label)
        q_row.addWidget(self.q_btn)
        adv_l.addLayout(q_row)

        self.advanced_panel.setVisible(False)  # start hidden
        nb_l.addWidget(self.advanced_panel)

        nb_l.addStretch(1)

        self.clear_button = QPushButton("Clear Loaded Images")
        self.clear_button.clicked.connect(self.clear_loaded_images)
        nb_l.addWidget(self.clear_button)
        nb_group.setLayout(nb_l)

        # — Continuum group —
        cont_group = QGroupBox("Continuum Sources")
        cont_l = QVBoxLayout()
        for name, attr in [("Red","red"), ("Green","green"), ("OSC","osc")]:
            btn = QPushButton(f"Load {name}")
            lbl = QLabel(f"No {name}")
            setattr(self, f"{attr}Button", btn)
            setattr(self, f"{attr}Label", lbl)
            btn.clicked.connect(lambda _, n=name: self.loadImage(n))
            cont_l.addWidget(btn); cont_l.addWidget(lbl)

            btn_sl = QPushButton(f"Load {name} (Starless)")
            lbl_sl = QLabel(f"No {name} (starless)")
            setattr(self, f"{attr}StarlessButton", btn_sl)
            setattr(self, f"{attr}StarlessLabel", lbl_sl)
            btn_sl.clicked.connect(lambda _, n=f"{name} (Starless)": self.loadImage(n))
            cont_l.addWidget(btn_sl); cont_l.addWidget(lbl_sl)

        cont_l.addStretch(1)
        cont_group.setLayout(cont_l)

        # — White balance diagnostics —
        wb_group   = QGroupBox("Star-Based WB")
        self.wb_l  = QVBoxLayout()
        self.wb_l.setAlignment(Qt.AlignmentFlag.AlignTop)
        wb_group.setLayout(self.wb_l)

        # put it in a scroll area so many entries won't overflow
        wb_scroll = QScrollArea()
        wb_scroll.setWidgetResizable(True)
        wb_container = QWidget()
        wb_container.setLayout(self.wb_l)
        wb_scroll.setWidget(wb_container)

        # assemble columns
        columns_layout.addWidget(nb_group,    1)
        columns_layout.addWidget(cont_group,  1)
        columns_layout.addWidget(wb_scroll,   2)

        # — Bottom row: Execute & status —
        bottom_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        bottom_layout.addWidget(self.execute_button, stretch=1)
        bottom_layout.addWidget(self.spinnerLabel,   stretch=1)
        bottom_layout.addWidget(self.statusLabel,    stretch=3)

        # put it all together
        main_layout.addLayout(columns_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.installEventFilter(self)

    def _toggle_advanced(self):
        show = not self.advanced_panel.isVisible()
        self.advanced_panel.setVisible(show)
        self.advanced_btn.setText("Advanced ▾" if show else "Advanced ▸")

    def _change_q(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Continuum Q Factor",
            "Q (scale of broadband subtraction, typical 0.6–1.0):",
            self.q_factor,
            0.10, 2.00, 2  # min, max, decimals
        )
        if ok:
            self.q_factor = float(val)
            self.q_label.setText(f"Continuum Q factor: {self.q_factor:.2f}")

    def _change_threshold(self):
        val, ok = QInputDialog.getDouble(
            self,
            "WB Threshold",
            "Sigma threshold for star detection:",
            self.threshold_value,
            0.5, 50.0, 1  # min, max, decimals
        )
        if ok:
            self.threshold_value = float(val)
            self.threshold_label.setText(f"WB star detect threshold: {self.threshold_value:.1f}")

    def _main_window(self) -> QMainWindow | None:
        # 1) explicit parent the tool may have been created with
        mw = self.parent_window
        if mw and hasattr(mw, "mdi"):
            return mw
        # 2) walk up the parent chain
        p = self.parent()
        while p is not None:
            if hasattr(p, "mdi"):
                return p  # main window
            p = p.parent()
        # 3) search top-level widgets
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "mdi"):
                return w
        return None

    def _iter_open_docs(self):
        """Yield (doc, title) for all open subwindows."""
        mw = self._main_window()
        if not mw or not hasattr(mw, "mdi"):
            return []
        out = []
        for sw in mw.mdi.subWindowList():
            w = sw.widget()
            d = getattr(w, "document", None)
            if d is not None:
                out.append((d, sw.windowTitle()))
        return out


    def refresh(self):
        if self.image_manager:
            # You might have a way to retrieve the current image and metadata.
            # For example, if your image_manager stores the current image,
            # you could do something like:
            return



    def clear_loaded_images(self):
        for attr in (
            "ha_image","sii_image","oiii_image","red_image","green_image","osc_image",
            "ha_starless_image","sii_starless_image","oiii_starless_image",
            "red_starless_image","green_starless_image","osc_starless_image"
        ):
            setattr(self, attr, None)

        self.haLabel.setText("No Ha")
        self.siiLabel.setText("No SII")
        self.oiiiLabel.setText("No OIII")
        self.redLabel.setText("No Red")
        self.greenLabel.setText("No Green")
        self.oscLabel.setText("No OSC")

        self.haStarlessLabel.setText("No Ha (starless)")
        self.siiStarlessLabel.setText("No SII (starless)")
        self.oiiiStarlessLabel.setText("No OIII (starless)")
        self.redStarlessLabel.setText("No Red (starless)")
        self.greenStarlessLabel.setText("No Green (starless)")
        self.oscStarlessLabel.setText("No OSC (starless)")

        self.combined_image = None
        self.statusLabel.setText("All loaded images cleared.")

    def loadImage(self, channel: str):
        """
        Prompt the user to load either from file or from ImageManager slots,
        for the given channel ("Ha", "SII", "OIII", "Red", "Green", "OSC").
        """
        source, ok = QInputDialog.getItem(
            self, f"Select {channel} Image Source", "Load image from:",
            ["From View", "From File"], editable=False
        )
        if not ok:
            return

        if source == "From File":
            result = self.loadImageFromFile(channel)
        else:
            result = self.loadImageFromView(channel)  

        if not result:
            return

        image, header, bit_depth, is_mono, name_or_path = result

        # Use view title if we got one; if it's a real path, show just the basename
        label_text = str(name_or_path) if name_or_path else "From View"


        try:
            if isinstance(name_or_path, str) and os.path.isabs(name_or_path):
                label_text = os.path.basename(name_or_path)
        except Exception:
            pass

        is_starless = "(Starless)" in channel
        base = channel.replace(" (Starless)", "")

        if base == "Ha":
            if is_starless:
                self.ha_starless_image = image
                self.haStarlessLabel.setText(label_text)
            else:
                self.ha_image = image
                self.haLabel.setText(label_text)
        elif base == "SII":
            if is_starless:
                self.sii_starless_image = image
                self.siiStarlessLabel.setText(label_text)
            else:
                self.sii_image = image
                self.siiLabel.setText(label_text)
        elif base == "OIII":
            if is_starless:
                self.oiii_starless_image = image
                self.oiiiStarlessLabel.setText(label_text)
            else:
                self.oiii_image = image
                self.oiiiLabel.setText(label_text)
        elif base == "Red":
            if is_starless:
                self.red_starless_image = image
                self.redStarlessLabel.setText(label_text)
            else:
                self.red_image = image
                self.redLabel.setText(label_text)
        elif base == "Green":
            if is_starless:
                self.green_starless_image = image
                self.greenStarlessLabel.setText(label_text)
            else:
                self.green_image = image
                self.greenLabel.setText(label_text)
        elif base == "OSC":
            if is_starless:
                self.osc_starless_image = image
                self.oscStarlessLabel.setText(label_text)
            else:
                self.osc_image = image
                self.oscLabel.setText(label_text)
        else:
            QMessageBox.critical(self, "Error", f"Unknown channel '{channel}'.")
            return

        # Store header and mono-flag for later saving
        self.original_header = header
        self.is_mono         = is_mono

    def _collect_open_documents(self):
        # kept for compatibility with callers; returns only docs
        return [d for d, _ in self._iter_open_docs()]

    def _select_document_via_dropdown(self, title: str):
        items = self._iter_open_docs()
        if not items:
            QMessageBox.information(self, f"Select View — {title}", "No open views/documents found.")
            return None

        # default to active view if present
        mw = self._main_window()
        active_doc = None
        if mw and mw.mdi.activeSubWindow():
            active_doc = getattr(mw.mdi.activeSubWindow().widget(), "document", None)

        if len(items) == 1:
            return items[0][0]

        names = [t for _, t in items]
        default_index = next((i for i, (d, _) in enumerate(items) if d is active_doc), 0)

        choice, ok = QInputDialog.getItem(
            self, f"Select View — {title}", "Choose:", names, default_index, False
        )
        if not ok:
            return None
        return items[names.index(choice)][0]

    def _image_from_doc(self, doc):
        """(np.ndarray, header, bit_depth, is_mono, file_path) from an ImageDocument."""
        arr = getattr(doc, "image", None)
        if arr is None:
            QMessageBox.warning(self, "No image", "Selected view has no image.")
            return None
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = False
        try:
            import numpy as np
            is_mono = isinstance(arr, np.ndarray) and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))
        except Exception:
            pass
        return arr, header, bit_depth, is_mono, meta.get("file_path")

    def loadImageFromView(self, channel: str):
        doc = self._select_document_via_dropdown(channel)
        if not doc:
            return None
        res = self._image_from_doc(doc)
        if not res:
            return None

        img, header, bit_depth, is_mono, _ = res

        # Build a human-friendly name for the label (view/subwindow title)
        title = ""
        try:
            title = doc.display_name()
        except Exception:
            mw = self._main_window()
            if mw and mw.mdi.activeSubWindow():
                title = mw.mdi.activeSubWindow().windowTitle()

        # Return with the "path" field set to the title so the caller can label it
        return img, header, bit_depth, is_mono, title


    def loadImageFromFile(self, channel: str):
        file_filter = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {channel} Image", "", file_filter)
        if not path:
            return None
        try:
            image, header, bit_depth, is_mono = legacy_load_image(path)  # ← use the alias
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {channel} image:\n{e}")
            return None
        return image, header, bit_depth, is_mono, path

    def loadImageFromSlot(self, channel: str):
        """
        Prompt the user to pick one of the ImageManager’s slots (using custom names if defined)
        and load that image.
        """
        if not self.image_manager:
            QMessageBox.critical(self, "Error", "ImageManager is not initialized. Cannot load image from slot.")
            return None

        # Look up the main window’s custom slot names
        main_win = getattr(self, "parent_window", None) or self.window()
        slot_names = getattr(main_win, "slot_names", {})

        # Build the list of display names (zero-based)
        display_names = [
            slot_names.get(i, f"Slot {i}") 
            for i in range(self.image_manager.max_slots)
        ]

        # Ask the user to choose one
        choice, ok = QInputDialog.getItem(
            self,
            f"Select Slot for {channel}",
            "Choose a slot:",
            display_names,
            0,
            False
        )
        if not ok or not choice:
            return None

        # Map back to the numeric index
        idx = display_names.index(choice)

        # Retrieve the image and metadata
        img = self.image_manager._images.get(idx)
        if img is None:
            QMessageBox.warning(self, "Empty Slot", f"{choice} is empty.")
            return None

        meta = self.image_manager._metadata.get(idx, {})
        return (
            img,
            meta.get("original_header"),
            meta.get("bit_depth", "Unknown"),
            meta.get("is_mono", False),
            meta.get("file_path", None)
        )


    def startContinuumSubtraction(self):
        # STARRED (with stars) continuum channels
        cont_red_starry   = self.red_image   if self.red_image   is not None else (self.osc_image[..., 0] if self.osc_image is not None else None)
        cont_green_starry = self.green_image if self.green_image is not None else (self.osc_image[..., 1] if self.osc_image is not None else None)

        # STARLESS continuum channels
        cont_red_starless   = self.red_starless_image   if self.red_starless_image   is not None else (self.osc_starless_image[..., 0] if self.osc_starless_image is not None else None)
        cont_green_starless = self.green_starless_image if self.green_starless_image is not None else (self.osc_starless_image[..., 1] if self.osc_starless_image is not None else None)

        # Build tasks per NB filter
        pairs = []
        def add_pair(name, nb_starry, cont_starry, nb_starless, cont_starless):
            has_starry   = (nb_starry is not None and cont_starry is not None)
            has_starless = (nb_starless is not None and cont_starless is not None)
            if has_starry or has_starless:
                pairs.append({
                    "name": name,
                    "nb": nb_starry,
                    "cont": cont_starry,
                    "nb_sl": nb_starless,
                    "cont_sl": cont_starless,
                    "starless_only": (has_starless and not has_starry),
                })

        add_pair("Ha",   self.ha_image,   cont_red_starry,   self.ha_starless_image,   cont_red_starless)
        add_pair("SII",  self.sii_image,  cont_red_starry,   self.sii_starless_image,  cont_red_starless)
        add_pair("OIII", self.oiii_image, cont_green_starry, self.oiii_starless_image, cont_green_starless)

        if not pairs:
            self.statusLabel.setText("Load at least one NB + matching continuum channel (or OSC).")
            return

        self.showSpinner()
        self._threads = []
        self._results = []
        self._pushed_results = False
        self._pending = 0

        # How many result signals do we expect in total?
        self._expected_results = sum(
            (1 if p["nb"]    is not None and p["cont"]    is not None else 0) +
            (1 if p["nb_sl"] is not None and p["cont_sl"] is not None else 0)
            for p in pairs
        )

        for p in pairs:
            t = ContinuumProcessingThread(
                p["nb"], p["cont"], self.linear_output_checkbox.isChecked(),
                starless_nb=p["nb_sl"], starless_cont=p["cont_sl"], starless_only=p["starless_only"],
                threshold=self.threshold_value, summary_gamma=self.summary_gamma, q_factor=self.q_factor
            )
            name = p["name"]  # avoid late binding in lambdas

            if p["nb"] is not None and p["cont"] is not None:
                self._pending += 1
                t.processing_complete.connect(
                    lambda img, stars, overlay, raw, after, n=f"{name} (starry)":
                        self._onOneResult(n, img, stars, overlay, raw, after)
                )

            if p["nb_sl"] is not None and p["cont_sl"] is not None:
                self._pending += 1
                t.processing_complete_starless.connect(
                    lambda img, stars, overlay, raw, after, n=f"{name} (starless)":
                        self._onOneResult(n, img, stars, overlay, raw, after)
                )

            t.status_update.connect(self.update_status_label)
            self._threads.append(t)
            t.start()


    def _onOneResult(self, filt, img, star_count, overlay_qimg, raw_pixels, after_pixels):
        # stash for later slot‐pushing
        self._results.append({
            "filter":  filt,
            "image":   img,
            "stars":   star_count,
            "overlay": overlay_qimg,
            "raw":     raw_pixels,
            "after":   after_pixels
        })

        # ---------- thumbnails / diagnostics ----------
        make_scatter = (
            isinstance(raw_pixels, np.ndarray) and
            raw_pixels.ndim == 2 and raw_pixels.shape[1] >= 2 and
            raw_pixels.shape[0] >= 3 and
            (cv2 is not None)
        )

        if make_scatter:
            nb_flux   = raw_pixels[:, 0].astype(np.float32, copy=False)
            cont_flux = raw_pixels[:, 1].astype(np.float32, copy=False)

            h, w = 200, 200
            scatter_img = np.ones((h, w, 3), np.uint8) * 255

            # 1) best-fit NB ≈ m·BB + c
            try:
                m, c = np.polyfit(cont_flux, nb_flux, 1)
                x0f, y0f = 0.0, c
                x1f, y1f = 1.0, m + c
                y0f = float(np.clip(y0f, 0.0, 1.0))
                y1f = float(np.clip(y1f, 0.0, 1.0))
                x0 = int(x0f * (w - 1)); y0 = int((1 - y0f) * (h - 1))
                x1 = int(x1f * (w - 1)); y1 = int((1 - y1f) * (h - 1))
                cv2.line(scatter_img, (x0, y0), (x1, y1), (0, 0, 255), 2)   # red line (BGR)
            except Exception:
                pass

            # 2) points
            xs = (np.clip(cont_flux, 0, 1) * (w - 1)).astype(int)
            ys = ((1 - np.clip(nb_flux, 0, 1)) * (h - 1)).astype(int)
            for x, y in zip(xs, ys):
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(scatter_img, (x, y), 2, (255, 0, 0), -1)     # blue points (BGR)

            # axes
            cv2.line(scatter_img, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 1)
            cv2.line(scatter_img, (0, 0), (0, h - 1), (0, 0, 0), 1)

            # labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            ((tw, _), _) = cv2.getTextSize("BB Flux", font, 0.5, 1)
            cv2.putText(scatter_img, "BB Flux", ((w - tw) // 2, h - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            for i, ch in enumerate("NB Flux"):
                cv2.putText(scatter_img, ch, (2, 15 + i*15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            qscatter = QImage(scatter_img.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            scatter_pix = QPixmap.fromImage(qscatter)

        # overlay thumbnail (always)
        thumb_pix = QPixmap.fromImage(overlay_qimg).scaled(
            200, 200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # assemble entry row
        entry = QWidget()
        elay  = QHBoxLayout(entry)
        elay.addWidget(QLabel(f"{filt}: {star_count} stars"))

        if make_scatter:
            scatter_label = QLabel()
            scatter_label.setPixmap(scatter_pix)
            scatter_label.setCursor(Qt.CursorShape.PointingHandCursor)
            elay.addWidget(scatter_label)
            self._clickable_images[scatter_label] = scatter_pix
            scatter_label.installEventFilter(self)

        overlay_label = QLabel()
        overlay_label.setPixmap(thumb_pix)
        overlay_label.setCursor(Qt.CursorShape.PointingHandCursor)
        elay.addWidget(overlay_label)
        self._clickable_images[overlay_label] = QPixmap.fromImage(overlay_qimg)
        overlay_label.installEventFilter(self)

        elay.addStretch(1)
        entry.setLayout(elay)
        self.wb_l.addWidget(entry)

        # ---------- call _pushResultsToDocs exactly once ----------
        if (not getattr(self, "_pushed_results", False)
            and len(self._results) == getattr(self, "_expected_results", 0)):
            self._pushed_results = True
            self.hideSpinner()
            self._pushResultsToDocs(self._results)


    def eventFilter(self, source, event):
        # catch mouse releases on any of our clickable labels
        if event.type() == QEvent.Type.MouseButtonRelease and source in self._clickable_images:
            pix = self._clickable_images[source]
            self._showEnlarged(pix)
            return True
        return super().eventFilter(source, event)

    def _showEnlarged(self, pixmap: QPixmap):
        """
        Detail View dialog with zoom controls, autostretch, and a 'push to document' action.
        """
        # --- helpers (local to this dialog) ---
        def qimage_from_float01(arr: np.ndarray) -> QImage:
            """float32 [0..1] -> QImage (RGB888 or Grayscale8), deep-copied."""
            a = np.clip(arr, 0.0, 1.0)
            if a.ndim == 2:  # mono
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w = u8.shape
                qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
                return qimg.copy()
            else:            # color
                if a.shape[2] == 1:
                    a = a[..., 0]
                    return qimage_from_float01(a)
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w, _ = u8.shape
                qimg = QImage(u8.data, w, h, 3*w, QImage.Format.Format_RGB888)
                return qimg.copy()

        def float01_from_qimage(qimg: QImage) -> np.ndarray:
            """QImage -> float32 [0..1]; supports Grayscale8 and RGB888 primarily."""
            if qimg.isNull():
                return np.zeros((1, 1), dtype=np.float32)
            fmt = qimg.format()
            if fmt != QImage.Format.Format_Grayscale8 and fmt != QImage.Format.Format_RGB888:
                # normalize to a friendly format
                qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                fmt = QImage.Format.Format_RGB888

            h = qimg.height()
            w = qimg.width()
            bpl = qimg.bytesPerLine()

            ptr = qimg.bits()
            ptr.setsize(h * bpl)
            buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))

            if fmt == QImage.Format.Format_Grayscale8:
                arr = buf[:, :w].astype(np.float32) / 255.0
                return arr
            else:  # RGB888
                arr = buf[:, :w*3].reshape((h, w, 3)).astype(np.float32) / 255.0
                return arr

        def qpixmap_from_float01(arr: np.ndarray) -> QPixmap:
            return QPixmap.fromImage(qimage_from_float01(arr))

        def percentile_autostretch(arr: np.ndarray, low=0.5, high=99.5) -> np.ndarray:
            """
            Simple contrast stretch using global percentiles.
            Works for mono or RGB; applies a single scale to all channels for coherence.
            """
            a = arr.astype(np.float32, copy=False)
            if a.ndim == 3:
                gray = np.mean(a, axis=2)
            else:
                gray = a
            p1, p2 = np.percentile(gray, [low, high])
            if p2 <= p1 + 1e-8:
                return np.clip(a, 0.0, 1.0)
            out = (a - p1) / (p2 - p1)
            return np.clip(out, 0.0, 1.0)

        # --- build dialog UI ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Detail View")
        dlg.resize(980, 820)

        outer = QVBoxLayout(dlg)

        # image area in a scroll view (so large zooms can be panned)
        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        img_container = QWidget()
        img_layout = QVBoxLayout(img_container)
        img_layout.setContentsMargins(0, 0, 0, 0)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Keep original image as float [0..1]
        base_qimg = pixmap.toImage()
        base_arr = float01_from_qimage(base_qimg)
        current_arr = base_arr.copy()

        # zoom state
        base_h = base_qimg.height()
        base_w = base_qimg.width()
        scale = 1.0

        def update_view():
            nonlocal scale
            # render current_arr -> pixmap, then scale
            pm = qpixmap_from_float01(current_arr)
            target_w = max(1, int(base_w * scale))
            target_h = max(1, int(base_h * scale))
            img_label.setPixmap(pm.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

        img_layout.addWidget(img_label)
        img_container.setLayout(img_layout)
        scroll.setWidget(img_container)
        outer.addWidget(scroll, stretch=1)

        # controls
        row = QHBoxLayout()
        btn_zoom_out = QPushButton("– Zoom")
        btn_zoom_reset = QPushButton("Reset Zoom")
        btn_zoom_in = QPushButton("+ Zoom")
        row.addWidget(btn_zoom_out)
        row.addWidget(btn_zoom_reset)
        row.addWidget(btn_zoom_in)

        row.addStretch(1)

        btn_autostretch = QPushButton("Autostretch")
        row.addWidget(btn_autostretch)

        btn_push = QPushButton("Push to New Document")
        row.addWidget(btn_push)

        btn_close = QPushButton("Close")
        row.addWidget(btn_close)

        outer.addLayout(row)

        # --- wire up actions ---
        def do_zoom_in():
            nonlocal scale
            scale = min(20.0, scale * 1.25)
            update_view()

        def do_zoom_out():
            nonlocal scale
            scale = max(0.05, scale / 1.25)
            update_view()

        def do_zoom_reset():
            nonlocal scale
            scale = 1.0
            update_view()

        def do_autostretch():
            nonlocal current_arr
            current_arr = stretch_color_image(current_arr, target_median=0.25)
            update_view()

        def do_push_to_doc():
            dm = getattr(self, "doc_manager", None)
            mw = self._main_window()
            if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
                QMessageBox.critical(dlg, "Detail View", "Cannot create document: missing DocManager or MainWindow.")
                return

            # keep float32 [0..1] for consistency with your pipeline
            img = current_arr.astype(np.float32, copy=False)

            # make a friendly unique name
            counter = getattr(self, "_detail_doc_counter", 0) + 1
            self._detail_doc_counter = counter
            name = f"DetailView_{counter}"

            meta = {
                "display_name": name,
                "file_path": name,
                "bit_depth": "32-bit floating point",
                "is_mono": (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)),
                "original_header": getattr(self, "original_header", None),
                "source": "Continuum Subtract — Detail View",
            }
            try:
                doc = dm.create_document(img, metadata=meta, name=name)
                mw._spawn_subwindow_for(doc)
                self.statusLabel.setText(f"Pushed detail view → '{name}'.")
            except Exception as e:
                QMessageBox.critical(dlg, "Detail View", f"Failed to create document:\n{e}")

        btn_zoom_in.clicked.connect(do_zoom_in)
        btn_zoom_out.clicked.connect(do_zoom_out)
        btn_zoom_reset.clicked.connect(do_zoom_reset)
        btn_autostretch.clicked.connect(do_autostretch)
        btn_push.clicked.connect(do_push_to_doc)
        btn_close.clicked.connect(dlg.accept)

        # initial render
        update_view()
        dlg.exec()


    def _onThreadFinished(self):
        self._pending -= 1
        if self._pending == 0:
            self.hideSpinner()
            self._pushResultsToDocs(self._results) 

    def _pushResultsToDocs(self, results):
        dm = getattr(self, "doc_manager", None)
        mw = self._main_window()
        if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
            QMessageBox.critical(self, "Continuum Subtract",
                                "Cannot create documents: missing DocManager or MainWindow.")
            return

        created = 0
        for entry in results:
            filt = entry["filter"]
            img  = np.asarray(entry["image"], dtype=np.float32)  # keep everything float32
            name = f"{filt}_ContSub"

            meta = {
                "display_name": name,                 # nice title in the UI
                "file_path": name,                    # placeholder path until user saves
                "bit_depth": "32-bit floating point",
                "is_mono": (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)),
                "original_header": self.original_header,
                "source": "Continuum Subtract",
            }

            try:
                # create a proper ImageDocument and register it
                doc = dm.create_document(img, metadata=meta, name=name)
                # show it as an MDI subwindow
                mw._spawn_subwindow_for(doc)
                created += 1
            except Exception as e:
                QMessageBox.critical(self, "Continuum Subtract",
                                    f"Failed to create document '{name}':\n{e}")

        self.statusLabel.setText(f"Created {created} document(s).")


    def _onThreadFinished(self):
        self._pending -= 1
        # do NOT push here if you already push in _onOneResult

    def update_status_label(self, message):
        self.statusLabel.setText(message)

    def showSpinner(self):
        self.spinnerLabel.setText("Processing…")
        self.spinnerLabel.show()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(False)

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerLabel.clear()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(True)



class ContinuumProcessingThread(QThread):
    processing_complete = pyqtSignal(np.ndarray, int, QImage, np.ndarray, np.ndarray)
    processing_complete_starless = pyqtSignal(np.ndarray, int, QImage, np.ndarray, np.ndarray)
    status_update = pyqtSignal(str)

    def __init__(self, nb_image, continuum_image, output_linear, *,
                 starless_nb=None, starless_cont=None, starless_only=False,
                 threshold: float = 5.0, summary_gamma: float = 0.6, q_factor: float = 0.8):
        super().__init__()
        self.nb_image = nb_image
        self.continuum_image = continuum_image
        self.output_linear = output_linear
        self.starless_nb = starless_nb
        self.starless_cont = starless_cont
        self.starless_only = starless_only
        self.background_reference = None
        self._recipe = None  # learned from starry pass

        # user knobs
        self.threshold = float(threshold)
        self.summary_gamma = float(summary_gamma)
        self.q_factor = float(q_factor)

    # ---------- small helpers ----------
    @staticmethod
    def _to_mono(img):
        a = np.asarray(img)
        if a.ndim == 3:
            if a.shape[2] == 3:
                return a[..., 0]  # use R channel for NB/cont slots when color
            if a.shape[2] == 1:
                return a[..., 0]
        return a

    @staticmethod
    def _as_rgb(nb, cont):
        r = np.asarray(nb,   dtype=np.float32)
        g = np.asarray(cont, dtype=np.float32)
        if r.ndim == 3: r = r[..., 0]
        if g.ndim == 3: g = g[..., 0]
        if r.dtype.kind in "ui":
            r = r / (255.0 if r.dtype == np.uint8 else 65535.0)
        if g.dtype.kind in "ui":
            g = g / (255.0 if g.dtype == np.uint8 else 65535.0)
        b = g
        return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _fit_ab(x, y):
        x = x.reshape(-1).astype(np.float32)
        y = y.reshape(-1).astype(np.float32)
        N = min(x.size, 100_000)
        if x.size > N:
            idx = np.random.choice(x.size, N, replace=False)
            x = x[idx]; y = y[idx]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)

    @staticmethod
    def _qimage_from_uint8(rgb_uint8: np.ndarray) -> QImage:
        """Create a deep-copied QImage from an HxWx3 uint8 array."""
        h, w = rgb_uint8.shape[:2]
        return QImage(rgb_uint8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _nonlinear_finalize(lin_img: np.ndarray) -> np.ndarray:
        """Stretch → subtract pedestal → curves, returned as float32 in [0,1]."""
        target_median = 0.25
        stretched = stretch_color_image(lin_img, target_median, True, False)
        final = stretched - 0.7 * np.median(stretched)
        final = np.clip(final, 0, 1)
        return apply_curves_adjustment(final, np.median(final), 0.5).astype(np.float32, copy=False)

    # ---------- BG neutral: return pedestals (no in-place surprises) ----------
    def _compute_bg_pedestal(self, rgb):
        height, width, _ = rgb.shape
        num_boxes, box_size, iterations = 200, 25, 25

        boxes = [(np.random.randint(0, height - box_size),
                  np.random.randint(0, width - box_size)) for _ in range(num_boxes)]
        best = np.full(num_boxes, np.inf, dtype=np.float32)

        for _ in range(iterations):
            for i, (y, x) in enumerate(boxes):
                if y + box_size <= height and x + box_size <= width:
                    patch = rgb[y:y+box_size, x:x+box_size]
                    med = np.median(patch) if patch.size else np.inf
                    best[i] = min(best[i], med)
                    sv = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy*box_size, x + dx*box_size
                            if 0 <= yy < height - box_size and 0 <= xx < width - box_size:
                                p2 = rgb[yy:yy+box_size, xx:xx+box_size]
                                if p2.size:
                                    sv.append(np.median(p2))
                    if sv:
                        k = int(np.argmin(sv))
                        y += (k // 3 - 1) * box_size
                        x += (k % 3  - 1) * box_size
                        boxes[i] = (y, x)

        # pick darkest
        darkest = np.inf; ref = None
        for y, x in boxes:
            if y + box_size <= height and x + box_size <= width:
                patch = rgb[y:y+box_size, x:x+box_size]
                med = np.median(patch) if patch.size else np.inf
                if med < darkest:
                    darkest, ref = med, patch

        ped = np.zeros(3, dtype=np.float32)
        if ref is not None:
            self.background_reference = np.median(ref.reshape(-1, 3), axis=0)
            chan_meds = np.median(rgb, axis=(0, 1))
            # pedestal to lift channels toward their own median
            ped = np.maximum(0.0, chan_meds - self.background_reference)

            # specifically lift G/B if below R reference
            r_ref = float(self.background_reference[0])
            for ch in (1, 2):
                if self.background_reference[ch] < r_ref:
                    ped[ch] += (r_ref - self.background_reference[ch])
        return ped

    @staticmethod
    def _apply_pedestal(rgb, ped):
        return np.clip(rgb + ped.reshape(1,1,3), 0.0, 1.0)

    @staticmethod
    def _normalize_red_to_green(rgb):
        r = rgb[...,0]; g = rgb[...,1]
        mad_r = float(np.mean(np.abs(r - np.mean(r))))
        mad_g = float(np.mean(np.abs(g - np.mean(g))))
        med_r = float(np.median(r))
        med_g = float(np.median(g))
        g_gain = (mad_g / max(mad_r, 1e-9))
        g_offs = (-g_gain * med_r + med_g)
        rgb2 = rgb.copy()
        rgb2[...,0] = np.clip(r * g_gain + g_offs, 0.0, 1.0)
        return rgb2, g_gain, g_offs

    def _linear_subtract(self, rgb, Q, green_median):
        r = rgb[...,0]; g = rgb[...,1]
        return np.clip(r - Q * (g - green_median), 0.0, 1.0)

    # ---------- main ----------
    def run(self):
        try:
            # STARLESS-ONLY early exit
            if (self.nb_image is None or self.continuum_image is None) and self.starless_only:
                self._run_starless_only()
                return

            recipe = None

            # ----- starry pass (learn recipe) -----
            if self.nb_image is not None and self.continuum_image is not None:
                rgb = self._as_rgb(self.nb_image, self.continuum_image)

                self.status_update.emit("Performing background neutralization...")
                ped = self._compute_bg_pedestal(rgb)
                rgb = self._apply_pedestal(rgb, ped)

                self.status_update.emit("Normalizing red to green…")
                rgb, g_gain, g_offs = self._normalize_red_to_green(rgb)

                self.status_update.emit("Performing star-based white balance…")
                balanced_rgb, star_count, star_overlay, raw_star_pixels, after_star_pixels = \
                    apply_star_based_white_balance(
                        rgb, threshold=self.threshold, autostretch=False,
                        reuse_cached_sources=True, return_star_colors=True
                    )

                # per-channel affine fit to reproduce WB later
                wb_a = np.zeros(3, np.float32)
                wb_b = np.zeros(3, np.float32)
                for c in range(3):
                    a, b = self._fit_ab(rgb[..., c], balanced_rgb[..., c])
                    wb_a[c], wb_b[c] = a, b

                green_med = float(np.median(balanced_rgb[..., 1]))
                Q = self.q_factor
                linear_image = self._linear_subtract(balanced_rgb, Q, green_med)

                # --- NEW: gamma brighten overlay for the summary ---
                g = max(self.summary_gamma, 1e-6)
                overlay_gamma = np.power(np.clip(star_overlay, 0.0, 1.0), g)
                overlay_uint8 = (overlay_gamma * 255).astype(np.uint8)
                qimg = self._qimage_from_uint8(overlay_uint8)

                if self.output_linear:
                    self.processing_complete.emit(
                        np.clip(linear_image, 0, 1), int(star_count), qimg,
                        np.array(raw_star_pixels), np.array(after_star_pixels)
                    )
                else:
                    self.status_update.emit("Linear → Non-linear stretch…")
                    target_median = 0.25
                    stretched = stretch_color_image(linear_image, target_median, True, False)
                    final = stretched - 0.7 * np.median(stretched)
                    final = np.clip(final, 0, 1)
                    final = apply_curves_adjustment(final, np.median(final), 0.5)
                    self.processing_complete.emit(
                        final.astype(np.float32, copy=False),
                        int(star_count), qimg,
                        np.array(raw_star_pixels), np.array(after_star_pixels)
                    )

                # learned recipe + fit data (reused for starless)
                recipe = {
                    "pedestal": ped,
                    "rnorm_gain": g_gain,
                    "rnorm_offs": g_offs,
                    "wb_a": wb_a,
                    "wb_b": wb_b,
                    "Q": Q,
                    "green_median": green_med,
                    "fit_qimg": qimg,
                    "fit_star_count": int(star_count),
                    "fit_raw": np.array(raw_star_pixels),
                    "fit_after": np.array(after_star_pixels),
                }

            # ----- starless paired pass (apply recipe) -----
            if self.starless_nb is not None and self.starless_cont is not None:
                if recipe is not None:
                    rgb = self._as_rgb(self.starless_nb, self.starless_cont)
                    # apply starry recipe exactly
                    rgb = self._apply_pedestal(rgb, recipe["pedestal"])
                    r = rgb[..., 0]
                    rgb[..., 0] = np.clip(r * recipe["rnorm_gain"] + recipe["rnorm_offs"], 0.0, 1.0)
                    for c in range(3):
                        rgb[..., c] = np.clip(rgb[..., c] * recipe["wb_a"][c] + recipe["wb_b"][c], 0.0, 1.0)

                    lin = self._linear_subtract(rgb, recipe["Q"], recipe["green_median"])

                    # reuse gamma-bright overlay & fit info from the starry pass
                    fit_qimg  = recipe["fit_qimg"]
                    fit_count = recipe["fit_star_count"]
                    fit_raw   = recipe["fit_raw"]
                    fit_after = recipe["fit_after"]

                    if self.output_linear:
                        self.processing_complete_starless.emit(
                            np.clip(lin, 0, 1), fit_count, fit_qimg, fit_raw, fit_after
                        )
                    else:
                        self.status_update.emit("Linear → Non-linear stretch (starless)…")
                        target_median = 0.25
                        stretched = stretch_color_image(lin, target_median, True, False)
                        final = stretched - 0.7 * np.median(stretched)
                        final = np.clip(final, 0, 1)
                        final = apply_curves_adjustment(final, np.median(final), 0.5)
                        self.processing_complete_starless.emit(
                            final.astype(np.float32, copy=False),
                            fit_count, fit_qimg, fit_raw, fit_after
                        )
                elif self.starless_only:
                    pass  # handled in _run_starless_only
        except Exception as e:
            try:
                self.status_update.emit(f"Continuum subtraction failed: {e}")
            except Exception:
                pass

    # ----- starless-only path (no WB; same math you had) -----
    def _run_starless_only(self):
        rgb = self._as_rgb(self.starless_nb, self.starless_cont)

        self.status_update.emit("Performing background neutralization…")
        ped = self._compute_bg_pedestal(rgb)
        rgb = self._apply_pedestal(rgb, ped)

        self.status_update.emit("Normalizing red to green…")
        rgb, _, _ = self._normalize_red_to_green(rgb)

        green_med = float(np.median(rgb[..., 1]))
        lin = self._linear_subtract(rgb, 0.9, green_med)

        # Blank overlay & empty star lists (no star detection in starless-only path)
        h, w = lin.shape[:2]
        blank = np.zeros((h, w, 3), np.uint8)
        qimg  = self._qimage_from_uint8(blank)
        empty = np.empty((0, 2), float)

        if self.output_linear:
            self.processing_complete_starless.emit(np.clip(lin, 0, 1), 0, qimg, empty, empty)
            return

        self.status_update.emit("Linear → Non-linear stretch…")
        final = self._nonlinear_finalize(lin)
        self.processing_complete_starless.emit(final, 0, qimg, empty, empty)