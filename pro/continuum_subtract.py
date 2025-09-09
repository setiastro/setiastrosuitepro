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
        self.spinnerLabel.setStyleSheet("color:#999; font-style:italic;")  # optional styling
        self.spinnerLabel.hide()

        self.statusLabel = QLabel("")  
        self.statusLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout = QVBoxLayout()  # overall vertical: columns + bottom row
        columns_layout = QHBoxLayout()  # holds the three groups

        # — NB group —
        nb_group = QGroupBox("Narrowband Filters")
        nb_l = QVBoxLayout()
        for name, attr in [("Ha","ha"), ("SII","sii"), ("OIII","oiii")]:
            btn = QPushButton(f"Load {name}")
            lbl = QLabel(f"No {name}")
            setattr(self, f"{attr}Button", btn)
            setattr(self, f"{attr}Label", lbl)
            btn.clicked.connect(lambda _,n=name: self.loadImage(n))
            nb_l.addWidget(btn)
            nb_l.addWidget(lbl)

        self.linear_output_checkbox = QCheckBox("Output Linear Image Only")
        nb_l.addWidget(self.linear_output_checkbox)    
        nb_l.addStretch(1)
        
        # ** clear-all button **
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
            btn.clicked.connect(lambda _,n=name: self.loadImage(n))
            cont_l.addWidget(btn)
            cont_l.addWidget(lbl)
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
        columns_layout.addWidget(nb_group,    1)  # stretch factor 1
        columns_layout.addWidget(cont_group,  1)  # stretch factor 1
        columns_layout.addWidget(wb_scroll,   2)  # stretch factor 2 (wider)

        # — Bottom row: Execute & status —
        bottom_layout = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        bottom_layout.addWidget(self.execute_button, stretch=1)
        # statusLabel must already exist
        bottom_layout.addWidget(self.spinnerLabel, stretch=1)
        bottom_layout.addWidget(self.statusLabel,   stretch=3)


        # put it all together
        main_layout.addLayout(columns_layout)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.installEventFilter(self)

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
        for attr in ("ha_image","sii_image","oiii_image","red_image","green_image","osc_image"):
            setattr(self, attr, None)

        self.haLabel.setText("No Ha")
        self.siiLabel.setText("No SII")
        self.oiiiLabel.setText("No OIII")
        self.redLabel.setText("No Red")
        self.greenLabel.setText("No Green")
        self.oscLabel.setText("No OSC")

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
        if channel == "Ha":
            self.ha_image = image
            self.haLabel .setText(label_text)
        elif channel == "SII":
            self.sii_image = image
            self.siiLabel.setText(label_text)
        elif channel == "OIII":
            self.oiii_image = image
            self.oiiiLabel.setText(label_text)
        elif channel == "Red":
            self.red_image = image
            self.redLabel.setText(label_text)
        elif channel == "Green":
            self.green_image = image
            self.greenLabel.setText(label_text)
        elif channel == "OSC":
            self.osc_image = image
            self.oscLabel.setText(label_text)
        else:
            # unexpected channel string
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
        # — build continuum channels with explicit None checks —
        if hasattr(self, "red_image") and self.red_image is not None:
            cont_red = self.red_image
        elif hasattr(self, "osc_image") and self.osc_image is not None:
            cont_red = self.osc_image[..., 0]
        else:
            cont_red = None

        if hasattr(self, "green_image") and self.green_image is not None:
            cont_green = self.green_image
        elif hasattr(self, "osc_image") and self.osc_image is not None:
            cont_green = self.osc_image[..., 1]
        else:
            cont_green = None

        # — build tasks as before —
        tasks = []
        if self.ha_image is not None and cont_red is not None:
            tasks.append(("Ha",  self.ha_image,  cont_red))
        if self.sii_image is not None and cont_red is not None:
            tasks.append(("SII", self.sii_image, cont_red))
        if self.oiii_image is not None and cont_green is not None:
            tasks.append(("OIII", self.oiii_image, cont_green))

        if not tasks:
            self.statusLabel.setText("Load at least one NB + matching continuum channel (or OSC).")
            return


        self.showSpinner()

        self._threads.clear()
        self._pending = len(tasks)
        self._results = []

        for name, nb, cont in tasks:
            t = ContinuumProcessingThread(nb, cont, self.linear_output_checkbox.isChecked())
            t.status_update.connect(self.update_status_label)
            t.processing_complete.connect(
                lambda img, stars, overlay, raw, after, n=name:
                    self._onOneResult(n, img, stars, overlay, raw, after)
            )
            t.finished.connect(self._onThreadFinished)
            self._threads.append(t)      # keep a reference
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

        # build scatter‐plot
        nb_flux   = raw_pixels[:, 0]
        cont_flux = raw_pixels[:, 1]
        h, w = 200, 200
        scatter_img = np.ones((h, w, 3), np.uint8) * 255

        # 1) Compute best-fit in flux space: NB ≈ m·BB + c
        m, c = np.polyfit(cont_flux, nb_flux, 1)

        # 2) Choose two BB positions to draw the line at (0 and 1)
        x0f, y0f = 0.0, c
        x1f, y1f = 1.0, m*1.0 + c

        # clip so we stay within [0,1]
        y0f = np.clip(y0f, 0.0, 1.0)
        y1f = np.clip(y1f, 0.0, 1.0)

        # map to pixel coords
        x0 = int(x0f * (w - 1))
        y0 = int((1 - y0f) * (h - 1))
        x1 = int(x1f * (w - 1))
        y1 = int((1 - y1f) * (h - 1))

        # draw the fit line (in blue, thickness 2)
        cv2.line(scatter_img, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # 3) Plot points
        xs = (cont_flux * (w - 1)).astype(int)
        ys = ((1 - nb_flux) * (h - 1)).astype(int)
        for x, y in zip(xs, ys):
            cv2.circle(scatter_img, (x, y), 2, (0, 0, 255), -1)
        # draw axes
        cv2.line(scatter_img, (0, h - 1), (w - 1, h - 1), (0, 0, 0), 1)  # x-axis
        cv2.line(scatter_img, (0, 0),       (0, h - 1),     (0, 0, 0), 1)  # y-axis

        # put “BB Flux” centered on the x-axis
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "BB Flux"
        ((tw, th), _) = cv2.getTextSize(text, font, 0.5, 1)
        x_text = (w - tw) // 2
        y_text = h - 5  # just above bottom
        cv2.putText(scatter_img, text, (x_text, y_text), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # put “NB Flux” vertically along the left
        vert_text = "NB Flux"
        # draw each character, stepping down
        for i, ch in enumerate(vert_text):
            # x is a few pixels right of the y-axis, y steps by 15px
            cv2.putText(scatter_img, ch, (2, 15 + i*15), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # convert to QPixmap
        qscatter = QImage(scatter_img.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        scatter_pix = QPixmap.fromImage(qscatter)

        # overlay thumbnail
        thumb_pix = QPixmap.fromImage(overlay_qimg).scaled(
            200, 200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # assemble entry row
        entry = QWidget()
        elay  = QHBoxLayout(entry)

        # 1) star count label
        elay.addWidget(QLabel(f"{filt}: {star_count} stars"))

        # 2) scatter thumbnail
        scatter_label = QLabel()
        scatter_label.setPixmap(scatter_pix)
        scatter_label.setCursor(Qt.CursorShape.PointingHandCursor)
        elay.addWidget(scatter_label)
        # remember it & install filter
        self._clickable_images[scatter_label] = scatter_pix
        scatter_label.installEventFilter(self)

        # 3) overlay thumbnail
        overlay_label = QLabel()
        overlay_label.setPixmap(thumb_pix)
        overlay_label.setCursor(Qt.CursorShape.PointingHandCursor)
        elay.addWidget(overlay_label)
        self._clickable_images[overlay_label] = QPixmap.fromImage(overlay_qimg)
        overlay_label.installEventFilter(self)

        elay.addStretch(1)
        entry.setLayout(elay)

        # add to the WB column
        self.wb_l.addWidget(entry)

    def eventFilter(self, source, event):
        # catch mouse releases on any of our clickable labels
        if event.type() == QEvent.Type.MouseButtonRelease and source in self._clickable_images:
            pix = self._clickable_images[source]
            self._showEnlarged(pix)
            return True
        return super().eventFilter(source, event)

    def _showEnlarged(self, pixmap):
        # simple dialog that just shows the pixmap at window-fitting size
        dlg = QDialog(self)
        dlg.setWindowTitle("Detail View")
        layout = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(pixmap.scaled(800, 800, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(lbl)
        dlg.resize(820, 820)
        dlg.exec()  # modal; user closes when done

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
    status_update = pyqtSignal(str)

    def __init__(self, nb_image, continuum_image, output_linear):
        super().__init__()
        self.nb_image = nb_image
        self.continuum_image = continuum_image
        self.output_linear = output_linear
        self.background_reference = None  # Store the background reference



    def run(self):
        # Ensure both images are mono
        if self.nb_image.ndim == 3 and self.nb_image.shape[2] == 3:
            self.nb_image = self.nb_image[..., 0]  # Take one channel for the NB image

        if self.continuum_image.ndim == 3 and self.continuum_image.shape[2] == 3:
            self.continuum_image = self.continuum_image[..., 0]  # Take one channel for the continuum image

        # Create RGB image
        r_combined = self.nb_image  # Use the normalized NB image as the Red channel
        g_combined = self.continuum_image # Use the normalized continuum image as the Green channel
        b_combined = self.continuum_image  # Use the normalized continuum image as the Blue channel


        # Stack the channels into a single RGB image
        combined_image = np.stack((r_combined, g_combined, b_combined), axis=-1)

        self.status_update.emit("Performing background neutralization...")
        QCoreApplication.processEvents()
            # Perform background neutralization
        self.background_neutralization(combined_image)

        # Normalize the red channel to the green channel
        combined_image[..., 0] = self.normalize_channel(combined_image[..., 0], combined_image[..., 1])

        self.status_update.emit("Performing star-based white balance…")
        balanced_rgb, star_count, star_overlay, raw_star_pixels, after_star_pixels = \
            apply_star_based_white_balance(
                combined_image,
                threshold=1.5,
                autostretch=False,
                reuse_cached_sources=True,
                return_star_colors=True
            )
        combined_image[:] = balanced_rgb   # replace working image with the white-balanced one
        
        self.status_update.emit(f"White balance complete ({star_count} stars).")
        QCoreApplication.processEvents()

        # Perform continuum subtraction
        linear_image = combined_image[..., 0] - 0.9*(combined_image[..., 1]-np.median(combined_image[..., 1]))

            # Check if the Output Linear checkbox is checked
        if self.output_linear:
            lin = np.clip(linear_image, 0, 1)
            # minimal placeholders
            h, w = lin.shape[:2]
            overlay_uint8 = np.zeros((h, w, 3), np.uint8)
            qimg = QImage(overlay_uint8.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            empty = np.empty((0, 2), dtype=float)
            self.processing_complete.emit(lin, 0, qimg, empty, empty)
            return

        self.status_update.emit("Subtraction complete.")
        QCoreApplication.processEvents()

        # Perform statistical stretch
        target_median = 0.25
        stretched_image = stretch_color_image(linear_image, target_median, True, False)

        # Final image adjustment
        final_image = stretched_image - 0.7*np.median(stretched_image)

        # Clip the final image to stay within [0, 1]
        final_image = np.clip(final_image, 0, 1)

        # Applies Curves Boost
        final_image = apply_curves_adjustment(final_image, np.median(final_image), 0.5)

        self.status_update.emit("Linear to Non-Linear Stretch complete.")
        QCoreApplication.processEvents()

        overlay_uint8 = (star_overlay * 255).astype(np.uint8)
        h2, w2 = overlay_uint8.shape[:2]
        bytes_per_line = 3 * w2
        qimg = QImage(
            overlay_uint8.data, w2, h2, bytes_per_line,
            QImage.Format.Format_RGB888
        ).copy()
        # Emit the final image for preview
        self.processing_complete.emit(
            final_image,          # → np.ndarray
            star_count,           # → int
            qimg,                 # → QImage
            np.array(raw_star_pixels),   # → np.ndarray
            np.array(after_star_pixels)  # → np.ndarray
        )

    def background_neutralization(self, rgb_image):
        height, width, _ = rgb_image.shape
        num_boxes = 200
        box_size = 25
        iterations = 25

        boxes = [(np.random.randint(0, height - box_size), np.random.randint(0, width - box_size)) for _ in range(num_boxes)]
        best_means = np.full(num_boxes, np.inf)

        for _ in range(iterations):
            for i, (y, x) in enumerate(boxes):
                if y + box_size <= height and x + box_size <= width:
                    patch = rgb_image[y:y + box_size, x:x + box_size]
                    patch_median = np.median(patch) if patch.size > 0 else np.inf

                    if patch_median < best_means[i]:
                        best_means[i] = patch_median

                    surrounding_values = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            surrounding_y = y + dy * box_size
                            surrounding_x = x + dx * box_size
                            
                            if (0 <= surrounding_y < height - box_size) and (0 <= surrounding_x < width - box_size):
                                surrounding_patch = rgb_image[surrounding_y:surrounding_y + box_size, surrounding_x:surrounding_x + box_size]
                                if surrounding_patch.size > 0:
                                    surrounding_values.append(np.median(surrounding_patch))

                    if surrounding_values:
                        dimmest_index = np.argmin(surrounding_values)
                        new_y = y + (dimmest_index // 3 - 1) * box_size
                        new_x = x + (dimmest_index % 3 - 1) * box_size
                        boxes[i] = (new_y, new_x)

        # After iterations, find the darkest box median
        darkest_value = np.inf
        background_box = None

        for box in boxes:
            y, x = box
            if y + box_size <= height and x + box_size <= width:
                patch = rgb_image[y:y + box_size, x:x + box_size]
                patch_median = np.median(patch) if patch.size > 0 else np.inf

                if patch_median < darkest_value:
                    darkest_value = patch_median
                    background_box = patch

        if background_box is not None:
            self.background_reference = np.median(background_box.reshape(-1, 3), axis=0)
            
            # Adjust the channels based on the median reference
            channel_medians = np.median(rgb_image, axis=(0, 1))

            # Adjust channels based on the red channel
            for channel in range(3):
                if self.background_reference[channel] < channel_medians[channel]:
                    pedestal = channel_medians[channel] - self.background_reference[channel]
                    rgb_image[..., channel] += pedestal

            # Specifically adjust G and B to match R
            r_median = self.background_reference[0]
            for channel in [1, 2]:  # Green and Blue channels
                if self.background_reference[channel] < r_median:
                    rgb_image[..., channel] += (r_median - self.background_reference[channel])

        self.status_update.emit("Background neutralization complete.")
        QCoreApplication.processEvents()
        return rgb_image
    
    def normalize_channel(self, image_channel, reference_channel):
        mad_image = np.mean(np.abs(image_channel - np.mean(image_channel)))
        mad_ref = np.mean(np.abs(reference_channel - np.mean(reference_channel)))

        median_image = np.median(image_channel)
        median_ref = np.median(reference_channel)

        # Apply the normalization formula
        normalized_channel = (
            image_channel * mad_ref / mad_image
            - (mad_ref / mad_image) * median_image
            + median_ref
        )

        self.status_update.emit("Color calibration complete.")
        QCoreApplication.processEvents()
        return np.clip(normalized_channel, 0, 1)  



    def continuum_subtraction(self, rgb_image):
        red_channel = rgb_image[..., 0]
        green_channel = rgb_image[..., 1]
        
        # Determine Q based on the selection (modify condition based on actual UI element)
        Q = 0.9 if self.output_linear else 1.0

        # Perform the continuum subtraction
        median_green = np.median(green_channel)
        result_image = red_channel - Q * (green_channel - median_green)
        
        return np.clip(result_image, 0, 1)  # Ensure values stay within [0, 1]
