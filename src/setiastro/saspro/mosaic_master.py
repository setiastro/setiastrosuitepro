# saspro/mosaic_master.py
"""
Mosaic Master + WCS/reprojection helpers.

Split out of star_alignment.py. Owns MosaicMasterDialog and its supporting
pieces (preview window, settings dialog, polynomial gradient removal, WCS
header sanitizing/parsing, astrometry.net blind-solve helpers, background
matching, and the view picker).

qs_int/qs_float/qs_bool and ASTROMETRY_API_URL still live in star_alignment.py
and are imported below. Callers that still do
`from ...star_alignment import MosaicMasterDialog` keep working via the
backward-compat __getattr__ shim in that module.
"""
from __future__ import annotations

import os
import sys
import math
import re
import json
import tempfile
import traceback
import warnings
import threading as _threading

import numpy as np
import numpy.ma as ma
import requests
import cv2

from astropy.io import fits
from astropy.io.fits import Header
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt6.QtCore import Qt, QPoint, QEvent, QProcess, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QComboBox, QMessageBox, QCheckBox,
    QFormLayout, QScrollArea, QFileDialog, QProgressDialog, QDialogButtonBox,
    QInputDialog, QApplication,QGroupBox,QSlider, QSpinBox, QDoubleSpinBox
)

from setiastro.saspro import astroalign
from setiastro.saspro.memory_utils import smart_zeros
from setiastro.saspro.legacy.image_manager import load_image, save_image
from setiastro.saspro.blink_comparator_pro import CustomDoubleSpinBox, CustomSpinBox
from setiastro.saspro.abe import _generate_sample_points as abe_generate_sample_points

try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

# --- interdependencies that remain in star_alignment.py ---
from setiastro.saspro.star_alignment import qs_int, qs_float, qs_bool, ASTROMETRY_API_URL

_AA_LOCK = _threading.Lock()

def _aa_find_transform_with_backoff(tgt_gray: np.ndarray, src_gray: np.ndarray):
    """
    Retry astroalign.find_transform() with progressively stricter detection,
    serializing SEP usage via _AA_LOCK; returns (transform_obj, (src_pts, tgt_pts)).
    """
    tgt32 = np.ascontiguousarray(tgt_gray.astype(np.float32))
    src32 = np.ascontiguousarray(src_gray.astype(np.float32))
    try:
        curr = sep.get_extract_pixstack()
        if curr < 1_500_000:
            sep.set_extract_pixstack(1_500_000)
    except Exception:
        pass

    tries = [
        dict(detection_sigma=120, min_area=5, max_control_points=400),
        dict(detection_sigma=50,  min_area=5, max_control_points=500),
        dict(detection_sigma=15,  min_area=5, max_control_points=600),
        dict(detection_sigma=5,  min_area=5, max_control_points=600),
    ]
    last_exc = None
    for kw in tries:
        try:
            global _AA_LOCK
            with _AA_LOCK:
                tform, (src_pts, tgt_pts) = astroalign.find_transform(tgt32, src32, **kw)

            # Decompose for a readable sanity check. For a panel already placed
            # by WCS these should be ~0 rotation, ~1 scale, few-px shift; large
            # values mean astroalign is fitting error the WCS didn't have.
            try:
                rot_deg = float(np.degrees(tform.rotation))
                scale = float(tform.scale)
                tx, ty = (float(tform.translation[0]),
                          float(tform.translation[1]))
                shift = float(np.hypot(tx, ty))
                n_match = len(src_pts)
                print(f"[astroalign] sigma={kw['detection_sigma']}: "
                      f"{n_match} stars matched | "
                      f"rot={rot_deg:+.3f} deg  scale={scale:.5f}  "
                      f"shift={shift:.2f}px (dx={tx:+.2f}, dy={ty:+.2f})")
                if abs(rot_deg) > 0.5 or abs(scale - 1.0) > 0.01 or shift > 25:
                    print(f"[astroalign]   WARNING: large correction — "
                          f"astroalign may be fitting a registration error "
                          f"the WCS did not have")
            except Exception as _pe:
                print(f"[astroalign] sigma={kw['detection_sigma']}: matched, "
                      f"transform print failed: {_pe}")

            return tform, (src_pts, tgt_pts)
        except Exception as e:
            last_exc = e
            if "internal pixel buffer full" in str(e).lower():
                try:
                    sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 5))
                except Exception:
                    pass
            continue
    raise last_exc

# --------------------------------------------------
# MosaicMasterDialog with blending/normalization integrated
# --------------------------------------------------
_NUM_FLOAT = {
    "CRPIX1","CRPIX2","CRVAL1","CRVAL2","CDELT1","CDELT2",
    "CD1_1","CD1_2","CD2_1","CD2_2","CROTA1","CROTA2","EQUINOX"
}
_3RD_AXIS_PREFIXES = ("NAXIS3","CTYPE3","CUNIT3","CRVAL3","CRPIX3","CDELT3","CD3_","PC3_","PC_3")

def _coerce_num(val, tp=float):
    if isinstance(val, (int, float)): return tp(val)
    s = str(val).strip().strip("'").strip()
    m = re.match(r"^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?", s)
    if m:
        return tp(float(m.group(0)))
    raise ValueError

def coerce_to_header(hdr_in):
    """
    Accept whatever the app hands us as a header and return an astropy Header,
    or None.

    Project files serialize headers to JSON, so a restored view carries a list
    of [key, value, comment] rows -- sometimes wrapped in an extra list -- while
    an on-disk load returns a real Header. Everything downstream wants a Header.
    """
    if hdr_in is None:
        return None
    if isinstance(hdr_in, Header):
        return hdr_in.copy()

    if isinstance(hdr_in, (bytes, bytearray)):
        try:
            hdr_in = hdr_in.decode("utf-8", "replace")
        except Exception:
            return None

    if isinstance(hdr_in, str):
        s = hdr_in.strip()
        if s[:1] in ("[", "{"):
            try:
                import json
                hdr_in = json.loads(s)
            except Exception:
                return None
        else:
            try:
                return Header.fromstring(s, sep="\n") if "\n" in s else Header.fromstring(s)
            except Exception:
                return None

    rows = []
    if isinstance(hdr_in, dict):
        # ProjectWriter._serialize_header_any wraps headers for JSON. Unwrap
        # before treating the dict's own keys as FITS cards -- otherwise the
        # wrapper's "format" key becomes the only card and the real WCS is
        # dropped on the floor.
        fmt = str(hdr_in.get("format", "")).lower()
        if fmt == "fits-cards" and isinstance(hdr_in.get("cards"), (list, tuple)):
            return coerce_to_header(hdr_in["cards"])
        if fmt == "dict" and isinstance(hdr_in.get("items"), dict):
            return coerce_to_header(hdr_in["items"])
        if fmt in ("repr", "unknown"):
            # repr(Header) is the concatenated card images -- fromstring can
            # read it back. Anything else is unrecoverable.
            txt = hdr_in.get("text")
            return coerce_to_header(txt) if isinstance(txt, str) else None
        rows = [(k, v, "") for k, v in hdr_in.items()]
    elif isinstance(hdr_in, (list, tuple)):
        seq = list(hdr_in)
        # Unwrap [[row, row, ...]] -- one extra level of nesting from the
        # serializer. Loop, in case there's more than one.
        while (len(seq) == 1 and isinstance(seq[0], (list, tuple))
               and seq[0] and isinstance(seq[0][0], (list, tuple))):
            seq = list(seq[0])
        for row in seq:
            if isinstance(row, (list, tuple)):
                if len(row) >= 3:
                    rows.append((row[0], row[1], row[2]))
                elif len(row) == 2:
                    rows.append((row[0], row[1], ""))
            elif hasattr(row, "keyword") and hasattr(row, "value"):   # a Card
                rows.append((row.keyword, row.value, getattr(row, "comment", "")))
    else:
        return None

    hdr = Header()
    for k, v, c in rows:
        try:
            key = str(k).strip().upper()
            # COMMENT/HISTORY need add_comment/add_history and repeat; none of
            # them carry WCS, so drop them.
            if not key or key in ("END", "COMMENT", "HISTORY", ""):
                continue
            if not (v is None or isinstance(v, (str, bool, int, float))):
                continue
            hdr[key] = (v, str(c) if c is not None else "")
        except Exception:
            continue

    return hdr if len(hdr) else None

def sanitize_wcs_header(hdr_in):
    """Return a cleaned astropy Header suitable for WCS(relax=True) with SIP kept."""
    if hdr_in is None:
        return None
    hdr = coerce_to_header(hdr_in)
    if hdr is None:
        return None

    # Drop any lingering 3rd-axis WCS bits
    for k in list(hdr.keys()):
        if any(k.startswith(pref) for pref in _3RD_AXIS_PREFIXES):
            try: del hdr[k]
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    # Minimal, sane defaults
    if not hdr.get("CTYPE1"): hdr["CTYPE1"] = "RA---TAN"
    if not hdr.get("CTYPE2"): hdr["CTYPE2"] = "DEC--TAN"

    # RADECSYS -> RADESYS (modern key)
    if "RADESYS" not in hdr and "RADECSYS" in hdr:
        hdr["RADESYS"] = str(hdr["RADECSYS"]).strip()
        try: del hdr["RADECSYS"]
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    # Coerce common numeric keys to the right types
    for k in _NUM_FLOAT:
        if k in hdr:
            try: hdr[k] = _coerce_num(hdr[k], float)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    # Declared inverse-SIP order with no coefficients is worse than none --
    # astropy builds a zero polynomial from it.
    if not any(re.match(r"^(AP|BP)_\d+_\d+$", str(k)) for k in hdr.keys()):
        for k in ("AP_ORDER", "BP_ORDER"):
            if k in hdr:
                del hdr[k]

    # SIP orders: ensure ints + pair up A/B and AP/BP if one is missing
    for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
        if k in hdr:
            try: hdr[k] = _coerce_num(hdr[k], int)
            except Exception: del hdr[k]
    if "A_ORDER" in hdr and "B_ORDER" not in hdr: hdr["B_ORDER"] = hdr["A_ORDER"]
    if "B_ORDER" in hdr and "A_ORDER" not in hdr: hdr["A_ORDER"] = hdr["B_ORDER"]
    if "AP_ORDER" in hdr and "BP_ORDER" not in hdr: hdr["BP_ORDER"] = hdr["AP_ORDER"]
    if "BP_ORDER" in hdr and "AP_ORDER" not in hdr: hdr["AP_ORDER"] = hdr["BP_ORDER"]

    # Keep axes sane
    if "WCSAXES" in hdr:
        try: hdr["WCSAXES"] = _coerce_num(hdr["WCSAXES"], int)
        except Exception: del hdr["WCSAXES"]
    if "NAXIS" in hdr:
        try: hdr["NAXIS"] = _coerce_num(hdr["NAXIS"], int)
        except Exception: del hdr["NAXIS"]

    return hdr

def _wcs_is_usable(w, hdr=None) -> bool:
    """
    A header carrying only CTYPE1/CTYPE2 still gives is_celestial == True --
    astropy fills in CRVAL=(0,0), CRPIX=(0,0), CDELT=(1,1). That is not an
    astrometric solution; it silently places every panel at the origin.
    Require the keys that make a solution a solution.
    """
    if w is None:
        return False
    try:
        if not w.is_celestial:
            return False
    except Exception:
        return False

    if hdr is not None:
        has_ref = all(k in hdr for k in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"))
        has_cd    = any(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
        has_pc    = any(k in hdr for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2"))
        has_cdelt = ("CDELT1" in hdr and "CDELT2" in hdr)
        if not (has_ref and (has_cd or has_pc or has_cdelt)):
            return False

    # Scale sanity: astro pixel scales are arcsec/px. The astropy default of
    # 1 deg/px is orders of magnitude past anything real.
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        sc = np.asarray(proj_plane_pixel_scales(w), float)
        if not np.all(np.isfinite(sc)) or np.any(sc <= 0.0) or np.any(sc > 1.0):
            return False
    except Exception:
        pass

    return True

def get_wcs_from_header(header):
    """Build a WCS while keeping SIP terms; suppress fix warnings; force 2D if needed."""
    if not header:
        return None
    hdr = sanitize_wcs_header(header)
    if hdr is None:
        return None

    naxis = 2 if hdr.get("NAXIS", 2) > 2 else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        try:
            w = WCS(hdr, naxis=naxis, relax=True)  # relax=True keeps SIP/AP/BP
            return w if _wcs_is_usable(w, hdr) else None
        except Exception:
            try:
                w = WCS(hdr, naxis=2, relax=True)
                return w if _wcs_is_usable(w, hdr) else None
            except Exception:
                return None
            
def robust_api_request(method, url, data=None, files=None, prompt_on_failure=False):
    """
    Sends an API request without automatic retries. If the request fails (network error or invalid JSON response),
    prompts the user if they want to start completely over. If the user chooses to try again,
    the function calls itself recursively.
    """
    try:
        if method == "GET":
            response = requests.get(url, timeout=600)
        elif method == "POST":
            response = requests.post(url, data=data, files=files, timeout=600)
        else:
            raise ValueError("Unsupported request method: " + method)

        response.raise_for_status()  # Raise HTTP errors (e.g., 500, 404)

        try:
            return response.json()  # Attempt to parse JSON
        except json.JSONDecodeError:
            error_message = f"Invalid JSON response from {url}."
            print(error_message)
            if prompt_on_failure:
                user_choice = QMessageBox.question(
                    None,
                    "Invalid Response",
                    f"{error_message}\nDo you want to start over?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if user_choice == QMessageBox.StandardButton.Yes:
                    return robust_api_request(method, url, data, files, prompt_on_failure=prompt_on_failure)
                else:
                    return None
            else:
                return None

    except requests.exceptions.RequestException as e:
        error_message = f"Network error when contacting {url}: {e}."
        print(error_message)
        if prompt_on_failure:
            user_choice = QMessageBox.question(
                None,
                "Network Error",
                f"{error_message}\nDo you want to start over?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if user_choice == QMessageBox.StandardButton.Yes:
                return robust_api_request(method, url, data, files, prompt_on_failure=prompt_on_failure)
            else:
                return None
        else:
            return None


def scale_image_for_display(image):
    """
    Scales a floating point image (0-1) to 8-bit (0-255) for display.
    """
    if np.max(image) == np.min(image):
        return np.zeros_like(image, dtype=np.uint8)  # Prevent division by zero
    scaled = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
    return scaled

def generate_minimal_fits_header(image):
    header = Header()
    header['SIMPLE'] = True
    # Set BITPIX according to the image’s data type.
    if np.issubdtype(image.dtype, np.integer):
        header['BITPIX'] = 16  # For 16-bit integer data.
    elif np.issubdtype(image.dtype, np.floating):
        header['BITPIX'] = -32  # For 32-bit float data.
    else:
        raise ValueError("Unsupported image data type for FITS header generation.")
    header['NAXIS'] = 2
    header['NAXIS1'] = image.shape[1]  # width
    header['NAXIS2'] = image.shape[0]  # height
    header['COMMENT'] = "Minimal header generated for blind solve"
    return header


class MosaicPreviewWindow(QDialog):
    def __init__(self, image_array, title="", parent=None, push_cb=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Preview")
        self._push_cb = push_cb

        # Keep the original array around for re-stretch or reset
        self.original_array = image_array.copy()
        # Current displayed array (8-bit or whatever you want)
        self.image_array = image_array.copy()

        # Zoom state
        self.zoom_factor = 1.0

        # Variables for panning (dragging)
        self.dragging = False
        self.last_mouse_pos = QPoint()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        # 1) QScrollArea to enable scrollbars for large images
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # 2) Label inside the scroll area
        self.preview_label = QLabel("No image yet.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)
        self.scroll_area.viewport().installEventFilter(self)

        # 3) Auto-Stretch Toggle
        self.stretch_toggle = QCheckBox("Auto-Stretch for Display")
        self.stretch_toggle.setChecked(True)
        self.stretch_toggle.stateChanged.connect(self.update_display)
        layout.addWidget(self.stretch_toggle)

        # 4) Button row (Zoom, Fit, etc.)
        button_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_out_button)

        self.fit_button = QPushButton("Fit to Preview")
        self.fit_button.clicked.connect(self.fit_to_preview)
        button_layout.addWidget(self.fit_button)

        self.autostretch_button = QPushButton("Reapply Stretch")
        self.autostretch_button.clicked.connect(self.autostretch_image)
        button_layout.addWidget(self.autostretch_button)

        self.push_btn = QPushButton("Push to New View")
        self.push_btn.clicked.connect(lambda: self._push_cb() if callable(self._push_cb) else None)
        button_layout.addWidget(self.push_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        # Finally, display the initial image
        self.display_image(self.image_array)

    def display_image(self, arr):
        """
        Convert array to QPixmap and display in preview_label.
        We'll respect self.zoom_factor to scale the pixmap.
        """
        if arr is None or arr.size == 0:
            print("WARNING: Trying to display an empty image.")
            return

        # Possibly apply auto-stretch
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr)
        else:
            # If it's already 8-bit or float, just convert to 8-bit safely
            arr_display = self.to_8bit(arr)
        
        # Convert single-channel => 3 channels if needed
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        # Make QImage => QPixmap
        h, w, c = arr_3ch.shape
        bytes_per_line = w * c
        qimg = QImage(arr_3ch.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Apply zoom factor
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        if new_w < 1: new_w = 1
        if new_h < 1: new_h = 1

        scaled_pixmap = pixmap.scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # Set the label to the scaled pixmap
        self.preview_label.setPixmap(scaled_pixmap)
        # Important: set the label size so the scroll area can scroll if it's bigger
        self.preview_label.resize(scaled_pixmap.size())

    def stretch_for_display(self, arr):
        """
        Applies an auto-stretch to improve visualization:
          1) Compute 0.5 and 99.5 percentiles
          2) Rescale to [0..255]
        """
        arr = arr.astype(np.float32, copy=False)
        mn, mx = np.percentile(arr, (0.5, 99.5))
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    def eventFilter(self, source, event):
        """
        Capture mouse events on the scroll_area.viewport():
          - Left-button press => start dragging
          - Mouse move => if dragging, pan
          - Left-button release => stop dragging
          - Wheel => zoom in/out
        """
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    return True  # We handled it
            elif event.type() == QEvent.Type.MouseMove:
                if self.dragging:
                    # Compute how far we moved
                    delta = event.pos() - self.last_mouse_pos
                    self.last_mouse_pos = event.pos()

                    # Adjust scrollbars
                    h_bar = self.scroll_area.horizontalScrollBar()
                    v_bar = self.scroll_area.verticalScrollBar()
                    h_bar.setValue(h_bar.value() - delta.x())
                    v_bar.setValue(v_bar.value() - delta.y())

                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.dragging = False
                    return True
            elif event.type() == QEvent.Type.Wheel:
                # Zoom in or out
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                event.accept()
                return True
        return super().eventFilter(source, event)

    def to_8bit(self, arr):
        """
        Simple fallback if not using auto-stretch:
        - If float in [0..1], multiply by 255
        - If already 8-bit, do nothing
        """
        if arr.dtype == np.uint8:
            return arr
        # else assume float [0..1]
        return (arr*255).clip(0,255).astype(np.uint8)

    def update_display(self):
        """
        Called when toggling the stretch checkbox. Re-display the image.
        """
        self.display_image(self.image_array)

    def autostretch_image(self):
        """
        Auto-stretch the original image and update preview.
        If the image has multiple channels, we do the same approach as stretch_for_display
        but typically you'd do something more advanced for color.
        """
        arr = self.original_array.copy()
        # e.g., if color, you might do a channel-by-channel approach
        # For simplicity, let's do a grayscale approach using the mean:
        if arr.ndim == 3:
            # we create a single channel for stretch
            g = np.mean(arr, axis=-1)
            arr_stretched = self.stretch_for_display(g)
            # Then broadcast back to 3 channels
            arr_stretched = np.stack([arr_stretched]*3, axis=-1)
        else:
            arr_stretched = self.stretch_for_display(arr)
        self.image_array = arr_stretched
        self.display_image(self.image_array)

    # ---------------------
    # ZOOM Methods
    # ---------------------
    
    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.display_image(self.image_array)

    
    def zoom_out(self):
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.05:
            self.zoom_factor = 0.05
        self.display_image(self.image_array)

    def fit_to_preview(self):
        """
        Scale the image so it fits inside the scroll_area's viewport.
        We'll measure the image's actual size, compare to the viewport,
        and adjust zoom_factor accordingly.
        """
        if self.image_array is None or self.image_array.size == 0:
            return

        # We'll figure out the image's *unzoomed* dimensions
        arr_display = self.image_array
        if self.stretch_toggle.isChecked():
            arr_display = self.stretch_for_display(arr_display)
        else:
            arr_display = self.to_8bit(arr_display)

        # Convert single-channel => 3 channels if needed, to find w,h
        if arr_display.ndim == 2:
            arr_3ch = np.stack([arr_display]*3, axis=-1)
        elif arr_display.ndim == 3 and arr_display.shape[2] == 1:
            arr_3ch = np.concatenate([arr_display, arr_display, arr_display], axis=2)
        else:
            arr_3ch = arr_display

        h, w, c = arr_3ch.shape

        # The scroll area viewport size
        viewport_size = self.scroll_area.viewport().size()
        vw, vh = viewport_size.width(), viewport_size.height()

        # Compute the scale factor to fit image inside viewport
        scale_w = vw / w if w else 1.0
        scale_h = vh / h if h else 1.0
        new_zoom = min(scale_w, scale_h)
        if new_zoom <= 0:
            new_zoom = 0.01

        self.zoom_factor = new_zoom
        self.display_image(self.image_array)

    def resizeEvent(self, event):
        """
        Refresh displayed pixmap when window is resized,
        only if we want the displayed image to keep fitting automatically.
        But typically, we won't auto-fit on window resize if user is controlling zoom.
        """
        super().resizeEvent(event)
        # Optionally do:
        # self.fit_to_preview()
        # or if you want to keep the user-chosen zoom, just re-display:
        self.display_image(self.image_array)

class MosaicSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        for _k in ("mosaic/num_stars",
                   "mosaic/translation_max_tolerance",
                   "mosaic/scale_min_tolerance",
                   "mosaic/scale_max_tolerance",
                   "mosaic/rotation_max_tolerance",
                   "mosaic/skew_max_tolerance"):
            if self.settings.contains(_k):
                self.settings.remove(_k)        
        self.setWindowTitle("Mosaic Master Settings")
        self.setMinimumWidth(400)
        self._build_ui()

    @staticmethod
    def _dspin(minimum, maximum, value, step, decimals=2):
        s = QDoubleSpinBox()
        s.setRange(minimum, maximum)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setKeyboardTracking(False)   # commit on Enter/focus-out, not per digit
        return s

    @staticmethod
    def _ispin(minimum, maximum, value, step=1):
        s = QSpinBox()
        s.setRange(minimum, maximum)
        s.setSingleStep(step)
        s.setValue(value)
        s.setKeyboardTracking(False)
        return s

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ---------------- Seam blending (always applies) ----------------
        blend_box = QGroupBox("Seam Blending")
        blend_form = QFormLayout(blend_box)

        saved_sharp = qs_float(self.settings, "mosaic/blend_sharp", 2.0)
        self.blendSharpSlider = QSlider(Qt.Orientation.Horizontal)
        self.blendSharpSlider.setRange(10, 60)      # value/10 -> 1.0 .. 6.0
        self.blendSharpSlider.setSingleStep(1)
        self.blendSharpSlider.setValue(int(round(saved_sharp * 10)))
        self.blendSharpSlider.setToolTip(
            "How sharply panels hand off across a seam.\n"
            "Low (1-2): wide, gradual blend -- hides large-scale brightness\n"
            "  differences, but needs good registration or stars can ghost.\n"
            "High (4-6): tight crossover near the seam midline -- safest when\n"
            "  panels disagree slightly, at the cost of visible gradient steps.\n"
            "Scale-free: same feel on thin and wide overlaps alike."
        )
        self.blendSharpValue = QLabel(f"{saved_sharp:.1f}")
        self.blendSharpValue.setMinimumWidth(28)
        self.blendSharpSlider.valueChanged.connect(
            lambda v: self.blendSharpValue.setText(f"{v / 10:.1f}"))

        row = QHBoxLayout()
        row.addWidget(self.blendSharpSlider, 1)
        row.addWidget(self.blendSharpValue)
        blend_form.addRow("Blend sharpness:", row)
        root.addWidget(blend_box)

        # ---------------- Star alignment (only when it runs) ----------------
        star_box = QGroupBox("Star Alignment")
        star_box.setToolTip(
            "Applies only to panels aligned on stars -- blind-solved or Seestar\n"
            "panels with no SIP. Panels with a SIP astrometric solution are\n"
            "placed by WCS and ignore these.")
        star_form = QFormLayout(star_box)

        self.fwhmSpin = self._dspin(
            0.0, 20.0, qs_float(self.settings, "mosaic/star_fwhm", 3.0), 0.1)
        star_form.addRow("Star detection FWHM:", self.fwhmSpin)

        self.sigmaSpin = self._dspin(
            0.0, 10.0, qs_float(self.settings, "mosaic/star_sigma", 3.0), 0.1)
        star_form.addRow("Star detection sigma:", self.sigmaSpin)

        self.polyDegreeSpin = self._ispin(
            1, 6, qs_int(self.settings, "mosaic/poly_degree", 3))
        self.polyDegreeSpin.setToolTip(
            "Degree of the polynomial-warp refinement, when that transform\n"
            "mode is selected for a star-aligned panel.")
        star_form.addRow("Polynomial degree:", self.polyDegreeSpin)

        root.addWidget(star_box)

        # ---------------- Buttons ----------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults |
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults
                       ).clicked.connect(self._restore_defaults)
        root.addWidget(buttons)

    def _restore_defaults(self):
        self.blendSharpSlider.setValue(20)
        self.fwhmSpin.setValue(3.0)
        self.sigmaSpin.setValue(3.0)
        self.polyDegreeSpin.setValue(3)

    def accept(self):
        s = self.settings
        s.setValue("mosaic/blend_sharp", self.blendSharpSlider.value() / 10.0)
        s.setValue("mosaic/star_fwhm", float(self.fwhmSpin.value()))
        s.setValue("mosaic/star_sigma", float(self.sigmaSpin.value()))
        s.setValue("mosaic/poly_degree", int(self.polyDegreeSpin.value()))
        s.sync()
        super().accept()

class PolyGradientRemoval:
    """
    A headless class that replicates the polynomial background removal
    logic from GradientRemovalDialog, minus the RBF step and UI code.

    Flow:
      1) Stretch the image (unlinked linear stretch).
      2) Downsample.
      3) Build an exclusion mask that:
         - Skips zero-valued pixels in any channel.
         - Optionally skip user-specified mask areas if desired (can pass mask to process()).
      4) Generate sample points from corners, borders, quartiles, do gradient_descent_to_dim_spot, skip bright areas.
      5) Fit a polynomial background and subtract it.
      6) Re-normalize median, clip to [0..1].
      7) Unstretch the final image back to the original domain.
    """

    def __init__(
        self,
        image: np.ndarray,
        poly_degree: int = 2,
        downsample_scale: int = 5,
        num_sample_points: int = 100
    ):
        """
        Args:
            image (np.ndarray): Input image in [0..1], shape (H,W) or (H,W,3), float32 recommended.
            poly_degree (int): Polynomial degree (1=linear,2=quadratic).
            downsample_scale (int): Factor for area downsampling.
            num_sample_points (int): Number of sample points to generate.
        """
        self.image = image.copy()
        self.poly_degree = poly_degree
        self.downsample_scale = downsample_scale
        self.num_sample_points = num_sample_points

        # For the stretch/unstretch logic
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = False

    def process(self, user_exclusion_mask: np.ndarray = None) -> np.ndarray:
        # 1) Stretch
        stretched = self.pixel_math_stretch(self.image)

        # 2) Downsample
        small_stretched = self.downsample_image(stretched, self.downsample_scale)
        h_s, w_s = small_stretched.shape[:2]

        # --- NEW: downsample user mask (≥0.5 keeps) ---
        mask_small = None
        if user_exclusion_mask is not None:
            m = user_exclusion_mask.astype(np.float32)
            mask_small = cv2.resize(m, (w_s, h_s), interpolation=cv2.INTER_AREA) >= 0.5

        # 4) Generate sample points using ABE’s sampler (fallback if missing)
        sample_points = self._gen_sample_points(
            small_stretched,
            num_points=self.num_sample_points,
            exclusion_mask=mask_small,
            patch_size=15,  # or make this configurable
        )

        # 5) Fit polynomial on the downsampled image
        poly_background_small = self.fit_polynomial_gradient(
            small_stretched, sample_points, degree=self.poly_degree
        )

        # Upscale background to full size
        poly_background = self.upscale_background(
            poly_background_small, stretched.shape[:2]
        )

        # Subtract
        after_poly = stretched - poly_background

        # Re-normalize median to original
        original_median = float(np.median(stretched))
        after_poly = self.normalize_image(after_poly, original_median)

        # Clip
        after_poly = np.clip(after_poly, 0, 1)

        # 6) Unstretch
        corrected = self.unstretch_image(after_poly)
        return corrected

    # --- NEW helper: delegate to ABE sampler, with a robust fallback ---
    def _gen_sample_points(
        self,
        small_image: np.ndarray,
        num_points: int,
        exclusion_mask: np.ndarray | None,
        patch_size: int = 15,
    ) -> np.ndarray:
        if abe_generate_sample_points is not None:
            return abe_generate_sample_points(
                small_image, num_points=num_points,
                exclusion_mask=exclusion_mask, patch_size=patch_size
            )

        # Fallback: simple grid (still respects exclusion_mask)
        H, W = small_image.shape[:2]
        grid = max(3, int(np.sqrt(max(9, num_points))))
        xs = np.linspace(10, max(11, W - 11), grid, dtype=int)
        ys = np.linspace(10, max(11, H - 11), grid, dtype=int)
        pts = []
        for y in ys:
            for x in xs:
                if exclusion_mask is not None and not exclusion_mask[y, x]:
                    continue
                pts.append((x, y))
        if not pts:
            pts = [(W // 2, H // 2)]
        return np.asarray(pts, dtype=np.int32)

    # ---------------------------------------------------------------
    # Helper: Stretch / Unstretch
    # ---------------------------------------------------------------
    def pixel_math_stretch(self, image: np.ndarray) -> np.ndarray:
        """
        Unlinked linear stretch using your existing Numba functions.

        Steps:
        1) If single-channel, replicate to 3-ch so we can store stats & do consistent logic.
        2) For each channel c: subtract the channel's min => data is >= 0.
        3) Compute the median after min subtraction for that channel.
        4) Call the appropriate Numba function:
            - If single-channel (was originally 1-ch), call numba_mono_final_formula
            on the 1-ch array.
            - If 3-ch color, call numba_color_final_formula_unlinked.
        5) Clip to [0,1].
        6) Store self.stretch_original_mins / medians so we can unstretch later.
        """
        target_median = 0.25

        # 1) Handle single-channel => replicate to 3 channels
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            self.was_single_channel = True
            image_3ch = np.stack([image.squeeze()] * 3, axis=-1)
        else:
            self.was_single_channel = False
            image_3ch = image

        image_3ch = image_3ch.astype(np.float32, copy=True)

        H, W, C = image_3ch.shape
        # We assume C=3 now.

        self.stretch_original_mins = []
        self.stretch_original_medians = []

        # 2) Subtract min per channel
        for c in range(C):
            cmin = image_3ch[..., c].min()
            image_3ch[..., c] -= cmin
            self.stretch_original_mins.append(float(cmin))

        # 3) Compute median after min subtraction
        medians_after_sub = []
        for c in range(C):
            cmed = float(np.median(image_3ch[..., c]))
            medians_after_sub.append(cmed)
        self.stretch_original_medians = medians_after_sub

        # 4) Apply the final formula with your Numba functions
        if self.was_single_channel:
            # If originally single-channel, let's do a single pass with numba_mono_final_formula
            # on the single channel. We can do that by extracting one channel from image_3ch.
            # Then replicate the result to 3 channels, or keep it as 1-ch?
            # Typically we keep it as 1-ch in the end, so let's do that.

            # We'll just pick channel 0, run the mono formula, store it back in a 2D array.
            mono_array = image_3ch[..., 0]  # shape (H,W)
            cmed = medians_after_sub[0]     # The median for that channel
            # We call the numba function
            from setiastro.saspro.legacy.numba_utils import numba_mono_final_formula
            stretched_mono = numba_mono_final_formula(mono_array, cmed, target_median)

            # Now place it back into image_3ch for consistency
            for c in range(3):
                image_3ch[..., c] = stretched_mono
        else:
            # 3-channel unlinked
            medians_rescaled = np.array(medians_after_sub, dtype=np.float32)
            # 'image_3ch' is our 'rescaled'
            from setiastro.saspro.legacy.numba_utils import numba_color_final_formula_unlinked
            stretched_3ch = numba_color_final_formula_unlinked(
                image_3ch, medians_rescaled, target_median
            )
            image_3ch = stretched_3ch

        # 5) Clip to [0..1]
        np.clip(image_3ch, 0.0, 1.0, out=image_3ch)
        image = image_3ch
        return image


    def unstretch_image(self, image: np.ndarray) -> np.ndarray:
        """
        Calls the Numba-optimized unstretch function.
        """
        image = image.astype(np.float32, copy=True)

        # Convert lists to NumPy arrays for efficient Numba processing
        stretch_original_medians = np.array(self.stretch_original_medians, dtype=np.float32)
        stretch_original_mins = np.array(self.stretch_original_mins, dtype=np.float32)

        # Call the Numba function
        from setiastro.saspro.legacy.numba_utils import numba_unstretch
        unstretched = numba_unstretch(image, stretch_original_medians, stretch_original_mins)

        if self.was_single_channel:
            # Convert back to grayscale
            unstretched = np.mean(unstretched, axis=2, keepdims=True)

        return unstretched


    # ---------------------------------------------------------------
    # Helper: Downsample
    # ---------------------------------------------------------------
    def downsample_image(self, image: np.ndarray, scale: int=6) -> np.ndarray:
        """
        Downsamples with area interpolation.
        """
        h, w = image.shape[:2]
        new_w = max(1, w//scale)
        new_h = max(1, h//scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)



    # ---------------------------------------------------------------
    # 5) Fit Polynomial
    # ---------------------------------------------------------------
    def fit_polynomial_gradient(self, image: np.ndarray, sample_points: np.ndarray, degree: int = 2, patch_size: int = 15) -> np.ndarray:
        """
        Optimized polynomial background fitting.
        - Extracts sample points using vectorized NumPy median calculations.
        - Solves for polynomial coefficients in parallel.
        - Precomputes polynomial basis terms for efficiency.
        """

        H, W = image.shape[:2]
        half_patch = patch_size // 2
        num_samples = len(sample_points)

        # Convert sample points to NumPy arrays
        sample_points = np.array(sample_points, dtype=np.int32)
        x_coords, y_coords = sample_points[:, 0], sample_points[:, 1]

        # Precompute polynomial design matrix
        from setiastro.saspro.legacy.numba_utils import build_poly_terms, evaluate_polynomial
        A = build_poly_terms(x_coords, y_coords, degree)

        # Extract sample values efficiently
        if image.ndim == 3 and image.shape[2] == 3:
            # Color image
            background = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                # Extract patches and compute medians using vectorized NumPy operations
                z_vals = np.array([
                    np.median(image[max(0, y-half_patch):min(H, y+half_patch+1),
                                    max(0, x-half_patch):min(W, x+half_patch+1), c])
                    for x, y in zip(x_coords, y_coords)
                ], dtype=np.float32)

                # Solve for polynomial coefficients
                coeffs = np.linalg.lstsq(A, z_vals, rcond=None)[0]

                # Generate full polynomial background
                background[..., c] = evaluate_polynomial(H, W, coeffs, degree)

        else:
            # Grayscale image
            background = np.zeros((H, W), dtype=np.float32)

            z_vals = np.array([
                np.median(image[max(0, y-half_patch):min(H, y+half_patch+1),
                                max(0, x-half_patch):min(W, x+half_patch+1)])
                for x, y in zip(x_coords, y_coords)
            ], dtype=np.float32)

            # Solve for polynomial coefficients
            coeffs = np.linalg.lstsq(A, z_vals, rcond=None)[0]

            # Generate full polynomial background
            background = evaluate_polynomial(H, W, coeffs, degree)

        return background
    # ---------------------------------------------------------------
    # 6) Upscale
    # ---------------------------------------------------------------
    def upscale_background(self, background: np.ndarray, out_shape: tuple) -> np.ndarray:
        """
        Resizes 'background' to out_shape=(H,W) using OpenCV interpolation.
        """
        oh, ow = out_shape

        if background.ndim == 3 and background.shape[2] == 3:
            # Resizing each channel efficiently without looping in Python
            return np.stack([cv2.resize(background[..., c], (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                            for c in range(3)], axis=-1)
        else:
            return cv2.resize(background, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    # ---------------------------------------------------------------
    # 7) Normalize
    # ---------------------------------------------------------------
    def normalize_image(self, image: np.ndarray, target_median: float) -> np.ndarray:
        """
        Shift image so its median matches target_median.
        """
        cmed = np.median(image)
        diff = target_median - cmed
        return image + diff

settings = QSettings("SetiAstro", "Seti Astro Suite Pro")

def save_api_key(api_key):
    settings.setValue("astrometry_api_key", api_key)  # Save to QSettings
    print("API key saved.")

def load_api_key():
    api_key = settings.value("astrometry_api_key", "")  # Load from QSettings
    if api_key:
        print("API key loaded.")
    return api_key


def _as_hwc(img):
    a = np.asarray(img, dtype=np.float32)
    return a[:, :, None] if a.ndim == 2 else a


def estimate_background_level(img, *, tiles=16, keep_frac=0.15, valid=None,
                              min_tile_valid=0.60, clip_sigma=2.5, clip_iters=3):
    """
    Robust per-channel sky level for one panel.

    Ranks tiles by luminance median, keeps the darkest `keep_frac`, then
    sigma-clips the pixels in those tiles (hard on the high side to kill
    stars/nebulosity, loose on the low side) and takes the per-channel median.

    Tile *selection* is done on luminance so all channels are measured over the
    same sky region -- that preserves the panel's background color.

    Returns (levels, sigmas), each shape (C,).
    """
    a = _as_hwc(img)
    H, W, C = a.shape
    finite = np.isfinite(a).all(axis=2)
    valid = finite if valid is None else (valid.astype(bool) & finite)
    if not valid.any():
        return np.zeros(C, np.float32), np.zeros(C, np.float32)

    lum = a.mean(axis=2)
    ys = np.linspace(0, H, tiles + 1).astype(int)
    xs = np.linspace(0, W, tiles + 1).astype(int)

    tile_lum, boxes = [], []
    for i in range(tiles):
        for j in range(tiles):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            v = valid[y0:y1, x0:x1]
            if v.size == 0 or v.sum() < min_tile_valid * v.size:
                continue
            tile_lum.append(float(np.median(lum[y0:y1, x0:x1][v])))
            boxes.append((y0, y1, x0, x1))

    if boxes:
        order = np.argsort(np.asarray(tile_lum, np.float32))
        k = max(1, int(round(keep_frac * len(order))))
        sel = [boxes[t] for t in order[:k]]
    else:
        sel = [(0, H, 0, W)]

    chunks = [a[y0:y1, x0:x1][valid[y0:y1, x0:x1]] for (y0, y1, x0, x1) in sel]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.zeros(C, np.float32), np.zeros(C, np.float32)
    px = np.concatenate(chunks, axis=0)          # (N, C)

    l = px.mean(axis=1)
    keep = np.ones(l.shape, bool)
    for _ in range(clip_iters):
        m = float(np.median(l[keep]))
        s = 1.4826 * float(np.median(np.abs(l[keep] - m)))
        if not np.isfinite(s) or s <= 0:
            break
        nxt = keep & (l < m + clip_sigma * s) & (l > m - 6.0 * clip_sigma * s)
        if nxt.sum() < 64 or nxt.sum() == keep.sum():
            break
        keep = nxt

    lev = np.median(px[keep], axis=0).astype(np.float32)
    sig = (1.4826 * np.median(np.abs(px[keep] - lev), axis=0)).astype(np.float32)
    return lev, sig

def _overlap_blocks(src, ref, mask, block=16, min_frac=0.9):
    """
    Reduce an overlap to block means before fitting.

    Per-pixel, a sky-dominated overlap carries almost no signal variance, so an
    OLS slope is attenuated by var(signal)/(var(signal)+var(noise)) and
    collapses toward zero. Noise averages down as 1/block; real structure does
    not. After an NxN reduction the same fit is well conditioned.

    Returns (S, R), each (Nblocks, C), or (None, None) if there isn't enough.
    """
    s, r = _as_hwc(src), _as_hwc(ref)
    C = s.shape[2]

    ys, xs = np.nonzero(mask)
    if ys.size < 8 * block * block:
        return None, None

    y0, x0 = int(ys.min()), int(xs.min())
    bh = (int(ys.max()) + 1 - y0) // block
    bw = (int(xs.max()) + 1 - x0) // block
    if bh < 2 or bw < 2:
        return None, None
    y1, x1 = y0 + bh * block, x0 + bw * block

    w = mask[y0:y1, x0:x1].astype(np.float32)
    cnt = w.reshape(bh, block, bw, block).sum(axis=(1, 3))
    ok = cnt >= (min_frac * block * block)
    n = int(ok.sum())
    if n < 32:
        return None, None

    S = np.empty((n, C), np.float64)
    R = np.empty((n, C), np.float64)
    for c in range(C):
        sb = (s[y0:y1, x0:x1, c] * w).reshape(bh, block, bw, block).sum(axis=(1, 3))
        rb = (r[y0:y1, x0:x1, c] * w).reshape(bh, block, bw, block).sum(axis=(1, 3))
        S[:, c] = sb[ok] / cnt[ok]
        R[:, c] = rb[ok] / cnt[ok]
    return S, R

def solve_linear_match(src, ref, mask, *, fit_scale=True, pivot=None,
                       block=16, iters=4, clip=2.5, gain_limits=(0.05, 20.0),
                       verbose=True):
    """
    Robust per-channel solve of  ref ~= a*(src - p) + p + b  over `mask`,
    where p is the pivot (the common sky level).

    Pivoting about the sky decorrelates a and b: after the stage-1 pedestal
    match the residual between panels is purely multiplicative on signal
    *above* sky, so that is the quantity the gain should scale.

    Returns (a, b, n_used) with a, b shape (C,). Apply as:
        out = (src - p) * a + p + b
    """
    s0 = _as_hwc(src)
    C = s0.shape[2]
    a_out = np.ones(C, np.float32)
    b_out = np.zeros(C, np.float32)

    p = np.zeros(C, np.float64)
    if pivot is not None:
        pv = np.asarray(pivot, np.float64).ravel()
        if pv.size:
            p = np.resize(pv, C)

    S, R = _overlap_blocks(src, ref, mask, block=block)
    if S is None:
        if verbose:
            print("[match] overlap too small to fit")
        return a_out, b_out, 0
    n_blocks = S.shape[0]

    for c in range(C):
        x = S[:, c] - p[c]
        y = R[:, c] - p[c]
        good = np.isfinite(x) & np.isfinite(y)
        aa, bb = 1.0, 0.0

        for _ in range(iters):
            n = int(good.sum())
            if n < 16:
                break
            if fit_scale:
                A = np.column_stack([x[good], np.ones(n)])
                sol, *_ = np.linalg.lstsq(A, y[good], rcond=None)
                aa, bb = float(sol[0]), float(sol[1])
            else:
                aa, bb = 1.0, float(np.median(y[good] - x[good]))

            res = y - (aa * x + bb)
            m = float(np.median(res[good]))
            sd = 1.4826 * float(np.median(np.abs(res[good] - m)))
            if not np.isfinite(sd) or sd <= 0:
                break
            nxt = good & (np.abs(res - m) < clip * sd)
            if nxt.sum() < 16 or nxt.sum() == good.sum():
                break
            good = nxt

        if not (np.isfinite(aa) and np.isfinite(bb)):
            aa, bb = 1.0, 0.0

        clamped = float(np.clip(aa, *gain_limits))
        if verbose and abs(clamped - aa) > 1e-6:
            print(f"[match] ch{c}: gain {aa:.3f} clamped to {clamped:.3f}")
        a_out[c] = clamped
        b_out[c] = float(bb)

    return a_out, b_out, n_blocks

class _ViewPickerDialog(QDialog):
    """Multi-select 'Add from View'. Checkable rows instead of a single-pick
    dropdown, so a whole mosaic's worth of panels goes in with one OK."""

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add from View")
        self.resize(480, 440)
        self._entries = entries

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Check the views to add to the mosaic:"))

        self._list = QListWidget()
        for e in entries:
            it = QListWidgetItem(e["picker_label"])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if e["already"]:
                # Present but not selectable -- re-adding would create a second
                # entry under the same view:// key, and remove_selected strips
                # every row matching a key, so one removal would take both.
                it.setCheckState(Qt.CheckState.Unchecked)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                it.setToolTip("Already added to this mosaic.")
            else:
                it.setCheckState(Qt.CheckState.Checked)
                it.setToolTip(e["key"])
            self._list.addItem(it)
        v.addWidget(self._list, 1)

        row = QHBoxLayout()
        b_all = QPushButton("All")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QPushButton("None")
        b_none.clicked.connect(lambda: self._set_all(False))
        b_wcs = QPushButton("Only solved")
        b_wcs.setToolTip("Check only views that already carry a WCS.")
        b_wcs.clicked.connect(self._only_wcs)
        for b in (b_all, b_none, b_wcs):
            row.addWidget(b)
        row.addStretch(1)
        self._count = QLabel("")
        row.addWidget(self._count)
        v.addLayout(row)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self._list.itemChanged.connect(lambda _it: self._update_count())
        self._update_count()

    def _enabled_rows(self):
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.flags() & Qt.ItemFlag.ItemIsEnabled:
                yield i, it

    def _set_all(self, on):
        st = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for _i, it in self._enabled_rows():
            it.setCheckState(st)

    def _only_wcs(self):
        for i, it in self._enabled_rows():
            it.setCheckState(Qt.CheckState.Checked
                             if self._entries[i]["wcs"] is not None
                             else Qt.CheckState.Unchecked)

    def _update_count(self):
        self._count.setText(f"{len(self.selected_entries())} selected")

    def selected_entries(self):
        return [self._entries[i] for i, it in self._enabled_rows()
                if it.checkState() == Qt.CheckState.Checked]

class MosaicMasterDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None, image_manager=None,
                 doc_manager=None, wrench_path=None, spinner_path=None,
                 list_open_docs_fn=None):
        super().__init__(parent)
        self.settings = settings
        self.image_manager = image_manager
        self._docman = doc_manager or getattr(parent, "doc_manager", None)

        # same pattern as StellarAlignmentDialog
        if list_open_docs_fn is None:
            cand = getattr(parent, "_list_open_docs", None)
            self._list_open_docs_fn = cand if callable(cand) else None
        else:
            self._list_open_docs_fn = list_open_docs_fn

        self.setWindowTitle("Mosaic Master")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self.wrench_path = wrench_path
        self.spinner_path = spinner_path

        self.resize(600, 400)
        self.loaded_images = []
        self.final_mosaic = None
        self.weight_mosaic = None
        self.wcs_metadata = None  # To store mosaic WCS header
        self.astap_exe = self.settings.value("paths/astap", "", type=str)

        # Variables to store stretching parameters:
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = False
        self._migrate_mosaic_settings()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Mosaic Master:\n"
            "1) Add images — linear FITS with a plate solve (WCS) work best\n"
            "2) Choose Reprojection Mode:\n"
            "....Exact — SIP-aware remap (recommended): reprojects each panel by its\n"
            "....   full WCS/SIP solution. Handles scale, rotation, flip, and field of\n"
            "....   view automatically — no manual tuning.\n"
            "....Fast — Homography: one global transform per panel. Quicker, but only\n"
            "....   exact for SIP-free TAN panels.\n"
            "3) Align & Create Mosaic\n"
            "4) Save to New View\n"
            "\n"
            "Advanced: the Transformation Type below only affects the optional star\n"
            "refinement after reprojection — leave it on Affine unless you have a reason."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        btn_layout = QHBoxLayout()

        add_btn = QPushButton("Add Image")
        add_btn.clicked.connect(self.add_image)
        btn_layout.addWidget(add_btn)

        add_from_view_btn = QPushButton("Add from View")
        add_from_view_btn.setToolTip("Add an image from any open View")
        add_from_view_btn.clicked.connect(self.add_image_from_view)
        btn_layout.addWidget(add_from_view_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)

        preview_btn = QPushButton("Preview Selected")
        preview_btn.clicked.connect(self.preview_selected)
        btn_layout.addWidget(preview_btn)

        align_btn = QPushButton("Align and Create Mosaic")
        align_btn.clicked.connect(self.align_images)
        btn_layout.addWidget(align_btn)

        save_btn = QPushButton("Save to New View")
        save_btn.clicked.connect(self.save_mosaic_to_new_view)
        btn_layout.addWidget(save_btn)

        wrench_btn = QPushButton()
        if self.wrench_path:
            wrench_btn.setIcon(QIcon(self.wrench_path))
        wrench_btn.setToolTip("Mosaic Settings")
        wrench_btn.clicked.connect(self.openSettings)
        btn_layout.addWidget(wrench_btn)

        layout.addLayout(btn_layout)

        # ------------------------------------------------------------------
        # Mode checkboxes (mutually exclusive: Seestar vs WCS-only)
        # ------------------------------------------------------------------
        checkbox_layout = QHBoxLayout()

        self.forceBlindCheckBox = QCheckBox("Force Blind Solve (ignore existing WCS)")
        checkbox_layout.addWidget(self.forceBlindCheckBox)

        self.seestarCheckBox = QCheckBox("Seestar Mode")
        self.seestarCheckBox.setToolTip(
            "When enabled, images are aligned iteratively using astroalign without plate solving."
        )
        checkbox_layout.addWidget(self.seestarCheckBox)

        self.wcsOnlyCheckBox = QCheckBox("Disable Star Alignment (WCS placement only)")
        self.wcsOnlyCheckBox.setToolTip(
            "Skips astroalign + refined alignment.\n"
            "Panels are only reprojected into the mosaic celestial-sphere frame using WCS, then blended.\n"
            "Useful when panels have little/no overlap but have valid WCS."
        )
        checkbox_layout.addWidget(self.wcsOnlyCheckBox)

        layout.addLayout(checkbox_layout)

        # Persisted WCS-only
        self.wcsOnlyCheckBox.setChecked(qs_bool(self.settings, "mosaic/wcs_only", False))



        # Helpers ----------------------------------------------------------
        def _set_checked(cb: QCheckBox, checked: bool):
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

        def _sync_wcs_only_ui():
            # WCS-only disables transform selection (because refinement is unused)
            wcs_only = self.wcsOnlyCheckBox.isChecked()
            if hasattr(self, "transform_combo"):
                self.transform_combo.setEnabled(not wcs_only)
                self.transform_combo.setToolTip(
                    "" if not wcs_only else "Disabled because WCS-only placement is enabled."
                )

        def _on_seestar_changed(state: int):
            # If Seestar is turned ON, force WCS-only OFF
            if self.seestarCheckBox.isChecked():
                _set_checked(self.wcsOnlyCheckBox, False)
            _sync_wcs_only_ui()

        def _on_wcs_only_changed(state: int):
            # Persist setting first (this one is user-visible preference)
            self.settings.setValue("mosaic/wcs_only", self.wcsOnlyCheckBox.isChecked())
            self.settings.sync()

            # If WCS-only is turned ON, force Seestar OFF
            if self.wcsOnlyCheckBox.isChecked():
                _set_checked(self.seestarCheckBox, False)
            _sync_wcs_only_ui()

        # Wire handlers (NO lambdas, no duplicates)
        self.seestarCheckBox.stateChanged.connect(_on_seestar_changed)
        self.wcsOnlyCheckBox.stateChanged.connect(_on_wcs_only_changed)

        # ------------------------------------------------------------------
        # Other controls
        # ------------------------------------------------------------------
        self.normalizeCheckBox = QCheckBox("Normalize images (median match)")
        self.normalizeCheckBox.setChecked(True)
        layout.addWidget(self.normalizeCheckBox)

        self.autostretchPreviewCheck = QCheckBox("Autostretch previews only")
        self.autostretchPreviewCheck.setToolTip("Only stretch display previews; keep data linear for processing.")
        self.autostretchPreviewCheck.setChecked(True)
        layout.addWidget(self.autostretchPreviewCheck)

        # Reprojection mode (persisted)
        self.reprojectModeLabel = QLabel("Reprojection mode:")
        self.reprojectModeCombo = QComboBox()
        self.reprojectModeCombo.addItems([
            "Exact — SIP-aware remap (recommended)",    # default
            "Fast — Homography (Global H)",
        ])
        self.reprojectModeCombo.setToolTip(
            "Exact — SIP-aware remap: per-pixel inverse WCS (tiled), honors SIP,\n"
            "footprint-limited with round-trip validation. Lanczos4 resampling.\n\n"
            "Fast — Homography: one global 3x3 per panel. Exact only for SIP-free\n"
            "TAN panels; with SIP present it can double stars near the edges."
        )

        _default_mode = self.settings.value(
            "mosaic/reproject_mode", "Exact — SIP-aware remap (recommended)", type=str)

        valid_modes = [self.reprojectModeCombo.itemText(i) for i in range(self.reprojectModeCombo.count())]
        if _default_mode not in valid_modes:
            # Covers the old labels, including the removed "Precise — Full WCS".
            _default_mode = "Exact — SIP-aware remap (recommended)"
        self.reprojectModeCombo.setCurrentText(_default_mode)
        self.reprojectModeCombo.currentTextChanged.connect(self._on_reproject_mode_changed)

        self.settings.sync()

        row = QHBoxLayout()
        row.addWidget(self.reprojectModeLabel)
        row.addWidget(self.reprojectModeCombo, 1)
        layout.addLayout(row)

        # Transform selection
        self.transform_combo = QComboBox()
        self.transform_combo.addItems([
            "Partial Affine Transform",
            "Affine Transform",
            "Homography Transform",
            "Polynomial Warp Based Transform"
        ])
        self.transform_combo.setCurrentIndex(1)
        layout.addWidget(QLabel("Select Transformation Method:"))
        layout.addWidget(self.transform_combo)

        # Now that transform_combo exists, apply WCS-only UI state
        _sync_wcs_only_ui()

        # List + status + spinner
        self.images_list = QListWidget()
        self.images_list.setSelectionMode(self.images_list.SelectionMode.SingleSelection)
        layout.addWidget(self.images_list)

        self.status_label = QLabel("Status: no images")
        layout.addWidget(self.status_label)

        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerMovie = QMovie(self.spinner_path) if self.spinner_path else None
        if self.spinnerMovie:
            self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()
        layout.addWidget(self.spinnerLabel)
        self.wcsOnlyCheckBox.clicked.connect(
            lambda *_: setattr(self, "_wcsonly_user_touched", True))
        self.setLayout(layout)

    @staticmethod
    def _wcs_has_sip(w):
        """True only if the WCS carries a usable forward SIP polynomial
        (order >= 2 with coefficients), not just declared keywords."""
        if w is None:
            return False
        sip = getattr(w, "sip", None)
        if sip is None:
            return False
        try:
            a, b = getattr(sip, "a", None), getattr(sip, "b", None)
            order = int(getattr(sip, "a_order", 0) or 0)
            if a is None or b is None or order < 2:
                return False
            return bool(np.any(a != 0.0) or np.any(b != 0.0))
        except Exception:
            return False

    def _refresh_sip_autodisable(self):
        """After the loaded-panel set changes, default the star-alignment
        toggle based on whether the panels carry SIP. Never override a manual
        choice the user has already made this session."""
        box = getattr(self, "wcsOnlyCheckBox", None)
        if box is None or getattr(self, "_wcsonly_user_touched", False):
            return

        solved = [it for it in self.loaded_images if it.get("wcs") is not None]
        if not solved:
            return
        n_sip = sum(1 for it in solved if self._wcs_has_sip(it["wcs"]))
        frac = n_sip / len(solved)

        box.blockSignals(True)   # programmatic change must not trip _touched
        if frac >= 0.8:
            box.setChecked(True)
            box.setText("Disable star alignment  —  SIP distortion detected "
                        "(WCS placement is more accurate)")
        elif frac == 0.0:
            box.setChecked(False)
            box.setText("Disable star alignment  —  no SIP; star alignment "
                        "recommended")
        else:
            box.setChecked(False)
            box.setText(f"Disable star alignment  —  mixed panels "
                        f"({n_sip}/{len(solved)} have SIP)")
        box.blockSignals(False)

    def _collect_view_candidates(self):
        """Every open view that has image data, tagged with WCS/already-added."""
        already = {d.get("path") for d in self.loaded_images}
        entries = []

        for title, doc in self._iter_docs():
            try:
                img = (doc.get_image()
                       if hasattr(doc, "get_image") and callable(doc.get_image)
                       else getattr(doc, "image", None))
            except Exception:
                img = None
            if not isinstance(img, np.ndarray):
                continue

            try:
                meta = (doc.get_metadata()
                        if hasattr(doc, "get_metadata") and callable(doc.get_metadata)
                        else getattr(doc, "metadata", {}) or {})
            except Exception:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}

            raw_hdr = (meta.get("original_header") or meta.get("header")
                       or meta.get("fits_header"))
            try:
                header = sanitize_wcs_header(raw_hdr) if raw_hdr is not None else None
                wcs_obj = get_wcs_from_header(header) if header else None
            except Exception as e:
                print(f"[Mosaic] header unreadable for {title!r}: "
                      f"{type(e).__name__}: {e}")
                header, wcs_obj = None, None

            h, w = img.shape[:2]
            display_name = str(title).strip() if title else "View"
            key = f"view://{id(doc)}"

            tags = []
            if wcs_obj is not None:
                tags.append("WCS")
            if key in already:
                tags.append("added")

            base_label = f"{display_name} — {w}×{h}"
            entries.append({
                "key": key,
                "display": display_name,
                "base_label": base_label,
                "picker_label": base_label + (f"   [{', '.join(tags)}]" if tags else ""),
                "image": img,
                "meta": meta,
                "header": header,
                "wcs": wcs_obj,
                "already": key in already,
            })
        return entries

    def _on_reproject_mode_changed(self, t: str):
        self.settings.setValue("mosaic/reproject_mode", t)
        self.settings.sync()


    def _migrate_mosaic_settings(self):
        s = self.settings
        _ = qs_int(s, "mosaic/poly_degree", 3)
        _ = qs_float(s, "mosaic/star_sigma", 3.0)
        _ = qs_float(s, "mosaic/star_fwhm", 3.0)
        # etc...
        s.sync()

    def _target_median_from_first(self, items):
        # Pick a stable target (median of first image after safe clipping)
        if not items:
            return 0.1
        a0 = items[0]["image"].astype(np.float32)
        if a0.ndim == 3:
            a0 = np.mean(a0, axis=2)
        m = np.median(np.clip(a0, np.percentile(a0, 1), np.percentile(a0, 99)))
        return float(max(m, 1e-6))

    @staticmethod
    def _bg_offset(levels, like):
        """Reshape a per-channel level vector so it broadcasts against `like`."""
        v = np.asarray(levels, np.float32).ravel()
        if v.size == 0:
            return np.float32(0.0)
        if like.ndim == 2:
            return np.float32(v.mean())
        C = like.shape[2]
        if v.size < C:
            v = np.concatenate([v, np.repeat(v[-1:], C - v.size)])
        return v[:C].astype(np.float32).reshape(1, 1, C)

    @staticmethod
    def _strip_pedestal(img, ped, coverage):
        """Remove the transport pedestal and zero everything outside coverage."""
        out = (img - ped).astype(np.float32, copy=False)
        out[~coverage] = 0.0
        return out

    @staticmethod
    def _linearize_wcs(w):
        """SIP is only valid inside the frame it was fit to. The mosaic canvas
        is much larger than any single panel, so the destination frame must be
        pure linear TAN. Source panels keep their SIP — that's what makes the
        inverse mapping correct."""
        out = w.deepcopy()
        try:
            out.sip = None
        except Exception:
            pass
        try:
            out.wcs.ctype = [str(c).replace("-SIP", "") for c in out.wcs.ctype]
        except Exception:
            pass
        return out

    def _normalize_linear(self, arr, target_med):
        """Linear median match only (no stretch/curves). Returns float32 array."""
        img = arr.astype(np.float32, copy=False)
        mono = img if img.ndim == 2 else np.mean(img, axis=2)
        med = np.median(mono)
        if med <= 0:
            return img
        gain = target_med / med
        return img * gain

    def _autostretch_if_requested(self, arr):
        """Used only for preview windows based on the checkbox."""
        if not self.autostretchPreviewCheck.isChecked():
            # Convert linear [0..1-ish] to 8-bit safely for preview
            a = arr.astype(np.float32)
            a = a / max(1e-6, np.percentile(a, 99.9))
            return (np.clip(a, 0, 1) * 255).astype(np.uint8)

        # Your existing stretch_for_display logic:
        return self.stretch_for_display(arr)

    def _sample_grid_pts(self, w, h):
        """9 control points: corners, edge midpoints, center."""
        xs = [0, w/2, w-1]
        ys = [0, h/2, h-1]
        pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)  # shape (9,2)
        return pts

    @staticmethod
    def _radec_to_vec(ra_deg, dec_deg):
        r = np.radians(np.asarray(ra_deg, float))
        d = np.radians(np.asarray(dec_deg, float))
        cd = np.cos(d)
        return np.stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)], axis=-1)

    @staticmethod
    def _gnomonic(ra, dec, ra0, dec0):
        """TAN-project (ra, dec) about (ra0, dec0). Degrees in, degrees out.
        Handles the RA wrap correctly, unlike a naive coordinate difference."""
        r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
        r0 = np.radians(float(ra0)); d0 = np.radians(float(dec0))
        cd, sd = np.cos(d), np.sin(d)
        cdr = np.cos(r - r0)
        den = np.sin(d0) * sd + np.cos(d0) * cd * cdr
        den = np.where(np.abs(den) < 1e-12, np.nan, den)
        x = np.degrees(cd * np.sin(r - r0) / den)
        y = np.degrees((np.cos(d0) * sd - np.sin(d0) * cd * cdr) / den)
        return x, y

    def _order_panels(self, items):
        """
        Reorder panels so the sequence starts near the field centre and every
        later panel overlaps sky that has already been placed.

        Footprints are computed with all_pix2world only -- forward SIP is a
        plain polynomial evaluation, so nothing iterates and nothing diverges,
        and this runs before the mosaic WCS exists.
        """
        n = len(items)
        if n < 3:
            return list(items)

        NS = 16
        per_panel, vecs = [], []
        for itm in items:
            h, w = itm["image"].shape[:2]
            xs = np.linspace(0, w - 1, NS); ys = np.linspace(0, h - 1, NS)
            px = np.concatenate([xs, xs, np.zeros(NS), np.full(NS, w - 1)])
            py = np.concatenate([np.zeros(NS), np.full(NS, h - 1), ys, ys])
            try:
                ra, dec = itm["wcs"].all_pix2world(px, py, 0)
                g = np.isfinite(ra) & np.isfinite(dec)
                ra, dec = ra[g], dec[g]
            except Exception:
                ra = dec = np.array([])
            per_panel.append((ra, dec))
            if ra.size:
                vecs.append(self._radec_to_vec(ra, dec))

        if not vecs:
            return list(items)

        # Field centre as a unit vector -- immune to the RA wrap
        c = np.concatenate(vecs, axis=0).mean(axis=0)
        c /= max(float(np.linalg.norm(c)), 1e-12)
        ra0 = float(np.degrees(np.arctan2(c[1], c[0])))
        dec0 = float(np.degrees(np.arcsin(np.clip(c[2], -1.0, 1.0))))

        boxes = []
        for (ra, dec) in per_panel:
            if ra.size == 0:
                boxes.append(None); continue
            x, y = self._gnomonic(ra, dec, ra0, dec0)
            g = np.isfinite(x) & np.isfinite(y)
            boxes.append((float(x[g].min()), float(y[g].min()),
                          float(x[g].max()), float(y[g].max())) if g.any() else None)

        def area(b):
            return (b[2] - b[0]) * (b[3] - b[1]) if b else 0.0

        def frac_overlap(a, b):
            """Shared area as a fraction of the smaller panel."""
            if a is None or b is None:
                return 0.0
            dx = min(a[2], b[2]) - max(a[0], b[0])
            dy = min(a[3], b[3]) - max(a[1], b[1])
            if dx <= 0 or dy <= 0:
                return 0.0
            return (dx * dy) / max(1e-12, min(area(a), area(b)))

        def d2(b):
            if b is None:
                return float("inf")
            cx = 0.5 * (b[0] + b[2]); cy = 0.5 * (b[1] + b[3])
            return cx * cx + cy * cy

        order = [min(range(n), key=lambda i: d2(boxes[i]))]
        remaining = set(range(n)) - set(order)

        while remaining:
            best, best_ov = None, 0.0
            for i in remaining:
                # Against each placed panel individually -- NOT their union box,
                # which would fill in the hollow of an L and invent overlap.
                ov = max((frac_overlap(boxes[i], boxes[j]) for j in order), default=0.0)
                if ov > best_ov:
                    best, best_ov = i, ov
            if best is None:
                best = min(remaining, key=lambda i: d2(boxes[i]))
                print(f"[Mosaic] {self._item_label(items[best])} shares no sky with "
                      f"any placed panel -- WCS placement, pedestal-only sky match.")
            order.append(best)
            remaining.discard(best)

        out = [items[i] for i in order]
        print("[Mosaic] panel order: " + " -> ".join(self._item_label(x) for x in out))
        return out

    def _compute_wcs_homography(self, src_wcs, dst_wcs, src_shape, dst_shape):
        """
        Build a single 3x3 homography H that maps src pixel coords -> dst pixel coords
        by sampling a small grid of tie-points via WCS.
        """
        h, w = src_shape[:2]
        pts_src = self._sample_grid_pts(w, h)                           # (N,2)
        # src pixels -> world (RA,DEC)
        ra, dec = src_wcs.pixel_to_world_values(pts_src[:,0], pts_src[:,1])
        # world -> dst pixels
        x_dst, y_dst = dst_wcs.world_to_pixel_values(ra, dec)

        pts_dst = np.column_stack([x_dst, y_dst]).astype(np.float32)    # (N,2)

        # Robust H (covers rotation/scale/shear and mild curvature over small fields)
        H, mask = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        return H

    def _warp_via_wcs_homography(self, img, src_wcs, dst_wcs, out_shape, H_cache=None):
        """
        Fast path: single homography warp per image (per mosaic WCS).
        If H_cache is provided as a dict, we reuse computed H.
        """
        H, W = out_shape
        key = id(src_wcs)
        if H_cache is not None and key in H_cache:
            H33 = H_cache[key]
        else:
            H33 = self._compute_wcs_homography(src_wcs, dst_wcs, img.shape, (H, W))
            if H_cache is not None:
                H_cache[key] = H33

        if img.ndim == 2:
            return cv2.warpPerspective(img, H33, (W, H), flags=cv2.INTER_LANCZOS4)
        else:
            # Warp each channel
            out = np.empty((H, W, img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = cv2.warpPerspective(img[..., c], H33, (W, H), flags=cv2.INTER_LANCZOS4)
            return out

    def _dst_footprint_bbox(self, src_shape, src_wcs, dst_wcs, out_shape, pad=16, n=96):
        """
        Bounding box (x0, y0, x1, y1) of the source panel inside the destination
        canvas, computed forward (all_pix2world -> wcs_world2pix) so no SIP
        inversion is involved. Returns None if the panel misses the canvas.
        """
        H, W = int(out_shape[0]), int(out_shape[1])
        h, w = int(src_shape[0]), int(src_shape[1])

        xs = np.linspace(0, w - 1, n)
        ys = np.linspace(0, h - 1, n)
        px = np.concatenate([xs, xs, np.zeros(n), np.full(n, w - 1)])
        py = np.concatenate([np.zeros(n), np.full(n, h - 1), ys, ys])

        try:
            ra, dec = src_wcs.all_pix2world(px, py, 0)      # forward SIP: no iteration
            dx, dy = dst_wcs.wcs_world2pix(ra, dec, 0)      # dst is linear: exact
        except Exception:
            return (0, 0, W, H)

        good = np.isfinite(dx) & np.isfinite(dy)
        if not good.any():
            return None

        x0 = max(0, int(np.floor(dx[good].min())) - pad)
        y0 = max(0, int(np.floor(dy[good].min())) - pad)
        x1 = min(W, int(np.ceil(dx[good].max())) + pad + 1)
        y1 = min(H, int(np.ceil(dy[good].max())) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _warp_via_wcs_remap_exact(self, src_img, src_wcs, dst_wcs, out_shape,
                                  tile=512, progress_cb=None):
        """
        Inverse mapping dst -> world -> src, tiled, restricted to the panel's
        actual footprint in the destination frame. Outside that box the SIP
        inverse cannot converge (the polynomial has no inverse beyond its fit
        domain), so we never ask it to. Every solved coordinate is validated by
        a forward round trip before it is used.
        """
        import warnings

        H, W = int(out_shape[0]), int(out_shape[1])
        is_color = (src_img.ndim == 3)
        nch = src_img.shape[2] if is_color else 1
        sh, sw = src_img.shape[:2]

        dst = (np.zeros((H, W, nch), np.float32) if is_color
               else np.zeros((H, W), np.float32))

        box = self._dst_footprint_bbox(src_img.shape, src_wcs, dst_wcs, (H, W))
        if box is None:
            print("[Mosaic] panel footprint falls outside the mosaic canvas.")
            return dst
        bx0, by0, bx1, by1 = box

        frac = ((bx1 - bx0) * (by1 - by0)) / float(max(1, W * H))
        print(f"[Mosaic] footprint x[{bx0}:{bx1}] y[{by0}:{by1}] "
              f"({frac * 100:.1f}% of canvas)")

        # --- round-trip tolerance: a quarter of a source pixel, in degrees ---
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            scale_deg = float(np.mean(proj_plane_pixel_scales(src_wcs)))
        except Exception:
            scale_deg = 1.0 / 3600.0
        tol_deg = 0.25 * scale_deg

        rejected = 0
        n_tiles = (max(1, -(-(by1 - by0) // tile)) *
                   max(1, -(-(bx1 - bx0) // tile)))
        done_tiles = 0

        with warnings.catch_warnings():
            # Residual non-convergence at the footprint edge is expected and is
            # handled by the validation below, not by hoping it doesn't happen.
            warnings.filterwarnings("ignore", message=".*all_world2pix.*")
            warnings.filterwarnings("ignore", message=".*failed to converge.*")

            for y0 in range(by0, by1, tile):
                y1 = min(y0 + tile, by1)
                hh = y1 - y0
                ys = np.arange(y0, y1, dtype=np.float64)[:, None]

                for x0 in range(bx0, bx1, tile):
                    x1 = min(x0 + tile, bx1)
                    ww = x1 - x0
                    xs = np.arange(x0, x1, dtype=np.float64)[None, :]

                    Xd = np.broadcast_to(xs, (hh, ww))
                    Yd = np.broadcast_to(ys, (hh, ww))

                    # dst is linear now, so wcs_pix2world is exact and cheap
                    ra, dec = dst_wcs.wcs_pix2world(Xd.ravel(), Yd.ravel(), 0)
                    sx, sy = src_wcs.all_world2pix(ra, dec, 0)   # iterative SIP inverse

                    mapx = sx.reshape(hh, ww).astype(np.float32)
                    mapy = sy.reshape(hh, ww).astype(np.float32)

                    bad = ~(np.isfinite(mapx) & np.isfinite(mapy))
                    bad |= (mapx < -1.0) | (mapx > sw) | (mapy < -1.0) | (mapy > sh)

                    # Round-trip validation: a diverged solve that happens to land
                    # inside the frame still fails to map back to the sky we asked
                    # for. Forward SIP is a plain polynomial evaluation — no
                    # iteration, no divergence.
                    ok = ~bad
                    if ok.any():
                        rx = np.where(ok, mapx, 0.0).astype(np.float64).ravel()
                        ry = np.where(ok, mapy, 0.0).astype(np.float64).ravel()
                        ra2, dec2 = src_wcs.all_pix2world(rx, ry, 0)
                        dra = ((ra2 - ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec))
                        err = np.hypot(dra, dec2 - dec).reshape(hh, ww)
                        bad |= ~np.isfinite(err)
                        bad |= (err > tol_deg)

                    if bad.any():
                        rejected += int(bad.sum())
                        mapx = np.where(bad, np.float32(-1.0), mapx)
                        mapy = np.where(bad, np.float32(-1.0), mapy)

                    if is_color:
                        for c in range(nch):
                            dst[y0:y1, x0:x1, c] = cv2.remap(
                                src_img[..., c], mapx, mapy,
                                interpolation=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                    else:
                        dst[y0:y1, x0:x1] = cv2.remap(
                            src_img, mapx, mapy,
                            interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

                    done_tiles += 1
                    if progress_cb is not None and (done_tiles % 8) == 0:
                        progress_cb(done_tiles / max(1, n_tiles))

        if rejected:
            tot = (bx1 - bx0) * (by1 - by0)
            print(f"[Mosaic] rejected {rejected} / {tot} mapped pixels "
                  f"({100.0 * rejected / max(1, tot):.1f}% of footprint box)")

        return dst
    def _get_astrometry_api_key(self) -> str:
        # Prefer QSettings; fall back to your legacy loader if present.
        key = self.settings.value("api/astrometry_key", "", type=str)
        if key:
            return key
        try:
            return load_api_key() or ""
        except Exception:
            return ""

    def _is_view_path(self, p) -> bool:
        return isinstance(p, str) and p.startswith("view://")

    def _push_mosaic_to_new_doc(self):
        if self.final_mosaic is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic available to push.")
            return
        img = self.final_mosaic.astype(np.float32, copy=False)   # ⚠️ keep dimensionality

        is_mono = (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))

        # Attach the mosaic WCS as a real header under 'original_header', so the
        # pushed view is plate-solved just like a save-to-new-view mosaic.
        hdr = None
        if self.wcs_metadata and any(self.wcs_metadata.values()):
            hdr = sanitize_wcs_header(self.wcs_metadata) or coerce_to_header(self.wcs_metadata)

        meta = {"step_name": "Mosaic Master", "is_mono": bool(is_mono)}
        if hdr is not None:
            meta["original_header"] = hdr

        dm = self._docman
        if dm is not None:
            newdoc = dm.open_array(img, metadata=meta, title="Mosaic")
            if hasattr(self.parent(), "_spawn_subwindow_for"):
                self.parent()._spawn_subwindow_for(newdoc)
            QMessageBox.information(self, "Mosaic Master", "Pushed to new view.")
        elif self.image_manager and hasattr(self.image_manager, "create_document"):
            self.image_manager.create_document(image=img, metadata=meta)
            QMessageBox.information(self, "Mosaic Master", "Pushed via image manager.")


    # ---- view helpers (same spirit as StellarAlignmentDialog) ----
    def _safe_call(self, maybe_callable):
        try:
            return maybe_callable() if callable(maybe_callable) else maybe_callable
        except Exception:
            return None

    def _title_for_doc(self, doc, preferred=None):
        # If we were given a candidate title and it's usable, keep it.
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()

        # Try common “nice” sources (call methods when present).
        for attr in ("display_name", "windowTitle", "objectName", "title", "name"):
            val = self._safe_call(getattr(doc, attr, None))
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Try metadata fallbacks.
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            for k in ("display_name", "title", "name"):
                val = meta.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()

        # Last resort.
        return f"Untitled View {id(doc)}"

    def _iter_docs(self):
        """
        Returns a list of (title:str, doc:any) pairs.
        Accepts helpers that yield either bare docs or (title, doc) tuples.
        """
        # Prefer host helper
        items = []
        if self._list_open_docs_fn:
            try:
                raw = list(self._list_open_docs_fn()) or []
            except Exception as e:
                print("[Mosaic] _list_open_docs_fn failed:", e)
                raw = []
            for it in raw:
                if isinstance(it, (tuple, list)) and len(it) >= 2:
                    t, d = it[0], it[1]
                    items.append((self._title_for_doc(d, preferred=t), d))
                else:
                    d = it
                    items.append((self._title_for_doc(d), d))

        if items:
            return items

        # Fallbacks from the doc manager
        docs = []
        try:
            if self._docman and hasattr(self._docman, "iter_documents") and callable(self._docman.iter_documents):
                docs = list(self._docman.iter_documents())
            elif self._docman and hasattr(self._docman, "documents"):
                docs = list(self._docman.documents)
        except Exception:
            docs = []

        return [(self._title_for_doc(d), d) for d in docs]


    def openSettings(self):
        dlg = MosaicSettingsDialog(self.settings, self)
        if dlg.exec():
            self.status_label.setText("Mosaic settings updated.")

    # ---------- Add / Remove ----------
    def add_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fz *.xisf *.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef)"
        )
        if not paths:
            return

        for path in paths:
            arr, header, bitdepth, ismono = load_image(path)
            header = sanitize_wcs_header(header) if header else None
            wcs_obj = get_wcs_from_header(header) if header else None

            base_name = os.path.basename(path)  # <- what you want in status strings

            d = {
                "path": path,            # internal identity
                "display": base_name,    # user-facing label
                "image": arr,
                "header": header,
                "wcs": wcs_obj,
                "bit_depth": bitdepth,
                "is_mono": ismono,
                "transform": None
            }
            self.loaded_images.append(d)

            # list text can include decorations like [WCS]
            list_text = base_name + (" [WCS]" if wcs_obj is not None else "")
            item = QListWidgetItem(list_text)
            item.setToolTip(path)  # still used for removal
            self.images_list.addItem(item)

        self.update_status()
        self._refresh_sip_autodisable()

    def add_image_from_view(self):
        entries = self._collect_view_candidates()
        if not entries:
            QMessageBox.information(self, "Add from View",
                                    "No views with image data are open.")
            return

        dlg = _ViewPickerDialog(entries, parent=self)
        if not dlg.exec():
            return

        chosen = dlg.selected_entries()
        if not chosen:
            return

        for e in chosen:
            img = e["image"]
            meta = e["meta"]
            self.loaded_images.append({
                "path": e["key"],
                "display": e["display"],
                "image": img,
                "header": e["header"],
                "wcs": e["wcs"],
                "bit_depth": meta.get("bit_depth"),
                "is_mono": meta.get("is_mono", img.ndim == 2),
                "transform": None,
            })

            item = QListWidgetItem(
                e["base_label"] + (" [WCS]" if e["wcs"] is not None else ""))
            item.setToolTip(e["key"])
            self.images_list.addItem(item)

        self.update_status()
        self._refresh_sip_autodisable()
        n_solved = sum(1 for e in chosen if e["wcs"] is not None)
        self.status_label.setText(
            f"{len(self.loaded_images)} images loaded "
            f"({len(chosen)} added, {n_solved} already solved).")


    def _item_label(self, item: dict) -> str:
        # 1) explicit display label wins
        d = item.get("display")
        if isinstance(d, str) and d.strip():
            return d.strip()

        # 2) fallback from path (disk or view://)
        p = item.get("path", "")
        if isinstance(p, str) and p.startswith("view://"):
            return "View"
        return os.path.basename(str(p)) if p else "Item"

    def remove_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Remove", "No item selected.")
            return
        for itm in s:
            row = self.images_list.row(itm)
            self.images_list.takeItem(row)
            p = itm.toolTip()
            self.loaded_images = [x for x in self.loaded_images if x["path"] != p]
        self.update_status()

    def update_status(self):
        c = len(self.loaded_images)
        self.status_label.setText(f"{c} images loaded.")

    # ---------- Preview ----------
    def preview_selected(self):
        s = self.images_list.selectedItems()
        if not s:
            QMessageBox.information(self, "Preview", "No item selected.")
            return
        path = s[0].toolTip()
        for d in self.loaded_images:
            if d["path"] == path:
                preview_image = d["image"]
                disp = self._autostretch_if_requested(preview_image)
                MosaicPreviewWindow(disp, title="Preview Selected", parent=self,
                                    push_cb=self._push_mosaic_to_new_doc).show()
                break

    # --- WCS/SIP helpers ---------------------------------------------------------
    def _ensure_image_naxis(self, hdr: dict | fits.Header, shape):
        try:
            h, w = (shape[0], shape[1]) if len(shape) >= 2 else (None, None)
            if h is not None and w is not None:
                hdr["NAXIS"]  = 2
                hdr["NAXIS1"] = int(w)
                hdr["NAXIS2"] = int(h)
        except Exception:
            pass

    def _coerce_float(self, hdr: dict, key: str):
        if key in hdr:
            try:
                hdr[key] = float(hdr[key])
            except Exception:
                try:
                    hdr[key] = float(str(hdr[key]).strip().strip("'"))
                except Exception:
                    del hdr[key]

    def _normalize_wcs_header(self, hdr: dict, img_shape):
        # Required CTYPE + RADESYS
        hdr.setdefault("CTYPE1", "RA---TAN")
        hdr.setdefault("CTYPE2", "DEC--TAN")
        # RADECSYS is deprecated → RADESYS
        if "RADECSYS" in hdr and "RADESYS" not in hdr:
            hdr["RADESYS"] = hdr.pop("RADECSYS")
        hdr.setdefault("RADESYS", "ICRS")
        hdr.setdefault("WCSAXES", 2)

        # Coerce numeric offenders occasionally written as strings
        for k in ("CRPIX1","CRPIX2","CRVAL1","CRVAL2","CD1_1","CD1_2","CD2_1","CD2_2","CDELT1","CDELT2","EQUINOX","CROTA1","CROTA2"):
            self._coerce_float(hdr, k)

        # SIP orders should be ints if present
        for k in ("A_ORDER","B_ORDER","AP_ORDER","BP_ORDER"):
            if k in hdr:
                try:
                    hdr[k] = int(hdr[k])
                except Exception:
                    del hdr[k]

        # Fill missing A/B vs AP/BP if only one is present
        if "A_ORDER" in hdr and "B_ORDER" not in hdr: hdr["B_ORDER"] = hdr["A_ORDER"]
        if "B_ORDER" in hdr and "A_ORDER" not in hdr: hdr["A_ORDER"] = hdr["B_ORDER"]
        if "AP_ORDER" in hdr and "BP_ORDER" not in hdr: hdr["BP_ORDER"] = hdr["AP_ORDER"]
        if "BP_ORDER" in hdr and "AP_ORDER" not in hdr: hdr["AP_ORDER"] = hdr["BP_ORDER"]

        # Make sure dimensions are present (avoids “more axes than image” warning)
        self._ensure_image_naxis(hdr, img_shape)

        return hdr

    def _build_wcs(self, hdr: dict, img_shape):
        # normalize then pass relax=True so SIP and non-standard keys are parsed
        nh = self._normalize_wcs_header(dict(hdr), img_shape)
        return WCS(nh, relax=True)


    # ---------- Align (Entry Point) ----------
    def align_images(self):
        # Seestar mode is its own pipeline
        if self.seestarCheckBox.isChecked():
            self.align_images_seestar_mode()
            return

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return
        try:
            blend_sharp = float(self.blendSharpSlider.value()) / 10.0
        except Exception:
            blend_sharp = 2.0
        blend_sharp = float(np.clip(blend_sharp, 1.0, 6.0))
        print(f"[Mosaic] seam blend sharpness = {blend_sharp:.1f}")

        # Show spinner and start animation.
        self.spinnerLabel.show()
        if self.spinnerMovie:
            self.spinnerMovie.start()
        QApplication.processEvents()

        # ------------------------------------------------------------
        # Progress dialog. Re-ranged per phase; _tick() also pumps the
        # event loop so Cancel stays responsive during long panels.
        # ------------------------------------------------------------
        progress = QProgressDialog("Preparing...", "Cancel", 0, 1, self)
        progress.setWindowTitle("Mosaic Master")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        def _tick(value=None, text=None):
            """Advance the bar, mirror the text to the status label, pump the
            event loop. Returns False if the user pressed Cancel."""
            if text is not None:
                progress.setLabelText(text)
                self.status_label.setText(text)
            if value is not None:
                progress.setValue(int(value))
            QApplication.processEvents()
            return not progress.wasCanceled()

        def _finish():
            try:
                progress.close()
            except Exception:
                pass
            if self.spinnerMovie:
                self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            QApplication.processEvents()

        # ------------------------------------------------------------
        # 1) Plate solve (unless already solved and not forcing blind)
        # ------------------------------------------------------------
        force_blind = self.forceBlindCheckBox.isChecked()
        images_to_process = (
            self.loaded_images if force_blind
            else [item for item in self.loaded_images if item.get("wcs") is None]
        )

        if images_to_process:
            progress.setRange(0, len(images_to_process))
            _tick(0, f"Plate solving 0/{len(images_to_process)}...")

        for _i_solve, item in enumerate(images_to_process):
            if not _tick(_i_solve,
                         f"Plate solving {_i_solve + 1}/{len(images_to_process)}: "
                         f"{self._item_label(item)}"):
                _finish()
                self.status_label.setText("Cancelled.")
                return

            # Ensure ASTAP path (or fall back to blind solve)
            if not self.astap_exe or not os.path.exists(self.astap_exe):
                executable_filter = "Executables (*.exe);;All Files (*)" if sys.platform.startswith("win") else "Executables (*)"
                new_path, _ = QFileDialog.getOpenFileName(self, "Select ASTAP Executable", "", executable_filter)
                if new_path:
                    self._save_astap_exe_to_settings(new_path)
                    self.astap_exe = new_path
                    QMessageBox.information(self, "Mosaic Master", "ASTAP path updated successfully.")
                else:
                    self.status_label.setText(f"No ASTAP path; blind solving {self._item_label(item)}...")
                    QApplication.processEvents()
                    solved_header = self.perform_blind_solve(item)
                    if solved_header:
                        solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                        item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
                    continue

            # Attempt ASTAP solve.
            self.status_label.setText(f"Attempting ASTAP solve for {self._item_label(item)}...")

            QApplication.processEvents()
            solved_header = self.attempt_astap_solve(item)

            if solved_header is None:
                self.status_label.setText(f"ASTAP failed for {self._item_label(item)}. Falling back to blind solve...")
                QApplication.processEvents()
                solved_header = self.perform_blind_solve(item)

            if solved_header:
                solved_header = self._normalize_wcs_header(solved_header, item["image"].shape)
                item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
            else:
                print(f"[Mosaic] Plate solving failed for {self._item_label(item)} ({item.get('path','')}).")

        # newly-solved panels may now carry (or lack) SIP -- refresh the default
        self._refresh_sip_autodisable()
        # ------------------------------------------------------------
        # 2) Gather WCS-valid panels
        # ------------------------------------------------------------
        wcs_items = [x for x in self.loaded_images if x.get("wcs") is not None]
        if not wcs_items:
            print("[Mosaic] No images have WCS; cannot build WCS mosaic.")
            _finish()
            return

        _tick(text="Ordering panels...")
        wcs_items = self._order_panels(wcs_items)

        # Panels must be at different sky positions. A degenerate WCS (one that
        # parsed but carries no real astrometry) puts them all at the origin and
        # collapses the mosaic to a single panel.
        cents = []
        for itm in wcs_items:
            h, w = itm["image"].shape[:2]
            try:
                ra, dec = itm["wcs"].all_pix2world([w / 2.0], [h / 2.0], 0)
                cents.append((float(ra[0]), float(dec[0])))
            except Exception:
                cents.append((float("nan"), float("nan")))
            print(f"[Mosaic] {self._item_label(itm)}: centre "
                  f"RA={cents[-1][0]:.5f} Dec={cents[-1][1]:.5f}")

        fin = np.array([c for c in cents if np.isfinite(c[0])], dtype=float)
        if len(fin) > 1:
            spread = float(np.hypot(*(fin.max(axis=0) - fin.min(axis=0))))
            print(f"[Mosaic] panel centre spread = {spread:.4f} deg")
            if spread < 1e-4:
                _finish()
                QMessageBox.warning(
                    self, "Mosaic Master",
                    "All panels report the same sky position, so the mosaic "
                    "would collapse onto one panel.\n\n"
                    "Their WCS parsed but carries no real astrometric solution. "
                    "Re-plate-solve the panels and try again.")
                return        
        # ------------------------------------------------------------
        # 3) Establish mosaic WCS + output bounding box
        # ------------------------------------------------------------
        reference_wcs = self._linearize_wcs(self._build_wcs(
            wcs_items[0]["wcs"].to_header(relax=True),
            wcs_items[0]["image"].shape
        ))

        min_x, min_y, max_x, max_y = self.compute_mosaic_bounding_box(wcs_items, reference_wcs)
        mosaic_width  = int(max_x - min_x)
        mosaic_height = int(max_y - min_y)

        if mosaic_width < 1 or mosaic_height < 1:
            print("[Mosaic] ERROR: Computed mosaic size invalid. Check WCS/inputs.")
            _finish()
            return

        mosaic_wcs = reference_wcs.deepcopy()
        mosaic_wcs.wcs.crpix[0] -= min_x
        mosaic_wcs.wcs.crpix[1] -= min_y
        self.wcs_metadata = mosaic_wcs.to_header(relax=True)

        # ------------------------------------------------------------
        # 4) Allocate accumulators
        # ------------------------------------------------------------
        is_color = any(not item["is_mono"] for item in wcs_items)

        self.stretch_original_mins = []
        self.stretch_original_medians = []
        self.was_single_channel = (not is_color)

        if is_color:
            self.final_mosaic, _ = smart_zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)
        else:
            self.final_mosaic, _ = smart_zeros((mosaic_height, mosaic_width), dtype=np.float32)
        self.weight_mosaic, _ = smart_zeros((mosaic_height, mosaic_width), dtype=np.float32)

        # ------------------------------------------------------------
        # 5) Background reference — measured once from the first panel
        # ------------------------------------------------------------
        did_normalize = bool(self.normalizeCheckBox.isChecked())

        # Pedestal added before warping so the valid-pixel footprint survives
        # every remap/warp (border fill is 0, real data lands near PED).
        self._BG_PED = 1.0
        self._bg_fit_gain = True   # False -> offset-only overlap fit

        self._ref_bg = None
        if did_normalize:
            try:
                ref_raw = wcs_items[0]["image"].astype(np.float32, copy=False)
                self._ref_bg, self._ref_sig = estimate_background_level(ref_raw)
                print(f"[Mosaic] reference sky level = {self._ref_bg} "
                      f"(sigma {self._ref_sig})")
            except Exception as e:
                print(f"[Mosaic] background estimate failed on reference: {e}")
                self._ref_bg = None

        use_bg_match = bool(did_normalize and self._ref_bg is not None)

        # Legacy fallback target, only used if the estimator bailed out
        if did_normalize and not use_bg_match:
            self._mosaic_target_median = self._target_median_from_first(
                [{"image": wcs_items[0]["image"]}]
            )
            print(f"[Mosaic] falling back to median match, "
                  f"target = {self._mosaic_target_median:.6g}")

        # Reprojection helpers cache
        if not hasattr(self, "_H_cache"):
            self._H_cache = {}

        # WCS-only toggle (no star alignment/refinement)
        wcs_only = bool(getattr(self, "wcsOnlyCheckBox", None) and self.wcsOnlyCheckBox.isChecked())

        # ------------------------------------------------------------
        # 6) Main loop: normalize -> reproject -> (optional) star align -> accumulate
        # ------------------------------------------------------------
        first_image = True
        PED = float(self._BG_PED)

        n_panels = len(wcs_items)
        SUBSTEPS = 4              # reproject, align, match, accumulate
        progress.setRange(0, n_panels * SUBSTEPS + 1)
        progress.setValue(0)

        for idx, itm in enumerate(wcs_items):
            arr = itm["image"]
            _base = idx * SUBSTEPS
            _name = self._item_label(itm)

            if not _tick(_base, f"Panel {idx + 1}/{n_panels} — reprojecting {_name}"):
                _finish()
                self.status_label.setText("Cancelled.")
                return

            img_lin = arr.astype(np.float32, copy=False)

            # --- stats kept only for the legacy unstretch path ---
            mono_for_stats = img_lin if img_lin.ndim == 2 else np.mean(img_lin, axis=2)
            self.stretch_original_mins.append(float(np.nanmin(mono_for_stats)))
            self.stretch_original_medians.append(float(np.nanmedian(mono_for_stats)))

            # --- background match: shift this panel's sky onto the reference sky ---
            if use_bg_match:
                lev_i, _ = estimate_background_level(img_lin)
                img_lin = (img_lin
                           - self._bg_offset(lev_i, img_lin)
                           + self._bg_offset(self._ref_bg, img_lin)).astype(np.float32, copy=False)
                print(f"[Mosaic] {self._item_label(itm)}: sky {lev_i} -> {self._ref_bg}")
            elif did_normalize:
                img_lin = self._normalize_linear(img_lin, float(self._mosaic_target_median))

            # --- carry the footprint through the warp on a pedestal ---
            img_ped = (img_lin + PED).astype(np.float32, copy=False)

            # --- Reprojection mode ---
            mode = self.reprojectModeCombo.currentText()

            if mode.startswith("Fast — Homography"):
                reprojected = self._warp_via_wcs_homography(
                    img_ped, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width),
                    H_cache=self._H_cache
                ).astype(np.float32, copy=False)
                reprojected = self._polish_reprojected(reprojected)

            else:
                # Exact SIP-aware remap. This IS the full spherical transform --
                # dst pixel -> world -> src pixel, per pixel, SIP on the source.
                # Also the default for any unrecognized/legacy mode string.
                reprojected = self._warp_via_wcs_remap_exact(
                    img_ped, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width),
                    tile=512,
                    progress_cb=lambda f, b=_base, i=idx, nm=_name: _tick(
                        b + f, f"Panel {i + 1}/{n_panels} — reprojecting {nm} "
                               f"({f * 100:.0f}%)")
                ).astype(np.float32, copy=False)
                reprojected = self._polish_reprojected(reprojected)

            # Pedestal-free grayscale for star detection (astroalign never sees PED)
            cov_r = (reprojected if reprojected.ndim == 2 else reprojected[..., 0]) > (0.995 * PED)
            reproj_red = self._strip_pedestal(
                reprojected if reprojected.ndim == 2 else reprojected[..., 0], PED, cov_r
            )

            _tick(_base + 1, f"Panel {idx + 1}/{n_panels} — aligning {_name}")

            # --- Optional star alignment/refinement -------------------------
            # SIP panels are placed by WCS alone: a global affine can't match a
            # per-pixel 4th-order SIP resample, and on a corner panel it trades
            # one edge's registration for another's. Star alignment runs only on
            # panels without SIP (blind solves, Seestar). The wcs_only checkbox
            # forces WCS-only for every panel regardless.
            # NOTE: `reprojected` keeps the pedestal on purpose, so the footprint
            # survives warpAffine (border fill = 0).
            panel_has_sip = self._wcs_has_sip(itm["wcs"])
            skip_star_align = wcs_only or panel_has_sip

            if skip_star_align:
                aligned = reprojected
                if first_image:
                    first_image = False
            else:
                # Pre-alignment footprint (pedestal intact) for the overlap test.
                gray_ped = reprojected[..., 0] if reprojected.ndim == 3 else reprojected
                if first_image:
                    aligned = reprojected
                    first_image = False
                else:
                    transform_method = self.transform_combo.currentText()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mos_now = self.final_mosaic / np.maximum(
                            self.weight_mosaic[..., None] if self.final_mosaic.ndim == 3
                            else self.weight_mosaic, 1e-12)
                    mos_now = np.nan_to_num(mos_now, nan=0.0, posinf=0.0, neginf=0.0)
                    mosaic_gray = mos_now if mos_now.ndim == 2 else np.mean(mos_now, axis=-1)

                    self.status_label.setText("Computing affine transform with astroalign...")
                    QApplication.processEvents()

                    # Star-match only in the overlap. Feeding the whole canvas
                    # spreads control points across regions that can't correspond,
                    # starving the RANSAC of usable bright anchors.
                    ov = (gray_ped > 0.5 * PED) & (self.weight_mosaic > 0)
                    n_ov_stars = int(ov.sum())
                    if n_ov_stars < 2000:
                        print(f"[Mosaic] {self._item_label(itm)}: overlap "
                              f"{n_ov_stars}px too small for star alignment, "
                              f"WCS placement only.")
                        transform_matrix = np.eye(2, 3, dtype=np.float32)
                    else:
                        ys, xs = np.nonzero(ov)
                        y0, y1 = int(ys.min()), int(ys.max()) + 1
                        x0, x1 = int(xs.min()), int(xs.max()) + 1
                        m = ov[y0:y1, x0:x1]
                        tgt = np.where(m, reproj_red[y0:y1, x0:x1], 0.0).astype(np.float32)
                        ref = np.where(m, mosaic_gray[y0:y1, x0:x1], 0.0).astype(np.float32)
                        print(f"[astroalign] overlap box {x1 - x0}x{y1 - y0}, "
                              f"{int(m.sum())} px")
                        try:
                            transform_obj, (src_pts, dst_pts) = _aa_find_transform_with_backoff(tgt, ref)
                            transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                        except Exception as e:
                            print(f"[Mosaic] astroalign fallback on {self._item_label(itm)}: "
                                  f"{type(e).__name__}: {e}")
                            self.status_label.setText(f"Astroalign failed: {e}. Using identity transform.")
                            transform_matrix = np.eye(2, 3, dtype=np.float32)

                    affine_aligned = cv2.warpAffine(
                        reprojected, transform_matrix, (mosaic_width, mosaic_height),
                        flags=cv2.INTER_LANCZOS4
                    )
                    aligned = affine_aligned

                    if transform_method in ["Homography Transform", "Polynomial Warp Based Transform"]:
                        self.status_label.setText(f"Starting refined alignment using {transform_method}...")
                        QApplication.processEvents()
                        refined_result = self.refined_alignment(affine_aligned, mosaic_gray,
                                                                method=transform_method)
                        if refined_result is not None:
                            aligned, best_inliers2 = refined_result
                            self.status_label.setText(
                                f"Refined alignment succeeded with {best_inliers2} inliers.")
                        else:
                            self.status_label.setText(
                                "Refined alignment failed; falling back to affine alignment.")

            # If mosaic is color but aligned is mono, expand for accumulation only
            if is_color and aligned.ndim == 2:
                aligned = np.repeat(aligned[..., None], 3, axis=2)

            # --- real footprint, then drop the pedestal ---
            gray_ped = aligned[..., 0] if aligned.ndim == 3 else aligned

            # A fully covered pixel is PED + data, and data >= 0, so anything
            # below PED had interpolation reach past the panel edge into
            # borderValue=0. Lanczos4 spans 8x8, so contamination extends ~4px
            # in; erode to clear the ringing band, where overshoot can push a
            # contaminated pixel back above PED.
            EDGE_TRIM = 12
            coverage = gray_ped > (0.995 * PED)
            if EDGE_TRIM > 0 and coverage.any():
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * EDGE_TRIM + 1, 2 * EDGE_TRIM + 1))
                coverage = cv2.erode(coverage.astype(np.uint8), k).astype(bool)

            if not coverage.any():
                print(f"[Mosaic] {self._item_label(itm)}: empty footprint, skipping.")
                continue
            aligned = self._strip_pedestal(aligned, PED, coverage)

            _tick(_base + 2, f"Panel {idx + 1}/{n_panels} — matching brightness ({_name})")

            # --- overlap refinement: solve mosaic ≈ a*panel + b where they share sky ---
            if use_bg_match and not first_image:
                overlap = coverage & (self.weight_mosaic > 0)
                n_ov = int(overlap.sum())
                if n_ov > 5000:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mos_est = self.final_mosaic / np.maximum(
                            self.weight_mosaic[..., None] if self.final_mosaic.ndim == 3
                            else self.weight_mosaic, 1e-12)
                    mos_est = np.nan_to_num(mos_est, nan=0.0, posinf=0.0, neginf=0.0)
                    a, b, n = solve_linear_match(aligned, mos_est, overlap,
                                                 fit_scale=self._bg_fit_gain,
                                                 pivot=self._ref_bg)
                    pv = self._bg_offset(self._ref_bg, aligned)
                    aligned = ((aligned - pv) * self._bg_offset(a, aligned)
                               + pv + self._bg_offset(b, aligned)
                               ).astype(np.float32, copy=False)
                    aligned[~coverage] = 0.0
                    print(f"[Mosaic] {self._item_label(itm)}: a={a}, b={b} "
                          f"({n} blocks)")
                else:
                    print(f"[Mosaic] {self._item_label(itm)}: overlap {n_ov} px, "
                          f"keeping pedestal-only match.")

            gray_aligned = aligned[..., 0] if aligned.ndim == 3 else aligned

            # --- Weight mask -------------------------------------------------
            # Feather that auto-scales to each seam's overlap. d_self is this
            # panel's distance-to-edge; the eventual division by weight_mosaic
            # means the *ratio* d_self / (d_self + d_other) is what sets the
            # blend, and that ratio spans exactly the available overlap -- wide
            # where overlap is generous, tight where it's thin. The raised-cosine
            # shaping controls how sharp the handoff is within that span.

            binary_mask = coverage.astype(np.uint8)
            d = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5).astype(np.float32)
            d = cv2.GaussianBlur(d, (15, 15), 0)
            d *= binary_mask
            d /= 100.0   # scale-invariant; keeps d**p in float32 range at high p

            # Raise the distance to a power BEFORE accumulation. After all panels
            # are summed, w_i / sum(w_j) = d_i^p / sum(d_j^p), which is the
            # overlap-relative feather. p=1 recovers the old linear ratio.
            smooth_mask = np.power(d, blend_sharp, dtype=np.float32)
            smooth_mask *= binary_mask

            _tick(_base + 3, f"Panel {idx + 1}/{n_panels} — blending {_name}")

            # --- Accumulate ---
            if is_color:
                self.final_mosaic += aligned * smooth_mask[..., np.newaxis]
            else:
                self.final_mosaic += gray_aligned * smooth_mask
            self.weight_mosaic += smooth_mask

            _tick(_base + SUBSTEPS, f"Panel {idx + 1}/{n_panels} done — {_name}")

        _tick(n_panels * SUBSTEPS, "Normalizing accumulated mosaic...")

        # ------------------------------------------------------------
        # 7) Final blend
        # ------------------------------------------------------------
        if is_color:
            self.final_mosaic = np.where(
                self.weight_mosaic[..., None] > 0,
                self.final_mosaic / np.maximum(self.weight_mosaic[..., None], 1e-12),
                self.final_mosaic
            )
        else:
            nz = (self.weight_mosaic > 0)
            self.final_mosaic[nz] = self.final_mosaic[nz] / np.maximum(self.weight_mosaic[nz], 1e-12)

        self.status_label.setText("Mosaic built. De-normalizing mosaic...")
        QApplication.processEvents()

        # ------------------------------------------------------------
        # 8) Optional “unstretch” (your existing logic)
        # ------------------------------------------------------------
        if (did_normalize and not use_bg_match and
            hasattr(self, "_mosaic_target_median") and float(self._mosaic_target_median) > 0 and
            getattr(self, "stretch_original_medians", None) and len(self.stretch_original_medians) > 0 and
            getattr(self, "stretch_original_mins", None) and len(self.stretch_original_mins) > 0):
            self.final_mosaic = self.unstretch_image(self.final_mosaic)

        self.status_label.setText("Final Mosaic Ready.")
        QApplication.processEvents()

        display_image = (np.stack([self.final_mosaic] * 3, axis=-1)
                        if self.final_mosaic.ndim == 2 else self.final_mosaic)
        display_image = self._autostretch_if_requested(display_image)

        _tick(n_panels * SUBSTEPS + 1, "Final Mosaic Ready.")
        _finish()

        MosaicPreviewWindow(
            display_image,
            title=("Final Mosaic (WCS-only)" if wcs_only else "Final Mosaic"),
            parent=self,
            push_cb=self._push_mosaic_to_new_doc
        ).show()

        self.status_label.setText(
            f"Final mosaic ready — {n_panels} panels, "
            f"{mosaic_width}×{mosaic_height}.")

      
    def _polish_reprojected(self, img: np.ndarray) -> np.ndarray:
        # 1) kill NaNs/Infs from reprojection
        out = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 2) trim tiny negative ringing
        out[out < 0] = 0.0

        # 3) optional: zero a 1px border to avoid remap edge seams
        if out.ndim == 2:
            out[:1, :] = 0; out[-1:, :] = 0; out[:, :1] = 0; out[:, -1:] = 0
        else:
            out[:1, :, :] = 0; out[-1:, :, :] = 0; out[:, :1, :] = 0; out[:, -1:, :] = 0

        return out

    def debayer_image(self, image, file_path, header):
        from setiastro.saspro.legacy.numba_utils import debayer_raw_fast, debayer_fits_fast  
        if file_path.lower().endswith(('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
            print(f"Debayering RAW image: {file_path}")
            return debayer_raw_fast(image)
        elif file_path.lower().endswith(('.fits', '.fit', '.fz', '.fz')):
            bayer_pattern = header.get('BAYERPAT')
            if bayer_pattern:
                print(f"Debayering FITS image: {file_path} with Bayer pattern {bayer_pattern}")
                return debayer_fits_fast(image, bayer_pattern)
        return image

    def refine_via_overlap(self, new_gray, mosaic_gray, rough_matrix):
        """
        1) Warp new_gray into mosaic coords using rough_matrix.
        2) Compute overlap region by checking where both warped_new_gray and mosaic_gray > 0.
        3) Restrict astroalign.find_transform() to that overlap only.
        4) Combine the refinement transform with rough_matrix so you have a single final 2×3 matrix
        mapping the original new_gray -> mosaic_gray coordinates.

        Returns the final 2×3 transform matrix for cv2.warpAffine, or None if no overlap or alignment fails.
        """

        # ------------------------------------------
        # A) Warp new_gray with rough_matrix
        # ------------------------------------------
        h_m, w_m = mosaic_gray.shape
        warped_new_gray = cv2.warpAffine(
            new_gray,
            rough_matrix,
            (w_m, h_m),   # (width, height)
            flags=cv2.INTER_LANCZOS4
        )

        # ------------------------------------------
        # B) Determine overlap region (both > 0)
        # ------------------------------------------
        mask_warped = (warped_new_gray > 0)
        mask_mosaic = (mosaic_gray > 0)
        overlap_mask = mask_warped & mask_mosaic

        overlap_pixels = np.count_nonzero(overlap_mask)
        print(f"[DEBUG] Overlap region has {overlap_pixels} pixels.")

        # If there’s not enough overlap, bail out
        if overlap_pixels < 50:
            # Not enough area for star matching
            return None

        # ------------------------------------------
        # C) Mask out everything except the overlap
        # ------------------------------------------
        # We'll use numpy.ma arrays so astroalign sees "valid" data only in overlap region
        warped_new_ma = ma.array(warped_new_gray, mask=~overlap_mask)
        mosaic_ma = ma.array(mosaic_gray, mask=~overlap_mask)

        # ------------------------------------------
        # D) Attempt astroalign on that overlap
        #    Note: the transform we get is from warped_new_gray -> mosaic_gray,
        #    i.e., in "mosaic coordinate space." So it should be *almost* identity
        #    if the rough transform was close, but with some tweak for rotation/translation.
        # ------------------------------------------
        try:
            refine_obj, (src_pts, dst_pts) = astroalign.find_transform(
                warped_new_ma, mosaic_ma,
                max_control_points=50,
                detection_sigma=2
            )
            print(f"[DEBUG] Overlap-limited astroalign success with {len(src_pts)} stars.")
        except Exception as e:
            print(f"[DEBUG] Overlap-limited astroalign failed: {e}")
            return None

        # ------------------------------------------
        # E) Combine refine_obj with the original rough_matrix
        # ------------------------------------------
        # refine_obj.params is typically a 3×3 (similarity).
        # rough_matrix is 2×3 for warpAffine.
        # We'll promote rough_matrix to 3×3, multiply, then convert back.
        refine_mat_3x3 = refine_obj.params  # e.g. a 3×3
        rough_mat_3x3 = np.eye(3, dtype=np.float32)
        rough_mat_3x3[:2, :] = rough_matrix

        # The final transform is refine_mat_3x3 * rough_mat_3x3
        #   (in linear-algebra order, the right-hand transform applies first)
        combined_3x3 = refine_mat_3x3 @ rough_mat_3x3

        # Convert that back to 2×3 for cv2
        final_transform = combined_3x3[:2, :].astype(np.float32)
        return final_transform


    def align_images_seestar_mode(self):
        """ 
        Align images in Seestar Mode by first selecting the center-most image (based on WCS centers)
        as the initial mosaic. Then, sort and add images by increasing distance from the mosaic center.
        If WCS information is present in the header, a rough pre-alignment is computed before refining
        with astroalign.find_transform. After processing all images, failed images are reattempted.

        Final image is produced by summing all warped images and dividing by the pixel counts 
        (sum-then-divide approach).
        """

        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        self.status_label.setText("🔄 Image Registration Started...")
        self.spinnerLabel.show()
        self.spinnerMovie.start()
        QApplication.processEvents()

        total_files = len(self.loaded_images)  # how many images in total
        if total_files == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Create a QProgressDialog
        progress = QProgressDialog("Aligning images...", "Cancel", 0, total_files, self)
        progress.setWindowTitle("Seestar Alignment")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.show()

        # We define how many pixels to zero out on each edge AFTER warp
        POST_WARP_BORDER = 10
        THRESHOLD_RATIO = 0.9  # if new pixel < 0.5 * mosaic pixel, skip it

        # -----------------------------------------------------
        # Helper: Extract the world-coordinate center from an image header
        # -----------------------------------------------------
        def get_wcs_center(item):
            header = item.get("header", None)
            if header is None:
                print(f"[DEBUG] No header for {item.get('path','')}")
                return None
            try:
                wcs = WCS(header)
                naxis1 = int(header.get("NAXIS1", 0))
                naxis2 = int(header.get("NAXIS2", 0))
                center_pix = np.array([naxis1 / 2.0, naxis2 / 2.0])
                center_world = wcs.all_pix2world(center_pix[None, :], 1)[0]
                print(f"[DEBUG] {item.get('path','')} center_world: {center_world}")
                return center_world
            except Exception as e:
                print(f"[DEBUG] Failed to get WCS center for {item.get('path','')}: {e}")
                return None

        # -----------------------------------------------------
        # Helper: Count how many stars appear in an image (2D or 3D)
        # -----------------------------------------------------
        def count_stars_in_image(image):
            """
            Use DAOStarFinder to return number of detected stars.
            Expects a 2D image; if 3D, we average over channels.
            You might need to tweak fwhm/threshold for your data.
            """
            from photutils.detection import DAOStarFinder
            if image.ndim == 3:
                image = np.mean(image, axis=2)

            mean_val, median_val, std_val = sigma_clipped_stats(image, sigma=3.0)
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std_val)
            sources = daofind(image - median_val)

            if sources is None:
                return 0
            else:
                return len(sources)

        # -----------------------------------------------------
        # 1) Gather all images with valid WCS and compute average center
        # -----------------------------------------------------
        centers = []
        valid_items = []
        for item in self.loaded_images:
            center = get_wcs_center(item)
            if center is not None:
                centers.append(center)
                valid_items.append(item)

        if not centers:
            QMessageBox.warning(self, "Align", "No images with valid WCS found.")
            return

        centers = np.array(centers)
        avg_center = np.mean(centers, axis=0)  # or np.median if you prefer

        # -----------------------------------------------------
        # Sort valid items by distance from average center
        # -----------------------------------------------------
        def distance_from_avg(item):
            center = get_wcs_center(item)
            dist = np.linalg.norm(center - avg_center) if center is not None else np.inf
            print(f"[DEBUG] {item['path']} distance from avg: {dist}")
            return dist

        valid_items.sort(key=distance_from_avg)
        for item in valid_items:
            print(f"[DEBUG] Sorted valid item: {item['path']}")

        # -----------------------------------------------------
        # Select from the top few “closest to center,” pick the one with the most stars as the base.
        # -----------------------------------------------------
        top_n = 5
        candidate_subset = valid_items[:top_n]

        best_item = None
        best_star_count = 0
        for candidate in candidate_subset:
            star_count = count_stars_in_image(candidate["image"])
            print(f"[DEBUG] {candidate['path']} star count: {star_count}")
            if star_count > best_star_count:
                best_star_count = star_count
                best_item = candidate

        if best_item is None:
            best_item = valid_items[0]

        base_item = best_item
        print(f"[DEBUG] Selected base image: {base_item['path']} with star count = {best_star_count}")

        # -----------------------------------------------------
        # Reorder loaded_images so that valid (WCS) items come first
        # -----------------------------------------------------
        remaining_items = []
        valid_paths = {v["path"] for v in valid_items}  # set of valid item paths

        for item in self.loaded_images:
            if item["path"] not in valid_paths:
                remaining_items.append(item)

        self.loaded_images = valid_items + remaining_items


        # -----------------------------------------------------
        # Helper: Crop a 5-pixel border from each edge
        # -----------------------------------------------------
        def crop_5px_border(img):
            if img.ndim == 3:
                h, w, c = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0, c), dtype=img.dtype)
                return img[10:-10, 10:-10, :]
            else:
                h, w = img.shape
                if h <= 10 or w <= 10:
                    return np.empty((0, 0), dtype=img.dtype)
                return img[10:-10, 10:-10]

        # -----------------------------------------------------
        # Helper: Convert image to float32 and normalize if needed
        # -----------------------------------------------------
        def ensure_float32_in_01(img):
            img = img.astype(np.float32, copy=False)
            mx = np.max(img)
            if mx <= 1.0:
                return img
            elif mx <= 255.0:
                return img / 255.0
            elif mx <= 65535.0:
                return img / 65535.0
            else:
                return img / mx if mx > 0 else img

        # -----------------------------------------------------
        # Helper: Compute a rough transform (rotation+translation) using WCS
        # -----------------------------------------------------
        def compute_rough_transform(new_header, ref_header):
            """Compute a rough transformation (rotation and translation) using WCS."""
            try:
                new_wcs = WCS(new_header)
                ref_wcs = WCS(ref_header)
                ref_naxis1 = int(ref_header.get('NAXIS1', 0))
                ref_naxis2 = int(ref_header.get('NAXIS2', 0))
                ref_center = np.array([ref_naxis1 / 2.0, ref_naxis2 / 2.0])
                world_center = ref_wcs.all_pix2world(ref_center[None, :], 1)
                new_center = new_wcs.all_world2pix(world_center, 1)[0]
                ref_offset = ref_center + np.array([1, 0])
                world_offset = ref_wcs.all_pix2world(ref_offset[None, :], 1)[0]
                new_offset = new_wcs.all_world2pix(world_offset[None, :], 1)[0]
                vector = new_offset - new_center
                angle = np.arctan2(vector[1], vector[0])
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                R = np.array([[cos_a, -sin_a],
                            [sin_a,  cos_a]])
                t = ref_center - R @ new_center
                rough = np.hstack([R, t.reshape(2, 1)]).astype(np.float32)
                return rough
            except Exception as e:
                print(f"Rough transform skipped: {e}")
                return None

        def compute_rough_transform_seestar(new_header, ref_header):
            """
            Builds a 2×3 transform matrix (rotation + translation) that
            roughly aligns an image with RA/DEC, pixel scale, and focal length
            to a reference image with the same kind of data.

            Required header fields (example):
            new_header["RA"]       (degrees)
            new_header["DEC"]      (degrees)
            new_header["XPIXSZ"]   (microns)
            new_header["YPIXSZ"]   (microns)
            new_header["FOCALLEN"] (mm)
            Similarly for ref_header.

            Returns:
            A 2×3 np.float32 array suitable for cv2.warpAffine,
            or None if missing data or something fails.
            """

            # 1) Parse required fields from new_header
            try:
                ra_new_deg = float(new_header["RA"])       # degrees
                dec_new_deg = float(new_header["DEC"])     # degrees
                xpixsz_new = float(new_header["XPIXSZ"])   # microns
                ypixsz_new = float(new_header["YPIXSZ"])   # microns
                flen_new   = float(new_header["FOCALLEN"]) # mm
            except KeyError as e:
                print(f"[DEBUG] new_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in new_header => {e}")
                return None

            # 2) Parse required fields from ref_header
            try:
                ra_ref_deg = float(ref_header["RA"])
                dec_ref_deg = float(ref_header["DEC"])
                xpixsz_ref = float(ref_header["XPIXSZ"])
                ypixsz_ref = float(ref_header["YPIXSZ"])
                flen_ref   = float(ref_header["FOCALLEN"])
            except KeyError as e:
                print(f"[DEBUG] ref_header missing key {e}. No rough transform.")
                return None
            except Exception as e:
                print(f"[DEBUG] parse error in ref_header => {e}")
                return None

            # 3) Compute average arcsec/pixel for each image
            #    arcsec_per_pix = 206265 * (XPIXSZ [um]) / (FOCALLEN [mm])
            #    (assuming 1 mm = 1000 um)
            #    If you want to be more precise, you might average XPIXSZ and YPIXSZ,
            #    or handle them separately if the camera has rectangular pixels.
            def arcsec_per_pixel(xpixsz, flen):
                return 206265.0 * (xpixsz / 1000.0) / flen

            scale_new = arcsec_per_pixel(xpixsz_new, flen_new)
            scale_ref = arcsec_per_pixel(xpixsz_ref, flen_ref)

            # For simplicity, let's assume the "reference" image defines the final scale.
            # If the images have different scales, we can do a ratio of scales => ~some "zoom."
            # Here we assume they are close enough, or you can do scale = scale_ref / scale_new
            # to get a small difference in pixel scale if needed.
            scale_ratio = scale_ref / scale_new  # how many ref pixels = 1 new pixel

            # 4) Compute difference in RA/DEC in arcseconds
            #    RA/DEC difference in degrees => multiply by 3600 to get arcseconds
            #    Then convert to pixel shift in reference coordinates.
            d_ra_deg  = (ra_new_deg  - ra_ref_deg)
            d_dec_deg = (dec_new_deg - dec_ref_deg)

            # For small fields, we can do a simple approximation ignoring spherical geometry:
            d_ra_arcsec  = d_ra_deg  * 3600.0 * np.cos(np.radians(dec_ref_deg))  # cos(dec) if you want
            d_dec_arcsec = d_dec_deg * 3600.0

            # If you want a more precise approach, you'd do a spherical trig approach,
            # but for small fields, this is usually good enough.

            # 5) Convert arcseconds => pixel shift in "reference" image coords
            dx_pix = d_ra_arcsec  / scale_ref
            dy_pix = - d_dec_arcsec / scale_ref
            # Note: we do "dy_pix = -d_dec_arcsec" if we assume +DEC => "up" in the image.
            # You may need to invert or swap RA/DEC sign depending on your orientation.

            # 6) Combine the scale ratio (if needed) and shift into a 2×3 transform.
            #    If you want to guess rotation from the difference in rotation angles,
            #    you can do that here. For now, we do no rotation => angle=0 => cos=1, sin=0.
            angle_rad = 0.0
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # scale_ratio is for "zoom," if you want to handle that.
            # If you prefer no scale difference, set scale_ratio = 1.0
            # or if the difference is small, ignore it.
            # scale_ratio = 1.0

            M = np.array([
                [scale_ratio*cos_a, -scale_ratio*sin_a, dx_pix],
                [scale_ratio*sin_a,  scale_ratio*cos_a, dy_pix]
            ], dtype=np.float32)

            print(f"[DEBUG] Seestar rough transform => dx_pix={dx_pix:.2f}, dy_pix={dy_pix:.2f}, scale={scale_ratio:.3f}")
            return M

        # -----------------------------------------------------
        # 1) Initialize mosaic_sum/mosaic_count from the base image
        # -----------------------------------------------------
        base_img = ensure_float32_in_01(base_item["image"])
        if base_item["path"].lower().endswith(('.fits', '.fit', '.fz')):
            if (base_img.ndim == 2 
                and "header" in base_item 
                and base_item["header"] is not None 
                and base_item["header"].get('BAYERPAT')):
                self.status_label.setText(f"Debayering {base_item['path']}")
                QApplication.processEvents()
                base_img = self.debayer_image(base_img, base_item["path"], base_item["header"])
                print(f"[DEBUG] Finished debayering base image: {base_item['path']}")

        base_img = crop_5px_border(base_img)
        if base_img.size == 0:
            QMessageBox.warning(self, "Crop Error", "Initial image is too small after cropping.")
            return
    
        base_median = np.median(base_img)

        # mosaic_sum holds the sum of pixel intensities
        mosaic_sum = base_img.astype(np.float32, copy=True)
        # mosaic_count tracks how many images contributed at each pixel
        if mosaic_sum.ndim == 3:
            # For color: count “contributed” if any channel is >0
            contrib_mask = np.any(mosaic_sum > 0, axis=2)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            for c in range(mosaic_sum.shape[2]):
                mosaic_count[..., c][contrib_mask] = 1.0
        else:
            contrib_mask = (mosaic_sum > 0)
            mosaic_count = np.zeros_like(mosaic_sum, dtype=np.float32)
            mosaic_count[contrib_mask] = 1.0

        # We’ll use a helper that returns a mono “view” for alignment:
        def get_current_mosaic_gray():
            """Return current mosaic as a float32 array in [0..1], averaged if color."""
            with np.errstate(divide='ignore', invalid='ignore'):
                stacked = mosaic_sum / np.maximum(mosaic_count, 1e-6)
            # If color, reduce to mono for astroalign
            if stacked.ndim == 3:
                return np.mean(stacked, axis=2).astype(np.float32)
            else:
                return stacked

        self.status_label.setText("Starting alignment of subsequent images...")
        QApplication.processEvents()

        # --- Helper: Process alignment for one image, then accumulate into mosaic_sum/count ---
        def process_alignment(item):
            nonlocal mosaic_sum, mosaic_count

            print(f"[DEBUG] Aligning image: {item['path']}")

            try:
                new_img = item["image"].astype(np.float32)
                print(f"[DEBUG] new_img initial shape={new_img.shape}, dtype={new_img.dtype}, "
                    f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                # Debayer if needed
                if item["path"].lower().endswith(('.fits', '.fit')):
                    if (new_img.ndim == 2 
                        and "header" in item 
                        and item["header"] is not None 
                        and item["header"].get('BAYERPAT')):
                        print(f"[DEBUG] Debayering image: {item['path']}")
                        new_img = self.debayer_image(new_img, item["path"], item["header"])
                        print(f"[DEBUG] debayered new_img shape={new_img.shape}, dtype={new_img.dtype}, "
                            f"min={np.min(new_img):.3f}, max={np.max(new_img):.3f}")

                new_img = crop_5px_border(new_img)
                if new_img.size == 0:
                    print(f"[DEBUG] new_img is empty after cropping => skip.")
                    self.status_label.setText(f"Skipping {item['path']} (too small after crop).")
                    QApplication.processEvents()
                    return False

                self.status_label.setText(f"Removing linear gradient from {item['path']}...")
                QApplication.processEvents()
                # Create an instance of PolyGradientRemoval with degree 1.
                poly_remover = PolyGradientRemoval(new_img, poly_degree=2, downsample_scale=5, num_sample_points=100)
                # Process the image to remove the gradient
                new_img = poly_remover.process()
                print("[DEBUG] Finished polynomial gradient removal on subframe.")

                # If mono:
                self.status_label.setText(f"Normalizing {item['path']}.")
                QApplication.processEvents()
                new_img = new_img-np.min(new_img)
                if new_img.ndim == 2:
                    new_img = stretch_mono_image(new_img, target_median=base_median, normalize=False, apply_curves=False, curves_boost=0.0)

                else:
                    # color approach - you might do a single ratio or call your stretch_color_image
                    new_img = stretch_color_image(new_img, target_median=base_median, linked=False, normalize=False, apply_curves=False, curves_boost=0.0)


                # If mosaic is color and new_img is mono, expand new_img
                # But remember, mosaic_sum is our "canvas"
                if mosaic_sum.ndim == 3 and new_img.ndim == 2:
                    print("[DEBUG] Expanding new_img from 2D => 3D to match mosaic channels.")
                    new_img = np.repeat(new_img[:, :, np.newaxis], mosaic_sum.shape[2], axis=2)
                elif mosaic_sum.ndim == 2 and new_img.ndim == 3:
                    print("[DEBUG] new_img is color but mosaic is mono => average new_img channels.")
                    new_img = np.mean(new_img, axis=2, dtype=np.float32)

                # Create the grayscale for astroalign from mosaic_sum/mosaic_count
                mosaic_gray = get_current_mosaic_gray()

                new_gray = new_img
                if new_gray.ndim == 3:
                    new_gray = np.mean(new_gray, axis=2)

                # Optional: mild stretch to help astroalign
                mosaic_gray = stretch_mono_image(mosaic_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)
                new_gray = stretch_mono_image(new_gray, target_median=0.1, normalize=False, apply_curves=False, curves_boost=0.0)

                # Attempt rough transform via WCS
                rough_matrix = None
                if ("header" in item and item["header"] is not None
                    and "CTYPE1" in item["header"] and "CTYPE2" in item["header"]):
                    # Normal WCS approach
                    rough_matrix = compute_rough_transform(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] WCS-based rough transform:\n", rough_matrix)
                else:
                    # Fallback: check if we have RA,DEC,XPIXSZ,FOCALLEN, etc.
                    rough_matrix = compute_rough_transform_seestar(item["header"], base_item["header"])
                    if rough_matrix is not None:
                        print("[DEBUG] Seestar rough transform:\n", rough_matrix)

                self.status_label.setText(f"Astroaligning for {item['path']}...")
                QApplication.processEvents()

                # Find transform
                transform_matrix = None
                try:
                    # astroalign wants double precision
                    transform_obj, (src_pts, dst_pts) = astroalign.find_transform(
                        new_gray,
                        mosaic_gray,
                        max_control_points=50,
                        detection_sigma=4,
                        min_area=5
                    )
                    transform_matrix = transform_obj.params[0:2, :].astype(np.float32)
                    print(f"[DEBUG] Astroalign success with {len(src_pts)} stars. transform_matrix:\n{transform_matrix}")

                except Exception as e:
                    print(f"[DEBUG] astroalign.find_transform failed: {e}")
                    self.status_label.setText(f"Alignment failed: {e}")
                    QApplication.processEvents()

                    # Fallback if rough_matrix is available
                    if rough_matrix is not None:
                        print("[DEBUG] Attempting refine_via_overlap fallback with rough_matrix.")
                        transform_matrix = self.refine_via_overlap(new_gray, mosaic_gray, rough_matrix)
                        if transform_matrix is None:
                            print("[DEBUG] Overlap approach also failed => skip this image.")
                            return False
                        print("[DEBUG] Overlap-based refinement succeeded. transform_matrix:\n", transform_matrix)
                    else:
                        print("[DEBUG] No rough transform => skipping image.")
                        return False

                if transform_matrix is None:
                    self.status_label.setText("No transform found; skipping image.")
                    print("[DEBUG] transform_matrix is None => returning False")
                    return False

                self.status_label.setText(f"Astroalign success for {item['path']}")
                QApplication.processEvents()

                # Warp the new image onto mosaic_sum's coordinate system
                h_m, w_m = mosaic_sum.shape[:2]
                new_h, new_w = new_img.shape[:2]

                mosaic_corners = np.array([[0, 0], [w_m, 0], [0, h_m], [w_m, h_m]], dtype=np.float32)
                new_img_corners = np.array([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]], dtype=np.float32)
                ones = np.ones((4, 1), dtype=np.float32)
                new_img_corners_hom = np.hstack([new_img_corners, ones])

                warped_corners = (transform_matrix @ new_img_corners_hom.T).T
                all_corners = np.vstack([mosaic_corners, warped_corners])
                min_xy = np.min(all_corners, axis=0)
                max_xy = np.max(all_corners, axis=0)

                margin = 10
                new_canvas_width = int(np.ceil(max_xy[0] - min_xy[0])) + margin
                new_canvas_height = int(np.ceil(max_xy[1] - min_xy[1])) + margin
                shift = -min_xy + np.array([margin / 2, margin / 2])
                shift_int = np.round(shift).astype(int)
                y0, x0 = shift_int[1], shift_int[0]

                print(f"[DEBUG] new_canvas_width={new_canvas_width}, new_canvas_height={new_canvas_height}")
                print(f"[DEBUG] shift={shift}, shift_int={shift_int}, y0={y0}, x0={x0}")

                # Expand mosaic_sum/mosaic_count to fit new canvas if needed
                expanded_sum = None
                expanded_count = None

                if mosaic_sum.ndim == 3:
                    channels = mosaic_sum.shape[2]
                    expanded_sum, _ = smart_zeros((new_canvas_height, new_canvas_width, channels), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    # Copy existing mosaic_sum/mosaic_count
                    expanded_sum[y0:y0+h_m, x0:x0+w_m, :] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m, :] = mosaic_count
                else:
                    expanded_sum, _ = smart_zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                    expanded_count = np.zeros_like(expanded_sum, dtype=np.float32)
                    expanded_sum[y0:y0+h_m, x0:x0+w_m] = mosaic_sum
                    expanded_count[y0:y0+h_m, x0:x0+w_m] = mosaic_count

                # Adjust transform for the shift
                new_transform = transform_matrix.copy()
                new_transform[0, 2] += shift[0]
                new_transform[1, 2] += shift[1]

                # Warp the new_img
                try:
                    warped_new = cv2.warpAffine(
                        new_img,
                        new_transform,
                        (new_canvas_width, new_canvas_height),
                        flags=cv2.INTER_LANCZOS4
                    )
                except cv2.error as cv2_err:
                    print(f"[OpenCV] warpAffine error => {cv2_err}")
                    return False

                # 1) Build the "border_mask" that excludes the outer POST_WARP_BORDER region
                h_w, w_w = warped_new.shape[:2]
                border_mask = np.ones((h_w, w_w), dtype=bool)
                b = POST_WARP_BORDER
                border_mask[:b, :] = False
                border_mask[-b:, :] = False
                border_mask[:, :b] = False
                border_mask[:, -b:] = False

                # 2) Build a "valid_mask" by warping an image of ones using INTER_NEAREST.
                ones_img = np.ones(new_img.shape[:2], dtype=np.uint8)
                warped_mask = cv2.warpAffine(
                    ones_img,
                    new_transform,
                    (new_canvas_width, new_canvas_height),
                    flags=cv2.INTER_NEAREST
                )
                valid_mask = (warped_mask == 1)

                # 2b) Compute a feathering weight from the valid_mask:
                # Compute the distance transform of the valid_mask.
                # This gives, for each pixel inside the valid region, its distance (in pixels)
                # from the nearest 0 (i.e. from the border of the valid region).
                FEATHER_RADIUS = 20  # adjust as needed; this is the maximum distance for feathering
                # Convert valid_mask to uint8 for distance transform.
                valid_uint8 = valid_mask.astype(np.uint8)
                dist = cv2.distanceTransform(valid_uint8, cv2.DIST_L2, 5)
                # Compute weights: linear ramp from 0 (at distance=0) to 1 (at distance>=FEATHER_RADIUS).
                feather_weights = np.clip(dist / FEATHER_RADIUS, 0, 1)

                # 3) Build "mosaic_gray" = the grayscale of (expanded_sum / expanded_count)
                mosaic_gray = np.zeros((new_canvas_height, new_canvas_width), dtype=np.float32)
                if expanded_sum.ndim == 2:
                    nonzero_mask = (expanded_count > 0)
                    mosaic_gray[nonzero_mask] = expanded_sum[nonzero_mask] / expanded_count[nonzero_mask]
                else:
                    sum_channels = np.sum(expanded_sum, axis=2)
                    sum_counts = np.sum(expanded_count, axis=2)
                    nonzero_mask = (sum_counts > 0)
                    mosaic_gray[nonzero_mask] = sum_channels[nonzero_mask] / sum_counts[nonzero_mask]

                # 4) Build new_gray for the warped image
                if warped_new.ndim == 2:
                    new_gray = warped_new
                else:
                    new_gray = np.mean(warped_new, axis=2)

                # 5) Build the brightness_mask: accept pixels where new_gray >= THRESHOLD_RATIO * mosaic_gray.
                brightness_mask = (new_gray >= THRESHOLD_RATIO * mosaic_gray)

                # 6) Combine masks.
                # Here we combine border_mask, brightness_mask, and valid_mask.
                # The final effective weight will be the feathering weight from within the valid_mask,
                # applied only where the other masks also pass.
                combined_mask = border_mask & valid_mask & brightness_mask

                # 7) Add only those pixels, using the feathering weights.
                if warped_new.ndim == 2:
                    # MONO: weight each pixel accordingly.
                    expanded_sum[combined_mask] += warped_new[combined_mask] * feather_weights[combined_mask]
                    expanded_count[combined_mask] += feather_weights[combined_mask]
                else:
                    # COLOR: do it per channel.
                    for c in range(warped_new.shape[2]):
                        expanded_sum[..., c][combined_mask] += warped_new[..., c][combined_mask] * feather_weights[combined_mask]
                        expanded_count[..., c][combined_mask] += feather_weights[combined_mask]

                # 8) Update mosaic_sum/mosaic_count.
                mosaic_sum = expanded_sum
                mosaic_count = expanded_count


                self.status_label.setText(f"Integrated image: {item['path']}")
                QApplication.processEvents()

                print("[DEBUG] process_alignment done successfully for this image.")
                return True

            except Exception as e:
                print(f"[DEBUG] process_alignment error => {e}")
                traceback.print_exc()
                return False

        # ---------------------------------------------------------
        # Process each subsequent image once
        # ---------------------------------------------------------
        failed_items = []
        # We'll do a normal for-loop with enumerate so we can pass index to progress
        for i, item in enumerate(self.loaded_images):
            # If user cancels
            if progress.wasCanceled():
                self.status_label.setText("Alignment canceled by user.")
                break

            # Update the progress label
            progress.setLabelText(f"Aligning image {i+1}/{total_files}: {item['path']}")
            progress.setValue(i)  # update the progress bar

            if item is base_item:
                continue

            success = process_alignment(item)
            if not success:
                failed_items.append(item)

            QApplication.processEvents()  # allow UI updates

        # Final step: mark progress done
        progress.setValue(total_files)

        # ---------------------------------------------------------
        # Reattempt failed images (max 1 additional attempt)
        # ---------------------------------------------------------
        max_retries = 1
        attempt = 1
        while failed_items and attempt <= max_retries:
            reattempt_count = len(failed_items)
            self.status_label.setText(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            print(f"Reattempting alignment for {reattempt_count} images (Attempt {attempt})")
            QApplication.processEvents()

            # 2) Reset the existing progress bar to track the reattempt pass
            progress.setRange(0, reattempt_count)
            progress.setValue(0)
            progress.setLabelText(f"Reattempt pass {attempt}...")

            current_failures = []
            for i, item in enumerate(failed_items):
                if progress.wasCanceled():
                    self.status_label.setText("Reattempt canceled by user.")
                    break

                progress.setLabelText(f"Reattempting {item['path']} ({i+1}/{reattempt_count})")
                progress.setValue(i)
                QApplication.processEvents()

                success = process_alignment(item)
                if not success:
                    current_failures.append(item)

            progress.setValue(reattempt_count)  # done reattempt pass

            failed_items = current_failures
            attempt += 1

        # All passes complete
        progress.close()
        self.status_label.setText("All alignment attempts complete.")
        QApplication.processEvents()

        # ---------------------------------------------------------
        # Finally, build the average mosaic = mosaic_sum / mosaic_count
        # ---------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            final_mosaic = np.zeros_like(mosaic_sum, dtype=np.float32)
            if final_mosaic.ndim == 2:
                nonzero = (mosaic_count > 0)
                final_mosaic[nonzero] = mosaic_sum[nonzero] / mosaic_count[nonzero]
            else:
                # color
                for c in range(final_mosaic.shape[2]):
                    chan_nonzero = (mosaic_count[..., c] > 0)
                    final_mosaic[..., c][chan_nonzero] = (
                        mosaic_sum[..., c][chan_nonzero] 
                        / mosaic_count[..., c][chan_nonzero]
                    )

        # Optional: you could do a final normalization
        max_val = np.max(final_mosaic)
        if max_val > 0:
            final_mosaic /= max_val

        self.final_mosaic = final_mosaic
        self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        self.status_label.setText("Seestar Mode alignment (sum/divide) complete.")
        QApplication.processEvents()

        display_image = self.stretch_for_display(self.final_mosaic)
        mosaic_win = MosaicPreviewWindow(display_image, title="Robot Telescope Mosaic", parent=self,
                    push_cb=self._push_mosaic_to_new_doc).show()

    # ---------- Star alignment using triangle matching ----------

    def refined_alignment(self, affine_aligned, mosaic_img, method="Homography Transform"):
        """
        Refined alignment that assumes affine_aligned is the result of the affine alignment step.
        It re-detects stars in the candidate (overlap) region and computes a refined transform from
        affine_aligned to mosaic_img. Then it applies the refined transform to affine_aligned and
        returns the fully warped image.
        
        Returns:
        - For "Homography Transform": (warped_image, inlier_count)
        - For "Polynomial Warp Based Transform": (warped_image, inlier_count)
        - If refinement fails, returns None.
        """
        print("\n--- Starting Refined Alignment ---")
        poly_degree = qs_int(self.settings, "mosaic/poly_degree", 3)
        self.status_label.setText("Refinement: Converting images to grayscale...")
        QApplication.processEvents()
        
        # Convert images to grayscale
        if affine_aligned.ndim == 3:
            affine_aligned_gray = np.mean(affine_aligned, axis=-1)
        else:
            affine_aligned_gray = affine_aligned
        if mosaic_img.ndim == 3:
            mosaic_gray = np.mean(mosaic_img, axis=-1)
        else:
            mosaic_gray = mosaic_img

        print("Grayscale conversion done.")
        
        # Compute overlap mask
        self.status_label.setText("Refinement: Computing overlap mask...")
        QApplication.processEvents()
        overlap_mask = (mosaic_gray > 0) & (affine_aligned_gray > 0)

        # Detect stars
        self.status_label.setText("Refinement: Detecting stars in mosaic and affine-aligned images...")
        QApplication.processEvents()
        # Increase max_stars to 50 for debugging purposes.
        mosaic_stars_masked = self.detect_stars(np.where(overlap_mask, mosaic_gray, 0), max_stars=300)
        new_stars_aligned = self.detect_stars(np.where(overlap_mask, affine_aligned_gray, 0), max_stars=300)

        # Debug: Print out the star lists.
        print("Mosaic stars (refined alignment):")
        for s in mosaic_stars_masked:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")
        print("New stars (refined alignment):")
        for s in new_stars_aligned:
            print(f"({s[0]:.2f}, {s[1]:.2f}) flux: {s[2]:.2f}")

        self.status_label.setText(f"Refinement: Detected {len(mosaic_stars_masked)} mosaic stars and {len(new_stars_aligned)} new stars.")
        QApplication.processEvents()

        if len(mosaic_stars_masked) < 4 or len(new_stars_aligned) < 4:
            self.status_label.setText("Refinement: Not enough stars detected in candidate region.")
            return None

        # Match stars using position and flux.
        self.status_label.setText("Refinement: Matching stars...")
        QApplication.processEvents()
        # For debugging, you might try a higher threshold.
        matches = self.match_stars(new_stars_aligned, mosaic_stars_masked, distance_thresh=10.0, flux_thresh=1.0)
        print(f"Matched stars: {len(matches)}")
        self.status_label.setText(f"Refinement: {len(matches)} matches found.")
        if len(matches) < 4:
            self.status_label.setText("Refinement: Not enough matched stars for refined transform.")
            return None

        src_pts = np.float32([match[0][:2] for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match[1][:2] for match in matches]).reshape(-1, 1, 2)

        if method == "Homography Transform":
            self.status_label.setText("Refinement: Computing homography transform...")
            QApplication.processEvents()
            H_refined, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.count_nonzero(mask)) if mask is not None else 0
            self.status_label.setText(f"Refinement: Homography computed with {inliers} inliers. Warping image...")
            QApplication.processEvents()
            if H_refined is None:
                self.status_label.setText("Refinement: Homography estimation failed.")
                return None
            warped_image = cv2.warpPerspective(affine_aligned, H_refined,
                                                (affine_aligned.shape[1], affine_aligned.shape[0]),
                                                flags=cv2.INTER_LANCZOS4)
            return (warped_image, inliers)
        elif method == "Polynomial Warp Based Transform":
            self.status_label.setText("Refinement: Computing Polynomial Warp based transform...")
            QApplication.processEvents()
            # Extract control points from matches.
            src_pts = np.float32([match[0][:2] for match in matches])
            dst_pts = np.float32([match[1][:2] for match in matches])
            # Call the static Polynomial Warp warp function.
            try:
                warped_image = MosaicMasterDialog.poly_warp(affine_aligned, src_pts, dst_pts, degree=poly_degree, status_label=self.status_label)
                inliers = len(matches)
                self.status_label.setText(f"Refinement: Polynomial Warp warp applied with {inliers} matched points.")
                return (warped_image, inliers)
            except Exception as e:
                self.status_label.setText(f"Refinement: Polynomial Warp warp failed: {e}")
                return None
        else:
            self.status_label.setText("Refinement: Unexpected transformation method.")
            return None

    @staticmethod
    def poly_warp(image, src_pts, dst_pts, degree=3, status_label=None):
        """
        Warp `image` using a polynomial transformation of the specified degree.
        
        The transformation is defined as:
        u = sum_{i+j<=degree} a_{ij} * x^i * y^j
        v = sum_{i+j<=degree} b_{ij} * x^i * y^j
        
        where the coefficients are solved via least squares using the control points.
        
        Parameters:
        image: Input image.
        src_pts: numpy array of shape (N,2) with control points in the source image.
        dst_pts: numpy array of shape (N,2) with corresponding control points in the destination.
        degree: Degree of the polynomial transformation (allowed 1 to 6, default=3).
        status_label: If provided, a Qt widget where progress messages are displayed.
        
        Returns:
        The warped image.
        """
        h, w = image.shape[:2]
        
        # Function to build the design matrix for points given a degree.
        def build_design_matrix(pts, degree):
            # pts: (N,2) array, where each row is (x, y)
            N = pts.shape[0]
            terms = []
            x = pts[:, 0]
            y = pts[:, 1]
            # Loop over exponents i and j with i+j <= degree.
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    terms.append((x**i) * (y**j))
            X = np.vstack(terms).T  # shape: (N, number_of_terms)
            return X

        # Build the design matrix for the control points.
        if status_label is not None:
            status_label.setText("Polynomial warp: Building design matrix for control points...")
            QApplication.processEvents()
        X = build_design_matrix(src_pts, degree)
        
        # Destination coordinates.
        U = dst_pts[:, 0]
        V = dst_pts[:, 1]
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Solving for polynomial coefficients...")
            QApplication.processEvents()
        
        # Solve the least-squares problem.
        coeffs_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        coeffs_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Computing full mapping for image...")
            QApplication.processEvents()
        
        # Build a full grid of coordinates.
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()
        pts_full = np.vstack([flat_x, flat_y]).T  # shape: (h*w, 2)
        
        # Build design matrix for the full grid.
        X_full = build_design_matrix(pts_full, degree)
        
        # Evaluate the polynomial mappings.
        map_u = np.dot(X_full, coeffs_u).reshape(h, w).astype(np.float32)
        map_v = np.dot(X_full, coeffs_v).reshape(h, w).astype(np.float32)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Full mapping computed. Remapping image...")
            QApplication.processEvents()
        
        # Remap the image.
        warped = cv2.remap(image, map_u, map_v, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Image warped.")
            QApplication.processEvents()
        
        return warped



    def stretch_for_display(self, arr):
        """
        Uses your global stretch_mono_image or stretch_color_image to produce
        a display-ready 8-bit image. For color images, uses unlinked stretching.
        """
        arr = arr.astype(np.float32)

        # Decide if it's mono or color based on shape
        if arr.ndim == 3 and arr.shape[2] == 3:
            # Color image => use stretch_color_image with unlinked stretching
            stretched = stretch_color_image(
                image=arr,
                target_median=0.25,   # Adjust if you prefer a different default
                linked=False,         # "unlinked" mode
                normalize=True,       # Ensures final values are in [0,1]
                apply_curves=False,   # Adjust if needed
                curves_boost=0.0,
                no_black_clip=True
            )
        else:
            # Mono image => use stretch_mono_image
            stretched = stretch_mono_image(
                image=arr,
                target_median=0.25,
                normalize=True,
                apply_curves=False,
                curves_boost=0.0,
                no_black_clip=True,
            )

        # Convert [0,1] => [0,255]
        disp = (stretched * 255.0).clip(0, 255).astype(np.uint8)
        return disp


    def detect_stars(self, image2d, max_stars=50):
        # Retrieve user-defined values for sigma and fwhm.
        from photutils.detection import DAOStarFinder
        sigma_val = self.settings.value("mosaic/star_sigma", 3.0, type=float)
        fwhm_val = self.settings.value("mosaic/star_fwhm", 3.0, type=float)
        
        mean_val, median_val, std_val = sigma_clipped_stats(image2d, sigma=3.0)
        # Use the user-defined fwhm and scale the threshold by the standard deviation.
        daofind = DAOStarFinder(threshold=sigma_val * std_val, fwhm=fwhm_val)
        sources = daofind(image2d - median_val)
        if sources is None or len(sources) == 0:
            return []
        x_coords = sources['xcentroid'].data
        y_coords = sources['ycentroid'].data
        flux = sources['flux'].data
        # Sort stars by brightness (flux) and select the top ones.
        sorted_indices = np.argsort(-flux)
        top_indices = sorted_indices[:max_stars]
        stars = [(x_coords[i], y_coords[i], flux[i]) for i in top_indices]
        return stars


    def match_stars(self, new_stars, mosaic_stars, distance_thresh=10.0, flux_thresh=0.2):
        """
        Matches stars between two lists.
        new_stars and mosaic_stars should be lists of (x, y, flux).
        
        This version normalizes the flux values in each list by dividing by the median flux.
        
        distance_thresh: maximum distance (in pixels) allowed for a match.
        flux_thresh: allowed absolute difference in normalized flux.
                    For example, if set to 0.2, then the normalized flux difference must be less than 0.2.
                    
        Returns a list of matched pairs: [(new_star, mosaic_star), ...]
        """
        # If either list is empty, return an empty match list.
        if not new_stars or not mosaic_stars:
            return []
        
        # Compute median fluxes.
        new_fluxes = [s[2] for s in new_stars]
        mosaic_fluxes = [s[2] for s in mosaic_stars]
        new_median = np.median(new_fluxes) if new_fluxes else 1.0
        mosaic_median = np.median(mosaic_fluxes) if mosaic_fluxes else 1.0

        # Normalize the flux for each star.
        norm_new_stars = [(s[0], s[1], s[2] / new_median) for s in new_stars]
        norm_mosaic_stars = [(s[0], s[1], s[2] / mosaic_median) for s in mosaic_stars]

        matches = []
        for ns in norm_new_stars:
            best_match = None
            best_distance = float('inf')
            for ms in norm_mosaic_stars:
                dx = ns[0] - ms[0]
                dy = ns[1] - ms[1]
                dist = np.hypot(dx, dy)
                # Check spatial proximity.
                if dist < distance_thresh and dist < best_distance:
                    # Check if the normalized fluxes are similar.
                    # Here, flux_thresh is an absolute threshold on the difference.
                    if abs(ns[2] - ms[2]) < flux_thresh:
                        best_match = ms
                        best_distance = dist
            if best_match is not None:
                matches.append((ns, best_match))
        return matches

    def estimate_transform(self, source_stars, dest_stars):
        min_len = min(len(source_stars), len(dest_stars))
        if min_len < 3:
            return None
        src_pts = np.float32(source_stars[:min_len])
        dst_pts = np.float32(dest_stars[:min_len])
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is None:
            return None
        full_mat = np.eye(3, dtype=np.float32)
        full_mat[:2] = matrix
        return full_mat


    def compute_mosaic_bounding_box(self, wcs_items, reference_wcs):
        """
        Compute the mosaic bounding box in pixel coordinates relative to a shared WCS frame,
        properly accounting for rotation and orientation dynamically.
        """
        all_pixels = []

        for itm in wcs_items:
            wcs = itm["wcs"]
            H, W = itm["image"].shape[:2]  # Use only the height and width

            # Get image corner coordinates in world coordinates (RA/Dec)
            pixel_corners = np.array([
                [0, 0],         # Top-left
                [W - 1, 0],     # Top-right
                [0, H - 1],     # Bottom-left
                [W - 1, H - 1]  # Bottom-right
            ])

            # Convert pixel to world coordinates (RA, Dec)
            world_coords = np.column_stack(wcs.pixel_to_world_values(pixel_corners[:, 0], pixel_corners[:, 1]))

            # Convert RA/Dec to pixel coordinates in the reference WCS
            sky_coords = SkyCoord(ra=world_coords[:, 0] * u.deg, dec=world_coords[:, 1] * u.deg, frame='icrs')
            pixel_coords = skycoord_to_pixel(sky_coords, reference_wcs)

            # Ensure we're only using the first two values (x, y)
            all_pixels.append(np.column_stack(pixel_coords[:2]))

        # Stack all pixel coordinates and compute bounding box
        all_pixels = np.vstack(all_pixels)

        min_x, max_x = np.min(all_pixels[:, 0]), np.max(all_pixels[:, 0])
        min_y, max_y = np.min(all_pixels[:, 1]), np.max(all_pixels[:, 1])

        # **Determine whether the mosaic is wider or taller dynamically**
        width = max_x - min_x
        height = max_y - min_y

        print(f"Detected Bounding Box (X={min_x} to {max_x}, Y={min_y} to {max_y})")
        print(f"Calculated Mosaic Size: Width={width}, Height={height}")
        self.status_label.setText(f"Detected Bounding Box: X={min_x} to {max_x}, Y={min_y} to {max_y}")
        self.status_label.setText(f"Calculated Mosaic Size: Width={width}, Height={height}")
        QApplication.processEvents()

        return int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x)), int(np.ceil(max_y))

    def save_mosaic_to_new_view(self):
        """
        Finalize and create a brand-new document/view in SASpro.
        """
        finalized = self.finalize_mosaic()
        if finalized is None:
            QMessageBox.information(self, "Mosaic Master", "No mosaic to save.")
            return

        img, meta = finalized

        if not self._docman:
            QMessageBox.warning(self, "Mosaic Master", "No document manager available.")
            return

        title = self._proposed_mosaic_title()

        try:
            if hasattr(self._docman, "open_array") and callable(self._docman.open_array):
                newdoc = self._docman.open_array(img, metadata=meta, title=title)
            elif hasattr(self._docman, "open_numpy") and callable(self._docman.open_numpy):
                newdoc = self._docman.open_numpy(img, metadata=meta, title=title)
            else:
                # very old API
                newdoc = self._docman.create_document(image=img, metadata=meta, name=title)

            # show it in a subwindow if the host provides the helper
            parent = self.parent() if hasattr(self, "parent") else None
            if parent and hasattr(parent, "_spawn_subwindow_for"):
                parent._spawn_subwindow_for(newdoc)

            QMessageBox.information(self, "Mosaic Master", "Created a new view for the mosaic.")
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Mosaic Master", f"Failed to create new view:\n{e}")

    def _proposed_mosaic_title(self) -> str:
        return "Mosaic"

    def finalize_mosaic(self):
        """
        Prepare the final mosaic (and its WCS metadata) and return (image, metadata).
        Does not push anywhere. Returns None on failure.

        The mosaic's WCS is the linear-TAN frame every tile was reprojected into
        (mosaic_wcs, stored as self.wcs_metadata in align_images), so it's the exact
        astrometric solution for the output canvas — not a fit. We attach it under
        'original_header' so the rest of SASpro (plate viewer, re-open, FITS save,
        Stellar Alignment) finds it the same way it does for any solved frame.
        """
        if self.final_mosaic is None:
            print("No mosaic to finalize.")
            return None

        is_mono = (self.final_mosaic.ndim == 2 or
                   (self.final_mosaic.ndim == 3 and self.final_mosaic.shape[2] == 1))

        # Build the header
        if not self.wcs_metadata or not any(self.wcs_metadata.values()):
            # No WCS available (e.g. Seestar-mode mosaic) — minimal header only.
            print("WCS metadata not available; creating minimal header.")
            hdr = self.create_minimal_fits_header(self.final_mosaic, is_mono)
        else:
            # Keep it as a real, sanitized Header so WCS(relax=True) reparses it.
            hdr = sanitize_wcs_header(self.wcs_metadata) or coerce_to_header(self.wcs_metadata)
            if hdr is None:
                print("WCS metadata present but unparseable; falling back to minimal header.")
                hdr = self.create_minimal_fits_header(self.final_mosaic, is_mono)

        meta = {
            "original_header": hdr,        # ← where SASpro reads WCS from
            "step_name": "Mosaic Master",
            "is_mono": bool(is_mono),
        }

        # Keep the array as-is (2D mono or 3-channel)
        return self.final_mosaic, meta

    def unstretch_image(self, image: np.ndarray) -> np.ndarray:
        """
        Best-effort inverse of the earlier median normalization.
        If mono, scale so median(image) -> original_median[0].
        If color, scale each channel so median(channel) -> original_median[c].
        Requires that self.stretch_original_medians / mins were populated per input.
        If stats are unavailable or inconsistent, returns image unchanged.
        """
        try:
            img = image.astype(np.float32, copy=True)

            orig_meds = getattr(self, "stretch_original_medians", None)
            orig_mins = getattr(self, "stretch_original_mins", None)
            if not orig_meds or not orig_mins:
                return img  # nothing to do

            # Use the FIRST frame’s recorded stats as the anchor for de-normalization.
            # (This is a heuristic; a true per-frame inverse is impossible after blending.)
            anchor_med = float(orig_meds[0])
            anchor_min = float(orig_mins[0])

            if anchor_med <= 0:
                return img

            if img.ndim == 2:
                cur_med = float(np.median(img))
                if cur_med > 0:
                    gain = anchor_med / cur_med
                    img = img * gain
                # keep dynamic range plausible
                img = np.clip(img, 0.0, 1.0)
                return img

            # color: per-channel median scaling (unlinked)
            for c in range(min(3, img.shape[2])):
                cur_med = float(np.median(img[..., c]))
                # choose a per-channel original median if available; fallback to anchor
                src_med = float(orig_meds[c]) if len(orig_meds) >= 3 else anchor_med
                if cur_med > 0 and src_med > 0:
                    gain = src_med / cur_med
                    img[..., c] *= gain

            img = np.clip(img, 0.0, 1.0)

            # If the mosaic was originally single-channel, keep it mono.
            if getattr(self, "was_single_channel", False) and img.ndim == 3:
                img = np.mean(img, axis=2).astype(np.float32)

            return img

        except Exception as e:
            print(f"[unstretch_image] fallback (no-op) due to: {e}")
            return image

    def _save_astap_exe_to_settings(self, path: str) -> None:
        path = os.path.normpath(os.path.expanduser(os.path.expandvars(path)))
        self.astap_exe = path
        self.settings.setValue("paths/astap", path)
        self.settings.sync()

    def _load_astrometry_api_key(self) -> str:
        # primary source: QSettings (matches SettingsDialog)
        key = self.settings.value("api/astrometry_key", "", type=str) or ""
        if key:
            return key
        # optional fallback to any app-level file helpers you already had
        try:
            k = load_api_key()
            return k or ""
        except Exception:
            return ""

    def _save_astrometry_api_key(self, key: str) -> None:
        key = (key or "").strip()
        self.settings.setValue("api/astrometry_key", key)
        self.settings.sync()
        # keep old helper in sync if you want
        try:
            save_api_key(key)
        except Exception:
            pass


    # ---------- Blind Solve via Astrometry.net ----------
    def perform_blind_solve(self, item):
        """
        Blind solve with Astrometry.net, construct WCS, keep SIP, and (optionally) update on-disk FITS.
        """
        while True:
            self.status_label.setText("Status: Logging in to Astrometry.net...")
            QApplication.processEvents()
            api_key = self._get_astrometry_api_key()
            if not api_key:
                api_key, ok = QInputDialog.getText(self, "Enter API Key", "Please enter your Astrometry.net API key:")
                if ok and api_key:
                    self.settings.setValue("api/astrometry_key", api_key)
                    self.settings.sync()
                else:
                    QMessageBox.warning(self, "API Key Required", "Blind solve canceled (no API key).")
                    return None

            session_key = self.login_to_astrometry(api_key)
            if session_key is None:
                if QMessageBox.question(self, "Login Failed",
                                        "Could not log in to Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Uploading image to Astrometry.net...")
            QApplication.processEvents()

            # Determine the file extension of the original image
            ext = os.path.splitext(str(item.get("path","")))[1].lower()
            if ext not in ('.fits', '.fit'):
                tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
                tmp.close()
                minimal_header = generate_minimal_fits_header(item["image"])
                save_image(
                    img_array=item["image"],
                    filename=tmp.name,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=minimal_header,
                    is_mono=item.get("is_mono", False)
                )
                upload_path = tmp.name
            else:
                upload_path = item["path"]

            subid = self.upload_image_to_astrometry(upload_path, session_key)
            if not subid:
                if QMessageBox.question(self, "Upload Failed",
                                        "Image upload failed or no subid returned. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Waiting for job ID...")
            QApplication.processEvents()
            job_id = self.poll_submission_status(subid)
            if not job_id:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Failed to retrieve job ID from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            self.status_label.setText("Status: Retrieving calibration data...")
            QApplication.processEvents()
            calibration_data = self.poll_calibration_data(job_id)
            if not calibration_data:
                if QMessageBox.question(self, "Blind Solve Failed",
                                        "Calibration data did not arrive from Astrometry.net. Try again?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                    continue
                else:
                    return None

            # Clean up tmp if we created one
            if ext not in ('.fits', '.fit', '.tif', '.tiff'):
                try:
                    os.remove(upload_path)
                except Exception:
                    pass

            break  # success

        # Build + normalize header, keep SIP, and build WCS with relax=True
        wcs_header = self.construct_wcs_header(calibration_data, item["image"].shape)
        wcs_header = self._normalize_wcs_header(wcs_header, item["image"].shape)
        item["wcs"] = self._build_wcs(wcs_header, item["image"].shape)

        dest = str(item.get("path", ""))
        if dest.lower().endswith((".fits", ".fit")) and not self._is_view_path(dest):
            self.update_fits_with_wcs(dest, calibration_data, wcs_header)
        else:
            print("Blind solve succeeded; not updating file on disk (view or non-FITS source).")

        return wcs_header


    def login_to_astrometry(self, api_key):
        url = ASTROMETRY_API_URL + "login"
        data = {'request-json': json.dumps({"apikey": api_key})}
        response = robust_api_request("POST", url, data=data, prompt_on_failure=True)
        if response and response.get("status") == "success":
            return response["session"]
        print("Login failed after multiple attempts.")
        QMessageBox.critical(self, "Login Failed", "Could not log in to Astrometry.net. Check your API key or internet connection.")
        return None

    def upload_image_to_astrometry(self, image_path, session_key):
        url = ASTROMETRY_API_URL + "upload"
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'request-json': json.dumps({
                    "publicly_visible": "y",
                    "allow_modifications": "d",
                    "session": session_key,
                    "allow_commercial_use": "d"
                })
            }
            response = robust_api_request("POST", url, data=data, files=files)
        if response and response.get("status") == "success":
            return response["subid"]
        QMessageBox.critical(self, "Upload Failed", "Image upload failed after multiple attempts.")
        return None

    def poll_submission_status(self, subid):
        from setiastro.saspro.main_helpers import non_blocking_sleep
        url = ASTROMETRY_API_URL + f"submissions/{subid}"
        for attempt in range(90):  # up to ~15 minutes
            response = robust_api_request("GET", url)
            if response:
                jobs = response.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
            print(f"Polling attempt {attempt+1}: Job ID not ready yet.")
            non_blocking_sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Failed to retrieve job ID from Astrometry.net after multiple attempts.")
        return None

    def poll_calibration_data(self, job_id):
        from setiastro.saspro.main_helpers import non_blocking_sleep
        url = ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/"
        for attempt in range(90):
            response = robust_api_request("GET", url)
            if response and 'ra' in response and 'dec' in response:
                print("Calibration data retrieved:", response)
                return response
            print(f"Calibration data not available yet (attempt {attempt+1})")
            non_blocking_sleep(10)
        QMessageBox.critical(self, "Blind Solve Failed", "Calibration data did not complete in the expected timeframe.")
        return None

    def construct_wcs_header(self, calibration_data, image_shape):
        h = Header()
        h['CTYPE1'] = 'RA---TAN'
        h['CTYPE2'] = 'DEC--TAN'
        h['CRPIX1'] = image_shape[1] / 2
        h['CRPIX2'] = image_shape[0] / 2
        h['CRVAL1'] = calibration_data['ra']
        h['CRVAL2'] = calibration_data['dec']
        scale = calibration_data['pixscale'] / 3600.0  # degrees/pixel
        orientation = math.radians(calibration_data['orientation'])
        h['CD1_1'] = -scale * np.cos(orientation)
        h['CD1_2'] = scale * np.sin(orientation)
        h['CD2_1'] = -scale * np.sin(orientation)
        h['CD2_2'] = -scale * np.cos(orientation)
        h['RADECSYS'] = 'ICRS'
        h['WCSAXES'] = 2
        print("Generated WCS header from calibration data.")
        return h

    def update_fits_with_wcs(self, filepath, calibration_data, wcs_header):
        if not filepath.lower().endswith(('.fits','.fit')):
            print("Not a FITS, skipping WCS update.")
            return
        try:
            with fits.open(filepath, mode='update') as hdul:
                hdr = hdul[0].header
                if 'NAXIS3' in hdr:
                    del hdr['NAXIS3']
                hdr['NAXIS'] = 2
                hdr['CTYPE1'] = 'RA---TAN'
                hdr['CTYPE2'] = 'DEC--TAN'
                hdr['CRVAL1'] = calibration_data['ra']
                hdr['CRVAL2'] = calibration_data['dec']
                # Determine H and W based on the data's dimensionality.
                if hdul[0].data.ndim == 3:
                    # Assume data are stored as (channels, height, width)
                    _, H, W = hdul[0].data.shape
                else:
                    H, W = hdul[0].data.shape[:2]
                hdr['CRPIX1'] = W/2.0
                hdr['CRPIX2'] = H/2.0
                scale = calibration_data['pixscale']/3600.0
                orientation = math.radians(calibration_data.get('orientation', 0.0))
                hdr['CD1_1'] = -scale * np.cos(orientation)
                hdr['CD1_2'] = scale * np.sin(orientation)
                hdr['CD2_1'] = -scale * np.sin(orientation)
                hdr['CD2_2'] = -scale * np.cos(orientation)
                hdr['WCSAXES'] = 2
                hdr['RADECSYS'] = 'ICRS'
                hdul.flush()
                print("WCS updated in FITS.")
            # Re-open to verify changes:
            with fits.open(filepath) as hdul_verify:
                print("Updated header keys:", hdul_verify[0].header.keys())
        except Exception as e:
            print(f"Error updating FITS with WCS: {e}")

    # ---------- Blind Solve via ASTAP ----------
    def attempt_astap_solve(self, item):
        """
        Attempt to plate-solve the image using ASTAP.
        Returns a solved header (as a dict) on success or None on failure.
        """
        # 1) Normalize the image (using your stretch_image).
        normalized_image = self.stretch_image(item["image"])
        
        # 2) Save normalized image to a temporary FITS file for ASTAP.
        try:
            tmp_path = self.save_temp_fits_image(normalized_image, item["path"])
        except Exception as e:
            print("Failed to save temporary FITS file:", e)
            return None

        # 3) Run ASTAP on the temporary file.
        process = QProcess(self)
        args = ["-f", tmp_path, "-r", "179", "-fov", "0", "-z", "0", "-wcs", "-sip"]
        print("Running ASTAP with arguments:", args)
        process.start(self.astap_exe, args)
        if not process.waitForStarted(5000):
            print("Failed to start ASTAP process:", process.errorString())
            os.remove(tmp_path)
            return None
        if not process.waitForFinished(300000):  # wait up to 5 minutes
            print("ASTAP process timed out.")
            os.remove(tmp_path)
            return None

        exit_code = process.exitCode()
        stdout = process.readAllStandardOutput().data().decode()
        stderr = process.readAllStandardError().data().decode()
        print("ASTAP exit code:", exit_code)
        print("ASTAP STDOUT:\n", stdout)
        print("ASTAP STDERR:\n", stderr)

        if exit_code != 0:
            try:
                os.remove(tmp_path)
            except Exception as e:
                print("Error removing temporary file:", e)
            return None

        # 4) Retrieve updated header from the temporary file.
        try:
            with fits.open(tmp_path, memmap=False) as hdul:
                solved_header = dict(hdul[0].header)
            # Remove some extraneous keywords
            solved_header.pop("COMMENT", None)
            solved_header.pop("HISTORY", None)
        except Exception as e:
            print("Error reading solved header:", e)
            os.remove(tmp_path)
            return None

        # 5) Merge .wcs file (if ASTAP wrote one) into the solved_header.
        wcs_path = os.path.splitext(tmp_path)[0] + ".wcs"
        if os.path.exists(wcs_path):
            try:
                with open(wcs_path, "r") as f:
                    content = f.read()
                pattern = r"(\w+)\s*=\s*('?[^/']*'?)[\s/]"
                for match in re.finditer(pattern, content):
                    key = match.group(1).strip().upper()
                    val = match.group(2).strip()
                    if val.startswith("'") and val.endswith("'"):
                        val = val[1:-1].strip()
                    solved_header[key] = val
            except Exception as e:
                print("Error reading .wcs file:", e)
            finally:
                try:
                    os.remove(wcs_path)
                except Exception as e:
                    print("Error removing .wcs file:", e)

        # Remove the END keyword if present
        solved_header.pop("END", None)
        # Remove any unneeded keywords
        for keyword in ["RANGE_LOW", "RANGE_HIGH", "HISTORY"]:
            solved_header.pop(keyword, None)

        # --- Ensure required WCS keys are present ---
        if "CTYPE1" not in solved_header or not solved_header["CTYPE1"].strip():
            solved_header["CTYPE1"] = "RA---TAN"
        if "CTYPE2" not in solved_header or not solved_header["CTYPE2"].strip():
            solved_header["CTYPE2"] = "DEC--TAN"
        if "RADECSYS" not in solved_header:
            solved_header["RADECSYS"] = "ICRS"
        if "WCSAXES" not in solved_header:
            solved_header["WCSAXES"] = 2

        # Convert known WCS keys to float or int
        expected_float_keys = {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"}
        expected_int_keys = {"NAXIS", "WCSAXES"}

        for key in expected_float_keys:
            if key in solved_header:
                try:
                    solved_header[key] = float(solved_header[key])
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to float.")

        for key in expected_int_keys:
            if key in solved_header:
                try:
                    solved_header[key] = int(float(solved_header[key]))
                except ValueError:
                    print(f"Warning: Could not convert {key}='{solved_header[key]}' to int.")

        dest = str(item.get("path", ""))

        ok_to_overwrite = (
            bool(dest)
            and not self._is_view_path(dest)
            and dest.lower().endswith((".fits", ".fit"))
        )

        if ok_to_overwrite:
            try:
                save_image(
                    img_array=normalized_image,
                    filename=dest,
                    original_format="fit",
                    bit_depth="32",
                    original_header=solved_header,
                    is_mono=False
                )
                print(f"✅ Updated FITS header with full WCS solution for {dest}.")
            except Exception as e:
                print("Error saving updated FITS file using save_image():", e)
                # don't re-raise; we can still continue in-memory
        else:
            print("Skipping on-disk write (source is a view or not a FITS path).")

        # cleanup temp files as you already do...
        return solved_header


    def save_temp_fits_image(self, normalized_image, image_path: str):
        """
        Save the normalized_image as a FITS file to a temporary file.
        
        If the original image is FITS, this method retrieves the stored metadata
        from the ImageManager and passes it directly to save_image().
        If not, it generates a minimal header.
        
        Returns the path to the temporary FITS file.
        """
        # Always save as FITS.
        selected_format = "fits"
        bit_depth = "32-bit floating point"
        is_mono = (normalized_image.ndim == 2 or 
                   (normalized_image.ndim == 3 and normalized_image.shape[2] == 1))
        
        # If the original image is FITS, try to get its stored metadata.
        original_header = None
        is_fits_path = isinstance(image_path, str) and image_path.lower().endswith((".fits", ".fit"))
        if is_fits_path and not self._is_view_path(image_path):
            if self.parent() and hasattr(self.parent(), "image_manager"):
                # Use the metadata from the current slot.
                _, meta = self.parent().image_manager.get_current_image_and_metadata()
                # Assume that meta already contains a proper 'original_header'
                # (or the entire meta is the header).
                original_header = meta.get("original_header", None)
            # If nothing is stored, fall back to creating a minimal header.
            if original_header is None:
                print("No stored FITS header found; creating a minimal header.")
                original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        else:
            # For non-FITS images, generate a minimal header.
            original_header = self.create_minimal_fits_header(normalized_image, is_mono)
        
        # Create a temporary filename.
        tmp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        
        try:
            # Call your global save_image() exactly as in AstroEditingSuite.
            save_image(
                img_array=normalized_image,
                filename=tmp_path,
                original_format=selected_format,
                bit_depth=bit_depth,
                original_header=original_header,
                is_mono=is_mono
                # (image_meta and file_meta can be omitted if not needed)
            )
            print(f"Temporary normalized FITS saved to: {tmp_path}")
        except Exception as e:
            print("Error saving temporary FITS file using save_image():", e)
            raise e
        return tmp_path

    def create_minimal_fits_header(self, img_array, is_mono=False):
        """
        Creates a minimal FITS header when the original header is missing.
        """

        header = Header()
        header['SIMPLE'] = (True, 'Standard FITS file')
        header['BITPIX'] = -32  # 32-bit floating-point data
        header['NAXIS'] = 2 if is_mono else 3
        header['NAXIS1'] = img_array.shape[2] if img_array.ndim == 3 and not is_mono else img_array.shape[1]  # Image width
        header['NAXIS2'] = img_array.shape[1] if img_array.ndim == 3 and not is_mono else img_array.shape[0]  # Image height
        if not is_mono:
            header['NAXIS3'] = img_array.shape[0] if img_array.ndim == 3 else 1  # Number of color channels
        header['BZERO'] = 0.0  # No offset
        header['BSCALE'] = 1.0  # No scaling


        return header

    def stretch_image(self, image):
        """
        Perform an unlinked linear stretch on the image.
        Each channel is stretched independently by subtracting its own minimum,
        recording its own median, and applying the stretch formula.
        Returns the stretched image in [0,1].
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # If the image is 2D or has one channel, convert to 3-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([image] * 3, axis=-1)

        image = image.astype(np.float32).copy()
        stretched_image = image.copy()
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        target_median = 0.02

        for c in range(3):
            channel_min = np.min(stretched_image[..., c])
            self.stretch_original_mins.append(channel_min)
            stretched_image[..., c] -= channel_min
            channel_median = np.median(stretched_image[..., c])
            self.stretch_original_medians.append(channel_median)
            if channel_median != 0:
                numerator = (channel_median - 1) * target_median * stretched_image[..., c]
                denominator = (
                    channel_median * (target_median + stretched_image[..., c] - 1)
                    - target_median * stretched_image[..., c]
                )
                denominator = np.where(denominator == 0, 1e-6, denominator)
                stretched_image[..., c] = numerator / denominator
            else:
                print(f"Channel {c} - Median is zero. Skipping stretch.")

        stretched_image = np.clip(stretched_image, 0.0, 1.0)
        self.was_single_channel = was_single_channel
        return stretched_image
