# saspro/mosaic_master.py
"""
Mosaic Master + WCS/reprojection helpers.

Split out of star_alignment.py. Owns MosaicMasterDialog and its supporting
pieces (preview window, settings dialog, polynomial gradient removal, WCS
header sanitizing/parsing, blind-solve helpers (in-house, then astrometry.net), background
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
import shutil
import gc
import traceback
import warnings
import threading as _threading

# Returned by the blind-solve chain when the user pressed Cancel.
_SOLVE_CANCELLED = object()

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
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie, QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QComboBox, QMessageBox, QCheckBox,
    QFormLayout, QScrollArea, QFileDialog, QProgressDialog, QDialogButtonBox,
    QInputDialog, QApplication,QGroupBox,QSlider, QSpinBox, QDoubleSpinBox,
    QWidget, QSplitter, QGridLayout, QTableWidgetItem
)

from setiastro.saspro import astroalign
from setiastro.saspro.memory_utils import smart_zeros
from setiastro.saspro.legacy.image_manager import load_image, save_image
from setiastro.saspro.blink_comparator_pro import CustomDoubleSpinBox, CustomSpinBox
from setiastro.saspro.abe import (
    _generate_sample_points as abe_generate_sample_points,
)
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.mosaic_master_ui import coverage_polygon
try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

# --- interdependencies that remain in star_alignment.py ---
from setiastro.saspro.star_alignment import qs_int, qs_float, qs_bool, ASTROMETRY_API_URL

# Star refinement aims for this many stars per overlap ("Stars to detect")
MOSAIC_STAR_TARGET = 200

_AA_LOCK = _threading.Lock()

def _aa_find_transform_with_backoff(tgt_gray: np.ndarray, src_gray: np.ndarray):
    """
    Retry astroalign.find_transform() with progressively stricter detection,
    serializing SEP usage via _AA_LOCK; returns (transform_obj, (src_pts, tgt_pts)).
    """
    import sep   # was never imported here: the pixel-stack raises below silently did nothing
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
        #sasprojectWriter._serialize_header_any wraps headers for JSON. Unwrap
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
        # Percentiles over real data only. A mosaic preview can be a quarter
        # empty; those zeros would own the low percentile and squash the ramp.
        valid = _valid_data_mask(arr, erode_px=0)       # always (H, W)
        sel = arr[valid]                                # (N,) mono or (N, C) colour
        if sel.size == 0:
            sel = arr.ravel()
        mn, mx = np.percentile(sel, (0.5, 99.5))
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

        self.starTargetSpin = self._ispin(
            20, 2000, qs_int(self.settings, "mosaic/star_target", MOSAIC_STAR_TARGET), 10)
        self.starTargetSpin.setToolTip(
            "Star refinement detects stars in each overlap at the highest threshold that\n"
            "finds at least this many (100σ, 50σ, 25σ … 3σ), the same threshold on both\n"
            "sides of the overlap. More stars: more matches, slower on rich fields.")
        star_form.addRow("Stars to detect:", self.starTargetSpin)

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
        self.starTargetSpin.setValue(MOSAIC_STAR_TARGET)
        self.polyDegreeSpin.setValue(3)

    def accept(self):
        s = self.settings
        s.setValue("mosaic/blend_sharp", self.blendSharpSlider.value() / 10.0)
        s.setValue("mosaic/star_target", int(self.starTargetSpin.value()))
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
        # Real-data mask for this frame. Sampling the background inside a
        # no-data region would anchor the polynomial fit to zeros and tilt the
        # fitted gradient across the whole frame. Combine with the caller's
        # mask when one is given (both use True = keep).
        self._valid = _valid_data_mask(self.image, erode_px=2)
        keep = self._valid
        if user_exclusion_mask is not None:
            keep = keep & (np.asarray(user_exclusion_mask).astype(bool))

        # 1) Stretch
        stretched = self.pixel_math_stretch(self.image)

        # 2) Downsample
        small_stretched = self.downsample_image(stretched, self.downsample_scale)
        h_s, w_s = small_stretched.shape[:2]

        # --- downsample the keep mask (≥0.5 keeps) ---
        mask_small = None
        if keep is not None and not keep.all():
            m = keep.astype(np.float32)
            mask_small = cv2.resize(m, (w_s, h_s), interpolation=cv2.INTER_AREA) >= 0.5
            if not mask_small.any():
                mask_small = None

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

        # Re-normalize median to original (real data only)
        original_median = float(np.median(stretched[self._valid])
                                if self._valid.any() else np.median(stretched))
        after_poly = self.normalize_image(after_poly, original_median,
                                          valid=self._valid)

        # Clip
        after_poly = np.clip(after_poly, 0, 1)

        # 6) Unstretch
        corrected = self.unstretch_image(after_poly)

        # No-data stays no-data: the subtraction above would otherwise leave a
        # non-zero residual there and the region would stop reading as a hole.
        if not self._valid.all():
            corrected[~self._valid, ...] = 0.0
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

        # Stats over real data only: a no-data region would set every channel
        # minimum to 0 and halve every median, so the stretch would be fitted
        # to the holes rather than to the sky.
        vmask = getattr(self, "_valid", None)
        if vmask is None or np.asarray(vmask).shape != (H, W):
            vmask = _valid_data_mask(image_3ch, erode_px=0)
        if not vmask.any():
            vmask = np.ones((H, W), bool)

        # 2) Subtract min per channel
        for c in range(C):
            cmin = float(image_3ch[..., c][vmask].min())
            image_3ch[..., c] -= cmin
            self.stretch_original_mins.append(float(cmin))

        # 3) Compute median after min subtraction
        medians_after_sub = []
        for c in range(C):
            cmed = float(np.median(image_3ch[..., c][vmask]))
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
    def normalize_image(self, image: np.ndarray, target_median: float,
                        valid: np.ndarray = None) -> np.ndarray:
        """
        Shift image so its median matches target_median.

        The median is taken over real data only -- no-data zeros would pull it
        down and the compensating shift would lift the whole frame.
        """
        if valid is None:
            valid = _valid_data_mask(image, erode_px=0)
        cmed = float(np.median(image[valid]) if np.any(valid) else np.median(image))
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


# No-data thinner than this (2*r+1 px) is a bad pixel or a bad line, not a
# region: an exact-zero pixel a PixInsight master carries, a dead column.
NODATA_SPECK_RADIUS = 2


def _nodata_regions(nodata, speck_radius: int = NODATA_SPECK_RADIUS) -> np.ndarray:
    """The no-data REGIONS of a no-data mask (bool): a frame's border, the
    empty part of a panel that is itself a mosaic. A morphological opening
    drops the specks and thin lines -- isolated bad pixels -- which are
    no-data at their own pixels but have no edge to trim."""
    nd = np.asarray(nodata, bool)
    if speck_radius <= 0 or not nd.any():
        return nd
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * speck_radius + 1,) * 2)
    r = 2 * speck_radius + 2
    kr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    # Beyond the array counts as no-data here: a slightly rotated frame in
    # its box leaves thin wedges of no-data along the array's edges, and those
    # are the frame's edge -- the opening must not take them for specks.
    pad = r + speck_radius + 1
    ndp = cv2.copyMakeBorder(nd.astype(np.uint8), pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=1)
    opened = cv2.morphologyEx(ndp, cv2.MORPH_OPEN, k)
    # The opening also shaves the thin tip where two no-data edges meet at a
    # sharp angle; take back the no-data right beside a region. Not whole
    # connected pieces: a dead column runs to the frame edge, joins the
    # no-data outside the frame once warped, and would grow into a stripe.
    region = ndp.astype(bool) & cv2.dilate(opened, kr).astype(bool)
    return region[pad:-pad, pad:-pad]


def _shrink_valid(valid, px: int) -> np.ndarray:
    """`valid` minus a px-wide band around its no-data regions (the ramp a
    resample leaves at an edge). Isolated bad pixels stay excluded but are
    NOT grown: eroding the whole mask turned every single bad pixel into a
    hole 2*px+1 wide, and where only one panel covered the sky those holes
    were black dots in the mosaic (GitHub issue #138; a 6 px erosion here and
    the 12 px coverage trim after the warp made them ~37 px across)."""
    v = np.asarray(valid, bool)
    if px <= 0 or v.all() or not v.any():
        return v
    region = _nodata_regions(~v)
    if not region.any():
        return v
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)
    return v & ~cv2.dilate(region.astype(np.uint8), k).astype(bool)


def _valid_data_mask(img, *, erode_px: int = 6) -> np.ndarray:
    """True where the panel actually carries data; False on no-data fill.

    A panel that is itself a mosaic is NOT a filled rectangle. The final blend
    writes exact 0.0 wherever weight_mosaic == 0, so an L-shaped 3-panel mosaic
    in a 2x2 bounding box arrives here with ~25% of its area as hard zero.
    Those pixels are *no data*, but every statistic in this module used to read
    them as very dark sky: estimate_background_level ranked them as the darkest
    tiles and returned a sky level of 0, the pedestal trick could not see them
    (they sit at exactly PED after the +PED shift), and the overlap fit treated
    the resulting constant as signal and collapsed the gain. One mask, used
    everywhere, is the fix.

    Zero is the no-data sentinel throughout SASpro's mosaic path -- warps use
    borderValue=0, _strip_pedestal zeroes outside coverage, _polish_reprojected
    zeroes the frame border. A real sky pixel is never exactly 0 in practice
    (it carries an offset and read noise), and a handful of genuinely-zero sky
    pixels dropped from a statistic costs nothing.

    For colour, a pixel counts as data if ANY channel is non-zero: a saturated
    or clipped channel must not knock out an otherwise valid pixel.

    The erosion removes the interpolation ramp along a footprint edge, where
    pixels are non-zero but are a blend of real data and fill. It is skipped
    when the frame is entirely valid, so an ordinary full-frame panel is not
    shaved, and when it would erase the mask outright.
    """
    a = _as_hwc(img)
    m = np.isfinite(a).all(axis=2) & (a != 0).any(axis=2)
    if erode_px > 0 and m.any() and not m.all():
        # only the no-data REGIONS are trimmed; an isolated zero pixel stays
        # a single excluded pixel (_shrink_valid, issue #138)
        eroded = _shrink_valid(m, erode_px)
        if eroded.any():
            m = eroded
    return m


def _valid_frac(mask) -> float:
    """Fraction of the frame that carries data, for logging."""
    try:
        return float(np.count_nonzero(mask)) / float(max(1, np.asarray(mask).size))
    except Exception:
        return 1.0


def _masked_median(img, valid=None):
    """Per-channel median over valid pixels only. Returns shape (C,)."""
    a = _as_hwc(img)
    if valid is None:
        valid = _valid_data_mask(a, erode_px=0)
    if not np.any(valid):
        return np.zeros(a.shape[2], np.float32)
    sel = a[valid]                                   # (N, C)
    return np.median(sel, axis=0).astype(np.float32)


def _panel_mad(img, *, mono=None, valid=None) -> float:
    """Raw median absolute deviation of a panel, computed identically to the
    Histogram statistics dialog so the two surfaces report the same number.

    Mirrors histogram.py's real-statistics MAD exactly: per channel, raw
    ``median(|x - median(x)|)`` with no 1.4826 scaling and no black-point
    normalization. (The old black-point subtraction was a no-op -- MAD is
    translation-invariant, so subtracting a pedestal never changed the value.)
    Large channels are subsampled with the same seeded RNG and 5M cap the
    Histogram uses, so the numbers agree bit-for-bit on clean data. Non-finite
    pixels are dropped per channel; that only diverges from the Histogram where
    the Histogram itself would return NaN, and it keeps the linear/stretched
    discriminator from silently collapsing to NaN on panels with masked borders.

    Linear astronomical data is sky-dominated and pathologically tight, so its
    MAD is tiny (order 1e-3); a display stretch lifts and spreads the midtones,
    raising the MAD by one to two orders of magnitude. That gap is what the
    caller uses to tell linear panels from stretched ones.

    For a colour panel each channel is measured exactly as the Histogram
    measures it and the per-channel MADs are averaged into the single scalar the
    caller needs (swap ``np.mean`` for ``np.max`` below to track the widest
    channel instead).

    No-data pixels are excluded via `valid` (computed from the panel when not
    supplied). A panel that is itself a mosaic carries a large block of exact
    zeros; counting those would drag the median down and inflate the MAD, and
    the caller uses this number to decide whether panels are on the same tone
    scale -- a decision that then prompts the user.
    """
    def _hist_mad_1d(chan, m=None) -> float:
        # One channel, identical to histogram.py's real-statistics MAD path.
        chan = np.asarray(chan, dtype=np.float32)
        flat = chan[m].ravel() if m is not None else chan.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return 0.0
        if flat.size > 20_000_000:
            idx = np.random.default_rng(0).choice(
                flat.size, size=5_000_000, replace=False)
            flat = flat[idx]
        cmed = float(np.median(flat))
        return float(np.median(np.abs(flat - cmed)))

    if mono is not None:
        if valid is None:
            valid = _valid_data_mask(mono, erode_px=0)
        return _hist_mad_1d(mono, valid)

    a = np.asarray(img)
    if valid is None:
        valid = _valid_data_mask(a, erode_px=0)
    if a.ndim == 2:
        return _hist_mad_1d(a, valid)

    mads = [_hist_mad_1d(a[..., c], valid) for c in range(a.shape[2])]
    return float(np.mean(mads)) if mads else 0.0

def estimate_background_level(img, *, tiles=16, keep_frac=0.15, valid=None,
                              min_tile_valid=0.60, clip_sigma=2.5, clip_iters=3):
    """
    Robust per-channel sky level for one panel.

    Ranks tiles by luminance median, keeps the darkest `keep_frac`, then
    sigma-clips the pixels in those tiles (hard on the high side to kill
    stars/nebulosity, loose on the low side) and takes the per-channel median.

    Tile *selection* is done on luminance so all channels are measured over the
    same sky region -- that preserves the panel's background color.

    `valid` defaults to the panel's real-data mask, NOT to "every finite pixel".
    That default is load-bearing: this routine ranks tiles by darkness and keeps
    the darkest 15%, so on a panel that is itself a mosaic every selected tile
    would be no-data fill and the reported sky level would be exactly 0 -- which
    then propagates into the pedestal match, the coverage test and the gain fit.
    Pass `valid` explicitly when the caller already has the mask.

    Returns (levels, sigmas), each shape (C,).
    """
    a = _as_hwc(img)
    H, W, C = a.shape
    finite = np.isfinite(a).all(axis=2)
    if valid is None:
        valid = _valid_data_mask(a) & finite
    else:
        valid = valid.astype(bool) & finite
    if not valid.any():
        print("[bg] no valid pixels in panel; sky level unmeasurable")
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

    # Belt and braces: whatever the caller believes about the overlap, a block
    # is only usable where BOTH sides carry real data. A no-data pixel on either
    # side is a constant, and constants are exactly what makes the OLS slope
    # below meaningless.
    mask = (np.asarray(mask).astype(bool)
            & np.isfinite(s).all(axis=2) & np.isfinite(r).all(axis=2)
            & (s != 0).any(axis=2) & (r != 0).any(axis=2))

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
                       block=16, iters=4, clip=2.5, gain_limits=(0.5, 2.0),
                       verbose=True):
    """
    Robust per-channel solve of  ref ~= a*(src - p) + p + b  over `mask`,
    where p is the pivot (the common sky level).

    Pivoting about the sky decorrelates a and b: after the stage-1 pedestal
    match the residual between panels is purely multiplicative on signal
    *above* sky, so that is the quantity the gain should scale.

    Two guards keep a failed fit from being applied as if it had succeeded:

    * If the source side of the overlap carries essentially no spread -- the
      signature of an overlap that is really no-data fill, or sky and nothing
      else -- the gain is not identifiable and we fall back to offset-only.
    * A gain outside `gain_limits` is REJECTED, not clamped. Panels off one rig,
      after a pedestal match, agree to within a few percent; anything past 2x is
      a broken fit, and clamping it to the limit still applies a wrong number.
      (The old behaviour clamped 0.005 -> 0.05 and then multiplied an entire
      mosaic by it, crushing the whole frame onto the sky level.)

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

    def _robust_sd(v):
        if v.size == 0:
            return 0.0
        m = float(np.median(v))
        return 1.4826 * float(np.median(np.abs(v - m)))

    for c in range(C):
        x = S[:, c] - p[c]
        y = R[:, c] - p[c]
        good = np.isfinite(x) & np.isfinite(y)
        aa, bb = 1.0, 0.0

        # Is the gain even identifiable? A degenerate overlap -- one whose
        # source side is a constant, which is what no-data fill lifted to sky
        # level looks like -- gives an OLS slope that is noise divided by noise.
        sx, sy = _robust_sd(x[good]), _robust_sd(y[good])
        fit_c = bool(fit_scale)
        if fit_c and not (np.isfinite(sx) and sx > 0.0 and sx >= 0.1 * max(sy, 0.0)):
            fit_c = False
            if verbose:
                print(f"[match] ch{c}: overlap carries no usable signal "
                      f"(src spread {sx:.3g} vs ref {sy:.3g}) -- offset only")

        for _ in range(iters):
            n = int(good.sum())
            if n < 16:
                break
            if fit_c:
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

        # Out-of-range gain => the fit failed. Drop it and keep the offset,
        # which is still well determined, instead of applying a clamped value.
        lo, hi = gain_limits
        if fit_c and not (lo <= aa <= hi):
            if verbose:
                print(f"[match] ch{c}: gain {aa:.3f} outside [{lo:g}, {hi:g}] "
                      f"-- fit rejected, offset only")
            aa = 1.0
            bb = (float(np.median(y[good] - x[good]))
                  if int(good.sum()) >= 16 else 0.0)

        a_out[c] = float(aa)
        b_out[c] = float(bb)

    return a_out, b_out, n_blocks

# ─────────────────────────────────────────────────────────────────────────────
# Advanced panel matching: one joint solve over every overlap
#
# The default match fits each panel against the mosaic built so far, in panel
# order, so its errors chain and depend on that order. This solves every panel
# together from every pairwise overlap, the way surface_mosaic equalizes tiles,
# with the deep-sky model: per panel and channel
#
#     out = (x - sky) * g + sky + b0 + b1*u + b2*v
#
# a gain about the common sky level (as solve_linear_match) plus an ADDITIVE
# background plane in panel-centred coordinates u, v in [-1, 1]. Additive,
# because deep-sky sky glow is additive and differs between nights and
# altitudes (unlike solar/lunar work, where the sky is black and only a gain is
# allowed). The reference panel is held fixed, which pins the solution.
#
# Guards, in the spirit of "less is more":
#   * block means (noise averages down, structure doesn't); blocks touching a
#     saturated star in either panel are dropped
#   * a pair whose overlap is sky only can't tell a gain from noise: it
#     constrains offsets and planes only (gain held at 1 for that pair)
#   * each block's noise is sky noise plus 2% of its own signal, for weights
#     and clipping alike: bright star/nebula blocks carry the gain, but no
#     single one dominates and none is thrown out for a 1% difference
#   * a slope is fitted only along an axis where the panel's overlaps bracket
#     its centre; with an overlap on one side only it would be read off the
#     curvature of one thin strip and extrapolated across the panel
#   * weak priors (gain toward 1, slopes toward 0) for what the overlaps
#     leave undetermined
#   * iterative per-pair MAD clipping of the residuals
#   * a gain outside 0.5-2 is REJECTED (held at 1 and re-solved), not clamped
# ─────────────────────────────────────────────────────────────────────────────
_ADV_BLOCK = 16
_ADV_MAX_BLOCKS = 20000
_ADV_SAT = 0.95          # block rejected if any channel tops this x that channel's max
_ADV_GAIN_PRIOR = 20.0   # gain toward 1, worth this many blocks
_ADV_SLOPE_PRIOR = 3.0   # plane slopes toward 0, worth this many blocks
_ADV_REL_SCATTER = 0.02  # blocks also scatter by this fraction of their signal
                         # (PSF, registration, shot noise), not just sky noise
_ADV_GAIN_SE = 0.02      # a pair informs a gain only if it pins it this well


def _adv_load(path, mmap=True):
    return np.load(path, mmap_mode="r" if mmap else None)


def _adv_pair_blocks(ri, rj):
    """Block means of the overlap of two stored panels, where both cover the
    block. Returns (Si, Sj, X, Y) -- Si/Sj (N, C), X/Y (N,) canvas block
    centres -- or None when the overlap is too small."""
    y0 = max(ri["bbox"][0], rj["bbox"][0]); y1 = min(ri["bbox"][1], rj["bbox"][1])
    x0 = max(ri["bbox"][2], rj["bbox"][2]); x1 = min(ri["bbox"][3], rj["bbox"][3])
    if y1 - y0 < 2 * _ADV_BLOCK or x1 - x0 < 2 * _ADV_BLOCK:
        return None

    def crop(r):
        oy, ox = r["bbox"][0], r["bbox"][2]
        im = _adv_load(r["img_path"]); cv = _adv_load(r["cov_path"])
        a = np.array(im[y0 - oy:y1 - oy, x0 - ox:x1 - ox], dtype=np.float32)
        m = np.array(cv[y0 - oy:y1 - oy, x0 - ox:x1 - ox], dtype=bool)
        del im, cv
        return a, m

    a, ma_ = crop(ri)
    b, mb = crop(rj)
    m = ma_ & mb
    n_px = int(m.sum())
    if n_px < 8 * _ADV_BLOCK * _ADV_BLOCK:
        return None
    block = max(_ADV_BLOCK, int(math.ceil(math.sqrt(n_px / float(_ADV_MAX_BLOCKS)))))
    H, W = m.shape
    bh, bw = H // block, W // block
    if bh < 1 or bw < 1:
        return None
    hh, ww = bh * block, bw * block
    mf = m[:hh, :ww].astype(np.float32)
    cnt = mf.reshape(bh, block, bw, block).sum(axis=(1, 3))
    ok = cnt >= 0.9 * block * block
    # a star clipped in ANY channel of either panel reads low there and would
    # bias that channel's gain: drop the block (per channel, not luminance)
    C = a.shape[2]
    sat_i = np.resize(np.asarray(ri["sat"], np.float64).ravel(), C)
    sat_j = np.resize(np.asarray(rj["sat"], np.float64).ravel(), C)
    for c in range(C):
        ca = np.where(m, a[..., c], 0.0)[:hh, :ww]
        cb = np.where(m, b[..., c], 0.0)[:hh, :ww]
        ok &= ca.reshape(bh, block, bw, block).max(axis=(1, 3)) < sat_i[c]
        ok &= cb.reshape(bh, block, bw, block).max(axis=(1, 3)) < sat_j[c]
    n = int(ok.sum())
    if n < 32:
        return None
    Si = np.empty((n, C), np.float64)
    Sj = np.empty((n, C), np.float64)
    for c in range(C):
        sa = (a[:hh, :ww, c] * mf).reshape(bh, block, bw, block).sum(axis=(1, 3))
        sb = (b[:hh, :ww, c] * mf).reshape(bh, block, bw, block).sum(axis=(1, 3))
        Si[:, c] = sa[ok] / cnt[ok]
        Sj[:, c] = sb[ok] / cnt[ok]
    by, bx = np.nonzero(ok)
    X = x0 + bx * block + 0.5 * block
    Y = y0 + by * block + 0.5 * block
    return Si, Sj, X, Y


def _adv_uv(rec, X, Y):
    return (X - rec["cx"]) / rec["hx"], (Y - rec["cy"]) / rec["hy"]


def _adv_mad(v):
    if v.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def _adv_solve_channel(pairs, c, pc, n, ref, iters=4, clip=3.0, gain_limits=(0.5, 2.0)):
    """Joint least squares for one channel. Returns (n, 4): g, b0, b1, b2."""
    free = [k for k in range(n) if k != ref]
    col = {k: 4 * t for t, k in enumerate(free)}
    NP = 4 * len(free)
    out = np.zeros((n, 4)); out[:, 0] = 1.0
    if NP == 0:
        return out, []

    k2 = _ADV_REL_SCATTER ** 2

    def floor_sigma(d, x2, keep):
        # sky-noise floor of a pair: residual scale over its fainter half
        # (a robust scale over every block would be set by the sky alone,
        # and bright blocks would all look like outliers)
        kk = keep & (x2 <= np.median(x2[keep]))
        v = d[kk] if int(kk.sum()) >= 16 else d[keep]
        return max(_adv_mad(v), 1e-12)

    rows = []
    for pr in pairs:
        xi = pr["Si"][:, c] - pc
        xj = pr["Sj"][:, c] - pc
        x2 = 0.5 * (xi * xi + xj * xj)
        keep = np.ones(xi.size, bool)
        sig = floor_sigma(xi - xj, x2, keep)
        # Is the gain identifiable from this overlap? Its standard error from
        # the blocks' signal against their noise; a sky-only overlap gives a
        # large one. (Not a robust spread of the signal: in a mostly-sky
        # overlap that sees only the sky and misses the stars and nebula
        # that carry the information.)
        info_g = float(np.sum(xi * xi / (sig * sig + k2 * x2)))
        informative = info_g > 0 and 1.0 / math.sqrt(info_g) < _ADV_GAIN_SE
        rows.append({"pr": pr, "xi": xi, "xj": xj, "x2": x2, "inf": bool(informative),
                     "keep": keep, "sig": sig})

    # A plane slope is fitted only along an axis where this panel's overlaps
    # BRACKET its centre (data on both sides). With an overlap on one side
    # only, the slope would be read off the curvature inside one thin strip
    # and extrapolated across the whole panel; it stays flat instead.
    side_u = {k: [0, 0] for k in free}
    side_v = {k: [0, 0] for k in free}
    for pr in pairs:
        for side, uu, vv in ((pr["i"], pr["ui"], pr["vi"]), (pr["j"], pr["uj"], pr["vj"])):
            if side == ref:
                continue
            side_u[side][0] += int(np.count_nonzero(uu < -0.33))
            side_u[side][1] += int(np.count_nonzero(uu > 0.33))
            side_v[side][0] += int(np.count_nonzero(vv < -0.33))
            side_v[side][1] += int(np.count_nonzero(vv > 0.33))
    free_u = {k: min(side_u[k]) >= 32 for k in free}
    free_v = {k: min(side_v[k]) >= 32 for k in free}

    fixed_gain = set()
    theta = np.zeros(NP)
    for _attempt in range(4):
        for _it in range(iters):
            A = np.zeros((NP, NP)); rhs = np.zeros(NP)
            info = {k: [0.0, 0.0, 0] for k in free}      # sum w, sum w x^2, rows
            for rw in rows:
                pr = rw["pr"]
                k = rw["keep"]
                nk = int(k.sum())
                if nk < 16:
                    continue
                # per block: sky noise plus a share of its own signal
                w = min(1.0, 2000.0 / nk) / (rw["sig"] ** 2 + k2 * rw["x2"][k])
                y = np.zeros(nk)
                cols, L = [], []
                for side, sgn, x, uu, vv in ((pr["i"], 1.0, rw["xi"][k], pr["ui"][k], pr["vi"][k]),
                                           (pr["j"], -1.0, rw["xj"][k], pr["uj"][k], pr["vj"][k])):
                    if side == ref:
                        y -= sgn * x                       # g = 1, no plane
                        continue
                    if rw["inf"] and side not in fixed_gain:
                        cols.append(col[side]); L.append(sgn * x)
                    else:
                        y -= sgn * x                       # gain held at 1 here
                    cols += [col[side] + 1, col[side] + 2, col[side] + 3]
                    L += [sgn * np.ones(nk), sgn * uu, sgn * vv]
                    info[side][0] += float(np.sum(w))
                    info[side][1] += float(np.sum(w * x * x))
                    info[side][2] += nk
                if not cols:
                    continue
                Lm = np.column_stack(L)
                ci = np.asarray(cols)
                A[np.ix_(ci, ci)] += Lm.T @ (w[:, None] * Lm)
                rhs[ci] += Lm.T @ (w * y)
            for k in free:
                sw, swx2, nr = info[k]
                g0 = col[k]
                if nr == 0:                                # not connected: identity
                    for t in range(4):
                        A[g0 + t, g0 + t] += 1.0
                    rhs[g0] += 1.0
                    continue
                mw = sw / nr
                lam_g = max(_ADV_GAIN_PRIOR * swx2 / nr, 1e-6 * mw)
                if k in fixed_gain:
                    lam_g = 1e9 * (lam_g + mw)
                A[g0, g0] += lam_g; rhs[g0] += lam_g * 1.0
                A[g0 + 1, g0 + 1] += 1e-3 * mw
                A[g0 + 2, g0 + 2] += (_ADV_SLOPE_PRIOR if free_u[k] else 1e9) * mw
                A[g0 + 3, g0 + 3] += (_ADV_SLOPE_PRIOR if free_v[k] else 1e9) * mw
            try:
                theta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, rhs, rcond=None)[0]

            # residuals -> per-pair scale and clipping
            def val(side, x, uu, vv, inf):
                if side == ref:
                    return x
                o = theta[col[side]:col[side] + 4]
                g = o[0] if (inf and side not in fixed_gain) else 1.0
                return g * x + o[1] + o[2] * uu + o[3] * vv
            for rw in rows:
                pr = rw["pr"]
                d = (val(pr["i"], rw["xi"], pr["ui"], pr["vi"], rw["inf"])
                     - val(pr["j"], rw["xj"], pr["uj"], pr["vj"], rw["inf"]))
                kk = rw["keep"]
                if int(kk.sum()) < 16:
                    continue
                med = float(np.median(d[kk]))
                s = floor_sigma(d - med, rw["x2"], kk)
                if not (np.isfinite(s) and s > 0):
                    continue
                rw["sig"] = s
                nxt = np.abs(d - med) < clip * np.sqrt(s * s + k2 * rw["x2"])
                if int(nxt.sum()) >= 16:
                    rw["keep"] = nxt

        lo, hi = gain_limits
        bad = [k for k in free if k not in fixed_gain
               and not (lo <= float(theta[col[k]]) <= hi)]
        if not bad:
            break
        fixed_gain |= set(bad)

    for k in free:
        out[k] = theta[col[k]:col[k] + 4]
        if k in fixed_gain or not np.all(np.isfinite(out[k])):
            out[k, 0] = 1.0
        if not np.all(np.isfinite(out[k])):
            out[k] = (1.0, 0.0, 0.0, 0.0)
    return out, sorted(fixed_gain)


def solve_global_panel_match(records, pivot, *, ref=0, log=print):
    """Gain + background plane for every stored panel from all pairwise
    overlaps, solved jointly per channel. Returns (params (n, C, 4), n_pairs);
    params[k, c] = (g, b0, b1, b2) for out = (x - sky)*g + sky + b0 + b1*u + b2*v."""
    n = len(records)
    C = int(records[0]["C"])
    p = np.resize(np.asarray(pivot, np.float64).ravel(), C)
    params = np.zeros((n, C, 4)); params[..., 0] = 1.0

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = _adv_pair_blocks(records[i], records[j])
            if r is None:
                continue
            Si, Sj, X, Y = r
            ui, vi = _adv_uv(records[i], X, Y)
            uj, vj = _adv_uv(records[j], X, Y)
            pairs.append({"i": i, "j": j, "Si": Si, "Sj": Sj, "X": X, "Y": Y,
                          "ui": ui, "vi": vi, "uj": uj, "vj": vj})
    if not pairs:
        log("[Mosaic+] no usable overlaps between panels; nothing to solve")
        return params, 0, []

    for c in range(C):
        res, fixed = _adv_solve_channel(pairs, c, p[c], n, ref)
        params[:, c, :] = res
        if fixed:
            log(f"[Mosaic+] ch{c}: gain of panel(s) {fixed} outside 0.5-2 -- "
                f"rejected, held at 1")

    # report: overlap mismatch before / after, in the data's units
    seams = []
    for pr in pairs:
        i, j = pr["i"], pr["j"]
        before, after = [], []
        for c in range(C):
            xi = pr["Si"][:, c] - p[c]; xj = pr["Sj"][:, c] - p[c]
            gi, a0, a1, a2 = params[i, c]; gj, c0, c1, c2 = params[j, c]
            ci_ = gi * xi + a0 + a1 * pr["ui"] + a2 * pr["vi"]
            cj_ = gj * xj + c0 + c1 * pr["uj"] + c2 * pr["vj"]
            before.append(float(np.median(np.abs(xi - xj))))
            after.append(float(np.median(np.abs(ci_ - cj_))))
        log(f"[Mosaic+] overlap {records[i]['name']} / {records[j]['name']}: "
            f"median |difference| {np.mean(before):.3g} -> {np.mean(after):.3g} "
            f"({pr['Si'].shape[0]} blocks)")
        try:
            hull = cv2.convexHull(np.column_stack([pr["X"], pr["Y"]]).astype(np.float32))
            hull = hull.reshape(-1, 2).astype(np.float64)
        except Exception:
            hull = None
        seams.append({"i": i, "j": j, "poly": hull, "before": float(np.mean(before)),
                      "after": float(np.mean(after)), "n": int(pr["Si"].shape[0])})
    for k in range(n):
        g = ", ".join(f"{params[k, c, 0]:.4f}" for c in range(C))
        b = ", ".join(f"{params[k, c, 1]:+.3g}" for c in range(C))
        t = max(float(np.max(np.abs(params[k, :, 2]))), float(np.max(np.abs(params[k, :, 3]))))
        log(f"[Mosaic+] {records[k]['name']}: gain [{g}], offset [{b}], "
            f"plane edge-to-centre <= {t:.3g}" + ("  (reference)" if k == ref else ""))
    return params, len(pairs), seams


def apply_panel_match(img, cov, rec, prm, pivot):
    """(x - sky)*g + sky + b0 + b1*u + b2*v on a stored panel crop; 0 outside cov."""
    out = np.array(img, dtype=np.float32, copy=True)
    H, W, C = out.shape
    p = np.resize(np.asarray(pivot, np.float64).ravel(), C)
    y0, _y1, x0, _x1 = rec["bbox"]
    U = ((np.arange(W, dtype=np.float32) + x0 + 0.5 - rec["cx"]) / rec["hx"])[None, :]
    V = ((np.arange(H, dtype=np.float32) + y0 + 0.5 - rec["cy"]) / rec["hy"])[:, None]
    for c in range(C):
        g, b0, b1, b2 = (float(v) for v in prm[c])
        pc = np.float32(p[c])
        out[..., c] = ((out[..., c] - pc) * np.float32(g) + pc + np.float32(b0)
                       + np.float32(b1) * U + np.float32(b2) * V)
    out[~np.asarray(cov, bool)] = 0.0
    return out


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

        self.resize(640, 520)
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
        """Panels on the left, the sky / result map on the right, one Build
        button. The options read by the build are behind the _opt_* methods."""
        from setiastro.saspro.mosaic_master_ui import PanelTable, MosaicMapWidget
        from setiastro.saspro.ser_stacker_dialog import CollapsibleGroup

        self.resize(1200, 780)
        s = self.settings
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal, self)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)

        # ══ left: panels and options ═══════════════════════════════════════
        left = QWidget(self)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)

        # What kind of mosaic: two picture cards, chosen first
        from PyQt6.QtWidgets import QButtonGroup
        from setiastro.saspro.mosaic_master_ui import ModeCardButton
        from setiastro.saspro.resources import mosaic_panels_path, mosaic_robotic_path
        self.panelsModeBtn = ModeCardButton(
            mosaic_panels_path, "Traditional Panels", "A few plate-solved panels", self)
        self.panelsModeBtn.setToolTip(
            "A few finished panels, each placed on the sky by its plate solution\n"
            "(solved when you build if needed), brightness-matched and blended.")
        self.roboticModeBtn = ModeCardButton(
            mosaic_robotic_path, "Robotic Mosaic", "Seestar and smart-telescope frames", self)
        self.roboticModeBtn.setToolTip(
            "Many short frames from a Seestar or other smart telescope (add the sub folder).\n"
            "Every frame is plate-solved against your Gaia catalogue, seeded by the\n"
            "pointing in its header, placed on one canvas by its own solution (so field\n"
            "rotation doesn't matter), and the frames are averaged. Solutions are\n"
            "cached, so a rebuild doesn't solve again.")
        self._sourceGroup = QButtonGroup(self)
        self._sourceGroup.setExclusive(True)
        self._sourceGroup.addButton(self.panelsModeBtn)
        self._sourceGroup.addButton(self.roboticModeBtn)
        (self.roboticModeBtn if str(s.value("mosaic/panel_source", "generic")) == "seestar"
         else self.panelsModeBtn).setChecked(True)
        self._sourceGroup.buttonToggled.connect(
            lambda _b, on: (self._on_source_changed(), self._mark_result_stale()) if on else None)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(self.panelsModeBtn, 1)
        mode_row.addWidget(self.roboticModeBtn, 1)
        lv.addLayout(mode_row)
        lv.addSpacing(4)

        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Panels</b>"))
        head.addStretch(1)
        self.addFilesBtn = QPushButton("+ Files…")
        self.addFilesBtn.setToolTip("Add panel files (FITS, XISF, TIFF, …).\n"
                                    "You can also drop files onto the list.")
        self.addFilesBtn.clicked.connect(self.add_image)
        self.addFolderBtn = QPushButton("+ Folder…")
        self.addFolderBtn.setToolTip("Add every image file in a folder (e.g. a Seestar sub-frame folder).")
        self.addFolderBtn.clicked.connect(self.add_folder)
        self.addViewsBtn = QPushButton("+ Views…")
        self.addViewsBtn.setToolTip("Add images that are open in SASpro.")
        self.addViewsBtn.clicked.connect(self.add_image_from_view)
        self.removeBtn = QPushButton("Remove")
        self.removeBtn.setToolTip("Remove the selected panels (Delete).")
        self.removeBtn.clicked.connect(self.remove_selected)
        for b in (self.addFilesBtn, self.addFolderBtn, self.addViewsBtn, self.removeBtn):
            head.addWidget(b)
        lv.addLayout(head)

        self.panel_table = PanelTable(self)
        self.panel_table.filesDropped.connect(self._add_paths)
        self.panel_table.cellClicked.connect(self._on_panel_cell_clicked)
        self.panel_table.cellDoubleClicked.connect(lambda _r, _c: self.preview_selected())
        self.panel_table.itemSelectionChanged.connect(self._on_table_selection)
        self.panel_table.setToolTip("Double-click a panel to preview it. Click the ★ column "
                                    "to choose the reference panel.")
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.panel_table,
                  activated=self.remove_selected)
        lv.addWidget(self.panel_table, 1)

        # ── main options ───────────────────────────────────────────────────
        og = QGridLayout()
        og.setHorizontalSpacing(8)
        og.setVerticalSpacing(6)


        self.brightnessCombo = QComboBox()
        self.brightnessCombo.addItem("Off", "off")
        self.brightnessCombo.addItem("Standard", "standard")
        self.brightnessCombo.addItem("Advanced — all overlaps at once", "advanced")
        self.brightnessCombo.setToolTip(
            "How panel brightness is matched.\n\n"
            "Standard: each panel's sky level is matched to the reference, then each\n"
            "panel is fitted to the mosaic built so far, one after another.\n\n"
            "Advanced: every panel is matched from all of its overlaps at once, with a\n"
            "gain and a smooth background tilt per panel. Errors don't chain from panel\n"
            "to panel and different nights' gradients line up. Builds twice: slower,\n"
            "and needs temporary disk space about the size of the panels.\n\n"
            "Off: panels are blended as they are.")
        _bright_default = ("advanced" if qs_bool(s, "mosaic/advanced_match", False)
                           else "standard")
        self._set_combo_data(self.brightnessCombo,
                             str(s.value("mosaic/brightness", _bright_default)))
        self.brightnessCombo.currentIndexChanged.connect(
            lambda _i: self._persist("mosaic/brightness", self.brightnessCombo.currentData()))

        self.reprojectModeCombo = QComboBox()
        self.reprojectModeCombo.addItems([
            "Exact — SIP-aware remap (recommended)",
            "Fast — Homography (Global H)",
        ])
        self.reprojectModeCombo.setToolTip(
            "How each panel is warped onto the mosaic's sky grid.\n\n"
            "Exact: per-pixel inverse WCS, honours SIP distortion, Lanczos resampling.\n"
            "Fast: one projective transform per panel; exact only without SIP, and\n"
            "with SIP it can double stars near the edges.")
        _mode = s.value("mosaic/reproject_mode", "Exact — SIP-aware remap (recommended)", type=str)
        if self.reprojectModeCombo.findText(_mode) < 0:
            _mode = "Exact — SIP-aware remap (recommended)"
        self.reprojectModeCombo.setCurrentText(_mode)
        self.reprojectModeCombo.currentTextChanged.connect(self._on_reproject_mode_changed)

        self.starRefineCombo = QComboBox()
        self.starRefineCombo.addItem("Affine (recommended)", "Affine Transform")
        self.starRefineCombo.addItem("Off — WCS placement only", "off")
        self.starRefineCombo.addItem("Partial affine", "Partial Affine Transform")
        self.starRefineCombo.addItem("Homography", "Homography Transform")
        self.starRefineCombo.addItem("Polynomial warp", "Polynomial Warp Based Transform")
        self.starRefineCombo.setToolTip(
            "After a panel is placed by its plate solution, its stars can be matched\n"
            "to the mosaic in the overlap for a final correction.\n\n"
            "Panels with SIP distortion are always placed by WCS alone: their\n"
            "plate solution is more accurate than one global star fit.\n"
            "Off: every panel by WCS alone (use when overlaps are small).")
        _refine_default = "off" if qs_bool(s, "mosaic/wcs_only", False) else "Affine Transform"
        self._set_combo_data(self.starRefineCombo,
                             str(s.value("mosaic/star_refine", _refine_default)))
        self.starRefineCombo.activated.connect(
            lambda _i: setattr(self, "_starrefine_user_touched", True))
        self.starRefineCombo.currentIndexChanged.connect(
            lambda _i: self._persist("mosaic/star_refine", self.starRefineCombo.currentData()))

        self.blendSharpSlider = QSlider(Qt.Orientation.Horizontal)
        self.blendSharpSlider.setRange(10, 60)          # value / 10 -> 1.0 .. 6.0
        self.blendSharpSlider.setValue(int(round(qs_float(s, "mosaic/blend_sharp", 2.0) * 10)))
        self.blendSharpSlider.setToolTip(
            "How sharply panels hand off across a seam.\n"
            "Soft: wide, gradual blend; hides small brightness differences, but needs\n"
            "good registration or stars can ghost.\n"
            "Sharp: tight crossover near the middle of the overlap; safest when panels\n"
            "disagree slightly, at the cost of a visible step.")
        self.blendSharpValue = QLabel(f"{self.blendSharpSlider.value() / 10:.1f}")
        self.blendSharpValue.setMinimumWidth(26)
        self.blendSharpSlider.valueChanged.connect(self._on_blend_changed)
        seam_row = QHBoxLayout()
        seam_row.addWidget(QLabel("soft"))
        seam_row.addWidget(self.blendSharpSlider, 1)
        seam_row.addWidget(QLabel("sharp"))
        seam_row.addSpacing(6)
        seam_row.addWidget(self.blendSharpValue)

        def _lab(text, tip):
            lb = QLabel(text)
            lb.setToolTip(tip)
            return lb
        og.addWidget(_lab("Brightness", self.brightnessCombo.toolTip()), 0, 0)
        og.addWidget(self.brightnessCombo, 0, 1)
        og.addWidget(_lab("Seams", self.blendSharpSlider.toolTip()), 0, 2)
        og.addLayout(seam_row, 0, 3)
        self.reprojectModeLabel = _lab("Placement", self.reprojectModeCombo.toolTip())
        og.addWidget(self.reprojectModeLabel, 1, 0)
        og.addWidget(self.reprojectModeCombo, 1, 1)
        self.starRefineLabel = _lab("Star refine", self.starRefineCombo.toolTip())
        og.addWidget(self.starRefineLabel, 1, 2)
        og.addWidget(self.starRefineCombo, 1, 3)
        self.seestarRejectCheck = QCheckBox("Reject satellites and other outliers")
        self.seestarRejectCheck.setToolTip(
            "Keeps each spot's three brightest and one faintest samples aside while the\n"
            "frames are stacked, then tests them against the other frames at that spot\n"
            "and takes out the ones that don't belong (satellite and aircraft trails,\n"
            "hot pixels, cosmic rays). Ordinary samples are never trimmed. Needs at least\n"
            "4 frames over a spot; costs almost no extra time.")
        self.seestarRejectCheck.setChecked(qs_bool(s, "mosaic/seestar_reject", True))
        self.seestarRejectCheck.toggled.connect(
            lambda on: self._persist("mosaic/seestar_reject", bool(on)))
        og.addWidget(self.seestarRejectCheck, 2, 0, 1, 4)
        og.setColumnStretch(1, 1)
        og.setColumnStretch(3, 1)
        lv.addLayout(og)

        self.optHint = QLabel("")
        self.optHint.setWordWrap(True)
        self.optHint.setStyleSheet("color: #9aa0aa;")
        lv.addWidget(self.optHint)

        # ── Preparation (collapsed) ────────────────────────────────────────
        self._prepGroup = CollapsibleGroup("Preparation — crop, colour, plate solving",
                                           self, collapsed=True)
        pf = self._prepGroup.content_layout()

        self.cropEachCheckBox = QCheckBox("Crop every panel")
        self.cropEachCheckBox.setToolTip(
            "Cut the same pixel margins off every panel before anything else.\n"
            "An existing plate solution is shifted with the crop, so it stays valid.")
        self.cropEachCheckBox.setChecked(qs_bool(s, "mosaic/crop_each", False))

        def _margin_spin(key, tip):
            sb = QSpinBox()
            sb.setRange(0, 100000)
            sb.setSuffix(" px")
            sb.setValue(qs_int(s, key, 0))
            sb.setKeyboardTracking(False)
            sb.setMaximumWidth(90)
            sb.setToolTip(tip)
            return sb
        self.cropLeftSpin = _margin_spin("mosaic/crop_left", "Pixels removed from the left edge of every panel.")
        self.cropRightSpin = _margin_spin("mosaic/crop_right", "Pixels removed from the right edge of every panel.")
        self.cropTopSpin = _margin_spin("mosaic/crop_top", "Pixels removed from the top edge of every panel.")
        self.cropBottomSpin = _margin_spin("mosaic/crop_bottom", "Pixels removed from the bottom edge of every panel.")
        crop_w = QWidget()
        cr = QHBoxLayout(crop_w)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.addWidget(self.cropEachCheckBox)
        for lab, sp in (("L", self.cropLeftSpin), ("R", self.cropRightSpin),
                        ("T", self.cropTopSpin), ("B", self.cropBottomSpin)):
            cr.addWidget(QLabel(lab))
            cr.addWidget(sp)
        cr.addStretch(1)
        pf.addRow(crop_w)

        def _persist_crop():
            self._persist("mosaic/crop_each", self.cropEachCheckBox.isChecked())
            for key, sp in (("mosaic/crop_left", self.cropLeftSpin), ("mosaic/crop_right", self.cropRightSpin),
                            ("mosaic/crop_top", self.cropTopSpin), ("mosaic/crop_bottom", self.cropBottomSpin)):
                self._persist(key, int(sp.value()))

        def _sync_crop(on):
            for sp in (self.cropLeftSpin, self.cropRightSpin, self.cropTopSpin, self.cropBottomSpin):
                sp.setEnabled(bool(on))
            _persist_crop()
        self.cropEachCheckBox.toggled.connect(_sync_crop)
        for sp in (self.cropLeftSpin, self.cropRightSpin, self.cropTopSpin, self.cropBottomSpin):
            sp.valueChanged.connect(lambda _v: _persist_crop())
        _sync_crop(self.cropEachCheckBox.isChecked())

        self.srgcEachCheckBox = QCheckBox("Survey colour/gradient correction (SRGC)")
        self.srgcEachCheckBox.setToolTip(
            "Survey Reference Gradient Correction on every panel after it is\n"
            "plate-solved. Fetches a public survey cutout for that field and\n"
            "subtracts only the additive gradient the survey does not explain.\n"
            "Needs a network connection and a plate solution on each panel.")
        self.srgcEachCheckBox.setChecked(
            qs_bool(s, "mosaic/srgc_each", qs_bool(s, "mosaic/adbe_each", False)))
        self.srgcSurveyCombo = QComboBox()
        try:
            from setiastro.saspro.survey_gradient import SURVEYS, DEFAULT_SURVEY
            for name in SURVEYS:
                self.srgcSurveyCombo.addItem(name)
            preferred = str(s.value("mosaic/srgc_survey",
                                    s.value("srgc/survey", DEFAULT_SURVEY)) or DEFAULT_SURVEY)
            idx = self.srgcSurveyCombo.findText(preferred)
            self.srgcSurveyCombo.setCurrentIndex(idx if idx >= 0 else 0)
        except Exception as e:
            self.srgcSurveyCombo.addItem("DSS2 colour (red / synth / blue) — all sky")
            print(f"[Mosaic] SRGC survey list unavailable: {e}")
        self.srgcSurveyCombo.setToolTip("Which public survey feeds the reference.\n"
                                        "DSS2 colour is all-sky. NSNS options are northern only.")
        self.srgcSurveyCombo.currentTextChanged.connect(
            lambda name: self._persist("mosaic/srgc_survey", str(name)))

        def _sync_srgc(on):
            self.srgcSurveyCombo.setEnabled(bool(on))
            self._persist("mosaic/srgc_each", bool(on))
        self.srgcEachCheckBox.toggled.connect(_sync_srgc)
        _sync_srgc(self.srgcEachCheckBox.isChecked())
        srgc_w = QWidget()
        sr = QHBoxLayout(srgc_w)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(self.srgcEachCheckBox)
        sr.addWidget(self.srgcSurveyCombo, 1)
        pf.addRow(srgc_w)

        self.forceBlindCheckBox = QCheckBox("Re-solve every panel (ignore existing plate solutions)")
        self.forceBlindCheckBox.setToolTip(
            "Solve every panel again, ignoring a plate solution that is already there.\n"
            "The in-house Gaia blind solver runs first; Astrometry.net only if it fails.")
        self.forceBlindCheckBox.toggled.connect(lambda _on: self._refresh_panels())
        pf.addRow(self.forceBlindCheckBox)
        lv.addWidget(self._prepGroup)

        # ── Star alignment details (collapsed) ─────────────────────────────
        self._starGroup = CollapsibleGroup("Star alignment details", self, collapsed=True)
        sf = self._starGroup.content_layout()
        self.starTargetSpin = QSpinBox(); self.starTargetSpin.setRange(20, 2000)
        self.starTargetSpin.setSingleStep(10)
        self.starTargetSpin.setValue(qs_int(s, "mosaic/star_target", MOSAIC_STAR_TARGET))
        self.starTargetSpin.setToolTip(
            "Star refinement detects stars in each overlap at the highest threshold that\n"
            "finds at least this many (100σ, 50σ, 25σ … 3σ), the same threshold on both\n"
            "sides of the overlap. More stars: more matches, slower on rich fields.")
        self.polyDegreeSpin = QSpinBox(); self.polyDegreeSpin.setRange(1, 6)
        self.polyDegreeSpin.setValue(qs_int(s, "mosaic/poly_degree", 3))
        self.polyDegreeSpin.setToolTip("Degree of the polynomial-warp refinement (Star refine: Polynomial warp).")
        for sp, key in ((self.starTargetSpin, "mosaic/star_target"),
                        (self.polyDegreeSpin, "mosaic/poly_degree")):
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v, k=key: self._persist(k, v))
        sf.addRow("Stars to detect:", self.starTargetSpin)
        sf.addRow("Polynomial degree:", self.polyDegreeSpin)
        self._starGroup.setToolTip("Used only for Traditional Panels refined on stars (Star refine),\n"
                                   "for panels without SIP distortion.")
        lv.addWidget(self._starGroup)

        # ── Output (collapsed) ─────────────────────────────────────────────
        self._outGroup = CollapsibleGroup("Output", self, collapsed=True)
        of = self._outGroup.content_layout()
        self.gaiaSolveCheckBox = QCheckBox("Plate solve the finished mosaic (in-house Gaia DR3)")
        self.gaiaSolveCheckBox.setToolTip(
            "Re-solve the finished mosaic against the local Gaia DR3 catalogue and\n"
            "attach that solution, tied to stars from every panel, instead of the\n"
            "reference panel's. Falls back to the geometric solution if it fails.")
        self.gaiaSolveCheckBox.setChecked(qs_bool(s, "mosaic/gaia_solve_final", True))
        self.gaiaSolveCheckBox.toggled.connect(
            lambda on: self._persist("mosaic/gaia_solve_final", bool(on)))
        of.addRow(self.gaiaSolveCheckBox)
        self.autostretchPreviewCheck = QCheckBox("Auto-stretch the full preview (data stays linear)")
        self.autostretchPreviewCheck.setChecked(qs_bool(s, "mosaic/autostretch_preview", True))
        self.autostretchPreviewCheck.toggled.connect(
            lambda on: self._persist("mosaic/autostretch_preview", bool(on)))
        of.addRow(self.autostretchPreviewCheck)
        lv.addWidget(self._outGroup)
        split.addWidget(left)

        # ══ right: sky map / result ═══════════════════════════════════════
        right = QWidget(self)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        top = QHBoxLayout()
        self.mapTitle = QLabel("<b>Sky map</b>")
        top.addWidget(self.mapTitle)
        top.addStretch(1)
        self.overlayCheck = QCheckBox("Panel map")
        self.overlayCheck.setChecked(True)
        self.overlayCheck.setToolTip("Show each panel's outline, gain and seam quality on the mosaic.")
        self.overlayCheck.hide()
        top.addWidget(self.overlayCheck)
        self.mapToggleBtn = QPushButton("Sky map")
        self.mapToggleBtn.setToolTip("Switch between the finished mosaic and the panels' sky map.")
        self.mapToggleBtn.hide()
        self.mapToggleBtn.clicked.connect(self._toggle_map_mode)
        top.addWidget(self.mapToggleBtn)
        self.fullPreviewBtn = QPushButton("Full Preview…")
        self.fullPreviewBtn.setToolTip("Open the finished mosaic in a zoomable window.")
        self.fullPreviewBtn.clicked.connect(self._open_full_preview)
        self.fullPreviewBtn.hide()
        top.addWidget(self.fullPreviewBtn)
        rv.addLayout(top)

        self.map = MosaicMapWidget(self)
        self.map.panelClicked.connect(self._select_panel_row)
        self.map.resultActivated.connect(self._open_full_preview)
        self.overlayCheck.toggled.connect(self.map.set_overlay_visible)
        rv.addWidget(self.map, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 6)
        split.setSizes([540, 640])

        # ══ bottom bar ════════════════════════════════════════════════════
        bar = QHBoxLayout()
        bar.addWidget(make_help_button("mosaic_master", self))
        self.status_label = QLabel("Add panels to begin.")
        # plate_solver's _set_status_ui looks for a child QLabel with exactly
        # this objectName to route the solver's progress into this window.
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        bar.addWidget(self.status_label, 1)

        self.spinnerLabel = QLabel(self)
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerMovie = QMovie(self.spinner_path) if self.spinner_path else None
        if self.spinnerMovie:
            self.spinnerLabel.setMovie(self.spinnerMovie)
        self.spinnerLabel.hide()
        bar.addWidget(self.spinnerLabel)

        # The next step is the coloured button: Build until there is a mosaic,
        # then Push to New View (green); Build turns plain and says Rebuild, and
        # turns blue again once the panels or the mode change.
        self.openViewBtn = QPushButton("Push to New View  ⇪")
        self.openViewBtn.setMinimumHeight(34)
        self.openViewBtn.setMinimumWidth(170)
        self.openViewBtn.setToolTip("Open the finished mosaic in SASpro as a new image\n"
                                    "(linear, with its plate solution). This window stays open.")
        self.openViewBtn.clicked.connect(self._open_result_as_view)
        bar.addWidget(self.openViewBtn)
        self.buildBtn = QPushButton("Build Mosaic  ▶")
        self.buildBtn.setMinimumHeight(34)
        self.buildBtn.setMinimumWidth(170)
        self.buildBtn.setToolTip("Solve any panel without a plate solution, then place,\n"
                                 "match and blend all panels into the mosaic.")
        self.buildBtn.clicked.connect(self._on_build)
        bar.addWidget(self.buildBtn)
        root.addLayout(bar)

        self._result_display = None
        self._result_stale = False
        self._on_source_changed()
        self._refresh_panels()
        self._update_action_buttons()

    # ── options read by the build ─────────────────────────────────────────
    def _opt_seestar(self) -> bool:
        return getattr(self, "roboticModeBtn", None) is not None and \
            self.roboticModeBtn.isChecked()

    def _source_mode(self) -> str:
        """'seestar' (Robotic Mosaic card) or 'generic' (Traditional Panels)."""
        return "seestar" if self._opt_seestar() else "generic"

    def _set_source_mode(self, mode: str):
        (self.roboticModeBtn if mode == "seestar" else self.panelsModeBtn).setChecked(True)

    def _opt_normalize(self) -> bool:
        return self.brightnessCombo.currentData() != "off"

    def _opt_advanced(self) -> bool:
        return self.brightnessCombo.currentData() == "advanced"

    def _opt_wcs_only(self) -> bool:
        return self.starRefineCombo.currentData() == "off"

    def _opt_transform(self) -> str:
        d = self.starRefineCombo.currentData()
        return "Affine Transform" if d in (None, "off") else str(d)

    # ── small UI helpers ──────────────────────────────────────────────────
    def _persist(self, key, value):
        self.settings.setValue(key, value)
        self.settings.sync()

    @staticmethod
    def _set_combo_data(combo, data):
        i = combo.findData(data)
        combo.setCurrentIndex(i if i >= 0 else 0)

    def _on_blend_changed(self, v):
        self.blendSharpValue.setText(f"{v / 10:.1f}")
        self._persist("mosaic/blend_sharp", v / 10.0)

    def _on_source_changed(self, *_):
        seestar = self._opt_seestar()
        self._persist("mosaic/panel_source", self._source_mode())
        tip = ("Not used for Seestar frames: each is placed by its own plate solution "
               "and the frames are averaged.")
        for w in (self.reprojectModeCombo, self.reprojectModeLabel,
                  self.starRefineCombo, self.starRefineLabel, self.brightnessCombo,
                  self.blendSharpSlider):
            w.setEnabled(not seestar)
        if getattr(self, "seestarRejectCheck", None) is not None:
            self.seestarRejectCheck.setVisible(seestar)
        if getattr(self, "_starGroup", None) is not None:
            self._starGroup.setEnabled(not seestar)
        self.reprojectModeCombo.setToolTip(tip if seestar else self.reprojectModeLabel.toolTip())
        if getattr(self, "panel_table", None) is not None:
            self.panel_table.placeholder = (
                "Drop Seestar / smart-telescope frames here\nor use + Folder for the sub folder\n\n"
                "Every frame is plate-solved and placed on the sky." if seestar else
                "Drop panel files here\nor use + Files / + Views\n\n"
                "Linear, plate-solved panels work best.")
            self.panel_table.viewport().update()
        if getattr(self, "map", None) is not None:
            self.map.empty_text = ("Frames appear here as footprints on the sky,\n"
                                   "from the pointing in their headers." if seestar else None)
            self.map.update()
        if getattr(self, "panel_table", None) is not None:
            self._refresh_panels()

    _BTN_BUILD = ("QPushButton { background: #2f6fd6; color: white; font-weight: bold;"
                  " padding: 6px 20px; border-radius: 4px; }"
                  "QPushButton:hover { background: #3b7ce6; }"
                  "QPushButton:disabled { background: #44484f; color: #9a9a9a; }")
    _BTN_PUSH = ("QPushButton { background: #23934f; color: white; font-weight: bold;"
                 " padding: 6px 20px; border-radius: 4px; }"
                 "QPushButton:hover { background: #2aa95c; }"
                 "QPushButton:disabled { background: #44484f; color: #9a9a9a; }")
    _BTN_PLAIN = "QPushButton { padding: 6px 20px; }"

    def _has_result(self) -> bool:
        return (getattr(self, "final_mosaic", None) is not None
                and getattr(self, "_result_display", None) is not None)

    def _mark_result_stale(self):
        """The panels or the mode changed since the mosaic was built: Build
        is the next step again (the old mosaic can still be pushed)."""
        self._result_stale = True
        self._update_action_buttons()

    def _update_action_buttons(self):
        """Colour the button for the next step: Build Mosaic (blue) until
        there is a mosaic, then Push to New View (green), with Build plain
        and reading Rebuild."""
        if getattr(self, "openViewBtn", None) is None:
            return
        building = bool(getattr(self, "_building", False))
        has = self._has_result()
        self.openViewBtn.setEnabled(has and not building)
        if building:
            return
        push_next = has and not getattr(self, "_result_stale", False)
        self.openViewBtn.setStyleSheet(self._BTN_PUSH if has else self._BTN_PLAIN)
        self.buildBtn.setStyleSheet(self._BTN_PLAIN if push_next else self._BTN_BUILD)
        self.buildBtn.setText("Rebuild Mosaic  ↻" if has else "Build Mosaic  ▶")
        self.openViewBtn.setDefault(push_next)
        self.buildBtn.setDefault(not push_next)

    def _set_building(self, on: bool):
        for w in (self.addFilesBtn, self.addViewsBtn, self.removeBtn, self.panel_table,
                  self.panelsModeBtn, self.roboticModeBtn,
                  self.brightnessCombo, self.reprojectModeCombo,
                  self.starRefineCombo, self.blendSharpSlider, self._prepGroup,
                  self.addFolderBtn, self.seestarRejectCheck,
                  self._starGroup, self._outGroup, self.fullPreviewBtn,
                  self.mapToggleBtn):
            w.setEnabled(not on)
        self.buildBtn.setEnabled(not on)
        if on:
            self.buildBtn.setText("Building…")
            self.openViewBtn.setEnabled(False)
        else:
            # only a build that produced a mosaic (_show_result) clears "stale"
            self._on_source_changed()
            self._update_action_buttons()

    def _on_build(self):
        if getattr(self, "_building", False):
            return
        if not self.loaded_images:
            self.status_label.setText("Add panels first.")
            return
        self._building = True
        self._build_report = None
        self._set_building(True)
        try:
            self.align_images()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Mosaic Master", f"The build failed:\n{e}")
        finally:
            self._building = False
            self._set_building(False)
            self._refresh_panels()

    # ── panel table + sky map ─────────────────────────────────────────────
    def _panel_info(self, item):
        """Thumbnail, size, scale, SIP, sky footprint and data stats for one
        panel, cached until its image or plate solution changes."""
        img = item.get("image")
        w = item.get("wcs")
        key = (id(img), id(w))
        c = item.get("_ui_cache")
        if c is not None and c.get("key") == key:
            return c
        if img is None:
            return self._panel_info_lazy(item, key)
        c = {"key": key, "scale": None, "sip": False, "fp": None}
        a = np.asarray(img)
        H, W = a.shape[:2]
        c["size"] = (W, H)
        step = max(1, int(max(H, W) // 200))
        sm = np.asarray(a[::step, ::step], dtype=np.float32)
        lum = sm if sm.ndim == 2 else sm.mean(axis=2)
        v = lum[np.isfinite(lum) & (lum != 0)]
        c["mad"] = float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0
        try:
            c["valid_px"] = int(np.count_nonzero(_valid_data_mask(a, erode_px=0)))
        except Exception:
            c["valid_px"] = int(W * H * (v.size / max(lum.size, 1)))
        c["valid_frac"] = c["valid_px"] / max(W * H, 1)
        try:                                     # small display thumbnail
            rgb = sm if sm.ndim == 3 else sm[..., None]
            rgb = np.repeat(rgb[..., :1], 3, axis=2) if rgb.shape[2] < 3 else rgb[..., :3]
            if v.size:
                lo, hi = np.percentile(v, (5, 99.7))
                t = np.clip((rgb - lo) / max(hi - lo, 1e-12), 0, 1) ** 0.45
            else:
                t = np.zeros_like(rgb)
            u8 = np.ascontiguousarray((t * 255).astype(np.uint8))
            qi = QImage(u8.data, u8.shape[1], u8.shape[0], 3 * u8.shape[1],
                        QImage.Format.Format_RGB888).copy()
            c["icon"] = QIcon(QPixmap.fromImage(qi).scaled(
                40, 30, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception:
            c["icon"] = None
        if w is not None:
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales
                c["scale"] = float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600.0
            except Exception:
                c["scale"] = None
            c["sip"] = self._wcs_has_sip(w)
            try:
                NS = 16
                xs = np.linspace(0, W - 1, NS); ys = np.linspace(0, H - 1, NS)
                px = np.concatenate([xs, np.full(NS, W - 1), xs[::-1], np.zeros(NS)])
                py = np.concatenate([np.zeros(NS), ys, np.full(NS, H - 1), ys[::-1]])
                ra, dec = w.all_pix2world(px, py, 0)
                g = np.isfinite(ra) & np.isfinite(dec)
                c["fp"] = (ra[g], dec[g]) if g.sum() >= 3 else None
            except Exception:
                c["fp"] = None
        item["_ui_cache"] = c
        return c

    def _panel_info_lazy(self, item, key):
        H, W = item.get("shape") or (0, 0)
        c = {"key": key, "size": (W, H), "mad": 0.0, "valid_px": W * H, "valid_frac": 1.0,
             "icon": None, "scale": None, "sip": False, "fp": None, "approx": False}
        w = item.get("wcs")
        try:
            if w is None:
                # position from the header pointing only: a north-up box, drawn dashed
                from setiastro.saspro.plate_solver import (
                    _parse_ra_deg, _parse_dec_deg, _estimate_scale_arcsec_from_header)
                h = item.get("header")
                hh = h if isinstance(h, Header) else coerce_to_header(h)
                ra, dec = _parse_ra_deg(hh), _parse_dec_deg(hh)
                sc = _estimate_scale_arcsec_from_header(hh)
                if ra is not None and dec is not None and sc:
                    w = WCS(naxis=2)
                    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
                    w.wcs.crval = [float(ra), float(dec)]
                    w.wcs.crpix = [W / 2.0 + 0.5, H / 2.0 + 0.5]
                    w.wcs.cd = np.array([[-sc / 3600.0, 0.0], [0.0, sc / 3600.0]])
                    c["approx"] = True
                    c["scale"] = float(sc)
            if w is not None:
                from astropy.wcs.utils import proj_plane_pixel_scales
                if c["scale"] is None:
                    c["scale"] = float(np.mean(proj_plane_pixel_scales(w.celestial))) * 3600.0
                c["sip"] = self._wcs_has_sip(w)
                NS = 8
                xs = np.linspace(0, W - 1, NS); ys = np.linspace(0, H - 1, NS)
                px = np.concatenate([xs, np.full(NS, W - 1), xs[::-1], np.zeros(NS)])
                py = np.concatenate([np.zeros(NS), ys, np.full(NS, H - 1), ys[::-1]])
                ra_, dec_ = w.all_pix2world(px, py, 0)
                g = np.isfinite(ra_) & np.isfinite(dec_)
                c["fp"] = (ra_[g], dec_[g]) if g.sum() >= 3 else None
        except Exception as e:
            print(f"[Mosaic] footprint unavailable for {item.get('path')}: {e}")
        item["_ui_cache"] = c
        return c

    def _reference_index(self, infos):
        """The panel the build will anchor on: a pinned one, else (as
        _order_panels) the plate-solved panel with the most real data."""
        items = self.loaded_images
        if self._opt_seestar():
            return None, False       # frames are averaged on a north-up canvas: no reference
        for i, it in enumerate(items):
            if it.get("pinned_ref"):
                return i, True
        if not items:
            return None, False
        cand = [i for i, it in enumerate(items) if it.get("wcs") is not None] or list(range(len(items)))
        d2 = self._centre_d2(infos)
        return max(cand, key=lambda i: (infos[i]["valid_px"], -d2.get(i, float("inf")))), False

    def _centre_d2(self, infos):
        """Squared distance of each solved panel's footprint centre from the
        field centre, on the tangent plane: _order_panels' tie-break."""
        fps = [(i, inf["fp"]) for i, inf in enumerate(infos) if inf.get("fp") is not None]
        if not fps:
            return {}
        vec = np.concatenate([self._radec_to_vec(ra, dec) for _i, (ra, dec) in fps], axis=0)
        c = vec.mean(axis=0)
        c /= max(float(np.linalg.norm(c)), 1e-12)
        ra0 = float(np.degrees(np.arctan2(c[1], c[0])))
        dec0 = float(np.degrees(np.arcsin(np.clip(c[2], -1.0, 1.0))))
        out = {}
        for i, (ra, dec) in fps:
            x, y = self._gnomonic(ra, dec, ra0, dec0)
            g = np.isfinite(x) & np.isfinite(y)
            if g.any():
                cx = 0.5 * (x[g].min() + x[g].max()); cy = 0.5 * (y[g].min() + y[g].max())
                out[i] = float(cx * cx + cy * cy)
        return out

    def _refresh_panels(self, only=None):
        """Rebuild the panel list, the sky map and the hint line. `only`: the
        rows whose cells changed (the rest are left as they are) -- for the
        Seestar solve loop, where a handful of 600 rows change at a time."""
        from setiastro.saspro.mosaic_master_ui import PanelTable as _PT, PANEL_COLORS
        t = getattr(self, "panel_table", None)
        if t is None:
            return
        items = self.loaded_images
        infos = [self._panel_info(it) for it in items]
        seestar = self._opt_seestar()
        force = self.forceBlindCheckBox.isChecked()
        ref, pinned = self._reference_index(infos)
        scales = [inf["scale"] for inf in infos if inf.get("scale")]
        med_scale = float(np.median(scales)) if scales else None
        mads = [inf["mad"] for inf in infos if inf["mad"] > 0]
        med_mad = float(np.median(mads)) if mads else None
        green, blue, amber, grey = (QColor(110, 200, 120), QColor(120, 170, 240),
                                    QColor(240, 180, 40), QColor(150, 150, 155))

        t.blockSignals(True)
        try:
            with t.bulk_update():
                if only is not None and t.rowCount() != len(items):
                    only = None
                if only is None:
                    t.setRowCount(len(items))
                n_warn = 0
                for r, (it, inf) in enumerate(zip(items, infos)):
                    if only is not None and r not in only:
                        continue
                    W, H = inf["size"]
                    warns = []
                    if med_mad and inf["mad"] > 0 and len(mads) > 1 and (
                            inf["mad"] > 8 * med_mad or inf["mad"] < med_mad / 8):
                        warns.append("Its tone scale is very different from the other panels: "
                                     "one may be stretched and the others linear. Seams won't match.")
                    if med_scale and inf.get("scale") and len(scales) > 1 and \
                            abs(inf["scale"] / med_scale - 1.0) > 0.10:
                        warns.append(f"Pixel scale {inf['scale']:.2f}″ differs from the others "
                                     f"({med_scale:.2f}″).")
                    n_warn += bool(warns)
                    notes = [str(it.get("path", "")), f"{W} × {H} px"]
                    if inf["valid_frac"] < 0.995:
                        notes.append(f"{100 * (1 - inf['valid_frac']):.0f}% of this frame is empty "
                                     f"(is it a mosaic itself?)")
                    cells = [None] * len(_PT.COLS)
                    cells[_PT.COL_IDX] = QTableWidgetItem(str(r + 1))
                    nm = QTableWidgetItem(("⚠ " if warns else "") + self._item_label(it))
                    if inf.get("icon") is not None:
                        nm.setIcon(inf["icon"])
                    nm.setToolTip("\n".join(notes + warns))
                    if warns:
                        nm.setForeground(amber)
                    cells[_PT.COL_NAME] = nm
                    state = it.get("solve_state")
                    if seestar and state == "failed":
                        sv = QTableWidgetItem("✗ Not solved")
                        sv.setForeground(QColor(235, 90, 80))
                        if it.get("failed_cached"):
                            sv.setToolTip(f"Plate solving failed in an earlier build: "
                                          f"{it.get('solve_error', '')}\n"
                                          "It isn't tried again, and the frame is left out of the "
                                          "mosaic.\nTurn on Re-solve every panel (Preparation) "
                                          "to try it again.")
                        else:
                            sv.setToolTip(f"Plate solving failed: {it.get('solve_error', '')}\n"
                                          "This frame is left out of the mosaic.")
                    elif seestar and state == "solving":
                        sv = QTableWidgetItem("Solving…")
                        sv.setForeground(blue)
                    elif seestar and it.get("wcs") is not None and not force:
                        sv = QTableWidgetItem("✓ Solved (cached)" if state == "cached" else "✓ Solved")
                        sv.setForeground(green)
                        sv.setToolTip("Plate solution " + ("from the cache of an earlier build."
                                                           if state == "cached" else "found."))
                    elif seestar:
                        sv = QTableWidgetItem("Solved at build")
                        sv.setForeground(blue)
                        sv.setToolTip("Plate-solved when you build, seeded by the pointing in the header.")
                    elif force:
                        sv = QTableWidgetItem("Re-solve at build")
                        sv.setForeground(blue)
                        sv.setToolTip("Re-solve every panel is on (Preparation).")
                    elif it.get("wcs") is not None:
                        sv = QTableWidgetItem("✓ Solved (SIP)" if inf["sip"] else "✓ Solved")
                        sv.setForeground(green)
                        sv.setToolTip("Has a plate solution" + (" with SIP distortion: placed by WCS, "
                                      "not star-refined." if inf["sip"] else "."))
                    else:
                        sv = QTableWidgetItem("Solved at build")
                        sv.setForeground(blue)
                        sv.setToolTip("No plate solution yet. The in-house Gaia blind solver runs\n"
                                      "when you build; Astrometry.net only if that fails.")
                    cells[_PT.COL_SOLVE] = sv
                    cells[_PT.COL_SCALE] = QTableWidgetItem(
                        f"{inf['scale']:.2f}″" if inf.get("scale") else "—")
                    rf = QTableWidgetItem("—" if seestar else ("★" if r == ref else "☆"))
                    rf.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    rf.setForeground(QColor(255, 215, 0) if (r == ref and not seestar) else grey)
                    if seestar:
                        rf.setToolTip("Seestar mosaics need no reference frame: the canvas is "
                                      "laid out north up from the plate solutions.")
                    if not seestar:
                        rf.setToolTip(("Reference panel (pinned): " if (r == ref and pinned) else
                                   "Reference panel (automatic): " if r == ref else "Click to make this the reference: ")
                                      + "it defines the mosaic's sky frame and sky level.\n"
                                        "Automatic choice: the plate-solved panel with the most real data.")
                    cells[_PT.COL_REF] = rf
                    for col, cell in enumerate(cells):
                        t.setItem(r, col, cell)
        finally:
            t.blockSignals(False)

        # sky map
        self._update_plan_map(infos, ref, PANEL_COLORS)

        # hint line
        hint = []
        if seestar:
            n_ok = sum(1 for it in items if it.get("wcs") is not None)
            n_bad = sum(1 for it in items if it.get("solve_state") == "failed")
            hint.append(f"Seestar: {len(items)} frames, {n_ok} plate-solved"
                        + (f", {n_bad} failed" if n_bad else "")
                        + ". Each frame is placed by its own plate solution and the frames are averaged.")
        else:
            sip_note = getattr(self, "_sip_note", "")
            if sip_note:
                hint.append(sip_note)
            n_uns = sum(1 for it in items if it.get("wcs") is None)
            if n_uns and not force:
                hint.append(f"{n_uns} panel{'s' if n_uns != 1 else ''} will be plate-solved when you build.")
        if n_warn and not seestar:
            hint.append(f"⚠ {n_warn} panel{'s' if n_warn != 1 else ''} with a warning (hover the name).")
        self.optHint.setText("  ".join(hint))
        self.optHint.setVisible(bool(hint))
        self.buildBtn.setEnabled(len(items) >= 1 and not getattr(self, "_building", False))

    def _update_plan_map(self, infos, ref, colors):
        """Footprints laid out in the reference panel's linear TAN frame: the
        frame the mosaic canvas is built in, so the plan looks like the result."""
        fps = [(i, inf["fp"]) for i, inf in enumerate(infos) if inf.get("fp") is not None]
        panels = []
        frame = None
        seestar = self._opt_seestar()
        fw_i = ref if (ref is not None and infos[ref].get("fp") is not None) else (
            fps[0][0] if fps else None)
        fw = self.loaded_images[fw_i].get("wcs") if fw_i is not None else None
        if seestar and fps:
            # the Seestar canvas is north up at the mean pointing: lay the plan out the same way
            vec = np.concatenate([self._radec_to_vec(ra, dec) for _i, (ra, dec) in fps], axis=0)
            cv_ = vec.mean(axis=0); cv_ /= max(float(np.linalg.norm(cv_)), 1e-12)
            sc = [infos[i].get("scale") for i, _ in fps if infos[i].get("scale")]
            s_deg = (float(np.median(sc)) if sc else 2.4) / 3600.0
            fw = WCS(naxis=2)
            fw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            fw.wcs.crval = [float(np.degrees(np.arctan2(cv_[1], cv_[0]))) % 360.0,
                            float(np.degrees(np.arcsin(np.clip(cv_[2], -1, 1))))]
            fw.wcs.crpix = [0.0, 0.0]
            fw.wcs.cd = np.array([[-s_deg, 0.0], [0.0, -s_deg]])
            frame = {"north": (0.0, -1.0), "east": (-1.0, 0.0), "deg_per_unit": s_deg}
        elif fw is not None:
            try:
                fW, fH = infos[fw_i]["size"]
                rc, dc = fw.wcs_pix2world([fW / 2.0], [fH / 2.0], 0)
                rc, dc = float(rc[0]), float(dc[0])
                c0 = np.array(fw.wcs_world2pix([rc], [dc], 0)).ravel()
                dd = 0.01
                nv = np.array(fw.wcs_world2pix([rc], [dc + dd], 0)).ravel() - c0
                ev = np.array(fw.wcs_world2pix([rc + dd / max(math.cos(math.radians(dc)), 1e-6)],
                                               [dc], 0)).ravel() - c0
                frame = {"north": (float(nv[0]), float(nv[1])), "east": (float(ev[0]), float(ev[1])),
                         "deg_per_unit": float(infos[fw_i].get("scale") or 3600.0) / 3600.0}
            except Exception as e:
                print(f"[Mosaic] sky map frame unavailable: {e}")
                frame = None
        if fps and frame is not None:
            # every footprint through the plan frame in ONE call (hundreds of
            # separate astropy calls cost a tenth of a second per refresh)
            try:
                lens = [len(np.atleast_1d(ra)) for _i, (ra, _d) in fps]
                X, Y = fw.wcs_world2pix(np.concatenate([np.atleast_1d(ra) for _i, (ra, _d) in fps]),
                                        np.concatenate([np.atleast_1d(d) for _i, (_r, d) in fps]), 0)
                cuts = np.cumsum([0] + lens)
                proj = [(np.asarray(X[a:b], float), np.asarray(Y[a:b], float))
                        for a, b in zip(cuts[:-1], cuts[1:])]
            except Exception:
                proj = [tuple(np.asarray(v, float) for v in fw.wcs_world2pix(ra, dec, 0))
                        for _i, (ra, dec) in fps]
            for (i, (ra, dec)), (x, y) in zip(fps, proj):
                g = np.isfinite(x) & np.isfinite(y)
                if g.sum() < 3:
                    continue
                inf = infos[i]
                it = self.loaded_images[i]
                W, H = inf["size"]
                tip = (f"{i + 1}. {self._item_label(it)}\n{W} × {H} px"
                       + (f", {inf['scale']:.2f}″/px" if inf.get("scale") else "")
                       + ("\nSIP distortion: placed by WCS" if inf.get("sip") else "")
                       + ("\nReference panel" if i == ref else ""))
                if seestar:
                    st = it.get("solve_state")
                    tip = (f"{i + 1}. {self._item_label(it)}\n"
                           + ("position from the header only (not solved yet)" if inf.get("approx")
                              else "plate-solved" + (" (cached)" if st == "cached" else ""))
                           + (f"\nsolve failed: {it.get('solve_error', '')}" if st == "failed" else ""))
                panels.append({"idx": i, "name": self._item_label(it),
                               "poly": np.column_stack([x[g], y[g]]),
                               "ref": (i == ref) and not seestar,
                               "approx": bool(inf.get("approx")),
                               "failed": it.get("solve_state") == "failed",
                               "color": colors[i % len(colors)], "tip": tip})
        n_uns = len(infos) - len(fps)
        if seestar:
            n_appr = sum(1 for d in panels if d["approx"])
            note = (f"{n_appr} placed from the header pointing (dashed) until solved"
                    if n_appr else "")
        else:
            note = (f"{n_uns} not plate-solved yet" if n_uns else "")
        self.map.set_plan(panels, note, frame, dense=seestar)

    def _on_panel_cell_clicked(self, row, col):
        from setiastro.saspro.mosaic_master_ui import PanelTable as _PT
        if col != _PT.COL_REF or not (0 <= row < len(self.loaded_images)):
            return
        it = self.loaded_images[row]
        was = bool(it.get("pinned_ref"))
        for x in self.loaded_images:
            x.pop("pinned_ref", None)
        if not was:
            it["pinned_ref"] = True
            self.status_label.setText(f"Reference panel: {self._item_label(it)} (pinned).")
        else:
            self.status_label.setText("Reference panel: automatic.")
        self._refresh_panels()

    def _on_table_selection(self):
        rows = sorted({i.row() for i in self.panel_table.selectionModel().selectedRows()})
        self.map.set_selected(rows)

    def _select_panel_row(self, idx):
        if 0 <= idx < self.panel_table.rowCount():
            self.panel_table.selectRow(idx)

    # ── Seestar: plate solve every frame, place by WCS, average ───────────
    def _seestar_adopt(self, item, header_str, state):
        from setiastro.saspro.plate_solver import _wcs_from_header_2d
        try:
            h = Header.fromstring(header_str)
            w = _wcs_from_header_2d(h, relax=True)
        except Exception as e:
            item["solve_state"] = "failed"
            item["solve_error"] = f"unusable solution ({e})"
            return False
        if w is None or not getattr(w, "is_celestial", False):
            item["solve_state"] = "failed"
            item["solve_error"] = "unusable solution"
            return False
        item["wcs"] = w
        item["solved_header"] = header_str
        item["solve_state"] = state
        item.pop("solve_error", None)
        return True

    def _seestar_pool(self, n_tasks):
        """Worker processes for the Seestar build (plate solving, then frame
        preparation), or None when they can't start here."""
        import setiastro.saspro.seestar_mosaic as sm
        try:
            from setiastro.saspro.worker_env import check_process_pools, process_pools_ok
            check_process_pools(log_fn=lambda m: print(f"[Seestar] {m}"))
            if not process_pools_ok() or n_tasks < 2:
                return None
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            from PyQt6.QtCore import QCoreApplication as _QCA
            ncpu = os.cpu_count() or 2
            nw = max(1, min(16, ncpu // 2, n_tasks))
            try:                                           # testing overrides
                nw = max(1, min(int(os.environ.get("SASPRO_SEESTAR_WORKERS") or nw), n_tasks))
            except ValueError:
                pass
            # each worker gets its share of the cores (CLAUDE.md lesson 8)
            nt = max(1, ncpu // nw)
            try:
                _e = int(os.environ.get("SASPRO_SEESTAR_THREADS") or nt)
                nt = 0 if _e < 0 else max(1, _e)           # -1: uncapped (for comparison)
            except ValueError:
                pass
            pool = ProcessPoolExecutor(max_workers=nw, mp_context=_mp.get_context("spawn"),
                                       initializer=sm.init_worker,
                                       initargs=(_QCA.organizationName(), _QCA.applicationName(), nt))
            pool._saspro_workers = nw
            print(f"[Seestar] {nw} worker processes, "
                  + (f"{nt} thread{'s' if nt != 1 else ''} each" if nt else "threads uncapped"))
            return pool
        except Exception as e:
            print(f"[Seestar] worker processes unavailable ({e}); working in this process")
            return None

    def _seestar_solve(self, todo, cache, progress, pool=None):
        """Plate-solve frames in worker processes (in this process if those
        can't start). Updates the table and the placement map as frames come
        in. Returns False if cancelled."""
        import concurrent.futures as cf
        import time as _time
        import setiastro.saspro.seestar_mosaic as sm
        n = len(todo)
        progress.setRange(0, n)
        progress.setValue(0)
        n_done = n_fail = 0
        t0 = _time.time()

        row_of = {id(x): r for r, x in enumerate(self.loaded_images)}
        changed = set()

        new_failures = []

        def _handle(it, res):
            nonlocal n_done, n_fail
            n_done += 1
            if id(it) in row_of:
                changed.add(row_of[id(it)])
            if res and res.get("ok") and res.get("header"):
                if self._seestar_adopt(it, res["header"], "solved"):
                    if res.get("shape"):
                        it["shape"] = tuple(res["shape"])
                    if not self._is_view_path(it.get("path", "")):
                        sm.cache_put(cache, it["path"], res["header"])
                    return
            n_fail += 1
            it["solve_state"] = "failed"
            it["solve_error"] = (res or {}).get("error") or "no solution"
            it.pop("failed_cached", None)
            # remembered only if the solver ran and said no (not a crashed worker)
            if res is not None and not res.get("_exception") \
                    and not self._is_view_path(it.get("path", "")):
                new_failures.append(it)

        nw = getattr(pool, "_saspro_workers", None) if pool is not None else None

        def _ui(force=False):
            # The progress line is cheap: every tick. The table and the sky map
            # are not (hundreds of rows and footprints): at most once a second,
            # and never more than a fifth of the time.
            now = _time.time()
            if force or (n_done != _ui.shown and now - _ui.last >= max(2.0, 10.0 * _ui.cost)):
                self._refresh_panels(only=None if force else set(changed))
                changed.clear()
                QApplication.processEvents()       # the repaint belongs to the cost
                _ui.last = _time.time()
                _ui.cost = _ui.last - now
                _ui.shown = n_done
            rate = n_done / max(now - t0, 1e-3)
            eta = (n - n_done) / rate if rate > 0 else 0
            progress.setValue(n_done)
            if n_done == 0 and nw:
                progress.setLabelText(f"Plate solving {n} frames on {nw} worker processes…\n"
                                      f"Starting the workers: the first solutions take about "
                                      f"20 s ({now - t0:.0f} s so far).")
            else:
                progress.setLabelText(f"Plate solving frames… {n_done}/{n}"
                                      + (f" ({n_fail} failed)" if n_fail else "")
                                      + (f" · about {eta / 60:.1f} min left" if n_done >= 3 else ""))
            QApplication.processEvents()
        _ui.last = 0.0
        _ui.cost = 0.0
        _ui.shown = -1

        for it in todo:
            it["solve_state"] = "solving"
            it.pop("solve_error", None)
        _ui(True)

        if pool is not None:
            print(f"[Seestar] plate solving {n} frames on {getattr(pool, '_saspro_workers', '?')} "
                  f"worker processes")
        try:
            if pool is not None:
                fut = {}
                for it in todo:
                    if it.get("image") is not None:
                        h = it.get("header")
                        hs = (h if isinstance(h, Header) else coerce_to_header(h) or Header()).tostring()
                        f = pool.submit(sm.solve_frame, it["path"], np.asarray(it["image"]), hs)
                    else:
                        f = pool.submit(sm.solve_frame, it["path"])
                    fut[f] = it
                pending = set(fut)
                while pending:
                    done, pending = cf.wait(pending, timeout=0.15, return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        try:
                            res = f.result()
                        except Exception as e:
                            res = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                   "_exception": True}
                        _handle(fut[f], res)
                    _ui()
                    if progress.wasCanceled():
                        for f in pending:
                            f.cancel()
                        for f in pending:
                            fut[f].pop("solve_state", None)
                        return False
            else:
                for it in todo:
                    if it.get("image") is not None:
                        h = it.get("header")
                        hs = (h if isinstance(h, Header) else coerce_to_header(h) or Header()).tostring()
                        res = sm.solve_frame(it["path"], np.asarray(it["image"]), hs)
                    else:
                        res = sm.solve_frame(it["path"])
                    _handle(it, res)
                    _ui(True)
                    if progress.wasCanceled():
                        for x in todo:
                            if x.get("solve_state") == "solving":
                                x.pop("solve_state", None)
                        return False
        finally:
            # Remember the frames that failed -- but only when others solved:
            # if every frame fails, the cause is the setup (no Gaia library,
            # wrong scale in the headers), not the frames, and it can be fixed.
            if new_failures and any(x.get("wcs") is not None and x.get("solve_state") != "failed"
                                    for x in self.loaded_images):
                for x in new_failures:
                    sm.cache_put_failure(cache, x["path"], x.get("solve_error"))
            sm.save_cache(cache)
            _ui(True)
        print(f"[Seestar] plate solved {n - n_fail}/{n} frames in {_time.time() - t0:.0f} s")
        return True

    @staticmethod
    def _run_live(fn, *a, **k):
        """Run fn on a helper thread and keep the window painting until it
        returns. For the few-second whole-canvas numpy steps between the
        Seestar phases: numpy releases the GIL inside its array loops."""
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            f = ex.submit(fn, *a, **k)
            while not f.done():
                cf.wait([f], timeout=0.05)
                QApplication.processEvents()
            return f.result()

    def _build_seestar(self):
        import concurrent.futures as cf
        import setiastro.saspro.seestar_mosaic as sm
        items = list(self.loaded_images)
        if not items:
            return
        progress = QProgressDialog("Plate solving frames…", "Cancel", 0, len(items), self)
        progress.setWindowTitle("Mosaic Master — Seestar")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.show()
        self.spinnerLabel.show()
        if self.spinnerMovie:
            self.spinnerMovie.start()
        QApplication.processEvents()
        pool = self._seestar_pool(len(items))
        try:
            # 1) plate solve whatever isn't solved (cache first)
            cache = getattr(self, "_seestar_cache", None) or sm.load_cache()
            self._seestar_cache = cache
            force = self.forceBlindCheckBox.isChecked()
            todo = []
            for it in items:
                if not force and it.get("wcs") is not None and it.get("solve_state") != "failed":
                    it.setdefault("solve_state", "solved")
                    continue
                if not force and not self._is_view_path(it.get("path", "")):
                    hs = sm.cache_get(cache, it["path"])
                    if hs and self._seestar_adopt(it, hs, "cached"):
                        continue
                    why = sm.cache_failure(cache, it["path"])
                    if why:
                        it["solve_state"] = "failed"
                        it["solve_error"] = why
                        it["failed_cached"] = True
                        continue
                todo.append(it)
            if todo and not self._seestar_solve(todo, cache, progress, pool):
                self.status_label.setText("Cancelled.")
                return
            solved = [it for it in items if it.get("wcs") is not None
                      and it.get("solve_state") != "failed"]
            n_fail = sum(1 for it in items if it.get("solve_state") == "failed")
            if not solved:
                QMessageBox.warning(self, "Mosaic Master",
                                    "No frame could be plate-solved, so nothing can be placed.\n"
                                    "Check that your Gaia catalogue is installed (Settings → "
                                    "Gaia DR3 Library) and that the frames show stars.")
                return

            # 2) the canvas around every solved frame
            def _date(it):
                h = it.get("header")
                try:
                    return str(h.get("DATE-OBS", "")) + str(it.get("path", ""))
                except Exception:
                    return str(it.get("path", ""))
            solved.sort(key=_date)
            shapes = [tuple(it.get("shape") or np.asarray(it["image"]).shape[:2]) for it in solved]
            try:
                cw, (Hc, Wc) = sm.make_canvas([it["wcs"] for it in solved], shapes)
            except Exception as e:
                QMessageBox.warning(self, "Mosaic Master", f"Could not lay out the mosaic:\n{e}")
                return
            Hms = [sm.frame_homography(it["wcs"], sh, cw) for it, sh in zip(solved, shapes)]
            boxes = [sm.canvas_bbox(h, sh, (Hc, Wc)) for h, sh in zip(Hms, shapes)]
            color = any(not it.get("is_mono", True) or sm.bayer_of(it.get("header") or {})
                        or (it.get("image") is not None and np.asarray(it["image"]).ndim == 3)
                        for it in solved)
            C = 3 if color else 1
            print(f"[Seestar] canvas {Wc} x {Hc} px for {len(solved)} frames")

            # 3) average in one pass (outliers are tested at the end, sm.CanvasStack)
            clip = 3.0 if self.seestarRejectCheck.isChecked() else None
            stack = sm.CanvasStack((Hc, Wc), C, clip_k=clip)
            if pool is not None:
                # the pool keeps every core busy: a few threads add each frame
                try:
                    stack.add_threads = max(1, int(os.environ.get("SASPRO_SEESTAR_ADD_THREADS") or 4))
                except ValueError:
                    stack.add_threads = 4
            dm = str(self.settings.value("stacking/debayer_method", "rcd") or "rcd").lower()
            crop = ((int(self.cropLeftSpin.value()), int(self.cropRightSpin.value()),
                     int(self.cropTopSpin.value()), int(self.cropBottomSpin.value()))
                    if self.cropEachCheckBox.isChecked() else (0, 0, 0, 0))
            passes = (1,)
            skies = []
            sky_of = {}
            nthreads = (getattr(pool, "_saspro_workers", 4) if pool is not None
                        else max(1, min(4, (os.cpu_count() or 2) // 2)))
            total = len(solved) * len(passes)
            progress.setRange(0, total)
            k_done = 0
            # frames in flight: enough to keep every worker busy
            depth = nthreads + 4
            slots, free_slots, slot_of = {}, [], {}
            if pool is not None:
                # prepared frames come back through shared memory (sm.attach_slot)
                try:
                    from multiprocessing import shared_memory
                    cap = sm.slot_bytes(max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes), 3)
                    for _ in range(depth):
                        sh = shared_memory.SharedMemory(create=True, size=cap)
                        slots[sh.name] = sh
                        free_slots.append(sh.name)
                except Exception as e:
                    print(f"[Seestar] shared memory unavailable ({e}); results come back pickled")
                    for sh in slots.values():
                        try:
                            sh.close(); sh.unlink()
                        except Exception:
                            pass
                    slots, free_slots = {}, []
            for ps in passes:
                # frames are prepared (read, debayer, sky, warp) in the worker
                # processes when there are some, and added here in order
                ex = pool if pool is not None else cf.ThreadPoolExecutor(max_workers=nthreads)
                window = {}
                try:
                    def _submit(i):
                        it = solved[i]
                        img = it.get("image")
                        slot = free_slots.pop() if free_slots else None
                        slot_of[i] = slot
                        return ex.submit(sm.prepare_frame, it["path"], Hms[i], boxes[i],
                                         image=(np.asarray(img) if img is not None else None),
                                         header=(it.get("header") if img is not None else None),
                                         debayer_method=dm, crop=crop, sky=sky_of.get(i),
                                         out_slot=slot)
                    window = {}
                    nxt = 0
                    for i in range(len(solved)):
                        while nxt < len(solved) and nxt < i + depth:
                            window[nxt] = _submit(nxt)
                            nxt += 1
                        fut_i = window.pop(i)
                        while not fut_i.done():
                            cf.wait([fut_i], timeout=0.1)
                            QApplication.processEvents()
                            if progress.wasCanceled():
                                fut_i.cancel()
                                for f in window.values():
                                    f.cancel()
                                self.status_label.setText("Cancelled.")
                                return
                        pr = fut_i.result()
                        if pr is not None and pr.get("slot"):
                            pr["data"], pr["weight"] = sm.slot_views(slots[pr["slot"]].buf,
                                                                     pr["shape"])
                        if pr is not None:
                            if pr["data"].shape[2] < C:
                                pr["data"] = np.repeat(pr["data"][..., :1], C, axis=2)
                            pr["data"] = pr["data"][..., :C]
                            stack.add(pr)
                            if ps == 1:
                                sky_of[i] = pr["sky"]
                                skies.append(pr["sky"][:C] if pr["sky"].size >= C
                                             else np.repeat(pr["sky"][:1], C))
                        pr = None                         # drop the slot views
                        if slot_of.get(i):
                            free_slots.append(slot_of.pop(i))
                        k_done += 1
                        progress.setValue(k_done)
                        progress.setLabelText(f"Stacking… frame {i + 1}/{len(solved)}")
                        QApplication.processEvents()
                        if progress.wasCanceled():
                            for f in window.values():
                                f.cancel()
                            self.status_label.setText("Cancelled.")
                            return
                finally:
                    if ex is not pool:
                        ex.shutdown(wait=False, cancel_futures=True)
                    if slots:
                        # a cancelled build can leave frames in flight: let the
                        # pool finish them before their slots go away
                        if pool is not None:
                            for f in window.values():
                                f.cancel()
                            cf.wait(list(window.values()), timeout=30)
                        for sh in slots.values():
                            try:
                                sh.close()
                                sh.unlink()
                            except Exception:
                                pass

            progress.setLabelText("Taking out satellites and other outliers…" if clip
                                  else "Finishing…")
            QApplication.processEvents()

            def _finish():
                o = stack.result()
                sk = (np.median(np.array(skies), axis=0).astype(np.float32) if skies
                      else np.zeros(C, np.float32))
                cv = np.asarray(stack.Wt) > 0
                o += sk[None, None, :C]
                o[~cv] = 0.0
                return o, np.array(stack.Wt, dtype=np.float32)
            out, self.weight_mosaic = self._run_live(_finish)
            self.final_mosaic = out[..., 0] if C == 1 else out
            hdr = cw.to_header(relax=True)
            hdr["NAXIS"] = 2
            hdr["NAXIS1"] = Wc
            hdr["NAXIS2"] = Hc
            exps = []
            for it in solved:
                try:
                    exps.append(float((it.get("header") or {}).get("EXPTIME", 0) or 0))
                except Exception:
                    pass
            exps = [e for e in exps if e > 0]
            if exps:
                hdr["EXPTIME"] = (float(np.median(exps)), "per frame (s)")
                hdr["TOTALEXP"] = (float(np.sum(exps)), "total integration (s)")
            hdr["NCOMBINE"] = (len(solved), "frames stacked")
            hdr["HISTORY"] = "SASpro Mosaic Master: Seestar frames plate-solved (Gaia DR3), placed by WCS, averaged"
            self.wcs_metadata = hdr
            self.was_single_channel = (C == 1)

            # the result map: frame outlines and coverage depth
            N = np.asarray(stack.N, np.float32)
            sc = min(1.0, 900.0 / max(Hc, Wc))
            heat = (cv2.resize(N, (max(1, int(Wc * sc)), max(1, int(Hc * sc))),
                               interpolation=cv2.INTER_AREA) if sc < 1.0 else N.copy())
            frames = [sm.canvas_corners(h, sh) for h, sh in zip(Hms, shapes)]
            self._build_report = {
                "mode": "seestar", "panels": [], "seams": [], "sigma": None,
                "shape": (Hc, Wc),
                "coverage": {"heat": heat, "max": int(N.max()) if N.size else 1,
                             "exptime": float(np.median(exps)) if exps else None,
                             "frames": frames}}
            rej = ""
            if clip:
                rej = (f", {100.0 * stack.rejected / max(stack.considered, 1):.2f}% of samples "
                       f"rejected as outliers")
            self._show_result(self._run_live(self._autostretch_if_requested, self.final_mosaic,
                                             self.autostretchPreviewCheck.isChecked()),
                              "Seestar Mosaic")
            self.status_label.setText(
                f"Seestar mosaic ready — {len(solved)} frames"
                + (f" ({n_fail} not solved, left out)" if n_fail else "")
                + f", {Wc}×{Hc} px, up to {int(N.max())} frames deep{rej}. "
                  "Plate-solved: the mosaic carries its WCS.")
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            try:
                progress.close()
            except Exception:
                pass
            if self.spinnerMovie:
                self.spinnerMovie.stop()
            self.spinnerLabel.hide()
            QApplication.processEvents()

    # ── result ────────────────────────────────────────────────────────────
    def _show_result(self, display_image, title="Mosaic"):
        """Show the finished mosaic in the map pane with its panel map."""
        try:
            from setiastro.saspro.imageops.stretch import stretch_color_image
            src_img = self.final_mosaic if self.final_mosaic is not None else display_image
            a = np.asarray(src_img, dtype=np.float32)
            if a.ndim == 2:
                a = a[..., None]
            if a.shape[2] == 1:
                a = np.repeat(a, 3, axis=2)
            H, W = a.shape[:2]
            sc = min(1.0, 1600.0 / max(H, W))
            small = (cv2.resize(a, (max(1, int(W * sc)), max(1, int(H * sc))),
                                interpolation=cv2.INTER_AREA) if sc < 1.0 else a.copy())
            small = np.nan_to_num(small, nan=0.0, posinf=0.0, neginf=0.0)
            disp = stretch_color_image(small, target_median=0.25, linked=True)
            u8 = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
            qimg = QImage(u8.data, u8.shape[1], u8.shape[0], 3 * u8.shape[1],
                          QImage.Format.Format_RGB888).copy()
        except Exception as e:
            print(f"[Mosaic] result thumbnail failed: {e}")
            return
        rep = getattr(self, "_build_report", None) or {}
        panels = list(rep.get("panels", []))
        seams = list(rep.get("seams", []))
        sigma = rep.get("sigma")
        names = [d.get("name", "") for d in panels]
        for k, d in enumerate(panels):
            lines = [f"{k + 1}. {d.get('name', '')}" + ("  (reference)" if d.get("ref") else "")]
            if d.get("align"):
                lines.append(f"Placement: {d['align']}")
            if d.get("gain") is not None:
                lines.append(f"Brightness: gain ×{d['gain']:.4f}, offset {d.get('offset', 0.0):+.3g}")
            d["tip"] = "\n".join(lines)
        for sm in seams:
            i, j = sm.get("i"), sm.get("j")
            a_ = names[i] if i is not None and 0 <= i < len(names) else "?"
            b_ = (names[j] if j is not None and 0 <= j < len(names) else "mosaic so far")
            lines = [f"Seam {a_} / {b_}"]
            if sm.get("before") is not None and sm.get("after") is not None:
                lines.append(f"Brightness difference: {sm['before']:.3g} before, {sm['after']:.3g} after matching")
                if sigma:
                    lines.append(f"= {sm['after'] / sigma:.2f}× the sky noise per pixel")
            sm["tip"] = "\n".join(lines)
        self.map.set_result(qimg, (W, H), panels, seams, sigma, coverage=rep.get("coverage"))
        self._result_display = display_image
        self._result_title = title
        self.mapTitle.setText("<b>Result</b>")
        self.overlayCheck.setVisible(bool(panels) or rep.get("coverage") is not None)
        self.mapToggleBtn.setText("Sky map")
        self.mapToggleBtn.show()
        self.fullPreviewBtn.show()
        self._result_stale = False
        self._update_action_buttons()

    def _toggle_map_mode(self):
        if self.map.mode() == "result":
            self.map.show_plan()
            self.mapTitle.setText("<b>Sky map</b>")
            self.mapToggleBtn.setText("Result")
            self.overlayCheck.hide()
        elif self.map.has_result():
            self.map.show_result()
            self.mapTitle.setText("<b>Result</b>")
            self.mapToggleBtn.setText("Sky map")
            _rep = getattr(self, "_build_report", None) or {}
            self.overlayCheck.setVisible(bool(_rep.get("panels")) or _rep.get("coverage") is not None)

    def _open_full_preview(self):
        if self._result_display is None:
            return
        MosaicPreviewWindow(self._result_display, title=getattr(self, "_result_title", "Mosaic"),
                            parent=self, push_cb=self._open_result_as_view).show()

    def _open_result_as_view(self):
        fin = self.finalize_mosaic()
        if fin is None:
            self.status_label.setText("No mosaic yet: build one first.")
            return
        img, meta = fin
        dm = self._docman
        if dm is None:
            QMessageBox.warning(self, "Mosaic Master", "No document manager is available.")
            return
        try:
            title = self._proposed_mosaic_title()
            if hasattr(dm, "open_array") and callable(dm.open_array):
                newdoc = dm.open_array(img, metadata=meta, title=title)
            elif hasattr(dm, "open_numpy") and callable(dm.open_numpy):
                newdoc = dm.open_numpy(img, metadata=meta, title=title)
            else:
                newdoc = dm.create_document(image=img, metadata=meta, name=title)
            parent = self.parent()
            if parent and hasattr(parent, "_spawn_subwindow_for"):
                parent._spawn_subwindow_for(newdoc)
            self.status_label.setText("Opened the mosaic as a new view.")
        except Exception as e:
            QMessageBox.critical(self, "Mosaic Master", f"Could not open the mosaic:\n{e}")

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
        """After the panel set changes: describe the SIP situation, and default
        Star refine to Off when most panels carry SIP (WCS placement is then
        more accurate), unless the user picked a setting this session."""
        combo = getattr(self, "starRefineCombo", None)
        solved = [it for it in self.loaded_images if it.get("wcs") is not None]
        if combo is None or not solved:
            self._sip_note = ""
            return
        n_sip = sum(1 for it in solved if self._wcs_has_sip(it["wcs"]))
        frac = n_sip / len(solved)
        if frac >= 0.8:
            self._sip_note = (f"{n_sip} of {len(solved)} panels carry SIP distortion: "
                              f"WCS placement is more accurate than star refinement.")
        elif n_sip:
            self._sip_note = (f"{n_sip} of {len(solved)} panels carry SIP distortion; "
                              f"those are always placed by WCS.")
        else:
            self._sip_note = ""
        if not getattr(self, "_starrefine_user_touched", False):
            want = "off" if frac >= 0.8 else "Affine Transform"
            combo.blockSignals(True)
            self._set_combo_data(combo, want)
            combo.blockSignals(False)

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
        _ = qs_int(s, "mosaic/star_target", MOSAIC_STAR_TARGET)
        # etc...
        s.sync()

    def _target_median_from_first(self, items):
        # Pick a stable target (median of first image after safe clipping).
        # No-data pixels are excluded: on a panel that is itself a mosaic they
        # would pull both percentiles and the median toward zero.
        if not items:
            return 0.1
        a0 = items[0]["image"].astype(np.float32)
        valid = _valid_data_mask(a0, erode_px=0)
        if a0.ndim == 3:
            a0 = np.mean(a0, axis=2)
        sel = a0[valid] if np.any(valid) else a0.ravel()
        if sel.size == 0:
            return 0.1
        m = np.median(np.clip(sel, np.percentile(sel, 1), np.percentile(sel, 99)))
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
    def _adv_store_panel(adv, idx, name, aligned, coverage):
        """Write one aligned panel (pedestal stripped, before any overlap fit)
        to disk, cropped to its footprint. The feather weight is added once it
        has been computed."""
        ys, xs = np.nonzero(coverage)
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        a3 = aligned if aligned.ndim == 3 else aligned[..., None]
        crop = np.ascontiguousarray(a3[y0:y1, x0:x1], dtype=np.float32)
        cov = np.ascontiguousarray(coverage[y0:y1, x0:x1])
        if cov.any():
            sat = _ADV_SAT * crop[cov].max(axis=0).astype(np.float64)   # per channel
        else:
            sat = np.full(crop.shape[2], np.inf)
        base = os.path.join(adv["dir"], f"p{idx:03d}")
        np.save(base + "_img.npy", crop)
        np.save(base + "_cov.npy", cov)
        return {"name": name, "bbox": (y0, y1, x0, x1), "C": int(crop.shape[2]),
                "img_path": base + "_img.npy", "cov_path": base + "_cov.npy",
                "smooth_path": base + "_w.npy", "sat": sat,
                "cx": 0.5 * (x0 + x1), "cy": 0.5 * (y0 + y1),
                "hx": max(0.5 * (x1 - x0), 1.0), "hy": max(0.5 * (y1 - y0), 1.0)}

    def _adv_rematch(self, adv, is_color, _tick):
        """Joint solve over every overlap, then rebuild the mosaic with it.
        Returns False if cancelled."""
        recs = adv["recs"]
        if len(recs) < 2:
            return True
        if not _tick(text=f"Advanced matching: solving {len(recs)} panels together…"):
            return False
        params, n_pairs, seams = solve_global_panel_match(recs, self._ref_bg, ref=0)
        if n_pairs == 0:
            return True
        rep = getattr(self, "_build_report", None)
        if rep is not None and len(rep.get("panels", [])) == len(recs):
            for k in range(len(recs)):
                rep["panels"][k]["gain"] = float(np.mean(params[k, :, 0]))
                rep["panels"][k]["offset"] = float(np.mean(params[k, :, 1]))
            if rep.get("sigma") is not None:
                rep["seams"] = seams
        self.final_mosaic.fill(0.0)
        self.weight_mosaic.fill(0.0)
        for k, rec in enumerate(recs):
            if not _tick(text=f"Advanced matching: rebuilding with panel "
                              f"{k + 1}/{len(recs)} ({rec['name']})"):
                return False
            img = _adv_load(rec["img_path"], mmap=False)
            cov = _adv_load(rec["cov_path"], mmap=False)
            w = _adv_load(rec["smooth_path"], mmap=False)
            corr = apply_panel_match(img, cov, rec, params[k], self._ref_bg)
            del img, cov
            y0, y1, x0, x1 = rec["bbox"]
            if is_color:
                if corr.shape[2] == 1:
                    corr = np.repeat(corr, 3, axis=2)
                self.final_mosaic[y0:y1, x0:x1] += corr * w[..., None]
            else:
                self.final_mosaic[y0:y1, x0:x1] += corr[..., 0] * w
            self.weight_mosaic[y0:y1, x0:x1] += w
            del corr, w
        print(f"[Mosaic+] mosaic rebuilt from a joint solve over {n_pairs} overlaps")
        return True

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

    def _normalize_linear(self, arr, target_med, valid=None):
        """Linear median match only (no stretch/curves). Returns float32 array.

        The median is taken over real data only; counting no-data zeros would
        drag it down and the compensating gain would blow the panel out.
        """
        img = arr.astype(np.float32, copy=False)
        mono = img if img.ndim == 2 else np.mean(img, axis=2)
        if valid is None:
            valid = _valid_data_mask(img, erode_px=0)
        sel = mono[valid] if np.any(valid) else mono.ravel()
        if sel.size == 0:
            return img
        med = float(np.median(sel))
        if med <= 0:
            return img
        gain = target_med / med
        return img * gain

    def _autostretch_if_requested(self, arr, on=None):
        """Used only for preview windows based on the checkbox. Pass `on` (the
        checkbox, read on the GUI thread) to run this on a helper thread."""
        if on is None:
            on = self.autostretchPreviewCheck.isChecked()
        if not on:
            # Convert linear [0..1-ish] to 8-bit safely for preview
            a = arr.astype(np.float32)
            v = _valid_data_mask(a, erode_px=0)
            sel = a[v] if np.any(v) else a.ravel()
            ref = np.percentile(sel, 99.9) if sel.size else 1.0
            a = a / max(1e-6, float(ref))
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
        Reorder panels so the sequence starts on the panel carrying the most
        real data and every later panel overlaps sky that has already been
        placed.

        The first panel is not merely first: it defines the mosaic's reference
        TAN frame AND the reference sky level every other panel is matched to.
        Seeding on real-data area rather than centrality means a large existing
        mosaic anchors the run instead of a single small tile -- the sky
        measurement is better supported, and the later panels (not the anchor)
        are the ones that get a gain fit, which is the better-conditioned way
        round.

        Footprints are computed with all_pix2world only -- forward SIP is a
        plain polynomial evaluation, so nothing iterates and nothing diverges,
        and this runs before the mosaic WCS exists.
        """
        n = len(items)
        # n == 2 used to return early, leaving whichever panel the user happened
        # to add first as the reference -- including a small tile ahead of a
        # whole mosaic. Two panels still deserve a considered order.
        if n < 2:
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

        # Seed on real-data area, falling back to centrality to break ties.
        valid_px = []
        for itm in items:
            try:
                valid_px.append(int(np.count_nonzero(
                    _valid_data_mask(itm["image"], erode_px=0))))
            except Exception:
                valid_px.append(int(np.prod(itm["image"].shape[:2])))

        pinned = [i for i, it in enumerate(items) if it.get("pinned_ref")]
        order = ([pinned[0]] if pinned else
                 [max(range(n), key=lambda i: (valid_px[i], -d2(boxes[i])))])
        print(f"[Mosaic] anchor panel: {self._item_label(items[order[0]])} "
              f"({valid_px[order[0]]:,} valid px) — defines the reference "
              f"frame and sky level")
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

    def _warp_via_wcs_homography(self, img, src_wcs, dst_wcs, out_shape,
                                 H_cache=None, src_mask=None):
        """
        Fast path: single homography warp per image (per mosaic WCS).
        If H_cache is provided as a dict, we reuse computed H.

        With `src_mask` (a bool array of the source's real-data pixels) the same
        transform is applied to the mask and the result is returned alongside,
        so the caller can tell real data from no-data fill in the output.
        Returns `dst` alone when src_mask is None, else `(dst, dst_mask)`.
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
            out = cv2.warpPerspective(img, H33, (W, H), flags=cv2.INTER_LANCZOS4)
        else:
            # Warp each channel
            out = np.empty((H, W, img.shape[2]), dtype=np.float32)
            for c in range(img.shape[2]):
                out[..., c] = cv2.warpPerspective(img[..., c], H33, (W, H), flags=cv2.INTER_LANCZOS4)

        if src_mask is None:
            return out

        # INTER_LINEAR, not LANCZOS4: a 0/1 mask rings badly under Lanczos and
        # the overshoot would push fill pixels back above the acceptance
        # threshold, which is the exact failure this mask exists to prevent.
        dstm = cv2.warpPerspective(
            src_mask.astype(np.float32), H33, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return out, dstm

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
                                  tile=512, progress_cb=None, src_mask=None):
        """
        Inverse mapping dst -> world -> src, tiled, restricted to the panel's
        actual footprint in the destination frame. Outside that box the SIP
        inverse cannot converge (the polynomial has no inverse beyond its fit
        domain), so we never ask it to. Every solved coordinate is validated by
        a forward round trip before it is used.

        With `src_mask` (a bool array of the source's real-data pixels) the same
        per-pixel maps are reused to remap the mask -- effectively free, since
        the expensive part is solving the coordinates -- and the warped mask is
        returned alongside the image. That is the only way to know which output
        pixels came from real data when the SOURCE itself has interior holes:
        the +PED trick can only detect pixels outside the warp footprint.
        Returns `dst` alone when src_mask is None, else `(dst, dst_mask)`.
        """
        import warnings

        H, W = int(out_shape[0]), int(out_shape[1])
        is_color = (src_img.ndim == 3)
        nch = src_img.shape[2] if is_color else 1
        sh, sw = src_img.shape[:2]

        dst = (np.zeros((H, W, nch), np.float32) if is_color
               else np.zeros((H, W), np.float32))
        want_mask = src_mask is not None
        dstm = np.zeros((H, W), np.float32) if want_mask else None
        mask_f = (np.ascontiguousarray(src_mask.astype(np.float32))
                  if want_mask else None)

        box = self._dst_footprint_bbox(src_img.shape, src_wcs, dst_wcs, (H, W))
        if box is None:
            print("[Mosaic] panel footprint falls outside the mosaic canvas.")
            return (dst, dstm) if want_mask else dst
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

                    if want_mask:
                        # INTER_LINEAR, not LANCZOS4: Lanczos rings on a 0/1
                        # edge and the overshoot would readmit fill pixels.
                        dstm[y0:y1, x0:x1] = cv2.remap(
                            mask_f, mapx, mapy,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

                    done_tiles += 1
                    if progress_cb is not None and (done_tiles % 8) == 0:
                        progress_cb(done_tiles / max(1, n_tiles))

        if rejected:
            tot = (bx1 - bx0) * (by1 - by0)
            print(f"[Mosaic] rejected {rejected} / {tot} mapped pixels "
                  f"({100.0 * rejected / max(1, tot):.1f}% of footprint box)")

        return (dst, dstm) if want_mask else dst

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
        start = str(self.settings.value("mosaic/last_dir", "") or "")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Panels", start,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.fts *.fz *.xisf "
            "*.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef)")
        if paths:
            self._persist("mosaic/last_dir", os.path.dirname(paths[0]))
            self._add_paths(paths)

    def _add_paths(self, paths):
        """Load panel files (from the file dialog or dropped on the list)."""
        have = {str(it.get("path")) for it in self.loaded_images}
        added, failed = 0, []
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for path in paths:
                if not path or str(path) in have or os.path.isdir(path):
                    continue
                if self._opt_seestar() and str(path).lower().endswith(
                        (".fit", ".fits", ".fts", ".fz")):
                    lazy = self._lazy_item(path)
                    if lazy is not None:
                        self.loaded_images.append(lazy)
                        have.add(str(path))
                        added += 1
                        if added % 25 == 0:
                            self.status_label.setText(f"Adding frames… {added}")
                            QApplication.processEvents()
                        continue
                self.status_label.setText(f"Loading {os.path.basename(path)}…")
                QApplication.processEvents()
                try:
                    arr, header, bitdepth, ismono = load_image(path)
                except Exception as e:
                    print(f"[Mosaic] could not load {path}: {e}")
                    arr = None
                if arr is None:
                    failed.append(os.path.basename(path))
                    continue
                header = sanitize_wcs_header(header) if header else None
                wcs_obj = get_wcs_from_header(header) if header else None
                self.loaded_images.append({
                    "path": path,
                    "display": os.path.basename(path),
                    "image": arr,
                    "header": header,
                    "wcs": wcs_obj,
                    "bit_depth": bitdepth,
                    "is_mono": ismono,
                    "transform": None,
                })
                have.add(str(path))
                added += 1
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_sip_autodisable()
        self._refresh_panels()
        if added:
            self._mark_result_stale()
        msg = f"{len(self.loaded_images)} panels loaded"
        if added:
            msg += f" ({added} added)"
        if failed:
            msg += f". Could not read: {', '.join(failed[:4])}" + ("…" if len(failed) > 4 else "")
        self.status_label.setText(msg + ".")

    def add_folder(self):
        start = str(self.settings.value("mosaic/last_dir", "") or "")
        d = QFileDialog.getExistingDirectory(self, "Add a Folder of Frames", start)
        if not d:
            return
        self._persist("mosaic/last_dir", d)
        exts = (".fit", ".fits", ".fts", ".fz", ".xisf", ".tif", ".tiff")
        paths = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith(exts) and os.path.isfile(os.path.join(d, f)))
        if not paths:
            self.status_label.setText("No image files in that folder.")
            return
        self._add_paths(paths)

    def _lazy_item(self, path):
        """A frame registered from its header only: pixels are read when the
        frame is solved and stacked, so hundreds of frames cost no memory."""
        try:
            hdr = fits.getheader(path)
        except Exception as e:
            print(f"[Mosaic] could not read the header of {path}: {e}")
            return None
        H = int(hdr.get("NAXIS2", 0) or 0); W = int(hdr.get("NAXIS1", 0) or 0)
        if H <= 0 or W <= 0:
            return None
        h2 = sanitize_wcs_header(hdr) if hdr else None
        item = {"path": path, "display": os.path.basename(path), "image": None,
                "lazy": True, "header": hdr, "wcs": get_wcs_from_header(h2) if h2 else None,
                "bit_depth": None, "is_mono": int(hdr.get("NAXIS", 2)) == 2 and not hdr.get("BAYERPAT"),
                "transform": None, "shape": (H, W)}
        if item["wcs"] is None:
            try:
                import setiastro.saspro.seestar_mosaic as _sm
                if not hasattr(self, "_seestar_cache"):
                    self._seestar_cache = _sm.load_cache()
                hs = _sm.cache_get(self._seestar_cache, path)
                if hs:
                    self._seestar_adopt(item, hs, "cached")
                else:
                    why = _sm.cache_failure(self._seestar_cache, path)
                    if why:
                        item["solve_state"] = "failed"
                        item["solve_error"] = why
                        item["failed_cached"] = True
            except Exception:
                pass
        return item

    def _ensure_loaded(self, item):
        if item.get("image") is not None:
            return True
        try:
            arr, header, bitdepth, ismono = load_image(item["path"])
        except Exception as e:
            print(f"[Mosaic] could not load {item.get('path')}: {e}")
            return False
        if arr is None:
            return False
        item["image"] = arr
        item["bit_depth"] = bitdepth
        item["is_mono"] = ismono
        item.pop("lazy", None)
        return True

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

        self._refresh_sip_autodisable()
        self._refresh_panels()
        if chosen:
            self._mark_result_stale()
        n_solved = sum(1 for e in chosen if e["wcs"] is not None)
        self.status_label.setText(
            f"{len(self.loaded_images)} panels loaded "
            f"({len(chosen)} added, {n_solved} already plate-solved).")

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
        rows = sorted({i.row() for i in self.panel_table.selectionModel().selectedRows()},
                      reverse=True)
        if not rows:
            self.status_label.setText("Select the panels to remove.")
            return
        for r in rows:
            if 0 <= r < len(self.loaded_images):
                del self.loaded_images[r]
        self._refresh_sip_autodisable()
        self._refresh_panels()
        self._mark_result_stale()
        self.update_status()

    def update_status(self):
        c = len(self.loaded_images)
        self.status_label.setText(f"{c} panel{'s' if c != 1 else ''} loaded."
                                  if c else "Add panels to begin.")

    # ---------- Preview ----------
    def preview_selected(self):
        rows = sorted({i.row() for i in self.panel_table.selectionModel().selectedRows()})
        if not rows or not (0 <= rows[0] < len(self.loaded_images)):
            self.status_label.setText("Select a panel to preview.")
            return
        d = self.loaded_images[rows[0]]
        if not self._ensure_loaded(d):
            self.status_label.setText(f"Could not read {self._item_label(d)}.")
            return
        disp = self._autostretch_if_requested(d["image"])
        MosaicPreviewWindow(disp, title=self._item_label(d), parent=self).show()

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


    # ---------- Pre-process (crop) and post-solve SRGC ----------
    def _stash_panel_sources(self):
        """Remember each panel as it was loaded, so a later run can re-crop
        from the original instead of cropping the already-cropped pixels."""
        for item in self.loaded_images:
            if "_src_image" in item:
                continue
            item["_src_image"] = item["image"]
            item["_src_header"] = item.get("header")
            item["_src_wcs"] = item.get("wcs")

    def _restore_panel_sources(self):
        for item in self.loaded_images:
            if "_src_image" not in item:
                continue
            item["image"] = item.pop("_src_image")
            item["header"] = item.pop("_src_header", item.get("header"))
            item["wcs"] = item.pop("_src_wcs", item.get("wcs"))

    def _sync_list_labels(self):
        self._refresh_panels()

    def _shift_wcs_for_crop(self, header, wcs, x0, y0, new_w, new_h):
        """Shift CRPIX by the crop origin and rewrite NAXIS to the new size.

        SIP stays valid: its polynomials are offsets from CRPIX, so moving
        CRPIX by the pixels that were cut off keeps the same sky reference.
        """
        src = None
        if isinstance(header, Header):
            src = header.copy()
        elif header is not None:
            src = sanitize_wcs_header(header)
            if src is not None:
                src = src.copy()
        if src is None and wcs is not None:
            try:
                src = wcs.to_header(relax=True)
            except Exception:
                src = None
        if src is None:
            return header, wcs

        for key, delta in (("CRPIX1", x0), ("CRPIX2", y0)):
            if key not in src:
                continue
            try:
                src[key] = float(src[key]) - float(delta)
            except Exception:
                pass
        src["NAXIS"] = 2
        src["NAXIS1"] = int(new_w)
        src["NAXIS2"] = int(new_h)

        new_wcs = get_wcs_from_header(src)
        if new_wcs is None and wcs is not None:
            try:
                new_wcs = wcs.deepcopy()
                new_wcs.wcs.crpix[0] -= float(x0)
                new_wcs.wcs.crpix[1] -= float(y0)
            except Exception:
                new_wcs = None
                print("[Mosaic] crop: could not shift the plate solution; "
                      "this panel will be solved again.")
        return src, new_wcs

    def _crop_panel(self, img, header, wcs, left, right, top, bottom):
        x0, y0 = int(left), int(top)
        h, w = img.shape[:2]
        x1, y1 = w - int(right), h - int(bottom)
        if img.ndim == 3:
            cropped = np.ascontiguousarray(img[y0:y1, x0:x1, :])
        else:
            cropped = np.ascontiguousarray(img[y0:y1, x0:x1])
        new_h, new_w = cropped.shape[:2]
        new_header, new_wcs = self._shift_wcs_for_crop(
            header, wcs, x0, y0, new_w, new_h)
        return cropped, new_header, new_wcs

    def _header_for_srgc(self, item):
        """Header carrying the panel's current plate solution for SRGC."""
        hdr = item.get("header")
        if isinstance(hdr, Header):
            return hdr
        if hdr is not None:
            cleaned = sanitize_wcs_header(hdr) or coerce_to_header(hdr)
            if cleaned is not None:
                return cleaned
        wcs = item.get("wcs")
        if wcs is not None:
            try:
                return wcs.to_header(relax=True)
            except Exception:
                pass
        return None

    def _run_srgc_on_panel(self, item, *, params, progress_cb=None):
        """Headless SRGC on one solved panel. Raises on hard failure."""
        from setiastro.saspro.survey_gradient import (
            SURVEYS, DEFAULT_SURVEY, SurveyFetchError,
            survey_gradient_model, apply_survey_gradient, reset_dead_hosts,
        )
        from setiastro.saspro.srgc_preset import srgc_params_from_preset

        params = srgc_params_from_preset(params)
        survey = params["survey"]
        info = SURVEYS.get(survey) or SURVEYS[DEFAULT_SURVEY]
        hdr = self._header_for_srgc(item)
        if hdr is None:
            raise SurveyFetchError(
                "No usable WCS on this panel. Plate-solve it first — "
                "SRGC fetches the reference by sky coordinates."
            )
        reset_dead_hosts()
        img = np.asarray(item["image"], dtype=np.float32)
        res = survey_gradient_model(
            img, hdr,
            channels=list(info.get("channels") or []),
            work_long_side=params["work_long_side"],
            degree=params["degree"],
            use_rbf=params["rbf"],
            num_samples=params["samples"],
            patch_size=params["patch"],
            exclusion_fraction=params["exclude"] / 100.0,
            protect_k=params["protect_k"],
            progress_cb=progress_cb,
        )
        strength = params["strength"] / 100.0
        out = apply_survey_gradient(img, res["gradient"], strength)
        item["image"] = np.asarray(out, dtype=np.float32)
        return res

    def _srgc_panels(self, *, tick=None) -> bool:
        """Run SRGC on every panel that has a WCS. Optional.

        ``tick(value, text)`` advances an outer progress dialog when given
        (the Align path already has one open). Otherwise a short dialog is
        created here. Returns False only when the user cancels. A failed
        panel keeps its pre-SRGC pixels and the mosaic continues.
        """
        if not bool(self.srgcEachCheckBox.isChecked()):
            return True

        targets = [it for it in self.loaded_images if it.get("wcs") is not None]
        if not targets:
            QMessageBox.warning(
                self, "Mosaic Master",
                "SRGC is checked, but no panel has a usable plate solution.\n"
                "Solve the panels first, or uncheck SRGC.")
            return True

        skipped = len(self.loaded_images) - len(targets)
        survey = str(self.srgcSurveyCombo.currentText() or "")
        params = {
            "survey": survey,
            "work_long_side": int(self.settings.value("srgc/work_long_side", 1024)),
            "degree": int(self.settings.value("srgc/degree", 1)),
            "rbf": qs_bool(self.settings, "srgc/rbf", True),
            "samples": int(self.settings.value("srgc/samples", 200)),
            "patch": int(self.settings.value("srgc/patch", 15)),
            "exclude": int(self.settings.value("srgc/exclude", 50)),
            "protect_k": float(self.settings.value("srgc/protect_k", 3.0)),
            "strength": int(self.settings.value("srgc/strength", 100)),
        }

        n = len(targets)
        own_progress = None
        if tick is None:
            own_progress = QProgressDialog("SRGC on panels...", "Cancel", 0, n, self)
            own_progress.setWindowTitle("Mosaic Master")
            own_progress.setWindowModality(Qt.WindowModality.WindowModal)
            own_progress.setAutoClose(False)
            own_progress.setAutoReset(False)
            own_progress.setMinimumDuration(0)
            own_progress.setValue(0)
            own_progress.show()
            QApplication.processEvents()

            def tick(value=None, text=None, _p=own_progress):
                if text is not None:
                    _p.setLabelText(text)
                    self.status_label.setText(text)
                if value is not None:
                    _p.setValue(int(value))
                QApplication.processEvents()
                return not _p.wasCanceled()

        failures = []
        try:
            if not tick(0, f"SRGC 0/{n}..."):
                self.status_label.setText("Cancelled.")
                return False
            for i, item in enumerate(targets):
                label = self._item_label(item)
                if not tick(i, f"SRGC {i + 1}/{n}: {label}"):
                    self.status_label.setText("Cancelled.")
                    return False

                def _cb(msg, _label=label, _tick=tick):
                    _tick(text=f"SRGC {_label}: {msg}")

                try:
                    res = self._run_srgc_on_panel(
                        item, params=params, progress_cb=_cb)
                    cov = float(res.get("coverage", 0.0)) * 100.0
                    print(f"[Mosaic] SRGC done: {label} "
                          f"(survey={params['survey']}, coverage={cov:.1f}%)")
                except Exception as e:
                    failures.append(f"{label}: {e}")
                    print(f"[Mosaic] SRGC failed for {label}: {e}")
            tick(n, f"SRGC done ({n} panel{'s' if n != 1 else ''}).")
        finally:
            if own_progress is not None:
                try:
                    own_progress.close()
                except Exception:
                    pass
                QApplication.processEvents()

        if skipped:
            print(f"[Mosaic] SRGC skipped {skipped} panel(s) with no WCS.")
        if failures:
            shown = "\n".join(failures[:8])
            extra = ""
            if len(failures) > 8:
                extra = f"\n… and {len(failures) - 8} more."
            QMessageBox.warning(
                self, "Mosaic Master",
                "SRGC failed on some panels. Those panels keep their pixels "
                "from before SRGC, and the mosaic will continue.\n\n"
                + shown + extra)
        return True

    def _preprocess_panels(self) -> bool:
        """Crop every panel. Optional.

        Returns False when the user cancels or the crop cannot be applied.
        Re-runs always start from the images as loaded, so changing the
        margins does not crop an already cropped panel. SRGC runs later,
        after plate solving, because it needs a WCS.
        """
        do_crop = bool(self.cropEachCheckBox.isChecked())
        if not do_crop:
            if any("_src_image" in it for it in self.loaded_images):
                self._restore_panel_sources()
                self._sync_list_labels()
                self.status_label.setText(f"{len(self.loaded_images)} images loaded.")
            return True

        left = int(self.cropLeftSpin.value())
        right = int(self.cropRightSpin.value())
        top = int(self.cropTopSpin.value())
        bottom = int(self.cropBottomSpin.value())
        if left == right == top == bottom == 0:
            QMessageBox.warning(
                self, "Mosaic Master",
                "Crop each image is checked, but every margin is 0.\n"
                "Set how many pixels to remove from each edge, or uncheck the option.")
            return False

        too_small = []
        for item in self.loaded_images:
            src = item["_src_image"] if "_src_image" in item else item["image"]
            h, w = src.shape[:2]
            if (w - left - right) < 32 or (h - top - bottom) < 32:
                too_small.append(f"{self._item_label(item)} ({w}×{h})")
        if too_small:
            QMessageBox.warning(
                self, "Mosaic Master",
                "Those margins would leave less than 32 pixels on:\n\n"
                + "\n".join(too_small))
            return False

        self._stash_panel_sources()

        n = len(self.loaded_images)
        progress = QProgressDialog("Preparing panels...", "Cancel", 0, max(1, n), self)
        progress.setWindowTitle("Mosaic Master")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        def _cancelled():
            self._restore_panel_sources()
            self._sync_list_labels()
            self.status_label.setText("Cancelled.")
            return False

        try:
            for step, item in enumerate(self.loaded_images, start=1):
                if progress.wasCanceled():
                    return _cancelled()
                label = self._item_label(item)
                progress.setLabelText(f"Cropping {step}/{n}: {label}")
                self.status_label.setText(f"Cropping {label}...")
                progress.setValue(step)
                QApplication.processEvents()
                img, header, wcs = self._crop_panel(
                    item["_src_image"], item.get("_src_header"),
                    item.get("_src_wcs"), left, right, top, bottom)
                item["image"] = img
                item["header"] = header
                item["wcs"] = wcs
                print(f"[Mosaic] cropped {label} by "
                      f"L{left} R{right} T{top} B{bottom} -> "
                      f"{img.shape[1]}×{img.shape[0]}")
        except Exception as e:
            self._restore_panel_sources()
            self._sync_list_labels()
            self.status_label.setText("Panel preparation failed.")
            QMessageBox.critical(
                self, "Mosaic Master",
                f"Could not prepare the panels:\n{e}")
            return False
        finally:
            try:
                progress.close()
            except Exception:
                pass
            QApplication.processEvents()

        self._sync_list_labels()
        self._refresh_sip_autodisable()
        return True

    # ---------- Align (Entry Point) ----------
    def align_images(self):
        if len(self.loaded_images) == 0:
            QMessageBox.warning(self, "Align", "No images to align.")
            return

        # Seestar frames: plate solve every frame, place by WCS, average.
        if self._opt_seestar():
            self._build_seestar()
            return

        # Frames added header-only (Seestar mode) need their pixels here.
        for it in self.loaded_images:
            if it.get("image") is None and not self._ensure_loaded(it):
                QMessageBox.warning(self, "Mosaic Master",
                                    f"Could not read {self._item_label(it)}.")
                return

        # Crop first. SRGC needs a WCS, so it runs after plate solving below.
        if not self._preprocess_panels():
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
        #sasprogress dialog. Re-ranged per phase; _tick() also pumps the
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

        self._solve_cancel_event().clear()
        for _i_solve, item in enumerate(images_to_process):
            if not _tick(_i_solve,
                         f"Plate solving {_i_solve + 1}/{len(images_to_process)}: "
                         f"{self._item_label(item)}"):
                _finish()
                self.status_label.setText("Cancelled.")
                return

            # In-house Gaia blind solve, then Astrometry.net if that fails.
            # perform_blind_solve attaches item["wcs"] itself. Rebuilding that
            # header here goes through dict() and rejects the plate solver's
            # own FOCALLEN history card, so a finished solve was thrown away.
            solved_header = self.perform_blind_solve(item)
            if solved_header is _SOLVE_CANCELLED:
                _finish()
                self.status_label.setText("Cancelled.")
                return
            if solved_header is not None and item.get("wcs") is None:
                try:
                    solved_header = self._normalize_wcs_header(
                        solved_header, item["image"].shape)
                    item["wcs"] = self._build_wcs(solved_header, item["image"].shape)
                except Exception as e:
                    print(f"[Mosaic] could not build WCS for "
                          f"{self._item_label(item)}: {e}")
                    item["wcs"] = None
            if item.get("wcs") is None:
                print(f"[Mosaic] Plate solving failed for {self._item_label(item)} ({item.get('path','')}).")

        # newly-solved panels may now carry (or lack) SIP -- refresh the default
        self._refresh_sip_autodisable()

        # ------------------------------------------------------------
        # 1b) SRGC — needs the plate solution from above
        # ------------------------------------------------------------
        if bool(self.srgcEachCheckBox.isChecked()):
            progress.setRange(0, max(1, sum(
                1 for it in self.loaded_images if it.get("wcs") is not None)))
            if not self._srgc_panels(tick=_tick):
                _finish()
                return

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
        did_normalize = bool(self._opt_normalize())

        # --- NEW: detect a mixed linear/stretched panel set ------------------
        # Affine normalization can only reconcile panels on the same tone scale.
        # Linear panels have a pathologically low MAD (~1e-3); a display stretch
        # raises MAD by 1-2 orders of magnitude. We don't use an absolute cutoff
        # (MAD is scale-dependent) — instead we flag when panels DISAGREE, i.e.
        # the spread of MADs across panels is large. A consistent set (all linear
        # or all similarly stretched) has MADs within a small factor of each other.
        if did_normalize and len(wcs_items) > 1:
            mads = []
            for itm in wcs_items:
                try:
                    mads.append((self._item_label(itm), _panel_mad(itm["image"])))
                except Exception:
                    mads.append((self._item_label(itm), 0.0))

            vals = np.array([mv for _, mv in mads if mv > 0], dtype=np.float64)
            if vals.size >= 2:
                mad_ratio = float(np.max(vals) / max(np.min(vals), 1e-9))
                # Ratio > ~8x means at least one panel is on a very different tone
                # scale from the others (the caliha1/caliha2 case is ~130x).
                if mad_ratio > 8.0:
                    med_mad = float(np.median(vals))
                    lo_names = [nm for nm, mv in mads if mv > 0 and mv < med_mad / 3.0]
                    hi_names = [nm for nm, mv in mads if mv > med_mad * 3.0]
                    detail = "\n".join(f"    {nm}: MAD={mv:.4g}" for nm, mv in mads)
                    msg = (
                        "Looks like you've mixed linear and stretched panels.\n\n"
                        "These panels are on very different tone scales "
                        f"(pixel spread differs by {mad_ratio:.0f}×), which usually "
                        "means one was stretched and the others are still linear. "
                        "Normalization can't blend across that — the seams won't "
                        "match.\n\n"
                        "Normally every panel should be either all linear, or all "
                        "stretched the same way.\n\n"
                        f"{detail}\n\n"
                        "Continue anyway?"
                    )
                    reply = QMessageBox.warning(
                        self, "Mixed Panel Scales", msg,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        # _finish() here too -- this early return used to leave
                        # the progress dialog and spinner running forever.
                        _finish()
                        self.status_label.setText("Cancelled — mixed panel scales.")
                        return

        # Pedestal added before warping so the valid-pixel footprint survives
        # every remap/warp (border fill is 0, real data lands near PED).
        self._BG_PED = 1.0
        self._bg_fit_gain = True   # False -> offset-only overlap fit

        self._ref_bg = None
        if did_normalize:
            try:
                ref_raw = wcs_items[0]["image"].astype(np.float32, copy=False)
                ref_valid = _valid_data_mask(ref_raw)
                self._ref_bg, self._ref_sig = estimate_background_level(
                    ref_raw, valid=ref_valid)
                print(f"[Mosaic] reference sky level = {self._ref_bg} "
                      f"(sigma {self._ref_sig}) measured over "
                      f"{100.0 * _valid_frac(ref_valid):.1f}% of "
                      f"{self._item_label(wcs_items[0])}")
                # A sky level of exactly zero means the measurement found no
                # real sky -- every panel would then be shifted onto nothing.
                if not np.any(np.asarray(self._ref_bg, np.float64) > 0):
                    print("[Mosaic] reference sky measured as zero; "
                          "falling back to median match.")
                    self._ref_bg = None
            except Exception as e:
                print(f"[Mosaic] background estimate failed on reference: {e}")
                self._ref_bg = None

        use_bg_match = bool(did_normalize and self._ref_bg is not None)

        # Advanced panel matching: keep every aligned panel (cropped to its
        # footprint, on disk) for one joint solve after the first pass.
        adv = None
        want_adv = bool(self._opt_advanced())
        if want_adv and not use_bg_match:
            print("[Mosaic+] advanced matching needs the sky-level match "
                  "(Normalize on, sky measurable); using the standard match.")
        elif want_adv:
            adv = {"dir": tempfile.mkdtemp(prefix="saspro_mosaic_"), "recs": []}
            print(f"[Mosaic+] advanced panel matching on; panels kept in {adv['dir']}")

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
        wcs_only = bool(self._opt_wcs_only())

        # ------------------------------------------------------------
        # 6) Main loop: normalize -> reproject -> (optional) star align -> accumulate
        # ------------------------------------------------------------
        first_image = True
        PED = float(self._BG_PED)

        # What the result map shows: each panel's footprint, how it was
        # placed, its brightness correction, and how well each seam matched.
        report = {"panels": [], "seams": [], "shape": (mosaic_height, mosaic_width),
                  "mode": ("advanced" if adv is not None else
                           "standard" if use_bg_match else "off"),
                  "sigma": (float(np.mean(np.asarray(self._ref_sig, np.float64)))
                            if use_bg_match and getattr(self, "_ref_sig", None) is not None
                            else None)}
        self._build_report = report

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

            # --- which source pixels are real data? -------------------------
            # A panel that is itself a mosaic has interior no-data (exact 0).
            # The pedestal trick downstream can only see pixels that fall
            # OUTSIDE the warp footprint, so interior holes have to be tracked
            # explicitly, from here all the way through to the coverage mask.
            src_valid = _valid_data_mask(img_lin)
            src_vfrac = _valid_frac(src_valid)
            if src_vfrac < 0.999:
                print(f"[Mosaic] {_name}: {100.0 * (1.0 - src_vfrac):.1f}% of the "
                      f"source frame is no-data (panel is itself a mosaic?)")
            if not src_valid.any():
                print(f"[Mosaic] {_name}: no valid pixels, skipping.")
                continue

            # --- stats kept only for the legacy unstretch path ---
            mono_for_stats = img_lin if img_lin.ndim == 2 else np.mean(img_lin, axis=2)
            _sv = mono_for_stats[src_valid]
            self.stretch_original_mins.append(
                float(np.nanmin(_sv)) if _sv.size else 0.0)
            self.stretch_original_medians.append(
                float(np.nanmedian(_sv)) if _sv.size else 0.0)

            # --- background match: shift this panel's sky onto the reference sky ---
            if use_bg_match:
                lev_i, _ = estimate_background_level(img_lin, valid=src_valid)
                img_lin = (img_lin
                           - self._bg_offset(lev_i, img_lin)
                           + self._bg_offset(self._ref_bg, img_lin)).astype(np.float32, copy=False)
                print(f"[Mosaic] {_name}: sky {lev_i} -> {self._ref_bg}")
            elif did_normalize:
                img_lin = self._normalize_linear(img_lin, float(self._mosaic_target_median),
                                                 valid=src_valid)

            # --- flatten the no-data region to the panel's own sky ------------
            # The sky shift above moved the holes along with everything else;
            # left alone they are a hard step against real data, and Lanczos
            # would ring across that step into pixels we are keeping. Filling
            # them flat removes the step. They are still excluded from the
            # mosaic -- the warped mask below is what does the excluding.
            if src_vfrac < 1.0:
                if np.may_share_memory(img_lin, arr):
                    img_lin = img_lin.copy()
                if img_lin.ndim == 3:
                    C = img_lin.shape[2]
                    if use_bg_match:
                        fv = np.resize(
                            np.asarray(self._ref_bg, np.float32).ravel(), C)
                    else:
                        fv = _masked_median(img_lin, src_valid)
                    img_lin[~src_valid] = fv.astype(np.float32)
                else:
                    fv = (float(np.mean(np.asarray(self._ref_bg, np.float32)))
                          if use_bg_match
                          else float(np.median(img_lin[src_valid])))
                    img_lin[~src_valid] = np.float32(fv)

            # --- carry the footprint through the warp on a pedestal ---
            img_ped = (img_lin + PED).astype(np.float32, copy=False)

            # --- Reprojection mode ---
            mode = self.reprojectModeCombo.currentText()

            if mode.startswith("Fast — Homography"):
                reprojected, cov_warp = self._warp_via_wcs_homography(
                    img_ped, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width),
                    H_cache=self._H_cache, src_mask=src_valid
                )
                reprojected = self._polish_reprojected(
                    reprojected.astype(np.float32, copy=False))

            else:
                # Exact SIP-aware remap. This IS the full spherical transform --
                # dst pixel -> world -> src pixel, per pixel, SIP on the source.
                # Also the default for any unrecognized/legacy mode string.
                reprojected, cov_warp = self._warp_via_wcs_remap_exact(
                    img_ped, itm["wcs"], mosaic_wcs, (mosaic_height, mosaic_width),
                    tile=512, src_mask=src_valid,
                    progress_cb=lambda f, b=_base, i=idx, nm=_name: _tick(
                        b + f, f"Panel {i + 1}/{n_panels} — reprojecting {nm} "
                               f"({f * 100:.0f}%)")
                )
                reprojected = self._polish_reprojected(
                    reprojected.astype(np.float32, copy=False))

            # Any pixel whose warped mask is not essentially 1 drew part of its
            # value from no-data and is not trustworthy.
            MASK_KEEP = 0.999
            cov_warp = np.asarray(cov_warp, np.float32)

            # Pedestal-free grayscale for star detection (astroalign never sees PED)
            cov_r = (((reprojected if reprojected.ndim == 2 else reprojected[..., 0])
                      > (0.5 * PED)) & (cov_warp >= MASK_KEEP))
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

            panel_gain = panel_offset = seam_rec = None
            if skip_star_align:
                align_note = ("WCS placement (SIP distortion)" if panel_has_sip
                              else "WCS placement (star refine off)")
                aligned = reprojected
            else:
                # Pre-alignment footprint (pedestal intact) for the overlap test.
                gray_ped = reprojected[..., 0] if reprojected.ndim == 3 else reprojected
                if first_image:
                    align_note = "reference panel"
                    aligned = reprojected
                else:
                    transform_method = self._opt_transform()
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
                    ov = ((gray_ped > 0.5 * PED) & (cov_warp >= MASK_KEEP)
                          & (self.weight_mosaic > 0))
                    n_ov_stars = int(ov.sum())
                    if n_ov_stars < 2000:
                        print(f"[Mosaic] {self._item_label(itm)}: overlap "
                              f"{n_ov_stars}px too small for star alignment, "
                              f"WCS placement only.")
                        align_note = "WCS placement (overlap too small for star matching)"
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
                            align_note = f"stars ({len(src_pts)} matched)"
                        except Exception as e:
                            print(f"[Mosaic] astroalign fallback on {self._item_label(itm)}: "
                                  f"{type(e).__name__}: {e}")
                            self.status_label.setText(f"Astroalign failed: {e}. Using identity transform.")
                            align_note = "WCS placement (star matching failed)"
                            transform_matrix = np.eye(2, 3, dtype=np.float32)

                    affine_aligned = cv2.warpAffine(
                        reprojected, transform_matrix, (mosaic_width, mosaic_height),
                        flags=cv2.INTER_LANCZOS4
                    )
                    aligned = affine_aligned
                    # The validity mask has to take the same ride, or it stops
                    # describing the image it belongs to.
                    cov_warp = cv2.warpAffine(
                        cov_warp, transform_matrix, (mosaic_width, mosaic_height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

                    if transform_method in ["Homography Transform", "Polynomial Warp Based Transform"]:
                        self.status_label.setText(f"Starting refined alignment using {transform_method}...")
                        QApplication.processEvents()
                        refined_result = self.refined_alignment(
                            affine_aligned, mosaic_gray, method=transform_method,
                            extra_mask=cov_warp, valid=cov_r)
                        if refined_result is not None:
                            aligned, best_inliers2, cov_warp = refined_result
                            align_note += f", {transform_method.split()[0].lower()} refinement ({best_inliers2} inliers)"
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

            # Two independent tests, both required:
            #
            #  * > 0.995*PED catches interpolation that reached past the panel
            #    EDGE into borderValue=0 (a fully covered pixel is PED + data,
            #    and data >= 0).
            #  * cov_warp catches no-data INSIDE the source frame. The pedestal
            #    test is blind to those: the holes sit at exactly PED after the
            #    +PED shift, so they pass it. This is what made a 3-panel mosaic
            #    report 100% coverage of a 2x2 canvas.
            #
            # Lanczos4 spans 8x8, so edge contamination extends ~4px in; erode
            # to clear the ringing band, where overshoot can push a contaminated
            # pixel back above PED.
            #
            # The pedestal test is at HALF the pedestal, not just under it: the
            # sky match shifts a panel whose sky is brighter than the
            # reference's DOWN, and its darkest real pixels (cold pixels, dark
            # nebula noise) go a little negative. At 0.995*PED each of those
            # failed, and the trim below grew it into a ~25 px black dot
            # (GitHub issue #138). Past the edge the warp falls to 0, far
            # below PED/2, and the band where it only partly falls is the one
            # the trim removes.
            EDGE_TRIM = 12
            coverage = (gray_ped > (0.5 * PED)) & (cov_warp >= MASK_KEEP)
            if EDGE_TRIM > 0 and coverage.any():
                coverage = _shrink_valid(coverage, EDGE_TRIM)

            if not coverage.any():
                print(f"[Mosaic] {_name}: empty footprint, skipping.")
                continue
            print(f"[Mosaic] {_name}: coverage {100.0 * _valid_frac(coverage):.1f}% "
                  f"of canvas ({int(coverage.sum())} px)")
            aligned = self._strip_pedestal(aligned, PED, coverage)

            adv_rec = None
            if adv is not None:
                try:
                    adv_rec = self._adv_store_panel(adv, idx, _name, aligned, coverage)
                except Exception as e:
                    print(f"[Mosaic+] could not store {_name} ({type(e).__name__}: {e}); "
                          f"advanced matching off")
                    shutil.rmtree(adv["dir"], ignore_errors=True)
                    adv = None

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
                    S0, R0 = _overlap_blocks(aligned, mos_est, overlap)
                    a, b, n = solve_linear_match(aligned, mos_est, overlap,
                                                 fit_scale=self._bg_fit_gain,
                                                 pivot=self._ref_bg)
                    pv = self._bg_offset(self._ref_bg, aligned)
                    aligned = ((aligned - pv) * self._bg_offset(a, aligned)
                               + pv + self._bg_offset(b, aligned)
                               ).astype(np.float32, copy=False)
                    aligned[~coverage] = 0.0
                    panel_gain = float(np.mean(a))
                    panel_offset = float(np.mean(b))
                    try:
                        S1, R1 = _overlap_blocks(aligned, mos_est, overlap)
                        if S0 is not None and S1 is not None:
                            seam_rec = {"i": len(report["panels"]), "j": None,
                                        "poly": coverage_polygon(overlap),
                                        "before": float(np.median(np.abs(S0 - R0))),
                                        "after": float(np.median(np.abs(S1 - R1))), "n": int(n)}
                    except Exception as e:
                        print(f"[Mosaic] seam measurement skipped: {e}")
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
            if adv is not None and adv_rec is not None:
                y0_, y1_, x0_, x1_ = adv_rec["bbox"]
                np.save(adv_rec["smooth_path"],
                        np.ascontiguousarray(smooth_mask[y0_:y1_, x0_:x1_]))
                adv["recs"].append(adv_rec)

            _tick(_base + 3, f"Panel {idx + 1}/{n_panels} — blending {_name}")

            # --- Accumulate ---
            if is_color:
                self.final_mosaic += aligned * smooth_mask[..., np.newaxis]
            else:
                self.final_mosaic += gray_aligned * smooth_mask
            self.weight_mosaic += smooth_mask

            report["panels"].append({
                "idx": next((k for k, x in enumerate(self.loaded_images) if x is itm), None),
                "name": _name, "poly": coverage_polygon(coverage),
                "align": align_note, "gain": panel_gain, "offset": panel_offset,
                "ref": not report["panels"]})
            if seam_rec is not None and report["sigma"] is not None:
                report["seams"].append(seam_rec)

            # Flip only once something is actually ON the canvas. Flipping it
            # inside the alignment branch (as before) let panel 0 fall into the
            # overlap-refinement block below and print a misleading
            # "overlap 0 px" against an all-zero weight map.
            first_image = False

            _tick(_base + SUBSTEPS, f"Panel {idx + 1}/{n_panels} done — {_name}")

        if adv is not None:
            try:
                if not self._adv_rematch(adv, is_color, _tick):
                    _finish()
                    self.status_label.setText("Cancelled.")
                    return
            except Exception as e:
                traceback.print_exc()
                print(f"[Mosaic+] advanced matching failed ({type(e).__name__}: {e}); "
                      f"keeping the standard match")
            finally:
                adv["recs"] = []
                gc.collect()
                shutil.rmtree(adv["dir"], ignore_errors=True)

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

        # ------------------------------------------------------------
        # 9) Re-solve the finished canvas against Gaia DR3
        # ------------------------------------------------------------
        # Runs before the preview opens so the pushed/saved mosaic carries the
        # catalogue-tied solution, not the geometric one it was built from.
        gaia_ok = False
        if getattr(self, "gaiaSolveCheckBox", None) and self.gaiaSolveCheckBox.isChecked():
            _tick(n_panels * SUBSTEPS + 1, "Plate solving the finished mosaic…")
            gaia_ok = self._gaia_solve_final_mosaic(
                say=lambda m: _tick(text=m))

        self.status_label.setText("Final Mosaic Ready.")
        QApplication.processEvents()

        display_image = (np.stack([self.final_mosaic] * 3, axis=-1)
                        if self.final_mosaic.ndim == 2 else self.final_mosaic)
        display_image = self._autostretch_if_requested(display_image)

        _tick(n_panels * SUBSTEPS + 1, "Final Mosaic Ready.")
        _finish()

        self._show_result(display_image,
                          "Final Mosaic (WCS-only)" if wcs_only else "Final Mosaic")

        self.status_label.setText(
            f"Final mosaic ready — {n_panels} panels, "
            f"{mosaic_width}×{mosaic_height}"
            + (", plate solved against Gaia DR3." if gaia_ok else "."))

      
    # ---------- Final plate solve (in-house Gaia DR3) ----------
    def _gaia_solve_final_mosaic(self, say=None):
        """Re-solve the finished canvas against the local Gaia DR3 catalogue
        and adopt that solution as the mosaic's WCS.

        Why bother, when the mosaic WCS is already exact? Because "exact" here
        means exact for the *projection*, not for the sky. Every panel was
        reprojected into a frame derived from the reference panel's own
        solution, so the whole canvas inherits that one panel's pointing error
        and its SIP residuals. Fitting the blended result against the catalogue
        uses stars from every panel at once, over a far larger baseline than
        any single tile, and ties the answer to Gaia rather than to a tile.
        It also attaches real SOLVRMS/SOLVP95/SOLVNSTR cards, which the
        geometric header has no way to provide.

        The existing mosaic WCS is handed over as the seed, so this is a
        refinement from a known-good starting point rather than a search.

        Returns True if a new solution was adopted. On any failure the
        geometric WCS is left exactly as it was -- a worse solve never
        replaces a good header.
        """
        def _say(msg):
            print(f"[Mosaic] {msg}")
            try:
                self.status_label.setText(msg)
                QApplication.processEvents()
            except Exception:
                pass
            if say is not None:
                try:
                    say(msg)
                except Exception:
                    pass

        if self.final_mosaic is None:
            return False

        # Imported lazily: plate_solver imports from star_alignment, which is
        # the module this one was split out of, so a top-level import here
        # would close a cycle at interpreter start.
        try:
            from setiastro.saspro.plate_solver import _solve_with_GAIA_threaded
        except Exception as e:
            _say(f"Gaia solve unavailable ({type(e).__name__}: {e}); "
                 f"keeping the geometric WCS.")
            return False

        seed = None
        if self.wcs_metadata:
            try:
                seed = sanitize_wcs_header(self.wcs_metadata) or \
                       coerce_to_header(self.wcs_metadata)
            except Exception:
                seed = None
        if seed is not None:
            self._ensure_image_naxis(seed, self.final_mosaic.shape)

        img = np.ascontiguousarray(self.final_mosaic.astype(np.float32, copy=False))
        h, w = img.shape[:2]
        _say(f"Plate solving the finished mosaic ({w}×{h}) against Gaia DR3…")

        # The mosaic WCS is the best seed this solver will ever be handed, so
        # do not let a stale "force blind" checkbox or a "none" seed mode from
        # an earlier single-frame solve throw it away. Saved and restored, in
        # the same style as the ASTAP passes in plate_solver itself.
        s = self.settings
        _orig_seed_mode = s.value("astap/seed_mode", "auto", type=str)
        _orig_force_blind = s.value("plate_solver/force_blind", False)
        solved = None
        try:
            if seed is not None:
                s.setValue("astap/seed_mode", "auto")
                s.setValue("plate_solver/force_blind", False)
                s.sync()

            ok, res = _solve_with_GAIA_threaded(
                self, s, img, seed, force_blind=(seed is None))

            if not ok and seed is not None:
                # The seed was perfect by construction, so a failure means the
                # fit itself could not close -- not that the pointing was
                # wrong. Still worth one blind pass off the quad index: a
                # mosaic is a big, star-rich field, which is the case blind
                # solving handles best.
                _say(f"Seeded Gaia solve failed ({res}); retrying blind…")
                ok, res = _solve_with_GAIA_threaded(self, s, img, None,
                                                    force_blind=True)
            if ok and isinstance(res, Header):
                solved = res
        except Exception as e:
            import traceback
            traceback.print_exc()
            _say(f"Gaia solve raised {type(e).__name__}: {e}; "
                 f"keeping the geometric WCS.")
            return False
        finally:
            try:
                s.setValue("astap/seed_mode", _orig_seed_mode)
                s.setValue("plate_solver/force_blind", _orig_force_blind)
                s.sync()
            except Exception:
                pass
            # Belt and braces: the solver's status popup is normally closed by
            # _solve_numpy_with_fallback, which we deliberately bypass to get
            # the Gaia pass on its own. If status text ever fails to reach a
            # label, that popup opens modelessly and nothing else would shut
            # it, leaving a "Plate Solving" window stuck on top.
            try:
                from setiastro.saspro.plate_solver import _status_popup_close
                _status_popup_close()
            except Exception:
                pass

        if solved is None:
            _say("Gaia solve did not converge; keeping the geometric WCS.")
            return False

        # Sanity-check before adopting. A solution that lands somewhere other
        # than where the geometry says the canvas points is a mis-solve, and
        # the geometric header is the safer of the two.
        if seed is not None:
            try:
                w_old = get_wcs_from_header(seed)
                w_new = get_wcs_from_header(solved)
                if w_old is not None and w_new is not None:
                    ra0, dec0 = w_old.all_pix2world([w / 2.0], [h / 2.0], 0)
                    ra1, dec1 = w_new.all_pix2world([w / 2.0], [h / 2.0], 0)
                    dra = ((float(ra1[0]) - float(ra0[0]) + 180.0) % 360.0 - 180.0)
                    dra *= np.cos(np.radians(float(dec0[0])))
                    sep = float(np.hypot(dra, float(dec1[0]) - float(dec0[0])))
                    _say(f"Gaia solution centre differs from the geometric "
                         f"centre by {sep * 60.0:.2f}'")
                    if sep > 1.0:
                        _say("That is more than a degree — treating the solve "
                             "as a mis-match and keeping the geometric WCS.")
                        return False
            except Exception as e:
                print(f"[Mosaic] centre comparison skipped: {e}")

        self._ensure_image_naxis(solved, self.final_mosaic.shape)
        self.wcs_metadata = solved

        bits = []
        for k, label in (("SOLVRMS", "RMS"), ("SOLVP95", "p95"),
                         ("SOLVNSTR", "stars")):
            try:
                if k in solved:
                    v = solved[k]
                    bits.append(f"{label}={v}\"" if k != "SOLVNSTR"
                                else f"{label}={int(v)}")
            except Exception:
                pass
        _say("Mosaic plate solved against Gaia DR3"
             + (f" — {', '.join(bits)}" if bits else "") + ".")
        return True

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
        if self.spinnerMovie:
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
                # the two image axes only: a colour FITS (NAXIS3 = 3) would
                # otherwise give a 3-axis WCS and the 2-D lookup fails
                wcs = WCS(header, naxis=2)
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

            No-data pixels are excluded from the background statistics, so the
            count reflects the real frame rather than the size of its holes.
            """
            from photutils.detection import DAOStarFinder
            valid = _valid_data_mask(image, erode_px=0)
            if image.ndim == 3:
                image = np.mean(image, axis=2)
            if not valid.any():
                return 0

            ignore = ~valid
            mean_val, median_val, std_val = sigma_clipped_stats(
                image, sigma=3.0, mask=ignore)
            if not np.isfinite(std_val) or std_val <= 0:
                return 0
            daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std_val)
            sources = daofind(image - median_val, mask=ignore)

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
            try:
                progress.close()
            except Exception:
                pass
            if self.spinnerMovie:
                self.spinnerMovie.stop()
            self.spinnerLabel.hide()
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
    
        # Median over real data only -- the base frame may itself carry no-data.
        _base_valid = _valid_data_mask(base_img, erode_px=0)
        base_median = float(np.median(base_img[_base_valid])
                            if _base_valid.any() else np.median(base_img))

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
                #sasprocess the image to remove the gradient
                new_img = poly_remover.process()
                print("[DEBUG] Finished polynomial gradient removal on subframe.")

                # If mono:
                self.status_label.setText(f"Normalizing {item['path']}.")
                QApplication.processEvents()
                # Minimum over real data, and keep no-data at zero afterwards:
                # subtracting a global min that came from a hole is a no-op,
                # but subtracting a real sky floor would lift the holes to a
                # non-zero value and they would stop reading as no-data.
                _nv = _valid_data_mask(new_img, erode_px=0)
                if _nv.any():
                    new_img = new_img - float(np.min(new_img[_nv]))
                    if not _nv.all():
                        new_img = np.clip(new_img, 0.0, None)
                        new_img[~_nv, ...] = 0.0
                else:
                    new_img = new_img - np.min(new_img)
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
        #sasprocess each subsequent image once
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
        if self.spinnerMovie:
            self.spinnerMovie.stop()
        self.spinnerLabel.hide()
        self.status_label.setText("Seestar Mode alignment (sum/divide) complete.")
        QApplication.processEvents()

        # Seestar mosaics are built by star matching alone, so the result has
        # no WCS at all. A blind Gaia solve is the only way this canvas ever
        # gets one -- self.wcs_metadata is None here, so _gaia_solve_final_mosaic
        # goes straight to the quad index.
        if getattr(self, "gaiaSolveCheckBox", None) and self.gaiaSolveCheckBox.isChecked():
            self._gaia_solve_final_mosaic()

        display_image = self.stretch_for_display(self.final_mosaic)
        self._show_result(display_image, "Robot Telescope Mosaic")

    # ---------- Star alignment using triangle matching ----------

    def refined_alignment(self, affine_aligned, mosaic_img,
                          method="Homography Transform", extra_mask=None,
                          valid=None):
        """
        Refined alignment that assumes affine_aligned is the result of the affine alignment step.
        It re-detects stars in the candidate (overlap) region and computes a refined transform from
        affine_aligned to mosaic_img. Then it applies the refined transform to affine_aligned and
        returns the fully warped image.

        `extra_mask` is the caller's validity map for affine_aligned. It is put
        through the identical warp and handed back, so the caller's notion of
        which pixels are real data keeps describing the image it belongs to.
        `valid` restricts star detection to real data on the panel side.

        Returns:
        - (warped_image, inlier_count, warped_extra_mask)
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
        if valid is not None:
            # Honour the caller's real-data map: a no-data region lifted to sky
            # level is > 0 and would otherwise be mistaken for overlap.
            overlap_mask &= np.asarray(valid, bool)

        # Detect stars. Pass the overlap as the statistics mask too -- the
        # zeroed-out remainder of the canvas would otherwise dominate the
        # sigma-clipped background and wreck the detection threshold.
        self.status_label.setText("Refinement: Detecting stars in mosaic and affine-aligned images...")
        QApplication.processEvents()
        # Increase max_stars to 50 for debugging purposes.
        # Both sides at ONE threshold, the lower of the two each would pick on
        # its own, so the two lists reach the same depth and their fluxes
        # (normalized by the median for matching) describe the same stars.
        mosaic_in = np.where(overlap_mask, mosaic_gray, 0)
        new_in = np.where(overlap_mask, affine_aligned_gray, 0)
        mosaic_stars_masked, sig_m = self.detect_stars(
            mosaic_in, max_stars=300, valid=overlap_mask, with_sigma=True)
        new_stars_aligned, sig_n = self.detect_stars(
            new_in, max_stars=300, valid=overlap_mask, with_sigma=True)
        if sig_m is not None and sig_n is not None and sig_m != sig_n:
            if sig_m > sig_n:
                mosaic_stars_masked, sig_m = self.detect_stars(
                    mosaic_in, max_stars=300, valid=overlap_mask, sigma=sig_n, with_sigma=True)
            else:
                new_stars_aligned, sig_n = self.detect_stars(
                    new_in, max_stars=300, valid=overlap_mask, sigma=sig_m, with_sigma=True)
        print(f"Star detection threshold: {sig_m}σ / {sig_n}σ")
        del mosaic_in, new_in

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
            warped_mask = extra_mask
            if extra_mask is not None:
                warped_mask = cv2.warpPerspective(
                    np.asarray(extra_mask, np.float32), H_refined,
                    (affine_aligned.shape[1], affine_aligned.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            return (warped_image, inliers, warped_mask)
        elif method == "Polynomial Warp Based Transform":
            self.status_label.setText("Refinement: Computing Polynomial Warp based transform...")
            QApplication.processEvents()
            # Extract control points from matches.
            src_pts = np.float32([match[0][:2] for match in matches])
            dst_pts = np.float32([match[1][:2] for match in matches])
            # Call the static Polynomial Warp warp function.
            try:
                warped_image = MosaicMasterDialog.poly_warp(affine_aligned, src_pts, dst_pts, degree=poly_degree, status_label=self.status_label)
                warped_mask = extra_mask
                if extra_mask is not None:
                    warped_mask = MosaicMasterDialog.poly_warp(
                        np.asarray(extra_mask, np.float32), src_pts, dst_pts,
                        degree=poly_degree, status_label=self.status_label,
                        interpolation=cv2.INTER_LINEAR)
                inliers = len(matches)
                self.status_label.setText(f"Refinement: Polynomial Warp warp applied with {inliers} matched points.")
                return (warped_image, inliers, warped_mask)
            except Exception as e:
                self.status_label.setText(f"Refinement: Polynomial Warp warp failed: {e}")
                return None
        else:
            self.status_label.setText("Refinement: Unexpected transformation method.")
            return None

    @staticmethod
    def poly_warp(image, src_pts, dst_pts, degree=3, status_label=None,
                  interpolation=cv2.INTER_LANCZOS4):
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
        warped = cv2.remap(image, map_u, map_v, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
        
        if status_label is not None:
            status_label.setText("Polynomial warp: Image warped.")
            QApplication.processEvents()
        
        return warped



    def stretch_for_display(self, arr):
        """
        Uses your global stretch_mono_image or stretch_color_image to produce
        a display-ready 8-bit image. For color images, uses unlinked stretching.

        A mosaic canvas is rarely full -- the corners of a rotated field and
        any unfilled panel slot are hard zeros. The stretch functions take no
        mask, and a median computed with those zeros included lands far below
        the real sky, so the stretch over-brightens everything and the preview
        washes out white. Substitute the valid median into the holes for the
        duration of the stretch, then put them back to black.
        """
        arr = arr.astype(np.float32)

        valid = _valid_data_mask(arr, erode_px=0)
        holes = None
        if valid.any() and not valid.all():
            holes = ~valid
            arr = arr.copy()
            if arr.ndim == 3:
                for c in range(arr.shape[2]):
                    arr[..., c][holes] = np.median(arr[..., c][valid])
            else:
                arr[holes] = np.median(arr[valid])

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
        if holes is not None:
            # No-data reads as black, not as mid-grey sky.
            if disp.ndim == 3:
                disp[holes, :] = 0
            else:
                disp[holes] = 0
        return disp


    def detect_stars(self, image2d, max_stars=50, valid=None, sigma=None, with_sigma=False):
        """Detect the brightest stars in `image2d`: [(x, y, flux), ...],
        brightest first, at most `max_stars`.

        The threshold is chosen by star count, not set by hand: the highest
        rung of sep_tiled.STAR_LADDER (100 sigma down to 3) that finds at least
        the user's "Stars to detect" (sep_tiled.waterfall_extract). `sigma`
        forces one threshold instead. With `with_sigma`, returns
        (stars, threshold used); the threshold is None when nothing could be
        measured.

        `valid` marks the real-data pixels; everything else is masked out of
        the background and the detection, so the zeroed remainder of a mosaic
        canvas can't drag the background down. It defaults to the frame's
        non-zero pixels.
        """
        import sep
        from setiastro.saspro.sep_tiled import waterfall_extract
        none = ([], None) if with_sigma else []

        image2d = np.asarray(image2d, dtype=np.float32)
        if valid is None:
            valid = _valid_data_mask(image2d, erode_px=0)
        valid = np.asarray(valid, bool)
        if not valid.any():
            return none
        data = np.where(valid & np.isfinite(image2d), image2d, 0.0).astype(np.float32)

        def _real_stars(cat):
            # hot pixels (a ~0.5 px) and streaks / cosmic rays, as in star alignment
            a = cat["a"]; b = cat["b"]
            return ((2.3548 * a >= 1.2) & (a > 0) & (b > 0)
                    & (1.0 - b / np.maximum(a, 1e-6) <= 0.6))

        target = qs_int(self.settings, "mosaic/star_target", MOSAIC_STAR_TARGET)
        r = waterfall_extract(sep, data, max(1, int(target)), mask=~valid, fixed=sigma,
                              keep=_real_stars)
        if r is None or len(r["cat"]) == 0:
            return ([], (None if r is None else r["sigma"])) if with_sigma else []
        cat = r["cat"]
        order = np.argsort(-np.asarray(cat["flux"]))[:max_stars]
        stars = [(float(cat["x"][i]), float(cat["y"][i]), float(cat["flux"][i])) for i in order]
        return (stars, r["sigma"]) if with_sigma else stars


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

        # Record the real footprint. Re-reading this mosaic as a panel later
        # infers no-data from exact zeros (see _valid_data_mask), which works
        # because the final blend leaves hard zeros outside coverage -- but an
        # explicit map is better than an inference, and survives any later step
        # that adds a pedestal. Consumers that don't know the key ignore it.
        try:
            if self.weight_mosaic is not None:
                cov = (self.weight_mosaic > 0)
                meta["mosaic_coverage"] = cov
                meta["mosaic_coverage_frac"] = float(_valid_frac(cov))
        except Exception as e:
            print(f"[Mosaic] could not attach coverage map: {e}")

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

            # Medians over real data only -- the unfilled parts of the canvas
            # are exact zeros and would drag every median down, so the
            # compensating gain would blow the mosaic out.
            valid = _valid_data_mask(img, erode_px=0)
            if not valid.any():
                return img

            if img.ndim == 2:
                cur_med = float(np.median(img[valid]))
                if cur_med > 0:
                    gain = anchor_med / cur_med
                    img = img * gain
                # keep dynamic range plausible
                img = np.clip(img, 0.0, 1.0)
                img[~valid] = 0.0
                return img

            # color: per-channel median scaling (unlinked)
            for c in range(min(3, img.shape[2])):
                cur_med = float(np.median(img[..., c][valid]))
                # choose a per-channel original median if available; fallback to anchor
                src_med = float(orig_meds[c]) if len(orig_meds) >= 3 else anchor_med
                if cur_med > 0 and src_med > 0:
                    gain = src_med / cur_med
                    img[..., c] *= gain

            img = np.clip(img, 0.0, 1.0)
            img[~valid, ...] = 0.0

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


    # ---------- Blind solve: in-house first, Astrometry.net second ----------
    def _set_solve_status(self, text: str):
        """Status line, and the plate-solve progress dialog when it is up.

        That dialog is window-modal, so it is the only text the user can
        see during a solve that takes minutes.
        """
        try:
            self.status_label.setText(text)
        except Exception:
            pass
        try:
            for dlg in self.findChildren(QProgressDialog):
                if dlg.isVisible():
                    dlg.setLabelText(text)
        except Exception:
            pass
        QApplication.processEvents()
        self._solve_was_cancelled()

    def _solve_cancel_event(self):
        event = getattr(self, "_solve_cancel", None)
        if event is None:
            event = _threading.Event()
            self._solve_cancel = event
        return event

    def _solve_was_cancelled(self) -> bool:
        """True once the plate-solve progress dialog has been cancelled.

        The flag stays set so the in-house worker, which only notices at its
        next status line, and the Astrometry.net poll loop see the same answer.
        """
        event = self._solve_cancel_event()
        if event.is_set():
            return True
        try:
            for dlg in self.findChildren(QProgressDialog):
                if dlg.isVisible() and dlg.wasCanceled():
                    event.set()
                    return True
        except Exception:
            pass
        return False

    def _fits_shape_matches(self, filepath, image_shape) -> bool:
        """True only when `filepath` is a FITS file of the same pixel size.

        A crop lives in memory. Writing that solution back would stamp the
        cropped WCS onto the original full frame.
        """
        if not isinstance(filepath, str) or self._is_view_path(filepath):
            return False
        if not filepath.lower().endswith((".fits", ".fit")) or not os.path.isfile(filepath):
            return False
        ih, iw = int(image_shape[0]), int(image_shape[1])
        try:
            with fits.open(filepath, memmap=False) as hdul:
                data = hdul[0].data
            if data is None:
                return False
            if data.ndim == 2:
                return int(data.shape[0]) == ih and int(data.shape[1]) == iw
            if data.ndim == 3:
                a, b, c = (int(x) for x in data.shape)
                if a <= 4:
                    return b == ih and c == iw
                if c <= 4:
                    return a == ih and b == iw
            return False
        except Exception as e:
            print(f"[Mosaic] could not compare FITS size before writing WCS: {e}")
            return False

    @staticmethod
    def _is_wcs_card(key: str) -> bool:
        ku = str(key).upper()
        if ku in {
            "WCSAXES", "RADESYS", "RADECSYS", "EQUINOX", "LONPOLE", "LATPOLE",
            "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
        }:
            return True
        if ku.startswith(("CTYPE", "CUNIT", "CRPIX", "CRVAL", "CDELT", "CROTA",
                          "CD", "PV", "SOLV", "BLND")):
            return True
        if ku.startswith("PC") and len(ku) > 2 and ku[2].isdigit():
            return True
        return bool(re.match(r"^(A|B|AP|BP)_\d+_\d+$", ku))

    def _write_wcs_cards(self, filepath, solved_header):
        """Replace the WCS in a FITS file with `solved_header`, leaving the pixels."""
        try:
            with fits.open(filepath, mode="update") as hdul:
                hdr = hdul[0].header
                for key in list(hdr.keys()):
                    if self._is_wcs_card(key):
                        try:
                            del hdr[key]
                        except Exception:
                            pass
                src = solved_header if isinstance(solved_header, Header) else Header(solved_header)
                for key in src.keys():
                    if not self._is_wcs_card(key):
                        continue
                    try:
                        hdr[key] = src[key]
                    except Exception:
                        pass
                hdul.flush()
            print(f"[Mosaic] wrote the solved WCS into {os.path.basename(filepath)}.")
        except Exception as e:
            print(f"[Mosaic] could not write WCS back to {filepath}: {e}")

    def _store_solved_header(self, item, solved_header, _calibration_data=None):
        dest = str(item.get("path", "") or "")
        if not self._fits_shape_matches(dest, item["image"].shape):
            print("[Mosaic] Blind solve succeeded; not updating the file on disk "
                  "(view, non-FITS, or the image size no longer matches the file).")
            return
        # Same writer for both solvers. The old Astrometry.net path painted a
        # new TAN solution over whatever SIP was already in the file, and the
        # next load applied both. The header kept in memory is the solution
        # that has to land on disk.
        self._write_wcs_cards(dest, solved_header)

    def _panel_seed_header(self, item):
        """Header kept only for its pixel scale. The position is ignored."""
        seed = item.get("header")
        if isinstance(seed, Header):
            return seed
        if seed is None:
            return None
        return sanitize_wcs_header(seed) or coerce_to_header(seed)

    def _blind_solve_inhouse(self, item):
        """Quad-index blind solve, refined against the local Gaia catalogue.

        Returns a usable header, or None so the caller can try Astrometry.net.
        """
        label = self._item_label(item)
        self._set_solve_status(f"Blind solving {label} with the in-house solver...")
        try:
            from setiastro.saspro.plate_solver import (
                _solve_with_GAIA_threaded, _status_popup_close,
                _wcs_from_header_2d)
        except Exception as e:
            print(f"[Mosaic] in-house blind solver unavailable: {e}")
            return None

        seed = self._panel_seed_header(item)
        try:
            ok, res = _solve_with_GAIA_threaded(
                self, self.settings,
                np.asarray(item["image"]),
                seed if isinstance(seed, Header) else None,
                force_blind=True,
                status_cb=self._set_solve_status,
                cancel_event=self._solve_cancel_event(),
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[Mosaic] in-house blind solve raised {type(e).__name__}: {e}")
            return None
        finally:
            try:
                _status_popup_close()
            except Exception:
                pass

        if self._solve_was_cancelled() or res == "cancelled":
            print(f"[Mosaic] in-house blind solve cancelled for {label}.")
            return _SOLVE_CANCELLED
        if not ok or not isinstance(res, Header):
            print(f"[Mosaic] in-house blind solve failed for {label}: {res}")
            return None

        # Same adoption as the plate solver. Rebuilding the header with
        # dict() + WCS() rejects the FOCALLEN history card the solver writes
        # ("705.0 -> 704.0 mm") and discards a solution that already succeeded.
        try:
            wcs = _wcs_from_header_2d(res, relax=True)
        except Exception as e:
            print(f"[Mosaic] in-house solution for {label} was not usable: {e}")
            return None
        if wcs is None or not getattr(wcs, "is_celestial", False):
            print(f"[Mosaic] in-house solution for {label} was not usable.")
            return None

        item["wcs"] = wcs
        item["header"] = res
        self._store_solved_header(item, res)
        print(f"[Mosaic] {label} solved with the in-house blind solver.")
        self._set_solve_status(f"Blind solved {label} with the in-house solver.")
        return res

    def perform_blind_solve(self, item):
        """
        Blind-solve one panel.

        The in-house Gaia solver runs first (quad index, then a catalogue
        refinement). Astrometry.net is the second choice, used only when
        that fails.
        """
        solved = self._blind_solve_inhouse(item)
        if solved is _SOLVE_CANCELLED:
            return _SOLVE_CANCELLED
        if solved is not None:
            return solved
        if self._solve_was_cancelled():
            return _SOLVE_CANCELLED
        self._set_solve_status(
            f"In-house blind solve failed for {self._item_label(item)}. "
            "Falling back to Astrometry.net...")
        return self._blind_solve_astrometry(item)

    def _blind_solve_astrometry(self, item):
        """
        Blind solve with Astrometry.net and store that TAN solution.

        Any SIP already in the file is removed. The calibration record has no
        distortion of its own, and leaving the old polynomial in place would
        bend the new pointing.
        """
        while True:
            self._set_solve_status("Status: Logging in to Astrometry.net...")
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

            self._set_solve_status("Status: Uploading image to Astrometry.net...")

            # Upload the file itself only when it is still the image we solved.
            # A crop, a view, or a non-FITS source has to go up as a temp FITS
            # of the pixels actually in memory.
            upload_path, created_tmp = self._prepare_astrometry_upload(item)
            if upload_path is None:
                return None

            try:
                subid = self.upload_image_to_astrometry(upload_path, session_key)
                if self._solve_was_cancelled():
                    return _SOLVE_CANCELLED
                if not subid:
                    if QMessageBox.question(self, "Upload Failed",
                                            "Image upload failed or no subid returned. Try again?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                        continue
                    return None

                self._set_solve_status("Status: Waiting for job ID...")
                job_id = self.poll_submission_status(subid)
                if job_id is _SOLVE_CANCELLED:
                    return _SOLVE_CANCELLED
                if not job_id:
                    if QMessageBox.question(self, "Blind Solve Failed",
                                            "Failed to retrieve job ID from Astrometry.net. Try again?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                        continue
                    return None

                self._set_solve_status("Status: Retrieving calibration data...")
                calibration_data = self.poll_calibration_data(job_id)
                if calibration_data is _SOLVE_CANCELLED:
                    return _SOLVE_CANCELLED
                if not calibration_data:
                    if QMessageBox.question(self, "Blind Solve Failed",
                                            "Calibration data did not arrive from Astrometry.net. Try again?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                        continue
                    return None
                break  # success
            finally:
                if created_tmp:
                    try:
                        os.remove(upload_path)
                    except Exception:
                        pass

        # Astrometry.net reports a TAN solution. Writing it has to remove any
        # SIP already in the file; stacking the two is done by _write_wcs_cards.
        wcs_header = self.construct_wcs_header(calibration_data, item["image"].shape)
        wcs_header = self._normalize_wcs_header(wcs_header, item["image"].shape)
        item["wcs"] = self._build_wcs(wcs_header, item["image"].shape)
        item["header"] = wcs_header
        self._store_solved_header(item, wcs_header, calibration_data)
        return wcs_header

    def _prepare_astrometry_upload(self, item):
        """Return (path, created_temp). (None, False) when the temp FITS cannot be written."""
        path = str(item.get("path", "") or "")
        if self._fits_shape_matches(path, item["image"].shape):
            return path, False
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
            tmp.close()
            save_image(
                img_array=item["image"],
                filename=tmp.name,
                original_format="fit",
                bit_depth="16-bit",
                original_header=generate_minimal_fits_header(item["image"]),
                is_mono=item.get("is_mono", False),
            )
            return tmp.name, True
        except Exception as e:
            print(f"[Mosaic] could not write a temporary FITS for Astrometry.net: {e}")
            return None, False


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
            if self._solve_was_cancelled():
                return _SOLVE_CANCELLED
            response = robust_api_request("GET", url)
            if response:
                jobs = response.get("jobs", [])
                if jobs and jobs[0] is not None:
                    return jobs[0]
            print(f"Polling attempt {attempt+1}: Job ID not ready yet.")
            non_blocking_sleep(10)
            if self._solve_was_cancelled():
                return _SOLVE_CANCELLED
        QMessageBox.critical(self, "Blind Solve Failed", "Failed to retrieve job ID from Astrometry.net after multiple attempts.")
        return None

    def poll_calibration_data(self, job_id):
        from setiastro.saspro.main_helpers import non_blocking_sleep
        url = ASTROMETRY_API_URL + f"jobs/{job_id}/calibration/"
        for attempt in range(90):
            if self._solve_was_cancelled():
                return _SOLVE_CANCELLED
            response = robust_api_request("GET", url)
            if response and 'ra' in response and 'dec' in response:
                print("Calibration data retrieved:", response)
                return response
            print(f"Calibration data not available yet (attempt {attempt+1})")
            non_blocking_sleep(10)
            if self._solve_was_cancelled():
                return _SOLVE_CANCELLED
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
        """Write ``wcs_header`` into the file, replacing any previous solution.

        Older code overwrote CRVAL/CD and left A_ORDER / PV in place, so a new
        TAN solution was evaluated with the previous distortion. NAXIS is left
        alone: a colour FITS stays a colour FITS.
        """
        if wcs_header is None:
            print("Not updating FITS: no solved header.")
            return
        self._write_wcs_cards(filepath, wcs_header)

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

        The per-channel min and median are taken over real data only. This
        feeds the temporary FITS handed to ASTAP: on a panel that is itself a
        mosaic the no-data zeros would set the minimum to 0 and halve the
        median, and the resulting over-bright stretch costs the solver stars.
        """
        was_single_channel = False  # Flag to check if image was single-channel

        # Validity from the ORIGINAL array, before the mono->3ch expansion.
        valid = _valid_data_mask(image, erode_px=0)

        # If the image is 2D or has one channel, convert to 3-channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            was_single_channel = True
            image = np.stack([np.squeeze(image)] * 3, axis=-1)

        image = image.astype(np.float32).copy()
        stretched_image = image.copy()
        self.stretch_original_mins = []
        self.stretch_original_medians = []
        target_median = 0.02

        if not valid.any():
            valid = np.ones(stretched_image.shape[:2], bool)

        for c in range(3):
            chan = stretched_image[..., c]
            channel_min = float(np.min(chan[valid]))
            self.stretch_original_mins.append(channel_min)
            stretched_image[..., c] -= channel_min
            channel_median = float(np.median(stretched_image[..., c][valid]))
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