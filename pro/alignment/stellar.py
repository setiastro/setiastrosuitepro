from __future__ import annotations

import os
import math
import random
import sys
import gc
import os as _os
import threading as _threading
import ctypes as _ctypes
import multiprocessing

# ---------------------------------------------------------------------
# Thread control (same as functions.py)
N = str(max(1, min( (os.cpu_count() or 8), 32 )))
os.environ.setdefault("OMP_NUM_THREADS", N)
os.environ.setdefault("OPENBLAS_NUM_THREADS", N)
os.environ.setdefault("MKL_NUM_THREADS", N)
os.environ.setdefault("NUMEXPR_NUM_THREADS", N)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", N)  # macOS Accelerate
try:
    import cv2
    cv2.setNumThreads(int(N))
except Exception:
    pass

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from itertools import combinations
from typing import Callable, Iterable, Tuple
import tempfile
import traceback
import requests
import numpy as np

import astroalign
import sep
import re
import warnings
import json
import time
from scipy.spatial import KDTree, Delaunay
from astropy.stats import sigma_clipped_stats
from astropy.io.fits import Header
from photutils.detection import DAOStarFinder
from astropy.table import vstack
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, get_sun
from astropy.wcs import FITSFixedWarning
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp
from pro.blink_comparator_pro import CustomDoubleSpinBox, CustomSpinBox
import numpy.ma as ma
from PyQt6.QtCore import Qt, QThread, QRunnable, QThreadPool, pyqtSignal, QObject, QTimer, QProcess, QPoint, QEvent, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QAbstractItemView, QListWidget, QInputDialog, QApplication, QProgressBar, QProgressDialog, 
    QRadioButton, QFileDialog, QComboBox, QMessageBox, QTextEdit, QDialogButtonBox, QTreeWidget,QCheckBox, QFormLayout, QListWidgetItem, QScrollArea, QTreeWidgetItem, QSpinBox, QDoubleSpinBox
)

from skimage.transform import warp, PolynomialTransform 

from pro.memory_utils import smart_zeros, should_use_memmap
from legacy.image_manager import load_image, save_image
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

try:
    from legacy.numba_utils import fast_star_detect
except Exception:
    fast_star_detect = None

from legacy.numba_utils import (
    rescale_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    rotate_180_numba,
    invert_image_numba,
)
from pro.abe import _generate_sample_points as abe_generate_sample_points

# CORE & FUNCTIONS IMPORTS
from pro.alignment.core import _gray2d, aa_find_transform_with_backoff, IDENTITY_2x3
from pro.alignment.functions import (
    _apply_affine_to_pts, _align_prefs, _cap_native_threads_once, _find_main_window_from_child,
    _resolve_doc_and_sw_by_ptr, _doc_from_sw, _warp_like_ref, run_star_alignment_headless,
    compute_pairs_astroalign, handle_shortcut, _fmt_doc_title, _list_open_docs_fallback,
    _doc_image, _active_doc_from_parent, _get_image_from_active_view, _push_image_to_active_view,
    _cap_points
)

# ---------------------------------------------------------------------
# Stellar Alignment (Dialog) — uses Active View or File (no slots)
# ---------------------------------------------------------------------

class StellarAlignmentDialog(QDialog):
    def __init__(self, parent, settings, doc_manager=None, list_open_docs_fn=None):

        super().__init__(parent)
        self.setWindowTitle("Stellar Alignment")
        self.setModal(True)

        self.settings = settings
        self.parent_window = parent
        self._docman = doc_manager or getattr(parent, "doc_manager", None)

        # allow caller to pass same helper used by RGBCombinationDialogPro
        if list_open_docs_fn is None:
            cand = getattr(parent, "_list_open_docs", None)
            self._list_open_docs_fn = cand if callable(cand) else None
        else:
            self._list_open_docs_fn = list_open_docs_fn

        self.stellar_source = None
        self.stellar_target = None
        self.aligned_image = None
        self.stretched_image = None
        self.autostretch_enabled = False
        self.source_was_mono = False
        self.target_was_mono = False

        self.source_file_path = None
        self.target_file_path = None
        self._align_progress_in_slot = False

        self.initUI()

    def initUI(self):
        L = QHBoxLayout(self)

        # left controls
        controls = QVBoxLayout()

        # SOURCE
        src_box = QGroupBox("Source Image (Reference)")
        src_l = QVBoxLayout(src_box)

        src_radios = QHBoxLayout()
        self.source_from_file_radio = QRadioButton("From File")
        self.source_from_view_radio = QRadioButton("From View")
        self.source_from_view_radio.setChecked(True)
        src_radios.addWidget(self.source_from_file_radio)
        src_radios.addWidget(self.source_from_view_radio)
        src_l.addLayout(src_radios)

        src_file_row = QHBoxLayout()
        self.btn_source_file = QPushButton("Select File")
        self.lbl_source_file = QLabel("No file selected")
        self.btn_source_file.clicked.connect(self.select_source_file)
        src_file_row.addWidget(self.btn_source_file)
        src_file_row.addWidget(self.lbl_source_file)
        src_l.addLayout(src_file_row)

        # view picker (any view – not only active)
        self.source_view_combo = QComboBox()
        src_view_row = QHBoxLayout()
        src_view_row.addWidget(self.source_view_combo, 1)
        btn_src_refresh = QPushButton("Refresh")
        btn_src_refresh.clicked.connect(self._populate_view_combos)
        src_view_row.addWidget(btn_src_refresh)
        src_l.addLayout(src_view_row)

        controls.addWidget(src_box)

        # TARGET
        tgt_box = QGroupBox("Target Image (To be Aligned)")
        tgt_l = QVBoxLayout(tgt_box)

        tgt_radios = QHBoxLayout()
        self.target_from_file_radio = QRadioButton("From File")
        self.target_from_view_radio = QRadioButton("From View")
        self.target_from_view_radio.setChecked(True)
        tgt_radios.addWidget(self.target_from_file_radio)
        tgt_radios.addWidget(self.target_from_view_radio)
        tgt_l.addLayout(tgt_radios)

        tgt_file_row = QHBoxLayout()
        self.btn_target_file = QPushButton("Select File")
        self.lbl_target_file = QLabel("No file selected")
        self.btn_target_file.clicked.connect(self.select_target_file)
        tgt_file_row.addWidget(self.btn_target_file)
        tgt_file_row.addWidget(self.lbl_target_file)
        tgt_l.addLayout(tgt_file_row)

        self.target_view_combo = QComboBox()
        tgt_view_row = QHBoxLayout()
        tgt_view_row.addWidget(self.target_view_combo, 1)
        btn_tgt_refresh = QPushButton("Refresh")
        btn_tgt_refresh.clicked.connect(self._populate_view_combos)
        tgt_view_row.addWidget(btn_tgt_refresh)
        tgt_l.addLayout(tgt_view_row)

        controls.addWidget(tgt_box)

        xform_box = QGroupBox("Transform / Distortion")
        xf = QFormLayout(xform_box)

        self.xf_model = QComboBox()
        self.xf_model.addItems([
            "Affine (fast)",
            "Homography (projective)",
            "Polynomial (order 3)",
            "Polynomial (order 4)",
        ])
        # map saved value to index
        prefs = _align_prefs(self.settings)
        _model = prefs["model"]
        idx = 0 if _model=="affine" else 1 if _model=="homography" else 2 if _model=="poly3" else 3
        self.xf_model.setCurrentIndex(idx)
        xf.addRow("Model:", self.xf_model)

        self.xf_maxcp = QSpinBox(); self.xf_maxcp.setRange(20, 2000); self.xf_maxcp.setValue(prefs["max_cp"])
        xf.addRow("Max control points:", self.xf_maxcp)

        self.xf_downsample = QSpinBox(); self.xf_downsample.setRange(1, 8); self.xf_downsample.setValue(prefs["downsample"])
        xf.addRow("Solve downsample:", self.xf_downsample)

        self.xf_h_reproj = QDoubleSpinBox(); self.xf_h_reproj.setRange(0.1, 10.0); self.xf_h_reproj.setDecimals(2)
        self.xf_h_reproj.setValue(prefs["h_reproj"])
        xf.addRow("Homog. RANSAC reproj (px):", self.xf_h_reproj)

        def _toggle_rows():
            is_h = (self.xf_model.currentIndex() == 1)  # 1 = Homography
            # Enable/disable only the control…
            self.xf_h_reproj.setEnabled(is_h)
            # …and its label in the form layout (if present)
            lab = xf.labelForField(self.xf_h_reproj)
            if lab is not None:
                lab.setEnabled(is_h)
        _toggle_rows()
        self.xf_model.currentIndexChanged.connect(lambda _ : _toggle_rows())

        controls.addWidget(xform_box)

        # run + status
        self.btn_run = QPushButton("Run Alignment")
        self.btn_run.clicked.connect(self.run_alignment)
        controls.addWidget(self.btn_run)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.status_label)

        L.addLayout(controls)

        # right: preview + actions
        right = QVBoxLayout()
        grp = QGroupBox("Aligned Image")
        rg = QVBoxLayout(grp)

        self.result_preview_label = QLabel()
        self.result_preview_label.setFixedSize(400, 400)
        self.result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rg.addWidget(self.result_preview_label)

        self.btn_autostretch = QPushButton("AutoStretch: OFF")
        self.btn_autostretch.clicked.connect(self.toggle_autostretch)
        rg.addWidget(self.btn_autostretch)

        # actions
        actions_row = QHBoxLayout()
        self.btn_apply_active = QPushButton("Apply to Active View")
        self.btn_create_view = QPushButton("Create New View")
        self.btn_apply_active.clicked.connect(self.apply_to_active_view)
        self.btn_create_view.clicked.connect(self.create_new_view)
        actions_row.addWidget(self.btn_apply_active)
        actions_row.addWidget(self.btn_create_view)
        rg.addLayout(actions_row)

        right.addWidget(grp)
        L.addLayout(right)

        # populate combos initially
        self._populate_view_combos()

    def _persist_xform_from_dialog(self):
        idx = self.xf_model.currentIndex()
        model = "affine" if idx==0 else ("homography" if idx==1 else ("poly3" if idx==2 else "poly4"))
        s = self.settings
        s.setValue("stacking/align/model", model)
        s.setValue("stacking/align/max_cp", int(self.xf_maxcp.value()))
        s.setValue("stacking/align/downsample", int(self.xf_downsample.value()))
        s.setValue("stacking/align/h_reproj", float(self.xf_h_reproj.value()))

    # ------------------------
    # Source/Target loaders (File / Active View)
    # ------------------------
    # inside StellarAlignmentDialog
    def _iter_docs(self) -> list[tuple[str, object]]:
        if self._list_open_docs_fn:
            try:
                return [(str(t), d) for (t, d) in self._list_open_docs_fn()]
            except Exception:
                pass
        return _list_open_docs_fallback(self.parent_window)

    def _populate_view_combos(self):
        items = self._iter_docs()

        def fill(cmb: QComboBox):
            cmb.blockSignals(True)
            cmb.clear()
            for title, doc in items:
                cmb.addItem(title, userData=doc)
            cmb.blockSignals(False)

        fill(self.source_view_combo)
        fill(self.target_view_combo)


    def load_source_from_view(self):
        doc = self.source_view_combo.currentData()
        img = _doc_image(doc)
        if img is None:
            QMessageBox.warning(self, "Error", "Selected source view has no image.")
            return
        self.source_was_mono = (img.ndim == 2)
        if self.source_was_mono:
            img = np.stack([img]*3, axis=-1)
        self.stellar_source = img
        self.lbl_source_file.setText(self.source_view_combo.currentText())

    def load_target_from_view(self):
        doc = self.target_view_combo.currentData()
        img = _doc_image(doc)
        if img is None:
            QMessageBox.warning(self, "Error", "Selected target view has no image.")
            return
        self.target_was_mono = (img.ndim == 2)
        if self.target_was_mono:
            img = np.stack([img]*3, axis=-1)
        self.stellar_target = img
        self.lbl_target_file.setText(self.target_view_combo.currentText())

    def select_source_file(self):
        default_dir = self.settings.value("working_directory", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Source Image", default_dir,
                    "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
        if not path:
            return
        image, header, bit_depth, is_mono = load_image(path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load source image.")
            return
        self.source_was_mono = bool(is_mono or image.ndim == 2)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        self.stellar_source = image
        self.lbl_source_file.setText(os.path.basename(path))

    def select_target_file(self):
        default_dir = self.settings.value("working_directory", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Target Image", default_dir,
                    "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg);;All Files (*)")
        if not path:
            return
        image, header, bit_depth, is_mono = load_image(path)
        if image is None:
            QMessageBox.warning(self, "Error", "Failed to load target image.")
            return
        self.target_was_mono = bool(is_mono or image.ndim == 2)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        self.stellar_target = image
        self.lbl_target_file.setText(os.path.basename(path))

    # ------------------------
    # Preview + stretch
    # ------------------------
    def toggle_autostretch(self):
        if self.aligned_image is None:
            QMessageBox.warning(self, "Stellar Alignment", "No aligned image available.")
            return
        self.autostretch_enabled = not self.autostretch_enabled
        self.btn_autostretch.setText(f"AutoStretch: {'ON' if self.autostretch_enabled else 'OFF'}")
        # recompute stretched version when turning on
        if self.autostretch_enabled:
            self.apply_autostretch()
        # draw
        img = self.stretched_image if self.autostretch_enabled and self.stretched_image is not None else self.aligned_image
        self.update_preview(self.result_preview_label, img)

    def apply_autostretch(self):
        if self.aligned_image is None:
            return
        a = self.aligned_image.astype(np.float32, copy=False)
        m = np.nanmax(a) if a.size else 1.0
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        self.stretched_image = np.clip(a / m, 0, 1)

    def update_preview(self, label, image):
        if image is None:
            return
        disp = image
        if disp.dtype != np.uint8:
            # simple preview scale to 8-bit
            m = float(np.nanmax(disp)) if disp.size else 1.0
            m = m if np.isfinite(m) and m > 0 else 1.0
            disp = np.clip(disp / m * 255.0, 0, 255).astype(np.uint8, copy=False)

        if disp.ndim == 3 and disp.shape[2] == 3:
            h, w, _ = disp.shape
            qimg = QImage(disp.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            h, w = disp.shape[:2]
            qimg = QImage(disp.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled = qimg.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(QPixmap.fromImage(scaled))

    # -----------------------------
    # Triangle helpers (kept from your version)
    # -----------------------------
    @staticmethod
    def compute_triangle_invariant(tri_points):
        d1 = np.linalg.norm(tri_points[0] - tri_points[1])
        d2 = np.linalg.norm(tri_points[1] - tri_points[2])
        d3 = np.linalg.norm(tri_points[2] - tri_points[0])
        sides = sorted([d1, d2, d3])
        if sides[0] == 0:
            return None
        return (sides[1] / sides[0], sides[2] / sides[0])

    @staticmethod
    def build_triangle_dict(coords):
        tri = Delaunay(coords)
        tri_dict = {}
        for simplex in tri.simplices:
            pts = coords[simplex]
            inv = StellarAlignmentDialog.compute_triangle_invariant(pts)
            if inv is None:
                continue
            inv_key = (round(inv[0], 2), round(inv[1], 2))
            tri_dict.setdefault(inv_key, []).append(simplex)
        return tri_dict

    @staticmethod
    def match_triangles(src_dict, tgt_dict, tol=0.1):
        matches = []
        for inv_src, src_tris in src_dict.items():
            for inv_tgt, tgt_tris in tgt_dict.items():
                if abs(inv_src[0] - inv_tgt[0]) < tol and abs(inv_src[1] - inv_tgt[1]) < tol:
                    for s in src_tris:
                        for t in tgt_tris:
                            matches.append((s, t))
        return matches

    @staticmethod
    def ransac_affine(src_coords, tgt_coords, matches, ransac_iter=500, inlier_thresh=3.0, update_callback=None):
        best_inliers = 0
        best_transform = None
        tgt_tree = KDTree(tgt_coords)
        total = ransac_iter
        for i in range(ransac_iter):
            src_tri, tgt_tri = random.choice(matches)
            pts_src = np.float32([src_coords[j] for j in src_tri])
            pts_tgt = np.float32([tgt_coords[j] for j in tgt_tri])
            transform, _ = cv2.estimateAffine2D(pts_src.reshape(-1, 1, 2),
                                                pts_tgt.reshape(-1, 1, 2),
                                                method=cv2.LMEDS)
            if transform is None:
                continue
            src_aug = np.hstack([src_coords, np.ones((src_coords.shape[0], 1))])
            transformed = (transform @ src_aug.T).T
            inliers = 0
            for pt in transformed:
                dist, _ = tgt_tree.query(pt)
                if dist < inlier_thresh:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_transform = np.eye(3, dtype=np.float32)
                best_transform[:2] = transform

            if update_callback is not None and (i % 10 == 0 or i == total - 1):
                progress = int(100 * i / total)
                update_callback(f"RANSAC progress: {progress}% (Best inliers: {best_inliers})")
        return best_transform, best_inliers

    def estimate_transform_ransac(self, source_stars, target_stars):
        src_coords = np.array([[s[0], s[1]] for s in source_stars])
        tgt_coords = np.array([[s[0], s[1]] for s in target_stars])
        self.status_label.setText("Computing Delaunay triangulation...")
        src_tri_dict = self.build_triangle_dict(src_coords)
        tgt_tri_dict = self.build_triangle_dict(tgt_coords)
        self.status_label.setText("Matching triangles...")
        matches = self.match_triangles(src_tri_dict, tgt_tri_dict, tol=0.1)
        if len(matches) == 0:
            self.status_label.setText("No triangle matches found!")
            return None, 0
        self.status_label.setText(f"Found {len(matches)} candidate triangle matches. Running RANSAC...")
        update_callback = lambda msg: self.status_label.setText(msg)
        best_transform, best_inliers = self.ransac_affine(
            src_coords, tgt_coords, matches, ransac_iter=1000, inlier_thresh=3.0, update_callback=update_callback
        )
        return best_transform, best_inliers

    # -----------------------------
    # Astroalign (with backoff) + warp
    # -----------------------------
    def aa_find_transform_with_backoff(self, tgt_gray: np.ndarray, src_gray: np.ndarray):
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
            dict(detection_sigma=5,  min_area=7,  max_control_points=75),
            dict(detection_sigma=12, min_area=9,  max_control_points=75),
            dict(detection_sigma=20, min_area=9,  max_control_points=75),
            dict(detection_sigma=30, min_area=11, max_control_points=75),
            dict(detection_sigma=50, min_area=11, max_control_points=75),
        ]
        last_exc = None
        for kw in tries:
            try:
                global _AA_LOCK
                with _AA_LOCK:
                    return astroalign.find_transform(tgt32, src32, **kw)
            except Exception as e:
                last_exc = e
                if "internal pixel buffer full" in str(e).lower():
                    try:
                        sep.set_extract_pixstack(int(sep.get_extract_pixstack() * 2))
                    except Exception:
                        pass
                continue
        raise last_exc

    def run_alignment(self):
        # Persist dialog choices back to QSettings (safe if the method isn’t present)
        self.status_label.setText("Starting Alignment…")
        QApplication.processEvents()
        try:
            self._persist_xform_from_dialog()
        except Exception:
            pass

        # Ensure sources are loaded
        if self.source_from_view_radio.isChecked() and self.stellar_source is None:
            self.load_source_from_view()
        if self.target_from_view_radio.isChecked() and self.stellar_target is None:
            self.load_target_from_view()

        if self.stellar_source is None:
            QMessageBox.warning(self, "Error", "Please choose a Source (file or view).")
            return
        if self.stellar_target is None:
            QMessageBox.warning(self, "Error", "Please choose a Target (file or view).")
            return

        # Local helpers (self-contained)
        def _cap_points(src_pts: np.ndarray, tgt_pts: np.ndarray, max_cp: int):
            if src_pts.shape[0] <= max_cp:
                return src_pts, tgt_pts
            idx = np.linspace(0, src_pts.shape[0] - 1, max_cp, dtype=int)
            return src_pts[idx], tgt_pts[idx]

        def _estimate_transform_from_pairs(model: str,
                                        src_xy: np.ndarray,
                                        tgt_xy: np.ndarray,
                                        h_reproj: float) -> tuple[str, object]:
            """
            Returns (kind, transform):
            kind="affine"      -> 2x3 float32
            kind="homography"  -> 3x3 float32
            kind="poly3|poly4" -> callable(img, out_hw)->img  (base warp + polynomial residual)
            """
            model = (model or "affine").lower()

            # Base model first (affine or homography)
            if model == "homography":
                H, _ = cv2.findHomography(src_xy, tgt_xy, method=cv2.RANSAC,
                                        ransacReprojThreshold=float(h_reproj))
                if H is None:
                    raise RuntimeError("Homography estimation failed.")
                base_kind, base_X = "homography", np.array(H, dtype=np.float64)
            else:
                A, _ = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC,
                                            ransacReprojThreshold=float(h_reproj))
                if A is None:
                    raise RuntimeError("Affine estimation failed.")
                base_kind, base_X = "affine", np.array(A, dtype=np.float64)

            if model not in ("poly3", "poly4"):
                return base_kind, base_X

            # Predict with base model
            if base_kind == "affine":
                ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
                P = np.hstack([src_xy, ones])
                pred_on_ref = (base_X @ P.T).T
            else:
                ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
                P = np.hstack([src_xy, ones]).T
                Q = (base_X @ P)
                pred_on_ref = (Q[:2, :] / Q[2:3, :]).T

            # Inlier selection for stable poly residual fit
            resid = np.linalg.norm(pred_on_ref - tgt_xy, axis=1)
            r_thresh = max(2.0, h_reproj * 1.5)
            inliers = resid < r_thresh
            if inliers.sum() < 20:
                return base_kind, base_X

            P_ref  = tgt_xy[inliers].astype(np.float32)
            P_pred = pred_on_ref[inliers].astype(np.float32)

            # Normalize to [0,1] domain for conditioning
            Hh, Ww = self.stellar_source.shape[:2]  # or use reference size if you prefer
            scale = np.array([Ww, Hh], dtype=np.float32)
            P_ref_n  = P_ref / scale
            P_pred_n = P_pred / scale

            order = 3 if model == "poly3" else 4
            t_poly = PolynomialTransform()
            ok = t_poly.estimate(P_ref_n, P_pred_n, order=order)  # ref_n -> basewarped_n
            if not ok:
                return base_kind, base_X

            def _warp_poly_residual(img: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
                Hout, Wout = out_hw

                # Pass A: base warp to reference grid
                if base_kind == "affine":
                    if img.ndim == 2:
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = cv2.warpAffine(img, base_X, (Wout, Hout),
                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        # Ensure contiguous channels
                        base_img = np.stack([cv2.warpAffine(np.ascontiguousarray(img[..., c]), base_X, (Wout, Hout),
                                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                            for c in range(img.shape[2])], axis=2)
                else:
                    if img.ndim == 2:
                        if not img.flags['C_CONTIGUOUS']:
                            img = np.ascontiguousarray(img)
                        base_img = cv2.warpPerspective(img, base_X, (Wout, Hout),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    else:
                        # Ensure contiguous channels
                        base_img = np.stack([cv2.warpPerspective(np.ascontiguousarray(img[..., c]), base_X, (Wout, Hout),
                                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                            for c in range(img.shape[2])], axis=2)

                # Pass B: polynomial residual via inverse_map (ref->basewarped), with normalization
                class _InvMap:
                    def __call__(self, coords):
                        coords_n = coords.astype(np.float32) / scale
                        mapped_n = t_poly(coords_n)
                        return mapped_n * scale

                inv = _InvMap()
                try:
                    out = warp(base_img.astype(np.float32, copy=False),
                            inverse_map=inv,
                            output_shape=(Hout, Wout),
                            preserve_range=True,
                            channel_axis=(-1 if base_img.ndim == 3 else None))
                except TypeError:
                    # older skimage: per-channel
                    if base_img.ndim == 2:
                        out = warp(base_img.astype(np.float32), inverse_map=inv,
                                output_shape=(Hout, Wout), preserve_range=True)
                    else:
                        out = np.stack([warp(np.ascontiguousarray(base_img[..., c].astype(np.float32)), inverse_map=inv,
                                            output_shape=(Hout, Wout), preserve_range=True)
                                        for c in range(base_img.shape[2])], axis=2)
                return out.astype(np.float32, copy=False)

            return f"poly{order}", _warp_poly_residual


        # Prepare grayscale for detection
        src = self.stellar_source
        tgt = self.stellar_target
        src_gray = np.mean(src, axis=2) if src.ndim == 3 else src
        tgt_gray = np.mean(tgt, axis=2) if tgt.ndim == 3 else tgt

        # Read dialog prefs
        model = ["affine", "homography", "poly3", "poly4"][self.xf_model.currentIndex()]
        max_cp = int(self.xf_maxcp.value())
        #ds = int(self.xf_downsample.value())
        h_reproj = float(self.xf_h_reproj.value())

        # Downsample for faster star matching (solve stage only)
        #if ds > 1:
        ##    new_ref = (max(1, src_gray.shape[1] // ds), max(1, src_gray.shape[0] // ds))
        #    new_tgt = (max(1, tgt_gray.shape[1] // ds), max(1, tgt_gray.shape[0] // ds))
        #    src_small = cv2.resize(src_gray, new_ref, interpolation=cv2.INTER_AREA)
        #    tgt_small = cv2.resize(tgt_gray, new_tgt, interpolation=cv2.INTER_AREA)
        #else:
        #    src_small, tgt_small = src_gray, tgt_gray
        src_small, tgt_small = src_gray, tgt_gray

        self.status_label.setText("Computing alignment with astroalign…")
        QApplication.processEvents()
        try:
            # NOTE: astroalign returns matched points as (src_pts, tgt_pts)
            #       but we called it with (tgt_small, src_small), so:
            #       src_pts are in tgt_small coords, tgt_pts in src_small coords
            transform_obj, (src_pts_s, tgt_pts_s) = self.aa_find_transform_with_backoff(tgt_small, src_small)
        except Exception as e:
            QMessageBox.warning(self, "Alignment Error", f"Astroalign failed: {e}")
            return

        # Convert to float32 arrays
        src_xy = np.asarray(src_pts_s, dtype=np.float32)
        tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)

        # Cap control points
        src_xy, tgt_xy = _cap_points(src_xy, tgt_xy, max_cp)

        # If we solved on a downsampled pair, re-fit transform at full resolution for accuracy
        #if ds > 1:
        #    src_xy *= ds
        #    tgt_xy *= ds

        # Estimate chosen transform on (possibly rescaled) full-res pairs
        try:
            kind, X = _estimate_transform_from_pairs(model, src_xy, tgt_xy, h_reproj)
        except Exception as e:
            QMessageBox.warning(self, "Alignment Error", f"Transform estimation failed: {e}")
            return

        self.status_label.setText("Warping target image…")
        QApplication.processEvents()
        H, W = src.shape[:2]

        # Apply the transform
        if kind == "affine":
            if tgt.ndim == 2:
                warped_target = cv2.warpAffine(
                    tgt, X, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                warped_target = np.stack(
                    [cv2.warpAffine(tgt[..., i], X, (W, H),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for i in range(tgt.shape[2])],
                    axis=2
                )
            transform_3x3 = np.eye(3, dtype=np.float32); transform_3x3[:2] = X
            self.show_transform_info(transform_3x3)

        elif kind == "homography":
            if tgt.ndim == 2:
                warped_target = cv2.warpPerspective(
                    tgt, X, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                warped_target = np.stack(
                    [cv2.warpPerspective(tgt[..., i], X, (W, H),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for i in range(tgt.shape[2])],
                    axis=2
                )
            # Optional: show homography info as well
            try:
                self.show_transform_info(np.array(X, dtype=np.float64, copy=False))
            except Exception:
                pass

        else:  # polynomial residual callable
            try:
                warped_target = X(tgt, (H, W))
            except Exception as e:
                QMessageBox.warning(self, "Alignment Error", f"Polynomial warp failed ({e}); falling back to affine.")


        # Store + preview (with optional AutoStretch)
        self.aligned_image = warped_target.astype(np.float32, copy=False)
        self.stretched_image = None
        if self.autostretch_enabled:
            self.apply_autostretch()

        disp = self.stretched_image if (self.autostretch_enabled and self.stretched_image is not None) else self.aligned_image
        self.update_preview(self.result_preview_label, disp)
        self.status_label.setText(f"Alignment complete ({model}).")
        QApplication.processEvents()
        QMessageBox.information(self, "Alignment Complete", f"Alignment completed using {model}.")



    def show_transform_info(self, matrix):
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        translation = (tx, ty)
        scale_x = np.sqrt(a * a + c * c)
        rotation_rad = np.arctan2(c, a)
        rotation_deg = np.degrees(rotation_rad)
        shear = (a * b + c * d) / (a * a + c * c) if (a * a + c * c) != 0 else 0.0
        det = a * d - b * c
        scale_y = det / scale_x if scale_x != 0 else 0.0

        info_text = (
            f"Transformation Matrix:\n\n"
            f"[{a:.3f}  {b:.3f}  {tx:.3f}]\n"
            f"[{c:.3f}  {d:.3f}  {ty:.3f}]\n"
            f"[0.000  0.000  1.000]\n\n"
            f"Translation: (tx, ty) = ({tx:.3f}, {ty:.3f})\n"
            f"Scaling: scale_x = {scale_x:.3f}, scale_y = {scale_y:.3f}\n"
            f"Rotation: {rotation_deg:.2f}°\n"
            f"Skew (shear): {shear:.3f}\n"
        )

        info_dialog = QDialog(self)
        info_dialog.setWindowTitle("Transformation Matrix Details")
        layout = QVBoxLayout(info_dialog)

        text_edit = QTextEdit(info_dialog)
        text_edit.setReadOnly(True)
        text_edit.setText(info_text)
        layout.addWidget(text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, info_dialog)
        button_box.accepted.connect(info_dialog.accept)
        layout.addWidget(button_box)

        info_dialog.show()

    def _output_image(self) -> np.ndarray | None:
        if self.aligned_image is None:
            QMessageBox.warning(self, "Stellar Alignment", "No aligned image. Run alignment first.")
            return None
        img = self.aligned_image
        # if original target was mono and we produced a 3-channel array, collapse back
        if self.target_was_mono and img.ndim == 3 and img.shape[2] >= 1:
            img = img[..., 0]
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        return img

    def apply_to_active_view(self):
        img = self._output_image()
        if img is None:
            return
        if not self._docman:
            QMessageBox.warning(self, "Stellar Alignment", "No document manager available.")
            return
        self._docman.update_active_document(
            img, metadata={"description": "Stellar aligned image"},
            step_name="Stellar Alignment"
        )
        QMessageBox.information(self, "Stellar Alignment", "Applied to active view.")
        self.accept()

    def _proposed_title(self) -> str:
        # try to build a friendly title from the source selection
        if self.source_from_view_radio.isChecked():
            doc = self.source_view_combo.currentData()
            try:
                name = doc.display_name() if callable(getattr(doc, "display_name", None)) else getattr(doc, "display_name", None)
                if not isinstance(name, str):
                    name = _fmt_doc_title(doc)
            except Exception:
                name = "Source"
            return f"Aligned_to_{name}"
        if self.source_file_path:
            return f"Aligned_to_{os.path.splitext(os.path.basename(self.source_file_path))[0]}"
        return "Aligned"

    def create_new_view(self):
        img = self._output_image()
        if img is None:
            return
        if not self._docman:
            QMessageBox.warning(self, "Stellar Alignment", "No document manager available.")
            return
        meta = {"step_name": "Stellar Alignment", "description": "Stellar aligned image",
                "is_mono": bool(self.target_was_mono)}
        newdoc = self._docman.open_array(img, metadata=meta, title=self._proposed_title())
        if hasattr(self.parent_window, "_spawn_subwindow_for"):
            self.parent_window._spawn_subwindow_for(newdoc)
        QMessageBox.information(self, "Stellar Alignment", "Created a new view.")
        self.accept()


# ---------------------------------------------------------------------
# Registration (batch) — unchanged behavior; no slots, only files or view as ref
# ---------------------------------------------------------------------
class RegistrationWorkerSignals(QObject):
    progress = pyqtSignal(str)
    result = pyqtSignal(str)
    error = pyqtSignal(str)
    result_transform = pyqtSignal(str, object)  # (orig_file_path, transform_matrix)


# Identity transform (2x3)







def _solve_delta_job(args):
    """
    Worker: compute incremental affine/similarity delta for one frame against the ref preview.
    args = (orig_path, current_transform_2x3, ref_small, Wref, Href,
            resample_flag, det_sigma, limit_stars, minarea,
            model, h_reproj)
    """
    try:
        import os
        import numpy as np
        import cv2
        import sep
        from astropy.io import fits

        (orig_path, current_transform_2x3, ref_small, Wref, Href,
         resample_flag, det_sigma, limit_stars, minarea,
         model, h_reproj) = args

        try:
            cv2.setNumThreads(1)
            try: cv2.ocl.setUseOpenCL(False)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        except Exception:
            pass

        # 1) read → gray float32
        with fits.open(orig_path, memmap=True) as hdul:
            arr = hdul[0].data
            if arr is None:
                return (orig_path, None, f"Could not load {os.path.basename(orig_path)}")
            gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
            gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # 2) pre-warp to REF size
        T_prev = np.asarray(current_transform_2x3, np.float32).reshape(2, 3)
        src_for_match = cv2.warpAffine(
            gray, T_prev, (Wref, Href),
            flags=resample_flag, borderMode=cv2.BORDER_REFLECT_101
        )

        # 3) denoise sparse islands to stabilize AA
        src_for_match = _suppress_tiny_islands(src_for_match, det_sigma=det_sigma, minarea=minarea)
        ref_small     = _suppress_tiny_islands(ref_small,     det_sigma=det_sigma, minarea=minarea)

        # 4) AA incremental delta on cropped ref
        m = (model or "affine").lower()
        if m in ("no_distortion", "nodistortion"):
            m = "similarity"

        if m == "similarity":
            tform = compute_similarity_transform_astroalign_cropped(
                src_for_match, ref_small,
                limit_stars=int(limit_stars) if limit_stars is not None else None,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                h_reproj=float(h_reproj)
            )
        else:
            tform = compute_affine_transform_astroalign_cropped(
                src_for_match, ref_small,
                limit_stars=int(limit_stars) if limit_stars is not None else None,
                det_sigma=float(det_sigma),
                minarea=int(minarea)
            )

        if tform is None:
            return (orig_path, None,
                    f"Astroalign failed for {os.path.basename(orig_path)} – skipping (no transform returned)")

        T_new = np.asarray(tform, np.float64).reshape(2, 3)
        return (orig_path, T_new, None)

    except Exception as e:
        return (args[0] if args else "<unknown>", None,
                f"Astroalign failed for {os.path.basename(args[0]) if args else '<unknown>'}: {e}")



def _residual_job_worker(args):
    """
    Process-safe worker for non-affine residual measurement.
    args = (path, ref_npy, model, h_reproj, det_sigma, minarea, limit_stars)
    Returns: (path, rms_px, err_or_None)
    """
    (path, ref_npy, model, h_reproj, det_sigma, minarea, limit_stars) = args
    try:
        import numpy as np  # re-imports are OK in spawned workers
        from astropy.io import fits

        # Load source (gray, float32, finite)
        with fits.open(path, memmap=True) as hdul:
            arr = hdul[0].data
            if arr is None:
                return (path, float("inf"), "Could not load")
            g = arr if arr.ndim == 2 else np.mean(arr, axis=2)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        # Memmap the shared reference
        ref_small = np.load(ref_npy, mmap_mode="r").astype(np.float32, copy=False)

        # Use the staticmethod that’s importable by workers
        _, _, rms, _ = StarRegistrationThread._aa_model_and_residual(
            g, ref_small, str(model).lower(),
            float(h_reproj), float(det_sigma), int(minarea),
            int(limit_stars) if limit_stars is not None else None
        )
        return (path, float(rms), None)

    except Exception as e:
        return (path, float("inf"), str(e))



def _suppress_tiny_islands(img32: np.ndarray, det_sigma: float, minarea: int) -> np.ndarray:
    """
    Zero out connected components smaller than `minarea`, using
    threshold = det_sigma * global RMS from SEP background.
    Returns float32 image, same shape as input.
    """

    import sep
    import cv2

    img32 = np.asarray(img32, np.float32, order="C")

    # Tame single-pixel spikes so they don't form components
    try:
        img32 = cv2.medianBlur(img32, 3)
    except Exception:
        pass

    # Background + threshold
    bkg = sep.Background(img32, bw=64, bh=64)
    back_img = bkg.back().astype(np.float32, copy=False)   # 2-D local background
    thresh   = float(det_sigma) * float(bkg.globalrms)

    # Candidates above background + thresh
    mask = (img32 > (back_img + thresh)).astype(np.uint8)

    # Label and prune tiny components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return img32

    keep = np.zeros(num, dtype=np.uint8)
    keep[stats[:, cv2.CC_STAT_AREA] >= int(minarea)] = 1
    keep[0] = 0  # background
    pruned = keep[labels]  # 2-D map of components to keep (1) / drop (0)

    # ✅ Replace tiny components with local background using element-wise where
    out = np.where(((mask == 1) & (pruned == 0)), back_img, img32)
    return out.astype(np.float32, copy=False)

# ─────────────────────────────────────────────────────────────
# Final warp+save worker (process-safe)
# ─────────────────────────────────────────────────────────────
def _finalize_write_job(args):
    """
    Process-safe worker: read full-res, compute/choose model, warp, save.
    Returns (orig_path, out_path or "", msg, success, drizzle_tuple or None)

    drizzle_tuple = (kind, matrix_or_None)
    """
    (orig_path, align_model, ref_shape, ref_npy_path,
     affine_2x3, h_reproj, output_directory,
     det_sigma, minarea, limit_stars) = args

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import numpy as np
    from astropy.io import fits
    import cv2

    try:
        cv2.setNumThreads(1)
        try: cv2.ocl.setUseOpenCL(False)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
    except Exception:
        pass

    debug_lines = []
    def dbg(s: str):
        # keep it short-ish; UI emits each line
        debug_lines.append(str(s))

    try:
        # 1) load source (full-res)
        with fits.open(orig_path, memmap=True) as hdul:
            img = hdul[0].data
            hdr = hdul[0].header
        if img is None:
            return (orig_path, "", f"⚠️ Failed to read {os.path.basename(orig_path)}", False, None)

        # Fix for white images: Normalize integer types to [0,1]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        is_mono = (img.ndim == 2)
        src_gray_full = img if is_mono else np.mean(img, axis=2)
        src_gray_full = np.nan_to_num(src_gray_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        img = np.ascontiguousarray(img)

        Href, Wref = ref_shape

        # 2) load reference via memmap
        ref2d = np.load(ref_npy_path, mmap_mode="r").astype(np.float32, copy=False)
        if ref2d.shape[:2] != (Href, Wref):
            return (orig_path, "", f"⚠️ Ref shape mismatch for {os.path.basename(orig_path)}", False, None)

        base = os.path.basename(orig_path)

        # helper: force affine to similarity (no shear)
        def _affine_to_similarity(A2x3: np.ndarray) -> np.ndarray:
            A2x3 = np.asarray(A2x3, np.float64).reshape(2, 3)
            R = A2x3[:, :2]
            t = A2x3[:, 2]
            U, S, Vt = np.linalg.svd(R)
            rot = U @ Vt
            if np.linalg.det(rot) < 0:
                U[:, -1] *= -1
                rot = U @ Vt
            s = float((S[0] + S[1]) * 0.5)
            Rsim = rot * s
            out = np.zeros((2, 3), dtype=np.float64)
            out[:, :2] = Rsim
            out[:, 2] = t
            return out

        # 3) choose transform
        model = (align_model or "affine").lower()
        if model in ("no_distortion", "nodistortion"):
            model = "similarity"

        kind = "affine"
        X = np.asarray(affine_2x3, np.float64).reshape(2, 3)

        if model != "affine":
            # ---- AA pairs (adaptive tiling) ----
            max_cp = None
            try:
                if limit_stars is not None and int(limit_stars) > 0:
                    max_cp = int(limit_stars)
            except Exception:
                max_cp = None

            dbg(f"[finalize] base={base} model={model} det_sigma={det_sigma} minarea={minarea} limit_stars={limit_stars}")

            AA_SCALE = 0.80  # finalize-only

            # ---- tiles=1 (center crop) ----
            src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
                src_gray_full, ref2d,
                scale=AA_SCALE,
                tiles=1,
                det_sigma=float(det_sigma),
                minarea=int(minarea),
                max_control_points=max_cp,
                _dbg=dbg
            )

            if src_xy is None or len(src_xy) < 8:
                dbg("[AA] tiles=1 too few matches")
                raise RuntimeError("astroalign produced too few matches")

            dbg(f"[AA] tiles=1 matches={len(src_xy)} best_tile_xy0={best_xy0}")

            spread_ok1 = _points_spread_ok(tgt_xy, Wref, Href, _dbg=dbg)
            dbg(f"[AA] spread_ok(tiles=1)={spread_ok1}")

            # ---- fallback: tiles=5 (corners + center) ----
            if not spread_ok1:
                src_xy5, tgt_xy5, best_P5, best_xy0_5 = _aa_find_pairs_multitile(
                    src_gray_full, ref2d,
                    scale=AA_SCALE,
                    tiles=5,  # <-- NEW primary fallback
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    max_control_points=max_cp,
                    _dbg=dbg
                )

                if src_xy5 is None or len(src_xy5) < 8:
                    dbg("[AA] tiles=5 too few matches; keeping tiles=1")
                else:
                    dbg(f"[AA] tiles=5 matches={len(src_xy5)} best_tile_xy0={best_xy0_5}")
                    spread_ok5 = _points_spread_ok(tgt_xy5, Wref, Href, _dbg=dbg)
                    dbg(f"[AA] spread_ok(tiles=5)={spread_ok5}")

                    # choose tiles=5 if it spreads better OR gives more matches
                    if spread_ok5 or len(src_xy5) > len(src_xy):
                        dbg("[AA] switching to tiles=5 result")
                        src_xy, tgt_xy = src_xy5, tgt_xy5
                        best_P, best_xy0 = best_P5, best_xy0_5
                    else:
                        dbg("[AA] keeping tiles=1 result (tiles=5 not better)")

            # ---- tertiary fallback: tiles=3 grid ----
            spread_ok_after = _points_spread_ok(tgt_xy, Wref, Href, _dbg=dbg)
            dbg(f"[AA] spread_ok(after tiles=5 check)={spread_ok_after}")

            if not spread_ok_after:
                src_xy3, tgt_xy3, best_P3, best_xy0_3 = _aa_find_pairs_multitile(
                    src_gray_full, ref2d,
                    scale=AA_SCALE,
                    tiles=3,
                    det_sigma=float(det_sigma),
                    minarea=int(minarea),
                    max_control_points=max_cp,
                    _dbg=dbg
                )

                if src_xy3 is None or len(src_xy3) < 8:
                    dbg("[AA] tiles=3 too few matches; keeping current result")
                else:
                    dbg(f"[AA] tiles=3 matches={len(src_xy3)} best_tile_xy0={best_xy0_3}")
                    spread_ok3 = _points_spread_ok(tgt_xy3, Wref, Href, _dbg=dbg)
                    dbg(f"[AA] spread_ok(tiles=3)={spread_ok3}")

                    if spread_ok3 or len(src_xy3) > len(src_xy):
                        dbg("[AA] switching to tiles=3 result")
                        src_xy, tgt_xy = src_xy3, tgt_xy3
                        best_P, best_xy0 = best_P3, best_xy0_3
                    else:
                        dbg("[AA] keeping current result (tiles=3 not better)")

            x0, y0 = best_xy0
            P = np.asarray(best_P, np.float64)

            # ---- base full-ref from best_P + best_xy0 ----
            if P.shape == (3, 3):
                base_kind0 = "homography"
                T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
                base_X0 = T @ P
            else:
                base_kind0 = "affine"
                A3 = np.vstack([P[0:2, :], [0,0,1]])
                T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
                base_X0 = (T @ A3)[0:2, :]

            hth = float(h_reproj)

            if model == "homography":
                H, inl = cv2.findHomography(src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth)
                ninl = int(inl.sum()) if inl is not None else 0
                dbg(f"[RANSAC] homography inliers={ninl}/{len(src_xy)} thr={hth}")

                if H is not None:
                    kind, X = "homography", np.asarray(H, np.float64)
                else:
                    kind, X = base_kind0, base_X0

            elif model == "similarity":
                A, inl = cv2.estimateAffinePartial2D(src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth)
                ninl = int(inl.sum()) if inl is not None else 0
                dbg(f"[RANSAC] similarity inliers={ninl}/{len(src_xy)} thr={hth}")

                if A is not None:
                    kind, X = "similarity", np.asarray(A, np.float64)
                else:
                    if base_kind0 == "affine":
                        kind, X = "similarity", _affine_to_similarity(base_X0)
                    else:
                        kind, X = base_kind0, base_X0

            elif model == "affine":
                kind, X = "affine", np.asarray(affine_2x3, np.float64)

            elif model in ("poly3", "poly4"):
                order = 3 if model == "poly3" else 4
                cx, cy = _fit_poly_xy(src_xy, tgt_xy, order=order)
                map_x, map_y = _poly_eval_grid(cx, cy, Wref, Href, order=order)
                kind, X = model, (map_x, map_y)

            else:
                dbg(f"[AA] unknown model '{model}', falling back to base {base_kind0}")
                kind, X = base_kind0, base_X0

        # 4) warp
        Hh, Ww = Href, Wref

        if kind in ("affine", "similarity"):
            A = np.asarray(X, np.float64).reshape(2, 3)

            if is_mono:
                aligned = cv2.warpAffine(
                    img, A, (Ww, Hh),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                aligned = np.stack([
                    cv2.warpAffine(
                        img[..., c], A, (Ww, Hh),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = ("affine", A.astype(np.float64))
            warp_label = ("similarity" if kind == "similarity" else "affine")

        elif kind == "homography":
            Hm = np.asarray(X, np.float64).reshape(3, 3)

            if is_mono:
                aligned = cv2.warpPerspective(
                    img, Hm, (Ww, Hh),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
            else:
                aligned = np.stack([
                    cv2.warpPerspective(
                        img[..., c], Hm, (Ww, Hh),
                        flags=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0
                    )
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = ("homography", Hm.astype(np.float64))
            warp_label = "homography"

        elif kind in ("poly3","poly4"):
            map_x, map_y = X
            if is_mono:
                aligned = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                aligned = np.stack([
                    cv2.remap(img[...,c], map_x, map_y, cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in range(img.shape[2])
                ], axis=2)

            drizzle_tuple = (align_model, None)
            warp_label = align_model

        if np.isnan(aligned).any() or np.isinf(aligned).any():
            aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

        # 5) save
        name, _ = os.path.splitext(base)
        if name.endswith("_n"):
            name = name[:-2]
        if not name.endswith("_n_r"):
            name += "_n_r"

        out_path = os.path.join(output_directory, f"{name}.fit")

        from legacy.image_manager import save_image as _legacy_save
        _legacy_save(img_array=aligned, filename=out_path, original_format="fit",
                     bit_depth=None, original_header=hdr, is_mono=is_mono)

        msg = (
            f"🌀 Distortion Correction on {base}: warp={warp_label}\n"
            f"💾 Wrote {os.path.basename(out_path)} [{warp_label}]"
        )
        if debug_lines:
            msg = "\n".join(debug_lines) + "\n" + msg
        return (orig_path, out_path, msg, True, drizzle_tuple)

    except Exception as e:
        if debug_lines:
            pre = "\n".join(debug_lines)
            return (orig_path, "", f"{pre}\n⚠️ Finalize error {os.path.basename(orig_path)}: {e}", False, None)
        return (orig_path, "", f"⚠️ Finalize error {os.path.basename(orig_path)}: {e}", False, None)


class StarRegistrationWorker(QRunnable):
    def __init__(self, file_path, original_file, current_transform,
                ref_stars, ref_triangles, output_directory,
                use_triangle=False, use_astroalign=False, reference_image=None,
                downsample_factor: int = 2, model_name: str = "affine"):

        super().__init__()
        self.file_path = file_path
        self.original_file = original_file
        self.current_transform = current_transform if current_transform is not None else IDENTITY_2x3
        self.ref_stars = ref_stars
        self.ref_triangles = ref_triangles
        self.output_directory = output_directory
        self.use_triangle = use_triangle
        self.use_astroalign = use_astroalign
        self.reference_image = reference_image  # 2D reference image
        self.downsample_factor = downsample_factor
        self.signals = RegistrationWorkerSignals()
        self.model_name = str(model_name).lower()

    def run(self):
        """
        Affine:
        - Apply current transform to a preview-sized image
        - Solve incremental delta vs reference preview
        - Emit the incremental delta (2x3) keyed by ORIGINAL path

        Non-affine (homography/poly3/4):
        - This QRunnable does not try to do residuals; it just reports and emits identity.
            The multi-process residual pass is handled by StarRegistrationThread.
        """
        try:
            _cap_native_threads_once()
            try:
                curr = sep.get_extract_pixstack()
                if curr < 1_500_000:
                    sep.set_extract_pixstack(1_500_000)
            except Exception:
                pass

            # --- Load ORIGINAL frame → grayscale float32 ---
            with fits.open(self.original_file, memmap=True) as hdul:
                arr = hdul[0].data
                if arr is None:
                    self.signals.error.emit(f"Could not load {self.original_file}")
                    return
                gray = arr if arr.ndim == 2 else np.mean(arr, axis=2)
                gray_small = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            ref_small = self.reference_image
            if ref_small is None:
                self.signals.error.emit("Worker missing reference preview.")
                return
            Href, Wref = ref_small.shape[:2]

            model = (self.model_name or "affine").lower()

            # --- Non-affine: don't accumulate here; identity + progress line only
            if model in ("homography", "poly3", "poly4"):
                self.signals.progress.emit(
                    f"Residual-only mode for {os.path.basename(self.original_file)} (model={model}); "
                    "emitting identity transform (handled by thread pass)."
                )
                self.signals.result_transform.emit(os.path.normpath(self.original_file), IDENTITY_2x3.copy())
                self.signals.result.emit(self.original_file)
                return

            # --- Affine incremental
            T_prev = np.array(self.current_transform, dtype=np.float32).reshape(2, 3)
            use_warp = not np.allclose(T_prev, np.array([[1,0,0],[0,1,0]], dtype=np.float32), rtol=1e-5, atol=1e-5)

            if use_warp and cv2 is not None:
                src_for_match = cv2.warpAffine(
                    gray_small, T_prev, (Wref, Href),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            else:
                if gray_small.shape != ref_small.shape and cv2 is not None:
                    src_for_match = cv2.resize(gray_small, (Wref, Href), interpolation=cv2.INTER_LINEAR)
                else:
                    src_for_match = gray_small

            try:
                transform = self.compute_affine_transform_astroalign(
                    src_for_match, ref_small, limit_stars=getattr(self, "limit_stars", None)
                )
            except Exception as e:
                msg = str(e)
                base = os.path.basename(self.original_file)
                if "of matching triangles exhausted" in msg.lower():
                    self.signals.error.emit(f"Astroalign failed for {base}: List of matching triangles exhausted")
                else:
                    self.signals.error.emit(f"Astroalign failed for {base}: {msg}")
                return

            if transform is None:
                base = os.path.basename(self.original_file)
                self.signals.error.emit(f"Astroalign failed for {base} – skipping (no transform returned)")
                return

            transform = np.array(transform, dtype=np.float64).reshape(2, 3)
            key = os.path.normpath(self.original_file)
            self.signals.result_transform.emit(key, transform)
            self.signals.progress.emit(
                f"Astroalign delta for {os.path.basename(self.original_file)} "
                f"(model={self.model_name}): dx={transform[0, 2]:.2f}, dy={transform[1, 2]:.2f}"
            )
            self.signals.result.emit(self.original_file)

        except Exception as e:
            self.signals.error.emit(f"Error processing {self.original_file}: {e}")



    @staticmethod
    def compute_affine_transform_astroalign(source_img, reference_img,
                                            scale=1.20,
                                            limit_stars: int | None = None,
                                            det_sigma: float = 12.0,
                                            minarea: int = 10):
        global _AA_LOCK
        import astroalign
        Hs, Ws = source_img.shape[:2]
        Hr, Wr = reference_img.shape[:2]

        h = min(int(round(Hs * scale)), Hr)
        w = min(int(round(Ws * scale)), Wr)
        y0 = max(0, (Hr - h) // 2)
        x0 = max(0, (Wr - w) // 2)
        ref_crop = reference_img[y0:y0+h, x0:x0+w]

        kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
        if limit_stars is not None:
            kwargs["max_control_points"] = int(limit_stars)

        try:
            with _AA_LOCK:
                tform, _ = astroalign.find_transform(
                    np.ascontiguousarray(source_img.astype(np.float32)),
                    np.ascontiguousarray(ref_crop.astype(np.float32)),
                    **kwargs
                )
        except TypeError:
            with _AA_LOCK:
                legacy_kwargs = {}
                if "max_control_points" in kwargs:
                    legacy_kwargs["max_control_points"] = kwargs["max_control_points"]
                tform, _ = astroalign.find_transform(
                    np.ascontiguousarray(source_img.astype(np.float32)),
                    np.ascontiguousarray(ref_crop.astype(np.float32)),
                    **legacy_kwargs
                )

        P = np.asarray(tform.params, dtype=np.float64)
        if P.shape == (3,3):
            T = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            return (T @ P)[0:2, :]
        elif P.shape == (2,3):
            A3 = np.vstack([P, [0,0,1]])
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            return (T @ A3)[0:2, :]
        return None

def _project_to_similarity(T2x3: np.ndarray) -> np.ndarray:
    T2x3 = np.asarray(T2x3, np.float64).reshape(2,3)
    R = T2x3[:, :2]
    t = T2x3[:, 2]
    U, S, Vt = np.linalg.svd(R)
    rot = U @ Vt
    if np.linalg.det(rot) < 0:
        U[:, -1] *= -1
        rot = U @ Vt
    s = float((S[0] + S[1]) * 0.5)  # uniform scale
    Rsim = rot * s
    out = np.zeros((2,3), np.float64)
    out[:, :2] = Rsim
    out[:, 2] = t
    return out

def _detect_stars_uniform(img32: np.ndarray,
                          det_sigma: float = 12.0,
                          minarea: int = 10,
                          grid=(4,4),
                          max_per_cell: int = 25,
                          max_total: int = 500) -> np.ndarray:
    """
    Fast star detection on float32 mono image.
    Returns Nx2 (x,y) float32 points spread across the image.
    """
    import numpy as np
    import sep

    img32 = np.asarray(img32, np.float32, order="C")
    H, W = img32.shape[:2]

    # SEP background / threshold
    bkg = sep.Background(img32, bw=64, bh=64)
    thresh = float(det_sigma) * float(bkg.globalrms)

    objs = sep.extract(img32 - bkg.back(), thresh, minarea=int(minarea))
    if objs is None or len(objs) == 0:
        return np.empty((0,2), np.float32)

    # sort by flux desc (brightest first)
    order = np.argsort(objs["flux"])[::-1]
    xs = objs["x"][order].astype(np.float32)
    ys = objs["y"][order].astype(np.float32)

    gy, gx = int(grid[0]), int(grid[1])
    cell_w = W / gx
    cell_h = H / gy

    keep_counts = np.zeros((gy, gx), dtype=np.int32)
    pts = []

    for x, y in zip(xs, ys):
        cx = int(x / cell_w)
        cy = int(y / cell_h)
        if cx < 0 or cy < 0 or cx >= gx or cy >= gy:
            continue
        if keep_counts[cy, cx] >= max_per_cell:
            continue
        keep_counts[cy, cx] += 1
        pts.append((x, y))
        if len(pts) >= max_total:
            break

    if not pts:
        return np.empty((0,2), np.float32)
    return np.asarray(pts, np.float32)


class StarRegistrationThread(QThread):
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)  # (done, total)

    def __init__(self, reference_image_path_or_view, files_to_align, output_directory,
                 max_refinement_passes=3, shift_tolerance=0.2, parent_window=None, align_prefs: dict | None = None):
        """
        reference_image_path_or_view: path string OR "__ACTIVE_VIEW__"
        If "__ACTIVE_VIEW__", we'll read the current active view as the reference frame.
        """
        super().__init__()
        self.reference = reference_image_path_or_view
        self.parent_window = parent_window
        self.original_files = [os.path.normpath(f) for f in files_to_align]
        self.files_to_align = self.original_files.copy()
        self.output_directory = os.path.normpath(output_directory)
        self.max_refinement_passes = max_refinement_passes
        self.shift_tolerance = shift_tolerance

        self.file_key_to_current_path = {f: f for f in self.original_files}
        self.alignment_matrices = {}
        self.transform_deltas = []
        self._done = 0
        self._total = len(self.original_files) * self.max_refinement_passes
        self.align_prefs = align_prefs or _align_prefs(QSettings())
        self.align_model = str(self.align_prefs.get("model", "affine")).lower()
        self.h_reproj   = float(self.align_prefs.get("h_reproj", 3.0))
        self.det_sigma   = float(self.align_prefs.get("det_sigma", 12.0))
        self.limit_stars = int(self.align_prefs.get("limit_stars", 500))
        self.minarea     = int(self.align_prefs.get("minarea", 10))
        self.downsample = int(self.align_prefs.get("downsample", 2))
        self.drizzle_xforms = {}  # {orig_norm_path: (kind, matrix)}

    @staticmethod
    def _aa_model_and_residual(src_gray: np.ndarray,
                            ref2d: np.ndarray,
                            model: str,
                            h_reproj: float,
                            det_sigma: float,
                            minarea: int,
                            max_control_points: int | None = None):
        """
        AA on a ~1.2× center crop; lift matches to full coords; re-estimate requested model.
        Returns: (kind, X, residual_rms_px, n_inliers)

        kind in {"affine","homography","similarity"} or base affine/homography if poly fails upstream.
        For poly3/4 we still return the base model here; finalize does the true residual warp.
        """
        import numpy as np
        import astroalign
        import cv2

        src = np.ascontiguousarray(src_gray.astype(np.float32))
        ref = np.ascontiguousarray(ref2d.astype(np.float32))
        Hs, Ws = src.shape[:2]
        Hr, Wr = ref.shape[:2]

        # ---- 1) center crop the reference to ~1.2× source ----
        scale = 1.20
        h = min(int(round(Hs * scale)), Hr)
        w = min(int(round(Ws * scale)), Wr)
        y0 = max(0, (Hr - h) // 2)
        x0 = max(0, (Wr - w) // 2)
        ref_crop = ref[y0:y0+h, x0:x0+w]

        kwargs = {"detection_sigma": float(det_sigma), "min_area": int(minarea)}
        if max_control_points is not None:
            kwargs["max_control_points"] = int(max_control_points)

        # ---- 2) astroalign correspondences (adaptive tiling) ----
        src_xy, tgt_xy, best_P, best_xy0 = _aa_find_pairs_multitile(
            src, ref,
            scale=1.20, tiles=1,
            det_sigma=det_sigma, minarea=minarea,
            max_control_points=max_control_points
        )

        if src_xy is None or len(src_xy) < 8:
            raise RuntimeError("astroalign produced too few matches")

        # ✅ your spread / covariance gate:
        if not _points_spread_ok(tgt_xy, Wr, Hr):
            src_xy2, tgt_xy2, best_P2, best_xy0_2 = _aa_find_pairs_multitile(
                src, ref,
                scale=1.20, tiles=3,           # 3x3 grid
                det_sigma=det_sigma, minarea=minarea,
                max_control_points=max_control_points
            )
            if src_xy2 is not None and len(src_xy2) > len(src_xy):
                src_xy, tgt_xy = src_xy2, tgt_xy2
                best_P, best_xy0 = best_P2, best_xy0_2

        # ---- 3) base full-ref transform from best_P + crop translation ----
        x0, y0 = best_xy0
        P = np.asarray(best_P, dtype=np.float64)
        if P.shape == (3, 3):
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_kind = "homography"
            base_X    = T @ P
        else:
            A3 = np.vstack([P[0:2,:], [0,0,1]])
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_kind = "affine"
            base_X    = (T @ A3)[0:2, :]

        # helper: force an affine 2x3 into nearest similarity (no shear)
        def _affine_to_similarity(A2x3: np.ndarray) -> np.ndarray:
            A2x3 = np.asarray(A2x3, np.float64).reshape(2, 3)
            R = A2x3[:, :2]
            t = A2x3[:, 2]
            # SVD to get closest rotation + uniform scale
            U, S, Vt = np.linalg.svd(R)
            rot = U @ Vt
            if np.linalg.det(rot) < 0:
                U[:, -1] *= -1
                rot = U @ Vt
            s = float((S[0] + S[1]) * 0.5)
            Rsim = rot * s
            out = np.zeros((2, 3), dtype=np.float64)
            out[:, :2] = Rsim
            out[:, 2] = t
            return out

        # ---- 4) re-estimate requested model (RANSAC) with lifted pairs ----
        hth = float(h_reproj)
        m = (model or "affine").lower()
        if m in ("no_distortion", "nodistortion"):
            m = "similarity"

        if m == "homography":
            H, inl = cv2.findHomography(src_xy, tgt_xy, cv2.RANSAC, ransacReprojThreshold=hth)
            if H is None:
                kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "homography", np.asarray(H, np.float64)
                inl_mask = inl.ravel().astype(bool)

        elif m == "affine":
            A, inl = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=hth)
            if A is None:
                kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "affine", np.asarray(A, np.float64)
                inl_mask = inl.ravel().astype(bool)

        elif m == "similarity":
            A, inl = cv2.estimateAffinePartial2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=hth)
            if A is None:
                # fallback: project base to similarity so we NEVER shear
                if base_kind == "affine":
                    kind, X = "similarity", _affine_to_similarity(base_X)
                else:
                    kind, X = base_kind, base_X
                inl_mask = None
            else:
                kind, X = "similarity", np.asarray(A, np.float64)
                inl_mask = inl.ravel().astype(bool)

        else:
            # poly3/4: report residual versus base model; finalize applies poly residual warp
            kind, X  = base_kind, base_X
            inl_mask = None

        # ---- 5) residual RMS (px) using whichever model we returned ----
        if kind == "homography":
            ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
            P3   = np.hstack([src_xy.astype(np.float32), ones]).T
            Q    = (np.asarray(X, np.float32) @ P3)
            pred = (Q[:2, :] / Q[2:3, :]).T
        else:  # affine or similarity (2x3)
            A2 = np.asarray(X, np.float32).reshape(2, 3)
            pred = (src_xy @ A2[:, :2].T) + A2[:, 2]

        if inl_mask is not None and inl_mask.sum() >= 10:
            res = np.linalg.norm(pred[inl_mask] - tgt_xy[inl_mask], axis=1)
            nin = int(inl_mask.sum())
        else:
            res = np.linalg.norm(pred - tgt_xy, axis=1)
            nin = int(res.shape[0])

        residual_rms = float(np.sqrt(np.mean(res**2))) if res.size else float("inf")
        return kind, X, residual_rms, nin



    def _estimate_model_transform(self, src_gray_full: np.ndarray) -> tuple[str, object]:
        """
        Fast, robust final transform: crop reference to ~1.2× source size (centered),
        solve on the crop, lift correspondences to FULL reference coords, then
        re-estimate the requested model (affine/homography or poly3/4). Returns (kind, X).
        """
        ref2d = self.reference_image_2d
        src = np.ascontiguousarray(src_gray_full.astype(np.float32))
        ref = np.ascontiguousarray(ref2d.astype(np.float32))
        Hs, Ws = src.shape[:2]
        Hr, Wr = ref.shape[:2]

        # ---- 1) center crop the reference to ~1.2× source ----
        scale = 1.20
        h = min(int(round(Hs * scale)), Hr)
        w = min(int(round(Ws * scale)), Wr)
        y0 = max(0, (Hr - h) // 2)
        x0 = max(0, (Wr - w) // 2)
        ref_crop = ref[y0:y0+h, x0:x0+w]

        # ---- 2) find_transform on the small pair; lift matches to full coords ----
        with _AA_LOCK:
            tform, (src_pts_s, tgt_pts_s) = astroalign.find_transform(src, ref_crop)

        src_xy = np.asarray(src_pts_s, dtype=np.float32)
        tgt_xy = np.asarray(tgt_pts_s, dtype=np.float32)
        tgt_xy[:, 0] += x0   # lift crop -> full
        tgt_xy[:, 1] += y0

        # Build a base full-ref transform from tform.params + crop translation
        P = np.asarray(tform.params, dtype=np.float64)
        if P.shape == (3,3):
            base_kind0 = "homography"
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_X0 = T @ P
        else:
            base_kind0 = "affine"
            A3 = np.vstack([P[0:2,:], [0,0,1]])
            T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
            base_X0 = (T @ A3)[0:2, :]

        # ---- 3) re-estimate requested model using full-coord pairs ----
        model = (self.align_model or "affine").lower()
        h_reproj = float(self.h_reproj)

        if model == "homography":
            H, _ = cv2.findHomography(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=h_reproj)
            if H is None:
                base_kind, base_X = base_kind0, base_X0
            else:
                base_kind, base_X = "homography", np.array(H, dtype=np.float64)
        elif model == "affine":
            A, _ = cv2.estimateAffine2D(src_xy, tgt_xy, method=cv2.RANSAC, ransacReprojThreshold=h_reproj)
            if A is None:
                base_kind, base_X = base_kind0, base_X0
            else:
                base_kind, base_X = "affine", np.array(A, dtype=np.float64)
        else:
            base_kind, base_X = base_kind0, base_X0  # for poly we refine from base

        # ---- 4) if not poly, we’re done ----
        if model not in ("poly3", "poly4"):
            return base_kind, base_X

        # ---- 5) poly residual refinement (unchanged logic, but with our pairs) ----
        if base_kind == "affine":
            pred_on_ref = _apply_affine_to_pts(base_X, src_xy)
        else:
            ones = np.ones((src_xy.shape[0], 1), dtype=np.float32)
            P3 = np.hstack([src_xy.astype(np.float32), ones]).T
            Q  = (np.asarray(base_X, np.float32) @ P3)
            pred_on_ref = (Q[:2, :] / Q[2:3, :]).T

        resid = np.linalg.norm(pred_on_ref - tgt_xy, axis=1)
        r_thresh = max(2.0, h_reproj * 1.5)
        inliers = resid < r_thresh
        if inliers.sum() < 20:
            return base_kind, base_X

        P_ref   = tgt_xy[inliers].astype(np.float32)
        P_pred  = pred_on_ref[inliers].astype(np.float32)

        Href, Wref = ref2d.shape[:2]
        scale_vec = np.array([Wref, Href], dtype=np.float32)
        P_ref_n  = P_ref  / scale_vec
        P_pred_n = P_pred / scale_vec

        order = 3 if model == "poly3" else 4
        t_poly = PolynomialTransform()
        ok = t_poly.estimate(P_ref_n, P_pred_n, order=order)  # ref_n -> basewarped_n
        if not ok:
            return base_kind, base_X

        def _warp_poly_residual(img: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
            Hh, Ww = out_hw
            # Pass A: base warp
            if base_kind == "affine":
                if img.ndim == 2:
                    img_base = cv2.warpAffine(img, base_X, (Ww, Hh),
                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    img_base = np.stack([cv2.warpAffine(img[..., c], base_X, (Ww, Hh),
                                                        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                        for c in range(img.shape[2])], axis=2)
            else:
                if img.ndim == 2:
                    img_base = cv2.warpPerspective(img, base_X, (Ww, Hh),
                                                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                else:
                    img_base = np.stack([cv2.warpPerspective(img[..., c], base_X, (Ww, Hh),
                                                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                                        for c in range(img.shape[2])], axis=2)

            class _InvMap:
                def __call__(self, coords):
                    coords_n = coords.astype(np.float32) / scale_vec
                    mapped_n = t_poly(coords_n)
                    return mapped_n * scale_vec

            try:
                out = warp(img_base.astype(np.float32, copy=False),
                        inverse_map=_InvMap(),
                        output_shape=(Hh, Ww),
                        preserve_range=True,
                        channel_axis=(-1 if img_base.ndim == 3 else None))
            except TypeError:
                if img_base.ndim == 2:
                    out = warp(img_base.astype(np.float32), inverse_map=_InvMap(),
                            output_shape=(Hh, Ww), preserve_range=True)
                else:
                    chs = [warp(img_base[..., c].astype(np.float32), inverse_map=_InvMap(),
                                output_shape=(Hh, Ww), preserve_range=True)
                        for c in range(img_base.shape[2])]
                    out = np.stack(chs, axis=2)
            return out.astype(np.float32, copy=False)

        return f"poly{order}", _warp_poly_residual



    def _warp_with_kind(self, img: np.ndarray, kind: str, X: object, out_hw: tuple[int,int]) -> np.ndarray:
        Hh, Ww = out_hw
        if kind == "affine":
            A = np.asarray(X, np.float64)
            if img.ndim == 2:
                return cv2.warpAffine(img, A, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.stack([cv2.warpAffine(img[..., c], A, (Ww, Hh),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            for c in range(img.shape[2])], axis=2)

        if kind.startswith("poly"):
            return X(img, (Hh, Ww))

        if kind == "homography":
            H = np.asarray(X, np.float64)
            if img.ndim == 2:
                return cv2.warpPerspective(img, H, (Ww, Hh), flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.stack([cv2.warpPerspective(img[..., c], H, (Ww, Hh),
                                                flags=cv2.INTER_LANCZOS4,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                            for c in range(img.shape[2])], axis=2)

        # TPS: X is a callable(img, out_hw) -> img
        return X(img, (Hh, Ww))


    def run(self):
        self.progress_update.emit(f"Alignment model = {self.align_model}")
        try:
            _cap_native_threads_once()

            # Resolve reference → 2D float32
            if isinstance(self.reference, str) and self.reference == "__ACTIVE_VIEW__":
                ref_img, _, _ = _get_image_from_active_view(self.parent_window)
                if ref_img is None:
                    self.registration_complete.emit(False, "Active view not available for reference.")
                    return
                ref2d = np.mean(ref_img, axis=2) if ref_img.ndim == 3 else ref_img
            else:
                ref_img, _, _, _ = load_image(self.reference)
                if ref_img is None:
                    self.registration_complete.emit(False, "Reference image failed to load!")
                    return
                ref2d = np.mean(ref_img, axis=2) if ref_img.ndim == 3 else ref_img

            ref2d = np.nan_to_num(ref2d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            self.reference_image_2d = ref2d

            # ✂️ No DAO/RANSAC: astroalign handles detection internally.

            # Single shared downsampled ref for workers
            #ds = max(1, int(self.align_prefs.get("downsample", 2)))
            #if ds > 1:
            #    new_hw = (max(1, ref2d.shape[1] // ds), max(1, ref2d.shape[0] // ds))  # (W, H)
            #    ref_small = cv2.resize(ref2d, new_hw, interpolation=cv2.INTER_AREA)
            #else:
            #    ref_small = ref2d
            #self.ref_small = np.ascontiguousarray(ref_small.astype(np.float32))
            self.ref_small = np.ascontiguousarray(ref2d.astype(np.float32))

            # Initialize transforms to identity for EVERY original frame
            self.alignment_matrices = {os.path.normpath(f): IDENTITY_2x3.copy() for f in self.original_files}
            self.delta_transforms = {}

            # Progress totals (units = number of worker completions across passes)
            self._done = 0
            self._total = len(self.original_files) * max(1, int(self.max_refinement_passes))

            # Registration passes (compute deltas only)
            for pass_idx in range(self.max_refinement_passes):
                self.progress_update.emit(f"⏳ Refinement Pass {pass_idx + 1}/{self.max_refinement_passes}…")
                success, msg = self.run_one_registration_pass(None, None, pass_idx)
                if not success:
                    any_aligned = any(x is not None for x in self.alignment_matrices.values())
                    if not any_aligned:
                        self.registration_complete.emit(False, "No frames could be aligned. Aborting.")
                        return
                    self.progress_update.emit("Partial success: some frames permanently failed.")
                    break

                # Convergence check on this pass’ deltas
                if self.transform_deltas and max(self.transform_deltas[-1]) < self.shift_tolerance:
                    self.progress_update.emit("✅ Convergence reached! Stopping refinement.")
                    break

            # Finalize: single full-res read → warp → write per frame
            self._finalize_writes()

            # Summary based on our known corpus
            total_count = len(self.original_files)
            aligned_count = sum(1 for f in self.original_files if os.path.exists(self.file_key_to_current_path.get(f, "")))
            summary = f"Registration complete. Valid frames: {aligned_count}/{total_count}."
            self.registration_complete.emit(True, summary)

        except Exception as e:
            self.registration_complete.emit(False, f"Error: {e}")


    def _increment_progress(self):
        self._done += 1
        self.progress_step.emit(self._done, self._total)

    # ─────────────────────────────────────────────────────────────
    # Drop-in replacement for: StarRegistrationThread.run_one_registration_pass
    # ─────────────────────────────────────────────────────────────
    def run_one_registration_pass(self, _ref_stars_unused, _ref_triangles_unused, pass_index):
        _cap_native_threads_once()
        import os
        import shutil
        import tempfile
        import cv2

        model = (self.align_model or "affine").lower()
        ref_small = np.ascontiguousarray(self.ref_small.astype(np.float32, copy=False))
        Href, Wref = ref_small.shape[:2]

        # --- Build reverse map: current_path -> original_key (handles bin2-upscale / rewrites)
        rev_current_to_orig = {}
        for orig_k, curr_p in self.file_key_to_current_path.items():
            rev_current_to_orig[os.path.normpath(curr_p)] = os.path.normpath(orig_k)

        # ---------- NON-AFFINE PATH: residuals-only ----------
        if model in ("homography", "poly3", "poly4"):
            work_list = list(self.original_files)

            from concurrent.futures import ProcessPoolExecutor, as_completed
            procs = max(2, min((os.cpu_count() or 8), 32))
            self.progress_update.emit(f"Using {procs} processes to measure residuals (model={model}).")

            tmpdir = tempfile.mkdtemp(prefix="sas_resid_")
            ref_npy = os.path.join(tmpdir, "ref_small.npy")
            try:
                np.save(ref_npy, ref_small)
            except Exception as e:
                try: shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                self.on_worker_error(f"Failed to persist residual reference: {e}")
                return False, "Residual pass aborted."

            pass_deltas = []
            try:
                from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
                import time

                jobs = [
                    (p, ref_npy, model, self.h_reproj, self.det_sigma, self.minarea, self.limit_stars)
                    for p in work_list
                ]
                total = len(jobs)
                done = 0

                self.progress_update.emit(f"Using {procs} processes to measure residuals (model={model}).")
                self.progress_step.emit(0, total)

                with ProcessPoolExecutor(max_workers=procs) as ex:
                    pending = {ex.submit(_residual_job_worker, j): j[0] for j in jobs}  # future -> ORIGINAL path
                    last_heartbeat = time.monotonic()

                    while pending:
                        done_set, pending = wait(pending, timeout=0.6, return_when=FIRST_COMPLETED)
                        # heartbeat if nothing finished for a bit
                        now = time.monotonic()
                        if not done_set and (now - last_heartbeat) > 2.0:
                            self.progress_update.emit(f"… measuring residuals ({done}/{total} done)")
                            last_heartbeat = now

                        for fut in done_set:
                            orig_pth = os.path.normpath(pending.pop(fut, "<unknown>")) if fut in pending else "<unknown>"
                            try:
                                pth, rms, err = fut.result()
                            except Exception as e:
                                pth, rms, err = (orig_pth, float("inf"), f"Worker crashed: {e}")

                            k_orig = os.path.normpath(pth or orig_pth)
                            if err:
                                self.on_worker_error(f"Residual measure failed for {os.path.basename(k_orig)}: {err}")
                                self.delta_transforms[k_orig] = float("inf")
                            else:
                                self.delta_transforms[k_orig] = float(rms)
                                self.progress_update.emit(
                                    f"[residuals] {os.path.basename(k_orig)} → RMS={rms:.2f}px"
                                )

                            done += 1
                            self.progress_step.emit(done, total)
                            last_heartbeat = now

                for orig in self.original_files:
                    pass_deltas.append(self.delta_transforms.get(os.path.normpath(orig), float("inf")))
                self.transform_deltas.append(pass_deltas)

                preview = ", ".join([f"{d:.2f}" if np.isfinite(d) else "∞" for d in pass_deltas[:10]])
                if len(pass_deltas) > 10:
                    preview += f" … ({len(pass_deltas)} total)"
                self.progress_update.emit(f"Pass {pass_index + 1}: residual RMS px [{preview}]")

                aligned_count = sum(1 for d in pass_deltas if np.isfinite(d) and d <= self.shift_tolerance)
                if aligned_count:
                    self.progress_update.emit(f"Within tolerance (≤ {self.shift_tolerance:.2f}px): {aligned_count} frame(s)")
                return True, "Residual pass complete."
            finally:
                try: shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        # ---------- AFFINE PATH (incremental delta accumulation) ----------
        resample_flag = cv2.INTER_AREA if pass_index == 0 else cv2.INTER_LINEAR

        if pass_index == 0:
            work_list = list(self.original_files)
        else:
            work_list = []
            for orig in self.original_files:
                k = os.path.normpath(orig)
                last_delta = self.delta_transforms.get(k, float("inf"))
                if not (last_delta < self.shift_tolerance):
                    work_list.append(orig)

        skipped = len(self.original_files) - len(work_list)
        if skipped > 0:
            self.progress_update.emit(
                f"Skipping {skipped} frame(s) already within {self.shift_tolerance:.2f}px."
            )
            for _ in range(skipped):
                self._increment_progress()

        if not work_list:
            self.transform_deltas.append([
                self.delta_transforms.get(os.path.normpath(f), 0.0)
                for f in self.original_files
            ])
            return True, "Pass complete (nothing to refine)."

        procs = max(2, min((os.cpu_count() or 8), 32))
        self.progress_update.emit(f"Using {procs} processes for stellar alignment (HW={os.cpu_count() or 8}).")

        timeout_sec = int(self.align_prefs.get("timeout_per_job_sec", 300))
        jobs = []
        for orig_key in work_list:
            ok = os.path.normpath(orig_key)
            current_path = os.path.normpath(self.file_key_to_current_path.get(ok, ok))
            current_transform = self.alignment_matrices.get(ok, IDENTITY_2x3)
            jobs.append((
                current_path,
                current_transform,
                ref_small, Wref, Href,
                resample_flag, float(self.det_sigma), int(self.limit_stars), int(self.minarea),
                model, float(self.h_reproj)
            ))

        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import time
        executor = ProcessPoolExecutor(max_workers=procs)

        try:
            fut_info, pending = {}, set()
            for j in jobs:
                f = executor.submit(_solve_delta_job, j)
                fut_info[f] = (time.monotonic(), j[0])  # j[0] = current_path
                pending.add(f)

            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for fut in done:
                    start_t, returned_path = fut_info.pop(fut, (None, "<unknown>"))
                    try:
                        curr_path_r, T_new, err = fut.result()
                    except Exception as e:
                        curr_path_r, T_new, err = (returned_path or "<unknown>", None, f"Worker crashed: {e}")

                    # Map CURRENT path back to ORIGINAL key for consistent accumulation
                    curr_norm = os.path.normpath(curr_path_r)
                    k_orig = rev_current_to_orig.get(curr_norm, curr_norm)

                    if err:
                        self.on_worker_error(err)
                        self._increment_progress()
                        continue

                    T_new = np.array(T_new, dtype=np.float64).reshape(2, 3)
                    if model in ("no_distortion", "nodistortion", "similarity"):
                        T_new = _project_to_similarity(T_new)                    
                    self.delta_transforms[k_orig] = float(np.hypot(T_new[0, 2], T_new[1, 2]))

                    T_prev = np.array(self.alignment_matrices.get(k_orig, IDENTITY_2x3), dtype=np.float64).reshape(2, 3)
                    prev_3 = np.vstack([T_prev, [0, 0, 1]])
                    new_3  = np.vstack([T_new,  [0, 0, 1]])
                    self.alignment_matrices[k_orig] = (new_3 @ prev_3)[0:2, :]

                    self.on_worker_progress(
                        f"Astroalign delta for {os.path.basename(curr_path_r)} "
                        f"(model={self.align_model}): dx={T_new[0, 2]:.2f}, dy={T_new[1, 2]:.2f}"
                    )
                    self._increment_progress()

                # Timeouts
                now = time.monotonic()
                forget = []
                for fut in pending:
                    start_t, orig_path = fut_info.get(fut, (None, "<unknown>"))
                    if start_t is None:
                        continue
                    if (now - start_t) > timeout_sec:
                        base = os.path.basename(orig_path or "<unknown>")
                        self.on_worker_error(f"Astroalign timeout for {base} (>{timeout_sec}s) – skipping")
                        forget.append(fut)
                        self._increment_progress()
                for fut in forget:
                    fut_info.pop(fut, None)
                    if fut in pending:
                        pending.remove(fut)

            pass_deltas, aligned_count = [], 0
            for orig in self.original_files:
                k = os.path.normpath(orig)
                d = self.delta_transforms.get(k, 0.0)
                pass_deltas.append(d)
                if d <= self.shift_tolerance:
                    aligned_count += 1

            self.transform_deltas.append(pass_deltas)
            preview = ", ".join([f"{d:.2f}" for d in pass_deltas[:10]])
            if len(pass_deltas) > 10:
                preview += f" … ({len(pass_deltas)} total)"
            self.progress_update.emit(f"Pass {pass_index + 1} delta shifts: [{preview}]")
            if aligned_count:
                self.progress_update.emit(f"Skipped (delta < {self.shift_tolerance:.2f}px): {aligned_count} frame(s)")
            return True, "Pass complete."
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass


    def on_worker_result_transform(self, persistent_key, new_transform):
        k = os.path.normpath(persistent_key)
        T_new = np.array(new_transform, dtype=np.float64).reshape(2, 3)

        self.delta_transforms[k] = float(np.hypot(T_new[0, 2], T_new[1, 2]))

        T_prev = np.array(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64).reshape(2, 3)
        prev_3 = np.vstack([T_prev, [0, 0, 1]])
        new_3  = np.vstack([T_new,  [0, 0, 1]])
        combined = new_3 @ prev_3
        self.alignment_matrices[k] = combined[0:2, :]

    def on_worker_progress(self, msg):
        self.progress_update.emit(msg)

    def on_worker_error(self, msg):
        self.progress_update.emit("Error: " + msg)

    def on_worker_result(self, out):
        print("Saved: " + out)

    # ----- Star detection (reference) -----
    def detect_stars(self, image):
        self.progress_update.emit("✨ Detecting stars in reference frame")
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        mean, median, std = sigma_clipped_stats(image)
        fwhm_list = [2.5, 3, 3.5, 4, 5, 6, 7]

        all_sources = []
        for fwhm in fwhm_list:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=4 * std)
            sources = daofind(image - median)
            if sources is not None and len(sources) > 0:
                all_sources.append(sources)

        if not all_sources:
            return np.empty((0, 2), dtype=np.float32)

        combined_sources = vstack(all_sources)
        x_rounded = np.round(combined_sources['xcentroid'], 1)
        y_rounded = np.round(combined_sources['ycentroid'], 1)
        xy_rounded = np.array([x_rounded, y_rounded]).T

        seen = {}
        unique_rows = []
        for i, (rx, ry) in enumerate(xy_rounded):
            key = (rx, ry)
            if key not in seen:
                seen[key] = True
                unique_rows.append(i)

        final_sources = combined_sources[unique_rows]
        star_coords = np.vstack([final_sources['xcentroid'], final_sources['ycentroid']]).T
        return star_coords.astype(np.float32)

    # ----- Triangle dict helpers -----
    def build_triangle_dict(self, coords):
        tri = Delaunay(coords)
        tri_dict = {}
        for simplex in tri.simplices:
            pts = coords[simplex]
            inv = self.compute_triangle_invariant(pts)
            if inv is None:
                continue
            inv_key = (round(inv[0], 2), round(inv[1], 2))
            tri_dict.setdefault(inv_key, []).append(simplex)
        return tri_dict

    @staticmethod
    def compute_triangle_invariant(tri_points):
        d1 = np.linalg.norm(tri_points[0] - tri_points[1])
        d2 = np.linalg.norm(tri_points[1] - tri_points[2])
        d3 = np.linalg.norm(tri_points[2] - tri_points[0])
        sides = sorted([d1, d2, d3])
        if sides[0] == 0:
            return None
        return (round(sides[1] / sides[0], 4), round(sides[2] / sides[0], 4))

    # ----- Validity + warp -----
    @staticmethod
    def is_valid_transform_static(matrix):
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        scale_x = np.sqrt(a ** 2 + c ** 2)
        scale_y = np.sqrt(b ** 2 + d ** 2)
        return (0.9 <= scale_x <= 1.1) and (0.9 <= scale_y <= 1.1)

    @staticmethod
    def apply_affine_transform_static(image, transform_matrix):
        T = np.array(transform_matrix, dtype=np.float32).reshape(2, 3)
        h, w = image.shape[:2]
        if image.ndim == 2:
            aligned = cv2.warpAffine(image, T, (w, h), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            chans = []
            for i in range(image.shape[2]):
                chans.append(cv2.warpAffine(image[:, :, i], T, (w, h), flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0))
            aligned = np.stack(chans, axis=2)
        return aligned

    # ─────────────────────────────────────────────────────────────
    # NEW METHOD: StarRegistrationThread._finalize_writes
    # ─────────────────────────────────────────────────────────────
    def _finalize_writes(self):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import tempfile
        import shutil
        import os

        self.drizzle_xforms = {}

        try:
            Href, Wref = self.reference_image_2d.shape[:2]
        except Exception:
            self.progress_update.emit("⚠️ No reference image available; aborting finalize.")
            return
        self._ref_shape_for_sasd = (Href, Wref)

        tmpdir = tempfile.mkdtemp(prefix="sas_align_")
        ref_npy = os.path.join(tmpdir, "ref2d.npy")
        try:
            np.save(ref_npy, np.asarray(self.reference_image_2d, dtype=np.float32))
        except Exception as e:
            self.progress_update.emit(f"⚠️ Failed to persist reference for workers: {e}")
            try: shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            return

        finalize_workers = int(self.align_prefs.get("finalize_workers", min(os.cpu_count() or 8, 8)))
        finalize_workers = max(2, finalize_workers)

        jobs = []
        for orig_path in self.original_files:
            k = os.path.normpath(orig_path)
            A = np.asarray(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float64)

            # 👉 If non-affine, we pass identity to make workers solve from scratch
            if self.align_model.lower() in ("homography", "poly3", "poly4"):
                A = IDENTITY_2x3.copy()

            jobs.append((
                orig_path,
                self.align_model,
                (Href, Wref),
                ref_npy,
                A,
                float(self.h_reproj),
                self.output_directory,
                float(self.det_sigma),
                int(self.minarea),
                int(self.limit_stars) if self.limit_stars is not None else None
            ))

        self.progress_update.emit(f"📝 Finalizing aligned outputs with {finalize_workers} processes…")

        ok = 0
        try:
            with ProcessPoolExecutor(max_workers=finalize_workers) as ex:
                futs = [ex.submit(_finalize_write_job, j) for j in jobs]
                for fut in as_completed(futs):
                    try:
                        orig_path, out_path, msg, success, drizzle = fut.result()
                    except Exception as e:
                        self.progress_update.emit(f"⚠️ Finalize worker crashed: {e}")
                        continue

                    if msg:
                        for line in (msg.splitlines() or [msg]):
                            self.progress_update.emit(line)

                    if success:
                        ok += 1
                        k = os.path.normpath(orig_path)
                        self.file_key_to_current_path[k] = out_path

                        if isinstance(drizzle, tuple) and len(drizzle) == 2:
                            kind, M = drizzle
                            try:
                                if kind == "affine":
                                    self.drizzle_xforms[k] = ("affine", np.asarray(M, np.float64).reshape(2, 3))
                                elif kind == "homography":
                                    self.drizzle_xforms[k] = ("homography", np.asarray(M, np.float64).reshape(3, 3))
                                else:
                                    self.drizzle_xforms[k] = (str(kind), None)  # poly3/4
                            except Exception:
                                pass
        finally:
            try: shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            gc.collect()  # Free memory after finalization

        try:
            sasd_path = os.path.join(self.output_directory, "alignment_transforms.sasd")
            self._save_alignment_transforms_sasd(sasd_path)
            self.progress_update.emit("✅ Transform file saved as alignment_transforms.sasd")
        except Exception as e:
            self.progress_update.emit(f"⚠️ Failed to save alignment_transforms.sasd: {e}")


    def _save_alignment_transforms_sasd(self, out_path: str):
        """
        SASD v2.1 format:

            REF_SHAPE: <H>, <W>
            REF_PATH: <reference path or __ACTIVE_VIEW__>
            MODEL: mixed               # informative; real model is per file

            FILE: <abs path to *_n.fit>
            KIND: affine|homography|tps
            MATRIX:
            a, b, tx
            c, d, ty

            FILE: <next>
            KIND: homography
            MATRIX:
            h00, h01, h02
            h10, h11, h12
            h20, h21, h22

        Blank line between blocks.
        """
        # reference geometry + path
        try:
            Href, Wref = self.reference_image_2d.shape[:2]
        except Exception:
            Href, Wref = 0, 0
        ref_path = self.reference if isinstance(self.reference, str) else "__ACTIVE_VIEW__"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"REF_SHAPE: {int(Href)}, {int(Wref)}\n")
            f.write(f"REF_PATH: {ref_path}\n")
            f.write("MODEL: mixed\n\n")

            # prefer model-aware drizzle_xforms; fall back to affine stack if missing
            for orig_key in self.original_files:
                k = os.path.normpath(orig_key)
                kind = None
                M = None

                if isinstance(getattr(self, "drizzle_xforms", None), dict) and k in self.drizzle_xforms:
                    kind, M = self.drizzle_xforms[k]
                else:
                    # fallback: affine-only (2x3) from accumulated alignment_matrices
                    M_aff = np.asarray(self.alignment_matrices.get(k, IDENTITY_2x3), dtype=np.float32)
                    kind, M = "affine", M_aff

                f.write(f"FILE: {k}\n")
                f.write(f"KIND: {kind}\n")
                f.write("MATRIX:\n")

                _fmt = lambda x: f"{float(x):.16g}"

                # then for writing:
                if kind == "homography":
                    H = np.asarray(M, np.float64).reshape(3, 3)
                    f.write(f"{_fmt(H[0,0])}, {_fmt(H[0,1])}, {_fmt(H[0,2])}\n")
                    f.write(f"{_fmt(H[1,0])}, {_fmt(H[1,1])}, {_fmt(H[1,2])}\n")
                    f.write(f"{_fmt(H[2,0])}, {_fmt(H[2,1])}, {_fmt(H[2,2])}\n\n")
                elif kind == "affine":
                    A = np.asarray(M, np.float64).reshape(2, 3)
                    f.write(f"{_fmt(A[0,0])}, {_fmt(A[0,1])}, {_fmt(A[0,2])}\n")
                    f.write(f"{_fmt(A[1,0])}, {_fmt(A[1,1])}, {_fmt(A[1,2])}\n\n")
                else:
                    f.write("MATRIX: \nUNSUPPORTED\n\n")

def _center_crop_params(Href, Wref, Hsrc, Wsrc, scale=1.10):
    # crop box centered in reference, sized ~ source*scale, but clamped to ref
    h = min(int(round(Hsrc * scale)), Href)
    w = min(int(round(Wsrc * scale)), Wref)
    y0 = max(0, (Href - h) // 2)
    x0 = max(0, (Wref - w) // 2)
    return y0, x0, h, w

def _crop(img, y0, x0, h, w):
    return img[y0:y0+h, x0:x0+w]

def _compose_with_ref_translation_affine(A_2x3, x0, y0):
    # A_full = T @ A  (homog), then take 2x3 back
    A = np.asarray(A_2x3, dtype=np.float64).reshape(2,3)
    A3 = np.vstack([A, [0,0,1]])
    T  = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
    return (T @ A3)[0:2,:]

def _compose_with_ref_translation_homography(H_3x3, x0, y0):
    H = np.asarray(H_3x3, dtype=np.float64).reshape(3,3)
    T = np.array([[1,0,x0],[0,1,y0],[0,0,1]], dtype=np.float64)
    return T @ H


# ---------------------------------------------------------------------
# Optional simple batch UI (file-based) — no slots
# ---------------------------------------------------------------------
class StarRegistrationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reference_image = None
        self.reference_image_path = None  # or "__ACTIVE_VIEW__"
        self.files_to_align = []
        self.output_directory = None
        self.thread = None
        self.parent_window = parent
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)        
        self.initUI()

        self._prog_timer = QTimer(self)
        self._prog_timer.setInterval(100)      # 10 Hz flush
        self._prog_timer.setSingleShot(True)
        self._prog_timer.timeout.connect(self._flush_progress)
        self._pending_prog = None
        self._in_progress_slot = False

    def initUI(self):
        self.setWindowTitle("Star Registration")
        self.setGeometry(200, 200, 600, 450)
        main_layout = QVBoxLayout(self)

        # Reference selection
        ref_layout = QHBoxLayout()
        self.ref_label = QLabel("Reference Image:")
        self.ref_path_label = QLabel("No reference selected")
        self.ref_path_label.setWordWrap(True)

        self.select_ref_active_button = QPushButton("From Active View")
        self.select_ref_active_button.clicked.connect(self.select_reference_from_active_view)

        self.select_ref_file_button = QPushButton("From File")
        self.select_ref_file_button.clicked.connect(self.select_reference_from_file)

        ref_layout.addWidget(self.ref_label)
        ref_layout.addWidget(self.ref_path_label)
        ref_layout.addWidget(self.select_ref_active_button)
        ref_layout.addWidget(self.select_ref_file_button)

        # File selection
        file_selection_layout = QHBoxLayout()
        self.add_files_button = QPushButton("Select Files")
        self.add_files_button.clicked.connect(self.select_files_to_align)
        self.add_directory_button = QPushButton("Select Directory")
        self.add_directory_button.clicked.connect(self.select_directory_to_align)
        file_selection_layout.addWidget(self.add_files_button)
        file_selection_layout.addWidget(self.add_directory_button)

        # Tree of files
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(1)
        self.tree_widget.setHeaderLabels(["Files to Align"])
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        tree_buttons_layout = QHBoxLayout()
        self.remove_selected_button = QPushButton("Remove Selected")
        self.remove_selected_button.clicked.connect(self.remove_selected_files)
        self.clear_tree_button = QPushButton("Clear All")
        self.clear_tree_button.clicked.connect(self.clear_tree)
        tree_buttons_layout.addWidget(self.remove_selected_button)
        tree_buttons_layout.addWidget(self.clear_tree_button)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setWordWrap(True)
        self.select_output_button = QPushButton("Select Output Folder")
        self.select_output_button.clicked.connect(self.select_output_directory)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path_label)
        output_layout.addWidget(self.select_output_button)

        # Progress + start
        self.progress_label = QLabel("Status: Waiting…")
        self.progress_label.setStyleSheet("color: blue; font-weight: bold;")

        self.start_button = QPushButton("Start Registration")
        self.start_button.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.start_button.clicked.connect(self.start_registration)

        # Add to layout
        main_layout.addLayout(ref_layout)
        main_layout.addLayout(file_selection_layout)
        main_layout.addWidget(self.tree_widget)
        main_layout.addLayout(tree_buttons_layout)
        main_layout.addLayout(output_layout)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.start_button)

    def _enqueue_progress(self, message: str) -> None:
        # Save only the latest message; start the coalescing timer
        self._pending_prog = str(message) if message is not None else ""
        if not self._prog_timer.isActive():
            self._prog_timer.start()

    def _flush_progress(self) -> None:
        if self._pending_prog is None:
            return
        # Hard non-reentrancy guard — don’t let update trigger itself
        if self._in_progress_slot:
            return
        self._in_progress_slot = True
        try:
            self.progress_label.setText(f"Status: {self._pending_prog}")
        finally:
            self._in_progress_slot = False
            self._pending_prog = None


    # Reference selection (no slots)
    def select_reference_from_active_view(self):
        self.reference_image_path = "__ACTIVE_VIEW__"
        self.ref_path_label.setText("(Active View)")

    def select_reference_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if file_path:
            self.reference_image_path = file_path
            self.ref_path_label.setText(os.path.basename(file_path))

    # File selection
    def select_files_to_align(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Align", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.fits *.fit *.xisf);;All Files (*)"
        )
        if files:
            for file in files:
                if file not in self.files_to_align:
                    self.files_to_align.append(file)
                    self.tree_widget.addTopLevelItem(QTreeWidgetItem([os.path.basename(file)]))

    def select_directory_to_align(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.fits', '.fit', '.xisf')
            new_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(exts)]
            for file in new_files:
                if file not in self.files_to_align:
                    self.files_to_align.append(file)
                    self.tree_widget.addTopLevelItem(QTreeWidgetItem([os.path.basename(file)]))

    # Manage tree
    def remove_selected_files(self):
        selected_items = self.tree_widget.selectedItems()
        for item in selected_items:
            file_name = item.text(0)
            for file_path in list(self.files_to_align):
                if os.path.basename(file_path) == file_name:
                    self.files_to_align.remove(file_path)
                    break
            index = self.tree_widget.indexOfTopLevelItem(item)
            self.tree_widget.takeTopLevelItem(index)

    def clear_tree(self):
        self.tree_widget.clear()
        self.files_to_align.clear()

    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if directory:
            self.output_directory = directory
            self.output_path_label.setText(directory)

    # Start/finish
    def start_registration(self):
        if not self.reference_image_path:
            QMessageBox.warning(self, "Missing Reference", "Please select a reference image (file or active view).")
            return
        if not self.files_to_align:
            QMessageBox.warning(self, "No Files", "Please add files to align before starting.")
            return
        if not self.output_directory:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory before starting.")
            return

        self.progress_label.setText("Status: Running…")
        self.progress_label.setStyleSheet("color: green; font-weight: bold;")

        self.thread = StarRegistrationThread(
            self.reference_image_path,
            self.files_to_align,
            self.output_directory,
            parent_window=self.parent_window
        )
        self.thread.progress_update.connect(self._enqueue_progress)
        self.thread.registration_complete.connect(self.registration_finished)
        self.thread.start()

    def update_progress(self, message):
        self.progress_label.setText(f"Status: {message}")

    def registration_finished(self, success, message):
        color = "green" if success else "red"
        self.progress_label.setText(f"Status: {message}")
        self.progress_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if success:
            QMessageBox.information(self, "Registration Complete", message)
        else:
            QMessageBox.warning(self, "Registration Error", message)


# --------------------------------------------------
# MosaicMasterDialog with blending/normalization integrated
# --------------------------------------------------



