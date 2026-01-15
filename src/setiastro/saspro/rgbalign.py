# pro/rgbalign.py
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication,
    QComboBox, QCheckBox, QMessageBox, QProgressBar, QPlainTextEdit,
    QSpinBox, QGridLayout, QWidget
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QCursor


import astroalign

import sep

import cv2

# try to reuse poly from star_alignment if present
try:
    from setiastro.saspro.star_alignment import PolynomialTransform
except Exception:
    PolynomialTransform = None


def _translate_channel(ch: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if cv2 is None:
        return ch
    H, W = ch.shape[:2]
    A = np.array([[1, 0, dx],
                [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        ch, A, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def _planet_centroid(ch: np.ndarray):
    """Return (cx, cy, area) of dominant planet blob, or None."""
    if cv2 is None:
        return None

    img = ch.astype(np.float32, copy=False)
    p1 = float(np.percentile(img, 1.0))
    p99 = float(np.percentile(img, 99.5))
    if p99 <= p1:
        return None

    scaled = (img - p1) * (255.0 / (p99 - p1))
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    scaled = cv2.GaussianBlur(scaled, (0, 0), 1.2)

    _, bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    j = int(np.argmax(areas)) + 1
    area = float(stats[j, cv2.CC_STAT_AREA])
    if area < 200:
        return None

    cx, cy = cents[j]
    return (float(cx), float(cy), area)


# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────
class RGBAlignWorker(QThread):
    progress = pyqtSignal(int, str)     # (percent, message)
    done = pyqtSignal(np.ndarray)       # aligned RGB image
    failed = pyqtSignal(str)
    EDGE_FRAC = 0.55      # 55% of max radius
    MIN_EDGE_PTS = 6      # want at least 6 out there
    EDGE_INNER_FRAC = 0.38   # toss center 38% radius
    MATCH_MAX_DIST = 10.0    # px, generous for CA
    MIN_MATCHES = 6


    def __init__(self, img: np.ndarray, model: str, sep_sigma: float = 3.0):
        super().__init__()
        self.img = img
        self.model = model
        self.sep_sigma = float(sep_sigma)

        self.r_xform = None
        self.b_xform = None
        self.r_pairs = None
        self.b_pairs = None

    def _pts_too_central(self, pts: np.ndarray | None, shape) -> bool:
        """
        Return True if the matched points are all bunched near the center.
        pts: (N, 2) in x,y
        shape: (H, W)
        """
        if pts is None or len(pts) == 0:
            return True
        h, w = shape[:2]
        cx, cy = w * 0.5, h * 0.5
        # distance of each point from center
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        rmax = np.hypot(cx, cy)  # radius to corner
        edge_mask = r > (self.EDGE_FRAC * rmax)
        return edge_mask.sum() < self.MIN_EDGE_PTS

    def _sep_detect_points(self, img: np.ndarray):
        """Return (N,2) points from SEP, brightest first, using user sigma."""
        if sep is None:
            return None
        data = img.astype(np.float32, copy=False)
        bkg = sep.Background(data)
        data_sub = data - bkg
        # use the promoted sigma here 👇
        objs = sep.extract(data_sub, self.sep_sigma, err=bkg.globalrms)
        if objs is None or len(objs) == 0:
            return None
        idx = np.argsort(objs["peak"])[::-1]
        pts = np.stack([objs["x"][idx], objs["y"][idx]], axis=1)
        return pts

    def _filter_edge_ring(self, pts: np.ndarray, shape, inner_frac=EDGE_INNER_FRAC):
        """Keep only points outside inner_frac * Rmax."""
        if pts is None or pts.size == 0:
            return None
        h, w = shape[:2]
        cx, cy = w * 0.5, h * 0.5
        r = np.hypot(pts[:,0] - cx, pts[:,1] - cy)
        rmax = np.hypot(cx, cy)
        mask = r >= (inner_frac * rmax)
        pts_edge = pts[mask]
        return pts_edge if pts_edge.size else None

    def _pair_edge_points(self, src_img, ref_img, shape):
        """Detect in BOTH images, keep only edge ring in REF, then NN-match in SRC."""
        ref_pts = self._sep_detect_points(ref_img)
        src_pts = self._sep_detect_points(src_img)
        if ref_pts is None or src_pts is None:
            return None, None

        ref_edge = self._filter_edge_ring(ref_pts, shape)
        if ref_edge is None:
            return None, None

        # brute-force NN, small N, so ok
        src_arr = np.asarray(src_pts, dtype=np.float32)
        pairs_src = []
        pairs_dst = []
        for (x_ref, y_ref) in ref_edge:
            dxy = src_arr - np.array([x_ref, y_ref], dtype=np.float32)
            dist = np.hypot(dxy[:,0], dxy[:,1])
            j = np.argmin(dist)
            if dist[j] <= self.MATCH_MAX_DIST:
                # src point is in the channel we want to warp → source
                pairs_src.append(src_arr[j])
                # ref point is the green channel → destination
                pairs_dst.append([x_ref, y_ref])

        if len(pairs_src) < self.MIN_MATCHES:
            return None, None

        return (np.array(pairs_src, dtype=np.float32),
                np.array(pairs_dst, dtype=np.float32))


    def run(self):
        if self.img is None or self.img.ndim != 3 or self.img.shape[2] < 3:
            self.failed.emit("Image must be RGB (3 channels).")
            return
        if astroalign is None:
            self.failed.emit("astroalign is not available.")
            return

        try:
            self.progress.emit(5, "Preparing channels…")
            R = np.ascontiguousarray(self.img[..., 0].astype(np.float32, copy=False))
            G = np.ascontiguousarray(self.img[..., 1].astype(np.float32, copy=False))
            B = np.ascontiguousarray(self.img[..., 2].astype(np.float32, copy=False))

            # R → G
            self.progress.emit(15, "Aligning Red → Green…")
            kind_R, X_R, (r_src, r_dst) = self._estimate_transform(R, G, self.model)
            self.r_xform = (kind_R, X_R)
            self.r_pairs = (r_src, r_dst)
            self.progress.emit(35, f"Red transform = {kind_R}")
            R_aligned = self._warp_channel(R, kind_R, X_R, G.shape)

            # B → G
            self.progress.emit(55, "Aligning Blue → Green…")
            kind_B, X_B, (b_src, b_dst) = self._estimate_transform(B, G, self.model)
            self.b_xform = (kind_B, X_B)
            self.b_pairs = (b_src, b_dst)
            self.progress.emit(75, f"Blue transform = {kind_B}")
            B_aligned = self._warp_channel(B, kind_B, X_B, G.shape)

            out = np.stack([R_aligned, G, B_aligned], axis=2).astype(self.img.dtype, copy=False)
            self.progress.emit(100, "Done.")
            self.done.emit(out)
        except Exception as e:
            self.failed.emit(str(e))

    def _planet_centroid(self, ch: np.ndarray):
        """
        Return (cx, cy, area) in pixel coordinates for the dominant planet blob.
        Robust for uint8/uint16/float images.
        """
        if cv2 is None:
            return None

        img = ch.astype(np.float32, copy=False)

        # Normalize to 0..255 for thresholding (robust percentile scaling)
        p1 = float(np.percentile(img, 1.0))
        p99 = float(np.percentile(img, 99.5))
        if p99 <= p1:
            return None

        scaled = (img - p1) * (255.0 / (p99 - p1))
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)

        # Blur to suppress banding/noise
        scaled = cv2.GaussianBlur(scaled, (0, 0), 1.2)

        # Otsu threshold to separate planet from background
        _, bw = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up small junk / fill holes a bit
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

        # Largest connected component = planet
        num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
        if num <= 1:
            return None

        # stats: [label, x, y, w, h, area] but area is stats[:, cv2.CC_STAT_AREA]
        areas = stats[1:, cv2.CC_STAT_AREA]
        j = int(np.argmax(areas)) + 1
        area = float(stats[j, cv2.CC_STAT_AREA])
        if area < 200:  # too tiny = probably noise
            return None

        cx, cy = cents[j]
        return (float(cx), float(cy), area)

    def _estimate_translation_by_centroid(self, src: np.ndarray, ref: np.ndarray):
        c_src = self._planet_centroid(src)
        c_ref = self._planet_centroid(ref)
        if c_src is None or c_ref is None:
            return None

        sx, sy, sa = c_src
        rx, ry, ra = c_ref

        dx = rx - sx
        dy = ry - sy
        return (dx, dy, (sx, sy, sa), (rx, ry, ra))


    # ───── helpers (basically mini versions of your big star alignment logic) ─────
    def _estimate_transform(self, src: np.ndarray, ref: np.ndarray, model: str):
        H, W = ref.shape[:2]

        # ── Planet centroid ADC mode ─────────────────────────────
        if model == "planet-centroid":
            t = self._estimate_translation_by_centroid(src, ref)
            if t is None:
                # fall back to affine/astroalign if centroid fails
                # (or you can hard-fail here if you prefer)
                pass
            else:
                dx, dy, src_info, ref_info = t
                # store "pairs" as tiny diagnostic info
                src_xy = np.array([[src_info[0], src_info[1]]], dtype=np.float32)
                dst_xy = np.array([[ref_info[0], ref_info[1]]], dtype=np.float32)
                X = (float(dx), float(dy))
                return ("translate", X, (src_xy, dst_xy))
        # ── 0) edge-only, SEP-based path ─────────────────────────────
        if model == "edge-sep":
            src_xy, dst_xy = self._pair_edge_points(src, ref, (H, W))
            if src_xy is not None and dst_xy is not None and cv2 is not None:
                # 0a) try homography first (better for corner warp)
                Hh, inliers = cv2.findHomography(
                    src_xy, dst_xy,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2000,
                    confidence=0.999,
                )
                if Hh is not None:
                    return ("homography", Hh, (src_xy, dst_xy))

                # 0b) fallback → affine
                A, inliers = cv2.estimateAffine2D(
                    src_xy, dst_xy,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2000,
                    confidence=0.999,
                )
                if A is not None:
                    return ("affine", A, (src_xy, dst_xy))
            # if SEP failed or cv2 missing → fall through to astroalign normal path

        # ─────────────────────────────────────────────────────────────
        # 1) astroalign normal pass
        # ─────────────────────────────────────────────────────────────
        tform, (src_pts, dst_pts) = astroalign.find_transform(
            np.ascontiguousarray(src),
            np.ascontiguousarray(ref),
            max_control_points=50,
            detection_sigma=5.0,
            min_area=5,
        )

        # 2) 'hungry' pass if too central
        if self._pts_too_central(dst_pts, ref.shape):
            tform2, (src_pts2, dst_pts2) = astroalign.find_transform(
                np.ascontiguousarray(src),
                np.ascontiguousarray(ref),
                max_control_points=120,
                detection_sigma=3.0,
                min_area=3,
            )
            if not self._pts_too_central(dst_pts2, ref.shape):
                tform, src_pts, dst_pts = tform2, src_pts2, dst_pts2

        # 3) original branching
        P = np.asarray(tform.params, dtype=np.float64)
        src_xy = np.asarray(src_pts, dtype=np.float32)
        dst_xy = np.asarray(dst_pts, dtype=np.float32)

        # affine
        if model == "affine":
            if cv2 is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            A, _ = cv2.estimateAffine2D(
                src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if A is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            return ("affine", A, (src_xy, dst_xy))

        # homography
        if model == "homography":
            if cv2 is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            Hh, _ = cv2.findHomography(
                src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if Hh is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            return ("homography", Hh, (src_xy, dst_xy))

        # poly3 / poly4
        if model in ("poly3", "poly4") and PolynomialTransform is not None and cv2 is not None:
            order = 3 if model == "poly3" else 4
            scale_vec = np.array([W, H], dtype=np.float32)
            src_n = src_xy / scale_vec
            dst_n = dst_xy / scale_vec

            t_poly = PolynomialTransform()
            ok = t_poly.estimate(dst_n, src_n, order=order)  # dst → src
            if not ok:
                Hh, _ = cv2.findHomography(
                    src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                return ("homography", Hh, (src_xy, dst_xy))

            def _warp_poly(img: np.ndarray, out_shape: tuple[int, int]):
                Hh_, Ww_ = out_shape
                yy, xx = np.mgrid[0:Hh_, 0:Ww_].astype(np.float32)
                coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
                coords_n = coords / scale_vec
                mapped_n = t_poly(coords_n)
                mapped = mapped_n * scale_vec
                map_x = mapped[:, 0].reshape(Hh_, Ww_).astype(np.float32)
                map_y = mapped[:, 1].reshape(Hh_, Ww_).astype(np.float32)
                return cv2.remap(
                    img, map_x, map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

            return (model, _warp_poly, (src_xy, dst_xy))

        # fallback → homography
        if cv2 is None:
            if P.shape == (3, 3):
                return ("homography", P, (src_xy, dst_xy))
            A3 = np.vstack([P[0:2, :], [0, 0, 1]])
            return ("homography", A3, (src_xy, dst_xy))

        Hh, _ = cv2.findHomography(
            src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        return ("homography", Hh, (src_xy, dst_xy))



    def _pick_edge_stars_with_sep(self, img, tiles=(3,3), per_tile=2):
        if sep is None:
            return []
        data = img.astype(np.float32, copy=False)
        bkg = sep.Background(data)
        data_sub = data - bkg
        objs = sep.extract(data_sub, 1.5, err=bkg.globalrms)
        H, W = data.shape[:2]
        th, tw = H // tiles[0], W // tiles[1]
        picked = []
        for ty in range(tiles[0]):
            for tx in range(tiles[1]):
                y0, y1 = ty*th, min((ty+1)*th, H)
                x0, x1 = tx*tw, min((tx+1)*tw, W)
                box = objs[
                    (objs['y'] >= y0) & (objs['y'] < y1) &
                    (objs['x'] >= x0) & (objs['x'] < x1)
                ]
                if len(box) == 0:
                    continue
                # brightest first
                box = box[np.argsort(box['peak'])][::-1][:per_tile]
                for o in box:
                    picked.append((float(o['x']), float(o['y'])))
        return picked


    def _warp_channel(self, ch: np.ndarray, kind: str, X, ref_shape):
        H, W = ref_shape[:2]

        if kind == "translate":
            dx, dy = X
            A = np.array([[1, 0, dx],
                        [0, 1, dy]], dtype=np.float32)
            return cv2.warpAffine(
                ch, A, (W, H),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        if kind == "affine":
            # Just assume cv2 is available (standard dependency) for perf
            A = np.asarray(X, dtype=np.float32).reshape(2, 3)
            return cv2.warpAffine(ch, A, (W, H), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind == "homography":
            Hm = np.asarray(X, dtype=np.float32).reshape(3, 3)
            return cv2.warpPerspective(ch, Hm, (W, H), flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind.startswith("poly"):
            return X(ch, (H, W))

        return ch


# ─────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────
class RGBAlignDialog(QDialog):
    def __init__(self, parent=None, document=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("RGB Align"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.parent = parent
        # document could be a view; try to unwrap
        self.doc_view = document
        self.doc = getattr(document, "document", document)
        self.image = getattr(self.doc, "image", None) if self.doc is not None else None

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Align R and B channels to G.\n"
                             "Select model and run."))

        hl = QHBoxLayout()
        hl.addWidget(QLabel(self.tr("Alignment model:")))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Planet Centroid (ADC)",   # NEW
            "EDGE",           # ← first, new default
            "Homography",
            "Affine",
            "Poly 3",
            "Poly 4",
        ])
        self.model_combo.setCurrentIndex(1)
        self.model_combo.setItemData(
            0,
            (
                "Planet Centroid (Atmospheric Dispersion)\n"
                "• Find planet disk in each channel\n"
                "• Compute centroid (moments)\n"
                "• Shift R and B so their centroids match G\n"
                "• Translation only (no warp), ideal for planets"
            ),
            Qt.ItemDataRole.ToolTipRole,
        )
        # tooltips for each mode
        self.model_combo.setItemData(
            1,
            (
                "EDGE (Edge-Detected Guided Estimator)\n"
                "• Detect stars in both channels with SEP\n"
                "• Keep only outer-ring stars (ignore center)\n"
                "• Try homography first for corner CA\n"
                "• If homography fails → try affine\n"
                "• If that fails → fall back to astroalign"
            ),
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            2,
            "Standard homography using astroalign matches (good general-purpose choice).",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            3,
            "Affine (shift + scale + rotate + shear). Good when channels are mostly parallel.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            4,
            "Polynomial (order 3). Use when you have mild field distortion.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            5,
            "Polynomial (order 4). Use for stronger distortion, but needs more/better matches.",
            Qt.ItemDataRole.ToolTipRole,
        )
        hl.addWidget(self.model_combo)
        lay.addLayout(hl)

        # ── SEP controls ─────────────────────────
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel(self.tr("SEP sigma:")))

        self.sep_spin = QSpinBox()
        self.sep_spin.setRange(1, 100)
        self.sep_spin.setValue(5)   # default; 1.5 was too hungry
        self.sep_spin.setToolTip("Detection threshold (σ) for SEP star finding in EDGE mode.\n"
                                 "Higher = fewer stars, lower = more stars.")
        sep_row.addWidget(self.sep_spin)

        self.btn_trial_sep = QPushButton(self.tr("Trial detect stars"))
        self.btn_trial_sep.setToolTip("Run SEP on the green channel with this sigma and report how many "
                                      "stars it finds and how many are in the EDGE ring.")
        self.btn_trial_sep.clicked.connect(self._trial_sep_detect)
        sep_row.addWidget(self.btn_trial_sep)

        lay.addLayout(sep_row)


        self.chk_new_doc = QCheckBox(self.tr("Create new document (keep original)"))
        self.chk_new_doc.setChecked(True)
        lay.addWidget(self.chk_new_doc)

        # progress
        self.progress_label = QLabel("Idle.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.progress_bar)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Transform summary will appear here…")
        self.summary_box.setMinimumHeight(140)
        # optional: monospace
        self.summary_box.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        lay.addWidget(self.summary_box)

        btns = QHBoxLayout()
        self.btn_run = QPushButton(self.tr("Align"))

        self.btn_manual = QPushButton(self.tr("Manual…"))
        self.btn_manual.setToolTip("Manual planetary RGB alignment (nudge channels by 1 px with preview).")

        self.btn_manual.clicked.connect(self._open_manual)

        self.btn_close = QPushButton(self.tr("Close"))
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_manual)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.btn_run.clicked.connect(self._start_align)
        self.btn_close.clicked.connect(self.close)

        self.worker: RGBAlignWorker | None = None

    def _open_manual(self):
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return

        dlg = RGBAlignManualDialog(self, self.image, planet_roi=("planet" in self.model_combo.currentText().lower()))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        rdx, rdy, bdx, bdy = dlg.result_offsets()
        self.summary_box.setPlainText(
            "[Manual Planetary]\n"
            f"R shift: dx={rdx}, dy={rdy}\n"
            f"B shift: dx={bdx}, dy={bdy}\n"
        )

        # apply full-res
        img = self.image
        R = img[..., 0].astype(np.float32, copy=False)
        G = img[..., 1].astype(np.float32, copy=False)
        B = img[..., 2].astype(np.float32, copy=False)

        R2 = _translate_channel(R, rdx, rdy)
        B2 = _translate_channel(B, bdx, bdy)

        out = np.stack([R2, G, B2], axis=2)
        if out.dtype != img.dtype:
            out = out.astype(img.dtype, copy=False)

        # same create-new-doc logic you already use
        try:
            if self.chk_new_doc.isChecked():
                dm = getattr(self.parent, "docman", None)
                if dm is not None:
                    dm.open_array(out, {"display_name": "RGB Aligned (Manual)"}, title="RGB Aligned (Manual)")
                else:
                    if hasattr(self.doc, "apply_edit"):
                        self.doc.apply_edit(out, {"step_name": "RGB Align (Manual)"}, step_name="RGB Align (Manual)")
                    else:
                        self.doc.image = out
            else:
                if hasattr(self.doc, "apply_edit"):
                    self.doc.apply_edit(out, {"step_name": "RGB Align (Manual)"}, step_name="RGB Align (Manual)")
                else:
                    self.doc.image = out
        except Exception as e:
            QMessageBox.warning(self, "RGB Align", f"Manual aligned image created, but applying failed:\n{e}")


    def _trial_sep_detect(self):
        if self.image is None:
            QMessageBox.warning(self, "RGB Align", "No image loaded.")
            return
        if sep is None:
            QMessageBox.warning(self, "RGB Align", "python-sep is not available.")
            return
        self.progress_label.setText(f"Trial Detection In Progress…")
        QApplication.processEvents()
        # use green channel as reference, same as align
        G = np.ascontiguousarray(self.image[..., 1].astype(np.float32, copy=False))
        sigma = float(self.sep_spin.value())

        # run a mini version of what the worker does
        bkg = sep.Background(G)
        data_sub = G - bkg
        objs = sep.extract(data_sub, sigma, err=bkg.globalrms)
        total = 0 if objs is None else len(objs)

        # compute how many are in the EDGE ring, using same logic/constants
        h, w = G.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        rmax = np.hypot(cx, cy)
        edge_inner = RGBAlignWorker.EDGE_INNER_FRAC * rmax

        if objs is not None and total > 0:
            r = np.hypot(objs["x"] - cx, objs["y"] - cy)
            edge_mask = r >= edge_inner
            edge_count = int(edge_mask.sum())
        else:
            edge_count = 0

        msg = (f"[Trial SEP]\n"
               f"sigma = {sigma}\n"
               f"total stars (green): {total}\n"
               f"outer-ring stars (used by EDGE): {edge_count}")
        self.summary_box.setPlainText(msg)
        self.progress_label.setText(f"Trial SEP: {total} stars, {edge_count} edge")


    def _start_align(self):
        if self.image is None:
            QMessageBox.warning(self, "RGB Align", "No image found in active view.")
            return
        if self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return
        if astroalign is None:
            QMessageBox.warning(self, "RGB Align", "astroalign is not available.")
            return

        model = self._selected_model()
        sep_sigma = float(self.sep_spin.value())
        self.progress_label.setText("Starting…")
        self.progress_bar.setValue(0)

        self.worker = RGBAlignWorker(self.image, model, sep_sigma=sep_sigma)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.done.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.start()
        self.btn_run.setEnabled(False)

    def _selected_model(self) -> str:
        txt = self.model_combo.currentText().lower()
        if "planet" in txt or "centroid" in txt or "adc" in txt:
            return "planet-centroid"
        if "edge" in txt:
            return "edge-sep"
        if "affine" in txt:
            return "affine"
        if "poly 3" in txt:
            return "poly3"
        if "poly 4" in txt:
            return "poly4"
        if "homography" in txt:
            return "homography"
        return "edge-sep"

    # slots
    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Failed.")
        QMessageBox.critical(self, "RGB Align", err)

    def _on_worker_done(self, out: np.ndarray):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Applying…")

        summary_lines = []
        w = self.worker  # type: ignore

        if w is not None:
            def _fmt_mat(M):
                return "\n".join(
                    ["    " + "  ".join(f"{v: .6f}" for v in row) for row in M]
                )

            def _spread_stats(pts, shape):
                if pts is None:
                    return "  points: 0"
                pts = np.asarray(pts, dtype=float)
                if pts.size == 0:
                    return "  points: 0"
                h, w_ = shape[:2]
                cx, cy = w_ * 0.5, h * 0.5
                if pts.ndim != 2 or pts.shape[1] != 2:
                    return f"  points: {len(pts)} (unusual shape {pts.shape})"
                r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                rmax = np.hypot(cx, cy)
                edge = r > (RGBAlignWorker.EDGE_FRAC * rmax)
                return (
                    f"  points: {len(pts)} "
                    f"(edge: {edge.sum()} ≥{RGBAlignWorker.EDGE_FRAC*100:.0f}%Rmax)"
                )

            h_img, w_img = self.image.shape[:2]

            # ── R → G ──
            if w.r_xform is not None:
                kind, X = w.r_xform
                summary_lines.append("Red → Green:")
                if w.r_pairs is not None and len(w.r_pairs) == 2:
                    summary_lines.append(_spread_stats(w.r_pairs[1], (h_img, w_img)))
                summary_lines.append(f"  model: {kind}")
                if kind == "translate":
                    dx, dy = X
                    summary_lines.append(f"  shift: dx={dx:.3f}, dy={dy:.3f}")                
                elif kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    Hm = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(Hm))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

            # ── B → G ──
            if w.b_xform is not None:
                kind, X = w.b_xform
                summary_lines.append("")
                summary_lines.append("Blue → Green:")
                if w.b_pairs is not None and len(w.b_pairs) == 2:
                    summary_lines.append(_spread_stats(w.b_pairs[1], (h_img, w_img)))
                summary_lines.append(f"  model: {kind}")
                if kind == "translate":
                    dx, dy = X
                    summary_lines.append(f"  shift: dx={dx:.3f}, dy={dy:.3f}")                
                elif kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    Hm = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(Hm))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

        summary_text = "\n".join(summary_lines) if summary_lines else "No transform info."
        self.summary_box.setPlainText(summary_text)

        if self.parent is not None and hasattr(self.parent, "_log") and callable(self.parent._log):
            self.parent._log("[RGB Align]\n" + summary_text)

        try:
            if self.chk_new_doc.isChecked():
                dm = getattr(self.parent, "docman", None)
                if dm is not None:
                    dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
                else:
                    if hasattr(self.doc, "apply_edit"):
                        self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                    else:
                        self.doc.image = out
            else:
                if hasattr(self.doc, "apply_edit"):
                    self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                else:
                    self.doc.image = out

            self.progress_label.setText("Done.")
        except Exception as e:
            self.progress_label.setText("Apply failed.")
            QMessageBox.warning(self, "RGB Align", f"Aligned image created, but applying failed:\n{e}")





def align_rgb_array(img: np.ndarray, model: str = "edge-sep", sep_sigma: float = 3.0) -> np.ndarray:
    """
    Headless core: returns a new RGB image with R,B aligned to G.
    Raises RuntimeError on problems.
    """
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise RuntimeError("Image must be RGB (3 channels).")
    if astroalign is None:
        raise RuntimeError("astroalign is not available.")

    worker = RGBAlignWorker(img, model, sep_sigma=sep_sigma)

    try:
        R = np.ascontiguousarray(img[..., 0].astype(np.float32, copy=False))
        G = np.ascontiguousarray(img[..., 1].astype(np.float32, copy=False))
        B = np.ascontiguousarray(img[..., 2].astype(np.float32, copy=False))

        def _estimate_and_warp(src, ref):
            # NOTE: _estimate_transform now returns 3 values
            kind, X, _pairs = worker._estimate_transform(src, ref, model)
            return worker._warp_channel(src, kind, X, ref.shape)

        R_aligned = _estimate_and_warp(R, G)
        B_aligned = _estimate_and_warp(B, G)

        out = np.stack([R_aligned, G, B_aligned], axis=2)
        if img.dtype != out.dtype:
            out = out.astype(img.dtype, copy=False)
        return out
    except Exception as e:
        raise RuntimeError(str(e))

def run_rgb_align_headless(main_window, document, preset: dict | None = None):
    if document is None:
        QMessageBox.warning(main_window, "RGB Align", "No active document.")
        return

    img = np.asarray(document.image)
    p = dict(preset or {})
    model = p.get("model", "edge").lower()
    sep_sigma = float(p.get("sep_sigma", 3.0))
    create_new = bool(p.get("new_doc", False))

    sb = getattr(main_window, "statusBar", None)
    if callable(sb):
        sb().showMessage(f"RGB Align ({model})…", 3000)

    try:
        out = align_rgb_array(img, model=model if model != "edge" else "edge-sep",
                              sep_sigma=sep_sigma)
    except Exception as e:
        QMessageBox.critical(main_window, "RGB Align (headless)", str(e))
        return

    if create_new:
        dm = getattr(main_window, "docman", None)
        if dm is not None:
            dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
        else:
            # fallback to replace if we can't create new
            try:
                document.apply_edit(out, {"step_name": "RGB Align"})
            except Exception:
                document.image = out
    else:
        # in-place
        if hasattr(document, "apply_edit"):
            document.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
        else:
            document.image = out

    if callable(sb):
        sb().showMessage("RGB Align done.", 3000)


class RGBAlignManualDialog(QDialog):
    """
    Manual channel translation for ADC / dispersion.
    Adjust R and B offsets relative to G with 1px steps and live preview.
    """
    def __init__(self, parent=None, image: np.ndarray | None = None, planet_roi: bool = False):
        super().__init__(parent)
        self.setWindowTitle("RGB Align — Manual (Planetary)")
        self.setModal(True)

        self.image = np.asarray(image) if image is not None else None
        self._timer = None
        self._result = None  # (r_dx, r_dy, b_dx, b_dy)

        # offsets
        self.r_dx = 0
        self.r_dy = 0
        self.b_dx = 0
        self.b_dy = 0

        lay = QVBoxLayout(self)

        self.lbl = QLabel("Nudge Red and Blue so they line up with Green (1 px steps).")
        lay.addWidget(self.lbl)

        # preview label
        self.preview = QLabel()
        self.preview.setMinimumSize(420, 320)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        lay.addWidget(self.preview)

        # controls grid
        row = QHBoxLayout()
        lay.addLayout(row)

        # controls
        row = QHBoxLayout()
        lay.addLayout(row)

        self.red_pad = OffsetPad("Red offset (relative to Green):")
        self.blue_pad = OffsetPad("Blue offset (relative to Green):")

        self.red_pad.changed.connect(self._on_change)
        self.blue_pad.changed.connect(self._on_change)

        row.addWidget(self.red_pad)
        row.addWidget(self.blue_pad)


        # buttons
        btns = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_apply = QPushButton("Apply")
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.btn_reset.clicked.connect(self._reset)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_close.clicked.connect(self.reject)

        # build a reasonable ROI for preview (center crop)
        self._roi = None
        if self.image is not None and self.image.ndim == 3 and self.image.shape[2] >= 3:
            H, W = self.image.shape[:2]

            if planet_roi:
                # Use green channel to find the planet (most stable)
                c = _planet_centroid(self.image[..., 1])
                if c is not None:
                    cx, cy, area = c
                    # Estimate radius from area, then choose a padded ROI
                    r = max(32.0, np.sqrt(area / np.pi))
                    s = int(np.clip(r * 3.2, 160, 900))  # ROI size = ~3.2x radius
                    cx_i, cy_i = int(round(cx)), int(round(cy))
                else:
                    cx_i, cy_i = W // 2, H // 2
                    s = int(np.clip(min(H, W) * 0.45, 160, 900))
            else:
                cx_i, cy_i = W // 2, H // 2
                s = int(np.clip(min(H, W) * 0.45, 160, 900))

            x0 = max(0, cx_i - s // 2)
            y0 = max(0, cy_i - s // 2)
            x1 = min(W, x0 + s)
            y1 = min(H, y0 + s)
            self._roi = (x0, y0, x1, y1)


        # debounce timer
        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_preview)
        if self._roi is not None:
            x0,y0,x1,y1 = self._roi
            self.lbl.setText(self.lbl.text() + f"\nPreview ROI: x={x0}:{x1}, y={y0}:{y1}")
        self._render_preview()

    def result_offsets(self):
        return self._result

    def _on_change(self, *_):
        self._timer.start(60)

    def _reset(self):
        self.red_pad.reset()
        self.blue_pad.reset()

    def _apply(self):
        rdx, rdy = self.red_pad.offsets()
        bdx, bdy = self.blue_pad.offsets()
        self._result = (int(rdx), int(rdy), int(bdx), int(bdy))
        self.accept()

    def _render_preview(self):
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] < 3:
            self.preview.setText("No RGB image.")
            return

        x0, y0, x1, y1 = self._roi if self._roi is not None else (0, 0, self.image.shape[1], self.image.shape[0])
        crop = self.image[y0:y1, x0:x1, :3]

        R = crop[..., 0].astype(np.float32, copy=False)
        G = crop[..., 1].astype(np.float32, copy=False)
        B = crop[..., 2].astype(np.float32, copy=False)

        rdx, rdy = self.red_pad.offsets()
        bdx, bdy = self.blue_pad.offsets()

        R2 = _translate_channel(R, rdx, rdy)
        B2 = _translate_channel(B, bdx, bdy)

        out = np.stack([R2, G, B2], axis=2)

        # display: robust scale to uint8
        out8 = self._to_u8_preview(out)
        qimg = QImage(out8.data, out8.shape[1], out8.shape[0], out8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    @staticmethod
    def _to_u8_preview(rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32, copy=False)
        # percentile stretch per-channel, simple + robust
        out = np.empty_like(x, dtype=np.uint8)
        for c in range(3):
            ch = x[..., c]
            p1 = float(np.percentile(ch, 1.0))
            p99 = float(np.percentile(ch, 99.5))
            if p99 <= p1:
                out[..., c] = 0
            else:
                y = (ch - p1) * (255.0 / (p99 - p1))
                out[..., c] = np.clip(y, 0, 255).astype(np.uint8)
        return out

class OffsetPad(QWidget):
    changed = pyqtSignal(int, int)  # dx, dy

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._dx = 0
        self._dy = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        hdr = QLabel(title)
        hdr.setStyleSheet("font-weight:600;")
        outer.addWidget(hdr)

        self.lbl = QLabel("dx=0  dy=0")
        self.lbl.setStyleSheet("color:#bbb;")
        outer.addWidget(self.lbl)

        grid = QGridLayout()
        grid.setSpacing(2)

        self.btn_up = QPushButton("↑")
        self.btn_dn = QPushButton("↓")
        self.btn_lt = QPushButton("←")
        self.btn_rt = QPushButton("→")
        self.btn_mid = QPushButton("⟲")  # reset

        for b in (self.btn_up, self.btn_dn, self.btn_lt, self.btn_rt, self.btn_mid):
            b.setFixedSize(34, 34)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        grid.addWidget(self.btn_up, 0, 1)
        grid.addWidget(self.btn_lt, 1, 0)
        grid.addWidget(self.btn_mid, 1, 1)
        grid.addWidget(self.btn_rt, 1, 2)
        grid.addWidget(self.btn_dn, 2, 1)

        outer.addLayout(grid)

        # connections
        self.btn_up.clicked.connect(lambda: self.nudge(0, -1))
        self.btn_dn.clicked.connect(lambda: self.nudge(0, +1))
        self.btn_lt.clicked.connect(lambda: self.nudge(-1, 0))
        self.btn_rt.clicked.connect(lambda: self.nudge(+1, 0))
        self.btn_mid.clicked.connect(self.reset)

    def set_offsets(self, dx: int, dy: int):
        self._dx = int(dx)
        self._dy = int(dy)
        self._update_label()
        self.changed.emit(self._dx, self._dy)

    def offsets(self):
        return (self._dx, self._dy)

    def reset(self):
        self.set_offsets(0, 0)

    def nudge(self, ddx: int, ddy: int, step: int = 1):
        self._dx += int(ddx) * int(step)
        self._dy += int(ddy) * int(step)
        self._update_label()
        self.changed.emit(self._dx, self._dy)

    def _update_label(self):
        self.lbl.setText(f"dx={self._dx:+d}  dy={self._dy:+d}")
