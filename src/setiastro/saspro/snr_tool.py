from __future__ import annotations

import numpy as np

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy.ma as ma

import sep
import time
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.vizier import Vizier

from PyQt6.QtCore import Qt, QRect, QEvent, QPointF, QRectF, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QWidget, QTextEdit, 
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,QComboBox, QGraphicsPathItem, QGraphicsEllipseItem,
    QMessageBox, QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem
)
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QPainterPath, QIcon

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.wcs.wcs import NoConvergence
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# IMPORTANT: use the centralized one (adjust import path to where you moved it)
from setiastro.saspro.sfcc import pickles_match_for_simbad

# Reuse useful SFCC pieces
from setiastro.saspro.sfcc import non_blocking_sleep  # already used in SFCC; optional
from setiastro.saspro.backgroundneutral import auto_rect_box, auto_rect_50x50
from setiastro.saspro.imageops.stretch import stretch_color_image
# We *intentionally* do NOT reuse SFCC pedestal-removal/clamp for photometry.

import socket

import multiprocessing as mp
import queue
import traceback

_EPS = 1e-12


def _to_float01_native(img: np.ndarray) -> np.ndarray:
    """
    Convert doc.image to float32 in [0,1] without changing channel count.
    - 2D stays 2D (mono)
    - 3D stays 3D (RGB or 1-channel)
    """
    a = np.asarray(img)
    if a.size == 0:
        raise ValueError("Empty image.")

    if a.dtype.kind in "ui":
        mx = float(np.iinfo(a.dtype).max)
        a = a.astype(np.float32) / (mx if mx > 0 else 1.0)
    else:
        a = a.astype(np.float32, copy=False)

    # keep native
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (1, 3) or (a.ndim == 3 and a.shape[2] >= 3):
        return a[..., :3] if a.shape[2] >= 3 else a
    raise ValueError(f"Unsupported image shape: {a.shape}")

def _stats_on_mask(img01: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """
    Returns per-channel (median, stddev) for pixels inside mask.
    Mono returns a 1-element list.
    """
    if img01.ndim == 2:
        vals = img01[mask]
        if vals.size < 10:
            return [(float("nan"), float("nan"))]
        med = float(np.median(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return [(med, std)]

    # 3D
    C = 1 if img01.shape[2] == 1 else img01.shape[2]
    out: list[tuple[float, float]] = []
    for c in range(C):
        plane = img01[..., 0] if img01.shape[2] == 1 else img01[..., c]
        vals = plane[mask]
        if vals.size < 10:
            out.append((float("nan"), float("nan")))
            continue
        med = float(np.median(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        out.append((med, std))
    return out

def _native_channel_count(img01: np.ndarray) -> int:
    if img01.ndim == 2:
        return 1
    if img01.ndim == 3:
        if img01.shape[2] == 1:
            return 1
        return 3 if img01.shape[2] >= 3 else img01.shape[2]
    return 1

def _bbox_from_mask(mask: np.ndarray) -> QRect:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return QRect()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return QRect(x0, y0, (x1 - x0 + 1), (y1 - y0 + 1))


def _expand_rect(r: QRect, margin: int, W: int, H: int) -> QRect:
    if r.isNull():
        return QRect()
    x0 = max(0, r.x() - margin)
    y0 = max(0, r.y() - margin)
    x1 = min(W, r.x() + r.width() + margin)
    y1 = min(H, r.y() + r.height() + margin)
    return QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _get_channel_plane(img01: np.ndarray, c: int) -> np.ndarray:
    """Return 2D plane for channel c for mono or RGB."""
    if img01.ndim == 2:
        return img01
    if img01.ndim == 3:
        if img01.shape[2] == 1:
            return img01[..., 0]
        return img01[..., c]
    return img01



def _mask_from_qrect(H: int, W: int, qrect) -> np.ndarray:
    # qrect is QRect
    x, y, w, h = qrect.x(), qrect.y(), qrect.width(), qrect.height()
    x0 = max(0, min(W, x))
    y0 = max(0, min(H, y))
    x1 = max(0, min(W, x + w))
    y1 = max(0, min(H, y + h))
    m = np.zeros((H, W), dtype=bool)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


class SNRToolDialog(QDialog):
    """
    Object SNR tool.
    Uses ONE target mask + an auto background rectangle (like Magnitude tool).
    """
    def __init__(self, parent, doc_manager, icon: QIcon | None = None):
        super().__init__(parent)
        self._main = parent
        self.doc_manager = doc_manager

        self.object_mask: np.ndarray | None = None
        self.bg_rect = None  # QRect

        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.setWindowTitle("Object SNR Calculation")
        self.resize(640, 420)

        v = QVBoxLayout(self)

        self.lbl = QLabel(
            "Pick a target region (signal of interest). Background will be auto-selected.\n"
            "SNR is computed per-channel using median(signal) − median(background), with noise from background stddev."
        )
        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)

        row = QHBoxLayout()
        self.btn_pick = QPushButton("Select Target Region…")
        self.btn_calc = QPushButton("Calculate SNR")
        self.btn_copy = QPushButton("Copy Results")
        row.addWidget(self.btn_pick)
        row.addWidget(self.btn_calc)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        v.addLayout(row)

        mid = QHBoxLayout()

        # left: text output
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        mid.addWidget(self.out, 2)

        # right: plots
        plots_box = QVBoxLayout()

        # channel selector for histogram (updates based on image)
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel:"))
        self.cmb_plot_channel = QComboBox()
        self.cmb_plot_channel.addItems(["L"])  # will be refreshed when image is known
        self.cmb_plot_channel.currentIndexChanged.connect(self._refresh_distribution_plot)
        ch_row.addWidget(self.cmb_plot_channel)
        ch_row.addStretch(1)
        plots_box.addLayout(ch_row)

        # tabs for plots
        self.tabs_plots = QTabWidget()

        # --- Tab 1: Distributions ---
        self.fig_dist = Figure(figsize=(4.0, 3.0))
        self.canvas_dist = FigureCanvas(self.fig_dist)
        dist_tab = QWidget()
        dist_tab_l = QVBoxLayout(dist_tab)
        dist_tab_l.setContentsMargins(0, 0, 0, 0)
        dist_tab_l.addWidget(self.canvas_dist)
        self.tabs_plots.addTab(dist_tab, "Distributions")
        # --- Tab 2: SNR bars ---
        self.fig_snr = Figure(figsize=(4.0, 3.0))
        self.canvas_snr = FigureCanvas(self.fig_snr)
        snr_tab = QWidget()
        snr_tab_l = QVBoxLayout(snr_tab)
        snr_tab_l.setContentsMargins(0, 0, 0, 0)
        snr_tab_l.addWidget(self.canvas_snr)
        self.tabs_plots.addTab(snr_tab, "SNR Chart")
        # --- Tab 3: ROI Preview ---
        self.lbl_roi = QLabel("ROI preview will appear after you run Calculate.")
        self.lbl_roi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_roi.setMinimumHeight(220)
        self.lbl_roi.setStyleSheet("border: 1px solid rgba(255,255,255,40); padding: 6px;")

        roi_tab = QWidget()
        roi_tab_l = QVBoxLayout(roi_tab)
        roi_tab_l.setContentsMargins(6, 6, 6, 6)
        roi_tab_l.addWidget(self.lbl_roi, 1)

        self.tabs_plots.addTab(roi_tab, "ROI Preview")

        # Keep a reference to the pixmap so it doesn't get GC'ed weirdly
        self._roi_pix = None
        plots_box.addWidget(self.tabs_plots, 1)

        mid.addLayout(plots_box, 1)
        v.addLayout(mid, 1)

        # cached last calc for plot refresh
        self._last_calc = None  # dict or None


        self.btn_close = QPushButton("Close")
        v.addWidget(self.btn_close, 0, Qt.AlignmentFlag.AlignRight)

        self.btn_pick.clicked.connect(self._pick_region)
        self.btn_calc.clicked.connect(self._calculate)
        self.btn_copy.clicked.connect(self._copy)
        self.btn_close.clicked.connect(self.close)

        self._write_header()
        self._refresh_distribution_plot()
        self._plot_snr_bars(["L"], [0.0], [0.0])

    def _active_doc(self):
        d = self.doc_manager.get_active_document()
        return d

    def _doc_image_float01(self) -> np.ndarray:
        doc = self._active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            raise ValueError("No active image.")
        return _to_float01_native(getattr(doc, "image"))


    def _write_header(self):
        self.out.setPlainText("Object SNR\n")


    # called by RegionPickerDialog (matches your Magnitude dialog pattern)
    def set_object_mask(self, mask: np.ndarray):
        self.object_mask = mask

    def set_background_rect(self, qrect):
        self.bg_rect = qrect

    def _pick_region(self):
        try:
            _ = self._doc_image_float01()  # validate image
        except Exception as e:
            QMessageBox.warning(self, "SNR", str(e))
            return

        dlg = RegionPickerDialog(parent=self, doc_manager=self.doc_manager, icon=self.windowIcon())
        # RegionPickerDialog should call back into set_object_mask + set_background_rect
        dlg.show()

    def _calculate(self):
        try:
            img = self._doc_image_float01()
        except Exception as e:
            QMessageBox.warning(self, "SNR", str(e))
            return

        if self.object_mask is None or int(self.object_mask.sum()) < 25:
            QMessageBox.warning(self, "SNR", "No target selected. Click 'Select Target Region…' first.")
            return

        H, W = img.shape[:2]
        if self.bg_rect is None:
            QMessageBox.warning(self, "SNR", "Background rectangle not set. Use the picker (it auto-finds background).")
            return
        self._ensure_plot_channels(img)
        bg_mask = _mask_from_qrect(H, W, self.bg_rect)

        # sanity
        if int(bg_mask.sum()) < 25:
            QMessageBox.warning(self, "SNR", "Background region too small.")
            return

        sig_stats = _stats_on_mask(img, self.object_mask)  # (median,std) but we only use median
        bg_stats  = _stats_on_mask(img, bg_mask)

        lines = []
        lines.append("Calculating Improved SNR\n")
        # Determine channel count + names based on native image
        if img.ndim == 2:
            ch_names = ["L"]
            C = 1
        elif img.ndim == 3 and img.shape[2] == 1:
            ch_names = ["L"]
            C = 1
        else:
            ch_names = ["R", "G", "B"]
            C = 3
        snr_vals: list[float] = []
        snr_db_vals: list[float] = []
        for c in range(C):
            Stotal = sig_stats[c][0]
            Bmed   = bg_stats[c][0]
            Bstd   = bg_stats[c][1]

            So = max(Stotal - Bmed, 0.0)
            noise = max(Bstd, _EPS)

            snr = So / noise
            snr_db = 10.0 * np.log10(max(snr, _EPS))  # matches your PI script
            snr_vals.append(float(snr))
            snr_db_vals.append(float(snr_db))
            lines.append(
                f"{ch_names[c]}: "
                f"Stotal(med)={Stotal:.6f}  "
                f"B(med)={Bmed:.6f}  "
                f"noise(std)={Bstd:.6f}  "
                f"So={So:.6f}  "
                f"SNR={snr:.3e}  {snr_db:.2f} dB"
            )

        lines.append("\nDone.")
        self.out.setPlainText("\n".join(lines))
        self._last_calc = {
            "img01": img,
            "obj_mask": self.object_mask.copy(),
            "bg_mask": bg_mask,
        }

        # update plots
        self._plot_snr_bars(ch_names, snr_vals, snr_db_vals)
        self._refresh_distribution_plot()
        self._update_roi_preview()

    def _ensure_plot_channels(self, img01: np.ndarray):
        """Update channel dropdown to match the active image (L or R/G/B)."""
        C = _native_channel_count(img01)
        names = ["L"] if C == 1 else ["R", "G", "B"]

        cur = self.cmb_plot_channel.currentText() if self.cmb_plot_channel.count() else ""
        self.cmb_plot_channel.blockSignals(True)
        self.cmb_plot_channel.clear()
        self.cmb_plot_channel.addItems(names)
        # restore selection if possible
        if cur in names:
            self.cmb_plot_channel.setCurrentText(cur)
        self.cmb_plot_channel.blockSignals(False)


    def _plot_snr_bars(self, ch_names: list[str], snr_vals: list[float], snr_db_vals: list[float]):
        self.fig_snr.clear()
        ax = self.fig_snr.add_subplot(111)

        x = np.arange(len(ch_names), dtype=float)
        # Plot SNR (linear). For readability, we plot linear but clamp extreme values a bit.
        snr_plot = np.array(snr_vals, dtype=float)
        snr_plot = np.clip(snr_plot, 0.0, np.nanmax(snr_plot) if np.isfinite(np.nanmax(snr_plot)) else 1.0)

        ax.bar(x - 0.15, snr_plot, width=0.3, label="SNR")
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names)
        ax.set_ylabel("SNR (linear)")

        # Second axis for dB
        ax2 = ax.twinx()
        ax2.bar(x + 0.15, snr_db_vals, width=0.3, label="dB")
        ax2.set_ylabel("SNR (dB)")

        ax.set_title("Object SNR")
        ax.grid(True, axis="y", alpha=0.25)

        # Simple legend handling across twin axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

        self.canvas_snr.draw_idle()

    def _roi_display_rgb01(self, crop01: np.ndarray) -> np.ndarray:
        """
        Convert a crop (mono or rgb) into display-ready RGB in [0,1].
        Uses your stretch_color_image for visibility.
        """
        a = np.asarray(crop01, dtype=np.float32)

        # ensure RGB for display
        if a.ndim == 2:
            a = np.dstack([a, a, a])
        elif a.ndim == 3 and a.shape[2] == 1:
            m = a[..., 0]
            a = np.dstack([m, m, m])
        elif a.ndim == 3 and a.shape[2] >= 3:
            a = a[..., :3]
        else:
            # fallback
            a = np.dstack([a, a, a])

        # stretch for view (doesn't affect measurement)
        try:
            disp = stretch_color_image(
                a,
                target_median=0.35,
                linked=True,
                normalize=False,
                apply_curves=False,
                curves_boost=0.0,
                blackpoint_sigma=3.5,
                no_black_clip=False,
                hdr_compress=False,
                hdr_amount=0.0,
                hdr_knee=0.75,
                luma_only=False,
                high_range=False,
            )
            return np.clip(disp, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)

    def _update_roi_preview(self):
        """
        Build a small cropped preview around the target + overlays.
        Requires: self._last_calc present.
        """
        if not self._last_calc:
            self.lbl_roi.setText("ROI preview will appear after you run Calculate.")
            self.lbl_roi.setPixmap(QPixmap())
            self._roi_pix = None
            return

        img01 = self._last_calc["img01"]
        obj_mask = self._last_calc["obj_mask"]
        bg_mask = self._last_calc["bg_mask"]

        H, W = img01.shape[:2]

        # bbox from target mask (tight)
        bbox = _bbox_from_mask(obj_mask)
        if bbox.isNull():
            self.lbl_roi.setText("No target mask (ROI preview unavailable).")
            self.lbl_roi.setPixmap(QPixmap())
            self._roi_pix = None
            return

        # expand a bit so you can see context
        crop_rect = _expand_rect(bbox, margin=40, W=W, H=H)

        # crop image + masks
        x, y, w, h = crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()

        crop_img = img01[y:y+h, x:x+w]
        crop_obj = obj_mask[y:y+h, x:x+w]
        crop_bg  = bg_mask[y:y+h, x:x+w]

        disp = self._roi_display_rgb01(crop_img)

        # make RGBA buffer for overlays
        rgb8 = np.ascontiguousarray((disp * 255.0).astype(np.uint8))
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb8
        rgba[..., 3] = 255

        # overlay: object mask in red
        # overlay: object mask in red (alpha blend)
        alpha_obj = 0.25  # <-- OPACITY: 0.0 transparent, 1.0 solid red
        obj_color = np.array([255, 0, 0], dtype=np.float32)

        base = rgba[..., :3].astype(np.float32)
        m = crop_obj

        base[m] = (1.0 - alpha_obj) * base[m] + alpha_obj * obj_color
        rgba[..., :3] = np.clip(base, 0, 255).astype(np.uint8)

        # overlay: background mask in gold-ish tint (subtle)
        gold = np.array([255, 215, 0], dtype=np.uint8)
        bg_idx = crop_bg & (~crop_obj)
        rgba[bg_idx, 0] = ((rgba[bg_idx, 0].astype(np.uint16) + gold[0]) // 2).astype(np.uint8)
        rgba[bg_idx, 1] = ((rgba[bg_idx, 1].astype(np.uint16) + gold[1]) // 2).astype(np.uint8)
        rgba[bg_idx, 2] = ((rgba[bg_idx, 2].astype(np.uint16) + gold[2]) // 2).astype(np.uint8)

        qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())  # copy to detach from numpy buffer
        self._roi_pix = pix

        # scale to label
        scaled = pix.scaled(self.lbl_roi.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_roi.setPixmap(scaled)
        self.lbl_roi.setText("")


    def _refresh_distribution_plot(self):
        """Redraw distribution plot from cached last calculation."""
        if not self._last_calc:
            # show empty
            self.fig_dist.clear()
            ax = self.fig_dist.add_subplot(111)
            ax.set_title("Object vs Background (run Calculate)")
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas_dist.draw_idle()
            return

        img01 = self._last_calc["img01"]
        obj_mask = self._last_calc["obj_mask"]
        bg_mask = self._last_calc["bg_mask"]

        C = _native_channel_count(img01)
        names = ["L"] if C == 1 else ["R", "G", "B"]
        ch = self.cmb_plot_channel.currentText()
        c = 0 if (C == 1) else max(0, min(2, names.index(ch) if ch in names else 0))

        obj_vals = _get_channel_plane(img01, c)[obj_mask]
        bg_vals  = _get_channel_plane(img01, c)[bg_mask]

        # downsample for speed
        maxn = 200_000
        if obj_vals.size > maxn:
            obj_vals = obj_vals[np.random.default_rng(0).choice(obj_vals.size, maxn, replace=False)]
        if bg_vals.size > maxn:
            bg_vals = bg_vals[np.random.default_rng(1).choice(bg_vals.size, maxn, replace=False)]

        self.fig_dist.clear()
        ax = self.fig_dist.add_subplot(111)

        # choose bins based on combined data range
        allv = np.concatenate([obj_vals.astype(np.float32, copy=False), bg_vals.astype(np.float32, copy=False)], axis=0)
        lo = float(np.nanpercentile(allv, 0.5)) if allv.size else 0.0
        hi = float(np.nanpercentile(allv, 99.5)) if allv.size else 1.0
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi <= lo:
            hi = lo + 1e-6

        bins = 80
        ax.hist(bg_vals, bins=bins, range=(lo, hi), alpha=0.5, label="Background", density=True)
        ax.hist(obj_vals, bins=bins, range=(lo, hi), alpha=0.5, label="Object", density=True)

        ax.set_title(f"Distributions ({names[c]})")
        ax.set_xlabel("Pixel value (0–1)")
        ax.set_ylabel("Density")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")

        self.canvas_dist.draw_idle()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # rescale ROI pixmap if we have one
        if self._roi_pix is not None and not self._roi_pix.isNull():
            scaled = self._roi_pix.scaled(self.lbl_roi.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_roi.setPixmap(scaled)

    def _copy(self):
        try:
            txt = self.out.toPlainText().strip()
            if not txt:
                return
            cb = self.parent().clipboard() if hasattr(self.parent(), "clipboard") else None
        except Exception:
            cb = None

        # simplest: QApplication clipboard
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(self.out.toPlainText())
        except Exception:
            QMessageBox.information(self, "Copy", "Could not access clipboard.")


class RegionPickerDialog(QDialog):
    """
    Pick ONE target rectangle (object). Background is auto-selected via auto_rect_50x50().
    Preview can be toggled to ABE hard_autostretch to see dim regions on linear data.
    """
    def __init__(self, parent, doc_manager, icon=None):
        super().__init__(parent)
        self._main = parent
        self.doc_manager = doc_manager
        self.doc = self.doc_manager.get_active_document()
        self._path = QPainterPath()
        self._path_item: QGraphicsPathItem | None = None
        self._ellipse_item: QGraphicsEllipseItem | None = None
        self._pen_live = QPen(QColor(0, 255, 0), 3, Qt.PenStyle.DashLine)
        self._pen_live.setCosmetic(True)
        self._pen_final = QPen(QColor(255, 0, 0), 3)
        self._pen_final.setCosmetic(True)
        self._has_final_selection = False
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.setWindowTitle("SNR Tool — Select Target Region")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(900, 600)

        self.auto_stretch = True
        self.zoom_factor = 1.0
        self._user_zoomed = False

        self.target_rect_scene = QRectF()
        self.target_item: QGraphicsRectItem | None = None
        self.bg_item: QGraphicsRectItem | None = None
        self._drawing = False
        self._origin_scene = QPointF()

        # --- scene/view ---
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._zoom_debounce_ms = 70
        self._interactive_timer = QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._end_interactive_present)

        self._interactive_active = False

        # Make interactive updates cheaper (optional but helps a lot)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # --- layout ---
        v = QVBoxLayout(self)
        self.lbl = QLabel("Draw a Target region (Box/Ellipse/Freehand). Background will be auto-selected (gold).")

        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Box", "Ellipse", "Freehand"])
        v.insertWidget(1, self.mode_combo)  # under label, above view

        v.addWidget(self.view, 1)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Use Target")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_find_bg = QPushButton("Find Background")
        self.btn_toggle = QPushButton("Disable Auto-Stretch" if self.auto_stretch else "Enable Auto-Stretch")

        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_toggle)
        btn_row.addWidget(self.btn_find_bg)
        v.addLayout(btn_row)

        from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_100)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addStretch(1)
        v.addLayout(zoom_row)


        # wiring
        self.btn_cancel.clicked.connect(self.close)
        self.btn_toggle.clicked.connect(self._toggle_autostretch)
        self.btn_find_bg.clicked.connect(self._on_find_background)
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(0.8))
        self.btn_zoom_100.clicked.connect(self.zoom_100)
        self.btn_zoom_fit.clicked.connect(self.fit_to_view)
        self.btn_apply.clicked.connect(self._on_use_target)

        # mouse events
        self.view.viewport().installEventFilter(self)

        # active doc tracking (optional, matches your style)
        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False
        self.destroyed.connect(lambda _=None: self._cleanup_connections())

        self._pixmap_item = None
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._load_image()
        QTimer.singleShot(0, self.fit_to_view)

    # ---------- doc helpers ----------
    def _begin_interactive_present(self):
        """
        Switch to FAST transform while user is actively zooming/panning.
        """
        if self._interactive_active:
            # restart debounce
            self._interactive_timer.start(self._zoom_debounce_ms)
            return

        self._interactive_active = True

        # FAST path: disable smooth pixmap transform
        try:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        except Exception:
            pass

        # keep debounce running
        self._interactive_timer.start(self._zoom_debounce_ms)

    def _end_interactive_present(self):
        """
        Restore SMOOTH transform after interaction stops, then repaint once.
        """
        self._interactive_active = False
        try:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        except Exception:
            pass

        # Force a final high-quality redraw
        try:
            self.view.viewport().update()
        except Exception:
            pass


    def _active_doc(self):
        d = self.doc_manager.get_active_document()
        return d if d is not None else self.doc

    def _doc_image_float01(self) -> np.ndarray:
        doc = self._active_doc()
        img = getattr(doc, "image", None)
        if img is None:
            raise ValueError("No active image.")

        img = np.asarray(img)
        if img.size == 0:
            raise ValueError("No active image.")

        # normalize integers → [0,1]
        if img.dtype.kind in "ui":
            mx = float(np.iinfo(img.dtype).max)
            img = img.astype(np.float32) / (mx if mx > 0 else 1.0)
        else:
            img = img.astype(np.float32, copy=False)

        # ---- allow mono for ROI picking (convert to RGB for DISPLAY only) ----
        if img.ndim == 2:
            return np.dstack([img, img, img]).astype(np.float32, copy=False)

        if img.ndim == 3:
            if img.shape[2] == 1:
                m = img[..., 0]
                return np.dstack([m, m, m]).astype(np.float32, copy=False)
            if img.shape[2] >= 3:
                return img[..., :3].astype(np.float32, copy=False)

        raise ValueError(f"Unsupported image shape for ROI picker: {img.shape}")

    def _capture_view_state(self):
        """
        Capture current view transform + center point in scene coords.
        """
        try:
            t = self.view.transform()
            center_scene = self.view.mapToScene(self.view.viewport().rect().center())
            return (t, center_scene)
        except Exception:
            return None

    def _restore_view_state(self, state):
        """
        Restore transform + center point.
        """
        if not state:
            return
        try:
            t, center_scene = state
            self.view.setTransform(t)
            self.view.centerOn(center_scene)
        except Exception:
            pass

    def _on_mode_changed(self, _idx):
        # If the user changes drawing mode, clear any existing selection
        # so it’s always “exactly one region”.
        self._clear_target_items()

    def _display_rgb01(self, img_rgb01: np.ndarray) -> np.ndarray:
        # preview-only stretch
        if not self.auto_stretch:
            return np.clip(img_rgb01, 0.0, 1.0).astype(np.float32, copy=False)

        # Use SASpro canonical stretch for preview (non-destructive; does NOT affect photometry)
        try:
            disp = stretch_color_image(
                img_rgb01,
                target_median=0.35,
                linked=True,          # better visibility for NB/odd color balance data
                normalize=False,       # keep the look stable; you can flip to True if you want punchier preview
                apply_curves=False,    # keep it “honest” and fast
                curves_boost=0.0,
                blackpoint_sigma=3.5,  # roughly similar vibe to your old "sigma=2"
                no_black_clip=False,
                hdr_compress=False,
                hdr_amount=0.0,
                hdr_knee=0.75,
                luma_only=False,
                high_range=False,
            )
            return np.clip(disp, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            # worst case: just show clipped linear
            return np.clip(img_rgb01, 0.0, 1.0).astype(np.float32, copy=False)


    # ---------- render ----------
    def _load_image(self, preserve_view_state=None, keep_overlays=True):
        try:
            img = self._doc_image_float01()
        except Exception as e:
            QMessageBox.warning(self, "No Image", str(e))
            self.close()
            return

        disp = self._display_rgb01(img)
        h, w, _ = disp.shape

        self._disp_buf8 = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
        qimg = QImage(self._disp_buf8.data, w, h, self._disp_buf8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        if self._pixmap_item is None or (not keep_overlays):
            self.scene.clear()
            self._pixmap_item = self.scene.addPixmap(pix)
            self._pixmap_item.setPos(0, 0)
        else:
            self._pixmap_item.setPixmap(pix)
            self._pixmap_item.setPos(0, 0)

        self.scene.setSceneRect(0, 0, pix.width(), pix.height())

        if preserve_view_state is not None:
            self._restore_view_state(preserve_view_state)
        else:
            self.view.resetTransform()
            self.view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._user_zoomed = False

        # If you're keeping overlays, DON'T blow away bg/target here.
        # Only auto-find bg if bg doesn't exist.
        if self.bg_item is None:
            self._on_find_background()


    def _toggle_autostretch(self):
        view_state = self._capture_view_state() if self._user_zoomed else None

        self.auto_stretch = not self.auto_stretch
        self.btn_toggle.setText("Disable Auto-Stretch" if self.auto_stretch else "Enable Auto-Stretch")

        # reload image but keep zoom/pan if user zoomed
        self._load_image(preserve_view_state=view_state, keep_overlays=True)

    def _zoom(self, factor: float):
        self._user_zoomed = True

        # start FAST interactive mode
        self._begin_interactive_present()

        cur = self.view.transform().m11()
        new_scale = cur * factor
        if new_scale < 0.01 or new_scale > 100.0:
            return

        self.view.scale(factor, factor)

    def zoom_100(self):
        self._user_zoomed = True
        self.view.resetTransform()
        self.view.scale(1.0, 1.0)


    def fit_to_view(self):
        self._user_zoomed = False
        self.view.resetTransform()
        if self._pixmap_item is not None:
            self.view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self.fit_to_view)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._user_zoomed:
            self.fit_to_view()

    # ---------- background auto-box ----------
    def _on_find_background(self):
        try:
            img = self._doc_image_float01()

            # Pull desired box size from the parent tool if available
            box = 50
            p = self.parent()
            if p is not None and hasattr(p, "bg_box_size"):
                try:
                    box = int(p.bg_box_size.value())
                except Exception:
                    box = 50

            x, y, w, h = auto_rect_box(img, box=box, margin=100)
            # (or auto_rect_50x50(img) if you want fixed behavior)
        except Exception as e:
            QMessageBox.warning(self, "Background", str(e))
            return

        if self.bg_item:
            try:
                self.scene.removeItem(self.bg_item)
            except Exception:
                pass

        pen = QPen(QColor(255, 215, 0), 3)  # gold
        pen.setCosmetic(True)
        rect_scene = QRectF(float(x), float(y), float(w), float(h))
        self.bg_item = self.scene.addRect(rect_scene, pen)


    def _target_mask(self) -> Optional[np.ndarray]:
        if self._pixmap_item is None:
            return None
        bounds = self._pixmap_item.boundingRect()
        W = int(bounds.width())
        H = int(bounds.height())
        if W <= 0 or H <= 0:
            return None

        mode = self.mode_combo.currentText()

        # start with empty mask
        mask_img = QImage(W, H, QImage.Format.Format_Grayscale8)
        mask_img.fill(0)

        p = QPainter(mask_img)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))

        if mode == "Box" and not self.target_rect_scene.isNull():
            p.drawRect(self.target_rect_scene.toRect())
        elif mode == "Ellipse" and not self.target_rect_scene.isNull():
            p.drawEllipse(self.target_rect_scene)
        elif mode == "Freehand" and not self._path.isEmpty():
            p.drawPath(self._path)

        p.end()

        ptr = mask_img.bits()
        ptr.setsize(mask_img.bytesPerLine() * H)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(H, mask_img.bytesPerLine())
        arr = arr[:, :W]
        return (arr > 0)


    # ---------- target drawing ----------
    def _clear_target_items(self):
        for it in (self.target_item, self._ellipse_item, self._path_item):
            if it is not None:
                try: self.scene.removeItem(it)
                except Exception: pass
        self.target_item = None
        self._ellipse_item = None
        self._path_item = None
        self.target_rect_scene = QRectF()
        self._path = QPainterPath()
        self._has_final_selection = False

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            et = event.type()

            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._has_final_selection:
                    self._clear_target_items()
                    self._has_final_selection = False
                self._drawing = True
                self._origin_scene = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                self._clear_target_items()

                if mode == "Freehand":
                    self._path = QPainterPath(self._origin_scene)
                    self._path_item = self.scene.addPath(self._path, self._pen_live)

                return True

            elif et == QEvent.Type.MouseMove and self._drawing:
                cur = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                if mode == "Box":
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self.target_item is None:
                        self.target_item = self.scene.addRect(self.target_rect_scene, self._pen_live)
                    else:
                        self.target_item.setRect(self.target_rect_scene)

                elif mode == "Ellipse":
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self._ellipse_item is None:
                        self._ellipse_item = self.scene.addEllipse(self.target_rect_scene, self._pen_live)
                    else:
                        self._ellipse_item.setRect(self.target_rect_scene)

                else:  # Freehand
                    self._path.lineTo(cur)
                    if self._path_item is not None:
                        self._path_item.setPath(self._path)

                return True

            elif et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self._drawing:
                self._drawing = False
                cur = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                if mode in ("Box", "Ellipse"):
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self.target_rect_scene.width() < 10 or self.target_rect_scene.height() < 10:
                        QMessageBox.warning(self, "Selection Too Small", "Please draw a larger target selection.")
                        self._clear_target_items()
                        return True

                    # finalize pen
                    if mode == "Box" and self.target_item is not None:
                        self.target_item.setPen(self._pen_final)
                    if mode == "Ellipse" and self._ellipse_item is not None:
                        self._ellipse_item.setPen(self._pen_final)

                else:  # Freehand
                    # close path and finalize
                    if self._path.elementCount() < 10:
                        QMessageBox.warning(self, "Selection Too Small", "Freehand selection too small.")
                        self._clear_target_items()
                        return True
                    self._path.closeSubpath()
                    if self._path_item is not None:
                        self._path_item.setPen(self._pen_final)
                        self._path_item.setPath(self._path)
                        

                return True

            elif et == QEvent.Type.Wheel:
                self._begin_interactive_present()
                angle = event.angleDelta().y()
                if angle == 0:
                    return True
                self._zoom(1.25 if angle > 0 else 0.8)
                return True

        return super().eventFilter(source, event)


    def _scene_rect_to_qrect(self, r: QRectF) -> QRect:
        if r is None or r.isNull():
            return QRect()
        bounds = self._pixmap_item.boundingRect() if self._pixmap_item else QRectF()
        W = int(bounds.width()); H = int(bounds.height())
        x = int(max(0.0, min(bounds.width(),  r.left())))
        y = int(max(0.0, min(bounds.height(), r.top())))
        w = int(max(1.0, min(bounds.width()  - x, r.width())))
        h = int(max(1.0, min(bounds.height() - y, r.height())))
        return QRect(x, y, w, h)

    def _on_use_target(self):
        mask = self._target_mask()
        if mask is None or int(mask.sum()) < 25:
            QMessageBox.warning(self, "No Target", "Draw a target selection first.")
            return

        # background is whatever is drawn (gold); if missing, recompute
        if self.bg_item is None:
            self._on_find_background()

        bgq = QRect()
        if self.bg_item is not None:
            bgq = self._scene_rect_to_qrect(self.bg_item.rect())

        # compute a bbox for info text (optional, but useful)
        try:
            ys, xs = np.nonzero(mask)
            if xs.size > 0 and ys.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox = QRect(x0, y0, (x1 - x0 + 1), (y1 - y0 + 1))
            else:
                bbox = QRect()
        except Exception:
            bbox = QRect()

        # push to parent MagnitudeToolDialog if it has setters
        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "set_object_mask"):
                parent.set_object_mask(mask)

            # (optional but recommended) also pass bg as mask if you add the setter
            if hasattr(parent, "set_background_rect"):
                parent.set_background_rect(bgq)

            # convenience: update label if present
            if hasattr(parent, "lbl_info"):
                parent.lbl_info.setText(
                    f"Target set: {int(mask.sum())} px"
                    + (f"  (bbox x={bbox.x()}, y={bbox.y()}, w={bbox.width()}, h={bbox.height()})" if not bbox.isNull() else "")
                    + "\n"
                    f"Background(auto): x={bgq.x()}, y={bgq.y()}, w={bgq.width()}, h={bgq.height()}"
                )

        self.close()


    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._load_image()

    def _cleanup_connections(self):
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False

    def closeEvent(self, e):
        try:
            self._cleanup_connections()
        finally:
            super().closeEvent(e)
