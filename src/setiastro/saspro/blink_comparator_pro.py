# pro/blink_comparator_pro.py
from __future__ import annotations

# ⬇️ keep your existing imports used by the code you pasted
import os
import re
import time
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional, List
from collections import defaultdict
# Qt
from PyQt6.QtCore import Qt, QTimer, QEvent, QPointF, QRectF, pyqtSignal, QSettings, QPoint, QCoreApplication
from PyQt6.QtGui import (QAction, QIcon, QImage, QPixmap, QBrush, QColor, QPalette,
                         QKeySequence, QWheelEvent, QShortcut, QDoubleValidator, QIntValidator)
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox, QProgressBar,
    QAbstractItemView, QMenu, QSplitter, QStyle, QScrollArea, QSlider, QDoubleSpinBox, QProgressDialog, QComboBox, QLineEdit, QApplication, QGridLayout, QCheckBox, QInputDialog,
    QMdiArea, QDialogButtonBox
)
from bisect import bisect_right
# 3rd-party (your code already expects these)
import cv2
import sep
import pyqtgraph as pg
from collections import OrderedDict
from setiastro.saspro.legacy.image_manager import load_image

from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image, siril_style_autostretch

from setiastro.saspro.legacy.numba_utils import debayer_fits_fast, debayer_raw_fast
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


from setiastro.saspro.star_metrics import measure_stars_sep

def _percentile_scale(arr, lo=0.5, hi=99.5):
    a = np.asarray(arr, dtype=np.float32)
    p1 = np.nanpercentile(a, lo)
    p2 = np.nanpercentile(a, hi)
    if not np.isfinite(p1) or not np.isfinite(p2) or p2 <= p1:
        return np.clip(a, 0.0, 1.0)
    return np.clip((a - p1) / (p2 - p1), 0.0, 1.0)

# ⬇️ your SASv2 classes — paste them unchanged (Qt6 compatible already)
class MetricsPanel(QWidget):
    """2×2 grid with clickable dots and draggable thresholds."""
    pointClicked = pyqtSignal(int, int)
    thresholdChanged = pyqtSignal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        # caching slots
        self._orig_images = None       # last list passed
        self.metrics_data = None       # list of 4 numpy arrays
        self.flags = None              # list of bools
        self._threshold_initialized = [False]*4
        self._open_previews = []
        self._show_guides = True  # default on (or False if you prefer)

        self.plots, self.scats, self.lines = [], [], []
        titles = [self.tr("FWHM (px)"), self.tr("Eccentricity"), self.tr("Background"), self.tr("Star Count")]
        for idx, title in enumerate(titles):
            pw = pg.PlotWidget()
            pw.setTitle(title)
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.getPlotItem().getViewBox().setBackgroundColor(
                self.palette().color(self.backgroundRole())
            )

            scat = pg.ScatterPlotItem(pen=pg.mkPen(None),
                                      brush=pg.mkBrush(100,100,255,200),
                                      size=8)
            scat.sigClicked.connect(lambda plot, pts, m=idx: self._on_point_click(m, pts))
            pw.addItem(scat)

            line = pg.InfiniteLine(pos=0, angle=0, movable=True,
                                   pen=pg.mkPen('r', width=2))
            line.sigPositionChangeFinished.connect(
                lambda ln, m=idx: self._on_line_move(m, ln))
            pw.addItem(line)

            # --- dashed reference lines: median + ±3σ (robust) ---
            median_ln = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                        pen=pg.mkPen((220, 220, 220, 170), width=1, style=Qt.PenStyle.DashLine))
            sigma_lo  = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                        pen=pg.mkPen((220, 220, 220, 120), width=1, style=Qt.PenStyle.DashLine))
            sigma_hi  = pg.InfiniteLine(pos=0, angle=0, movable=False,
                                        pen=pg.mkPen((220, 220, 220, 120), width=1, style=Qt.PenStyle.DashLine))

            # keep them behind points/threshold visually
            median_ln.setZValue(-10)
            sigma_lo.setZValue(-10)
            sigma_hi.setZValue(-10)

            pw.addItem(median_ln)
            pw.addItem(sigma_lo)
            pw.addItem(sigma_hi)

            # create the lists once
            if not hasattr(self, "median_lines"):
                self.median_lines = []
                self.sigma_lines = []   # list of (lo, hi)

            self.median_lines.append(median_ln)
            self.sigma_lines.append((sigma_lo, sigma_hi))
            grid.addWidget(pw, idx//2, idx%2)
            self.plots.append(pw)
            self.scats.append(scat)
            self.lines.append(line)

    def set_guides_visible(self, on: bool):
        self._show_guides = bool(on)

        if not self._show_guides:
            # ✅ hide immediately
            if hasattr(self, "median_lines"):
                for ln in self.median_lines:
                    ln.hide()
            if hasattr(self, "sigma_lines"):
                for lo, hi in self.sigma_lines:
                    lo.hide()
                    hi.hide()
            return

        # ✅ turning ON: recompute/restore based on what’s currently plotted
        self._refresh_guides_from_current_plot()

    def _refresh_guides_from_current_plot(self):
        """Recompute/position guide lines using current plot data (if any)."""
        if not getattr(self, "_show_guides", True):
            return
        if not hasattr(self, "median_lines") or not hasattr(self, "sigma_lines"):
            return
        # Use the scatter data already in each panel
        for m, scat in enumerate(self.scats):
            x, y = scat.getData()[:2]
            if y is None or len(y) == 0:
                self.median_lines[m].hide()
                lo, hi = self.sigma_lines[m]
                lo.hide(); hi.hide()
                continue

            med, sig = self._median_and_robust_sigma(np.asarray(y, dtype=np.float32))
            mline = self.median_lines[m]
            lo_ln, hi_ln = self.sigma_lines[m]

            if np.isfinite(med):
                mline.setPos(med); mline.show()
            else:
                mline.hide()

            if np.isfinite(med) and np.isfinite(sig) and sig > 0:
                lo = med - 3.0 * sig
                hi = med + 3.0 * sig
                if m == 3:
                    lo = max(0.0, lo)
                lo_ln.setPos(lo); hi_ln.setPos(hi)
                lo_ln.show(); hi_ln.show()
            else:
                lo_ln.hide(); hi_ln.hide()


    @staticmethod
    def _median_and_robust_sigma(y: np.ndarray):
        """Return (median, sigma) using MAD-based robust sigma. Ignores NaN/Inf."""
        y = np.asarray(y, dtype=np.float32)
        finite = np.isfinite(y)
        if not finite.any():
            return np.nan, np.nan
        v = y[finite]
        med = float(np.nanmedian(v))
        mad = float(np.nanmedian(np.abs(v - med)))
        sigma = 1.4826 * mad  # robust sigma estimate
        return med, float(sigma)


    @staticmethod
    def _compute_one(i_entry):
        """
        Compute (FWHM, eccentricity, star_count) using SEP on a *2x downsampled*
        mono float32 frame.

        - Downsample is fixed at 2x (linear), using AREA.
        - FWHM is converted back to full-res pixel units by multiplying by 2.
        Optionally multiply by sqrt(2) if you want to compensate for the
        AREA downsample's effective smoothing (see fwhm_factor below).
        - Eccentricity is scale-invariant.
        - Star count should be closer to full-res if we also scale minarea
        from 16 -> 4 (area scales by 1/4).
        """

        import cv2
        import sep

        idx, entry = i_entry
        img = entry["image_data"]

        data = np.asarray(img)
        h0, w0 = data.shape[:2]

        # ----------------------------
        # 1) Normalize to float32 mono [0..1]
        # ----------------------------
        if data.ndim == 3:
            data = data.mean(axis=2)

        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0
        else:
            data = data.astype(np.float32, copy=False)

        # Guard: SEP expects finite values
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)

        # ----------------------------
        # 2) Fixed 2x downsample (linear /2)
        # ----------------------------
        # Use integer decimation by resize to preserve speed and consistency.
        new_w = max(16, w0 // 2)
        new_h = max(16, h0 // 2)
        ds = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ----------------------------
        # 3) SEP pipeline (same as before, but minarea scaled)
        # ----------------------------
        try:
            bkg = sep.Background(ds)
            back = bkg.back()
            try:
                gr = float(bkg.globalrms)
            except Exception:
                gr = float(np.median(np.asarray(bkg.rms(), dtype=np.float32)))

            # minarea: 16 at full-res ~= 4 at 2x downsample (area /4)
            minarea = 4

            cat = sep.extract(
                ds - back,
                thresh=7.0,
                err=gr,
                minarea=minarea,
                clean=True,
                deblend_nthresh=32,
            )

            if len(cat) > 0:
                # FWHM via geometric-mean sigma (old Blink)
                sig = np.sqrt(cat["a"] * cat["b"]).astype(np.float32, copy=False)
                fwhm_ds = float(np.nanmedian(2.3548 * sig))

                # ----------------------------
                # 4) Convert FWHM back to full-res
                # ----------------------------
                # Pure geometric reconversion: *2
                # If you want the "noise reduction" compensation you mentioned:
                #   multiply by sqrt(2) instead of 2, or 2*sqrt(2) depending on intent.
                #
                # Most consistent with "true full-res pixels" is factor = 2.
                # If you insist on smoothing-compensation, set factor = 2*np.sqrt(2)
                # (because you still have to undo scale, and then add smoothing term).
                fwhm_factor = 2.0  # change to (2.0 * np.sqrt(2.0)) if you really want it
                fwhm = fwhm_ds * fwhm_factor

                # TRUE eccentricity
                a = np.maximum(cat["a"].astype(np.float32, copy=False), 1e-12)
                b = np.clip(cat["b"].astype(np.float32, copy=False), 0.0, None)
                q = np.clip(b / a, 0.0, 1.0)
                e_true = np.sqrt(np.maximum(0.0, 1.0 - q * q))
                ecc = float(np.nanmedian(e_true))

                star_cnt = int(len(cat))
            else:
                fwhm, ecc, star_cnt = np.nan, np.nan, 0

        except Exception:
            fwhm, ecc, star_cnt = 10.0, 1.0, 0

        orig_back = entry.get("orig_background", np.nan)
        return idx, fwhm, ecc, orig_back, star_cnt



    def compute_all_metrics(self, loaded_images) -> bool:
        """
        Run SEP over the full list in parallel using threads and cache results.
        Uses *downsampled* SEP for speed + lower RAM.
        Returns True if metrics were computed, False if user canceled.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        import numpy as np
        import psutil
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QProgressDialog, QApplication

        n = len(loaded_images)
        if n == 0:
            self._orig_images = []
            self.metrics_data = [np.array([])] * 4
            self.flags = []
            self._threshold_initialized = [False] * 4
            return True

        # ----------------------------
        # 1) Allocate result arrays
        # ----------------------------
        m0 = np.full(n, np.nan, dtype=np.float32)  # FWHM (full-res px units)
        m1 = np.full(n, np.nan, dtype=np.float32)  # Eccentricity
        m2 = np.full(n, np.nan, dtype=np.float32)  # Background (cached)
        m3 = np.full(n, np.nan, dtype=np.float32)  # Star count
        flags = [e.get("flagged", False) for e in loaded_images]

        # ----------------------------
        # 2) Progress dialog (Cancel)
        # ----------------------------
        prog = QProgressDialog(self.tr("Computing frame metrics…"), self.tr("Cancel"), 0, n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        prog.show()
        QApplication.processEvents()

        cpu = os.cpu_count() or 1

        # ----------------------------
        # 3) Worker sizing by RAM (downsample-aware)
        # ----------------------------
        # Estimate using the same max_dim as _compute_one (default 1024).
        # Use first frame to estimate scale.
        max_dim = int(loaded_images[0].get("metrics_max_dim", 1024))
        h0, w0 = loaded_images[0]["image_data"].shape[:2]
        scale = 1.0
        if max(h0, w0) > max_dim:
            scale = max_dim / float(max(h0, w0))

        hd = max(16, int(round(h0 * scale)))
        wd = max(16, int(round(w0 * scale)))

        # float32 mono downsample buffer
        bytes_per = hd * wd * 4

        # SEP allocates extra maps; budget ~3x to be safe.
        budget_per_worker = int(bytes_per * 3.0)

        avail = psutil.virtual_memory().available
        max_by_mem = max(1, int(avail / max(budget_per_worker, 1)))

        # Don’t exceed CPU, and don’t go crazy high even if RAM is huge
        workers = max(1, min(cpu, max_by_mem, 24))

        tasks = [(i, loaded_images[i]) for i in range(n)]
        done = 0
        canceled = False

        try:
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = {exe.submit(self._compute_one, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    if prog.wasCanceled():
                        canceled = True
                        break

                    try:
                        idx, fwhm, ecc, orig_back, star_cnt = fut.result()
                    except Exception:
                        idx = futures.get(fut, 0)
                        fwhm, ecc, orig_back, star_cnt = np.nan, np.nan, np.nan, 0

                    if 0 <= idx < n:
                        m0[idx] = fwhm
                        m1[idx] = ecc
                        m2[idx] = orig_back
                        m3[idx] = float(star_cnt)

                    done += 1
                    prog.setValue(done)
                    QApplication.processEvents()
        finally:
            prog.close()

        if canceled:
            # IMPORTANT: leave caches alone; caller handles clear/return
            return False

        # ----------------------------
        # 4) Stash results
        # ----------------------------
        self._orig_images = loaded_images
        self.metrics_data = [m0, m1, m2, m3]
        self.flags = flags
        self._threshold_initialized = [False] * 4
        return True

    def plot(self, loaded_images, indices=None):
        """
        Plot metrics for loaded_images.
        If indices is given (list/array of ints), only those frames are shown.
        """
        # empty clear
        if not loaded_images:
            self.metrics_data = None
            for pw, scat, line in zip(self.plots, self.scats, self.lines):
                scat.setData(x=[], y=[])
                line.setPos(0)
                pw.getPlotItem().getViewBox().update()
                pw.repaint()

            # ✅ hide guides too
            if hasattr(self, "median_lines"):
                for ln in self.median_lines:
                    ln.hide()
            if hasattr(self, "sigma_lines"):
                for lo, hi in self.sigma_lines:
                    lo.hide()
                    hi.hide()
            return

        # compute & cache on first call or new image list
        if self._orig_images is not loaded_images or self.metrics_data is None:
            ok = self.compute_all_metrics(loaded_images)
            if not ok or self.metrics_data is None:
                # user declined/canceled -> clear plots and exit cleanly
                for pw, scat, line in zip(self.plots, self.scats, self.lines):
                    scat.setData(x=[], y=[])
                    line.setPos(0)
                    pw.getPlotItem().getViewBox().update()
                    pw.repaint()
                return


        # default to all indices
        if indices is None:
            indices = np.arange(len(loaded_images), dtype=int)

        # store for later recoloring
        self._cur_indices = np.array(indices, dtype=int)

        x = np.arange(len(indices))

        for m, (pw, scat, line) in enumerate(zip(self.plots, self.scats, self.lines)):
            arr = self.metrics_data[m]
            y   = arr[indices]

            brushes = [
                pg.mkBrush(255,0,0,200) if self.flags[idx] else pg.mkBrush(100,100,255,200)
                for idx in indices
            ]
            scat.setData(x=x, y=y, brush=brushes, pen=pg.mkPen(None), size=8)

            # --- update dashed reference lines (median + ±3σ) ---
            if getattr(self, "_show_guides", True):
                try:
                    med, sig = self._median_and_robust_sigma(y)
                    mline = self.median_lines[m]
                    lo_ln, hi_ln = self.sigma_lines[m]

                    if np.isfinite(med):
                        mline.setPos(med)
                        mline.show()
                    else:
                        mline.hide()

                    if np.isfinite(med) and np.isfinite(sig) and sig > 0:
                        lo = med - 3.0 * sig
                        hi = med + 3.0 * sig
                        if m == 3:
                            lo = max(0.0, lo)
                        lo_ln.setPos(lo); hi_ln.setPos(hi)
                        lo_ln.show(); hi_ln.show()
                    else:
                        lo_ln.hide(); hi_ln.hide()
                except Exception:
                    if hasattr(self, "median_lines") and m < len(self.median_lines):
                        self.median_lines[m].hide()
                        a, b = self.sigma_lines[m]
                        a.hide(); b.hide()
            else:
                # guides disabled -> force-hide
                if hasattr(self, "median_lines") and m < len(self.median_lines):
                    self.median_lines[m].hide()
                    a, b = self.sigma_lines[m]
                    a.hide(); b.hide()


            # initialize threshold line once
            if not self._threshold_initialized[m]:
                mx, mn = np.nanmax(y), np.nanmin(y)
                span   = mx-mn if mx!=mn else 1.0
                line.setPos((mx+0.05*span) if m<3 else 0)
                self._threshold_initialized[m] = True

    def _refresh_scatter_colors(self):
        if not hasattr(self, "_cur_indices") or self._cur_indices is None:
            # default to all indices
            self._cur_indices = np.arange(len(self.flags or []), dtype=int)

        for scat in self.scats:
            x, y = scat.getData()[:2]
            brushes = []
            for xi in x:
                li = int(xi)
                gi = self._cur_indices[li] if 0 <= li < len(self._cur_indices) else 0
                brushes.append(pg.mkBrush(255,0,0,200) if (self.flags and gi < len(self.flags) and self.flags[gi])
                            else pg.mkBrush(100,100,255,200))
            scat.setData(x=x, y=y, brush=brushes)

    def remove_frames(self, removed_idx: List[int]):
        """
        Drop frames from cached arrays and flags (no recomputation).
        removed_idx: global indices in the *old* ordering.
        """
        if self.metrics_data is None or not removed_idx:
            return
        import numpy as _np
        removed = _np.unique(_np.asarray(removed_idx, dtype=int))
        n = len(self.flags or [])
        if n == 0:
            return
        keep = _np.ones(n, dtype=bool)
        keep[removed[removed < n]] = False

        # shrink cached arrays and flags
        self.metrics_data = [arr[keep] for arr in self.metrics_data]
        if self.flags is not None:
            self.flags = list(_np.asarray(self.flags)[keep])

    def refresh_colors_and_status(self):
        """Recolor dots based on self.flags; caller should also update the window status."""
        self._refresh_scatter_colors()

    def _on_point_click(self, metric_idx, points):
        for pt in points:
            # local index on the currently plotted subset
            li = int(round(pt.pos().x()))

            # map to global index
            if hasattr(self, "_cur_indices") and self._cur_indices is not None and 0 <= li < len(self._cur_indices):
                gi = int(self._cur_indices[li])
            else:
                gi = li  # fallback (e.g., "All")

            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                # preview the correct global frame
                entry  = self._orig_images[gi]
                img    = entry['image_data']
                is_mono= entry.get('is_mono', False)
                dlg = ImagePreviewDialog(img, is_mono)
                dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                dlg.show()
                self._open_previews.append(dlg)
                dlg.destroyed.connect(lambda _=None, d=dlg:
                    self._open_previews.remove(d) if d in self._open_previews else None)
            else:
                # emit the correct global frame index so Blink flags the right leaf
                self.pointClicked.emit(metric_idx, gi)

    def _on_line_move(self, metric_idx, line):
        self.thresholdChanged.emit(metric_idx, line.value())

class MetricsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self._thresholds_per_group: dict[str, List[float|None]] = {}
        self.setWindowTitle(self.tr("Frame Metrics"))
        self.resize(800, 600)

        vbox = QVBoxLayout(self)

        # ← **new** instructions label
        instr = QLabel(self.tr(
            "Instructions:\n"
            " • Use the filter dropdown to restrict by FILTER.\n"
            " • Click a dot to flag/unflag a frame.\n"
            " • Shift-click a dot to preview the image.\n"
            " • Drag the red lines to set thresholds."
        ),
            self
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #ccc; font-size: 12px;")
        vbox.addWidget(instr)
        self.chk_guides = QCheckBox(self.tr("Show median and ±3σ guides"), self)
        self.chk_guides.setChecked(True)  # default on
        self.chk_guides.toggled.connect(self._on_toggle_guides)
        vbox.addWidget(self.chk_guides)
        # → filter selector
        self.group_combo = QComboBox(self)
        self.group_combo.addItem(self.tr("All"))
        self.group_combo.currentTextChanged.connect(self._on_group_change)
        vbox.addWidget(self.group_combo)

        # → the 2×2 metrics panel
        self.metrics_panel = MetricsPanel(self)
        vbox.addWidget(self.metrics_panel)

        # keep status up‐to‐date when things happen
        self.metrics_panel.thresholdChanged.connect(self._update_status)
        self.metrics_panel.pointClicked   .connect(self._update_status)

        # ← status label
        self.status_label = QLabel("", self)
        vbox.addWidget(self.status_label)

        # internal storage
        self._all_images = []
        self._current_indices: Optional[List[int]] = None

    def _on_toggle_guides(self, on: bool):
        if hasattr(self, "metrics_panel") and self.metrics_panel is not None:
            self.metrics_panel.set_guides_visible(on)


    def _update_status(self, *args):
        """Recompute and show: Flagged Items X / Y (Z%).  Robust to stale indices."""
        flags = getattr(self.metrics_panel, "flags", []) or []
        nflags = len(flags)

        # what subset are we currently looking at?
        idxs = self._current_indices if self._current_indices is not None else range(nflags)

        total = 0
        flagged_cnt = 0

        for i in idxs:
            # i can be np.int64 or a stale index from before a move/delete
            j = int(i)
            if 0 <= j < nflags:
                total += 1
                if flags[j]:
                    flagged_cnt += 1
            else:
                # stale index → just skip it
                continue

        pct = (flagged_cnt / total * 100.0) if total else 0.0
        self.status_label.setText(self.tr("Flagged Items {0}/{1}  ({2:.1f}%)").format(flagged_cnt, total, pct))


    def set_images(self, loaded_images, order=None):
        self._all_images = loaded_images
        self._order_all = list(order) if order is not None else list(range(len(loaded_images)))

        # ─── rebuild the combo-list of FILTER groups ─────────────
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem(self.tr("All"))
        seen = set()
        for entry in loaded_images:
            filt = entry.get('header', {}).get('FILTER', 'Unknown')
            if filt not in seen:
                seen.add(filt)
                self.group_combo.addItem(filt)
        self.group_combo.blockSignals(False)

        # ─── reset & seed per-group thresholds ────────────────────
        self._thresholds_per_group.clear()
        self._thresholds_per_group["All"] = [None]*4
        for entry in loaded_images:
            filt = entry.get('header', {}).get('FILTER', 'Unknown')
            if filt not in self._thresholds_per_group:
                self._thresholds_per_group[filt] = [None]*4

        # ─── compute & cache all metrics once ────────────────────
        self.metrics_panel.compute_all_metrics(self._all_images)

        # ─── show “All” by default and plot ───────────────────────
        self._current_indices = self._order_all
        self._apply_thresholds("All")
        self.metrics_panel.plot(self._all_images, indices=self._current_indices)
        self.metrics_panel.set_guides_visible(self.chk_guides.isChecked())
        self._update_status()

    def _reindex_list_after_remove(self, lst: List[int] | None, removed: List[int]) -> List[int] | None:
        """Return lst with removed indices dropped and others shifted."""
        if lst is None:
            return None
        from bisect import bisect_right
        removed = sorted(set(int(i) for i in removed))
        rset = set(removed)
        def new_idx(old):
            return old - bisect_right(removed, old)
        return [new_idx(i) for i in lst if i not in rset]

    def _rebuild_groups_from_images(self):
        """Rebuild the FILTER combobox from current _all_images, keep current if possible."""
        cur = self.group_combo.currentText()
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem(self.tr("All"))
        seen = set()
        for entry in self._all_images:
            filt = (entry.get('header', {}) or {}).get('FILTER', 'Unknown')
            if filt not in seen:
                self.group_combo.addItem(filt)
                seen.add(filt)
        self.group_combo.blockSignals(False)
        # restore selection if still valid
        idx = self.group_combo.findText(cur)
        if idx >= 0:
            self.group_combo.setCurrentIndex(idx)
        else:
            self.group_combo.setCurrentIndex(0)

    def remove_indices(self, removed: List[int]):
        """
        Called when some frames were deleted/moved out of the list.
        Does NOT recompute metrics. Just trims cached arrays and re-plots.

        Robust against:
        - removed indices referring to the old list (out of range)
        - metrics_panel arrays being a different length than _all_images
        - stale _order_all / _current_indices containing out-of-bounds indices
        """
        if not removed:
            return

        # Unique + int
        removed = sorted({int(i) for i in removed})

        # ---- 1) Trim metrics panel caches SAFELY ----
        # Prefer panel's current frame count, because it represents the arrays we must slice.
        n_panel = getattr(self.metrics_panel, "n_frames", None)
        if callable(n_panel):
            n_panel = n_panel()
        if not isinstance(n_panel, int) or n_panel <= 0:
            # fallback: infer from metrics_data if present
            md = getattr(self.metrics_panel, "metrics_data", None)
            if md is not None and len(md) and md[0] is not None:
                try:
                    n_panel = int(len(md[0]))
                except Exception:
                    n_panel = 0
            else:
                n_panel = 0

        if n_panel > 0:
            removed_panel = [i for i in removed if 0 <= i < n_panel]
            if removed_panel:
                self.metrics_panel.remove_frames(removed_panel)
        # else: panel has nothing (or isn't initialized) — just continue with ordering cleanup

        # ---- 2) Update ordering arrays with the SAME removed set (but clamp later) ----
        self._order_all = self._reindex_list_after_remove(self._order_all, removed)
        if self._current_indices is not None:
            self._current_indices = self._reindex_list_after_remove(self._current_indices, removed)

        # ---- 3) Rebuild groups (filters may have disappeared) ----
        self._rebuild_groups_from_images()

        # ---- 4) Plot with VALID indices only ----
        n_imgs = len(self._all_images) if self._all_images is not None else 0

        def _sanitize_indices(ixs):
            if not ixs:
                return []
            out = []
            seen = set()
            for i in ixs:
                try:
                    ii = int(i)
                except Exception:
                    continue
                if 0 <= ii < n_imgs and ii not in seen:
                    seen.add(ii)
                    out.append(ii)
            return out

        indices = self._current_indices if self._current_indices is not None else self._order_all
        indices = _sanitize_indices(indices)

        # If the current group became empty, fall back to "all"
        if not indices and n_imgs:
            indices = list(range(n_imgs))
            self._current_indices = indices  # optional: keeps UI consistent

        self.metrics_panel.plot(self._all_images, indices=indices)

        # ---- 5) Recolor & status ----
        self.metrics_panel.refresh_colors_and_status()
        self._update_status()

    def _on_group_change(self, name: str):
        if name == self.tr("All"):
            self._current_indices = self._order_all
        else:
            # preserve Tree order inside the chosen FILTER
            filt = name
            self._current_indices = [
                i for i in self._order_all
                if (self._all_images[i].get('header', {}) or {}).get('FILTER', 'Unknown') == filt
            ]
        self._apply_thresholds(name)
        self.metrics_panel.plot(self._all_images, indices=self._current_indices)

    def _on_panel_threshold_change(self, metric_idx: int, new_val: float):
        """User just dragged a threshold line."""
        grp = self.group_combo.currentText()
        # save it for this group
        self._thresholds_per_group[grp][metric_idx] = new_val
        self.metrics_panel.plot(self._all_images, indices=self._current_indices)
        self.metrics_panel.set_guides_visible(self.chk_guides.isChecked())
        # (if you also want immediate re-flagging in the tree, keep your BlinkTab logic hooked here)

    def _apply_thresholds(self, group_name: str):
        """Restore the four InfiniteLine positions for a given group."""
        saved = self._thresholds_per_group.get(group_name, [None]*4)
        for idx, line in enumerate(self.metrics_panel.lines):
            if saved[idx] is not None:
                line.setPos(saved[idx])
            # if saved[idx] is None, we leave it so that
            # the panel’s own auto-init can run on next plot()

    def update_metrics(self, loaded_images, order=None):
        if loaded_images is not self._all_images:
            self.set_images(loaded_images, order=order)
        else:
            if order is not None:
                self._order_all = list(order)
    # re-plot the current group with the new ordering
            self._on_group_change(self.group_combo.currentText())

class BlinkComparatorPro(QDialog):
    sendToStacking = pyqtSignal(list, str)

    def __init__(self, doc_manager=None, parent=None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self.setWindowTitle(self.tr("Blink Comparator"))
        self.resize(1200, 700)

        self.tab = BlinkTab(doc_manager=self.doc_manager, parent=self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tab)
        self.setLayout(layout)

        # bridge tab → dialog
        self.tab.sendToStacking.connect(self.sendToStacking)


class BlinkTab(QWidget):
    imagesChanged = pyqtSignal(int)
    sendToStacking = pyqtSignal(list, str)
    def __init__(self, image_manager=None, doc_manager=None, parent=None):
        super().__init__(parent)  

        self.image_paths = []  # Store the file paths of loaded images
        self.loaded_images = []  # Store the image objects (as numpy arrays)
        self.image_labels = []  # Store corresponding file names for the TreeWidget
        self.doc_manager = doc_manager        # ⬅️ new
        self.image_manager = image_manager            # ⬅️ ensure we don't use it
        self.metrics_window: Optional[MetricsWindow] = None
        self.zoom_level = 0.5  # Default zoom level
        self.dragging = False  # Track whether the mouse is dragging
        self.last_mouse_pos = None  # Store the last mouse position
        self.thresholds_by_group: dict[str, List[float|None]] = {}
        self.aggressive_stretch_enabled = False
        self.current_sigma = 3.7
        self.current_pixmap = None
        self._last_preview_name = None
        self._pending_preview_timer = QTimer(self)
        self._pending_preview_timer.setSingleShot(True)
        self._pending_preview_timer.setInterval(40)  # 40–80ms is plenty
        self._pending_preview_item = None
        self._pending_preview_timer.timeout.connect(self._do_preview_update)
        self.play_fps = 1  # default fps (200 ms/frame)
        self._view_center_norm = None
        self.initUI()
        self.init_shortcuts()

    def initUI(self):
        main_layout = QHBoxLayout(self)


        # Create a QSplitter to allow resizing between left and right panels
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left Column for the file loading and TreeView
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        # --------------------
        # Instruction Label
        # --------------------
        instruction_text = self.tr("Press 'F' to flag/unflag an image.\nRight-click on an image for more options.")
        self.instruction_label = QLabel(instruction_text, self)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-weight: bold;")  # Optional: Make the text bold for emphasis

        self.instruction_label.setStyleSheet(f"""
            QLabel {{
                font-weight: bold;
            }}
        """)

        # Add the instruction label to the left layout at the top
        left_layout.addWidget(self.instruction_label)

        # Horizontal layout for "Select Images" and "Select Directory" buttons
        button_layout = QHBoxLayout()

        # "Select Images" Button
        self.fileButton = QPushButton(self.tr('Select Images'), self)
        self.fileButton.clicked.connect(self.openFileDialog)
        button_layout.addWidget(self.fileButton)

        # "Select Directory" Button
        self.dirButton = QPushButton(self.tr('Select Directory'), self)
        self.dirButton.clicked.connect(self.openDirectoryDialog)
        button_layout.addWidget(self.dirButton)

        self.addButton = QPushButton(self.tr("Add Additional"), self)
        self.addButton.clicked.connect(self.addAdditionalImages)
        button_layout.addWidget(self.addButton)

        left_layout.addLayout(button_layout)

        self.metrics_button = QPushButton(self.tr("Show Metrics"), self)
        self.metrics_button.clicked.connect(self.show_metrics)
        left_layout.addWidget(self.metrics_button)

        push_row = QHBoxLayout()
        self.send_lights_btn = QPushButton(self.tr("→ Stacking: Lights"), self)
        self.send_lights_btn.setToolTip(self.tr("Send selected (or all) blink files to the Stacking Suite → Light tab"))
        self.send_lights_btn.clicked.connect(self._send_to_stacking_lights)
        push_row.addWidget(self.send_lights_btn)

        self.send_integ_btn = QPushButton(self.tr("→ Stacking: Integration"), self)
        self.send_integ_btn.setToolTip(self.tr("Send selected (or all) blink files to the Stacking Suite → Image Integration tab"))
        self.send_integ_btn.clicked.connect(self._send_to_stacking_integration)
        push_row.addWidget(self.send_integ_btn)

        left_layout.addLayout(push_row)

        # Playback controls (left arrow, play, pause, right arrow)
        playback_controls_layout = QHBoxLayout()

        # Left Arrow Button
        self.left_arrow_button = QPushButton(self)
        self.left_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
        self.left_arrow_button.clicked.connect(self.previous_item)
        playback_controls_layout.addWidget(self.left_arrow_button)

        # Play Button
        self.play_button = QPushButton(self)
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.start_playback)
        playback_controls_layout.addWidget(self.play_button)

        # Pause Button
        self.pause_button = QPushButton(self)
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.clicked.connect(self.stop_playback)
        playback_controls_layout.addWidget(self.pause_button)

        # Right Arrow Button
        self.right_arrow_button = QPushButton(self)
        self.right_arrow_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.right_arrow_button.clicked.connect(self.next_item)
        playback_controls_layout.addWidget(self.right_arrow_button)

        left_layout.addLayout(playback_controls_layout)

        # ----- Playback speed controls -----
        # ----- Playback speed controls (0.1–10.0 fps) -----
        speed_layout = QHBoxLayout()

        speed_label = QLabel(self.tr("Speed:"), self)
        speed_layout.addWidget(speed_label)

        # Slider maps 1..100 -> 0.1..10.0 fps
        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(int(round(self.play_fps * 10)))  # play_fps is float
        self.speed_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.speed_slider.setToolTip(self.tr("Playback speed (0.1–10.0 fps)"))
        speed_layout.addWidget(self.speed_slider, 1)

        # Custom float spin (your class)
        self.speed_spin = CustomDoubleSpinBox(
            minimum=0.1, maximum=10.0, initial=self.play_fps, step=0.1, parent=self
        )
        speed_layout.addWidget(self.speed_spin)

        # IMPORTANT: remove any old direct connects like:
        # self.speed_slider.valueChanged.connect(self.speed_spin.setValue)
        # self.speed_spin.valueChanged.connect(self.speed_slider.setValue)

        # Use lambdas to cast types correctly
        self.speed_slider.valueChanged.connect(lambda v: self.speed_spin.setValue(v / 10.0))          # int -> float
        self.speed_spin.valueChanged.connect(lambda f: self.speed_slider.setValue(int(round(f * 10))))  # float -> int

        self.speed_slider.valueChanged.connect(self._apply_playback_interval)
        self.speed_spin.valueChanged.connect(self._apply_playback_interval)

        left_layout.addLayout(speed_layout)

        self.export_button = QPushButton(self.tr("Export Video…"), self)
        self.export_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.export_button.clicked.connect(self.export_blink_video)
        left_layout.addWidget(self.export_button)

        # Tree view for file names
        self.fileTree = QTreeWidget(self)
        self.fileTree.setColumnCount(1)
        self.fileTree.setHeaderLabels([self.tr("Image Files")])
        self.fileTree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)  # Allow multiple selections
        #self.fileTree.itemClicked.connect(self.on_item_clicked)
        self.fileTree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.fileTree.customContextMenuRequested.connect(self.on_right_click)
        self.fileTree.currentItemChanged.connect(self._on_current_item_changed_safe)
        self.fileTree.setStyleSheet("""
                QTreeWidget::item:selected {
                    background-color: #3a75c4;  /* Blue background for selected items */
                    color: #ffffff;  /* White text color */
                }
            """)
        left_layout.addWidget(self.fileTree)

        # "Clear Flags" Button
        self.clearFlagsButton = QPushButton(self.tr('Clear Flags'), self)
        self.clearFlagsButton.clicked.connect(self.clearFlags)
        left_layout.addWidget(self.clearFlagsButton)

        # "Clear Images" Button
        self.clearButton = QPushButton(self.tr('Clear Images'), self)
        self.clearButton.clicked.connect(self.clearImages)
        left_layout.addWidget(self.clearButton)

        # Add progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        # Add loading message label
        self.loading_label = QLabel(self.tr("Loading images..."), self)
        left_layout.addWidget(self.loading_label)
        self.imagesChanged.emit(len(self.loaded_images)) 

        # Set the layout for the left widget
        left_widget.setLayout(left_layout)

        # Add the left widget to the splitter
        splitter.addWidget(left_widget)

        # Right Column for Image Preview
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom / preview toolbar (standardized)
        zoom_controls_layout = QHBoxLayout()

        self.zoom_in_btn  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.zoom_out_btn = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.fit_btn      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.fit_btn.clicked.connect(self.fit_to_preview)

        zoom_controls_layout.addWidget(self.zoom_in_btn)
        zoom_controls_layout.addWidget(self.zoom_out_btn)
        zoom_controls_layout.addWidget(self.fit_btn)

        zoom_controls_layout.addStretch(1)

        # Keep Aggressive Stretch as a text toggle (it’s not really a zoom action)
        self.aggressive_button = QPushButton(self.tr("Aggressive Stretch"), self)
        self.aggressive_button.setCheckable(True)
        self.aggressive_button.clicked.connect(self.toggle_aggressive)
        zoom_controls_layout.addWidget(self.aggressive_button)

        right_layout.addLayout(zoom_controls_layout)

        # Scroll area for the preview
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.viewport().installEventFilter(self)

        # QLabel for the image preview
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.preview_label)

        right_layout.addWidget(self.scroll_area)

        # Set the layout for the right widget
        right_widget.setLayout(right_layout)

        # Add the right widget to the splitter
        splitter.addWidget(right_widget)

        # Set initial splitter sizes
        splitter.setSizes([300, 700])  # Adjust proportions as needed

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Set the main layout for the widget
        self.setLayout(main_layout)

        # Initialize playback timer
        self.playback_timer = QTimer(self)
        self._apply_playback_interval()  # sets interval based on self.play_fps
        self.playback_timer.timeout.connect(self.next_item)

        # Connect the selection change signal to update the preview when arrow keys are used
        self.fileTree.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.scroll_area.horizontalScrollBar().valueChanged.connect(lambda _: self._capture_view_center_norm())
        self.scroll_area.verticalScrollBar().valueChanged.connect(lambda _: self._capture_view_center_norm())
        self.imagesChanged.connect(self._update_loaded_count_label)

    @staticmethod
    def _ensure_float01(img):
        """
        Convert to float32 and force into [0..1] using:
        - if min < 0: subtract min
        - if max > 1: divide by max
        Works for mono or RGB. Handles NaN/Inf safely.
        """
        arr = np.asarray(img, dtype=np.float32)

        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros_like(arr, dtype=np.float32)

        mn = float(arr[finite].min())
        if mn < 0.0:
            arr = arr - mn

        # recompute after possible shift
        finite = np.isfinite(arr)
        mx = float(arr[finite].max()) if finite.any() else 0.0
        if mx > 1.0:
            if mx > 0.0:
                arr = arr / mx

        return np.clip(arr, 0.0, 1.0)


    def _aggressive_display_boost(self, x01: np.ndarray, strength: float = 3.7) -> np.ndarray:
        """
        Stronger display stretch on top of an already stretched image.
        Input/Output are float32 in [0..1].
        Robust: percentile normalize + asinh boost.
        """
        x = np.asarray(x01, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        x = np.clip(x, 0.0, 1.0)

        # Robust normalize: ignore extreme outliers so we actually expand contrast
        lo = float(np.percentile(x, 0.25))
        hi = float(np.percentile(x, 99.75))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
            return x  # nothing to do, but never return black

        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)

        # Asinh boost (stronger -> more aggressive midtone lift)
        k = max(1.0, float(strength) * 1.25)  # tune multiplier to taste
        y = np.arcsinh(k * y) / np.arcsinh(k)

        return np.clip(y, 0.0, 1.0)


    # --------------------------------------------
    # NEW: collect paths & emit to stacking
    # --------------------------------------------
    def _collect_paths_for_stacking(self) -> list[str]:
        """
        Priority:
        1) if user has rows selected in the tree → use those
        2) else → use all loaded image_paths
        """
        paths: list[str] = []

        selected_items = self.fileTree.selectedItems()
        if selected_items:
            for it in selected_items:
                p = it.data(0, Qt.ItemDataRole.UserRole)
                if not p:
                    # some code uses text as path, fall back
                    p = it.text(0)
                if p:
                    paths.append(p)
        else:
            # no selection → send all
            for p in self.image_paths:
                if p:
                    paths.append(p)

        # de-dup, keep order
        seen = set()
        unique_paths = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        return unique_paths

    def _send_to_stacking_lights(self):
        paths = self._collect_paths_for_stacking()
        if not paths:
            QMessageBox.information(self, self.tr("No images"), self.tr("There are no images to send."))
            return
        self.sendToStacking.emit(paths, "lights")

    def _send_to_stacking_integration(self):
        paths = self._collect_paths_for_stacking()
        if not paths:
            QMessageBox.information(self, self.tr("No images"), self.tr("There are no images to send."))
            return
        self.sendToStacking.emit(paths, "integration")


    def export_blink_video(self):
        """Export the blink sequence to a video. Defaults to all frames in current tree order."""
        # Ensure we have frames
        leaves = self.get_all_leaf_items()
        if not leaves:
            QMessageBox.information(self, self.tr("No Images"), self.tr("Load images before exporting."))
            return

        # Ask options first (size, fps, selection scope)
        opts = self._ask_video_options(default_fps=float(self.play_fps))
        if opts is None:
            return
        target_w, target_h = opts["size"]
        fps = max(0.1, min(60.0, float(opts["fps"])))
        only_selected = bool(opts.get("only_selected", False))

        # Decide frame order
        if only_selected:
            sel_leaves = [it for it in self.fileTree.selectedItems() if it.childCount() == 0]
            if not sel_leaves:
                QMessageBox.information(self, self.tr("No Selection"), self.tr("No individual frames selected."))
                return
            names = {it.text(0).lstrip("⚠️ ").strip() for it in sel_leaves}
            order = [i for i in self._tree_order_indices()
                    if os.path.basename(self.image_paths[i]) in names]
        else:
            order = self._tree_order_indices()

        if not order:
            QMessageBox.information(self, self.tr("No Frames"), self.tr("Nothing to export."))
            return

        if len(order) < 2:
            ret = QMessageBox.question(
                self, self.tr("Only one frame"),
                self.tr("You're about to export a video with a single frame. Continue?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        # Ask where to save
        out_path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export Blink Video"), "blink.mp4", self.tr("Video (*.mp4 *.avi)")
        )
        if not out_path:
            return
        # Let _open_video_writer_portable decide the real extension; we pass requested
        writer, out_path, backend = self._open_video_writer_portable(out_path, (target_w, target_h), fps)
        if writer is None:
            QMessageBox.critical(self, self.tr("Export"),
                self.tr("No compatible video codec found.\n\n"
                "Tip: install FFmpeg or `pip install imageio[ffmpeg]` for a portable fallback.")
            )
            return

        # Progress UI
        prog = QProgressDialog(self.tr("Rendering video…"), self.tr("Cancel"), 0, len(order), self)
        prog.setWindowTitle(self.tr("Export Blink Video"))
        prog.setAutoClose(True)
        prog.setMinimumDuration(300)

        using_imageio = (backend == "imageio-ffmpeg")
        frames_written = 0

        try:
            for i, idx in enumerate(order):
                if prog.wasCanceled():
                    break

                entry = self.loaded_images[idx]
                f = self._make_display_frame(entry)  # uint8, gray or RGB

                # Ensure 3-channel RGB
                if f.ndim == 2:
                    f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)

                # Letterbox into target (keep aspect)
                tw, th = (target_w, target_h)
                h, w = f.shape[:2]
                s = min(tw / float(w), th / float(h))
                nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
                resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                rgb_canvas = np.zeros((th, tw, 3), dtype=np.uint8)
                x0, y0 = (tw - nw) // 2, (th - nh) // 2
                rgb_canvas[y0:y0+nh, x0:x0+nw] = resized

                if using_imageio:
                    writer.append_data(rgb_canvas)  # RGB
                else:
                    writer.write(cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR))  # BGR
                frames_written += 1

                prog.setValue(i + 1)
                QApplication.processEvents()
        finally:
            try:
                writer.close() if using_imageio else writer.release()
            except Exception:
                pass

        if prog.wasCanceled():
            try:
                os.remove(out_path)
            except Exception:
                pass
            QMessageBox.information(self, self.tr("Export"), self.tr("Export canceled."))
            return

        if frames_written == 0:
            QMessageBox.critical(self, self.tr("Export"), self.tr("No frames were written (codec/back-end issue?)."))
            return

        QMessageBox.information(self, self.tr("Export"), self.tr("Saved: {0}\nFrames: {1} @ {2} fps").format(out_path, frames_written, fps))



    def _ask_video_options(self, default_fps: float):
        """Options dialog for size, fps, and whether to limit to current selection."""
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Video Options"))
        layout = QGridLayout(dlg)

        # Size
        layout.addWidget(QLabel(self.tr("Size:")), 0, 0)
        size_combo = QComboBox(dlg)
        size_combo.addItem("HD 1280×720", (1280, 720))
        size_combo.addItem("Full HD 1920×1080", (1920, 1080))
        size_combo.addItem("Square 1080×1080", (1080, 1080))
        size_combo.setCurrentIndex(0)
        layout.addWidget(size_combo, 0, 1)

        # FPS
        layout.addWidget(QLabel(self.tr("FPS:")), 1, 0)
        fps_edit = QDoubleSpinBox(dlg)
        fps_edit.setRange(0.1, 60.0)
        fps_edit.setDecimals(2)
        fps_edit.setSingleStep(0.1)
        fps_edit.setValue(float(default_fps))
        layout.addWidget(fps_edit, 1, 1)

        # Only selected?
        only_selected = QCheckBox(self.tr("Export only selected frames"), dlg)
        only_selected.setChecked(False)  # default: export everything in tree order
        layout.addWidget(only_selected, 2, 0, 1, 2)

        # Buttons
        btns = QHBoxLayout()
        ok = QPushButton(self.tr("OK"), dlg); cancel = QPushButton(self.tr("Cancel"), dlg)
        ok.clicked.connect(dlg.accept); cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns, 3, 0, 1, 2)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return {
            "size": size_combo.currentData(),
            "fps": fps_edit.value(),
            "only_selected": only_selected.isChecked()
        }



    def _make_display_frame(self, entry):
        stored = entry['image_data']
        use_aggr = bool(self.aggressive_stretch_enabled)

        if not use_aggr:
            if stored.dtype == np.uint8:
                return stored
            elif stored.dtype == np.uint16:
                return (stored >> 8).astype(np.uint8)
            else:
                # ✅ display-only normalization for float / weird ranges
                f01 = self._ensure_float01(stored)
                return (f01 * 255.0).astype(np.uint8)

        base01 = self._as_float01(stored)

        if base01.ndim == 2:
            disp01 = self._aggressive_display_boost(base01, strength=self.current_sigma)
        else:
            lum = base01.mean(axis=2).astype(np.float32)
            lum_boost = self._aggressive_display_boost(lum, strength=self.current_sigma)
            gain = lum_boost / (lum + 1e-6)
            disp01 = np.clip(base01 * gain[..., None], 0.0, 1.0)

        return (disp01 * 255.0).astype(np.uint8)



    def _fit_letterbox(self, frame_bgr_or_rgb, target_size):
        """
        Fit 'frame' into target_size with letterboxing (black borders).
        Accepts uint8, shape (H,W,3). Returns BGR uint8 (H_t,W_t,3).
        """
        tw, th = target_size
        h, w = frame_bgr_or_rgb.shape[:2]
        # Compute scale to fit inside
        s = min(tw / float(w), th / float(h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))

        # Resize (OpenCV uses BGR—this function doesn’t swap channels)
        resized = cv2.resize(frame_bgr_or_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        # Pad into target
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        x0 = (tw - nw) // 2
        y0 = (th - nh) // 2
        out[y0:y0+nh, x0:x0+nw] = resized if resized.ndim == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return out

    def _open_video_writer_portable(self, requested_path: str, size: tuple[int, int], fps: float):
        """
        Try several (container, fourcc) combos that work across platforms.
        Returns (writer, out_path, backend_name). If OpenCV fails, tries imageio-ffmpeg.
        Never writes a probe frame, so no accidental extra first frame.
        """
        tw, th = size
        candidates = [
            (".mp4", "mp4v", "OpenCV-mp4v"),
            (".mp4", "avc1", "OpenCV-avc1"),   # H.264 if available
            (".mp4", "H264", "OpenCV-H264"),
            (".avi", "MJPG", "OpenCV-MJPG"),
            (".avi", "XVID", "OpenCV-XVID"),
        ]
        base, _ = os.path.splitext(requested_path)

        # Try OpenCV containers/codecs first (without writing a test frame)
        for ext, fourcc_tag, label in candidates:
            out_path = base + ext
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)

            # open/close once to check the container initialization
            vw = cv2.VideoWriter(out_path, fourcc, float(fps), (tw, th))
            ok = vw.isOpened()
            try:
                vw.release()
            except Exception:
                pass

            # some backends leave a tiny stub — clean it up before the real open
            try:
                if os.path.exists(out_path) and os.path.getsize(out_path) < 1024:
                    os.remove(out_path)
            except Exception:
                pass

            if ok:
                vw2 = cv2.VideoWriter(out_path, fourcc, float(fps), (tw, th))
                if vw2.isOpened():
                    return vw2, out_path, label

        # Fallback: imageio-ffmpeg (portable, needs imageio[ffmpeg])
        try:
            import imageio
            writer = imageio.get_writer(base + ".mp4", fps=float(fps), macro_block_size=None)  # expects RGB frames
            return writer, base + ".mp4", "imageio-ffmpeg"
        except Exception:
            return None, None, None




    def _update_loaded_count_label(self, n: int):
        # pluralize nicely
        self.loading_label.setText(self.tr("Loaded {0} image{1}.").format(n, 's' if n != 1 else ''))

    def _apply_playback_interval(self, *_):
        # read from custom spin if present (support both .value() and .value attribute)
        fps = float(getattr(self, "play_fps", 1.0))

        if hasattr(self, "speed_spin") and self.speed_spin is not None:
            try:
                v = getattr(self.speed_spin, "value", None)
                if callable(v):
                    fps = float(v())          # QDoubleSpinBox-style
                elif v is not None:
                    fps = float(v)            # CustomDoubleSpinBox stores numeric attribute
                else:
                    # last-resort: try Qt API name
                    fps = float(self.speed_spin.value())
            except Exception:
                # fall back to existing play_fps
                pass

        fps = max(0.1, min(10.0, fps))
        self.play_fps = fps

        if hasattr(self, "playback_timer") and self.playback_timer is not None:
            self.playback_timer.setInterval(int(round(1000.0 / fps)))  # 0.1 fps -> 10000 ms


    def _on_current_item_changed_safe(self, current, previous):
        if not current:
            return

        # If mouse is down, defer a bit, but DO NOT capture the item
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            QTimer.singleShot(120, self._center_if_no_mouse)
            return

        # Defer to allow selection to settle, then ensure the *current* item is visible
        QTimer.singleShot(0, self._ensure_current_visible)

    def _ensure_current_visible(self):
        item = self.fileTree.currentItem()
        if item is not None:
            self.fileTree.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)

    def _center_if_no_mouse(self):
        if QApplication.mouseButtons() == Qt.MouseButton.NoButton:
            item = self.fileTree.currentItem()
            if item is not None:
                self.fileTree.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)

    def _leaf_path(self, item: QTreeWidgetItem) -> str | None:
        """Return full path for a leaf item, preferring UserRole; fallback to basename match."""
        if not item or item.childCount() > 0:
            return None

        p = item.data(0, Qt.ItemDataRole.UserRole)
        if p and isinstance(p, str):
            return p

        # fallback: basename match (legacy items)
        name = item.text(0).lstrip("⚠️ ").strip()
        if not name:
            return None
        return next((x for x in self.image_paths if os.path.basename(x) == name), None)


    def _leaf_index(self, item: QTreeWidgetItem) -> int | None:
        """Return index into image_paths/loaded_images for a leaf item."""
        p = self._leaf_path(item)
        if not p:
            return None
        try:
            return self.image_paths.index(p)
        except ValueError:
            return None


    def _set_leaf_display(self, item: QTreeWidgetItem, *, base_name: str, flagged: bool, full_path: str):
        """Update a leaf item's text + UserRole consistently."""
        disp = base_name
        if flagged:
            disp = f"⚠️ {disp}"
        item.setText(0, disp)
        item.setData(0, Qt.ItemDataRole.UserRole, full_path)


    def clearFlags(self):
        """Clear all flagged states, update tree icons & metrics."""
        # 1) Reset internal flag state
        for entry in self.loaded_images:
            entry['flagged'] = False

        # 2) Update tree widget: strip any "⚠️ " prefix and reset color
        normal = self.fileTree.palette().color(QPalette.ColorRole.WindowText)
        for item in self.get_all_leaf_items():
            name = item.text(0).lstrip("⚠️ ")
            item.setText(0, name)
            item.setForeground(0, QBrush(normal))

        # 3) If metrics window is open, refresh its dots & status
        if self.metrics_window:
            panel = self.metrics_window.metrics_panel
            panel.flags = [False] * len(self.loaded_images)
            panel._refresh_scatter_colors()
            # update the "Flagged Items X/Y" label
            self.metrics_window._update_status()

    # inside BlinkTab
    def _sync_metrics_flags(self):
        if self.metrics_window:
            panel = self.metrics_window.metrics_panel
            panel.flags = [entry['flagged'] for entry in self.loaded_images]
            panel._refresh_scatter_colors()
            # after a move/delete, current_indices might be stale → refresh text safely
            self.metrics_window._update_status()


    def addAdditionalImages(self):
        """Let the user pick more images to append to the blink list."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Add Additional Images"),
            "",
            self.tr("Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;All Files (*)")
        )
        # filter out duplicates
        new_paths = [p for p in file_paths if p not in self.image_paths]
        if not new_paths:
            QMessageBox.information(self, self.tr("No New Images"), self.tr("No new images selected or already loaded."))
            return
        self._appendImages(new_paths)

    def _appendImages(self, file_paths):
        # decide dtype exactly as in loadImages
        mem = psutil.virtual_memory()
        avail = mem.available / (1024**3)
        if avail <= 16:
            target_dtype = np.uint8
        elif avail <= 32:
            target_dtype = np.uint16
        else:
            target_dtype = np.float32

        total_new = len(file_paths)
        self.progress_bar.setRange(0, total_new)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        # load one-by-one (or you could parallelize as you like)
        for i, path in enumerate(sorted(file_paths, key=lambda p: self._natural_key(os.path.basename(p)))):
            try:
                _, hdr, bit_depth, is_mono, stored, back = self._load_one_image(path, target_dtype)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

            # append to our master lists
            self.image_paths.append(path)
            self.loaded_images.append({
                'file_path':      path,
                'image_data':     stored,
                'header':         hdr or {},
                'bit_depth':      bit_depth,
                'is_mono':        is_mono,
                'flagged':        False,
                'orig_background': back
            })

            # update progress bar
            self.progress_bar.setValue(i+1)
            QApplication.processEvents()

            # and add it into the tree under the correct object/filter/exp
            self.add_item_to_tree(path)

        # update status
        self.loading_label.setText(self.tr("Loaded {0} images.").format(len(self.loaded_images)))
        if self.metrics_window and self.metrics_window.isVisible():
            self.metrics_window.update_metrics(self.loaded_images, order=self._tree_order_indices())

        self.imagesChanged.emit(len(self.loaded_images)) 

    def show_metrics(self):
        if self.metrics_window is None:
            self.metrics_window = MetricsWindow()
            mp = self.metrics_window.metrics_panel
            mp.pointClicked.connect(self.on_metrics_point)
            mp.thresholdChanged.connect(self.on_threshold_change)

        order = self._tree_order_indices()
        self.metrics_window.set_images(self.loaded_images, order=order)
        panel = self.metrics_window.metrics_panel
        self.thresholds_by_group[self.tr("All")] = [line.value() for line in panel.lines]
        self.metrics_window.show()
        self.metrics_window.raise_()

    def on_metrics_point(self, metric_idx, frame_idx):
        item = self.get_tree_item_for_index(frame_idx)
        if not item:
            return
        self._toggle_flag_on_item(item)  

    def _as_float01(self, arr):
        """Convert any stored dtype to float32 in [0..1], with safety normalization."""
        if arr.dtype == np.uint8:
            out = arr.astype(np.float32) / 255.0
            return out

        if arr.dtype == np.uint16:
            out = arr.astype(np.float32) / 65535.0
            return out

        # float path (or anything else): normalize if needed
        out = np.asarray(arr, dtype=np.float32)

        if out.size == 0:
            return out

        # handle NaNs/Infs early
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        mn = float(out.min())
        if mn < 0.0:
            out = out - mn  # shift so min becomes 0

        mx = float(out.max())
        if mx > 1.0 and mx > 0.0:
            out = out / mx  # scale so max becomes 1

        return np.clip(out, 0.0, 1.0)



    def on_threshold_change(self, metric_idx, threshold):
        panel = self.metrics_window.metrics_panel
        if panel.metrics_data is None:
            return

        # figure out which FILTER group we're in
        group = self.metrics_window.group_combo.currentText()
        # ensure we have a 4-slot list for this group
        thr_list = self.thresholds_by_group.setdefault(group, [None]*4)
        # store the new threshold for this metric
        thr_list[metric_idx] = threshold

        # build the list of indices to re-evaluate
        if group == self.tr("All"):
            indices = range(len(self.loaded_images))
        else:
            indices = [
                i for i, e in enumerate(self.loaded_images)
                if e.get('header', {}).get('FILTER','Unknown') == group
            ]

        # re‐flag only those frames in this group, OR across all 4 metrics
        for i in indices:
            entry = self.loaded_images[i]
            flagged = False
            for m, thr in enumerate(thr_list):
                if thr is None:
                    continue
                val = panel.metrics_data[m][i]
                if np.isnan(val):
                    continue
                if (m < 3 and val > thr) or (m == 3 and val < thr):
                    flagged = True
                    break
            entry['flagged'] = flagged

            # update the tree icon
            item = self.get_tree_item_for_index(i)
            if item:
                RED = Qt.GlobalColor.red
                normal = self.fileTree.palette().color(QPalette.ColorRole.WindowText)
                name = item.text(0).lstrip("⚠️ ")
                if flagged:
                    item.setText(0, f"⚠️ {name}")
                    item.setForeground(0, QBrush(RED))
                else:
                    item.setText(0, name)
                    item.setForeground(0, QBrush(normal))

        # now push the *entire* up-to-date flagged list into the panel
        panel.flags = [e['flagged'] for e in self.loaded_images]
        panel._refresh_scatter_colors()
        self.metrics_window._update_status()

    def _rebuild_tree_from_loaded(self):
        """Rebuild the left tree from self.loaded_images without reloading or recomputing."""
        self.fileTree.clear()
        from collections import defaultdict

        grouped = defaultdict(list)
        for entry in self.loaded_images:
            hdr = entry.get('header', {}) or {}
            obj = hdr.get('OBJECT', 'Unknown')
            fil = hdr.get('FILTER', 'Unknown')
            exp = hdr.get('EXPOSURE', 'Unknown')
            grouped[(obj, fil, exp)].append(entry['file_path'])

        # natural sort within each leaf group
        for key, paths in grouped.items():
            paths.sort(key=lambda p: self._natural_key(os.path.basename(p)))

        by_object = defaultdict(lambda: defaultdict(dict))
        for (obj, fil, exp), paths in grouped.items():
            by_object[obj][fil][exp] = paths

        for obj in sorted(by_object, key=lambda o: o.lower()):
            obj_item = QTreeWidgetItem([self.tr("Object: {0}").format(obj)])
            self.fileTree.addTopLevelItem(obj_item)
            obj_item.setExpanded(True)

            for fil in sorted(by_object[obj], key=lambda f: f.lower()):
                filt_item = QTreeWidgetItem([self.tr("Filter: {0}").format(fil)])
                obj_item.addChild(filt_item)
                filt_item.setExpanded(True)

                for exp in sorted(by_object[obj][fil], key=lambda e: str(e).lower()):
                    exp_item = QTreeWidgetItem([self.tr("Exposure: {0}").format(exp)])
                    filt_item.addChild(exp_item)
                    exp_item.setExpanded(True)

                    for p in by_object[obj][fil][exp]:
                        leaf = QTreeWidgetItem([os.path.basename(p)])
                        leaf.setData(0, Qt.ItemDataRole.UserRole, p)
                        exp_item.addChild(leaf)

        # 🔹 Re-apply flagged styling
        RED = Qt.GlobalColor.red
        normal = self.fileTree.palette().color(QPalette.ColorRole.WindowText)

        for idx, entry in enumerate(self.loaded_images):
            item = self.get_tree_item_for_index(idx)
            if not item:
                continue
            base = os.path.basename(self.image_paths[idx])
            if entry.get("flagged", False):
                item.setText(0, f"⚠️ {base}")
                item.setForeground(0, QBrush(RED))
            else:
                item.setText(0, base)
                item.setForeground(0, QBrush(normal))



    def _after_list_changed(self, removed_indices: List[int] | None = None):
        self._rebuild_tree_from_loaded()
        self.imagesChanged.emit(len(self.loaded_images))

        if self.metrics_window and self.metrics_window.isVisible():
            # ✅ safest: rebind images + rebuild plot order from tree
            self.metrics_window.set_images(self.loaded_images, order=self._tree_order_indices())
            self._sync_metrics_flags()

    def get_tree_item_for_index(self, idx):
        target_path = self.image_paths[idx]
        for item in self.get_all_leaf_items():
            p = item.data(0, Qt.ItemDataRole.UserRole)
            if p == target_path:
                return item
        return None


    def compute_metric(self, metric_idx, entry):
        """Recompute a single metric for one image.  Use cached orig_background for metric 2."""
        # metric 2 is the pre-stretch background we already computed
        if metric_idx == 2:
            return entry.get('orig_background', np.nan)

        # otherwise rebuild a float32 [0..1] array from whatever dtype we stored
        img = entry['image_data']
        if img.dtype == np.uint8:
            data = img.astype(np.float32)/255.0
        elif img.dtype == np.uint16:
            data = img.astype(np.float32)/65535.0
        else:
            data = np.asarray(img, dtype=np.float32)
        if data.ndim == 3:
            data = data.mean(axis=2)

        # run SEP for the other metrics
        bkg = sep.Background(data)
        back, gr, rr = bkg.back(), bkg.globalback, bkg.globalrms
        cat = sep.extract(data - back, 5.0, err=gr, minarea=9)
        if len(cat)==0:
            return np.nan

        sig = np.sqrt(cat['a']*cat['b'])
        if metric_idx == 0:
            return np.nanmedian(2.3548*sig)
        elif metric_idx == 1:
            return np.nanmedian(1 - (cat['b']/cat['a']))
        else:  # metric_idx == 3 (star count)
            return len(cat)


    def init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        toggle_shortcut = QShortcut(QKeySequence("Space"), self.fileTree)
        def _toggle_play():
            if self.playback_timer.isActive():
                self.stop_playback()
            else:
                self.start_playback()
        toggle_shortcut.activated.connect(_toggle_play)        
        # Create a shortcut for the "F" key to flag images
        flag_shortcut = QShortcut(QKeySequence("F"), self.fileTree)
        flag_shortcut.activated.connect(self.flag_current_image)

    def openDirectoryDialog(self):
        """Allow users to select a directory and load all images within it recursively."""
        directory = QFileDialog.getExistingDirectory(self, self.tr("Select Directory"), "")
        if directory:
            # Supported image extensions
            supported_extensions = (
                '.png', '.tif', '.tiff', '.fits', '.fit',
                '.xisf', '.cr2', '.nef', '.arw', '.dng', '.raf',
                '.orf', '.rw2', '.pef'
            )

            # Collect all image file paths recursively
            new_file_paths = []
            for root, _, files in os.walk(directory):
                for file in sorted(files, key=str.lower):  # 🔹 Sort alphabetically (case-insensitive)
                    if file.lower().endswith(supported_extensions):
                        full_path = os.path.join(root, file)
                        if full_path not in self.image_paths:  # Avoid duplicates
                            new_file_paths.append(full_path)

            if new_file_paths:
                self.loadImages(new_file_paths)
            else:
                QMessageBox.information(self, self.tr("No Images Found"), self.tr("No supported image files were found in the selected directory."))


    def clearImages(self):
        """Clear all loaded images and reset the tree view."""
        confirmation = QMessageBox.question(
            self,
            self.tr("Clear All Images"),
            self.tr("Are you sure you want to clear all loaded images?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmation == QMessageBox.StandardButton.Yes:
            self.stop_playback()
            self.image_paths.clear()
            self.loaded_images.clear()
            self.image_labels.clear()
            self.fileTree.clear()
            self.preview_label.clear()
            self.preview_label.setText(self.tr('No image selected.'))
            self.current_pixmap = None
            self.progress_bar.setValue(0)
            self.loading_label.setText(self.tr("Loading images..."))
            self.imagesChanged.emit(len(self.loaded_images)) 

            # (legacy) if you still have this, you can delete it:
            # self.thresholds = [None, None, None, None]

            # also reset the metrics panel (if it’s open)
            if self.metrics_window is not None:
                mp = self.metrics_window.metrics_panel
                # clear out old data & reset flags / thresholds
                mp.metrics_data = None
                mp._threshold_initialized = [False]*4
                for scat in mp.scats:
                    scat.clear()
                for line in mp.lines:
                    line.setPos(0)

                # clear per‐group threshold storage
                self.metrics_window._thresholds_per_group.clear()

        # finally, tell the MetricsWindow to fully re‐init with no images
        if self.metrics_window is not None:
            self.metrics_window.update_metrics([])
   


    @staticmethod
    def _load_one_image(file_path: str, target_dtype):
        """Load + pre-process one image & return all metadata."""

        # 1) load
        image, header, bit_depth, is_mono = load_image(file_path)
        if image is None or image.size == 0:
            msg = QCoreApplication.translate("BlinkTab", "Empty image")
            raise ValueError(msg)

        # 2) optional debayer
        if is_mono:
            image = BlinkTab.debayer_image(image, file_path, header)

        image = BlinkTab._ensure_float01(image)

        data = np.asarray(image, dtype=np.float32, order='C')
        if data.ndim == 3:
            data = data.mean(axis=2)
        bkg = sep.Background(data)
        global_back = bkg.globalback

        target_med = 0.25
        if image.ndim == 2:
            stretched = stretch_mono_image(image, target_med)
        else:
            stretched = stretch_color_image(image, target_med, linked=False)

        clipped = np.clip(stretched, 0.0, 1.0)
        if target_dtype is np.uint8:
            stored = (clipped * 255).astype(np.uint8)
        elif target_dtype is np.uint16:
            stored = (clipped * 65535).astype(np.uint16)
        else:
            stored = clipped.astype(np.float32)

        return file_path, header, bit_depth, is_mono, stored, global_back

    @staticmethod
    def debayer_image(image, file_path, header):
        """Check if image is OSC (One-Shot Color) and debayer if required."""
        if file_path.lower().endswith(('.fits', '.fit')):
            bayer_pattern = header.get('BAYERPAT', None)
            if bayer_pattern:
                image = debayer_fits_fast(image, bayer_pattern)
        elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')):
            image = debayer_raw_fast(image, bayer_pattern="RGGB")
        return image

    @staticmethod
    def _natural_key(path: str):
        """
        Split a filename into text and integer chunks so that 
        “…_2.fit” sorts before “…_10.fit”.
        """
        name = os.path.basename(path)
        return [int(tok) if tok.isdigit() else tok.lower()
                for tok in re.split(r'(\d+)', name)]

    def loadImages(self, file_paths):
        # 0) early out
        if not file_paths:
            return

        # ---------- NEW: natural sort the list of filenames ----------
        file_paths = sorted(file_paths, key=lambda p: self._natural_key(os.path.basename(p)))

        # 1) pick dtype based on RAM
        mem = psutil.virtual_memory()
        avail = mem.available / (1024**3)
        if avail <= 16:
            target_dtype = np.uint8
        elif avail <= 32:
            target_dtype = np.uint16
        else:
            target_dtype = np.float32

        total = len(file_paths)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        self.image_paths.clear()
        self.loaded_images.clear()
        self.fileTree.clear()

        # ---------- NEW: Retry-aware parallel load ----------
        MAX_RETRIES = 2
        RETRY_DELAY = 2
        remaining = list(file_paths)
        completed = []
        attempt = 0

        while remaining and attempt <= MAX_RETRIES:
            
            total_cpus = os.cpu_count() or 1
            reserved_cpus = min(4, max(1, int(total_cpus * 0.25)))
            max_workers = max(1, min(total_cpus - reserved_cpus, 60))

            futures = {}
            failed = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for path in remaining:
                    futures[executor.submit(self._load_one_image, path, target_dtype)] = path
                for fut in as_completed(futures):
                    path = futures[fut]
                    try:
                        result = fut.result()
                        completed.append(result)
                        done = len(completed)
                        self.progress_bar.setValue(int(100 * done / total))
                        QApplication.processEvents()
                    except Exception as e:
                        print(f"[WARN][Attempt {attempt}] Failed to load {path}: {e}")
                        failed.append(path)

            remaining = failed
            attempt += 1
            if remaining:
                print(f"[Retry] {len(remaining)} images will be retried after {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

        if remaining:
            print(f"[FAILURE] These files failed to load after {MAX_RETRIES} retries:")
            for path in remaining:
                print(f"  - {path}")

        # ---------- Unpack completed results ----------
        for path, header, bit_depth, is_mono, stored, back in completed:
            header = header or {}
            self.image_paths.append(path)
            self.loaded_images.append({
                'file_path':      path,
                'image_data':     stored,
                'header':         header,
                'bit_depth':      bit_depth,
                'is_mono':        is_mono,
                'flagged':        False,
                'orig_background': back
            })

        # 3) rebuild object/filter/exposure tree
        grouped = defaultdict(list)
        for entry in self.loaded_images:
            hdr = entry['header']
            obj = hdr.get('OBJECT', 'Unknown')
            filt = hdr.get('FILTER', 'Unknown')
            exp = hdr.get('EXPOSURE', 'Unknown')
            grouped[(obj, filt, exp)].append(entry['file_path'])

        for key, paths in grouped.items():
            paths.sort(key=lambda p: self._natural_key(os.path.basename(p)))
        by_object = defaultdict(lambda: defaultdict(dict))
        for (obj, filt, exp), paths in grouped.items():
            by_object[obj][filt][exp] = paths

        for obj in sorted(by_object, key=lambda o: o.lower()):
            obj_item = QTreeWidgetItem([f"Object: {obj}"])
            self.fileTree.addTopLevelItem(obj_item)
            obj_item.setExpanded(True)

            for filt in sorted(by_object[obj], key=lambda f: f.lower()):
                filt_item = QTreeWidgetItem([f"Filter: {filt}"])
                obj_item.addChild(filt_item)
                filt_item.setExpanded(True)

                for exp in sorted(by_object[obj][filt], key=lambda e: str(e).lower()):
                    exp_item = QTreeWidgetItem([f"Exposure: {exp}"])
                    filt_item.addChild(exp_item)
                    exp_item.setExpanded(True)

                    for p in by_object[obj][filt][exp]:
                        leaf = QTreeWidgetItem([os.path.basename(p)])
                        leaf.setData(0, Qt.ItemDataRole.UserRole, p)  
                        exp_item.addChild(leaf)

        self.loading_label.setText(self.tr("Loaded {0} images.").format(len(self.loaded_images)))
        self.progress_bar.setValue(100)
        self.imagesChanged.emit(len(self.loaded_images))
        if self.metrics_window and self.metrics_window.isVisible():
            self.metrics_window.update_metrics(self.loaded_images, order=self._tree_order_indices())


    def findTopLevelItemByName(self, name):
        """Find a top-level item in the tree by its name."""
        for index in range(self.fileTree.topLevelItemCount()):
            item = self.fileTree.topLevelItem(index)
            if item.text(0) == name:
                return item
        return None

    def findChildItemByName(self, parent, name):
        """Find a child item under a given parent by its name."""
        for index in range(parent.childCount()):
            child = parent.child(index)
            if child.text(0) == name:
                return child
        return None


    def _toggle_flag_on_item(self, item: QTreeWidgetItem, *, sync_metrics: bool = True):
        idx = self._leaf_index(item)
        if idx is None:
            return

        entry = self.loaded_images[idx]
        entry['flagged'] = not bool(entry.get('flagged', False))

        RED = Qt.GlobalColor.red
        normal_color = self.fileTree.palette().color(QPalette.ColorRole.WindowText)

        base = os.path.basename(self.image_paths[idx])

        if entry['flagged']:
            item.setText(0, f"⚠️ {base}")
            item.setForeground(0, QBrush(RED))
        else:
            item.setText(0, base)
            item.setForeground(0, QBrush(normal_color))

        # Keep UserRole correct (in case this was a legacy leaf)
        item.setData(0, Qt.ItemDataRole.UserRole, self.image_paths[idx])

        if sync_metrics:
            self._sync_metrics_flags()

    def flag_current_image(self):
        item = self.fileTree.currentItem()
        if not item:
            QMessageBox.warning(self, self.tr("No Selection"), self.tr("No image is currently selected to flag."))
            return
        self._toggle_flag_on_item(item)   # ← this now updates the metrics panel too
        self.next_item()


    def on_current_item_changed(self, current, previous):
        """Ensure the selected item is visible by scrolling to it."""
        if current:
            self.fileTree.scrollToItem(current, QAbstractItemView.ScrollHint.PositionAtCenter)

    def previous_item(self):
        """Select the previous item in the TreeWidget."""
        current_item = self.fileTree.currentItem()
        if current_item:
            all_items = self.get_all_leaf_items()
            current_index = all_items.index(current_item)
            if current_index > 0:
                previous_item = all_items[current_index - 1]
            else:
                previous_item = all_items[-1]  # Loop back to the last item
            self.fileTree.setCurrentItem(previous_item)
            #self.on_item_clicked(previous_item, 0)  # Update the preview

    def next_item(self):
        """Select the next item in the TreeWidget, looping back to the first item if at the end."""
        current_item = self.fileTree.currentItem()
        if current_item:
            # Get all leaf items
            all_items = self.get_all_leaf_items()

            # Check if the current item is in the leaf items
            try:
                current_index = all_items.index(current_item)
            except ValueError:
                # If the current item is not a leaf, move to the first leaf item
                print("Current item is not a leaf. Selecting the first leaf item.")
                if all_items:
                    next_item = all_items[0]
                    self.fileTree.setCurrentItem(next_item)
                    self.on_item_clicked(next_item, 0)
                return

            # Select the next leaf item or loop back to the first
            if current_index < len(all_items) - 1:
                next_item = all_items[current_index + 1]
            else:
                next_item = all_items[0]  # Loop back to the first item

            self.fileTree.setCurrentItem(next_item)
            #self.on_item_clicked(next_item, 0)  # Update the preview
        else:
            print("No current item selected.")

    def get_all_leaf_items(self):
        """Get a flat list of all leaf items (actual files) in the TreeWidget."""
        def recurse(parent):
            items = []
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child.childCount() == 0:  # It's a leaf item
                    items.append(child)
                else:
                    items.extend(recurse(child))
            return items

        root = self.fileTree.invisibleRootItem()
        return recurse(root)

    def start_playback(self):
        """Start playing through the items in the TreeWidget."""
        if self.playback_timer.isActive():
            return

        leaves = self.get_all_leaf_items()
        if not leaves:
            QMessageBox.information(self, self.tr("No Images"), self.tr("Load some images first."))
            return

        # Ensure a current leaf item is selected
        cur = self.fileTree.currentItem()
        if cur is None or cur.childCount() > 0:
            self.fileTree.setCurrentItem(leaves[0])

        # Honor current fps setting
        self._apply_playback_interval()
        self.playback_timer.start()

    def stop_playback(self):
        """Stop playing through the items."""
        if self.playback_timer.isActive():
            self.playback_timer.stop()


    def openFileDialog(self):
        """Allow users to select multiple images and add them to the existing list."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            self.tr("Open Images"),
            "",
            self.tr("Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef);;All Files (*)")
        )
        
        # Filter out already loaded images to prevent duplicates
        new_file_paths = [path for path in file_paths if path not in self.image_paths]

        if new_file_paths:
            self.loadImages(new_file_paths)
        else:
            QMessageBox.information(self, self.tr("No New Images"), self.tr("No new images were selected or all selected images are already loaded."))


    def debayer_fits(self, image_data, bayer_pattern):
        """Debayer a FITS image using a basic Bayer pattern (2x2)."""
        if bayer_pattern == 'RGGB':
            # RGGB Bayer pattern
            r = image_data[::2, ::2]  # Red
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            b = image_data[1::2, 1::2]  # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'BGGR':
            # BGGR Bayer pattern
            b = image_data[::2, ::2]  # Blue
            g1 = image_data[::2, 1::2]  # Green 1
            g2 = image_data[1::2, ::2]  # Green 2
            r = image_data[1::2, 1::2]  # Red

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GRBG':
            # GRBG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            r = image_data[::2, 1::2]  # Red
            b = image_data[1::2, ::2]  # Blue
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        elif bayer_pattern == 'GBRG':
            # GBRG Bayer pattern
            g1 = image_data[::2, ::2]  # Green 1
            b = image_data[::2, 1::2]  # Blue
            r = image_data[1::2, ::2]  # Red
            g2 = image_data[1::2, 1::2]  # Green 2

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(self.tr("Unsupported Bayer pattern: {0}").format(bayer_pattern))

    def remove_item_from_tree(self, file_path):
        """Remove a specific item from the tree view based on file path."""
        file_name = os.path.basename(file_path)
        root = self.fileTree.invisibleRootItem()

        def recurse(parent):
            for index in range(parent.childCount()):
                child = parent.child(index)
                if child.text(0).endswith(file_name):
                    parent.removeChild(child)
                    return True
                if recurse(child):
                    return True
            return False

        recurse(root)

    def add_item_to_tree(self, file_path):
        """Add a specific item to the tree view based on file path."""
        # Extract metadata for grouping
        image_entry = next((img for img in self.loaded_images if img['file_path'] == file_path), None)
        if not image_entry:
            return

        header = image_entry['header']
        object_name = header.get('OBJECT', 'Unknown') if header else 'Unknown'
        filter_name = header.get('FILTER', 'Unknown') if header else 'Unknown'
        exposure_time = header.get('EXPOSURE', 'Unknown') if header else 'Unknown'

        # Group images by filter and exposure time
        group_key = (object_name, filter_name, exposure_time)

        # Find or create the object item
        object_item = self.findTopLevelItemByName(f"Object: {object_name}")
        if not object_item:
            object_item = QTreeWidgetItem([f"Object: {object_name}"])
            self.fileTree.addTopLevelItem(object_item)
            object_item.setExpanded(True)

        # Find or create the filter item
        filter_item = self.findChildItemByName(object_item, f"Filter: {filter_name}")
        if not filter_item:
            filter_item = QTreeWidgetItem([f"Filter: {filter_name}"])
            object_item.addChild(filter_item)
            filter_item.setExpanded(True)

        # Find or create the exposure item
        exposure_item = self.findChildItemByName(filter_item, f"Exposure: {exposure_time}")
        if not exposure_item:
            exposure_item = QTreeWidgetItem([f"Exposure: {exposure_time}"])
            filter_item.addChild(exposure_item)
            exposure_item.setExpanded(True)

        # Add the file item
        file_name = os.path.basename(file_path)
        item = QTreeWidgetItem([file_name])
        item.setData(0, Qt.ItemDataRole.UserRole, file_path)
        exposure_item.addChild(item)

    def _tree_order_indices(self) -> list[int]:
        """Return the indices of loaded_images in the exact order the Tree shows."""
        order = []
        for leaf in self.get_all_leaf_items():
            path = leaf.data(0, Qt.ItemDataRole.UserRole)
            if not path:
                # fallback by basename if old items exist
                name = leaf.text(0).lstrip("⚠️ ").strip()
                path = next((p for p in self.image_paths if os.path.basename(p) == name), None)
            if path and path in self.image_paths:
                order.append(self.image_paths.index(path))
        return order

    def debayer_raw(self, raw_image_data, bayer_pattern="RGGB"):
        """Debayer a RAW image based on the Bayer pattern, ensuring even dimensions."""
        H, W = raw_image_data.shape
        # Crop to even dimensions if necessary
        if H % 2 != 0:
            raw_image_data = raw_image_data[:H-1, :]
        if W % 2 != 0:
            raw_image_data = raw_image_data[:, :W-1]
        
        if bayer_pattern == 'RGGB':
            r = raw_image_data[::2, ::2]      # Red
            g1 = raw_image_data[::2, 1::2]     # Green 1
            g2 = raw_image_data[1::2, ::2]     # Green 2
            b = raw_image_data[1::2, 1::2]     # Blue

            # Average green channels
            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        elif bayer_pattern == 'BGGR':
            b = raw_image_data[::2, ::2]      # Blue
            g1 = raw_image_data[::2, 1::2]     # Green 1
            g2 = raw_image_data[1::2, ::2]     # Green 2
            r = raw_image_data[1::2, 1::2]     # Red

            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        elif bayer_pattern == 'GRBG':
            g1 = raw_image_data[::2, ::2]     # Green 1
            r = raw_image_data[::2, 1::2]      # Red
            b = raw_image_data[1::2, ::2]      # Blue
            g2 = raw_image_data[1::2, 1::2]     # Green 2

            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        elif bayer_pattern == 'GBRG':
            g1 = raw_image_data[::2, ::2]     # Green 1
            b = raw_image_data[::2, 1::2]      # Blue
            r = raw_image_data[1::2, ::2]      # Red
            g2 = raw_image_data[1::2, 1::2]     # Green 2

            g = (g1 + g2) / 2
            return np.stack([r, g, b], axis=-1)
        else:
            raise ValueError(self.tr("Unsupported Bayer pattern: {0}").format(bayer_pattern))

    

    def on_item_clicked(self, item, column):
        self.fileTree.setFocus()
        if not item or item.childCount() > 0:
            return

        file_path = self._leaf_path(item)
        if not file_path:
            return

        self._capture_view_center_norm()

        try:
            idx = self.image_paths.index(file_path)
        except ValueError:
            return

        entry = self.loaded_images[idx]

        # ✅ single source of truth (handles aggressive + mono + color)
        disp8 = self._make_display_frame(entry)

        qimage = self.convert_to_qimage(disp8)
        self.current_pixmap = QPixmap.fromImage(qimage)
        self.apply_zoom()


    def _capture_view_center_norm(self):
        """Remember the current viewport center as a fraction of the content size."""
        sa = self.scroll_area
        vp = sa.viewport()
        content_w = max(1, self.preview_label.width())
        content_h = max(1, self.preview_label.height())
        if content_w <= 1 or content_h <= 1:
            return
        hbar = sa.horizontalScrollBar()
        vbar = sa.verticalScrollBar()
        cx = hbar.value() + vp.width()  / 2.0
        cy = vbar.value() + vp.height() / 2.0
        self._view_center_norm = (cx / content_w, cy / content_h)

    def _restore_view_center_norm(self):
        """Restore the viewport center captured earlier (if any)."""
        if not self._view_center_norm:
            return
        sa = self.scroll_area
        vp = sa.viewport()
        content_w = max(1, self.preview_label.width())
        content_h = max(1, self.preview_label.height())
        cx = self._view_center_norm[0] * content_w
        cy = self._view_center_norm[1] * content_h
        hbar = sa.horizontalScrollBar()
        vbar = sa.verticalScrollBar()
        h_target = int(round(cx - vp.width()  / 2.0))
        v_target = int(round(cy - vp.height() / 2.0))
        h_target = max(hbar.minimum(), min(hbar.maximum(), h_target))
        v_target = max(vbar.minimum(), min(vbar.maximum(), v_target))
        # Set after layout settles to avoid fighting size changes
        QTimer.singleShot(0, lambda: (hbar.setValue(h_target), vbar.setValue(v_target)))

    def apply_zoom(self):
        """Apply current zoom to pixmap without losing scroll position."""
        if not self.current_pixmap:
            return

        # keep current center if we already showed something
        had_content = (self.preview_label.pixmap() is not None) and (self.preview_label.width() > 0)

        if had_content:
            self._capture_view_center_norm()
        else:
            # first time: default center
            self._view_center_norm = (0.5, 0.5)

        # scale and show
        base_w = self.current_pixmap.width()
        base_h = self.current_pixmap.height()
        scaled_w = max(1, int(round(base_w * self.zoom_level)))
        scaled_h = max(1, int(round(base_h * self.zoom_level)))

        scaled = self.current_pixmap.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_label.resize(scaled.size())

        # restore the center we captured (or 0.5,0.5 for first time)
        self._restore_view_center_norm()

    def wheelEvent(self, event: QWheelEvent):
        # Check the vertical delta to determine zoom direction.
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        # Accept the event so it isn’t propagated further (e.g. to the scroll area).
        event.accept()


    def zoom_in(self):
        """Increase the zoom level and refresh the image."""
        self.zoom_level = min(self.zoom_level * 1.2, 3.0)  # Cap at 3x
        self.apply_zoom()


    def zoom_out(self):
        """Decrease the zoom level and refresh the image."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.05)  # Cap at 0.2x
        self.apply_zoom()


    def fit_to_preview(self):
        """Adjust the zoom level so the image fits within the QScrollArea viewport."""
        if self.current_pixmap:
            # Get the size of the QScrollArea's viewport
            viewport_size = self.scroll_area.viewport().size()
            pixmap_size = self.current_pixmap.size()

            # Calculate the zoom level required to fit the pixmap in the QScrollArea viewport
            width_ratio = viewport_size.width() / pixmap_size.width()
            height_ratio = viewport_size.height() / pixmap_size.height()
            self.zoom_level = min(width_ratio, height_ratio)

            # Apply the zoom level
            self.apply_zoom()
        else:
            print("No image loaded. Cannot fit to preview.")
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No image loaded. Cannot fit to preview."))

    def _is_leaf(self, item: Optional[QTreeWidgetItem]) -> bool:
        return bool(item and item.childCount() == 0)

    def on_right_click(self, pos):
        item = self.fileTree.itemAt(pos)
        if not self._is_leaf(item):
            # Optional: expand/collapse-only menu, or just ignore
            return

        menu = QMenu(self)

        push_action = QAction(self.tr("Open in Document Window"), self)
        push_action.triggered.connect(lambda: self.push_to_docs(item))
        menu.addAction(push_action)

        rename_action = QAction(self.tr("Rename"), self)
        rename_action.triggered.connect(lambda: self.rename_item(item))
        menu.addAction(rename_action)

        # 🔹 NEW: batch rename selected
        batch_rename_action = QAction(self.tr("Batch Rename Selected…"), self)
        batch_rename_action.triggered.connect(self.batch_rename_items)
        menu.addAction(batch_rename_action)

        move_action = QAction(self.tr("Move Selected Items"), self)
        move_action.triggered.connect(self.move_items)
        menu.addAction(move_action)

        delete_action = QAction(self.tr("Delete Selected Items"), self)
        delete_action.triggered.connect(self.delete_items)
        menu.addAction(delete_action)

        menu.addSeparator()

        batch_delete_action = QAction(self.tr("Delete All Flagged Images"), self)
        batch_delete_action.triggered.connect(self.batch_delete_flagged_images)
        menu.addAction(batch_delete_action)

        batch_move_action = QAction(self.tr("Move All Flagged Images"), self)
        batch_move_action.triggered.connect(self.batch_move_flagged_images)
        menu.addAction(batch_move_action)

        # 🔹 NEW: rename all flagged images
        rename_flagged_action = QAction(self.tr("Rename Flagged Images…"), self)
        rename_flagged_action.triggered.connect(self.rename_flagged_images)
        menu.addAction(rename_flagged_action)

        menu.addSeparator()

        send_lights_act = QAction(self.tr("Send to Stacking → Lights"), self)
        send_lights_act.triggered.connect(self._send_to_stacking_lights)
        menu.addAction(send_lights_act)

        send_integ_act = QAction(self.tr("Send to Stacking → Integration"), self)
        send_integ_act.triggered.connect(self._send_to_stacking_integration)
        menu.addAction(send_integ_act)

        menu.exec(self.fileTree.mapToGlobal(pos))


    def push_to_docs(self, item: QTreeWidgetItem):
        """
        Push the currently selected blink leaf image into DocManager as a new document,
        preserving all original metadata (original_header, meta, bit_depth, is_mono, etc.)
        and swapping ONLY the numpy image array.
        """
        if not item or item.childCount() > 0:
            return

        # --- Resolve full path safely (UserRole-first) ---
        file_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not file_path or not isinstance(file_path, str):
            # legacy fallback: try to map by displayed name
            file_name = item.text(0).lstrip("⚠️ ").strip()
            file_path = next((p for p in self.image_paths if os.path.basename(p) == file_name), None)

        if not file_path:
            return

        try:
            idx = self.image_paths.index(file_path)
        except ValueError:
            return

        entry = self.loaded_images[idx]

        # --- Find main window + doc manager ---
        mw = self._main_window()
        dm = self.doc_manager or (getattr(mw, "docman", None) if mw else None)
        if not mw or not dm:
            QMessageBox.warning(self, self.tr("Document Manager"), self.tr("Main window or DocManager not available."))
            return

        # --- Build the swapped payload (image replaced, metadata preserved) ---
        # Whatever you're storing as entry['image_data'] (uint16/float/etc), normalize to float01 for display pipeline.
        # If your DocManager expects native dtype instead, swap _as_float01 for your native image.
        np_image_f01 = self._as_float01(entry["image_data"]).astype(np.float32, copy=False)

        # Preserve your full load_image return structure as much as possible:
        # load_image returns: image, original_header, bit_depth, is_mono, meta
        original_header = entry.get("original_header", entry.get("header", None))
        bit_depth       = entry.get("bit_depth", None)
        is_mono         = entry.get("is_mono", None)
        meta            = entry.get("meta", {})

        # Keep meta dict style your app uses; add source tag without clobbering
        if isinstance(meta, dict):
            meta = dict(meta)
            meta.setdefault("source", "BlinkComparatorPro")
            meta.setdefault("file_path", file_path)

        # This is the "all the other stuff" you wanted preserved
        payload = {
            "file_path": file_path,
            "original_header": original_header,
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "meta": meta,
            "source": "BlinkComparatorPro",
        }

        title = os.path.basename(file_path)

        # --- Create document using whatever DocManager API exists ---
        doc = None
        try:
            # Preferred: if you have a method that mirrors open_file/load_image shape
            if hasattr(dm, "open_from_load_image"):
                # (image, original_header, bit_depth, is_mono, meta)
                doc = dm.open_from_load_image(np_image_f01, original_header, bit_depth, is_mono, meta, title=title)

            elif hasattr(dm, "open_array"):
                # Some of your code expects metadata in doc.metadata; pass payload whole
                doc = dm.open_array(np_image_f01, metadata=payload, title=title)

            elif hasattr(dm, "open_numpy"):
                doc = dm.open_numpy(np_image_f01, metadata=payload, title=title)

            elif hasattr(dm, "create_document"):
                # Try both signatures
                try:
                    doc = dm.create_document(image=np_image_f01, metadata=payload, name=title)
                except TypeError:
                    doc = dm.create_document(np_image_f01, payload, title)

            else:
                raise AttributeError("DocManager lacks a known creation method")

        except Exception as e:
            QMessageBox.critical(self, self.tr("Doc Manager"), self.tr("Failed to create document:\n{0}").format(e))
            return

        if doc is None:
            QMessageBox.critical(self, self.tr("Doc Manager"), self.tr("DocManager returned no document."))
            return

        # --- Hand off to DocManager flow (DocManager should trigger MDI + window creation) ---
        try:
            # If your architecture already auto-spawns windows on documentAdded,
            # you should NOT call mw._spawn_subwindow_for(doc) here.
            if hasattr(dm, "add_document"):
                dm.add_document(doc)
            elif hasattr(dm, "register_document"):
                dm.register_document(doc)
            else:
                # If open_array/open_numpy already registers the doc internally, do nothing.
                pass

            # If you *must* spawn manually (older path), keep as fallback
            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(doc)

            if hasattr(mw, "_log"):
                mw._log(f"Blink → opened '{title}' as new document")

        except Exception as e:
            QMessageBox.critical(self, self.tr("UI"), self.tr("Failed to open subwindow:\n{0}").format(e))



    # optional shim to keep any old calls working
    def push_image_to_manager(self, item):
        self.push_to_docs(item)



    def rename_item(self, item: QTreeWidgetItem):
        if not item or item.childCount() > 0:
            return

        idx = self._leaf_index(item)
        if idx is None:
            return

        old_path = self.image_paths[idx]
        old_base = os.path.basename(old_path)

        new_name, ok = QInputDialog.getText(
            self,
            self.tr("Rename Image"),
            self.tr("Enter new name:"),
            text=old_base
        )
        if not ok:
            return

        new_name = (new_name or "").strip()
        if not new_name:
            return

        new_path = os.path.join(os.path.dirname(old_path), new_name)

        # Avoid overwrite
        if os.path.exists(new_path):
            QMessageBox.critical(self, self.tr("Error"), self.tr("A file with that name already exists."))
            return

        try:
            os.rename(old_path, new_path)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to rename the file: {0}").format(e))
            return

        # Update internal structures
        self.image_paths[idx] = new_path
        self.loaded_images[idx]['file_path'] = new_path

        # Update the leaf item
        flagged = bool(self.loaded_images[idx].get("flagged", False))
        self._set_leaf_display(item, base_name=new_name, flagged=flagged, full_path=new_path)

        # Rebuild so natural sort stays correct and groups update
        self._after_list_changed()
        self._sync_metrics_flags()


    def rename_flagged_images(self):
        """Prefix all *flagged* images on disk and in the tree."""
        # Collect indices of flagged frames
        flagged_indices = [i for i, e in enumerate(self.loaded_images)
                           if e.get("flagged", False)]

        if not flagged_indices:
            QMessageBox.information(
                self,
                self.tr("Rename Flagged Images"),
                self.tr("There are no flagged images to rename.")
            )
            return

        # Small dialog like in your mockup: just a prefix field
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Rename flagged images"))
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel(self.tr("Prefix to add to flagged image filenames:"), dlg))

        prefix_edit = QLineEdit(dlg)
        prefix_edit.setText("Bad_")  # sensible default
        layout.addWidget(prefix_edit)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        prefix = prefix_edit.text()
        if prefix is None:
            prefix = ""
        prefix = prefix.strip()
        if not prefix:
            # Allow empty but warn – otherwise user may be confused
            ret = QMessageBox.question(
                self,
                self.tr("No Prefix"),
                self.tr("No prefix entered. This will not change any filenames.\n\n"
                "Continue anyway?"),
                QMessageBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        successes = 0
        failures = []

        for idx in flagged_indices:
            old_path = self.image_paths[idx]
            directory, base = os.path.split(old_path)

            new_base = f"{prefix}{base}"
            new_path = os.path.join(directory, new_base)

            # Skip if unchanged
            if new_path == old_path:
                continue

            # Avoid overwriting an existing file
            if os.path.exists(new_path):
                failures.append((old_path, "target already exists"))
                continue

            try:
                os.rename(old_path, new_path)
            except Exception as e:
                failures.append((old_path, str(e)))
                continue

            # Update internal paths
            self.image_paths[idx] = new_path
            self.loaded_images[idx]["file_path"] = new_path

            # Update tree item text + UserRole data
            item = self.get_tree_item_for_index(idx)
            if item is not None:
                # preserve ⚠️ prefix
                disp_name = new_base
                if self.loaded_images[idx].get("flagged", False):
                    disp_name = f"⚠️ {disp_name}"
                item.setText(0, disp_name)
                item.setData(0, Qt.ItemDataRole.UserRole, new_path)

            successes += 1

        # Rebuild tree so new names are naturally re-sorted, keep flags
        self._after_list_changed()
        # Also sync the metrics panel flags/colors
        self._sync_metrics_flags()

        msg = self.tr("Renamed {0} flagged image{1}.").format(successes, 's' if successes != 1 else '')
        if failures:
            msg += self.tr("\n\n{0} file(s) could not be renamed:").format(len(failures))
            for old, err in failures[:10]:  # don’t spam too hard
                msg += f"\n• {os.path.basename(old)} – {err}"

        QMessageBox.information(self, self.tr("Rename Flagged Images"), msg)


    def batch_rename_items(self):
        """Batch rename selected leaf items by adding a prefix and/or suffix."""
        selected_items = [it for it in self.fileTree.selectedItems() if it and it.childCount() == 0]
        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No individual image items selected for renaming."))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Batch Rename"))
        dialog_layout = QVBoxLayout(dialog)

        dialog_layout.addWidget(QLabel(self.tr("Enter a prefix or suffix to rename selected files:"), dialog))

        form_layout = QHBoxLayout()
        prefix_field = QLineEdit(dialog)
        prefix_field.setPlaceholderText(self.tr("Prefix"))
        form_layout.addWidget(prefix_field)

        mid_label = QLabel(self.tr("filename"), dialog)
        mid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form_layout.addWidget(mid_label)

        suffix_field = QLineEdit(dialog)
        suffix_field.setPlaceholderText(self.tr("Suffix"))
        form_layout.addWidget(suffix_field)
        dialog_layout.addLayout(form_layout)

        btns = QHBoxLayout()
        ok_button = QPushButton(self.tr("OK"), dialog)
        cancel_button = QPushButton(self.tr("Cancel"), dialog)
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        btns.addWidget(ok_button)
        btns.addWidget(cancel_button)
        dialog_layout.addLayout(btns)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        prefix = (prefix_field.text() or "").strip()
        suffix = (suffix_field.text() or "").strip()

        if not prefix and not suffix:
            QMessageBox.information(self, self.tr("Batch Rename"), self.tr("No prefix or suffix entered. Nothing to do."))
            return

        renamed = 0
        failures = []

        # Work on indices so we can update lists safely
        indices = []
        for it in selected_items:
            idx = self._leaf_index(it)
            if idx is not None:
                indices.append((idx, it))

        for idx, it in indices:
            old_path = self.image_paths[idx]
            directory, base = os.path.split(old_path)

            new_base = f"{prefix}{base}{suffix}"
            new_path = os.path.join(directory, new_base)

            if new_path == old_path:
                continue

            if os.path.exists(new_path):
                failures.append((old_path, self.tr("target already exists")))
                continue

            try:
                os.rename(old_path, new_path)
            except Exception as e:
                failures.append((old_path, str(e)))
                continue

            # Update internal lists
            self.image_paths[idx] = new_path
            self.loaded_images[idx]["file_path"] = new_path

            # Update leaf item
            flagged = bool(self.loaded_images[idx].get("flagged", False))
            self._set_leaf_display(it, base_name=new_base, flagged=flagged, full_path=new_path)

            renamed += 1

        # Rebuild so group headers + natural order stay correct
        self._after_list_changed()
        self._sync_metrics_flags()

        msg = self.tr("Batch renamed {0} file{1}.").format(renamed, "s" if renamed != 1 else "")
        if failures:
            msg += self.tr("\n\n{0} file(s) failed:").format(len(failures))
            for old, err in failures[:10]:
                msg += f"\n• {os.path.basename(old)} – {err}"
        QMessageBox.information(self, self.tr("Batch Rename"), msg)


    def batch_delete_flagged_images(self):
        """Delete all flagged images."""
        flagged_images = [img for img in self.loaded_images if img['flagged']]
        
        if not flagged_images:
            QMessageBox.information(self, self.tr("No Flagged Images"), self.tr("There are no flagged images to delete."))
            return

        confirmation = QMessageBox.question(
            self,
            self.tr("Confirm Batch Deletion"),
            self.tr("Are you sure you want to permanently delete {0} flagged images? This action is irreversible.").format(len(flagged_images)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirmation == QMessageBox.StandardButton.Yes:
            removed_indices = []
            # snapshot the indices before mutation
            for img in flagged_images:
                try:
                    removed_indices.append(self.image_paths.index(img['file_path']))
                except ValueError:
                    pass

            # perform deletions
            for img in flagged_images:
                file_path = img['file_path']
                try:
                    os.remove(file_path)
                except Exception as e:
                    ...
                # remove from structures
                if file_path in self.image_paths:
                    self.image_paths.remove(file_path)
                if img in self.loaded_images:
                    self.loaded_images.remove(img)
                self.remove_item_from_tree(file_path)

            QMessageBox.information(self, self.tr("Batch Deletion"), self.tr("Deleted {0} flagged images.").format(len(removed_indices)))

            # 🔁 refresh tree + metrics (no recompute)
            self._after_list_changed(removed_indices)

    def batch_move_flagged_images(self):
        """Move all flagged images to a selected directory AND remove them from the blink list."""
        flagged_indices = [i for i, e in enumerate(self.loaded_images) if e.get("flagged", False)]
        if not flagged_indices:
            QMessageBox.information(self, self.tr("No Flagged Images"), self.tr("There are no flagged images to move."))
            return

        destination_dir = QFileDialog.getExistingDirectory(self, self.tr("Select Destination Folder"), "")
        if not destination_dir:
            return

        failures = []

        # Move first (use current paths from indices)
        for i in flagged_indices:
            src_path = self.image_paths[i]
            dest_path = os.path.join(destination_dir, os.path.basename(src_path))
            try:
                os.rename(src_path, dest_path)
            except Exception as e:
                failures.append((src_path, str(e)))

        # Remove from lists ONLY if move succeeded
        # Build a set of indices to remove: those that did NOT fail
        failed_src = {p for p, _ in failures}
        removed_indices = [i for i in flagged_indices if self.image_paths[i] not in failed_src]

        removed_indices = sorted(set(removed_indices), reverse=True)
        for idx in removed_indices:
            if 0 <= idx < len(self.image_paths):
                del self.image_paths[idx]
            if 0 <= idx < len(self.loaded_images):
                del self.loaded_images[idx]

        if removed_indices:
            self._after_list_changed(removed_indices)

        if failures:
            msg = self.tr("Moved {0} flagged file(s). {1} failed:").format(len(removed_indices), len(failures))
            for p, err in failures[:10]:
                msg += f"\n• {os.path.basename(p)} – {err}"
            QMessageBox.warning(self, self.tr("Batch Move"), msg)
        else:
            QMessageBox.information(self, self.tr("Batch Move"), self.tr("Moved and removed {0} flagged image(s).").format(len(removed_indices)))


    def move_items(self):
        """Move selected leaf images to a selected directory AND remove them from the blink list."""
        selected_items = [it for it in self.fileTree.selectedItems() if it and it.childCount() == 0]
        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No individual image items selected for moving."))
            return

        new_dir = QFileDialog.getExistingDirectory(self, self.tr("Select Destination Folder"), "")
        if not new_dir:
            return

        removed_indices = []
        failures = []

        # Collect (idx, old_path, item) first to avoid index drift
        triplets = []
        for it in selected_items:
            p = self._leaf_path(it)
            if not p:
                continue
            try:
                idx = self.image_paths.index(p)
            except ValueError:
                continue
            triplets.append((idx, p, it))

        for idx, old_path, it in triplets:
            base = os.path.basename(old_path)
            new_path = os.path.join(new_dir, base)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                failures.append((old_path, str(e)))
                continue

            removed_indices.append(idx)

            # remove leaf from tree immediately (optional; _after_list_changed will rebuild anyway)
            #parent = it.parent() or self.fileTree.invisibleRootItem()
            #parent.removeChild(it)

        # Purge arrays descending
        removed_indices = sorted(set(removed_indices), reverse=True)
        for idx in removed_indices:
            if 0 <= idx < len(self.image_paths):
                del self.image_paths[idx]
            if 0 <= idx < len(self.loaded_images):
                del self.loaded_images[idx]

        if removed_indices:
            self._after_list_changed(removed_indices)

        if failures:
            msg = self.tr("Moved {0} file(s). {1} failed:").format(len(removed_indices), len(failures))
            for old, err in failures[:10]:
                msg += f"\n• {os.path.basename(old)} – {err}"
            QMessageBox.warning(self, self.tr("Move Selected Items"), msg)
        else:
            QMessageBox.information(self, self.tr("Move Selected Items"), self.tr("Moved and removed {0} item(s).").format(len(removed_indices)))

    def delete_items(self):
        """Delete selected leaf images from disk and remove them from the blink list."""
        selected_items = [it for it in self.fileTree.selectedItems() if it and it.childCount() == 0]
        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No individual image items selected for deletion."))
            return

        reply = QMessageBox.question(
            self,
            self.tr("Confirm Deletion"),
            self.tr("Are you sure you want to permanently delete {0} selected images? This action is irreversible.").format(len(selected_items)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed_indices = []
        failures = []

        # Snapshot first
        triplets = []
        for it in selected_items:
            p = self._leaf_path(it)
            if not p:
                continue
            try:
                idx = self.image_paths.index(p)
            except ValueError:
                continue
            triplets.append((idx, p, it))

        for idx, path, it in triplets:
            try:
                os.remove(path)
            except Exception as e:
                failures.append((path, str(e)))
                continue

            removed_indices.append(idx)

            # remove from tree immediately (optional)
            parent = it.parent() or self.fileTree.invisibleRootItem()
            parent.removeChild(it)

        # Purge arrays descending
        removed_indices = sorted(set(removed_indices), reverse=True)
        for idx in removed_indices:
            if 0 <= idx < len(self.image_paths):
                del self.image_paths[idx]
            if 0 <= idx < len(self.loaded_images):
                del self.loaded_images[idx]

        # Clear preview safely
        self.preview_label.clear()
        self.preview_label.setText(self.tr("No image selected."))
        self.current_pixmap = None

        if removed_indices:
            self._after_list_changed(removed_indices)

        if failures:
            msg = self.tr("Deleted {0} file(s). {1} failed:").format(len(removed_indices), len(failures))
            for p, err in failures[:10]:
                msg += f"\n• {os.path.basename(p)} – {err}"
            QMessageBox.warning(self, self.tr("Delete Selected Items"), msg)
        else:
            QMessageBox.information(self, self.tr("Delete Selected Items"), self.tr("Deleted {0} item(s).").format(len(removed_indices)))


    def eventFilter(self, source, event):
        """Handle mouse events for dragging."""
        if source == self.scroll_area.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                # Start dragging
                self.dragging = True
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.Type.MouseMove and self.dragging:
                # Handle dragging
                delta = event.pos() - self.last_mouse_pos
                self.scroll_area.horizontalScrollBar().setValue(
                    self.scroll_area.horizontalScrollBar().value() - delta.x()
                )
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() - delta.y()
                )
                self.last_mouse_pos = event.pos()
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self.dragging = False
                self._capture_view_center_norm()  # remember where the user panned to
                return True
        return super().eventFilter(source, event)

    def on_selection_changed(self, selected, deselected):
        items = self.fileTree.selectedItems()
        if not items:
            return
        item = items[0]

        # if a group got selected, ignore (or auto-drill to first leaf if you prefer)
        if item.childCount() > 0:
            return

        name = item.text(0).lstrip("⚠️ ").strip()
        if self._last_preview_name == name:
            return  # no-op, same item

        # debounce: only preview the last selection after brief idle
        self._pending_preview_item = item
        self._pending_preview_timer.start()

    def _do_preview_update(self):
        item = self._pending_preview_item
        if not item or item.treeWidget() is None:   # ← item got deleted
            return
        cur = self.fileTree.currentItem()
        if cur is not item:
            return
        name = item.text(0).lstrip("⚠️ ").strip()
        self._last_preview_name = name
        self.on_item_clicked(item, 0)

    def toggle_aggressive(self):
        self.aggressive_stretch_enabled = self.aggressive_button.isChecked()
        cur = self.fileTree.currentItem()
        if cur:
            self._last_preview_name = None  # force reload even if same item
            self.on_item_clicked(cur, 0)

    def convert_to_qimage(self, img_array):
        if img_array.dtype == np.uint8:
            arr8 = img_array
        elif img_array.dtype == np.uint16:
            arr8 = (img_array.astype(np.float32) / 65535.0 * 255.0).clip(0,255).astype(np.uint8)
        else:
            # ✅ display-only normalize floats outside 0..1
            f01 = self._ensure_float01(img_array)
            arr8 = (f01 * 255.0).astype(np.uint8)

        h, w = arr8.shape[:2]
        buffer = arr8.tobytes()

        if arr8.ndim == 3:
            # RGB
            return QImage(buffer, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            # grayscale
            return QImage(buffer, w, h, w, QImage.Format.Format_Grayscale8)

    def _main_window(self):
        w = self
        from PyQt6.QtWidgets import QMainWindow, QApplication
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parentWidget()
        if w is not None:
            return w
        # fallback: scan toplevels
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None

# Import centralized widgets
from setiastro.saspro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox
from setiastro.saspro.widgets.preview_dialogs import ImagePreviewDialog


BlinkComparatorPro = BlinkTab

# ⬇️ paste your SASv2 code here (exactly as you sent), then end with:
class BlinkComparatorPro(BlinkTab):
    """Alias class so the main app can import a SASpro-named tool."""
    pass

