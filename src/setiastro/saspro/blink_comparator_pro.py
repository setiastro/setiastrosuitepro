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
from PyQt6.QtCore import Qt, QTimer, QEvent, QPointF, QRectF, pyqtSignal, QSettings, QPoint
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

            grid.addWidget(pw, idx//2, idx%2)
            self.plots.append(pw)
            self.scats.append(scat)
            self.lines.append(line)

    @staticmethod
    def _compute_one(i_entry):
        idx, entry = i_entry
        img = entry['image_data']

        # normalize to float32 mono [0..1] exactly like live
        data = np.asarray(img)
        if data.ndim == 3:
            data = data.mean(axis=2)
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0
        else:
            data = data.astype(np.float32, copy=False)

        try:
            # --- match old Blink’s SEP pipeline ---
            bkg = sep.Background(data)
            back = bkg.back()
            try:
                gr = float(bkg.globalrms)
            except Exception:
                # some SEP builds only expose per-cell rms map
                gr = float(np.median(np.asarray(bkg.rms(), dtype=np.float32)))

            cat = sep.extract(
                data - back,
                thresh=7.0,
                err=gr,
                minarea=16,
                clean=True,
                deblend_nthresh=32,
            )

            if len(cat) > 0:
                # FWHM via geometric-mean sigma (old Blink)
                sig = np.sqrt(cat['a'] * cat['b']).astype(np.float32, copy=False)
                fwhm = float(np.nanmedian(2.3548 * sig))

                # TRUE eccentricity: e = sqrt(1 - (b/a)^2)  (old Blink)
                # guard against divide-by-zero and NaNs
                a = np.maximum(cat['a'].astype(np.float32, copy=False), 1e-12)
                b = np.clip(cat['b'].astype(np.float32, copy=False), 0.0, None)
                q = np.clip(b / a, 0.0, 1.0)  # b/a
                e_true = np.sqrt(np.maximum(0.0, 1.0 - q * q))
                ecc = float(np.nanmedian(e_true))

                star_cnt = int(len(cat))
            else:
                fwhm, ecc, star_cnt = np.nan, np.nan, 0

        except Exception:
            # same sentinel behavior as before
            fwhm, ecc, star_cnt = 10.0, 1.0, 0

        orig_back = entry.get('orig_background', np.nan)
        return idx, fwhm, ecc, orig_back, star_cnt


    def compute_all_metrics(self, loaded_images):
        """Run SEP over the full list in parallel using threads and cache results."""
        n = len(loaded_images)
        if n == 0:
            # Clear any previous state and bail
            self._orig_images = []
            self.metrics_data = [np.array([])]*4
            self.flags = []
            self._threshold_initialized = [False]*4
            return

        # Heads-up dialog (as you already had)
        settings = QSettings()
        show = settings.value("metrics/showWarning", True, type=bool)
        if show:
            msg = QMessageBox(self)
            msg.setWindowTitle(self.tr("Heads-up"))
            msg.setText(self.tr(
                "This is going to use ALL your CPU cores and the UI may lock up until it finishes.\n\n"
                "Continue?"
            ))
            msg.setStandardButtons(QMessageBox.StandardButton.Yes |
                                QMessageBox.StandardButton.No)
            cb = QCheckBox(self.tr("Don't show again"), msg)
            msg.setCheckBox(cb)
            if msg.exec() != QMessageBox.StandardButton.Yes:
                return
            if cb.isChecked():
                settings.setValue("metrics/showWarning", False)

        # pre-allocate result arrays
        m0 = np.full(n, np.nan, dtype=np.float32)  # FWHM
        m1 = np.full(n, np.nan, dtype=np.float32)  # Eccentricity
        m2 = np.full(n, np.nan, dtype=np.float32)  # Background (cached)
        m3 = np.full(n, np.nan, dtype=np.float32)  # Star count
        flags = [e.get('flagged', False) for e in loaded_images]

        # progress dialog
        prog = QProgressDialog(self.tr("Computing frame metrics…"), self.tr("Cancel"), 0, n, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        prog.show()
        QApplication.processEvents()

        workers = min(os.cpu_count() or 1, 60)
        tasks = [(i, loaded_images[i]) for i in range(n)]
        done = 0  # <-- FIX: initialize before incrementing

        try:
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = {exe.submit(self._compute_one, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    if prog.wasCanceled():
                        break
                    try:
                        idx, fwhm, ecc, orig_back, star_cnt = fut.result()
                    except Exception:
                        # On failure, leave NaNs/sentinels and continue
                        idx, fwhm, ecc, orig_back, star_cnt = futures[fut], np.nan, np.nan, np.nan, 0
                    m0[idx], m1[idx], m2[idx], m3[idx] = fwhm, ecc, orig_back, float(star_cnt)
                    done += 1
                    prog.setValue(done)
                    QApplication.processEvents()
        finally:
            prog.close()

        # stash results
        self._orig_images = loaded_images
        self.metrics_data = [m0, m1, m2, m3]
        self.flags = flags
        self._threshold_initialized = [False]*4


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
            return

        # compute & cache on first call or new image list
        if self._orig_images is not loaded_images or self.metrics_data is None:
            self.compute_all_metrics(loaded_images)

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
        """
        if not removed:
            return
        removed = sorted(set(int(i) for i in removed))

        # 1) shrink cached arrays in the panel
        self.metrics_panel.remove_frames(removed)

        # 2) update our “master” list and ordering (object identity unchanged)
        #    (BlinkTab will already have mutated the underlying list for us)
        self._order_all = self._reindex_list_after_remove(self._order_all, removed)
        self._current_indices = self._reindex_list_after_remove(self._current_indices, removed)

        # 3) rebuild group list (filters may have disappeared)
        self._rebuild_groups_from_images()

        # 4) replot current group with updated order
        indices = self._current_indices if self._current_indices is not None else self._order_all
        self.metrics_panel.plot(self._all_images, indices=indices)

        # 5) recolor & status
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
                disp8 = stored
            elif stored.dtype == np.uint16:
                disp8 = (stored >> 8).astype(np.uint8)
            else:
                disp8 = (np.clip(stored, 0.0, 1.0) * 255.0).astype(np.uint8)
            return disp8

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
        # read from custom spin if present
        fps = float(self.speed_spin.value) if hasattr(self, "speed_spin") else float(getattr(self, "play_fps", 1.0))
        fps = max(0.1, min(10.0, fps))
        self.play_fps = fps
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

    def toggle_aggressive(self):
        self.aggressive_stretch_enabled = self.aggressive_button.isChecked()
        # force a redisplay of the current image
        cur = self.fileTree.currentItem()
        if cur:
            self.on_item_clicked(cur, 0)

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
                obj_item.addChild(fil_item)
                fil_item.setExpanded(True)
                for exp in sorted(by_object[obj][fil], key=lambda e: str(e).lower()):
                    exp_item = QTreeWidgetItem([self.tr("Exposure: {0}").format(exp)])
                    fil_item.addChild(exp_item)
                    exp_item.setExpanded(True)
                    for p in by_object[obj][fil][exp]:
                        leaf = QTreeWidgetItem([os.path.basename(p)])
                        leaf.setData(0, Qt.ItemDataRole.UserRole, p)
                        exp_item.addChild(leaf)

        # 🔹 NEW: re-apply flagged styling
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
        """Call after you mutate image_paths/loaded_images. Keeps UI + metrics in sync w/o recompute."""
        # 1) rebuild the tree (groups collapse if empty)
        self._rebuild_tree_from_loaded()
        self.imagesChanged.emit(len(self.loaded_images))

        # 2) refresh metrics (if open) WITHOUT recomputing SEP
        if self.metrics_window and self.metrics_window.isVisible():
            if removed_indices:
                # drop points and reindex
                self.metrics_window._all_images = self.loaded_images
                self.metrics_window.remove_indices(list(removed_indices))
            else:
                # just order changed or paths changed -> replot current group
                self.metrics_window.update_metrics(
                    self.loaded_images,
                    order=self._tree_order_indices()
                )

    def get_tree_item_for_index(self, idx):
        target = os.path.basename(self.image_paths[idx])
        for item in self.get_all_leaf_items():
            if item.text(0).lstrip("⚠️ ") == target:
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
            raise ValueError(self.tr("Empty image"))

        # 2) optional debayer
        if is_mono:
            # adjust this call to match your debayer signature
            image = BlinkTab.debayer_image(image, file_path, header)

        # ✅ NEW: force 0..1 range BEFORE SEP + stretch
        image = BlinkTab._ensure_float01(image)

        # 3) SEP background on mono float32
        data = np.asarray(image, dtype=np.float32, order='C')
        if data.ndim == 3:
            data = data.mean(axis=2)
        bkg = sep.Background(data)
        global_back = bkg.globalback

        # 4) stretch
        target_med = 0.25
        if image.ndim == 2:
            stretched = stretch_mono_image(image, target_med)
        else:
            stretched = stretch_color_image(image, target_med, linked=False)

        # 5) cast to target_dtype
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
        file_name = item.text(0).lstrip("⚠️ ")
        file_path = next((p for p in self.image_paths if os.path.basename(p) == file_name), None)
        if file_path is None:
            return

        idx = self.image_paths.index(file_path)
        entry = self.loaded_images[idx]
        entry['flagged'] = not entry['flagged']

        RED = Qt.GlobalColor.red
        palette = self.fileTree.palette()
        normal_color = palette.color(QPalette.ColorRole.WindowText)

        if entry['flagged']:
            item.setText(0, f"⚠️ {file_name}")
            item.setForeground(0, QBrush(RED))
        else:
            item.setText(0, file_name)
            item.setForeground(0, QBrush(normal_color))

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

        name = item.text(0).lstrip("⚠️ ").strip()
        file_path = next((p for p in self.image_paths if os.path.basename(p) == name), None)
        if not file_path:
            return

        self._capture_view_center_norm()

        idx = self.image_paths.index(file_path)
        entry = self.loaded_images[idx]
        stored = entry['image_data']  # already stretched & clipped at load time

        # --- Fast path: just display what we cached in RAM ---
        if not self.aggressive_stretch_enabled:
            # Convert to 8-bit only if needed (no additional stretch)
            if stored.dtype == np.uint8:
                disp8 = stored
            elif stored.dtype == np.uint16:
                disp8 = (stored >> 8).astype(np.uint8)   # ~ /257, quick & vectorized
            else:  # float32 in [0..1]
                disp8 = (np.clip(stored, 0.0, 1.0) * 255.0).astype(np.uint8)

        else:
            # Aggressive mode: compute only here (from float01)
            base01 = self._as_float01(stored)
            # Siril-style autostretch
            if base01.ndim == 2:
                st = siril_style_autostretch(base01, sigma=self.current_sigma)
                disp01 = self._as_float01(st)   # <-- IMPORTANT: handles 0..255 or 0..1 correctly
            else:
                base01 = self._as_float01(stored)

                if base01.ndim == 2:
                    disp01 = self._aggressive_display_boost(base01, strength=self.current_sigma)
                else:
                    lum = base01.mean(axis=2).astype(np.float32)
                    lum_boost = self._aggressive_display_boost(lum, strength=self.current_sigma)
                    gain = lum_boost / (lum + 1e-6)
                    disp01 = np.clip(base01 * gain[..., None], 0.0, 1.0)

                disp8 = (disp01 * 255.0).astype(np.uint8)


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


    def push_to_docs(self, item):
        # Resolve file + entry
        file_name = item.text(0).lstrip("⚠️ ")
        file_path = next((p for p in self.image_paths if os.path.basename(p) == file_name), None)
        if not file_path:
            return
        idx   = self.image_paths.index(file_path)
        entry = self.loaded_images[idx]

        # Find main window + doc manager
        mw = self._main_window()
        dm = self.doc_manager or (getattr(mw, "docman", None) if mw else None)
        if not mw or not dm:
            QMessageBox.warning(self, self.tr("Document Manager"), self.tr("Main window or DocManager not available."))
            return

        # Prepare image + metadata for a real document
        np_image_f01 = self._as_float01(entry['image_data'])  # ensure float32 [0..1]
        metadata  = {
            'file_path': file_path,
            'original_header': entry.get('header', {}),
            'bit_depth': entry.get('bit_depth'),
            'is_mono': entry.get('is_mono'),
            'source': 'BlinkComparatorPro',
        }
        title = os.path.basename(file_path)

        # Create the document using whatever API your DocManager has
        doc = None
        try:
            if hasattr(dm, "open_array"):
                doc = dm.open_array(np_image_f01, metadata=metadata, title=title)
            elif hasattr(dm, "open_numpy"):
                doc = dm.open_numpy(np_image_f01, metadata=metadata, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=np_image_f01, metadata=metadata, name=title)
            else:
                raise AttributeError(self.tr("DocManager lacks open_array/open_numpy/create_document"))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Doc Manager"), self.tr("Failed to create document:\n{0}").format(e))
            return

        if doc is None:
            QMessageBox.critical(self, self.tr("Doc Manager"), self.tr("DocManager returned no document."))
            return

        # SHOW it: ask the main window to spawn an MDI subwindow
        try:
            mw._spawn_subwindow_for(doc)
            if hasattr(mw, "_log"):
                mw._log(f"Blink → opened '{title}' as new document")
        except Exception as e:
            QMessageBox.critical(self, self.tr("UI"), self.tr("Failed to open subwindow:\n{0}").format(e))


    # optional shim to keep any old calls working
    def push_image_to_manager(self, item):
        self.push_to_docs(item)



    def rename_item(self, item):
        """Allow the user to rename the selected image."""
        current_name = item.text(0).lstrip("⚠️ ")
        new_name, ok = QInputDialog.getText(self, self.tr("Rename Image"), self.tr("Enter new name:"), text=current_name)

        if ok and new_name:
            file_path = next((path for path in self.image_paths if os.path.basename(path) == current_name), None)
            if file_path:
                # Get the new file path with the new name
                new_file_path = os.path.join(os.path.dirname(file_path), new_name)

                try:
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"File renamed from {current_name} to {new_name}")
                    
                    # Update the image paths and tree view
                    self.image_paths[self.image_paths.index(file_path)] = new_file_path
                    item.setText(0, new_name)
                except Exception as e:
                    QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to rename the file: {0}").format(e))

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
        """Batch rename selected items by adding a prefix or suffix."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No items selected for renaming."))
            return

        # Create a custom dialog for entering the prefix and suffix
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr("Batch Rename"))
        dialog_layout = QVBoxLayout(dialog)

        instruction_label = QLabel(self.tr("Enter a prefix or suffix to rename selected files:"))
        dialog_layout.addWidget(instruction_label)

        # Create fields for prefix and suffix
        form_layout = QHBoxLayout()

        prefix_field = QLineEdit(dialog)
        prefix_field.setPlaceholderText(self.tr("Prefix"))
        form_layout.addWidget(prefix_field)

        current_filename_label = QLabel("currentfilename", dialog)
        form_layout.addWidget(current_filename_label)

        suffix_field = QLineEdit(dialog)
        suffix_field.setPlaceholderText(self.tr("Suffix"))
        form_layout.addWidget(suffix_field)

        dialog_layout.addLayout(form_layout)

        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton(self.tr("OK"), dialog)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton(self.tr("Cancel"), dialog)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)

        dialog_layout.addLayout(button_layout)

        # Show the dialog and handle user input
        if dialog.exec() == QDialog.DialogCode.Accepted:
            prefix = prefix_field.text().strip()
            suffix = suffix_field.text().strip()

            # Rename each selected file
            for item in selected_items:
                current_name = item.text(0)
                file_path = next((path for path in self.image_paths if os.path.basename(path) == current_name), None)

                if file_path:
                    # Construct the new filename
                    directory = os.path.dirname(file_path)
                    new_name = f"{prefix}{current_name}{suffix}"
                    new_file_path = os.path.join(directory, new_name)

                    try:
                        # Rename the file
                        os.rename(file_path, new_file_path)
                        print(f"File renamed from {file_path} to {new_file_path}")

                        # Update the paths and tree view
                        self.image_paths[self.image_paths.index(file_path)] = new_file_path
                        item.setText(0, new_name)

                    except Exception as e:
                        print(f"Failed to rename {file_path}: {e}")
                        QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to rename the file: {0}").format(e))

            print(f"Batch renamed {len(selected_items)} items.")

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
        """Move all flagged images to a selected directory."""
        flagged_images = [img for img in self.loaded_images if img['flagged']]
        
        if not flagged_images:
            QMessageBox.information(self, self.tr("No Flagged Images"), self.tr("There are no flagged images to move."))
            return

        # Select destination directory
        destination_dir = QFileDialog.getExistingDirectory(self, self.tr("Select Destination Folder"), "")
        if not destination_dir:
            return  # User canceled

        for img in flagged_images:
            src_path = img['file_path']
            file_name = os.path.basename(src_path)
            dest_path = os.path.join(destination_dir, file_name)

            try:
                os.rename(src_path, dest_path)
                print(f"Moved flagged image from {src_path} to {dest_path}")
            except Exception as e:
                print(f"Failed to move {src_path}: {e}")
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to move {0}: {1}").format(src_path, e))
                continue

            # Update data structures
            self.image_paths.remove(src_path)
            self.image_paths.append(dest_path)
            img['file_path'] = dest_path
            img['flagged'] = False  # Reset flag if desired

            # Update tree view
            self.remove_item_from_tree(src_path)
            self.add_item_to_tree(dest_path)

        QMessageBox.information(self, self.tr("Batch Move"), self.tr("Moved {0} flagged images.").format(len(flagged_images)))
        self._after_list_changed(removed_indices=None)

    def move_items(self):
        """Move selected images *and* remove them from the tree+metrics."""
        selected_items = self.fileTree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No items selected for moving."))
            return

        # Ask where to move
        new_dir = QFileDialog.getExistingDirectory(self,
                                                self.tr("Select Destination Folder"),
                                                "")
        if not new_dir:
            return

        # Keep track of which on‐disk paths we actually moved
        moved_old_paths = []
        removed_indices = []

        for item in selected_items:
            name = item.text(0).lstrip("⚠️ ")
            old_path = next((p for p in self.image_paths 
                            if os.path.basename(p) == name), None)
            if not old_path:
                continue
            removed_indices.append(self.image_paths.index(old_path)) 

            new_path = os.path.join(new_dir, name)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to move {0}: {1}").format(old_path, e))
                continue

            moved_old_paths.append(old_path)

            # 1) Remove the leaf from the tree
            parent = item.parent() or self.fileTree.invisibleRootItem()
            parent.removeChild(item)

        # 2) Purge them from your internal lists
        for idx in sorted(removed_indices, reverse=True):
            del self.image_paths[idx]
            del self.loaded_images[idx]

        self._after_list_changed(removed_indices)
        print(f"Moved and removed {len(removed_indices)} items.")



    def delete_items(self):
        """Delete the selected items from the tree, the loaded images list, and the file system."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, self.tr("Warning"), self.tr("No items selected for deletion."))
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            self.tr('Confirm Deletion'),
            self.tr("Are you sure you want to permanently delete {0} selected images? This action is irreversible.").format(len(selected_items)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        removed_indices = []
        if reply == QMessageBox.StandardButton.Yes:
            for item in selected_items:
                file_name = item.text(0).lstrip("⚠️ ")
                file_path = next((path for path in self.image_paths if os.path.basename(path) == file_name), None)
                if file_path:
                    try:
                        idx = self.image_paths.index(file_path)
                        removed_indices.append(idx)  # collect BEFORE mutation
                        ...
                        os.remove(file_path)
                    except Exception as e:
                        ...
            # Remove from widgets
            for item in selected_items:
                parent = item.parent() or self.fileTree.invisibleRootItem()
                parent.removeChild(item)

            # Purge arrays (descending order)
            for idx in sorted(removed_indices, reverse=True):
                del self.image_paths[idx]
                del self.loaded_images[idx]

            # Clear preview
            self.preview_label.clear()
            self.preview_label.setText(self.tr('No image selected.'))
            self.current_image = None

            # 🔁 refresh tree + metrics (no recompute)
            self._after_list_changed(removed_indices)

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
        """Convert numpy image array to QImage."""
        # 1) Bring everything into a uint8 (0–255) array
        if img_array.dtype == np.uint8:
            arr8 = img_array
        elif img_array.dtype == np.uint16:
            # downscale 16-bit → 8-bit
            arr8 = (img_array.astype(np.float32) / 65535.0 * 255.0).clip(0,255).astype(np.uint8)
        else:
            # assume float in [0..1]
            arr8 = (img_array.clip(0.0, 1.0) * 255.0).astype(np.uint8)

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

