# pro/blink_comparator_pro.py
from __future__ import annotations

# ⬇️ keep your existing imports used by the code you pasted
import os, re, time, psutil, numpy as np
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
    QMdiArea
)

# 3rd-party (your code already expects these)
import cv2, sep, pyqtgraph as pg

from legacy.image_manager import load_image

from imageops.stretch import stretch_color_image, stretch_mono_image, siril_style_autostretch

from numba_utils import debayer_fits_fast, debayer_raw_fast


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
        titles = ["FWHM (px)", "Eccentricity", "Background", "Star Count"]
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
        """
        Worker: run SEP on one image entry.
        Returns (idx, fwhm, ecc, orig_back, star_count).
        If SEP overflows its internal pixel buffer, we catch it and
        return sentinel “bad” values so the frame will be flagged.
        """
        idx, entry = i_entry

        # rebuild normalized mono data
        img = entry['image_data']
        if img.dtype == np.uint8:
            data = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            data = img.astype(np.float32) / 65535.0
        else:
            data = np.asarray(img, dtype=np.float32)
        if data.ndim == 3:
            data = data.mean(axis=2)

        # detection parameters
        thresh   = 7.0    # σ threshold
        min_area = 16     # at least a 4×4 blob

        try:
            bkg = sep.Background(data)
            back, gb, gr = bkg.back(), bkg.globalback, bkg.globalrms

            cat = sep.extract(
                data - back,
                thresh,
                err=gr,
                minarea=min_area,
                clean=True,
                deblend_nthresh=32
            )

            if len(cat) > 0:
                # σ = geometric mean of the two RMS axes
                sig      = np.sqrt(cat['a'] * cat['b'])
                fwhm     = np.nanmedian(2.3548 * sig)

                # true eccentricity e = sqrt(1 - (b/a)^2)
                ratios   = np.clip(cat['b'] / cat['a'], 0, 1)
                ecc_vals = np.sqrt(1.0 - ratios**2)
                ecc      = np.nanmedian(ecc_vals)

                star_cnt = len(cat)
            else:
                fwhm, ecc, star_cnt = np.nan, np.nan, 0

        except Exception as e:
            # catch SEP overflow (or any other) and mark as “bad” frame
            # you can even check `if "internal pixel buffer full" in str(e):`
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
            msg.setWindowTitle("Heads-up")
            msg.setText(
                "This is going to use ALL your CPU cores and the UI may lock up until it finishes.\n\n"
                "Continue?"
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes |
                                QMessageBox.StandardButton.No)
            cb = QCheckBox("Don't show again", msg)
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
        prog = QProgressDialog("Computing frame metrics…", "Cancel", 0, n, self)
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


    def _on_point_click(self, metric_idx, points):
        for pt in points:
            frame_idx = int(round(pt.pos().x()))
            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                entry  = self._orig_images[frame_idx]
                img    = entry['image_data']
                is_mono= entry.get('is_mono', False)
                dlg = ImagePreviewDialog(img, is_mono)
                dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  
                dlg.show()
                self._open_previews.append(dlg)  # <-- hold a reference

                # optionally prune closed ones:
                dlg.destroyed.connect(lambda _=None, d=dlg: 
                      self._open_previews.remove(d)
                      if d in self._open_previews else None)
            else:
                self.pointClicked.emit(metric_idx, frame_idx)

    def _on_line_move(self, metric_idx, line):
        self.thresholdChanged.emit(metric_idx, line.value())

class MetricsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self._thresholds_per_group: dict[str, List[float|None]] = {}
        self.setWindowTitle("Frame Metrics")
        self.resize(800, 600)

        vbox = QVBoxLayout(self)

        # ← **new** instructions label
        instr = QLabel(
            "Instructions:\n"
            " • Use the filter dropdown to restrict by FILTER.\n"
            " • Click a dot to flag/unflag a frame.\n"
            " • Shift-click a dot to preview the image.\n"
            " • Drag the red lines to set thresholds.",
            self
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #ccc; font-size: 12px;")
        vbox.addWidget(instr)

        # → filter selector
        self.group_combo = QComboBox(self)
        self.group_combo.addItem("All")
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
        """Recompute and show: Flagged Items X / Y (Z%)."""
        flags = getattr(self.metrics_panel, 'flags', []) or []
        # which subset are we in?
        idxs = self._current_indices if self._current_indices is not None else range(len(flags))
        total = len(idxs)
        flagged = sum(flags[i] for i in idxs)
        pct = (flagged/total*100) if total else 0.0
        self.status_label.setText(f"Flagged Items {flagged}/{total}  ({pct:.1f}%)")

    def set_images(self, loaded_images):
        """Initialize with a brand-new set of images."""
        self._all_images = loaded_images

        # ─── rebuild the combo-list of FILTER groups ─────────────
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItem("All")
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
        self._current_indices = None
        self._apply_thresholds("All")
        self.metrics_panel.plot(self._all_images, indices=None)

        # ─── update the flagged-items status label ───────────────
        self._update_status()


    def _on_group_change(self, name: str):
        """Re-plot for the selected FILTER group."""
        if name == "All":
            self._current_indices = None
        else:
            # collect indices matching this filter
            self._current_indices = [
                i for i, e in enumerate(self._all_images)
                if e.get('header', {}).get('FILTER', 'Unknown') == name
            ]

        # apply saved thresholds for this group
        self._apply_thresholds(name)
        # re-draw
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

    def update_metrics(self, loaded_images):
        """
        Called whenever BlinkTab.loadImages or clearImages fires.
        If it's a new list, re-init; otherwise just re-plot current group.
        """
        if loaded_images is not self._all_images:
            self.set_images(loaded_images)
        else:
            # same list, just redraw current selection
            self._on_group_change(self.group_combo.currentText())



class BlinkTab(QWidget):
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
        instruction_text = "Press 'F' to flag/unflag an image.\nRight-click on an image for more options."
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
        self.fileButton = QPushButton('Select Images', self)
        self.fileButton.clicked.connect(self.openFileDialog)
        button_layout.addWidget(self.fileButton)

        # "Select Directory" Button
        self.dirButton = QPushButton('Select Directory', self)
        self.dirButton.clicked.connect(self.openDirectoryDialog)
        button_layout.addWidget(self.dirButton)

        self.addButton = QPushButton("Add Additional", self)
        self.addButton.clicked.connect(self.addAdditionalImages)
        button_layout.addWidget(self.addButton)

        left_layout.addLayout(button_layout)

        self.metrics_button = QPushButton("Show Metrics", self)
        self.metrics_button.clicked.connect(self.show_metrics)
        left_layout.addWidget(self.metrics_button)



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

        speed_label = QLabel("Speed:", self)
        speed_layout.addWidget(speed_label)

        # Slider maps 1..100 -> 0.1..10.0 fps
        self.speed_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(int(round(self.play_fps * 10)))  # play_fps is float
        self.speed_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.speed_slider.setToolTip("Playback speed (0.1–10.0 fps)")
        speed_layout.addWidget(self.speed_slider, 1)

        # Custom float spin (your class)
        self.speed_spin = CustomDoubleSpinBox(
            minimum=0.1, maximum=10.0, initial=self.play_fps, step=0.1, suffix=" fps", parent=self
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

        # Tree view for file names
        self.fileTree = QTreeWidget(self)
        self.fileTree.setColumnCount(1)
        self.fileTree.setHeaderLabels(["Image Files"])
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
        self.clearFlagsButton = QPushButton('Clear Flags', self)
        self.clearFlagsButton.clicked.connect(self.clearFlags)
        left_layout.addWidget(self.clearFlagsButton)

        # "Clear Images" Button
        self.clearButton = QPushButton('Clear Images', self)
        self.clearButton.clicked.connect(self.clearImages)
        left_layout.addWidget(self.clearButton)

        # Add progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        # Add loading message label
        self.loading_label = QLabel("Load images...", self)
        left_layout.addWidget(self.loading_label)

        # Set the layout for the left widget
        left_widget.setLayout(left_layout)

        # Add the left widget to the splitter
        splitter.addWidget(left_widget)

        # Right Column for Image Preview
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        # Zoom controls: Add Zoom In and Zoom Out buttons
        zoom_controls_layout = QHBoxLayout()

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        zoom_controls_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        zoom_controls_layout.addWidget(self.zoom_out_button)

        self.fit_to_preview_button = QPushButton("Fit to Preview")
        self.fit_to_preview_button.clicked.connect(self.fit_to_preview)
        zoom_controls_layout.addWidget(self.fit_to_preview_button)

        self.aggressive_button = QPushButton("Aggressive Stretch", self)
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


    def _apply_playback_interval(self, *_):
        # read from custom spin if present
        fps = float(self.speed_spin.value()) if hasattr(self, "speed_spin") else float(getattr(self, "play_fps", 1.0))
        fps = max(0.1, min(10.0, fps))
        self.play_fps = fps
        self.playback_timer.setInterval(int(round(1000.0 / fps)))  # 0.1 fps -> 10000 ms

    def _on_current_item_changed_safe(self, current, previous):
        if not current:
            return

        # If a mouse button is currently pressed, don't scroll now—defer a bit
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            QTimer.singleShot(120, lambda: self._center_if_no_mouse(current))
            return

        # Let selection settle, then gently ensure it's visible (no jump)
        QTimer.singleShot(0, lambda: self.fileTree.scrollToItem(
            current, QAbstractItemView.ScrollHint.EnsureVisible
        ))

    def _center_if_no_mouse(self, item):
        # Only center if the mouse is up AND the item is still current
        if QApplication.mouseButtons() == Qt.MouseButton.NoButton and item is self.fileTree.currentItem():
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

    def addAdditionalImages(self):
        """Let the user pick more images to append to the blink list."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Additional Images",
            "",
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)"
        )
        # filter out duplicates
        new_paths = [p for p in file_paths if p not in self.image_paths]
        if not new_paths:
            QMessageBox.information(self, "No New Images", "No new images selected or already loaded.")
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
        self.loading_label.setText(f"Loaded {len(self.loaded_images)} images.")
        if self.metrics_window and self.metrics_window.isVisible():
            self.metrics_window.update_metrics(self.loaded_images)

    def show_metrics(self):
        if self.metrics_window is None:
            self.metrics_window = MetricsWindow()
            mp = self.metrics_window.metrics_panel
            mp.pointClicked.connect(self.on_metrics_point)
            mp.thresholdChanged.connect(self.on_threshold_change)

        # ← here ←
        self.metrics_window.set_images(self.loaded_images)
        panel = self.metrics_window.metrics_panel
        self.thresholds_by_group["All"] = [ line.value() for line in panel.lines ]
        self.metrics_window.show()
        self.metrics_window.raise_()

    def on_metrics_point(self, metric_idx, frame_idx):
        # Toggle the flagged state on the image…
        item = self.get_tree_item_for_index(frame_idx)
        if not item:
            return
        self._toggle_flag_on_item(item)

        # Now update the panel’s flags and refresh
        panel = self.metrics_window.metrics_panel
        panel.flags = [entry['flagged'] for entry in self.loaded_images]
        panel._refresh_scatter_colors()
        self.metrics_window._update_status()

    def _as_float01(self, arr):
        """Fast conversion to float [0..1] without any new stretching logic."""
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        # assume float already in [0..1]
        return np.asarray(arr, dtype=np.float32)

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
        if group == "All":
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
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", "")
        if directory:
            # Supported image extensions
            supported_extensions = (
                '.png', '.tif', '.tiff', '.fits', '.fit',
                '.xisf', '.cr2', '.nef', '.arw', '.dng',
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
                QMessageBox.information(self, "No Images Found", "No supported image files were found in the selected directory.")


    def clearImages(self):
        """Clear all loaded images and reset the tree view."""
        confirmation = QMessageBox.question(
            self,
            "Clear All Images",
            "Are you sure you want to clear all loaded images?",
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
            self.preview_label.setText('No image selected.')
            self.current_pixmap = None
            self.progress_bar.setValue(0)
            self.loading_label.setText("Loading images...")

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
            raise ValueError("Empty image")

        # 2) optional debayer
        if is_mono:
            # adjust this call to match your debayer signature
            image = BlinkTab.debayer_image(image, file_path, header)

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
        elif file_path.lower().endswith(('.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')):
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
                        exp_item.addChild(leaf)

        self.loading_label.setText(f"Loaded {len(self.loaded_images)} images.")
        self.progress_bar.setValue(100)
        if self.metrics_window and self.metrics_window.isVisible():
            self.metrics_window.update_metrics(self.loaded_images)

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


    def _toggle_flag_on_item(self, item: QTreeWidgetItem):
        """Toggle the flagged state on this tree item and its loaded_images entry."""
        file_name = item.text(0).lstrip("⚠️ ")
        # find the matching image entry
        file_path = next((p for p in self.image_paths if os.path.basename(p) == file_name), None)
        if file_path is None:
            return

        idx = self.image_paths.index(file_path)
        entry = self.loaded_images[idx]
        entry['flagged'] = not entry['flagged']

        # update the tree view
        RED = Qt.GlobalColor.red
        palette = self.fileTree.palette()
        normal_color = palette.color(QPalette.ColorRole.WindowText)

        if entry['flagged']:
            item.setText(0, f"⚠️ {file_name}")
            item.setForeground(0, QBrush(RED))
        else:
            item.setText(0, file_name)
            item.setForeground(0, QBrush(normal_color))

    def flag_current_image(self):
        """Called by the F-key: toggle flag on the currently selected item."""
        item = self.fileTree.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "No image is currently selected to flag.")
            return
        self._toggle_flag_on_item(item)
        self.next_item()  # Move to the next item after flagging


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
            QMessageBox.information(self, "No Images", "Load some images first.")
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
            "Open Images",
            "",
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.cr2 *.cr3 *.nef *.arw *.dng *.orf *.rw2 *.pef);;All Files (*)"
        )
        
        # Filter out already loaded images to prevent duplicates
        new_file_paths = [path for path in file_paths if path not in self.image_paths]

        if new_file_paths:
            self.loadImages(new_file_paths)
        else:
            QMessageBox.information(self, "No New Images", "No new images were selected or all selected images are already loaded.")


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
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

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
        exposure_item.addChild(item)



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
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")

    

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
            if base01.ndim == 2:
                disp = siril_style_autostretch(base01, sigma=self.current_sigma)
            else:
                # apply per-channel or linked, your choice; keeping it simple here
                disp = siril_style_autostretch(base01, sigma=self.current_sigma)
            disp8 = (np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8)

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
            QMessageBox.warning(self, "Warning", "No image loaded. Cannot fit to preview.")

    def _is_leaf(self, item: Optional[QTreeWidgetItem]) -> bool:
        return bool(item and item.childCount() == 0)

    def on_right_click(self, pos):
        item = self.fileTree.itemAt(pos)
        if not self._is_leaf(item):
            # Optional: expand/collapse-only menu, or just ignore
            return

        menu = QMenu(self)

        push_action = QAction("Open in Document Window", self)
        push_action.triggered.connect(lambda: self.push_to_docs(item))
        menu.addAction(push_action)

        rename_action = QAction("Rename", self)
        rename_action.triggered.connect(lambda: self.rename_item(item))
        menu.addAction(rename_action)

        move_action = QAction("Move Selected Items", self)
        move_action.triggered.connect(self.move_items)
        menu.addAction(move_action)

        delete_action = QAction("Delete Selected Items", self)
        delete_action.triggered.connect(self.delete_items)
        menu.addAction(delete_action)

        menu.addSeparator()
        batch_delete_action = QAction("Delete All Flagged Images", self)
        batch_delete_action.triggered.connect(self.batch_delete_flagged_images)
        menu.addAction(batch_delete_action)

        batch_move_action = QAction("Move All Flagged Images", self)
        batch_move_action.triggered.connect(self.batch_move_flagged_images)
        menu.addAction(batch_move_action)

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
            QMessageBox.warning(self, "Document Manager", "Main window or DocManager not available.")
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
                raise AttributeError("DocManager lacks open_array/open_numpy/create_document")
        except Exception as e:
            QMessageBox.critical(self, "Doc Manager", f"Failed to create document:\n{e}")
            return

        if doc is None:
            QMessageBox.critical(self, "Doc Manager", "DocManager returned no document.")
            return

        # SHOW it: ask the main window to spawn an MDI subwindow
        try:
            mw._spawn_subwindow_for(doc)
            if hasattr(mw, "_log"):
                mw._log(f"Blink → opened '{title}' as new document")
        except Exception as e:
            QMessageBox.critical(self, "UI", f"Failed to open subwindow:\n{e}")


    # optional shim to keep any old calls working
    def push_image_to_manager(self, item):
        self.push_to_docs(item)



    def rename_item(self, item):
        """Allow the user to rename the selected image."""
        current_name = item.text(0).lstrip("⚠️ ")
        new_name, ok = QInputDialog.getText(self, "Rename Image", "Enter new name:", text=current_name)

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
                    QMessageBox.critical(self, "Error", f"Failed to rename the file: {e}")

    def batch_rename_items(self):
        """Batch rename selected items by adding a prefix or suffix."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for renaming.")
            return

        # Create a custom dialog for entering the prefix and suffix
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Rename")
        dialog_layout = QVBoxLayout(dialog)

        instruction_label = QLabel("Enter a prefix or suffix to rename selected files:")
        dialog_layout.addWidget(instruction_label)

        # Create fields for prefix and suffix
        form_layout = QHBoxLayout()

        prefix_field = QLineEdit(dialog)
        prefix_field.setPlaceholderText("Prefix")
        form_layout.addWidget(prefix_field)

        current_filename_label = QLabel("currentfilename", dialog)
        form_layout.addWidget(current_filename_label)

        suffix_field = QLineEdit(dialog)
        suffix_field.setPlaceholderText("Suffix")
        form_layout.addWidget(suffix_field)

        dialog_layout.addLayout(form_layout)

        # Add OK and Cancel buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK", dialog)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel", dialog)
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
                        QMessageBox.critical(self, "Error", f"Failed to rename the file: {e}")

            print(f"Batch renamed {len(selected_items)} items.")

    def batch_delete_flagged_images(self):
        """Delete all flagged images."""
        flagged_images = [img for img in self.loaded_images if img['flagged']]
        
        if not flagged_images:
            QMessageBox.information(self, "No Flagged Images", "There are no flagged images to delete.")
            return

        confirmation = QMessageBox.question(
            self,
            "Confirm Batch Deletion",
            f"Are you sure you want to permanently delete {len(flagged_images)} flagged images? This action is irreversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirmation == QMessageBox.StandardButton.Yes:
            for img in flagged_images:
                file_path = img['file_path']
                try:
                    os.remove(file_path)
                    print(f"Deleted flagged image: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
                    QMessageBox.critical(self, "Error", f"Failed to delete {file_path}: {e}")

                # Remove from data structures
                self.image_paths.remove(file_path)
                self.loaded_images.remove(img)

                # Remove from tree view
                self.remove_item_from_tree(file_path)

            QMessageBox.information(self, "Batch Deletion", f"Deleted {len(flagged_images)} flagged images.")

    def batch_move_flagged_images(self):
        """Move all flagged images to a selected directory."""
        flagged_images = [img for img in self.loaded_images if img['flagged']]
        
        if not flagged_images:
            QMessageBox.information(self, "No Flagged Images", "There are no flagged images to move.")
            return

        # Select destination directory
        destination_dir = QFileDialog.getExistingDirectory(self, "Select Destination Folder", "")
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
                QMessageBox.critical(self, "Error", f"Failed to move {src_path}: {e}")
                continue

            # Update data structures
            self.image_paths.remove(src_path)
            self.image_paths.append(dest_path)
            img['file_path'] = dest_path
            img['flagged'] = False  # Reset flag if desired

            # Update tree view
            self.remove_item_from_tree(src_path)
            self.add_item_to_tree(dest_path)

        QMessageBox.information(self, "Batch Move", f"Moved {len(flagged_images)} flagged images.")


    def move_items(self):
        """Move selected images *and* remove them from the tree+metrics."""
        selected_items = self.fileTree.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for moving.")
            return

        # Ask where to move
        new_dir = QFileDialog.getExistingDirectory(self,
                                                "Select Destination Folder",
                                                "")
        if not new_dir:
            return

        # Keep track of which on‐disk paths we actually moved
        moved_old_paths = []

        for item in selected_items:
            name = item.text(0).lstrip("⚠️ ")
            old_path = next((p for p in self.image_paths 
                            if os.path.basename(p) == name), None)
            if not old_path:
                continue

            new_path = os.path.join(new_dir, name)
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to move {old_path}: {e}")
                continue

            moved_old_paths.append(old_path)

            # 1) Remove the leaf from the tree
            parent = item.parent() or self.fileTree.invisibleRootItem()
            parent.removeChild(item)

        # 2) Purge them from your internal lists
        for old in moved_old_paths:
            idx = self.image_paths.index(old)
            del self.image_paths[idx]
            del self.loaded_images[idx]

        # 3) Update your “loaded X images” label
        self.loading_label.setText(f"Loaded {len(self.loaded_images)} images.")

        # 4) Tell metrics to re-initialize on the new list
        if self.metrics_window and self.metrics_window.isVisible():
            self.metrics_window.update_metrics(self.loaded_images)

        print(f"Moved and removed {len(moved_old_paths)} items.")




    def delete_items(self):
        """Delete the selected items from the tree, the loaded images list, and the file system."""
        selected_items = self.fileTree.selectedItems()

        if not selected_items:
            QMessageBox.warning(self, "Warning", "No items selected for deletion.")
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            'Confirm Deletion',
            f"Are you sure you want to permanently delete {len(selected_items)} selected images? This action is irreversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for item in selected_items:
                file_name = item.text(0).lstrip("⚠️ ")
                file_path = next((path for path in self.image_paths if os.path.basename(path) == file_name), None)

                if file_path:
                    try:
                        # Remove the image from image_paths
                        if file_path in self.image_paths:
                            self.image_paths.remove(file_path)
                            print(f"Image path {file_path} removed from image_paths.")
                        else:
                            print(f"Image path {file_path} not found in image_paths.")

                        # Remove the corresponding image from loaded_images
                        matching_image_data = next((entry for entry in self.loaded_images if entry['file_path'] == file_path), None)
                        if matching_image_data:
                            self.loaded_images.remove(matching_image_data)
                            print(f"Image {file_name} removed from loaded_images.")
                        else:
                            print(f"Image {file_name} not found in loaded_images.")

                        # Delete the file from the filesystem
                        os.remove(file_path)
                        print(f"File {file_path} deleted successfully.")

                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        QMessageBox.critical(self, "Error", f"Failed to delete the image file: {e}")

            # Remove the selected items from the TreeWidget
            for item in selected_items:
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.fileTree.indexOfTopLevelItem(item)
                    if index != -1:
                        self.fileTree.takeTopLevelItem(index)

            print(f"Deleted {len(selected_items)} items.")

            # Clear the preview if the deleted items include the currently displayed image
            self.preview_label.clear()
            self.preview_label.setText('No image selected.')

            self.current_image = None

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
        if not item:
            return
        # item might have changed selection; ensure it’s still selected/current
        cur = self.fileTree.currentItem()
        if cur is not item:
            return

        name = item.text(0).lstrip("⚠️ ").strip()
        self._last_preview_name = name

        # kick the preview (reuse your existing loader)
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

# NOTE: If you referenced CustomDoubleSpinBox or ImagePreviewDialog,
# replace them with these tiny local shims:

class CustomDoubleSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, minimum=0.0, maximum=10.0, initial=0.0, step=0.1,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._value = initial
        self.suffix = suffix

        # Line edit
        self.lineEdit = QLineEdit(f"{initial:.3f}")
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        validator = QDoubleValidator(self.minimum, self.maximum, 3, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lineEdit.setValidator(validator)
        self.lineEdit.editingFinished.connect(self.onEditingFinished)

        # Up/down buttons
        self.upButton = QToolButton(); self.upButton.setText("▲")
        self.downButton = QToolButton(); self.downButton.setText("▼")
        self.upButton.clicked.connect(self.increaseValue)
        self.downButton.clicked.connect(self.decreaseValue)

        # Buttons layout
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.upButton)
        buttonLayout.addWidget(self.downButton)
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)

        # Optional suffix label
        self.suffixLabel = QLabel(self.suffix) if self.suffix else None
        if self.suffixLabel:
            self.suffixLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Main layout
        mainLayout = QHBoxLayout()
        mainLayout.setSpacing(2)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.lineEdit)
        mainLayout.addLayout(buttonLayout)
        if self.suffixLabel:
            mainLayout.addWidget(self.suffixLabel)
        self.setLayout(mainLayout)

        self.updateButtonStates()

    def updateButtonStates(self):
        self.upButton.setEnabled(self._value < self.maximum)
        self.downButton.setEnabled(self._value > self.minimum)

    def increaseValue(self):
        self.setValue(self._value + self.step)

    def decreaseValue(self):
        self.setValue(self._value - self.step)

    def onEditingFinished(self):
        try:
            new_val = float(self.lineEdit.text())
        except ValueError:
            new_val = self._value
        self.setValue(new_val)

    def setValue(self, val: float):
        if val < self.minimum:
            val = self.minimum
        elif val > self.maximum:
            val = self.maximum
        if abs(val - self._value) > 1e-9:
            self._value = val
            self.lineEdit.setText(f"{val:.3f}")
            self.valueChanged.emit(val)
            self.updateButtonStates()

    def value(self) -> float:
        return self._value

class CustomSpinBox(QWidget):
    """
    A simple custom spin box that mimics QSpinBox functionality.
    Emits valueChanged(int) when the value changes.
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, minimum=0, maximum=100, initial=0, step=1, parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._value = initial

        # Create a line edit to show the value.
        self.lineEdit = QLineEdit(str(initial))
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        # Optionally, restrict input to integers using a validator.
        
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        self.lineEdit.editingFinished.connect(self.editingFinished)

        # Create up and down buttons with arrow text or icons.
        self.upButton = QToolButton()
        self.upButton.setText("▲")
        self.downButton = QToolButton()
        self.downButton.setText("▼")
        self.upButton.clicked.connect(self.increaseValue)
        self.downButton.clicked.connect(self.decreaseValue)

        # Arrange the buttons vertically.
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.upButton)
        buttonLayout.addWidget(self.downButton)
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)

        # Arrange the line edit and buttons horizontally.
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.lineEdit)
        mainLayout.addLayout(buttonLayout)
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)

        self.updateButtonStates()

    @property
    def value(self):
        return self._value

    def setValue(self, val):
        if val < self.minimum:
            val = self.minimum
        elif val > self.maximum:
            val = self.maximum
        if val != self._value:
            self._value = val
            self.lineEdit.setText(str(val))
            self.valueChanged.emit(val)
            self.updateButtonStates()

    def updateButtonStates(self):
        self.upButton.setEnabled(self._value < self.maximum)
        self.downButton.setEnabled(self._value > self.minimum)

    def increaseValue(self):
        self.setValue(self._value + self.step)

    def decreaseValue(self):
        self.setValue(self._value - self.step)

    def editingFinished(self):
        try:
            newVal = int(self.lineEdit.text())
        except ValueError:
            newVal = self._value
        self.setValue(newVal)


class ImagePreviewDialog(QDialog):
    def __init__(self, np_image, is_mono=False):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.resize(640, 480)  # Set initial size
        self.autostretch_enabled = False  # Autostretch toggle for preview
        self.is_mono = is_mono  # Store is_mono flag
        self.zoom_factor = 1.0  # Track the zoom level

        # Store the 32-bit numpy image for reference
        self.np_image = np_image

        # Set up the layout and the scroll area
        layout = QVBoxLayout(self)

        # Autostretch and Zoom Buttons
        button_layout = QHBoxLayout()
        self.autostretch_button = QPushButton("AutoStretch (Off)")
        self.autostretch_button.setCheckable(True)
        self.autostretch_button.toggled.connect(self.toggle_autostretch)
        button_layout.addWidget(self.autostretch_button)

        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_layout.addWidget(self.zoom_out_button)

        layout.addLayout(button_layout)

        # Scroll area for displaying the image
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        layout.addWidget(self.scroll_area)

        # Set up the QLabel to display the image
        self.image_label = QLabel()
        self.display_qimage(self.np_image)  # Display the image with the initial numpy array
        self.scroll_area.setWidget(self.image_label)

        # Set up mouse dragging
        self.dragging = False
        self.drag_start_pos = QPoint()

        # Enable mouse wheel for zooming
        self.image_label.installEventFilter(self)

        # Center the scroll area on initialization
        QTimer.singleShot(0, self.center_scrollbars)  # Delay to ensure layout is set

    def display_qimage(self, np_img):
        """Convert a numpy array to QImage and display it at the current zoom level."""
        display_image_uint8 = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)

        if len(display_image_uint8.shape) == 3 and display_image_uint8.shape[2] == 3:
            # RGB image
            height, width, channels = display_image_uint8.shape
            bytes_per_line = 3 * width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        elif len(display_image_uint8.shape) == 2:
            # Grayscale image
            height, width = display_image_uint8.shape
            bytes_per_line = width
            qimage = QImage(display_image_uint8.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            raise ValueError(f"Unexpected image shape: {display_image_uint8.shape}")

        # Apply zoom
        pixmap = QPixmap.fromImage(qimage)
        scaled_width = int(pixmap.width() * self.zoom_factor)  # Convert to integer
        scaled_height = int(pixmap.height() * self.zoom_factor)  # Convert to integer
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()


    def toggle_autostretch(self, checked):
        self.autostretch_enabled = checked
        self.autostretch_button.setText("AutoStretch (On)" if checked else "AutoStretch (Off)")
        self.apply_autostretch()

    def apply_autostretch(self):
        """Apply or remove autostretch while maintaining 32-bit precision."""
        print("applying autostretch")
        target_median = 0.25  # Target median for stretching

        if self.autostretch_enabled:
            if self.np_image.ndim == 2:
                # mono stretch path
                stretched = stretch_mono_image(self.np_image, target_median)
                display_image = np.stack([stretched]*3, axis=-1)
            elif self.np_image.ndim == 3 and self.np_image.shape[2] == 3:
                # color stretch path
                display_image = stretch_color_image(self.np_image, target_median, linked=False)
            else:
                raise ValueError(f"Unexpected image shape for autostretch: {self.np_image.shape}")
        else:
            # autostretch off: just show original
            if self.np_image.ndim == 2:
                display_image = np.stack([self.np_image]*3, axis=-1)
            else:
                display_image = self.np_image


        print(f"Debug: Display image shape before QImage conversion: {display_image.shape}")
        self.display_qimage(display_image)



    
    def zoom_in(self):
        """Increase the zoom factor and refresh the display."""
        self.zoom_factor *= 1.2  # Increase zoom by 20%
        if self.autostretch_enabled:
            self.apply_autostretch()
        else:
            self.display_qimage(self.np_image)

    
    def zoom_out(self):
        """Decrease the zoom factor and refresh the display."""
        self.zoom_factor /= 1.2  # Decrease zoom by 20%
        if self.autostretch_enabled:
            self.apply_autostretch()
        else:
            self.display_qimage(self.np_image)

    def eventFilter(self, source, event):
        """Handle mouse wheel events for zooming."""
        if source == self.image_label and event.type() == QEvent.Type.Wheel:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            return True
        return super().eventFilter(source, event)

    def mousePressEvent(self, event):
        """Start dragging if the left mouse button is pressed."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        """Handle dragging to move the scroll area."""
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y()
            )
            self.drag_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """Stop dragging when the left mouse button is released."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def center_scrollbars(self):
        """Centers the scrollbars to start in the middle of the image."""
        h_scroll = self.scroll_area.horizontalScrollBar()
        v_scroll = self.scroll_area.verticalScrollBar()
        h_scroll.setValue((h_scroll.maximum() + h_scroll.minimum()) // 2)
        v_scroll.setValue((v_scroll.maximum() + v_scroll.minimum()) // 2)

    def resizeEvent(self, event):
        """Handle resizing of the dialog."""
        super().resizeEvent(event)
        self.display_qimage(self.np_image)


BlinkComparatorPro = BlinkTab

# ⬇️ paste your SASv2 code here (exactly as you sent), then end with:
class BlinkComparatorPro(BlinkTab):
    """Alias class so the main app can import a SASpro-named tool."""
    pass
