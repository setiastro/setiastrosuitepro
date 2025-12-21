# pro/histogram.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QSettings, QTimer, QEvent, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QMessageBox, QToolButton, QInputDialog, QSplitter, QSizePolicy, QHeaderView
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QPalette

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

def _to_float_preserve(img):
    if img is None: return None
    a = np.asarray(img)
    return a.astype(np.float32, copy=False) if a.dtype != np.float32 else a



class HistogramDialog(QDialog):
    """
    Per-document histogram (non-modal).
    - Connects to ImageDocument.changed and repaints automatically.
    - Multiple dialogs can be open at once (each bound to one doc).
    """
    pivotPicked = pyqtSignal(float)  # normalized [0..1] x position for GHS pivot
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Histogram"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.doc = document
        self.image = _to_float_preserve(document.image)

        self.zoom_factor = 1.0   # 1.0 = 100%
        self.log_scale   = False # log X
        self.log_y       = False # log Y
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._eps_log         = 1e-6      # first log bin edge (for labels)

        # for mapping clicks → normalized x
        self._click_mapping = None  # dict or None
        self.settings = QSettings()
        self.sensor_max01 = 1.0
        self.sensor_native_max = None     # user ADU max (e.g., 65532)
        self.native_theoretical_max = None

        # histogram cache
        self._bin_count       = 512
        self._bin_edges_lin   = None      # np.ndarray | None
        self._bin_edges_log   = None      # np.ndarray | None
        self._counts_lin      = None      # list[np.ndarray] | None
        self._counts_log      = None      # list[np.ndarray] | None
        self._is_color        = False
        self._eps_log         = 1e-6      # first log bin edge (for labels)

        self._load_sensor_max_setting()
        self._build_ui()

        # debounce timer for resize / splitter moves
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(80)  # ms; tweak if you want snappier/slower
        self._resize_timer.timeout.connect(self._draw_histogram)

        # prime histogram & stats from initial image
        self._recompute_hist_cache()
        self._update_stats()

        # wire up to this specific document
        self.doc.changed.connect(self._on_doc_changed)
        # If the doc object goes away, close this dialog
        self.doc.destroyed.connect(self.deleteLater)

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._doc_conn = False
        if getattr(self, "doc", None) is not None:
            try:
                self.doc.destroyed.connect(self._on_doc_destroyed)
                self._doc_conn = True
            except Exception:
                pass

        # Do the first draw once the widget has a real size
        QTimer.singleShot(0, self._draw_histogram)



    # ---------- UI ----------
    def _build_ui(self):
        # Make it start at a sensible size
        self.setMinimumSize(800, 400)
        self.resize(900, 500)

        main_layout = QVBoxLayout(self)

        # --- top area: splitter with histogram + stats ---
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # left: scroll area + label for the pixmap
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.hist_label = QLabel(self)
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.hist_label)
        self.hist_label.installEventFilter(self)
        self.hist_label.setToolTip(self.tr(
            "Ctrl+Click on the histogram to send that intensity as the "
            "pivot to Hyperbolic Stretch (if open)."
        ))
        self.scroll_area.viewport().installEventFilter(self)

        splitter.addWidget(self.scroll_area)

        # right: stats table
        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(7)
        self.stats_table.setColumnCount(1)
        self.stats_table.setVerticalHeaderLabels([
            self.tr("Min"), self.tr("Max"), self.tr("Median"), self.tr("StdDev"),
            self.tr("MAD"), self.tr("Low Clipped"), self.tr("High Clipped")
        ])

        # Let it grow/shrink with the splitter
        self.stats_table.setMinimumWidth(320)
        self.stats_table.setSizePolicy(
            QSizePolicy.Policy.Preferred,      # <- was Fixed
            QSizePolicy.Policy.Expanding,
        )

        # Make the columns use available width nicely
        hdr = self.stats_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # hdr.setStretchLastSection(True)
        splitter.addWidget(self.stats_table)

        # Give more space to histogram side by default
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        # Explicit initial sizes so it doesn't start with a tiny histogram
        splitter.setSizes([650, 250])

        QTimer.singleShot(0, self._adjust_stats_width)

        main_layout.addWidget(splitter)

        # --- controls row (unchanged except for being below splitter) ---
        ctl = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setRange(50, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        ctl.addWidget(QLabel(self.tr("Zoom:")))
        ctl.addWidget(self.zoom_slider)

        self.btn_logx = QPushButton(self.tr("Toggle Log X-Axis"), self)
        self.btn_logx.setCheckable(True)
        self.btn_logx.toggled.connect(self._toggle_log_x)
        ctl.addWidget(self.btn_logx)

        self.btn_logy = QPushButton(self.tr("Toggle Log Y-Axis"), self)
        self.btn_logy.setCheckable(True)
        self.btn_logy.toggled.connect(self._toggle_log_y)
        ctl.addWidget(self.btn_logy)

        self.btn_sensor_max = QToolButton(self)
        self.btn_sensor_max.setText("?")
        self.btn_sensor_max.setToolTip(self.tr(
            "Set your camera's true saturation level for clipping warnings.\n"
            "Tip: take an overexposed frame and see its max ADU."
        ))
        self.btn_sensor_max.clicked.connect(self._prompt_sensor_max)
        ctl.addWidget(self.btn_sensor_max)

        main_layout.addLayout(ctl)

        btn_close = QPushButton(self.tr("Close"), self)
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close)

        self.setLayout(main_layout)



    # ---------- slots ----------
    def _on_doc_changed(self):
        self.image = _to_float_preserve(self.doc.image)
        self._recompute_hist_cache()
        self._update_stats()
        self._draw_histogram()

    def _on_zoom_changed(self, v: int):
        self.zoom_factor = v / 100.0
        self._draw_histogram()

    def _toggle_log_x(self, on: bool):
        self.log_scale = bool(on)
        self._draw_histogram()

    def _toggle_log_y(self, on: bool):
        self.log_y = bool(on)
        self._draw_histogram()

    # ---------- drawing ----------
    # ---------- drawing ----------
    def _draw_histogram(self):
        # nothing to draw yet
        if self.image is None or self._bin_edges_lin is None:
            self.hist_label.clear()
            return

        # use available size in the scroll area's viewport
        if self.scroll_area is not None:
            vp = self.scroll_area.viewport()
            avail_w = max(200, vp.width())
            avail_h = max(200, vp.height())
        else:
            avail_w = 512
            avail_h = 300

        base_width = avail_w
        height = avail_h
        width = int(base_width * self.zoom_factor)

        # layout margins
        left_margin    = 32   # room for Y labels
        top_margin     = 12   # room so top ticks/text aren't clipped
        bottom_margin  = 24   # room for X labels
        axis_y         = height - bottom_margin
        usable_h       = max(1, axis_y - top_margin)
        plot_width     = max(1, width - left_margin)

        # choose edges + raw counts from cache
        if self.log_scale:
            bin_edges   = self._bin_edges_log
            counts_list = self._counts_log
        else:
            bin_edges   = self._bin_edges_lin
            counts_list = self._counts_lin

        if bin_edges is None or counts_list is None:
            self.hist_label.clear()
            return

        bin_count = len(bin_edges) - 1

        # precompute log range if needed
        if self.log_scale:
            # guard: avoid log10(<=0)
            be0 = float(bin_edges[0])
            if be0 <= 0:
                be0 = self._eps_log
            log_min = np.log10(be0)
            log_max = 0.0
        else:
            log_min = None
            log_max = None

        # map X-domain edge → pixel X
        def x_pos(edge: float) -> int:
            if self.log_scale:
                if edge <= 0:
                    edge = self._eps_log
                if abs(log_max - log_min) < 1e-12:
                    return left_margin
                return left_margin + int(
                    (np.log10(edge) - log_min) / (log_max - log_min) * plot_width
                )
            else:
                return left_margin + int(edge * plot_width)

        # --- convert counts → display values (linear or log Y) ---
        vals_list: list[np.ndarray] = []
        max_val = 0.0
        for counts in counts_list:
            if self.log_y:
                vals = np.log10(counts + 1.0)
            else:
                vals = counts.astype(np.float32)
            if vals.size:
                max_val = max(max_val, float(vals.max()))
            vals_list.append(vals)

        if max_val <= 0:
            max_val = 1.0

        # theme colors
        pal = self.window().palette() if self.window() else self.palette()
        bg_color   = pal.color(QPalette.ColorRole.Window)
        text_color = pal.color(QPalette.ColorRole.Text)

        if bg_color.lightness() < 128:
            axis_color  = QColor(210, 210, 210)
            label_color = QColor(245, 245, 245)
        else:
            axis_color  = QColor(40, 40, 40)
            label_color = text_color

        grid_color = QColor(axis_color)
        grid_color.setAlpha(60)
        grid_pen = QPen(grid_color)
        grid_pen.setWidth(1)

        pm = QPixmap(width, height)
        pm.fill(bg_color)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # helper: map normalized [0,1] → Y pixel (0 at bottom, 1 at top)
        def y_pos(norm: float) -> int:
            # norm in [0,1], map 0→axis_y, 1→top_margin
            return int(top_margin + (1.0 - norm) * usable_h)

        # ----- draw bars -----
        if self._is_color:
            colors = [
                QColor(255, 0, 0, 140),
                QColor(0, 180, 0, 140),
                QColor(0, 0, 255, 140),
            ]
            for ch_idx, vals in enumerate(vals_list):
                hn = vals / max_val
                p.setPen(QPen(colors[ch_idx]))
                for i in range(bin_count):
                    x0 = x_pos(float(bin_edges[i]))
                    x1 = x_pos(float(bin_edges[i + 1]))
                    w  = max(1, x1 - x0)
                    h  = int(hn[i] * usable_h)
                    y0 = axis_y - h
                    p.drawRect(x0, y0, w, h)
        else:
            vals = vals_list[0]
            hn = vals / max_val
            p.setPen(QPen(axis_color))
            for i in range(bin_count):
                x0 = x_pos(float(bin_edges[i]))
                x1 = x_pos(float(bin_edges[i + 1]))
                w  = max(1, x1 - x0)
                h  = int(hn[i] * usable_h)
                y0 = axis_y - h
                p.drawRect(x0, y0, w, h)

        # ----- axes -----
        p.setPen(QPen(axis_color, 2))
        # X axis at axis_y, Y axis from top_margin down to axis_y
        p.drawLine(left_margin, axis_y, width - 1, axis_y)
        p.drawLine(left_margin, top_margin, left_margin, axis_y)

        p.setFont(QFont("Arial", 10))

        # ----- X ticks + grid -----
        if self.log_scale:
            ticks = np.logspace(np.log10(bin_edges[0]), 0.0, 11)
            for t in ticks:
                x = x_pos(float(t))
                if left_margin < x < width - 1:
                    p.setPen(grid_pen)
                    p.drawLine(x, top_margin, x, axis_y)
                p.setPen(axis_color)
                p.drawLine(x, axis_y, x, axis_y - 5)
                p.setPen(label_color)
                p.drawText(x - 18, axis_y + bottom_margin - 8, f"{t:.3f}")
        else:
            ticks = np.linspace(0.0, 1.0, 11)
            for t in ticks:
                x = x_pos(float(t))
                if left_margin < x < width - 1:
                    p.setPen(grid_pen)
                    p.drawLine(x, top_margin, x, axis_y)
                p.setPen(axis_color)
                p.drawLine(x, axis_y, x, axis_y - 5)
                p.setPen(label_color)
                p.drawText(x - 10, axis_y + bottom_margin - 8, f"{t:.1f}")

        # ----- Y ticks + grid -----
        n_yticks = 6
        if self.log_y:
            exps = np.linspace(0.0, max_val, n_yticks)
            norms = exps / max_val
            labels = [f"{10**e:.0f}" for e in exps]
        else:
            vals_for_ticks = np.linspace(0.0, max_val, n_yticks)
            norms = vals_for_ticks / max_val
            labels = [f"{v:.0f}" for v in vals_for_ticks]

        for i, (yn, lab) in enumerate(zip(norms, labels)):
            y = y_pos(float(yn))
            if 0 < i < n_yticks - 1:
                p.setPen(grid_pen)
                p.drawLine(left_margin, y, width - 1, y)
            p.setPen(axis_color)
            p.drawLine(left_margin - 5, y, left_margin, y)
            p.setPen(label_color)
            p.drawText(2, y + 4, lab)

        # --- draw effective-max marker if user set one ---
        if self.sensor_max01 < 0.9999:
            x = x_pos(self.sensor_max01)
            p.setPen(QPen(QColor(220, 0, 0), 2, Qt.PenStyle.DashLine))
            p.drawLine(x, top_margin, x, axis_y)
            p.drawText(min(x + 4, width - 80), top_margin + 12,
                       self.tr("True Max {0:.4f}").format(self.sensor_max01))
        # store mapping info for Ctrl+click → normalized x
        try:
            self._click_mapping = {
                "left_margin": left_margin,
                "plot_width": plot_width,
                "axis_y": axis_y,
                "top_margin": top_margin,
                "height": height,
                "log_scale": bool(self.log_scale),
                "log_min": log_min,
                "log_max": log_max,
            }
        except Exception:
            self._click_mapping = None
        p.end()
        self.hist_label.setPixmap(pm)
        self.hist_label.resize(pm.size())

    def _x_pix_to_u(self, x_pix: int) -> float | None:
        """
        Map a horizontal pixel coordinate (in the label) to a normalized
        intensity in [0..1], respecting linear / log X modes.
        """
        m = self._click_mapping
        if not m:
            return None

        left = m["left_margin"]
        width = max(1, m["plot_width"])
        if x_pix < left or x_pix > left + width:
            return None

        t = (x_pix - left) / float(width)
        t = max(0.0, min(1.0, t))

        if not m["log_scale"]:
            # linear: domain is already [0..1]
            return float(t)

        # log X: t in [0..1] corresponds to [10^log_min .. 10^log_max] (log_max ~ 0)
        log_min = m.get("log_min", None)
        log_max = m.get("log_max", None)
        if log_min is None or log_max is None or abs(log_max - log_min) < 1e-12:
            return float(t)

        log_v = log_min + t * (log_max - log_min)
        v = 10.0 ** log_v
        # v is in (eps .. 1]; clamp to [0..1]
        return float(max(0.0, min(1.0, v)))


    def _recompute_hist_cache(self):
        """Compute histograms once for the current image.

        This is called when the document image changes. Resizing / zooming
        will only redraw using this cached data.
        """
        img = self.image
        self._bin_edges_lin = None
        self._bin_edges_log = None
        self._counts_lin    = None
        self._counts_log    = None
        self._is_color      = False
        self._eps_log       = 1e-6

        if img is None:
            return

        a = img
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]

        if a.ndim == 3 and a.shape[2] == 3:
            chans = [a[..., i] for i in range(3)]
            self._is_color = True
        else:
            chan = a if a.ndim == 2 else a[..., 0]
            chans = [chan]
            self._is_color = False

        bin_count = self._bin_count

        # --- linear X bins ---
        bin_edges_lin = np.linspace(0.0, 1.0, bin_count + 1).astype(np.float32)
        counts_lin: list[np.ndarray] = []
        for c in chans:
            counts, _ = np.histogram(c.ravel(), bins=bin_edges_lin)
            counts_lin.append(counts.astype(np.float32))

        # --- log X bins ---
        pos = a[a > 0]
        eps = max(1e-6, float(pos.min())) if pos.size else 1e-6
        log_min, log_max = np.log10(eps), 0.0
        if abs(log_max - log_min) < 1e-12:
            bin_edges_log = np.linspace(eps, 1.0, bin_count + 1).astype(np.float32)
        else:
            bin_edges_log = np.logspace(log_min, log_max, bin_count + 1).astype(np.float32)

        counts_log: list[np.ndarray] = []
        for c in chans:
            counts, _ = np.histogram(c.ravel(), bins=bin_edges_log)
            counts_log.append(counts.astype(np.float32))

        self._bin_edges_lin = bin_edges_lin
        self._bin_edges_log = bin_edges_log
        self._counts_lin    = counts_lin
        self._counts_log    = counts_log
        self._eps_log       = float(eps)


    def _schedule_redraw(self):
        # Only bother if visible; restart timer each time
        if self.isVisible():
            self._resize_timer.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_redraw()

    def eventFilter(self, obj, event):
        # Ctrl+click on the histogram pixmap → emit pivotPicked(u)
        if obj is self.hist_label and event.type() == QEvent.Type.MouseButtonPress:
            if (event.button() == Qt.MouseButton.LeftButton and
                    (event.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                pos = event.position().toPoint()
                u = self._x_pix_to_u(pos.x())
                if u is not None:
                    # emit normalized pivot in [0..1]
                    self.pivotPicked.emit(u)
                    event.accept()
                    return True

        # When the splitter moves, the scroll_area viewport gets a Resize event
        if self.scroll_area is not None and obj is self.scroll_area.viewport():
            if event.type() == QEvent.Type.Resize:
                self._schedule_redraw()

        return super().eventFilter(obj, event)

    def _update_stats(self):
        if self.image is None:
            return

        img = self.image
        # determine channels
        if img.ndim == 3 and img.shape[2] == 3:
            chans = [img[..., i] for i in range(3)]
            self.stats_table.setColumnCount(3)
            self.stats_table.setHorizontalHeaderLabels(["R", "G", "B"])
        else:
            chan = img if img.ndim == 2 else img[..., 0]
            chans = [chan]
            self.stats_table.setColumnCount(1)
            self.stats_table.setHorizontalHeaderLabels(["Gray"])

        eps = 1e-6  # tolerance for "exactly 0/1" after float ops

        row_defs = [
            (self.tr("Min"),          lambda c: float(np.min(c)),                "{:.4f}"),
            (self.tr("Max"),          lambda c: float(np.max(c)),                "{:.4f}"),
            (self.tr("Median"),       lambda c: float(np.median(c)),             "{:.4f}"),
            (self.tr("StdDev"),       lambda c: float(np.std(c)),                "{:.4f}"),
            (self.tr("MAD"),          lambda c: float(np.median(np.abs(c - np.median(c)))), "{:.4f}"),
            (self.tr("Low Clipped"),  lambda c: _clip_fmt(c, low=True,  eps=eps), "{}"),
            (self.tr("High Clipped"), lambda c: _clip_fmt(c, low=False, eps=eps), "{}"),
        ]

        def _clip_fmt(c, low: bool, eps: float):
            flat = np.ravel(c)
            n = flat.size if flat.size else 1
            if low:
                k = int(np.count_nonzero(flat <= eps))
            else:
                hi_thr = max(eps, self.sensor_max01 - eps)
                k = int(np.count_nonzero(flat >= hi_thr))
            pct = 100.0 * k / n
            return f"{k} ({pct:.3f}%)"

        # apply labels + sizes
        self.stats_table.setRowCount(len(row_defs))
        self.stats_table.setVerticalHeaderLabels([lab for lab, _, _ in row_defs])

        # fill cells
        for r, (lab, fn, fmt) in enumerate(row_defs):
            for c_idx, c_arr in enumerate(chans):
                val = fn(c_arr)
                text = fmt.format(val)
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # --- visual pop for non-trivial clipping ---
                if lab in (self.tr("Low Clipped"), self.tr("High Clipped")):
                    # text looks like: "123 (0.456%)"
                    try:
                        pct_str = text.split("(")[1].split("%")[0]
                        pct = float(pct_str)
                    except Exception:
                        pct = 0.0

                    # thresholds you can tweak
                    # <0.01%: ignore
                    # 0.01–0.1%: mild warning
                    # 0.1–1%: clear warning
                    # >1%: strong warning
                    if pct >= 1.0:
                        it.setBackground(QColor(100, 30, 30))  # strong red tint
                    elif pct >= 0.1:
                        it.setBackground(QColor(70, 30, 30))  # medium red tint
                    elif pct >= 0.01:
                        it.setBackground(QColor(40, 30, 30))  # mild red tint

                self.stats_table.setItem(r, c_idx, it)

        self._adjust_stats_width()        

    def _theoretical_native_max_from_meta(self):
        meta = getattr(self.doc, "metadata", None) or {}
        bd = str(meta.get("bit_depth", "")).lower()

        if "16-bit" in bd:
            return 65535
        if "8-bit" in bd:
            return 255
        if "32-bit unsigned" in bd:
            return 4294967295
        return None

    def _settings_key_for_native_max(self, native_theoretical_max):
        if native_theoretical_max == 65535:
            return "histogram/sensor_max_native_16"
        if native_theoretical_max == 255:
            return "histogram/sensor_max_native_8"
        if native_theoretical_max == 4294967295:
            return "histogram/sensor_max_native_32u"
        return "histogram/sensor_max_native_generic"

    def _load_sensor_max_setting(self):
        self.native_theoretical_max = self._theoretical_native_max_from_meta()
        if self.native_theoretical_max:
            key = self._settings_key_for_native_max(self.native_theoretical_max)
            val = self.settings.value(key, None)
            if val is not None:
                try:
                    self.sensor_native_max = float(val)
                except Exception:
                    self.sensor_native_max = None

        self._recompute_effective_max01()

    def _recompute_effective_max01(self):
        if self.native_theoretical_max and self.sensor_native_max:
            self.sensor_max01 = float(self.sensor_native_max) / float(self.native_theoretical_max)
            self.sensor_max01 = float(np.clip(self.sensor_max01, 1e-6, 1.0))
        else:
            self.sensor_max01 = 1.0

    def _prompt_sensor_max(self):
        self.native_theoretical_max = self._theoretical_native_max_from_meta()

        if self.native_theoretical_max:
            key = self._settings_key_for_native_max(self.native_theoretical_max)
            current = self.sensor_native_max or self.native_theoretical_max

            val, ok = QInputDialog.getInt(
                self,
                self.tr("Sensor True Max (ADU)"),
                self.tr("Enter your sensor's true saturation value in native ADU.\n"
                "(Typical max for this file type is {0})\n\n"
                "You can measure this by taking a deliberately overexposed frame\n"
                "and reading its maximum pixel value.").format(self.native_theoretical_max),
                int(current),
                1,
                int(self.native_theoretical_max)
            )
            if ok:
                self.sensor_native_max = float(val)
                self.settings.setValue(key, float(val))
        else:
            # float images / unknown depth: allow normalized max
            val, ok = QInputDialog.getDouble(
                self,
                self.tr("Histogram Effective Max"),
                self.tr("Enter effective maximum for clipping (normalized units)."),
                float(self.sensor_max01),
                1e-6,
                1.0,
                6
            )
            if ok:
                self.sensor_max01 = float(val)
                self.settings.setValue("histogram/sensor_max01_generic", float(val))

        self._recompute_effective_max01()
        self._update_stats()      # High Clipped row depends on sensor_max01
        self._draw_histogram()

    def _adjust_stats_width(self):
        """Resize stats table so all columns are visible without a scrollbar."""
        if not self.stats_table:
            return

        # Let Qt compute natural column widths
        self.stats_table.resizeColumnsToContents()
        self.stats_table.resizeRowsToContents()

        vh = self.stats_table.verticalHeader()
        frame = self.stats_table.frameWidth()

        total_w = vh.width() + 2 * frame

        for col in range(self.stats_table.columnCount()):
            total_w += self.stats_table.columnWidth(col)

        # Room for a possible vertical scrollbar
        vbar = self.stats_table.verticalScrollBar()
        if vbar is not None:
            total_w += vbar.sizeHint().width()

        # A tiny padding so text isn't tight
        total_w += 6

        self.stats_table.setMinimumWidth(total_w)


    def _on_doc_destroyed(self, *args):
        # Called when the owner/document goes away.
        try:
            # Avoid re-entrancy; schedule deletion safely.
            self.deleteLater()
        except RuntimeError:
            pass

    def closeEvent(self, event):
        # Cleanly disconnect to avoid stray callbacks.
        if getattr(self, "_doc_conn", False) and getattr(self, "doc", None) is not None:
            try:
                self.doc.destroyed.disconnect(self._on_doc_destroyed)
            except (TypeError, RuntimeError):
                pass
            self._doc_conn = False

        super().closeEvent(event)
