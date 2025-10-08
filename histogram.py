# pro/histogram.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont

def _to_float01(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    a = np.asarray(img)
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        return (a.astype(np.float32) / float(info.max)).clip(0, 1)
    if a.dtype.kind == "f":
        # assume already normalized-ish; softly normalize if above 1
        mx = float(a.max()) if a.size else 1.0
        return (a.astype(np.float32) / (mx if mx > 1.0 else 1.0)).clip(0, 1)
    return a.astype(np.float32)

class HistogramDialog(QDialog):
    """
    Per-document histogram (non-modal).
    - Connects to ImageDocument.changed and repaints automatically.
    - Multiple dialogs can be open at once (each bound to one doc).
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.doc = document
        self.image = _to_float01(document.image)

        self.zoom_factor = 1.0   # 1.0 = 100%
        self.log_scale   = False # log X
        self.log_y       = False # log Y
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()

        # wire up to this specific document
        self.doc.changed.connect(self._on_doc_changed)
        # If the doc object goes away, close this dialog
        self.doc.destroyed.connect(self.deleteLater)

        # Render initial
        self._draw_histogram()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._doc_conn = False
        if getattr(self, "doc", None) is not None:
            try:
                self.doc.destroyed.connect(self._on_doc_destroyed)
                self._doc_conn = True
            except Exception:
                pass        

    # ---------- UI ----------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        top = QHBoxLayout()
        # scroll area + label for the pixmap
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setFixedSize(520, 310)
        self.scroll_area.setWidgetResizable(False)

        self.hist_label = QLabel(self)
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.hist_label)
        top.addWidget(self.scroll_area)

        # stats table
        self.stats_table = QTableWidget(self)
        self.stats_table.setRowCount(4)
        self.stats_table.setColumnCount(1)
        self.stats_table.setVerticalHeaderLabels(["Min", "Max", "Median", "StdDev"])
        self.stats_table.setFixedWidth(360)
        top.addWidget(self.stats_table)

        main_layout.addLayout(top)

        # controls
        ctl = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setRange(50, 1000)   # 50%..1000%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(10)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)

        ctl.addWidget(QLabel("Zoom:"))
        ctl.addWidget(self.zoom_slider)

        self.btn_logx = QPushButton("Toggle Log X-Axis", self)
        self.btn_logx.setCheckable(True)
        self.btn_logx.toggled.connect(self._toggle_log_x)
        ctl.addWidget(self.btn_logx)

        self.btn_logy = QPushButton("Toggle Log Y-Axis", self)
        self.btn_logy.setCheckable(True)
        self.btn_logy.toggled.connect(self._toggle_log_y)
        ctl.addWidget(self.btn_logy)

        main_layout.addLayout(ctl)

        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close)

        self.setLayout(main_layout)

    # ---------- slots ----------
    def _on_doc_changed(self):
        self.image = _to_float01(self.doc.image)
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
    def _draw_histogram(self):
        if self.image is None:
            self.hist_label.clear()
            return

        img = self.image
        # squeeze 1-channel 3D → 2D
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]

        base_width = 512
        height = 300
        width = int(base_width * self.zoom_factor)
        bin_count = 512

        # choose bin edges (0..1 domain)
        if self.log_scale:
            # epsilon > 0 to avoid log(0). If entire image is 0, use tiny eps.
            eps = max(1e-6, float(np.min(img[img > 0])) if np.any(img > 0) else 1e-6)
            log_min, log_max = np.log10(eps), 0.0
            if abs(log_max - log_min) < 1e-12:
                # degenerate → fallback linear
                bin_edges = np.linspace(eps, 1.0, bin_count + 1)
                def x_pos(edge):  # noqa
                    if eps >= 1.0:
                        return 0
                    return int((edge - eps) / max(1e-12, (1.0 - eps)) * width)
            else:
                bin_edges = np.logspace(log_min, log_max, bin_count + 1)
                def x_pos(edge):  # noqa
                    return int((np.log10(edge) - log_min) / (log_max - log_min) * width)
        else:
            bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
            def x_pos(edge):  # noqa
                return int(edge * width)

        # prepare canvas
        pm = QPixmap(width, height)
        pm.fill(QColor("white"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # draw bars
        if img.ndim == 3 and img.shape[2] == 3:
            colors = [QColor(255, 0, 0, 140), QColor(0, 180, 0, 140), QColor(0, 0, 255, 140)]
            for ch in range(3):
                hist, _ = np.histogram(img[..., ch].ravel(), bins=bin_edges)
                hist = hist.astype(np.float32)
                if self.log_y:
                    hist = np.log10(hist + 1.0)
                mv = float(hist.max())
                if mv > 0:
                    hist /= mv
                p.setPen(QPen(colors[ch]))
                for i in range(bin_count):
                    x0 = x_pos(bin_edges[i]); x1 = x_pos(bin_edges[i+1])
                    w  = max(1, x1 - x0)
                    h  = int(hist[i] * height)
                    p.drawRect(x0, height - h, w, h)
        else:
            # mono
            gray = img if img.ndim == 2 else img[..., 0]
            hist, _ = np.histogram(gray.ravel(), bins=bin_edges)
            hist = hist.astype(np.float32)
            if self.log_y:
                hist = np.log10(hist + 1.0)
            mv = float(hist.max())
            if mv > 0:
                hist /= mv
            p.setPen(QPen(QColor(0, 0, 0)))
            for i in range(bin_count):
                x0 = x_pos(bin_edges[i]); x1 = x_pos(bin_edges[i+1])
                w  = max(1, x1 - x0)
                h  = int(hist[i] * height)
                p.drawRect(x0, height - h, w, h)

        # axis
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.drawLine(0, height - 1, width, height - 1)

        # ticks + labels
        p.setFont(QFont("Arial", 10))
        if self.log_scale:
            # tick values from eps..1 (10 ticks)
            eps = bin_edges[0]
            ticks = np.logspace(np.log10(eps), 0.0, 11)
            for t in ticks:
                x = x_pos(t)
                p.drawLine(x, height - 1, x, height - 6)
                p.drawText(x - 18, height - 10, f"{t:.3f}")
        else:
            ticks = np.linspace(0.0, 1.0, 11)
            for t in ticks:
                x = x_pos(t)
                p.drawLine(x, height - 1, x, height - 6)
                p.drawText(x - 10, height - 10, f"{t:.1f}")

        p.end()
        self.hist_label.setPixmap(pm)
        self.hist_label.resize(pm.size())

        self._update_stats()

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

        labels = ["Min", "Max", "Median", "StdDev"]
        rows = {
            "Min":    [float(np.min(c))     for c in chans],
            "Max":    [float(np.max(c))     for c in chans],
            "Median": [float(np.median(c))  for c in chans],
            "StdDev": [float(np.std(c))     for c in chans],
        }

        self.stats_table.setRowCount(4)
        for r, lab in enumerate(labels):
            for c, val in enumerate(rows[lab]):
                it = QTableWidgetItem(f"{val:.4f}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setItem(r, c, it)

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