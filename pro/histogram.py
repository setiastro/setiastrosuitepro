# pro/histogram.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QScrollArea,
    QTableWidget, QTableWidgetItem, QMessageBox, QToolButton, QInputDialog
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
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.doc = document
        self.image = _to_float_preserve(document.image)

        self.zoom_factor = 1.0   # 1.0 = 100%
        self.log_scale   = False # log X
        self.log_y       = False # log Y
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.settings = QSettings()
        self.sensor_max01 = 1.0
        self.sensor_native_max = None     # user ADU max (e.g., 65532)
        self.native_theoretical_max = None
        self._load_sensor_max_setting()
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
        # we'll set row/col labels dynamically in _update_stats
        self.stats_table.setRowCount(7)
        self.stats_table.setColumnCount(1)
        self.stats_table.setVerticalHeaderLabels([
            "Min", "Max", "Median", "StdDev",
            "MAD", "Low Clipped", "High Clipped"
        ])
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

        self.btn_sensor_max = QToolButton(self)
        self.btn_sensor_max.setText("?")
        self.btn_sensor_max.setToolTip(
            "Set your camera's true saturation level for clipping warnings.\n"
            "Tip: take an overexposed frame and see its max ADU."
        )
        self.btn_sensor_max.clicked.connect(self._prompt_sensor_max)
        ctl.addWidget(self.btn_sensor_max)

        main_layout.addLayout(ctl)

        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close)

        self.setLayout(main_layout)

    # ---------- slots ----------
    def _on_doc_changed(self):
        self.image = _to_float_preserve(self.doc.image)
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

        # --- draw effective-max marker if user set one ---
        if self.sensor_max01 < 0.9999:
            x = x_pos(self.sensor_max01)
            p.setPen(QPen(QColor(220, 0, 0), 2, Qt.PenStyle.DashLine))
            p.drawLine(x, 0, x, height)
            p.drawText(min(x + 4, width - 60), 12, f"True Max {self.sensor_max01:.4f}")

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

        eps = 1e-6  # tolerance for "exactly 0/1" after float ops

        row_defs = [
            ("Min",          lambda c: float(np.min(c)),                "{:.4f}"),
            ("Max",          lambda c: float(np.max(c)),                "{:.4f}"),
            ("Median",       lambda c: float(np.median(c)),             "{:.4f}"),
            ("StdDev",       lambda c: float(np.std(c)),                "{:.4f}"),
            ("MAD",          lambda c: float(np.median(np.abs(c - np.median(c)))), "{:.4f}"),
            ("Low Clipped",  lambda c: _clip_fmt(c, low=True,  eps=eps), "{}"),
            ("High Clipped", lambda c: _clip_fmt(c, low=False, eps=eps), "{}"),
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
                if lab in ("Low Clipped", "High Clipped"):
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
                "Sensor True Max (ADU)",
                f"Enter your sensor's true saturation value in native ADU.\n"
                f"(Typical max for this file type is {self.native_theoretical_max})\n\n"
                "You can measure this by taking a deliberately overexposed frame\n"
                "and reading its maximum pixel value.",
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
                "Histogram Effective Max",
                "Enter effective maximum for clipping (normalized units).",
                float(self.sensor_max01),
                1e-6,
                1.0,
                6
            )
            if ok:
                self.sensor_max01 = float(val)
                self.settings.setValue("histogram/sensor_max01_generic", float(val))

        self._recompute_effective_max01()
        self._draw_histogram()


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