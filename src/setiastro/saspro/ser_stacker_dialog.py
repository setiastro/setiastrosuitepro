# src/setiastro/saspro/ser_stacker_dialog.py
from __future__ import annotations
import os
import traceback
import numpy as np

from typing import Optional, Union, Sequence

SourceSpec = Union[str, Sequence[str]]

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QEvent, QTimer
from PyQt6.QtWidgets import (
    QWidget, QSpinBox, QMessageBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFormLayout, QComboBox, QDoubleSpinBox, QCheckBox, QTextEdit, QProgressBar,
    QScrollArea, QSlider, QToolButton
)

from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPixmap

from setiastro.saspro.ser_stack_config import SERStackConfig

from setiastro.saspro.ser_stacker import stack_ser, analyze_ser, AnalyzeResult
from setiastro.saspro.ser_stacker import _shift_image

def _source_basename_from_source(source: SourceSpec | None) -> str:
    """
    Best-effort base name from the SER source:
      - if source is "path/to/file.ser" -> "file"
      - if source is [paths...] -> first path stem
      - else -> ""
    """
    try:
        if isinstance(source, str) and source.strip():
            p = source.strip()
            base = os.path.basename(p)
            stem, _ = os.path.splitext(base)
            return stem.strip()
        if isinstance(source, (list, tuple)) and len(source) > 0:
            first = source[0]
            if isinstance(first, str) and first.strip():
                base = os.path.basename(first.strip())
                stem, _ = os.path.splitext(base)
                return stem.strip()
    except Exception:
        pass
    return ""


def _derive_view_base_title(main, doc) -> str:
    """
    Prefer the active view's title (respecting per-view rename/override),
    fallback to the document display name, then to doc.name, and finally 'Image'.
    Also strips any decorations if available.
    """
    # 1) Ask main for a subwindow for this document, if it exposes a helper
    try:
        if hasattr(main, "_subwindow_for_document"):
            sw = main._subwindow_for_document(doc)
            if sw:
                w = sw.widget() if hasattr(sw, "widget") else sw
                if hasattr(w, "_effective_title"):
                    t = w._effective_title() or ""
                else:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                if hasattr(w, "_strip_decorations"):
                    t, _ = w._strip_decorations(t)
                if t.strip():
                    return t.strip()
    except Exception:
        pass

    # 2) Try scanning MDI for a subwindow whose widget holds this document
    try:
        mdi = (getattr(main, "mdi_area", None)
               or getattr(main, "mdiArea", None)
               or getattr(main, "mdi", None))
        if mdi and hasattr(mdi, "subWindowList"):
            for sw in mdi.subWindowList():
                w = sw.widget()
                if getattr(w, "document", None) is doc:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                    if hasattr(w, "_strip_decorations"):
                        t, _ = w._strip_decorations(t)
                    if t.strip():
                        return t.strip()
    except Exception:
        pass

    # 3) Fallback to document's display name (then name, then generic)
    try:
        if hasattr(doc, "display_name"):
            t = doc.display_name()
            if t and t.strip():
                return t.strip()
    except Exception:
        pass

    return (getattr(doc, "name", "") or "Image").strip()


def _push_as_new_doc(
    main,
    source_doc,
    arr: np.ndarray,
    *,
    title_suffix: str = "_stack",
    source: str = "Planetary Stacker",
    source_path: SourceSpec | None = None,
):
    dm = getattr(main, "docman", None)
    if not dm or not hasattr(dm, "open_array"):
        return None

    try:
        # --- Base title ---
        base = ""
        if source_doc is not None:
            base = _derive_view_base_title(main, source_doc) or ""
        if not base:
            base = _source_basename_from_source(source_path) or ""
        if not base:
            base = "Stack"

        # Avoid double suffix
        suf = title_suffix or ""
        if suf and base.lower().endswith(suf.lower()):
            title = base
        else:
            title = f"{base}{suf}"

        x = np.asarray(arr)
        # keep mono mono
        if x.ndim == 3 and x.shape[2] == 1:
            x = x[..., 0]
        x = x.astype(np.float32, copy=False)

        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": bool(x.ndim == 2),
            "source": source,
        }

        newdoc = dm.open_array(x, metadata=meta, title=title)

        if hasattr(main, "_spawn_subwindow_for"):
            main._spawn_subwindow_for(newdoc)

        return newdoc
    except Exception:
        return None


class APEditorDialog(QDialog):
    """
    AP editor (AutoStakkert-ish):
    - Scrollable preview (fits to window by default)
    - Zoom controls (+/-/slider, Fit, 1:1)
    - Constant on-screen AP box thickness (draw boxes after scaling)
    - Left click: add AP
    - Right click: delete nearest AP
    """
    def __init__(
        self,
        parent=None,
        *,
        ref_img01: np.ndarray,
        ap_size: int,
        ap_spacing: int,
        ap_min_mean: float,
        initial_centers: np.ndarray | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Alignment Points (APs)")
        self.setModal(True)
        self.resize(1000, 750)

        self._ref = np.asarray(ref_img01, dtype=np.float32)
        self._H, self._W = self._ref.shape[:2]

        self._ap_size = int(ap_size)
        self._ap_spacing = int(ap_spacing)
        self._ap_min_mean = float(ap_min_mean)

        self._centers = None if initial_centers is None else np.asarray(initial_centers, dtype=np.int32).copy()

        # zoom state
        self._zoom = 1.0
        self._fit_pending = True  # do initial "fit to window" after first show

        # ---- Build UI ---------------------------------------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # scroll area + pix label
        self._pix = QLabel(self)
        self._pix.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pix.setMouseTracking(True)
        self._pix.setStyleSheet("background:#111;")  # makes the viewport look sane

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(False)       # we control label size
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._pix)

        outer.addWidget(self._scroll, 1)

        # Zoom row (under preview)
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = QPushButton("–", self)
        self.btn_zoom_in = QPushButton("+", self)
        self.btn_fit = QPushButton("Fit", self)
        self.btn_100 = QPushButton("1:1", self)

        self.sld_zoom = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_zoom.setRange(10, 400)     # percent
        self.sld_zoom.setValue(100)
        self.lbl_zoom = QLabel("100%", self)
        self.lbl_zoom.setStyleSheet("color:#aaa; min-width:60px;")

        self.btn_zoom_out.setFixedWidth(34)
        self.btn_zoom_in.setFixedWidth(34)

        zoom_row.addWidget(QLabel("Zoom:", self))
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.sld_zoom, 1)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.lbl_zoom)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.btn_fit)
        zoom_row.addWidget(self.btn_100)

        outer.addLayout(zoom_row)

        # hint + buttons
        self._lbl_hint = QLabel("Left click: add AP   |   Right click: delete nearest AP   |   Ctrl+Wheel: zoom", self)
        self._lbl_hint.setStyleSheet("color:#aaa;")
        outer.addWidget(self._lbl_hint, 0)

        # --- AP settings (in-dialog) ---
        ap_row = QHBoxLayout()

        self.lbl_ap = QLabel("AP:", self)
        self.lbl_ap.setStyleSheet("color:#aaa;")

        self.spin_ap_size = QSpinBox(self)
        self.spin_ap_size.setRange(16, 256)
        self.spin_ap_size.setSingleStep(8)
        self.spin_ap_size.setValue(int(self._ap_size))

        self.spin_ap_spacing = QSpinBox(self)
        self.spin_ap_spacing.setRange(8, 256)
        self.spin_ap_spacing.setSingleStep(8)
        self.spin_ap_spacing.setValue(int(self._ap_spacing))
        self.spin_ap_min_mean = QDoubleSpinBox(self)
        self.spin_ap_min_mean.setRange(0.0, 1.0)
        self.spin_ap_min_mean.setDecimals(3)
        self.spin_ap_min_mean.setSingleStep(0.005)
        self.spin_ap_min_mean.setValue(float(self._ap_min_mean))
        self.spin_ap_min_mean.setToolTip("Minimum mean intensity (0..1) required for an AP tile to be placed.")

        ap_row.addWidget(self.lbl_ap)
        ap_row.addSpacing(6)
        ap_row.addWidget(QLabel("Size", self))
        ap_row.addWidget(self.spin_ap_size)
        ap_row.addSpacing(10)
        ap_row.addWidget(QLabel("Spacing", self))
        ap_row.addWidget(self.spin_ap_spacing)
        ap_row.addSpacing(10)
        ap_row.addWidget(QLabel("Min mean", self))
        ap_row.addWidget(self.spin_ap_min_mean)        
        ap_row.addStretch(1)

        outer.addLayout(ap_row, 0)

        btn_row = QHBoxLayout()
        self.btn_auto = QPushButton("Auto-place", self)
        self.btn_clear = QPushButton("Clear", self)
        self.btn_ok = QPushButton("OK", self)
        self.btn_cancel = QPushButton("Cancel", self)
        btn_row.addWidget(self.btn_auto)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)
        outer.addLayout(btn_row)

        # signals
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_auto.clicked.connect(self._do_autoplace)
        self.btn_clear.clicked.connect(self._do_clear)

        self.btn_fit.clicked.connect(self._fit_to_window)
        self.btn_100.clicked.connect(lambda: self._set_zoom(1.0))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.sld_zoom.valueChanged.connect(self._on_zoom_slider)

        # intercept mouse clicks on label
        self._pix.mousePressEvent = self._on_mouse_press  # type: ignore

        self._ap_debounce = QTimer(self)
        self._ap_debounce.setSingleShot(True)
        self._ap_debounce.setInterval(250)  # ms
        self._ap_debounce.timeout.connect(self._apply_ap_params_and_relayout)
        self.spin_ap_min_mean.valueChanged.connect(self._schedule_ap_relayout)

        # apply redraw when changed
        self.spin_ap_size.valueChanged.connect(self._schedule_ap_relayout)
        self.spin_ap_spacing.valueChanged.connect(self._schedule_ap_relayout)

        # enable Ctrl+Wheel zoom on the scroll area's viewport
        self._scroll.viewport().installEventFilter(self)

        # precompute display base image (uint8) once
        self._base_u8 = self._make_display_u8(self._ref)

        # init centers
        if self._centers is None:
            self._do_autoplace()

        # first render
        self._render()

    def ap_size(self) -> int:
        return int(self._ap_size)

    def ap_spacing(self) -> int:
        return int(self._ap_spacing)

    def ap_min_mean(self) -> float:
        return float(self._ap_min_mean)

    def _schedule_ap_relayout(self):
        # Restart the timer each change
        try:
            self._ap_debounce.start()
        except Exception:
            # fallback: apply immediately if timer fails for some reason
            self._apply_ap_params_and_relayout()

    def _apply_ap_params_and_relayout(self):
        # Commit params
        self._ap_size = int(self.spin_ap_size.value())
        self._ap_spacing = int(self.spin_ap_spacing.value())
        self._ap_min_mean = float(self.spin_ap_min_mean.value())

        # Re-autoplace using the updated params
        self._do_autoplace()



    def showEvent(self, e):
        super().showEvent(e)
        if self._fit_pending:
            self._fit_pending = False
            self._fit_to_window()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # keep a "fit" feel when the dialog is resized, but don't fight the user
        # only auto-fit if they're near fit zoom already
        # (comment this out if you *never* want auto-adjust)
        # self._fit_to_window()
        self._render()

    def eventFilter(self, obj, event):
        # Ctrl+wheel zoom
        try:
            if obj is self._scroll.viewport():
                if event.type() == QEvent.Type.Wheel:
                    if bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                        delta = event.angleDelta().y()
                        if delta > 0:
                            self._set_zoom(self._zoom * 1.15)
                        elif delta < 0:
                            self._set_zoom(self._zoom / 1.15)
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def ap_centers(self) -> np.ndarray:
        if self._centers is None:
            return np.zeros((0, 2), dtype=np.int32)
        return self._centers

    # ---------- image prep ----------
    @staticmethod
    def _make_display_u8(img01: np.ndarray) -> np.ndarray:
        mono = img01 if img01.ndim == 2 else img01[..., 0]
        mono = np.clip(mono, 0.0, 1.0)

        lo = float(np.percentile(mono, 1.0))
        hi = float(np.percentile(mono, 99.5))
        if hi <= lo + 1e-8:
            hi = lo + 1e-3

        v = (mono - lo) / (hi - lo)
        v = np.clip(v, 0.0, 1.0)
        return (v * 255.0 + 0.5).astype(np.uint8)

    # ---------- zoom helpers ----------
    def _on_zoom_slider(self, value: int):
        z = float(value) / 100.0
        self._set_zoom(z)

    def _set_zoom(self, z: float):
        z = float(z)
        z = max(0.10, min(4.00, z))  # clamp 10%..400%
        self._zoom = z

        block = self.sld_zoom.blockSignals(True)
        try:
            self.sld_zoom.setValue(int(round(z * 100.0)))
        finally:
            self.sld_zoom.blockSignals(block)

        self.lbl_zoom.setText(f"{int(round(z * 100.0))}%")
        self._render()

    def _fit_to_window(self):
        # fit image into scroll viewport with a little padding
        vw = max(1, self._scroll.viewport().width() - 10)
        vh = max(1, self._scroll.viewport().height() - 10)
        if self._W <= 0 or self._H <= 0:
            return
        z = min(vw / float(self._W), vh / float(self._H))
        self._set_zoom(z)

    def _on_ap_params_changed(self):
        # Update internal params
        self._ap_size = int(self.spin_ap_size.value())
        self._ap_spacing = int(self.spin_ap_spacing.value())

        # Just redraw boxes (does not re-place points automatically)
        self._render()


    # ---------- drawing ----------
    def _render(self):
        u8 = self._base_u8
        # ✅ memmap/FITS-safe: QImage assumes row-major contiguous when we pass bytesPerLine=w
        if not u8.flags["C_CONTIGUOUS"]:
            u8 = np.ascontiguousarray(u8)

        h, w = u8.shape[:2]

        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        base_pm = QPixmap.fromImage(qimg)  # already detached above

        # scale to display zoom (keeps UI sane)
        zw = max(1, int(round(w * self._zoom)))
        zh = max(1, int(round(h * self._zoom)))
        pm = base_pm.scaled(
            zw, zh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # draw AP boxes in *display coords* so thickness doesn't scale
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        s_img = int(max(8, self._ap_size))
        s_disp = max(2, int(round(s_img * self._zoom)))
        half_disp = s_disp // 2

        pen = QPen(QColor(0, 255, 0), 2)  # constant on-screen thickness
        p.setPen(pen)

        if self._centers is not None and self._centers.size > 0:
            for cx, cy in self._centers.tolist():
                x = int(round(cx * self._zoom))
                y = int(round(cy * self._zoom))
                p.drawRect(int(x - half_disp), int(y - half_disp), int(s_disp), int(s_disp))

        p.end()

        self._pix.setPixmap(pm)
        self._pix.setFixedSize(pm.size())

    # ---------- actions ----------
    def _do_autoplace(self):
        from setiastro.saspro.ser_stacker import _autoplace_aps  # reuse exact logic
        self._centers = _autoplace_aps(self._ref, self._ap_size, self._ap_spacing, self._ap_min_mean)
        self._render()

    def _do_clear(self):
        self._centers = np.zeros((0, 2), dtype=np.int32)
        self._render()

    # ---------- mouse handling ----------
    def _on_mouse_press(self, ev):
        pm = self._pix.pixmap()
        if pm is None:
            return

        # display coords in the label
        dx = float(ev.position().x())
        dy = float(ev.position().y())

        # map to image coords
        ix = int(round(dx / max(1e-6, self._zoom)))
        iy = int(round(dy / max(1e-6, self._zoom)))

        # clamp to image bounds
        ix = max(0, min(self._W - 1, ix))
        iy = max(0, min(self._H - 1, iy))

        if ev.button() == Qt.MouseButton.LeftButton:
            self._add_point(ix, iy)
        elif ev.button() == Qt.MouseButton.RightButton:
            self._delete_nearest(ix, iy)

    def _add_point(self, x: int, y: int):
        s = int(max(8, self._ap_size))
        half = s // 2

        # ensure AP box fits fully
        x = max(half, min(self._W - 1 - half, x))
        y = max(half, min(self._H - 1 - half, y))

        if self._centers is None or self._centers.size == 0:
            self._centers = np.asarray([[x, y]], dtype=np.int32)
        else:
            self._centers = np.vstack([self._centers, np.asarray([[x, y]], dtype=np.int32)])
        self._render()

    def _delete_nearest(self, x: int, y: int):
        if self._centers is None or self._centers.size == 0:
            return

        pts = self._centers.astype(np.float32)
        d2 = (pts[:, 0] - float(x)) ** 2 + (pts[:, 1] - float(y)) ** 2
        j = int(np.argmin(d2))

        # radius in image pixels (so behavior is stable regardless of zoom)
        radius = max(10.0, float(self._ap_size) * 0.6)
        if float(d2[j]) <= radius * radius:
            self._centers = np.delete(self._centers, j, axis=0)
            self._render()

class QualityGraph(QWidget):
    """
    AS-style quality plot (sorted curve expected):
    - Curve: q[0] best ... q[N-1] worst
    - Vertical cutoff line at keep_k
    - Midrange horizontal line (min/max midpoint)
    - True median horizontal line labeled 'Med'
    - Click / drag adjusts keep line and emits keepChanged(k, N)
    """
    keepChanged = pyqtSignal(int, int)  # keep_k, total_N

    def __init__(self, parent=None):
        super().__init__(parent)
        self._q: np.ndarray | None = None
        self._keep_k: int | None = None
        self.setMinimumHeight(160)
        self._dragging = False

    def set_data(self, q: np.ndarray | None, keep_k: int | None = None):
        self._q = None if q is None else np.asarray(q, dtype=np.float32)
        self._keep_k = keep_k
        self.update()

    def _plot_rect(self):
        # room for labels
        return self.rect().adjusted(34, 10, -10, -22)

    def _x_to_keep_k(self, x: float) -> int | None:
        if self._q is None or self._q.size < 2:
            return None
        r = self._plot_rect()
        if r.width() <= 1:
            return None
        N = int(self._q.size)

        # clamp x to plot rect
        xx = max(float(r.left()), min(float(r.right()), float(x)))

        # map x back to index i in [0..N-1]
        t = (xx - float(r.left())) / float(max(1, r.width()))
        i = int(round(t * float(N - 1)))
        i = max(0, min(N - 1, i))

        # keep_k is count of frames kept => i=0 means keep 1, i=N-1 means keep N
        return int(i + 1)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        self._dragging = True
        k = self._x_to_keep_k(ev.position().x())
        if k is not None:
            self._keep_k = int(k)
            self.update()
            self.keepChanged.emit(int(k), int(self._q.size))  # type: ignore
        ev.accept()

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return super().mouseMoveEvent(ev)
        k = self._x_to_keep_k(ev.position().x())
        if k is not None:
            if self._keep_k != int(k):
                self._keep_k = int(k)
                self.update()
                self.keepChanged.emit(int(k), int(self._q.size))  # type: ignore
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            ev.accept()
            return
        return super().mouseReleaseEvent(ev)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        r = self._plot_rect()

        # frame
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawRect(r)

        if self._q is None or self._q.size < 2:
            p.setPen(QPen(QColor(160, 160, 160), 1))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Analyze to see quality graph")
            p.end()
            return

        q = self._q
        N = int(q.size)

        qmin = float(np.min(q))
        qmax = float(np.max(q))
        if qmax <= qmin + 1e-12:
            qmax = qmin + 1e-6

        def y_for(val: float) -> float:
            return r.bottom() - ((val - qmin) / (qmax - qmin)) * r.height()

        # ---- horizontal reference lines ----
        # 1) midrange (between min/max) - dashed
        qmid = qmin + 0.5 * (qmax - qmin)
        ymid = y_for(qmid)
        pen_mid = QPen(QColor(120, 120, 120), 1)
        pen_mid.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen_mid)
        p.drawLine(int(r.left()), int(ymid), int(r.right()), int(ymid))

        # 2) true median of q - dotted (or dash-dot)
        qmed = float(np.median(q))
        ymed = y_for(qmed)
        pen_med = QPen(QColor(160, 160, 160), 1)
        pen_med.setStyle(Qt.PenStyle.DotLine)
        p.setPen(pen_med)
        p.drawLine(int(r.left()), int(ymed), int(r.right()), int(ymed))

        # small "Med" label on the right of the median line
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawText(int(r.right()) - 34, int(ymed) - 2, "Med")

        # ---- curve ----
        p.setPen(QPen(QColor(0, 220, 0), 2))
        lastx = lasty = None
        for i in range(N):
            x = r.left() + (i / (N - 1)) * r.width()
            y = y_for(float(q[i]))
            if lastx is not None:
                p.drawLine(int(lastx), int(lasty), int(x), int(y))
            lastx, lasty = x, y

        # ---- cutoff line ----
        if self._keep_k is not None and N > 1:
            k = int(max(1, min(N, int(self._keep_k))))
            xcut = r.left() + ((k - 1) / (N - 1)) * r.width()
            p.setPen(QPen(QColor(255, 220, 0), 2))
            p.drawLine(int(xcut), int(r.top()), int(xcut), int(r.bottom()))

        # ---- labels ----
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawText(
            self.rect().adjusted(6, 0, 0, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            "Best",
        )
        p.drawText(
            self.rect().adjusted(0, 0, -6, 0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            "Worst",
        )

        # y labels: max, mid, median, min
        p.drawText(4, int(r.top()) + 10, f"{qmax:.3g}")
        p.drawText(4, int(ymid) + 4,    f"{qmid:.3g}")   # midrange
        p.drawText(4, int(ymed) + 4,    f"{qmed:.3g}")   # true median
        p.drawText(4, int(r.bottom()),  f"{qmin:.3g}")

        p.end()

class _AnalyzeWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, phase
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SERStackConfig, *, debayer: bool, to_rgb: bool, ref_mode: str, ref_count: int):
        super().__init__()
        self.cfg = cfg
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)
        self.ref_mode = ref_mode
        self.ref_count = int(ref_count)
        self._cancel = False
        self._worker_realign: _ReAlignWorker | None = None

    def run(self):
        try:
            def cb(done: int, total: int, phase: str):
                self.progress.emit(int(done), int(total), str(phase))

            ar = analyze_ser(
                self.cfg,
                debayer=self.debayer,
                to_rgb=self.to_rgb,
                bayer_pattern=getattr(self.cfg, "bayer_pattern", None),
                ref_mode=self.ref_mode,
                ref_count=self.ref_count,
                progress_cb=cb,
            )
            self.finished_ok.emit(ar)
        except Exception as e:
            msg = f"{e}\n\n{traceback.format_exc()}"
            self.failed.emit(msg)

class _StackWorker(QThread):
    progress = pyqtSignal(int, int, str)        # ✅ add this
    finished_ok = pyqtSignal(object, object)    # out(np.ndarray), diag(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: SERStackConfig, analysis: AnalyzeResult | None, *, debayer: bool, to_rgb: bool):
        super().__init__()
        self.cfg = cfg
        self.analysis = analysis
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)

    def run(self):
        try:
            print(f"tracking mode = {getattr(self.cfg, 'track_mode', 'planetary')}")
            def cb(done: int, total: int, phase: str):
                self.progress.emit(int(done), int(total), str(phase))

            out, diag = stack_ser(
                self.cfg.source,
                roi=self.cfg.roi,
                debayer=self.debayer,
                to_rgb=self.to_rgb,
                bayer_pattern=getattr(self.cfg, "bayer_pattern", None),  # ✅ add this
                keep_percent=float(getattr(self.cfg, "keep_percent", 20.0)),
                track_mode=str(getattr(self.cfg, "track_mode", "planetary")),
                surface_anchor=getattr(self.cfg, "surface_anchor", None),
                analysis=self.analysis,
                local_warp=True,
                progress_cb=cb,
                drizzle_scale=float(getattr(self.cfg, "drizzle_scale", 1.0)),
                drizzle_pixfrac=float(getattr(self.cfg, "drizzle_pixfrac", 0.80)),
                drizzle_kernel=str(getattr(self.cfg, "drizzle_kernel", "gaussian")),
                drizzle_sigma=float(getattr(self.cfg, "drizzle_sigma", 0.0)),
                keep_mask=getattr(self.cfg, "keep_mask", None),
            )


            self.finished_ok.emit(out, diag)
        except Exception as e:
            msg = f"{e}\n\n{traceback.format_exc()}"
            self.failed.emit(msg)


class _ReAlignWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, phase
    finished_ok = pyqtSignal(object)       # updated AnalyzeResult
    failed = pyqtSignal(str)

    def __init__(self, cfg: SERStackConfig, analysis: AnalyzeResult, *, debayer: bool, to_rgb: bool):
        super().__init__()
        self.cfg = cfg
        self.analysis = analysis
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)

    def run(self):
        try:
            def cb(done: int, total: int, phase: str):
                self.progress.emit(int(done), int(total), str(phase))

            from setiastro.saspro.ser_stacker import realign_ser  # you’ll add this below
            out_analysis = realign_ser(
                self.cfg,
                self.analysis,
                debayer=self.debayer,
                to_rgb=self.to_rgb,
                bayer_pattern=getattr(self.cfg, "bayer_pattern", None),
                progress_cb=cb,
            )
            self.finished_ok.emit(out_analysis)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class SERStackerDialog(QDialog):
    """
    Dedicated stacking UI (AutoStakkert-like direction):
    - Keeps viewer separate from stacking.
    - V1: track mode, keep %, uses ROI + optional surface anchor from viewer.
    - Later: alignment points (manual/auto), quality graph, drizzle, etc.
    """

    # Main app can connect this to "push to new view"
    stackProduced = pyqtSignal(object, object)  # out(np.ndarray), diag(dict)

    def __init__(
        self,
        parent=None,
        *,
        main,
        source_doc=None,
        ser_path: Optional[str] = None,   # ✅ typed + default
        source: Optional[SourceSpec] = None,
        roi=None,
        track_mode: str = "planetary",
        surface_anchor=None,
        debayer: bool = True,
        keep_percent: float = 20.0,
        bayer_pattern: Optional[str] = None,
        planet_min_val=0.02, planet_use_norm=False, planet_norm_hi_pct=99.5,
        planet_thresh_pct=92.0, planet_smooth_sigma=1.5, **kwargs
    ):
        super().__init__(parent)
        self.setWindowTitle("Planetary Stacker - Beta")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._bayer_pattern = bayer_pattern
        self._keep_mask = None  # np.ndarray bool shape (N,) or None

        self._planet_min_val = float(planet_min_val)
        self._planet_use_norm = bool(planet_use_norm)
        self._planet_norm_hi_pct = float(planet_norm_hi_pct)
        self._planet_thresh_pct = float(planet_thresh_pct)
        self._planet_smooth_sigma = float(planet_smooth_sigma)

        # ---- Normalize inputs ------------------------------------------------
        # If caller provided only `source`, treat string-source as ser_path too.
        if source is None:
            source = ser_path

        # If source is a single path string and ser_path is missing, fill it.
        if ser_path is None and isinstance(source, str) and source:
            ser_path = source

        if source is None:
            raise ValueError("SERStackerDialog requires source (path or list of paths).")

        self._main = main
        self._source = source
        self._source_doc = source_doc
        self.setMinimumWidth(980)     # or 1024 if you want it beefier
        self.resize(1040, 720)        # initial size (width, height)
        # IMPORTANT: now _ser_path is never empty for the common SER case
        self._ser_path = ser_path

        self._track_mode = track_mode
        self._roi = roi
        self._surface_anchor = surface_anchor
        self._debayer = bool(debayer)
        self._keep_percent = float(keep_percent)

        self._analysis = None
        self._worker_analyze = None
        self._worker = None
        self._last_out = None
        self._last_diag = None
        try:
            if isinstance(self._source, (list, tuple)):
                self._append_log(f"Source: sequence ({len(self._source)} frames)  first={self._source[0]}")
            else:
                self._append_log(f"Source: {self._source}")
        except Exception:
            pass

        self._build_ui()

        # defaults
        self.cmb_track.setCurrentText(
            "Planetary" if track_mode == "planetary" else ("Surface" if track_mode == "surface" else "Off")
        )
        self.spin_keep.setValue(float(keep_percent))
        self.chk_debayer.setChecked(bool(debayer))
        self._update_anchor_warning()
        try:
            if isinstance(self._source, (list, tuple)):
                self._append_log(f"Source: sequence ({len(self._source)} frames)")
            else:
                self._append_log(f"Source: {self._source}")
        except Exception:
            self._append_log("Source: (unknown)")
        self._append_log(f"ROI: {roi if roi is not None else '(full frame)'}")
        if track_mode == "surface":
            self._append_log(f"Surface anchor (ROI-space): {surface_anchor}")

    # ---------------- UI ----------------
    def _build_ui(self):
        # ----- Dialog layout -----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Split into two columns so we don't exceed monitor height:
        # Left: settings/analyze/actions/progress
        # Right: quality graph + log
        cols = QHBoxLayout()
        cols.setSpacing(10)
        outer.addLayout(cols, 1)

        left = QVBoxLayout()
        left.setSpacing(8)
        right = QVBoxLayout()
        right.setSpacing(8)

        cols.addLayout(left, 0)
        cols.addLayout(right, 1)

        # =========================
        # LEFT COLUMN
        # =========================

        # --- Stack Settings ---
        gb = QGroupBox("Stack Settings", self)
        form = QFormLayout(gb)

        self.cmb_track = QComboBox(self)
        self.cmb_track.addItems(["Planetary", "Surface", "Off"])

        self.spin_keep = QDoubleSpinBox(self)
        self.spin_keep.setRange(0.1, 100.0)
        self.spin_keep.setDecimals(1)
        self.spin_keep.setSingleStep(1.0)
        self.spin_keep.setValue(20.0)

        self.chk_debayer = QCheckBox("Debayer (Bayer SER)", self)
        self.chk_debayer.setChecked(True)

        self.lbl_anchor = QLabel("", self)
        self.lbl_anchor.setWordWrap(True)

        form.addRow("Tracking", self.cmb_track)
        form.addRow("Keep %", self.spin_keep)
        form.addRow("", self.chk_debayer)
        form.addRow("Surface anchor", self.lbl_anchor)

        left.addWidget(gb, 0)

        # --- Drizzle ---
        gbD = QGroupBox("Drizzle", self)
        fD = QFormLayout(gbD)

        self.spin_pixfrac = QDoubleSpinBox(self)
        self.spin_pixfrac.setRange(0.30, 1.00)
        self.spin_pixfrac.setDecimals(2)
        self.spin_pixfrac.setSingleStep(0.05)
        self.spin_pixfrac.setValue(0.80)

        self.cmb_kernel = QComboBox(self)
        self.cmb_kernel.addItems(["Gaussian", "Circle", "Square"])

        self.spin_sigma = QDoubleSpinBox(self)
        self.spin_sigma.setRange(0.00, 10.00)
        self.spin_sigma.setDecimals(2)
        self.spin_sigma.setSingleStep(0.05)
        self.spin_sigma.setValue(0.00)   # 0 = auto
        self.spin_sigma.setToolTip("Gaussian sigma in output pixels (0 = auto from pixfrac)")

        # scale row: combo + info button in same row
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)

        self.cmb_drizzle = QComboBox(self)
        self.cmb_drizzle.addItems(["Off (1x)", "1.5x", "2x"])

        self.btn_drizzle_info = QToolButton(self)
        self.btn_drizzle_info.setText("?")
        self.btn_drizzle_info.setToolTip("Drizzle info")
        self.btn_drizzle_info.setFixedSize(22, 22)

        scale_row.addWidget(self.cmb_drizzle, 1)
        scale_row.addWidget(self.btn_drizzle_info, 0)

        scale_row_w = QWidget(self)
        scale_row_w.setLayout(scale_row)

        fD.addRow("Scale", scale_row_w)
        fD.addRow("Pixfrac", self.spin_pixfrac)
        fD.addRow("Kernel", self.cmb_kernel)
        fD.addRow("Sigma", self.spin_sigma)

        def _sync_drizzle_ui():
            t = self.cmb_drizzle.currentText()
            off = "Off" in t
            self.spin_pixfrac.setEnabled(not off)
            self.cmb_kernel.setEnabled(not off)

            k = self.cmb_kernel.currentText().lower()
            is_gauss = ("gaussian" in k)
            self.spin_sigma.setEnabled((not off) and is_gauss)

            # sensible defaults when enabling drizzle
            if off:
                return
            if "1.5" in t:
                if abs(self.spin_pixfrac.value() - 0.80) < 1e-6 or self.spin_pixfrac.value() in (0.70,):
                    self.spin_pixfrac.setValue(0.80)
            elif "2" in t:
                if abs(self.spin_pixfrac.value() - 0.70) < 1e-6 or self.spin_pixfrac.value() in (0.80,):
                    self.spin_pixfrac.setValue(0.70)

        self.cmb_drizzle.currentIndexChanged.connect(lambda _=None: _sync_drizzle_ui())
        self.cmb_kernel.currentIndexChanged.connect(lambda _=None: _sync_drizzle_ui())
        _sync_drizzle_ui()

        def _show_drizzle_info():
            QMessageBox.information(
                self,
                "Drizzle Info",
                "Drizzle increases output resolution by resampling and re-depositing pixels.\n\n"
                "Compute cost:\n"
                "• 1.5× drizzle ≈ 225% compute (2.25×)\n"
                "• 2× drizzle ≈ 400% compute (4×)\n\n"
                "Pixfrac (drop shrink):\n"
                "• Controls how large each input pixel’s “drop” is in the output grid.\n"
                "• Lower pixfrac = tighter drops (sharper, but can create gaps/noise).\n"
                "• Higher pixfrac = smoother coverage (less noise, slightly softer).\n\n"
                "When drizzle helps:\n"
                "• Best when you are under-sampled and you have good alignment / many frames.\n"
                "• Helps most with stable seeing and lots of usable frames.\n\n"
                "When drizzle may NOT help:\n"
                "• If you’re already well-sampled (common around f/10–f/20 depending on pixel size),\n"
                "  gains can be minimal.\n"
                "• If seeing is very poor, drizzle often just magnifies blur/noise.\n\n"
                "Tip: Start with 1.5× and pixfrac ~0.8. If coverage looks sparse/noisy, increase pixfrac."
            )

        self.btn_drizzle_info.clicked.connect(_show_drizzle_info)

        left.addWidget(gbD, 0)

        # --- Analyze settings (no graph in left column anymore) ---
        gbA = QGroupBox("Analyze", self)
        fA = QFormLayout(gbA)

        self.cmb_ref = QComboBox(self)
        self.cmb_ref.addItems(["Best frame", "Best stack (N)"])

        self.spin_refN = QSpinBox(self)
        self.spin_refN.setRange(2, 200)
        self.spin_refN.setValue(10)

        self.spin_ap_min = QDoubleSpinBox(self)
        self.spin_ap_min.setRange(0.0, 1.0)
        self.spin_ap_min.setDecimals(3)
        self.spin_ap_min.setSingleStep(0.005)
        self.spin_ap_min.setValue(0.03)
        fA.addRow("AP min mean (0..1)", self.spin_ap_min)

        self.btn_edit_aps = QPushButton("(2) Edit APs…", self)
        self.btn_edit_aps.setEnabled(False)
        fA.addRow("", self.btn_edit_aps)

        self.spin_ap_size = QSpinBox(self)
        self.spin_ap_size.setRange(16, 256)
        self.spin_ap_size.setSingleStep(8)
        self.spin_ap_size.setValue(64)

        self.spin_ap_spacing = QSpinBox(self)
        self.spin_ap_spacing.setRange(8, 256)
        self.spin_ap_spacing.setSingleStep(8)
        self.spin_ap_spacing.setValue(48)

        fA.addRow("Reference", self.cmb_ref)
        fA.addRow("Ref stack N", self.spin_refN)

        self.cmb_ap_scale = QComboBox(self)
        self.cmb_ap_scale.addItems(["Single", "Multi-scale (2× / 1× / ½×)"])
        fA.addRow("AP scale", self.cmb_ap_scale)

        self.chk_ssd_bruteforce = QCheckBox("SSD refine: brute force (slower, can rescue tough data)", self)
        self.chk_ssd_bruteforce.setChecked(False)
        fA.addRow("", self.chk_ssd_bruteforce)

        fA.addRow("AP size (px)", self.spin_ap_size)
        fA.addRow("AP spacing (px)", self.spin_ap_spacing)

        left.addWidget(gbA, 0)

        # --- Action buttons ---
        row = QHBoxLayout()
        self.btn_analyze = QPushButton("(1) Analyze", self)
        self.btn_analyze.setEnabled(True)
        self.btn_blink = QPushButton("(3) Blink Keepers", self)   # ✅ new
        self.btn_stack = QPushButton("(4) Stack Now", self)
        self.btn_close = QPushButton("Close", self)

        row.addWidget(self.btn_analyze)
        row.addStretch(1)
        row.addWidget(self.btn_blink) 
        row.addStretch(1)
        row.addWidget(self.btn_stack)
        row.addWidget(self.btn_close)

        left.addLayout(row, 0)

        # --- Progress ---
        self.prog = QProgressBar(self)
        self.prog.setRange(0, 0)
        self.prog.setVisible(False)
        left.addWidget(self.prog, 0)

        self.lbl_prog = QLabel("", self)
        self.lbl_prog.setStyleSheet("color:#aaa;")
        self.lbl_prog.setVisible(False)
        left.addWidget(self.lbl_prog, 0)

        left.addStretch(1)

        # =========================
        # RIGHT COLUMN
        # =========================

        # --- Quality Graph ---
        gbQ = QGroupBox("Quality", self)
        vQ = QVBoxLayout(gbQ)
        vQ.setContentsMargins(8, 8, 8, 8)
        vQ.setSpacing(6)

        self.graph = QualityGraph(self)
        self.graph.setMinimumHeight(180)
        self.graph.setMinimumWidth(480)  # keeps the right column from scrunching

        # small hint under the graph
        self.lbl_graph_hint = QLabel("Tip: click the graph to set Keep cutoff.", self)
        self.lbl_graph_hint.setStyleSheet("color:#888; font-size:11px;")
        self.lbl_graph_hint.setWordWrap(True)

        vQ.addWidget(self.graph, 1)
        vQ.addWidget(self.lbl_graph_hint, 0)

        right.addWidget(gbQ, 1)

        # --- Log ---
        gbL = QGroupBox("Log", self)
        vL = QVBoxLayout(gbL)
        vL.setContentsMargins(8, 8, 8, 8)

        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        self.log.setPlaceholderText("Log…")

        vL.addWidget(self.log, 1)
        right.addWidget(gbL, 1)

        # =========================
        # Signals / wiring
        # =========================

        self.btn_close.clicked.connect(self.close)
        self.btn_stack.clicked.connect(self._start_stack)
        self.btn_blink.clicked.connect(self._blink_keepers)
        self.cmb_track.currentIndexChanged.connect(self._update_anchor_warning)
        self.btn_analyze.clicked.connect(self._start_analyze)
        self.btn_edit_aps.clicked.connect(self._edit_aps)
        self.spin_keep.valueChanged.connect(self._on_keep_changed)

        # Keep % edits update the cutoff line
        self.spin_keep.valueChanged.connect(self._update_graph_cutoff)

        # Clicking on the graph updates Keep %
        def _on_graph_keep_changed(k: int, total: int):
            total = max(1, int(total))
            k = max(1, min(total, int(k)))
            pct = 100.0 * float(k) / float(total)

            block = self.spin_keep.blockSignals(True)
            try:
                self.spin_keep.setValue(float(pct))
            finally:
                self.spin_keep.blockSignals(block)

            # update graph line (using current analysis ordering)
            self._update_graph_cutoff()
            self._append_log(f"Keep set from graph: {pct:.1f}% ({k}/{total})")

        self.graph.keepChanged.connect(_on_graph_keep_changed)

    # ---------------- helpers ----------------
    def _edit_aps(self):
        if self._analysis is None:
            return

        try:
            dlg = APEditorDialog(
                self,
                ref_img01=self._analysis.ref_image,
                ap_size=int(self.spin_ap_size.value()),
                ap_spacing=int(self.spin_ap_spacing.value()),
                ap_min_mean=float(self.spin_ap_min.value()),
                initial_centers=getattr(self._analysis, "ap_centers", None),
            )
            if dlg.exec() == QDialog.DialogCode.Accepted:
                centers = dlg.ap_centers()
                self._analysis.ap_centers = centers

                # ✅ pull size/spacing changes from the editor back into the main UI
                try:
                    self.spin_ap_size.setValue(int(dlg.ap_size()))
                    self.spin_ap_spacing.setValue(int(dlg.ap_spacing()))
                    self.spin_ap_min.setValue(float(dlg.ap_min_mean()))
                except Exception:
                    pass

                self._append_log(
                    f"APs set: {int(centers.shape[0])} points   "
                    f"(size={int(self.spin_ap_size.value())}, spacing={int(self.spin_ap_spacing.value())})"
                )

                # Recompute alignment only (no full analyze)
                cfg = self._make_cfg()

                self.lbl_prog.setVisible(True)
                self.prog.setVisible(True)
                self.prog.setRange(0, 100)
                self.prog.setValue(0)
                self.lbl_prog.setText("Re-aligning with APs…")
                self.btn_stack.setEnabled(False)
                self.btn_analyze.setEnabled(False)
                self.btn_edit_aps.setEnabled(False)

                self._worker_realign = _ReAlignWorker(cfg, self._analysis, debayer=bool(self.chk_debayer.isChecked()), to_rgb=False)
                self._worker_realign.progress.connect(self._on_analyze_progress)  # reuse progress UI
                self._worker_realign.finished_ok.connect(self._on_realign_ok)
                self._worker_realign.failed.connect(self._on_analyze_fail)
                self._worker_realign.start()
            else:
                self._append_log("AP edit cancelled.")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "AP Editor Error", f"{e}\n\n{tb}")
            self._append_log(f"AP editor failed: {e}")
            self._append_log(tb)

    def _on_realign_ok(self, ar: AnalyzeResult):
        self._analysis = ar

        self.prog.setVisible(False)
        self.lbl_prog.setVisible(False)

        self.btn_stack.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_edit_aps.setEnabled(True)
        self.btn_close.setEnabled(True)

        self._append_log("Re-align done (dx/dy/conf updated from APs).")



    def _append_log(self, s: str):
        try:
            self.log.append(s)
        except Exception:
            pass

    def _track_mode_value(self) -> str:
        t = self.cmb_track.currentText().strip().lower()
        if t.startswith("planet"):
            return "planetary"
        if t.startswith("surface"):
            return "surface"
        return "off"

    def _update_anchor_warning(self):
        mode = self._track_mode_value()
        if mode != "surface":
            self.lbl_anchor.setText("(not used)")
            self.lbl_anchor.setStyleSheet("color:#888;")
            return

        if self._surface_anchor is None:
            self.lbl_anchor.setText("REQUIRED (set in SER Viewer with Ctrl+Shift+drag)")
            self.lbl_anchor.setStyleSheet("color:#c66;")
            return

        x, y, w, h = [int(v) for v in self._surface_anchor]

        # Always show ROI-space (that’s what the tracker uses)
        txt = f"✅ ROI-space: x={x}, y={y}, w={w}, h={h}"

        # If an ROI is set, also show full-frame coords for sanity/debug
        if self._roi is not None:
            rx, ry, rw, rh = [int(v) for v in self._roi]
            fx = rx + x
            fy = ry + y
            txt += f"   |   Full-frame: x={fx}, y={fy}, w={w}, h={h}"

        self.lbl_anchor.setText(txt)
        self.lbl_anchor.setStyleSheet("color:#4a4;")


    # ---------------- actions ----------------
    def _start_analyze(self):
        mode = self._track_mode_value()
        if mode == "surface" and self._surface_anchor is None:
            self._append_log("Surface mode requires an anchor. Set it in the viewer (Ctrl+Shift+drag).")
            return

        ref_mode = "best_stack" if self.cmb_ref.currentText().lower().startswith("best stack") else "best_frame"
        refN = int(self.spin_refN.value()) if ref_mode == "best_stack" else 1

        cfg = self._make_cfg()

        self.btn_analyze.setEnabled(False)
        self.btn_stack.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.lbl_prog.setVisible(True)
        self.lbl_prog.setText("Analyzing…")
        self.prog.setVisible(True)
        self.prog.setRange(0, 100)
        self.prog.setValue(0)

        self._worker_analyze = _AnalyzeWorker(
            cfg,
            debayer=bool(self.chk_debayer.isChecked()),
            to_rgb=False,
            ref_mode=ref_mode,
            ref_count=refN,
        )
        self._worker_analyze.finished_ok.connect(self._on_analyze_ok)
        self._worker_analyze.failed.connect(self._on_analyze_fail)
        self._worker_analyze.progress.connect(self._on_analyze_progress)
        self._worker_analyze.start()


    def _on_analyze_progress(self, done: int, total: int, phase: str):
        total = max(1, int(total))
        done = max(0, min(total, int(done)))
        pct = int(round(100.0 * done / total))
        self.prog.setRange(0, 100)
        self.prog.setValue(pct)
        self.lbl_prog.setText(f"{phase}: {done}/{total} ({pct}%)")


    def _on_analyze_ok(self, ar: AnalyzeResult):
        self._analysis = ar

        self.prog.setVisible(False)
        self.lbl_prog.setVisible(False)

        self.btn_analyze.setEnabled(True)
        self.btn_stack.setEnabled(True)
        self.btn_blink.setEnabled(True)
        self.btn_close.setEnabled(True)

        self._append_log(f"Analyze done. frames={ar.frames_total}  track={ar.track_mode}")
        self._append_log(f"Ref: {ar.ref_mode} (N={ar.ref_count})")

        # update graph (time-order) + cutoff marker based on keep%
        k = int(round(ar.frames_total * (float(self.spin_keep.value()) / 100.0)))
        k = max(1, min(ar.frames_total, k))
        q_sorted = ar.quality[ar.order]
        self.graph.set_data(q_sorted, keep_k=k)
        self.btn_edit_aps.setEnabled(True)

    def _on_analyze_fail(self, msg: str):
        self.prog.setVisible(False)
        self.lbl_prog.setVisible(False)

        self.btn_analyze.setEnabled(True)
        had_analysis = self._analysis is not None and getattr(self._analysis, "ref_image", None) is not None
        self.btn_stack.setEnabled(bool(had_analysis))
        self.btn_edit_aps.setEnabled(bool(had_analysis))
        self.btn_close.setEnabled(True)

        self._append_log("ANALYZE FAILED:")
        self._append_log(msg)

    def _update_graph_cutoff(self):
        if self._analysis is None:
            return
        n = int(self._analysis.frames_total)
        k = int(round(n * (float(self.spin_keep.value()) / 100.0)))
        k = max(1, min(n, k))
        q_sorted = self._analysis.quality[self._analysis.order]
        self.graph.set_data(q_sorted, keep_k=k)

    def _blink_keepers(self):
        if self._analysis is None:
            return

        N = int(self._analysis.frames_total)
        keep_k = int(round(N * (float(self.spin_keep.value()) / 100.0)))
        keep_k = max(1, min(N, keep_k))

        cfg = self._make_cfg()
        cfg.keep_mask = getattr(cfg, "keep_mask", None)

        try:
            dlg = BlinkKeepersDialog(
                self,
                cfg=cfg,
                analysis=self._analysis,
                debayer=bool(self.chk_debayer.isChecked()),
                to_rgb=False,
                keep_k=keep_k,
            )
            if dlg.exec() == QDialog.DialogCode.Accepted:
                km = dlg.keep_mask_all_frames()
                self._keep_mask = km

                # log stats
                kept = int(np.count_nonzero(km))
                self._append_log(f"Blink Keepers: kept {kept}/{N} after manual rejects.")
            else:
                self._append_log("Blink Keepers cancelled (no changes).")
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Blink Keepers Error", f"{e}\n\n{tb}")
            self._append_log(f"Blink Keepers failed: {e}")
            self._append_log(tb)


    def _make_cfg(self) -> SERStackConfig:
        scale_text = self.cmb_drizzle.currentText()
        if "1.5" in scale_text:
            drizzle_scale = 1.5
        elif "2" in scale_text:
            drizzle_scale = 2.0
        else:
            drizzle_scale = 1.0

        drizzle_kernel = self.cmb_kernel.currentText().strip().lower()  # gaussian/circle/square

        return SERStackConfig(
            source=self._source,
            roi=self._roi,
            track_mode=self._track_mode_value(),
            surface_anchor=self._surface_anchor,
            keep_percent=float(self.spin_keep.value()),
            bayer_pattern=self._bayer_pattern,
            keep_mask=getattr(self, "_keep_mask", None),

            ap_size=int(self.spin_ap_size.value()),
            ap_spacing=int(self.spin_ap_spacing.value()),
            ap_min_mean=float(self.spin_ap_min.value()),
            ap_multiscale=(self.cmb_ap_scale.currentIndex() == 1),
            ssd_refine_bruteforce=bool(
                getattr(self, "chk_ssd_bruteforce", None) and self.chk_ssd_bruteforce.isChecked()
            ),

            # ✅ NEW: planetary centroid knobs (add UI controls or set defaults)
            planet_smooth_sigma=self._planet_smooth_sigma,
            planet_thresh_pct=self._planet_thresh_pct,
            planet_min_val=self._planet_min_val,
            planet_use_norm=self._planet_use_norm,
            planet_norm_hi_pct=self._planet_norm_hi_pct,

            # drizzle
            drizzle_scale=float(drizzle_scale),
            drizzle_pixfrac=float(self.spin_pixfrac.value()),
            drizzle_kernel=str(drizzle_kernel),
            drizzle_sigma=float(self.spin_sigma.value()),
        )

    def _on_keep_changed(self, _v):
        self._keep_mask = None

    def _start_stack(self):
        mode = self._track_mode_value()
        if mode == "surface" and self._surface_anchor is None:
            self._append_log("Surface mode requires an anchor. Set it in the viewer (Ctrl+Shift+drag).")
            return

        cfg = self._make_cfg()
        cfg.keep_mask = self._keep_mask
        debayer = bool(self.chk_debayer.isChecked())

        self.btn_stack.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_edit_aps.setEnabled(False)
        scale_text = self.cmb_drizzle.currentText()
        if "1.5" in scale_text:
            drizzle_scale = 1.5
        elif "2" in scale_text:
            drizzle_scale = 2.0
        else:
            drizzle_scale = 1.0

        drizzle_kernel = self.cmb_kernel.currentText().strip().lower()  # "gaussian"/"circle"/"square"
        drizzle_pixfrac = float(self.spin_pixfrac.value())
        drizzle_sigma = float(self.spin_sigma.value())
        self.lbl_prog.setVisible(True)
        self.prog.setVisible(True)
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        self.lbl_prog.setText("Stack: 0%")

        self._append_log("Stacking...")

        self._worker = _StackWorker(cfg, analysis=self._analysis, debayer=debayer, to_rgb=False)
        self._worker.progress.connect(self._on_analyze_progress)   # ✅ reuse your progress handler
        self._worker.finished_ok.connect(self._on_stack_ok)
        self._worker.failed.connect(self._on_stack_fail)
        self._worker.start()

    def _on_stack_ok(self, out, diag):
        self._last_out = out
        self._last_diag = diag

        self.prog.setVisible(False)
        self.lbl_prog.setVisible(False)

        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_edit_aps.setEnabled(True)

        self._append_log(f"Done. Kept {diag.get('frames_kept')} / {diag.get('frames_total')}")
        self._append_log(f"Track: {diag.get('track_mode')}  ROI: {diag.get('roi_used')}")

        # ✅ Create the new stacked document (GUI thread)
        newdoc = _push_as_new_doc(
            self._main,
            self._source_doc,
            out,
            title_suffix="_stack",
            source="Planetary Stacker",
            source_path=self._source,
        )

        # Optional: stash diag on the document metadata (handy later)
        if newdoc is not None:
            try:
                md = getattr(newdoc, "metadata", None)
                if md is None:
                    md = {}
                    setattr(newdoc, "metadata", md)
                md["ser_stack_diag"] = diag
            except Exception:
                pass

        # Keep emitting too (so other callers can hook it)
        self.stackProduced.emit(out, diag)


    def _on_stack_fail(self, msg: str):
        self.prog.setVisible(False)
        self.btn_stack.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self._append_log("FAILED:")
        self._append_log(msg)

class BlinkKeepersDialog(QDialog):
    """
    Blink through the frames currently selected to keep, allow user to reject any.
    Returns a keep_mask (bool) for ALL frames, True=keep.
    """
    def __init__(self, parent=None, *, cfg: SERStackConfig, analysis: AnalyzeResult,
                 debayer: bool, to_rgb: bool, keep_k: int):
        super().__init__(parent)
        self.setWindowTitle("Blink Keepers")
        self.resize(1000, 750)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()  # so keypress works immediately
        self.cfg = cfg
        self.analysis = analysis
        self.debayer = bool(debayer)
        self.to_rgb = bool(to_rgb)

        self.N = int(analysis.frames_total)
        keep_k = max(1, min(self.N, int(keep_k)))

        # keeper frame indices in original frame space
        self.keepers = np.asarray(analysis.order[:keep_k], dtype=np.int32)

        # rejection only over the keeper list
        self.rejected = np.zeros((self.keepers.size,), dtype=bool)

        # ---- UI ----
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        self.lbl = QLabel(self)
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setStyleSheet("background:#111;")
        self.lbl.setMinimumHeight(480)
        outer.addWidget(self.lbl, 1)
        # --- Instructions / shortcuts ---
        self.lbl_help = QLabel(self)
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setStyleSheet(
            "color:#9aa; background:#151515; border:1px solid #2a2a2a;"
            "border-radius:6px; padding:6px; font-size:11px;"
        )
        self.lbl_help.setText(
            "Shortcuts: "
            "←/→ (or ↑/↓) = Prev/Next   |   PgUp/PgDn = Prev/Next\n"
            "R or Space = Toggle Reject + Next   |   Backspace = Toggle Reject + Prev\n"
            "Esc = Cancel   |   Enter = OK"
        )
        outer.addWidget(self.lbl_help, 0)
        info_row = QHBoxLayout()
        self.lbl_info = QLabel("", self)
        self.lbl_info.setStyleSheet("color:#bbb;")
        self.lbl_info.setWordWrap(True)
        info_row.addWidget(self.lbl_info, 1)

        self.btn_toggle = QPushButton("Reject", self)
        self.btn_toggle.setCheckable(True)
        info_row.addWidget(self.btn_toggle, 0)

        outer.addLayout(info_row)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev", self)
        self.btn_next = QPushButton("Next ▶", self)

        self.sld = QSlider(Qt.Orientation.Horizontal, self)
        self.sld.setRange(0, max(0, self.keepers.size - 1))
        self.sld.setValue(0)

        self.lbl_pos = QLabel("", self)
        self.lbl_pos.setStyleSheet("color:#aaa; min-width:90px;")

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.sld, 1)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_pos)
        outer.addLayout(nav)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("OK", self)
        self.btn_cancel = QPushButton("Cancel", self)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        outer.addLayout(btns)
        # --- Display controls (preview only) ---
        disp_row = QHBoxLayout()

        self.sld_bright = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_bright.setRange(-100, 100)   # percent-ish shift
        self.sld_bright.setValue(0)
        self.sld_bright.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sld_bright.setToolTip("Brightness (preview only)")

        self.lbl_bright = QLabel("Bright: 0", self)
        self.lbl_bright.setStyleSheet("color:#aaa; min-width:90px;")

        disp_row.addWidget(QLabel("Preview", self))
        disp_row.addSpacing(6)
        disp_row.addWidget(self.sld_bright, 1)
        disp_row.addWidget(self.lbl_bright, 0)

        outer.addLayout(disp_row)

        # ---- signals ----
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))
        self.sld.valueChanged.connect(self._show_index)
        self.btn_toggle.clicked.connect(lambda: self._toggle_reject_and_advance(+1))
        self.sld_bright.valueChanged.connect(self._on_brightness_changed)


        # ---- load source ----
        from setiastro.saspro.imageops.serloader import open_planetary_source
        self.src = open_planetary_source(
            self.cfg.source,
            cache_items=20,
        )
        self._debayer = bool(debayer)
        self._bayer_pattern = getattr(self.cfg, "bayer_pattern", None)
        self._force_rgb = True
        self.lbl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sld.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._preview_brightness = 0.0  # -1..+1

        self._show_index(0)

    def _on_brightness_changed(self, v: int):
        # map [-100..100] -> [-1..1]
        self._preview_brightness = float(int(v)) / 100.0
        self.lbl_bright.setText(f"Bright: {int(v)}")
        self._show_index(self.sld.value())


    def _toggle_reject_at(self, idx: int):
        if 0 <= idx < self.rejected.size:
            self.rejected[idx] = ~self.rejected[idx]
            self._update_labels()

    def _toggle_reject_and_advance(self, step: int = +1):
        i = int(self.sld.value())
        if self.keepers.size <= 0:
            return

        self._toggle_reject_at(i)

        # advance (clamped)
        j = i + int(step)
        j = max(0, min(int(self.keepers.size) - 1, j))
        self.sld.setValue(j)

    def keyPressEvent(self, e):
        k = e.key()
        mods = e.modifiers()

        # ignore if user is holding Ctrl/Alt/Meta (don’t fight standard shortcuts)
        if mods & (Qt.KeyboardModifier.ControlModifier |
                Qt.KeyboardModifier.AltModifier |
                Qt.KeyboardModifier.MetaModifier):
            super().keyPressEvent(e)
            return

        if k in (Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self._step(+1)
            e.accept()
            return

        if k in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self._step(-1)
            e.accept()
            return

        # R toggles reject and moves to next
        if k == Qt.Key.Key_R:
            self._toggle_reject_and_advance(+1)
            e.accept()
            return

        # Space does the same (nice for rapid triage)
        if k == Qt.Key.Key_Space:
            self._toggle_reject_and_advance(+1)
            e.accept()
            return

        # Optional: backspace toggles and moves back
        if k == Qt.Key.Key_Backspace:
            self._toggle_reject_and_advance(-1)
            e.accept()
            return
        if k == Qt.Key.Key_Escape:
            self.reject()
            e.accept()
            return
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            e.accept()
            return
        super().keyPressEvent(e)


    def _step(self, d: int):
        i = int(self.sld.value()) + int(d)
        i = max(0, min(int(self.keepers.size) - 1, i))
        self.sld.setValue(i)

    def _toggle_reject_current(self, checked: bool):
        i = int(self.sld.value())
        if 0 <= i < self.rejected.size:
            self.rejected[i] = bool(checked)
            self._update_labels()

    def _update_labels(self):
        i = int(self.sld.value())
        fi = int(self.keepers[i]) if self.keepers.size else 0
        q = float(self.analysis.quality[fi]) if self.analysis.quality is not None else 0.0

        is_rej = bool(self.rejected[i]) if self.rejected.size else False

        self.lbl_pos.setText(f"{i+1}/{int(self.keepers.size)}")
        self.lbl_info.setText(
            f"Keeper #{i+1}  |  Frame index: {fi}  |  Quality: {q:.6g}  |  "
            f"{'REJECTED' if is_rej else 'KEEP'}"
        )

        # ✅ colorize text when rejected
        if is_rej:
            self.lbl_info.setStyleSheet("color:#f66; font-weight:600;")   # red
            self.lbl_pos.setStyleSheet("color:#f66; min-width:90px;")
            # optional: make the button look “danger”
            self.btn_toggle.setStyleSheet("background:#3a1111; color:#f66;")
        else:
            self.lbl_info.setStyleSheet("color:#bbb;")
            self.lbl_pos.setStyleSheet("color:#aaa; min-width:90px;")
            self.btn_toggle.setStyleSheet("")  # back to default

        block = self.btn_toggle.blockSignals(True)
        try:
            self.btn_toggle.setChecked(is_rej)
            self.btn_toggle.setText("Un-reject" if is_rej else "Reject")
        finally:
            self.btn_toggle.blockSignals(block)


    @staticmethod
    def _disp_u8(mono01: np.ndarray, *, brightness: float = 0.0) -> np.ndarray:
        """
        Preview stretch for blink:
        - percentile stretch (1..99.5)
        - brightness shifts the window up/down (preview only)
          brightness in [-1..1] => shifts by ~10% of window size
        """
        mono = np.asarray(mono01, dtype=np.float32)
        mono = np.clip(mono, 0.0, 1.0)

        lo = float(np.percentile(mono, 1.0))
        hi = float(np.percentile(mono, 99.5))
        if hi <= lo + 1e-8:
            hi = lo + 1e-3

        # --- brightness: shift the stretch window ---
        b = float(np.clip(brightness, -1.0, 1.0))
        span = (hi - lo)

        # shift by up to 10% of span (tweakable)
        shift = -b * 0.50 * span
        lo2 = lo + shift
        hi2 = hi + shift

        # keep sane bounds
        lo2 = float(np.clip(lo2, 0.0, 1.0))
        hi2 = float(np.clip(hi2, lo2 + 1e-6, 1.0))

        v = (mono - lo2) / (hi2 - lo2)
        v = np.clip(v, 0.0, 1.0)

        return (v * 255.0 + 0.5).astype(np.uint8)

    
    def _show_index(self, i: int):
        if self.keepers.size == 0:
            return
        i = int(max(0, min(int(self.keepers.size) - 1, int(i))))
        fi = int(self.keepers[i])

        roi = getattr(self.cfg, "roi", None)
        img = self.src.get_frame(
            fi,
            roi=roi,
            debayer=bool(self._debayer),
            to_float01=True,
            force_rgb=bool(self._force_rgb),
            bayer_pattern=getattr(self, "_bayer_pattern", None),
        ).astype(np.float32, copy=False)

        # ✅ apply analyze global alignment (same as stack_ser does first)
        gdx = float(self.analysis.dx[int(fi)]) if (getattr(self.analysis, "dx", None) is not None) else 0.0
        gdy = float(self.analysis.dy[int(fi)]) if (getattr(self.analysis, "dy", None) is not None) else 0.0
        img = _shift_image(img, gdx, gdy)

        # display mono channel
        if img.ndim == 3:
            img = img[..., 0]

        u8 = self._disp_u8(img, brightness=getattr(self, "_preview_brightness", 0.0))


        # ✅ memmap/FITS-safe: guarantee tight row stride for bytesPerLine=w
        if not u8.flags["C_CONTIGUOUS"]:
            u8 = np.ascontiguousarray(u8)

        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pm = QPixmap.fromImage(qimg)

        self.lbl.setPixmap(pm.scaled(
            self.lbl.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        self._update_labels()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show_index(self.sld.value())

    def keep_mask_all_frames(self) -> np.ndarray:
        """
        Convert keeper+rejected into a full N-length keep mask.
        """
        km = np.zeros((self.N,), dtype=bool)
        km[self.keepers] = True
        # turn off rejected keepers
        if self.keepers.size:
            km[self.keepers[self.rejected]] = False
        return km
