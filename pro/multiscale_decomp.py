# pro/multiscale_decomp.py
from __future__ import annotations
import numpy as np
import cv2

from dataclasses import dataclass
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QToolButton, QSlider, QSplitter,
)


class _ZoomPanView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._panning = False
        self._pan_start = None

    def wheelEvent(self, ev):
        # Ctrl+wheel optional – but I’ll make plain wheel zoom since you asked
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_start is not None:
            delta = ev.pos() - self._pan_start
            self._pan_start = ev.pos()

            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


# ─────────────────────────────────────────────
# Core math (your backbone)
# ─────────────────────────────────────────────

def _blur_gaussian(img01: np.ndarray, sigma: float) -> np.ndarray:
    k = int(max(3, 2 * round(3 * sigma) + 1))  # odd
    return cv2.GaussianBlur(img01, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def multiscale_decompose(img01: np.ndarray, layers: int, base_sigma: float = 1.0):
    c = img01.astype(np.float32, copy=False)
    details = []
    for k in range(layers):
        sigma = base_sigma * (2 ** k)
        c_next = _blur_gaussian(c, sigma)
        w = c - c_next
        details.append(w)
        c = c_next
    residual = c
    return details, residual

def multiscale_reconstruct(details, residual):
    out = residual.astype(np.float32, copy=True)
    for w in details:
        out += w
    return out

def soft_threshold(x: np.ndarray, t: float):
    a = np.abs(x)
    return np.sign(x) * np.maximum(0.0, a - t)

def apply_layer_ops(
    w: np.ndarray,
    bias_gain: float,
    thr: float,
    amount: float,
    denoise_strength: float = 0.0,
    sigma: float | np.ndarray | None = None,
):
    w2 = w

    # 1) Noise reduction step (MMT-style NR)
    if denoise_strength > 0.0:
        if sigma is None:
            sigma = float(np.std(w2))
        # 3σ at denoise=1, scaled linearly
        t_dn = max(0.0, denoise_strength * 3.0 * float(sigma))
        if t_dn > 0.0:
            w_dn = soft_threshold(w2, t_dn)
            # Blend original vs denoised based on denoise_strength
            w2 = (1.0 - denoise_strength) * w2 + denoise_strength * w_dn

    # 2) Existing threshold + bias shaping (can act as detail shaping / extra NR)
    if thr > 0:
        wt = soft_threshold(w2, thr)
        w2 = (1.0 - amount) * w2 + amount * wt
    if abs(bias_gain - 1.0) > 1e-6:
        w2 = w2 * bias_gain
    return w2

# ─────────────────────────────────────────────
# Layer config
# ─────────────────────────────────────────────


@dataclass
class LayerCfg:
    enabled: bool = True
    bias_gain: float = 1.0        # 1.0 = unchanged
    thr: float = 0.0              # soft threshold in detail domain
    amount: float = 0.0           # 0..1 blend toward thresholded
    denoise: float = 0.0          # 0..1 additional noise reduction


# ─────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────

class MultiscaleDecompDialog(QDialog):
    """
    PI-style multiscale decomposition (MMT-like) dialog.
    Decompose into detail layers + residual, let user tune per-layer ops, preview, apply to doc.
    """
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.setWindowTitle("Multiscale Decomposition")
        self.setMinimumSize(1050, 700)
        self.residual_enabled = True
        self._layer_noise = None  # list[float] per detail layer

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1] (match your other tools’ convention)
        img = np.asarray(base)
        img = img.astype(np.float32, copy=False)
        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(img)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        self._orig_shape = img.shape
        self._orig_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

        # force display buffer to 3ch (like your other dialogs)
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        else:
            img3 = img[:, :, :3]

        self._image = img3.copy()      # working linear image (edited on Apply only)
        self._preview_img = img3.copy()

        # decomposition cache
        self._cached_layers = None     # list[np.ndarray]
        self._cached_residual = None
        self._cached_key = None        # (layers, base_sigma)

        # per-layer configs
        self.layers = 4
        self.base_sigma = 1.0
        self.cfgs: list[LayerCfg] = [LayerCfg() for _ in range(self.layers)]

        # debounce preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._rebuild_preview)

        self._build_ui()
        self._recompute_decomp(force=True)
        self._rebuild_preview()
        QTimer.singleShot(0, self._fit_view)

    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # Splitter between preview (left) and controls (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ----- LEFT: preview -----
        left_widget = QWidget(self)
        left = QVBoxLayout(left_widget)

        self.scene = QGraphicsScene(self)
        self.view = _ZoomPanView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        left.addWidget(self.view)

        zoom_row = QHBoxLayout()

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setIcon(QIcon.fromTheme("zoom-out"))
        self.zoom_out_btn.setToolTip("Zoom Out")

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setIcon(QIcon.fromTheme("zoom-in"))
        self.zoom_in_btn.setToolTip("Zoom In")

        self.fit_btn = QToolButton()
        self.fit_btn.setIcon(QIcon.fromTheme("zoom-fit-best"))
        self.fit_btn.setToolTip("Fit to Preview")

        self.one_to_one_btn = QToolButton()
        self.one_to_one_btn.setIcon(QIcon.fromTheme("zoom-original"))
        self.one_to_one_btn.setToolTip("1:1")

        self.zoom_out_btn.clicked.connect(lambda: self.view.scale(0.8, 0.8))
        self.zoom_in_btn.clicked.connect(lambda: self.view.scale(1.25, 1.25))
        self.fit_btn.clicked.connect(self._fit_view)
        self.one_to_one_btn.clicked.connect(self._one_to_one)

        zoom_row.addStretch(1)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.fit_btn)
        zoom_row.addWidget(self.one_to_one_btn)
        zoom_row.addStretch(1)

        left.addLayout(zoom_row)

        # ----- RIGHT: controls -----
        right_widget = QWidget(self)
        right = QVBoxLayout(right_widget)

        gb_global = QGroupBox("Global")
        form = QFormLayout(gb_global)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 10)
        self.spin_layers.setValue(self.layers)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.3, 5.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(self.base_sigma)

        self.cb_linked_rgb = QCheckBox("Linked RGB (apply same params to all channels)")
        self.cb_linked_rgb.setChecked(True)

        self.combo_preview = QComboBox()
        self._refresh_preview_combo()

        form.addRow("Layers:", self.spin_layers)
        form.addRow("Base sigma:", self.spin_sigma)
        form.addRow(self.cb_linked_rgb)
        form.addRow("Layer preview:", self.combo_preview)

        right.addWidget(gb_global)

        # Layers table
        gb_layers = QGroupBox("Layers")
        v = QVBoxLayout(gb_layers)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["On", "Layer", "Scale", "Gain", "Thr", "Amt", "NR", "Type"]
        )

        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        v.addWidget(self.table)
        right.addWidget(gb_layers, stretch=1)

        # Per-layer editor (now with sliders)
        gb_edit = QGroupBox("Selected Layer")
        ef = QFormLayout(gb_edit)
        self.lbl_sel = QLabel("Layer: —")

        # --- Spin boxes ---
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 3.0)
        self.spin_gain.setSingleStep(0.05)
        self.spin_gain.setValue(1.0)

        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 0.10)
        self.spin_thr.setSingleStep(0.001)
        self.spin_thr.setDecimals(4)

        self.spin_amt = QDoubleSpinBox()
        self.spin_amt.setRange(0.0, 1.0)
        self.spin_amt.setSingleStep(0.05)

        self.spin_denoise = QDoubleSpinBox()
        self.spin_denoise.setRange(0.0, 1.0)
        self.spin_denoise.setSingleStep(0.05)
        self.spin_denoise.setValue(0.0)

        # --- Sliders (int ranges, mapped to spins) ---
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(0, 300)         # 0..3.00
        self.slider_thr = QSlider(Qt.Orientation.Horizontal)
        self.slider_thr.setRange(0, 100)          # 0..0.10 (×0.001)
        self.slider_amt = QSlider(Qt.Orientation.Horizontal)
        self.slider_amt.setRange(0, 100)          # 0..1.00
        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 100)      # 0..1.00

        # Layout rows: label -> [slider | spinbox]
        ef.addRow(self.lbl_sel)

        gain_row = QHBoxLayout()
        gain_row.addWidget(self.slider_gain)
        gain_row.addWidget(self.spin_gain)
        ef.addRow("Gain:", gain_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(self.slider_thr)
        thr_row.addWidget(self.spin_thr)
        ef.addRow("Threshold:", thr_row)

        amt_row = QHBoxLayout()
        amt_row.addWidget(self.slider_amt)
        amt_row.addWidget(self.spin_amt)
        ef.addRow("Amount:", amt_row)

        dn_row = QHBoxLayout()
        dn_row.addWidget(self.slider_denoise)
        dn_row.addWidget(self.spin_denoise)
        ef.addRow("Denoise:", dn_row)

        right.addWidget(gb_edit)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Document")
        self.btn_close = QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_close)
        right.addLayout(btn_row)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # ----- Signals -----
        self.spin_layers.valueChanged.connect(self._on_layers_changed)
        self.spin_sigma.valueChanged.connect(self._on_global_changed)
        self.combo_preview.currentIndexChanged.connect(self._schedule_preview)

        self.table.itemSelectionChanged.connect(self._on_table_select)

        # spinboxes -> layer cfg
        self.spin_gain.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_thr.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_amt.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_denoise.valueChanged.connect(self._on_layer_editor_changed)

        # sliders -> spinboxes
        self.slider_gain.valueChanged.connect(self._on_gain_slider_changed)
        self.slider_thr.valueChanged.connect(self._on_thr_slider_changed)
        self.slider_amt.valueChanged.connect(self._on_amt_slider_changed)
        self.slider_denoise.valueChanged.connect(self._on_dn_slider_changed)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_close.clicked.connect(self.reject)

        # Signals
        self.spin_layers.valueChanged.connect(self._on_layers_changed)
        self.spin_sigma.valueChanged.connect(self._on_global_changed)
        self.combo_preview.currentIndexChanged.connect(self._schedule_preview)

        self.table.itemSelectionChanged.connect(self._on_table_select)

        self.spin_gain.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_thr.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_amt.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_denoise.valueChanged.connect(self._on_layer_editor_changed)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_close.clicked.connect(self.reject)

    # ---------- Preview plumbing ----------
    def _schedule_preview(self):
        self._preview_timer.start(60)

    def _recompute_decomp(self, force: bool = False):
        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())
        key = (layers, base_sigma)

        if (not force) and self._cached_key == key and self._cached_layers is not None:
            return

        self.layers = layers
        self.base_sigma = base_sigma

        self._cached_layers, self._cached_residual = multiscale_decompose(
            self._image, layers=self.layers, base_sigma=self.base_sigma
        )
        self._cached_key = key

        self._layer_noise = []
        for w in self._cached_layers:
            sigma = float(np.std(w)) if w.size else 0.0
            if sigma <= 0.0:
                sigma = 1e-6
            self._layer_noise.append(sigma)

        # ensure cfg list matches layer count
        if len(self.cfgs) != self.layers:
            old = self.cfgs[:]
            self.cfgs = [LayerCfg() for _ in range(self.layers)]
            for i in range(min(len(old), self.layers)):
                self.cfgs[i] = old[i]

        self._rebuild_table()
        self._refresh_preview_combo()

    def _rebuild_preview(self):
        self._recompute_decomp(force=False)
        details = self._cached_layers
        residual = self._cached_residual
        if details is None or residual is None:
            return

        # apply per-layer ops
        tuned = []
        for i, w in enumerate(details):
            cfg = self.cfgs[i]
            if not cfg.enabled:
                tuned.append(np.zeros_like(w))
            else:
                sigma = None
                if self._layer_noise is not None and i < len(self._layer_noise):
                    sigma = self._layer_noise[i]
                tuned.append(
                    apply_layer_ops(
                        w,
                        cfg.bias_gain,
                        cfg.thr,
                        cfg.amount,
                        cfg.denoise,
                        sigma,
                    )
                )

        # reconstruction (keep raw version for visualization)
        res = residual if self.residual_enabled else np.zeros_like(residual)
        out_raw = multiscale_reconstruct(tuned, res)
        out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)

        sel = self.combo_preview.currentData()
        if sel is None or sel == "final":
            if not self.residual_enabled:
                # Detail-only visualization: SAME style as detail-layer preview
                d = out_raw.astype(np.float32, copy=False)
                vis = 0.5 + d * 4.0      # same gain as single-layer view
                self._preview_img = np.clip(vis, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                self._preview_img = out

        elif sel == "residual":
            self._preview_img = np.clip(residual, 0, 1)

        else:
            # sel is int index of detail layer
            w = tuned[int(sel)]
            vis = np.clip(0.5 + (w * 4.0), 0.0, 1.0)
            self._preview_img = vis.astype(np.float32, copy=False)

        self._refresh_pix()



    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        h, w = arr.shape[:2]
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _refresh_pix(self):
        self.pix.setPixmap(self._np_to_qpix(self._preview_img))
        self.scene.setSceneRect(self.pix.boundingRect())

    def _fit_view(self):
        if self.pix.pixmap().isNull():
            return
        self.view.resetTransform()
        self.view.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def _one_to_one(self):
        self.view.resetTransform()

    # ---------- Table / layer editing ----------
    def _on_gain_slider_changed(self, v: int):
        # 0..300 -> 0.00..3.00
        val = v / 100.0
        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(val)
        self.spin_gain.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_thr_slider_changed(self, v: int):
        # 0..100 -> 0.000..0.100
        val = v / 1000.0
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(val)
        self.spin_thr.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_amt_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_amt.blockSignals(True)
        self.spin_amt.setValue(val)
        self.spin_amt.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_dn_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_denoise.blockSignals(True)
        self.spin_denoise.setValue(val)
        self.spin_denoise.blockSignals(False)
        self._on_layer_editor_changed()


    def _rebuild_table(self):
        self.table.blockSignals(True)
        try:
            # +1 row for residual ("R")
            self.table.setRowCount(self.layers + 1)

            # detail rows
            for i in range(self.layers):
                cfg = self.cfgs[i]

                item_on = QTableWidgetItem("")
                item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item_on.setCheckState(Qt.CheckState.Checked if cfg.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(i, 0, item_on)

                self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
                self.table.setItem(i, 2, QTableWidgetItem(f"{self.base_sigma * (2**i):.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{cfg.bias_gain:.2f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{cfg.thr:.4f}"))
                self.table.setItem(i, 5, QTableWidgetItem(f"{cfg.amount:.2f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{cfg.denoise:.2f}"))
                self.table.setItem(i, 7, QTableWidgetItem("D"))

            # residual row
            r = self.layers
            item_on = QTableWidgetItem("")
            item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_on.setCheckState(
                Qt.CheckState.Checked if self.residual_enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(r, 0, item_on)

            self.table.setItem(r, 1, QTableWidgetItem("R"))
            self.table.setItem(r, 2, QTableWidgetItem("—"))
            self.table.setItem(r, 3, QTableWidgetItem("1.00"))
            self.table.setItem(r, 4, QTableWidgetItem("0.0000"))
            self.table.setItem(r, 5, QTableWidgetItem("0.00"))
            self.table.setItem(r, 6, QTableWidgetItem("0.00"))
            self.table.setItem(r, 7, QTableWidgetItem("R"))

        finally:
            self.table.blockSignals(False)

        # connect once (avoid stacking connects)
        try:
            self.table.itemChanged.disconnect(self._on_table_item_changed)
        except Exception:
            pass
        self.table.itemChanged.connect(self._on_table_item_changed)

        if self.layers > 0 and not self.table.selectedItems():
            self.table.selectRow(0)
            self._load_layer_into_editor(0)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()

        # Residual row
        if r == self.layers:
            if c == 0:
                self.residual_enabled = (item.checkState() == Qt.CheckState.Checked)
                self._schedule_preview()
            # ignore other edits for residual
            return

        if not (0 <= r < len(self.cfgs)):
            return

        cfg = self.cfgs[r]

        if c == 0:
            # On/off
            cfg.enabled = (item.checkState() == Qt.CheckState.Checked)
            self._schedule_preview()
            return

        # numeric columns: Gain(3), Thr(4), Amt(5), NR(6)
        try:
            text = item.text().strip()
            val = float(text) if text else 0.0
        except Exception:
            return

        if c == 3:
            cfg.bias_gain = val
        elif c == 4:
            cfg.thr = val
        elif c == 5:
            cfg.amount = val
        elif c == 6:
            cfg.denoise = val
        else:
            return

        # If this row is currently selected, update editor widgets too
        if getattr(self, "_selected_layer", None) == r:
            self._load_layer_into_editor(r)

        self._schedule_preview()



    def _on_table_select(self):
        rows = {it.row() for it in self.table.selectedItems()}
        if not rows:
            return
        r = min(rows)
        self._load_layer_into_editor(r)

    def _load_layer_into_editor(self, idx: int):
        self._selected_layer = idx

        if idx == self.layers:
            self.lbl_sel.setText("Layer: R (Residual)")
            for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                    self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
                w.setEnabled(False)
            return

        for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
            w.setEnabled(True)

        cfg = self.cfgs[idx]
        self.lbl_sel.setText(f"Layer: {idx+1} / {self.layers}")

        # spins + sliders in sync
        self.spin_gain.blockSignals(True)
        self.spin_thr.blockSignals(True)
        self.spin_amt.blockSignals(True)
        self.spin_denoise.blockSignals(True)

        self.slider_gain.blockSignals(True)
        self.slider_thr.blockSignals(True)
        self.slider_amt.blockSignals(True)
        self.slider_denoise.blockSignals(True)
        try:
            self.spin_gain.setValue(cfg.bias_gain)
            self.spin_thr.setValue(cfg.thr)
            self.spin_amt.setValue(cfg.amount)
            self.spin_denoise.setValue(cfg.denoise)

            self.slider_gain.setValue(int(round(cfg.bias_gain * 100.0)))
            self.slider_thr.setValue(int(round(cfg.thr * 1000.0)))
            self.slider_amt.setValue(int(round(cfg.amount * 100.0)))
            self.slider_denoise.setValue(int(round(cfg.denoise * 100.0)))
        finally:
            self.spin_gain.blockSignals(False)
            self.spin_thr.blockSignals(False)
            self.spin_amt.blockSignals(False)
            self.spin_denoise.blockSignals(False)
            self.slider_gain.blockSignals(False)
            self.slider_thr.blockSignals(False)
            self.slider_amt.blockSignals(False)
            self.slider_denoise.blockSignals(False)



    def _on_layer_editor_changed(self):
        idx = getattr(self, "_selected_layer", None)
        if idx is None or not (0 <= idx < len(self.cfgs)):
            return
        cfg = self.cfgs[idx]
        cfg.bias_gain = float(self.spin_gain.value())
        cfg.thr = float(self.spin_thr.value())
        cfg.amount = float(self.spin_amt.value())
        cfg.denoise = float(self.spin_denoise.value())

        # keep table in sync
        self.table.blockSignals(True)
        try:
            self.table.item(idx, 3).setText(f"{cfg.bias_gain:.2f}")
            self.table.item(idx, 4).setText(f"{cfg.thr:.4f}")
            self.table.item(idx, 5).setText(f"{cfg.amount:.2f}")
            self.table.item(idx, 6).setText(f"{cfg.denoise:.2f}")
        finally:
            self.table.blockSignals(False)

        self._schedule_preview()

    def _on_layers_changed(self):
        self._recompute_decomp(force=True)
        self._schedule_preview()

    def _on_global_changed(self):
        self._recompute_decomp(force=True)
        self._schedule_preview()

    def _refresh_preview_combo(self):
        self.combo_preview.blockSignals(True)
        try:
            self.combo_preview.clear()
            self.combo_preview.addItem("Final", userData="final")
            self.combo_preview.addItem("R (Residual)", userData="residual")
            for i in range(self.layers):
                self.combo_preview.addItem(f"Detail Layer {i+1}", userData=i)
        finally:
            self.combo_preview.blockSignals(False)

    # ---------- Apply to doc ----------
    def _commit_to_doc(self):
        self._recompute_decomp(force=False)

        details = self._cached_layers
        residual = self._cached_residual
        if details is None or residual is None:
            return

        tuned = []
        for i, w in enumerate(details):
            cfg = self.cfgs[i]
            if not cfg.enabled:
                tuned.append(np.zeros_like(w))
            else:
                sigma = None
                if self._layer_noise is not None and i < len(self._layer_noise):
                    sigma = self._layer_noise[i]
                tuned.append(
                    apply_layer_ops(
                        w,
                        cfg.bias_gain,
                        cfg.thr,
                        cfg.amount,
                        cfg.denoise,
                        sigma,
                    )
                )

        res = residual if self.residual_enabled else np.zeros_like(residual)
        out = multiscale_reconstruct(tuned, res)
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        # convert back to mono if original was mono
        if self._orig_mono:
            mono = out[..., 0]
            if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                mono = mono[:, :, None]
            out_final = mono.astype(np.float32, copy=False)
        else:
            out_final = out

        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out_final, step_name="Multiscale Decomposition")
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out_final, step_name="Multiscale Decomposition")
            else:
                self._doc.image = out_final
        except Exception as e:
            QMessageBox.critical(self, "Multiscale Decomposition", f"Failed to write to document:\n{e}")
            return

        # refresh active view if main window has hook
        if hasattr(self.parent(), "_refresh_active_view"):
            try:
                self.parent()._refresh_active_view()
            except Exception:
                pass

        self.accept()


class _MultiScaleDecompPresetDialog(QDialog):
    """
    Preset editor for Multiscale Decomposition (headless + shortcuts).
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Multiscale Decomposition — Preset")
        init = dict(initial or {})

        v = QVBoxLayout(self)

        # ---- Global ----
        gb = QGroupBox("Global")
        form = QFormLayout(gb)

        self.sp_layers = QSpinBox()
        self.sp_layers.setRange(1, 10)
        self.sp_layers.setValue(int(init.get("layers", 4)))

        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.3, 5.0)
        self.sp_sigma.setDecimals(2)
        self.sp_sigma.setSingleStep(0.1)
        self.sp_sigma.setValue(float(init.get("base_sigma", 1.0)))

        self.cb_linked = QCheckBox("Linked RGB channels")
        self.cb_linked.setChecked(bool(init.get("linked_rgb", True)))

        form.addRow("Layers:", self.sp_layers)
        form.addRow("Base sigma:", self.sp_sigma)
        form.addRow("", self.cb_linked)

        v.addWidget(gb)

        # ---- Layers ----
        gb_layers = QGroupBox("Per-Layer Settings")
        lv = QVBoxLayout(gb_layers)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["On", "Layer", "Gain", "Threshold", "Amount", "Denoise"])

        self.table.verticalHeader().setVisible(False)
        lv.addWidget(self.table)

        v.addWidget(gb_layers)

        # ---- Buttons ----
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        self._populate_table(init)

    def _populate_table(self, init: dict):
        layers = int(self.sp_layers.value())
        cfgs = init.get("layers_cfg", [])

        self.table.setRowCount(layers)

        for i in range(layers):
            cfg = cfgs[i] if i < len(cfgs) else {}

            # Enabled
            chk = QTableWidgetItem("")
            chk.setFlags(chk.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(
                Qt.CheckState.Checked if cfg.get("enabled", True)
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(i, 0, chk)

            self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{float(cfg.get('gain', 1.0)):.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{float(cfg.get('thr', 0.0)):.4f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{float(cfg.get('amount', 0.0)):.2f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{float(cfg.get('denoise', 0.0)):.2f}"))


    def result_dict(self) -> dict:
        layers = int(self.sp_layers.value())
        out_layers = []

        for r in range(layers):
            enabled = self.table.item(r, 0).checkState() == Qt.CheckState.Checked
            gain = float(self.table.item(r, 2).text())
            thr = float(self.table.item(r, 3).text())
            amt = float(self.table.item(r, 4).text())
            try:
                dn = float(self.table.item(r, 5).text())
            except Exception:
                dn = 0.0

            out_layers.append({
                "enabled": enabled,
                "gain": gain,
                "thr": thr,
                "amount": amt,
                "denoise": dn,
            })


        return {
            "layers": layers,
            "base_sigma": float(self.sp_sigma.value()),
            "linked_rgb": bool(self.cb_linked.isChecked()),
            "layers_cfg": out_layers,
        }
