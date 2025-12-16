# pro/multiscale_decomp.py
from __future__ import annotations
import numpy as np
import cv2

from dataclasses import dataclass
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox
)

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

def apply_layer_ops(w: np.ndarray, bias_gain: float, thr: float, amount: float):
    w2 = w
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

        # LEFT: preview
        left = QVBoxLayout()
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        left.addWidget(self.view)

        zoom_row = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_1to1 = QPushButton("1:1")
        self.btn_fit.clicked.connect(self._fit_view)
        self.btn_1to1.clicked.connect(self._one_to_one)
        zoom_row.addStretch(1)
        zoom_row.addWidget(self.btn_fit)
        zoom_row.addWidget(self.btn_1to1)
        zoom_row.addStretch(1)
        left.addLayout(zoom_row)

        # RIGHT: controls
        right = QVBoxLayout()

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
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["On", "Layer", "Scale", "Gain", "Thr", "Amt"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        v.addWidget(self.table)
        right.addWidget(gb_layers, stretch=1)

        # Per-layer editor
        gb_edit = QGroupBox("Selected Layer")
        ef = QFormLayout(gb_edit)
        self.lbl_sel = QLabel("Layer: —")

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

        ef.addRow(self.lbl_sel)
        ef.addRow("Gain:", self.spin_gain)
        ef.addRow("Threshold:", self.spin_thr)
        ef.addRow("Amount:", self.spin_amt)

        right.addWidget(gb_edit)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Document")
        self.btn_close = QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_close)
        right.addLayout(btn_row)

        root.addLayout(left, stretch=2)
        root.addLayout(right, stretch=1)

        # Signals
        self.spin_layers.valueChanged.connect(self._on_layers_changed)
        self.spin_sigma.valueChanged.connect(self._on_global_changed)
        self.combo_preview.currentIndexChanged.connect(self._schedule_preview)

        self.table.itemSelectionChanged.connect(self._on_table_select)

        self.spin_gain.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_thr.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_amt.valueChanged.connect(self._on_layer_editor_changed)

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
                tuned.append(apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount))

        out = multiscale_reconstruct(tuned, residual)
        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        # layer preview mode
        sel = self.combo_preview.currentData()
        if sel is None or sel == "final":
            self._preview_img = out
        elif sel == "residual":
            self._preview_img = np.clip(residual, 0, 1)
        else:
            # sel is int index of detail layer
            w = tuned[int(sel)]
            # visualize detail layer: map around 0.5
            vis = np.clip(0.5 + (w * 4.0), 0.0, 1.0)  # gain for visibility
            self._preview_img = vis

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
        self.view.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def _one_to_one(self):
        self.view.resetTransform()

    # ---------- Table / layer editing ----------
    def _rebuild_table(self):
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(self.layers)
            for i in range(self.layers):
                cfg = self.cfgs[i]
                # On
                item_on = QTableWidgetItem("")
                item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item_on.setCheckState(Qt.CheckState.Checked if cfg.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(i, 0, item_on)

                self.table.setItem(i, 1, QTableWidgetItem(str(i+1)))
                self.table.setItem(i, 2, QTableWidgetItem(f"{self.base_sigma*(2**i):.2f}"))

                self.table.setItem(i, 3, QTableWidgetItem(f"{cfg.bias_gain:.2f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{cfg.thr:.4f}"))
                self.table.setItem(i, 5, QTableWidgetItem(f"{cfg.amount:.2f}"))
        finally:
            self.table.blockSignals(False)

        # react to checkbox toggles / edits
        self.table.itemChanged.connect(self._on_table_item_changed)

        # select first row by default
        if self.layers > 0 and not self.table.selectedItems():
            self.table.selectRow(0)
            self._load_layer_into_editor(0)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()
        if not (0 <= r < len(self.cfgs)):
            return
        cfg = self.cfgs[r]
        if c == 0:
            cfg.enabled = (item.checkState() == Qt.CheckState.Checked)
            self._schedule_preview()

    def _on_table_select(self):
        rows = {it.row() for it in self.table.selectedItems()}
        if not rows:
            return
        r = min(rows)
        self._load_layer_into_editor(r)

    def _load_layer_into_editor(self, idx: int):
        cfg = self.cfgs[idx]
        self._selected_layer = idx
        self.lbl_sel.setText(f"Layer: {idx+1} / {self.layers}")
        self.spin_gain.blockSignals(True)
        self.spin_thr.blockSignals(True)
        self.spin_amt.blockSignals(True)
        try:
            self.spin_gain.setValue(cfg.bias_gain)
            self.spin_thr.setValue(cfg.thr)
            self.spin_amt.setValue(cfg.amount)
        finally:
            self.spin_gain.blockSignals(False)
            self.spin_thr.blockSignals(False)
            self.spin_amt.blockSignals(False)

    def _on_layer_editor_changed(self):
        idx = getattr(self, "_selected_layer", None)
        if idx is None or not (0 <= idx < len(self.cfgs)):
            return
        cfg = self.cfgs[idx]
        cfg.bias_gain = float(self.spin_gain.value())
        cfg.thr = float(self.spin_thr.value())
        cfg.amount = float(self.spin_amt.value())

        # keep table in sync
        self.table.blockSignals(True)
        try:
            self.table.item(idx, 3).setText(f"{cfg.bias_gain:.2f}")
            self.table.item(idx, 4).setText(f"{cfg.thr:.4f}")
            self.table.item(idx, 5).setText(f"{cfg.amount:.2f}")
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
            self.combo_preview.addItem("Residual", userData="residual")
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
                tuned.append(apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount))

        out = multiscale_reconstruct(tuned, residual)
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
