from __future__ import annotations

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Star Preview UI (SEP Demo)"
SCRIPT_GROUP    = "Samples"
SCRIPT_SHORTCUT = ""   # optional

# -------------------------
# Star Preview UI sample
# -------------------------

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QMessageBox, QApplication, QWidget
)

# your libs already bundled in SASpro
from imageops.stretch import stretch_color_image, stretch_mono_image
from imageops.starbasedwhitebalance import apply_star_based_white_balance

# (optional) for applying result back to active doc
from pro.whitebalance import apply_white_balance_to_doc


def _to_float01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img).astype(np.float32, copy=False)
    if a.size == 0:
        return a
    m = float(np.nanmax(a))
    if np.isfinite(m) and m > 1.0:
        a = a / m
    return np.clip(a, 0.0, 1.0)


class StarPreviewDialog(QDialog):
    """
    Sample script UI:
    - Shows active image (auto-updates when subwindow changes)
    - Runs SEP detection + ellipse overlay
    - Zoom controls + Fit/1:1
    - Demo Apply WB to active image
    """
    def __init__(self, ctx, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Sample Script: Star Preview UI")
        self.resize(980, 640)

        self._zoom = 1.0
        self._img01: np.ndarray | None = None
        self._overlay01: np.ndarray | None = None

        self._build_ui()
        self._wire()

        # debounce for slider/checkbox
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._rebuild_overlay)

        # watch active base doc so preview isn't blank
        try:
            dm = getattr(self.ctx.app, "doc_manager", None)
            if dm is not None and hasattr(dm, "activeBaseChanged"):
                dm.activeBaseChanged.connect(lambda _=None: self._load_active_image())
        except Exception:
            pass

        # initial load
        QTimer.singleShot(0, self._load_active_image)

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        self.preview = QLabel("No active image.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("border: 1px solid #333; background:#1f1f1f;")
        self.preview.setMinimumSize(720, 420)
        root.addWidget(self.preview, stretch=1)

        # Zoom bar
        zrow = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_fit      = QPushButton("Fit")
        self.btn_1to1     = QPushButton("1:1")
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_fit)
        zrow.addWidget(self.btn_1to1)
        zrow.addStretch(1)
        root.addLayout(zrow)

        # SEP controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("SEP threshold (σ):"))
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(1, 100)
        self.thr_slider.setValue(50)
        self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        ctrl.addWidget(self.thr_slider, stretch=1)

        self.thr_label = QLabel("50")
        self.thr_label.setFixedWidth(30)
        ctrl.addWidget(self.thr_label)

        self.chk_autostretch = QCheckBox("Autostretch preview")
        self.chk_autostretch.setChecked(True)
        ctrl.addWidget(self.chk_autostretch)

        root.addLayout(ctrl)

        # bottom buttons
        brow = QHBoxLayout()
        brow.addStretch(1)
        self.btn_apply_demo = QPushButton("Apply WB to Active Image (demo)")
        self.btn_close = QPushButton("Close")
        brow.addWidget(self.btn_apply_demo)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

    def _wire(self):
        self.btn_close.clicked.connect(self.reject)

        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_fit.clicked.connect(self._zoom_fit)
        self.btn_1to1.clicked.connect(lambda: self._set_zoom(1.0))

        self.thr_slider.valueChanged.connect(self._on_thr_changed)
        self.chk_autostretch.toggled.connect(lambda _=None: self._debounce.start())

        self.btn_apply_demo.clicked.connect(self._apply_demo_wb)

    # ------------- Active image -------------
    def _load_active_image(self):
        try:
            doc = self.ctx.active_document()
        except Exception:
            doc = None

        if doc is None or getattr(doc, "image", None) is None:
            self._img01 = None
            self._overlay01 = None
            self.preview.setText("No active image.")
            self.preview.setPixmap(QPixmap())
            return

        img = _to_float01(np.asarray(doc.image))
        self._img01 = img
        self._zoom_fit()
        self._rebuild_overlay()

    # ------------- SEP overlay -------------
    def _on_thr_changed(self, v: int):
        self.thr_label.setText(str(v))
        self._debounce.start()

    def _rebuild_overlay(self):
        if self._img01 is None:
            return
        try:
            thr = float(self.thr_slider.value())
            auto = bool(self.chk_autostretch.isChecked())

            img = self._img01
            # if mono, make a fake RGB for visualization / SEP expects gray anyway
            if img.ndim == 2:
                rgb = np.repeat(img[..., None], 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                rgb = np.repeat(img, 3, axis=2)
            else:
                rgb = img

            # Use your WB star detector just for overlay
            # (balanced output ignored; we only want overlay + count)
            _balanced, count, overlay = apply_star_based_white_balance(
                rgb, threshold=thr, autostretch=auto,
                reuse_cached_sources=False, return_star_colors=False
            )

            self._overlay01 = overlay
            self._render_pixmap()
            self.setWindowTitle(f"Sample Script: Star Preview UI  —  {count} stars")

        except Exception as e:
            self._overlay01 = None
            self.preview.setText(f"Star detection failed:\n{e}")

    # ------------- Rendering / zoom -------------
    def _render_pixmap(self):
        if self._overlay01 is None:
            return
        ov = np.clip(self._overlay01, 0, 1)
        h, w, c = ov.shape
        qimg = QImage((ov * 255).astype(np.uint8).data, w, h, 3*w, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg)

        # apply zoom
        zw = int(pm.width() * self._zoom)
        zh = int(pm.height() * self._zoom)
        pmz = pm.scaled(zw, zh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview.setPixmap(pmz)

    def _set_zoom(self, z: float):
        self._zoom = float(np.clip(z, 0.05, 20.0))
        self._render_pixmap()

    def _zoom_fit(self):
        if self._overlay01 is None and self._img01 is None:
            return
        # fit based on raw image size
        base = self._overlay01 if self._overlay01 is not None else self._img01
        h, w = base.shape[:2]
        vw = max(1, self.preview.width())
        vh = max(1, self.preview.height())
        self._zoom = min(vw / w, vh / h)
        self._render_pixmap()

    # ------------- Demo apply -------------
    def _apply_demo_wb(self):
        try:
            doc = self.ctx.active_document()
            if doc is None:
                raise RuntimeError("No active document.")
            # Reuse your headless preset WB as an example of applying edits
            preset = {"mode": "star", "threshold": float(self.thr_slider.value())}
            apply_white_balance_to_doc(doc, preset)
            QMessageBox.information(self, "Demo", "White Balance applied to active image.")
            # refresh preview after edit
            self._load_active_image()
        except Exception as e:
            QMessageBox.critical(self, "Demo", f"Failed to apply WB:\n{e}")


def run(ctx):
    """
    SASpro entry point.
    """
    w = StarPreviewDialog(ctx, parent=ctx.app)
    w.exec()
