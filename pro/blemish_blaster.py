# pro/blemish_blaster.py
from __future__ import annotations
import math
import numpy as np
from typing import Optional
from PyQt6.QtCore import Qt, QEvent, QPointF, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QAction, QKeySequence, QColor, QWheelEvent, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QSlider,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QMessageBox, QScrollArea, QCheckBox, QDoubleSpinBox    
)
from imageops.stretch import stretch_color_image, stretch_mono_image 

from dataclasses import dataclass

@dataclass
class BlemishOp:
    x: int
    y: int
    radius: int
    feather: float
    opacity: float
    channels: list[int]


# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────

class _BBWorkerSignals(QObject):
    finished = pyqtSignal(np.ndarray)

class _BlemishWorker(QRunnable):
    def __init__(self, image: np.ndarray, x: int, y: int, radius: int, feather: float, opacity: float,
                 channels_to_process: list[int]):
        super().__init__()
        self.image = image.copy()
        self.x, self.y = int(x), int(y)
        self.radius = int(radius)
        self.feather = float(feather)
        self.opacity = float(opacity)
        self.channels_to_process = channels_to_process
        self.signals = _BBWorkerSignals()

    @pyqtSlot()
    def run(self):
        out = self._remove_blemish(
            self.image, self.x, self.y, self.radius, self.feather, self.opacity, self.channels_to_process
        )
        self.signals.finished.emit(out)

    # ── the exact SASv2 logic (minor tidy) ────────────────────────────────────
    def _remove_blemish(self, image, x, y, radius, feather, opacity, channels_to_process):
        corrected_image = image.copy()
        h, w = image.shape[:2]

        # 6 neighbors
        angles = [0, 60, 120, 180, 240, 300]
        centers = []
        for ang in angles:
            r = math.radians(ang)
            dx = int(math.cos(r) * (radius * 1.5))
            dy = int(math.sin(r) * (radius * 1.5))
            centers.append((x + dx, y + dy))

        tgt_median = self._median_circle(image, x, y, radius, channels_to_process)
        neigh_medians = [self._median_circle(image, cx, cy, radius, channels_to_process) for (cx, cy) in centers]

        diffs = [abs(m - tgt_median) for m in neigh_medians]
        idxs = np.argsort(diffs)[:3]
        sel_centers = [centers[i] for i in idxs]

        for c in channels_to_process:
            for i in range(max(y - radius, 0), min(y + radius + 1, h)):
                yi = i - y
                for j in range(max(x - radius, 0), min(x + radius + 1, w)):
                    xj = j - x
                    dist = math.hypot(xj, yi)
                    if dist > radius:
                        continue

                    weight = 1.0 if feather <= 0 else max(0.0, min(1.0, (radius - dist) / (radius * feather)))

                    samples = []
                    for (cx, cy) in sel_centers:
                        sj = j + (cx - x)
                        si = i + (cy - y)
                        if 0 <= si < h and 0 <= sj < w:
                            if image.ndim == 2:
                                samples.append(image[si, sj])
                            elif image.ndim == 3 and image.shape[2] == 1:
                                samples.append(image[si, sj, 0])
                            elif image.ndim == 3 and c < image.shape[2]:
                                samples.append(image[si, sj, c])

                    if samples:
                        median_val = float(np.median(samples))
                    else:
                        if image.ndim == 2:
                            median_val = float(image[i, j])
                        elif image.ndim == 3 and image.shape[2] == 1:
                            median_val = float(image[i, j, 0])
                        else:
                            median_val = float(image[i, j, c])

                    if image.ndim == 2:
                        orig = float(image[i, j])
                        corrected_image[i, j] = (1 - opacity * weight) * orig + (opacity * weight) * median_val
                    elif image.ndim == 3 and image.shape[2] == 1:
                        orig = float(image[i, j, 0])
                        corrected_image[i, j, 0] = (1 - opacity * weight) * orig + (opacity * weight) * median_val
                    elif image.ndim == 3 and c < image.shape[2]:
                        orig = float(image[i, j, c])
                        corrected_image[i, j, c] = (1 - opacity * weight) * orig + (opacity * weight) * median_val

        return corrected_image

    def _median_circle(self, image, cx, cy, radius, channels):
        vals = []
        y0 = max(cy - radius, 0); y1 = min(cy + radius + 1, image.shape[0])
        x0 = max(cx - radius, 0); x1 = min(cx + radius + 1, image.shape[1])
        if y0 >= y1 or x0 >= x1:
            return 0.0

        for c in channels:
            if image.ndim == 2:
                roi = image[y0:y1, x0:x1]
            elif image.ndim == 3:
                if image.shape[2] == 1:
                    roi = image[y0:y1, x0:x1, 0]
                elif c < image.shape[2]:
                    roi = image[y0:y1, x0:x1, c]
                else:
                    continue
            else:
                continue

            yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
            mask = (xx - (cx - x0))**2 + (yy - (cy - y0))**2 <= radius**2
            vals.extend(roi[mask].ravel())

        return float(np.median(vals)) if len(vals) else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class BlemishBlasterDialogPro(QDialog):
    """
    Interactive blemish remover (preview + click to heal) that writes back to the
    provided document when 'Apply' is pressed.
    """
    def __init__(self, parent, doc):
        super().__init__(parent)
        self.setWindowTitle("Blemish Blaster")
        self.setMinimumSize(900, 650)

        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1]
        self._orig_shape = base.shape
        self._orig_mono = (base.ndim == 2) or (base.ndim == 3 and base.shape[2] == 1)

        img = np.asarray(base, dtype=np.float32)
        if img.dtype.kind in "ui":
            # Best-effort normalize
            maxv = float(np.nanmax(img)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        # display buffer is 3-channels for visualization
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img3 = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        self._image   = img3.copy()      # linear, edited by worker
        self._display = self._image.copy()

        # ── Scene/View (unchanged) ─────────────────────────────────────────
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        self.circle = QGraphicsEllipseItem()
        self.circle.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
        self.circle.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.circle.setVisible(False)
        self.scene.addItem(self.circle)

        # scroll container
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.view)

        # ── Controls
        ctrls = QGroupBox("Controls")
        form  = QFormLayout(ctrls)

        # existing sliders
        self.s_radius  = QSlider(Qt.Orientation.Horizontal); self.s_radius.setRange(1, 900); self.s_radius.setValue(12)
        self.s_feather = QSlider(Qt.Orientation.Horizontal); self.s_feather.setRange(0, 100); self.s_feather.setValue(50)
        self.s_opacity = QSlider(Qt.Orientation.Horizontal); self.s_opacity.setRange(0, 100); self.s_opacity.setValue(100)
        form.addRow("Radius:",  self.s_radius)
        form.addRow("Feather:", self.s_feather)
        form.addRow("Opacity:", self.s_opacity)

        # --- PREVIEW AUTOSTRETCH (display only) ---
        self.cb_autostretch = QCheckBox("Auto-stretch preview")
        self.cb_autostretch.setChecked(True)
        form.addRow(self.cb_autostretch)

        self.s_target_median = QDoubleSpinBox()
        self.s_target_median.setRange(0.01, 0.60)
        self.s_target_median.setSingleStep(0.01)
        self.s_target_median.setDecimals(3)
        self.s_target_median.setValue(0.25)
        form.addRow("Target median:", self.s_target_median)

        self.cb_linked = QCheckBox("Linked color channels")
        self.cb_linked.setChecked(True)
        form.addRow(self.cb_linked)

        # react to UI
        self.cb_autostretch.toggled.connect(self._update_display_autostretch)
        self.s_target_median.valueChanged.connect(self._update_display_autostretch)
        self.cb_linked.toggled.connect(self._update_display_autostretch)
        # (nice-to-have: disable fields when off)
        self.cb_autostretch.toggled.connect(lambda on: (self.s_target_median.setEnabled(on),
                                                        self.cb_linked.setEnabled(on)))

        # buttons / layout (unchanged)
        bb = QHBoxLayout()

        self.btn_undo  = QPushButton("Undo")
        self.btn_redo  = QPushButton("Redo")
        self.btn_apply = QPushButton("Apply to Document")
        self.btn_close = QPushButton("Close")

        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)

        bb.addStretch()
        bb.addWidget(self.btn_undo)
        bb.addWidget(self.btn_redo)
        bb.addSpacing(12)
        bb.addWidget(self.btn_apply)
        bb.addWidget(self.btn_close)

        main = QVBoxLayout(self)
        main.addWidget(self.scroll)
        main.addWidget(ctrls)
        main.addLayout(bb)

        # behavior
        self._threadpool = QThreadPool.globalInstance()
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_undo.clicked.connect(self._undo_step)
        self.btn_redo.clicked.connect(self._redo_step)
        # undo/redo inside dialog (simple)
        self._undo, self._redo = [], []
        self._update_undo_redo_buttons()

        # wheel zoom
        self._zoom = 1.0

        self._update_display_autostretch()

        # shortcuts
        a_undo = QAction(self); a_undo.setShortcut(QKeySequence.StandardKey.Undo); a_undo.triggered.connect(self._undo_step)
        a_redo = QAction(self); a_redo.setShortcut(QKeySequence.StandardKey.Redo); a_redo.triggered.connect(self._redo_step)
        self.addAction(a_undo); self.addAction(a_redo)

    def _update_undo_redo_buttons(self):
        try:
            self.btn_undo.setEnabled(len(self._undo) > 0)
            self.btn_redo.setEnabled(len(self._redo) > 0)
        except Exception:
            pass


    def _update_display_autostretch(self):
        """Rebuilds self._display from the current linear working image."""
        src = self._image  # linear data (HxWx3)
        if not self.cb_autostretch.isChecked():
            self._display = src.astype(np.float32, copy=False)
            self._refresh_pix()
            return

        tm = float(self.s_target_median.value())
        if not self._orig_mono:
            # true color source
            disp = stretch_color_image(src, target_median=tm, linked=self.cb_linked.isChecked(),
                                    normalize=False, apply_curves=False)
        else:
            # original was mono; channels in src are identical
            mono = src[..., 0]
            mono_st = stretch_mono_image(mono, target_median=tm, normalize=False, apply_curves=False)
            disp = np.stack([mono_st]*3, axis=-1)

        self._display = disp.astype(np.float32, copy=False)
        self._refresh_pix()


    # ── Event filter for hover/click + wheel zoom
    def eventFilter(self, src, ev):
        if src is self.view.viewport():
            if ev.type() == QEvent.Type.MouseMove:
                pos = self.view.mapToScene(ev.position().toPoint())
                r = self.s_radius.value()
                self.circle.setRect(pos.x()-r, pos.y()-r, 2*r, 2*r)
                self.circle.setVisible(True)
            elif ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                pos = self.view.mapToScene(ev.position().toPoint())
                self._heal_at(pos)
                return True
            elif ev.type() == QEvent.Type.Wheel:
                self._wheel_zoom(ev)
                return True
        return super().eventFilter(src, ev)

    # ── Heal logic
    def _heal_at(self, scene_pos: QPointF):
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        if not (0 <= x < self._image.shape[1] and 0 <= y < self._image.shape[0]):
            return
        radius  = int(self.s_radius.value())
        feather = float(self.s_feather.value()) / 100.0
        opacity = float(self.s_opacity.value()) / 100.0

        chans = [0, 1, 2]  # we always run on the 3-channel display buffer
        worker = _BlemishWorker(self._image, x, y, radius, feather, opacity, chans)
        worker.signals.finished.connect(self._on_worker_done)
        self.setEnabled(False)
        self._threadpool.start(worker)

    def _on_worker_done(self, corrected: np.ndarray):
        self._undo.append(self._image.copy()); self._redo.clear()
        self._image = corrected.astype(np.float32, copy=False)
        self._display = self._image.copy()
        self._update_display_autostretch()
        self.setEnabled(True)
        self._update_undo_redo_buttons()

    # ── Zoom
    def _wheel_zoom(self, ev: QWheelEvent):
        step = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
        newz = min(4.0, max(0.05, self._zoom * step))
        if abs(newz - self._zoom) < 1e-4:
            return
        self._zoom = newz
        self.view.resetTransform()
        self.view.scale(self._zoom, self._zoom)

    # ── Undo/Redo
    def _undo_step(self):
        if not self._undo: 
            return
        self._redo.append(self._image.copy())
        self._image = self._undo.pop()
        self._display = self._image.copy()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    def _redo_step(self):
        if not self._redo: 
            return
        self._undo.append(self._image.copy())
        self._image = self._redo.pop()
        self._display = self._image.copy()
        self._update_display_autostretch()
        self._update_undo_redo_buttons()

    # ── Commit back to the document
    def _commit_to_doc(self):
        # convert back to original channels if needed
        out = self._image
        if self._orig_mono:
            # collapse to single channel
            mono = np.mean(out, axis=2, dtype=np.float32)
            if len(self._orig_shape) == 3 and self._orig_shape[2] == 1:
                mono = mono[:, :, None]
            out = mono.astype(np.float32, copy=False)
        else:
            # ensure 3 channels
            if out.ndim == 2:
                out = np.repeat(out[:, :, None], 3, axis=2)
            elif out.ndim == 3 and out.shape[2] >= 3:
                out = out[:, :, :3]

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        # Try common doc APIs
        applied = False
        try:
            if hasattr(self._doc, "set_image"):
                self._doc.set_image(out, step_name="Blemish Blaster"); applied = True
            elif hasattr(self._doc, "apply_numpy"):
                self._doc.apply_numpy(out, step_name="Blemish Blaster"); applied = True
            elif hasattr(self._doc, "image"):
                self._doc.image = out; applied = True
        except Exception as e:
            QMessageBox.critical(self, "Blemish Blaster", f"Failed to write to document:\n{e}")
            return

        if applied and hasattr(self.parent(), "_refresh_active_view"):
            try: self.parent()._refresh_active_view()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        self.accept()

    # ── display helpers
    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        if arr.ndim == 2:
            h, w = arr.shape
            arr = np.repeat(arr[:, :, None], 3, axis=2)
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        else:
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _refresh_pix(self):
        self.pix.setPixmap(self._np_to_qpix(self._display))
        # auto-fit only on first paint; here just ensure the circle hides until move
        self.circle.setVisible(False)
