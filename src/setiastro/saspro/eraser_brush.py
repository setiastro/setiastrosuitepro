# src/setiastro/saspro/eraser_brush.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Eraser brush for Clone Stamp and Blemish Blaster.

The eraser paints a fill through the same feathered brush the clone uses, as if
the clone source were a patch of one flat value:

    black         0.0 in every channel (SASpro's no-data value)
    image median  each channel's own median (R, G and B separately)
    color         a chosen color, its values written straight into the image

Typical use: nebula remnants or other artifacts left in a stars-only image.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QColorDialog, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
)


def fill_dab_inplace(img: np.ndarray, tx: int, ty: int, color: np.ndarray,
                     mask_full: np.ndarray, r: int, opacity: float) -> None:
    """One eraser dab, in place: img (HxWx3 float32) becomes
    (1 - a) * img + a * color under the brush, a = mask * opacity.
    Where a is 1 the result is exactly `color` (black is exactly 0.0)."""
    h, w = img.shape[:2]
    x0, y0 = tx - r, ty - r
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, tx + r + 1), min(h, ty + r + 1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    a = mask_full[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * np.float32(opacity)
    if a.max() <= 0:
        return
    a3 = a[:, :, None]
    tgt = img[cy0:cy1, cx0:cx1, :]
    tgt[:] = (1.0 - a3) * tgt + a3 * np.asarray(color, np.float32).reshape(1, 1, 3)


def display_fill(linear: np.ndarray, display: np.ndarray, color: np.ndarray,
                 linked: bool = False, max_samples: int = 250_000) -> np.ndarray:
    """The value `color` shows as in a preview buffer.

    The auto-stretch preview is, per channel, a non-decreasing function of the
    pixel value with f(0) = 0 and f(1) = 1, so the fill's preview value can be
    read off the image's own (value, preview) pairs instead of re-running the
    stretch: linear interpolation between the nearest sampled values below and
    above (np.interp on the sorted samples, without the sort). A linked stretch
    (or a mono image) is one function for all three channels, so linked=True
    pools their samples: a fill outside one channel's own range still lands
    between real samples."""
    color = np.asarray(color, np.float32)
    if display is linear:
        return color.copy()
    h, w = linear.shape[:2]
    s = max(1, int(math.ceil(math.sqrt(h * w / float(max_samples)))))
    xs = np.ascontiguousarray(linear[::s, ::s, :3]).reshape(-1, 3)
    ys = np.ascontiguousarray(display[::s, ::s, :3]).reshape(-1, 3)
    out = np.empty(3, np.float32)
    for c in range(3):
        if linked:
            x, y = xs.reshape(-1), ys.reshape(-1)
        else:
            x, y = xs[:, c], ys[:, c]
        v = float(color[c])
        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0          # f(0) = 0, f(1) = 1
        below = np.flatnonzero(x <= v)
        if below.size:
            i = below[np.argmax(x[below])]
            if x[i] >= x0:
                x0, y0 = float(x[i]), float(y[i])
        above = np.flatnonzero(x >= v)
        if above.size:
            j = above[np.argmin(x[above])]
            if x[j] <= x1:
                x1, y1 = float(x[j]), float(y[j])
        out[c] = y0 if x1 <= x0 else y0 + (y1 - y0) * (v - x0) / (x1 - x0)
    return np.clip(out, 0.0, 1.0)


def channel_medians(img: np.ndarray, mono: bool) -> np.ndarray:
    """Each channel's median of an HxWx3 image (3 floats). A mono image's
    three channels are copies, so its one median is used for all three."""
    from setiastro.saspro.imageops.stretch import _channel_medians
    src = img[..., :1] if mono else img[..., :3]
    med = _channel_medians(src).astype(np.float32)
    if not np.all(np.isfinite(med)):
        med = np.array([np.nanmedian(src[..., c]) for c in range(src.shape[2])], np.float32)
        med = np.nan_to_num(med, nan=0.0)
    if mono:
        med = np.repeat(med[:1], 3)
    return med


class StrokeTiles:
    """Pre-stroke copies of the 64x64 tiles a brush stroke touches, so one
    stroke is one undo entry (the Clone Stamp's scheme). patches is a list of
    (y0, x0, copy); the tiles never overlap."""
    TILE = 64

    def __init__(self, img: np.ndarray):
        self.img = img
        self.seen = set()
        self.patches = []

    def touch(self, tx: int, ty: int, r: int) -> None:
        h, w = self.img.shape[:2]
        x0, x1 = max(0, tx - r), min(w, tx + r + 1)
        y0, y1 = max(0, ty - r), min(h, ty + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        T = self.TILE
        for tiy in range(y0 // T, (y1 - 1) // T + 1):
            for tix in range(x0 // T, (x1 - 1) // T + 1):
                if (tiy, tix) in self.seen:
                    continue
                self.seen.add((tiy, tix))
                py0, px0 = tiy * T, tix * T
                self.patches.append((py0, px0, self.img[py0:py0 + T, px0:px0 + T].copy()))


class EraserFillPicker(QWidget):
    """Black / Image median / Color for the eraser. fill() is the value the
    eraser paints (3 floats in the image's own units). The choice and the color
    are remembered, shared by Clone Stamp and Blemish Blaster."""

    KEY_FILL = "eraser/fill"
    KEY_COLOR = "eraser/color"

    def __init__(self, image_fn, mono: bool, parent=None):
        super().__init__(parent)
        self._image_fn = image_fn          # () -> the dialog's working HxWx3 image
        self._mono = bool(mono)
        self._median = None

        s = QSettings()
        kind = str(s.value(self.KEY_FILL, "black") or "black")
        col = QColor(str(s.value(self.KEY_COLOR, "#808080") or "#808080"))
        self._color = col if col.isValid() else QColor(128, 128, 128)

        self.rb_black = QRadioButton(self.tr("Black"))
        self.rb_median = QRadioButton(self.tr("Image median"))
        self.rb_color = QRadioButton(self.tr("Color"))
        self.rb_black.setToolTip(self.tr("Paints 0.0 in every channel."))
        if self._mono:
            self.rb_median.setToolTip(self.tr("Paints the image's median, measured when you choose this."))
        else:
            self.rb_median.setToolTip(self.tr(
                "Paints each channel's own median (R, G and B separately), measured when you choose this."))
        self.rb_color.setToolTip(self.tr(
            "Paints the chosen color's values straight into the image. "
            "With Auto-stretch preview on, it shows brighter than the swatch."))
        self._group = QButtonGroup(self)
        for i, rb in enumerate((self.rb_black, self.rb_median, self.rb_color)):
            self._group.addButton(rb, i)

        self.btn_color = QPushButton(self)
        self.btn_color.setFixedSize(34, 20)
        self.btn_color.setToolTip(self.tr("Choose the fill color"))
        self.btn_color.clicked.connect(self._pick_color)

        self.lbl_value = QLabel(self)
        self.lbl_value.setStyleSheet("color:#888; font-size:10px;")
        self.lbl_value.setWordWrap(True)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.rb_black)
        row.addWidget(self.rb_median)
        row.addWidget(self.rb_color)
        row.addWidget(self.btn_color)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(row)
        lay.addWidget(self.lbl_value)

        {"median": self.rb_median, "color": self.rb_color}.get(kind, self.rb_black).setChecked(True)
        self._paint_swatch()
        self._group.idToggled.connect(self._on_toggled)

    # ── the fill ───────────────────────────────────────────────
    def kind(self) -> str:
        if self.rb_median.isChecked():
            return "median"
        if self.rb_color.isChecked():
            return "color"
        return "black"

    def fill(self) -> np.ndarray:
        """The value the eraser paints, 3 floats."""
        k = self.kind()
        if k == "median":
            if self._median is None:
                self._measure_median()
            if self._median is not None:
                return self._median.copy()
            return np.zeros(3, np.float32)
        if k == "color":
            return self._color_values()
        return np.zeros(3, np.float32)

    def _color_values(self) -> np.ndarray:
        r, g, b = self._color.getRgbF()[:3]
        v = np.array([r, g, b], np.float32)
        if self._mono:
            v[:] = np.float32(v.mean())      # what a mono image keeps on Apply
        return v

    def _measure_median(self) -> None:
        img = self._image_fn()
        if img is None:
            self._median = None
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._median = channel_medians(img, self._mono)
        finally:
            QApplication.restoreOverrideCursor()

    # ── UI ─────────────────────────────────────────────────────
    def _on_toggled(self, _id: int, on: bool) -> None:
        if not on:
            return
        if self.kind() == "median":
            self._median = None              # measured fresh each time it is chosen
            if self.isVisible():
                self._measure_median()
        try:
            QSettings().setValue(self.KEY_FILL, self.kind())
        except Exception:
            pass
        self._show_value()

    def _pick_color(self) -> None:
        col = QColorDialog.getColor(self._color, self, self.tr("Eraser color"))
        if not col.isValid():
            return
        self._color = col
        try:
            QSettings().setValue(self.KEY_COLOR, col.name())
        except Exception:
            pass
        self._paint_swatch()
        if self.rb_color.isChecked():
            self._show_value()
        else:
            self.rb_color.setChecked(True)

    def _paint_swatch(self) -> None:
        v = self._color_values()
        q = QColor.fromRgbF(float(v[0]), float(v[1]), float(v[2]))
        self.btn_color.setStyleSheet(f"background:{q.name()}; border:1px solid #666;")

    def _show_value(self) -> None:
        k = self.kind()
        if k == "median" and self._median is None:
            self.lbl_value.setText("")
            return
        v = self.fill()
        if self._mono or (v[0] == v[1] == v[2]):
            self.lbl_value.setText(self.tr("Fills with {0}").format(f"{float(v[0]):.4g}"))
        else:
            self.lbl_value.setText(self.tr("Fills with R {0}  G {1}  B {2}").format(
                *(f"{float(x):.4g}" for x in v)))

    def showEvent(self, ev):
        super().showEvent(ev)
        if self.kind() == "median" and self._median is None:
            self._measure_median()
        self._show_value()
