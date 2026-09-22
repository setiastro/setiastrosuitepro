"""Viewport-crop display widget for SASpro image subwindows.

Drop-in replacement for the QLabel that ImageSubWindow puts inside its
QScrollArea. Instead of holding a full-resolution (or 8192-capped) pixmap of
the whole image scaled to the current zoom, this widget reports the FULL scaled
image size to the scroll area (so scrollbar ranges are correct) but paints only
the currently exposed rectangle, rasterized at true 1:1 for the current scale.

Consequences:
  * Peak allocation and paint cost are O(viewport), independent of zoom level
    or image size.
  * There is no per-side render cap, so deep pixel-peeping shows real source
    pixels (nearest-neighbour upsample) rather than an under-resolved overview.

The owner (ImageSubWindow) supplies the pixels through one method:

    owner._rasterize_widget_rect(wx, wy, ww, wh) -> (QImage, ox, oy) | None

where (wx, wy, ww, wh) is the exposed rectangle in widget coordinates and the
returned QImage is drawn at widget offset (ox, oy). (ox, oy) may sit slightly
left/above the exposed rect (it snaps to whole source pixels for exact
alignment); QPainter clips the overhang.

Compatibility surface preserved for existing callers of ``self.label``:
  * ``pixmap()``       returns the most recent viewport crop (a fine drag
                       thumbnail / representative image for existing consumers).
  * ``setPixmap(pm)``  enters an override mode that paints *pm* directly. Used by
                       the live layer-composite preview; cleared by the next
                       ``request_full_size()`` (i.e. the next real render).
  * ``clear()``        drops source/override and repaints empty.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPixmap


class ViewportImageCanvas(QLabel):
    def __init__(self, owner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner = owner
        self._full_w = 1
        self._full_h = 1
        self._override_pm = None        # QPixmap painted directly (layer preview)
        self._last_crop = None          # QPixmap: most recent viewport crop
        # Let Qt clear the (tiny) margin area around a centred small image.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    # -- sizing ------------------------------------------------------------- #
    def request_full_size(self, w: int, h: int) -> None:
        """Set the logical full (scaled) image size and repaint. Clears any
        layer-preview override. Allocates nothing image-sized."""
        w = max(1, int(w))
        h = max(1, int(h))
        self._override_pm = None
        if (w, h) != (self._full_w, self._full_h):
            self._full_w, self._full_h = w, h
            self.setFixedSize(w, h)     # drives QScrollArea scrollbar ranges
        self.update()

    def full_size(self) -> tuple[int, int]:
        return self._full_w, self._full_h

    # -- drop-in QLabel surface --------------------------------------------- #
    def setPixmap(self, pm) -> None:                      # type: ignore[override]
        """Paint *pm* directly until the next request_full_size(). Used by the
        transient layer-composite preview during drags."""
        self._override_pm = pm if (pm is not None and not pm.isNull()) else None
        if self._override_pm is not None:
            self._last_crop = self._override_pm
        self.update()

    def pixmap(self):                                     # type: ignore[override]
        return self._last_crop

    def clear(self) -> None:                              # type: ignore[override]
        self._override_pm = None
        self._last_crop = None
        self._full_w = self._full_h = 1
        self.setFixedSize(1, 1)
        self.update()

    # -- painting ----------------------------------------------------------- #
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        try:
            if self._override_pm is not None:
                p.drawPixmap(0, 0, self._override_pm)
                return

            owner = self._owner
            if owner is None:
                return

            rect = ev.rect().intersected(self.rect())
            if rect.isEmpty():
                return

            res = None
            try:
                res = owner._rasterize_widget_rect(
                    rect.x(), rect.y(), rect.width(), rect.height()
                )
            except Exception:
                res = None
            if not res:
                return

            img, ox, oy = res
            if img is None or img.isNull():
                return

            p.drawImage(int(ox), int(oy), img)
            try:
                self._last_crop = QPixmap.fromImage(img)
            except Exception:
                pass
        finally:
            p.end()