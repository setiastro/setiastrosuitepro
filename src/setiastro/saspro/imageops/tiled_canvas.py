# saspro/tiled_canvas.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
#
# Drop-in replacement for ViewportImageCanvas that caches rasterized tiles so
# pan and zoom are pure blits instead of per-frame re-rasterization.
#
# Model (CPU-only; no GPU/QRhi):
#   * The image is a fixed grid of source-space tiles (TILE_DIM px each). Each
#     visible tile is rasterized ONCE at source resolution -- crop -> LUT/mask
#     -> uint8 RGB QImage -> QPixmap -- via owner._render_source_tile(), and
#     cached. paintEvent blits the tiles that intersect the exposed rect,
#     letting QPainter scale each cached tile to the current zoom.
#   * Because tiles are cached at SOURCE resolution, changing zoom does not
#     rebuild them: the same pixmaps are drawn at a different dest rect. Panning
#     likewise reuses tiles. Rasterization happens only on first sight of a tile
#     or when the LUT changes (invalidate()).
#   * A small whole-image proxy (downscaled from owner._pm_src, <= PROXY_MAX on
#     the long edge) is drawn first as a backdrop, so not-yet-built tiles show an
#     approximate image rather than a gap, and a far zoom-out can skip building
#     every tile.
#   * Tiles are LRU-evicted once the cache exceeds CACHE_BUDGET_BYTES.
#
# Owner contract (ImageSubWindow):
#   owner.scale                      -> float, current zoom (widget = src*scale)
#   owner._render_source_tile(sx,sy,sw,sh) -> QImage|None
#           Source-resolution RGB of that source rect with the display LUT and
#           mask overlay applied. NO geometric scaling (the canvas scales on
#           blit). Runs on the GUI thread in Stage 1.
#   owner._pm_src                    -> QPixmap|None, the capped full pixmap
#           _render() already maintains; used only to build the proxy.
#
# Stage 1: synchronous tile fill (still far cheaper than re-rasterizing every
# frame). The _get_tile miss path is the single seam where a QThreadPool fill
# (Stage 2) drops in later.

from __future__ import annotations

from collections import OrderedDict

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPixmap

from setiastro.saspro.imageops.viewport_canvas import ViewportImageCanvas

TILE_DIM = 256                     # source px per tile (grid cell)
PROXY_MAX = 2048                   # long-edge cap of the backdrop proxy
CACHE_BUDGET_BYTES = 256 * 1024 * 1024   # ~256 MB of cached tiles


# ---- pure geometry / budget helpers (no Qt; unit-tested) ------------------ #
def visible_tile_range(vx, vy, vw, vh, scale, tile_dim, src_w, src_h):
    """Inclusive (tx0, ty0, tx1, ty1) tile indices covering the widget-space
    rect (vx, vy, vw, vh) at `scale`, clamped to the image. Returns None if the
    rect maps outside the image or inputs are degenerate."""
    if scale <= 0 or src_w <= 0 or src_h <= 0 or vw <= 0 or vh <= 0:
        return None
    sx0 = int((vx) / scale)
    sy0 = int((vy) / scale)
    sx1 = int((vx + vw) / scale) + 1
    sy1 = int((vy + vh) / scale) + 1
    sx0 = max(0, min(sx0, src_w - 1))
    sy0 = max(0, min(sy0, src_h - 1))
    sx1 = max(0, min(sx1, src_w))
    sy1 = max(0, min(sy1, src_h))
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    return (sx0 // tile_dim, sy0 // tile_dim,
            (sx1 - 1) // tile_dim, (sy1 - 1) // tile_dim)


def tile_source_rect(tx, ty, tile_dim, src_w, src_h):
    """Source rect (sx, sy, sw, sh) of tile (tx, ty), clamped at right/bottom
    edges. Returns None if the tile is entirely outside the image."""
    sx = tx * tile_dim
    sy = ty * tile_dim
    if sx >= src_w or sy >= src_h or sx < 0 or sy < 0:
        return None
    sw = min(tile_dim, src_w - sx)
    sh = min(tile_dim, src_h - sy)
    if sw <= 0 or sh <= 0:
        return None
    return (sx, sy, sw, sh)


def evict_to_budget(od: "OrderedDict", budget_bytes: int, bytes_of):
    """Pop oldest entries from an insertion-ordered dict until the summed
    bytes_of(value) is within budget. Returns the number evicted. Assumes the
    most-recently-used items were moved to the end (move_to_end on hit)."""
    total = sum(bytes_of(v) for v in od.values())
    evicted = 0
    while total > budget_bytes and len(od) > 1:
        _key, val = od.popitem(last=False)   # oldest
        total -= bytes_of(val)
        evicted += 1
    return evicted


class TiledImageCanvas(ViewportImageCanvas):
    # Subclass of ViewportImageCanvas so every isinstance(self.label,
    # ViewportImageCanvas) check in subwindow.py keeps matching. The base
    # __init__ sets _owner, _full_w/_full_h, _override_pm and WA_OpaquePaintEvent;
    # we override request_full_size/pixmap/clear/paintEvent below.
    def __init__(self, owner, *args, **kwargs):
        super().__init__(owner, *args, **kwargs)
        self._src_w = 0
        self._src_h = 0
        self._tiles: "OrderedDict[tuple, QPixmap]" = OrderedDict()
        self._proxy = None                       # QPixmap backdrop (small)
        self._proxy_dirty = True
        self._interacting = False                # fast filter during gestures

    # -- sizing / lifecycle ------------------------------------------------- #
    def request_full_size(self, w: int, h: int, src_w: int, src_h: int):
        """Set the full (scaled) widget size and the source dimensions, then
        repaint. Clears any layer-preview override. Tiles are scale-independent,
        so a pure zoom (same src, new full size) keeps the whole cache."""
        w = max(1, int(w)); h = max(1, int(h))
        src_w = max(1, int(src_w)); src_h = max(1, int(src_h))
        self._override_pm = None
        if (src_w, src_h) != (self._src_w, self._src_h):
            # A different image/source: the tile grid changes meaning -> reset.
            self._src_w, self._src_h = src_w, src_h
            self._tiles.clear()
            self._proxy = None
            self._proxy_dirty = True
        if (w, h) != (self._full_w, self._full_h):
            self._full_w, self._full_h = w, h
            self.setFixedSize(w, h)
        self.update()

    def full_size(self):
        return self._full_w, self._full_h

    def invalidate(self):
        """LUT/data changed: drop cached tiles and the proxy so they rebuild
        with the new mapping. Call this where _render used to rebuild _pm_src."""
        self._tiles.clear()
        self._proxy = None
        self._proxy_dirty = True
        self.update()

    def set_interacting(self, on: bool):
        """During an active pan/zoom gesture, prefer the fast (nearest) filter;
        on settle, allow smooth downscaling."""
        self._interacting = bool(on)

    # -- drop-in QLabel surface --------------------------------------------- #
    def setPixmap(self, pm):                              # type: ignore[override]
        self._override_pm = pm if (pm is not None and not pm.isNull()) else None
        self.update()

    def pixmap(self):                                    # type: ignore[override]
        # Whole-image proxy is a fine thumbnail for drag consumers.
        return self._proxy

    def clear(self):                                     # type: ignore[override]
        self._override_pm = None
        self._tiles.clear()
        self._proxy = None
        self._proxy_dirty = True
        self._full_w = self._full_h = 1
        self._src_w = self._src_h = 0
        self.setFixedSize(1, 1)
        self.update()

    # -- tile / proxy management ------------------------------------------- #
    @staticmethod
    def _pm_bytes(pm) -> int:
        try:
            return pm.width() * pm.height() * (pm.depth() // 8 or 4)
        except Exception:
            return 0

    def _ensure_proxy(self):
        if self._proxy is not None and not self._proxy_dirty:
            return
        self._proxy = None
        self._proxy_dirty = False
        base = getattr(self._owner, "_pm_src", None)
        if base is None or base.isNull():
            return
        long_edge = max(base.width(), base.height())
        if long_edge <= PROXY_MAX:
            self._proxy = base
        else:
            f = PROXY_MAX / float(long_edge)
            self._proxy = base.scaled(
                max(1, int(base.width() * f)), max(1, int(base.height() * f)),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    def _get_tile(self, tx, ty):
        key = (tx, ty)
        pm = self._tiles.get(key)
        if pm is not None:
            self._tiles.move_to_end(key)         # LRU touch
            return pm
        rect = tile_source_rect(tx, ty, TILE_DIM, self._src_w, self._src_h)
        if rect is None:
            return None
        # Stage 2 seam: dispatch this to a QThreadPool and return None until the
        # finished tile lands (proxy covers it meanwhile). Stage 1 fills inline.
        try:
            qimg = self._owner._render_source_tile(*rect)
        except Exception:
            qimg = None
        if qimg is None or qimg.isNull():
            return None
        pm = QPixmap.fromImage(qimg)
        self._tiles[key] = pm
        evict_to_budget(self._tiles, CACHE_BUDGET_BYTES, self._pm_bytes)
        return pm

    # -- painting ----------------------------------------------------------- #
    def paintEvent(self, ev):
        p = QPainter(self)
        try:
            if self._override_pm is not None:
                p.drawPixmap(0, 0, self._override_pm)
                return

            scale = float(getattr(self._owner, "scale", 1.0) or 1.0)
            if scale <= 0 or self._src_w <= 0 or self._src_h <= 0:
                return

            rect = ev.rect().intersected(self.rect())
            if rect.isEmpty():
                return

            # Interactive gestures: fast (nearest) scaling; smooth on settle.
            # Upscaling (zoom-in) always stays nearest for crisp pixels.
            smooth = (not self._interacting) and (scale < 1.0)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

            # Backdrop proxy: approximate image under any not-yet-built tile.
            self._ensure_proxy()
            if self._proxy is not None:
                p.drawPixmap(
                    QRectF(0.0, 0.0, float(self._full_w), float(self._full_h)),
                    self._proxy,
                    QRectF(0.0, 0.0, float(self._proxy.width()), float(self._proxy.height())),
                )

            rng = visible_tile_range(
                rect.x(), rect.y(), rect.width(), rect.height(),
                scale, TILE_DIM, self._src_w, self._src_h,
            )
            if rng is None:
                return
            tx0, ty0, tx1, ty1 = rng
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    pm = self._get_tile(tx, ty)
                    if pm is None:
                        continue          # proxy shows through
                    dx = (tx * TILE_DIM) * scale
                    dy = (ty * TILE_DIM) * scale
                    p.drawPixmap(
                        QRectF(dx, dy, pm.width() * scale, pm.height() * scale),
                        pm,
                        QRectF(0.0, 0.0, float(pm.width()), float(pm.height())),
                    )
        finally:
            p.end()