# pro/widgets/graphics_views.py
"""
Centralized graphics view widgets for Seti Astro Suite Pro.

Provides reusable zoomable and interactive QGraphicsView widgets.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWheelEvent, QPainter
from PyQt6.QtWidgets import QGraphicsView


class ZoomableGraphicsView(QGraphicsView):
    """
    A QGraphicsView with mouse wheel zoom (Ctrl+wheel) and drag support.
    
    Features:
    - Ctrl+wheel to zoom in/out
    - Scroll hand drag mode
    - Smooth pixmap transform
    - Configurable zoom limits and step
    
    Usage:
        view = ZoomableGraphicsView()
        view.setScene(scene)
        view.zoom_in()
        view.zoom_out()
        view.fit_to_item(pixmap_item)
    """
    
    def __init__(self, scene=None, parent=None, *,
                 zoom_min: float = 0.05,
                 zoom_max: float = 12.0,
                 zoom_step: float = 1.25):
        super().__init__(parent)
        if scene is not None:
            self.setScene(scene)
        
        self._zoom = 1.0
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._zoom_step = zoom_step
        
        # Default configuration
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle wheel events - zoom with Ctrl modifier."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                event.accept()
                return
            self._apply_zoom(
                up=(delta > 0),
                anchor=QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )
            event.accept()
        else:
            super().wheelEvent(event)
    
    def _apply_zoom(self, up: bool, anchor: QGraphicsView.ViewportAnchor | None = None):
        """Apply zoom in/out with optional anchor point."""
        old_anchor = self.transformationAnchor()
        if anchor is not None:
            self.setTransformationAnchor(anchor)
        
        step = self._zoom_step if up else (1.0 / self._zoom_step)
        new_zoom = max(self._zoom_min, min(self._zoom_max, self._zoom * step))
        factor = new_zoom / self._zoom
        
        if factor != 1.0:
            self.scale(factor, factor)
            self._zoom = new_zoom
        
        if anchor is not None:
            self.setTransformationAnchor(old_anchor)
    
    def zoom_in(self):
        """Zoom in centered on the view."""
        self._apply_zoom(True, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)
    
    def zoom_out(self):
        """Zoom out centered on the view."""
        self._apply_zoom(False, anchor=QGraphicsView.ViewportAnchor.AnchorViewCenter)
    
    def fit_to_item(self, item):
        """Fit the view to show the entire item."""
        if item is None:
            return
        # Handle both pixmap items and generic items
        if hasattr(item, 'pixmap') and item.pixmap().isNull():
            return
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
    
    # Alias for compatibility
    fit_item = fit_to_item
    
    @property
    def zoom_level(self) -> float:
        """Get current zoom level."""
        return self._zoom
    
    def set_zoom(self, level: float):
        """Set zoom to a specific level."""
        level = max(self._zoom_min, min(self._zoom_max, level))
        factor = level / self._zoom
        if factor != 1.0:
            self.scale(factor, factor)
            self._zoom = level
    
    def reset_zoom(self):
        """Reset zoom to 1:1."""
        self.resetTransform()
        self._zoom = 1.0
