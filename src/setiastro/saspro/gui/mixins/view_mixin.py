# pro/gui/mixins/view_mixin.py
"""
View management mixin for AstroSuiteProMainWindow.

This mixin contains all view-related functionality: tiling, cascading,
zooming, autostretch, and view layout management.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

if TYPE_CHECKING:
    pass


class ViewMixin:
    """
    Mixin for view management.
    
    Provides methods for arranging, zooming, and managing MDI subwindows.
    """

    def _auto_fit_all_subwindows(self):
        """Apply auto-fit to every visible subwindow when the mode is enabled."""
        if not getattr(self, "_auto_fit_on_resize", False):
            return

        subs = self._visible_subwindows()
        if not subs:
            return

        # Remember current active so we can restore it
        prev_active = self.mdi.activeSubWindow()

        for sw in subs:
            # Make this subwindow active so _zoom_active_fit() works on it
            self.mdi.setActiveSubWindow(sw)
            self._zoom_active_fit()

        # Restore previously active subwindow if still around
        if prev_active and prev_active in subs:
            self.mdi.setActiveSubWindow(prev_active)

    def _visible_subwindows(self):
        """Return list of visible, non-minimized subwindows."""
        subs = [sw for sw in self.mdi.subWindowList()
                if sw.isVisible() and not (sw.windowState() & Qt.WindowState.WindowMinimized)]
        return subs


    def _tile_views(self):
        """Tile all subwindows."""
        self.mdi.tileSubWindows()
        self._auto_fit_all_subwindows()

    def _tile_views_direction(self, direction: str):
        """
        Tile views in a specific direction.
        
        Args:
            direction: 'v' for vertical columns, 'h' for horizontal rows
        """
        subs = self._visible_subwindows()
        if not subs:
            return
        area = self.mdi.viewport().rect()
        # account for MDI viewport origin in global coords
        off = self.mdi.viewport().mapTo(self.mdi, area.topLeft())
        origin_x, origin_y = off.x(), off.y()

        n = len(subs)
        if direction == "v":  # columns
            col_w = max(1, area.width() // n)
            for i, sw in enumerate(subs):
                sw.setGeometry(origin_x + i*col_w, origin_y, col_w, area.height())
        else:  # rows
            row_h = max(1, area.height() // n)
            for i, sw in enumerate(subs):
                sw.setGeometry(origin_x, origin_y + i*row_h, area.width(), row_h)

        self._auto_fit_all_subwindows()

    def _tile_views_grid(self):
        """Arrange subwindows in a near-square grid across the MDI area."""
        subs = self._visible_subwindows()
        if not subs:
            return
        area = self.mdi.viewport().rect()
        off = self.mdi.viewport().mapTo(self.mdi, area.topLeft())
        origin_x, origin_y = off.x(), off.y()

        n = len(subs)
        # rows x cols ~ square
        cols = int(max(1, math.ceil(math.sqrt(n))))
        rows = int(max(1, math.ceil(n / cols)))

        cell_w = max(1, area.width() // cols)
        cell_h = max(1, area.height() // rows)

        for idx, sw in enumerate(subs):
            r = idx // cols
            c = idx % cols
            sw.setGeometry(origin_x + c*cell_w, origin_y + r*cell_h, cell_w, cell_h)

        self._auto_fit_all_subwindows()

    def _zoom_step_active(self, direction: int):
        """
        Zoom the active view in or out by a fixed factor.
        
        Args:
            direction: > 0 for zoom in, < 0 for zoom out
        """
        sw = self.mdi.activeSubWindow()
        if not sw:
            return

        view = sw.widget()
        try:
            cur_scale = float(getattr(view, "scale", 1.0))
        except Exception:
            cur_scale = 1.0

        # Reasonable step factor
        step = 1.25
        factor = step if direction > 0 else 1.0 / step

        new_scale = cur_scale * factor
        # Clamp to sane bounds
        new_scale = max(1e-4, min(32.0, new_scale))

        # Manual zoom -> we are no longer in a "perfect fit" state
        try:
            self.act_zoom_fit.setChecked(False)
        except Exception:
            pass

        # Prefer anchor-based zoom so we keep the current scroll-center stable
        if hasattr(view, "_zoom_at_anchor") and callable(view._zoom_at_anchor):
            try:
                rel = float(new_scale) / max(cur_scale, 1e-12)
                view._zoom_at_anchor(rel)
                return
            except Exception:
                pass

        # Fallback: absolute set_scale without forcing recentering
        if hasattr(view, "set_scale") and callable(view.set_scale):
            try:
                view.set_scale(float(new_scale))
                return
            except Exception:
                pass

    def _zoom_active_1_1(self):
        """Zoom active view to 100% (1:1 pixel scale)."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(view, "set_scale") and callable(view.set_scale):
            try:
                view.set_scale(1.0)
            except Exception:
                pass

    def _zoom_active_fit(self):
        """Fit the active view's image to its viewport."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        self._zoom_active_1_1()
        
        # Get sizes
        img_w, img_h = self._infer_image_size(view)
        if not img_w or not img_h:
            return

        vp = self._viewport_widget(view)
        vw, vh = max(1, vp.width()), max(1, vp.height())

        # Compute uniform scale (minus a hair to avoid scrollbars fighting)
        scale = min((vw - 2) / img_w, (vh - 2) / img_h)
        # Clamp to sane bounds
        scale = max(1e-4, min(32.0, scale))
        self._sync_fit_auto_visual()
        
        # Apply using view API if available
        if hasattr(view, "set_scale") and callable(view.set_scale):
            try:
                view.set_scale(float(scale))
                self._ensure_smooth_resample(view)
                self._center_view(view)
                return
            except Exception:
                pass

        # Fallback: relative zoom using _zoom_at_anchor
        try:
            cur = float(getattr(view, "scale", 1.0))
            factor = scale / max(cur, 1e-12)
            if hasattr(view, "_zoom_at_anchor") and callable(view._zoom_at_anchor):
                view._zoom_at_anchor(float(factor))
                self._center_view(view)
                return
        except Exception:
            pass

    def _ensure_smooth_resample(self, view):
        """
        Make sure the view is using smooth interpolation for the current scale.
        Different view widgets in SASpro may implement this differently, so we
        try a few known hooks safely.
        """
        # 1) Best case: explicit API
        for name in ("set_smooth_scaling", "set_interpolation", "set_smooth", "enable_smooth_scaling"):
            fn = getattr(view, name, None)
            if callable(fn):
                try:
                    fn(True)
                    return
                except Exception:
                    pass

        # 2) Some views store a mode flag
        for attr in ("smooth_scaling", "_smooth_scaling", "_use_smooth_scaling", "use_smooth_scaling"):
            if hasattr(view, attr):
                try:
                    setattr(view, attr, True)
                    # kick a repaint/update if available
                    try:
                        view.update()
                    except Exception:
                        pass
                    return
                except Exception:
                    pass

        # 3) QLabel pixmap scaling: if you have a custom "rebuild pixmap" method, call it
        for name in ("_rebuild_pixmap", "_update_pixmap", "_render_scaled", "rebuild_pixmap"):
            fn = getattr(view, name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass


    def _infer_image_size(self, view):
        """Return (img_w, img_h) in device-independent pixels (ints), best-effort."""
        # Preferred: from the label's pixmap
        try:
            pm = getattr(view, "label", None).pixmap() if hasattr(view, "label") else None
            if pm and not pm.isNull():
                dpr = max(1.0, float(pm.devicePixelRatio()))
                return int(round(pm.width() / dpr)), int(round(pm.height() / dpr))
        except Exception:
            pass

        # Next: from the document image
        try:
            doc = getattr(view, "document", None)
            if doc and getattr(doc, "image", None) is not None:
                import numpy as np
                h, w = np.asarray(doc.image).shape[:2]
                return int(w), int(h)
        except Exception:
            pass

        # Fallback: from attributes some views keep
        for w_key, h_key in (("image_width", "image_height"), ("_img_w", "_img_h")):
            w = getattr(view, w_key, None)
            h = getattr(view, h_key, None)
            if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                return int(w), int(h)

        return None, None

    def _viewport_widget(self, view):
        """Return the viewport widget used to display the image."""
        try:
            if hasattr(view, "scroll") and hasattr(view.scroll, "viewport"):
                return view.scroll.viewport()
            # Some views are QGraphicsView/QAbstractScrollArea-like
            if hasattr(view, "viewport"):
                return view.viewport()
        except Exception:
            pass
        # Worst case: the view itself
        return view

    def _center_view(self, view):
        """Center the content after a zoom change, if possible."""
        try:
            vp = self._viewport_widget(view)
            hbar = view.scroll.horizontalScrollBar() if hasattr(view, "scroll") else None
            vbar = view.scroll.verticalScrollBar() if hasattr(view, "scroll") else None
            lbl = getattr(view, "label", None)
            if vp and hbar and vbar and lbl:
                cx = max(0, lbl.width() // 2 - vp.width() // 2)
                cy = max(0, lbl.height() // 2 - vp.height() // 2)
                hbar.setValue(min(hbar.maximum(), cx))
                vbar.setValue(min(vbar.maximum(), cy))
        except Exception:
            pass

    def _sync_fit_auto_visual(self):
        """Sync the Fit button's checked state with auto-fit mode."""
        on = bool(getattr(self, "_auto_fit_on_resize", False))
        if hasattr(self, "act_zoom_fit"):
            self.act_zoom_fit.blockSignals(True)
            try:
                self.act_zoom_fit.setChecked(on)
            finally:
                self.act_zoom_fit.blockSignals(False)

    def _toggle_auto_fit_on_resize(self, checked: bool):
        """Toggle auto-fit on resize mode."""
        self._auto_fit_on_resize = bool(checked)
        self.settings.setValue("view/auto_fit_on_resize", self._auto_fit_on_resize)
        self._sync_fit_auto_visual()
        if checked:
            self._zoom_active_fit()

    def _on_view_resized(self):
        """Called whenever an ImageSubWindow emits resized(). Debounced."""
        if not getattr(self, "_auto_fit_on_resize", False):
            return
        if hasattr(self, "_auto_fit_timer") and self._auto_fit_timer is not None:
            if self._auto_fit_timer.isActive():
                self._auto_fit_timer.stop()
            self._auto_fit_timer.start()

    def _apply_auto_fit_resize(self):
        """Run the actual Fit after the resize settles."""
        if not getattr(self, "_auto_fit_on_resize", False):
            return
        self._zoom_active_fit()

    def _toggle_autostretch(self, on: bool):
        """Toggle autostretch for the active view."""
        sw = self.mdi.activeSubWindow()
        if sw:
            sw.widget().set_autostretch(on)
            self._log(f"Display-Stretch {'ON' if on else 'OFF'} -> {sw.windowTitle()}")

    def _set_hard_autostretch_from_action(self, checked: bool):
        """Set hard autostretch profile from toolbar action."""
        from PyQt6.QtCore import QSignalBlocker
        
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()

        # mirror the action's check to the view profile
        if hasattr(view, "set_autostretch_profile"):
            view.set_autostretch_profile("hard" if checked else "normal")

        # ensure it's visible
        if not getattr(view, "autostretch_enabled", False):
            view.set_autostretch(True)
            self._sync_autostretch_action(True)

        self._log(f"Display-Stretch profile -> {'HARD' if checked else 'NORMAL'}  ({sw.windowTitle()})")

    def _toggle_hard_autostretch(self):
        """Toggle between hard and normal autostretch profiles."""
        from PyQt6.QtCore import QSignalBlocker
        
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        
        # flip profile
        new_profile = "hard" if not getattr(view, "is_hard_autostretch", lambda: False)() else "normal"
        if hasattr(view, "set_autostretch_profile"):
            view.set_autostretch_profile(new_profile)
            
        # ensure autostretch is ON so the change is visible immediately
        if not getattr(view, "autostretch_enabled", False):
            view.set_autostretch(True)
            self._sync_autostretch_action(True)

        # reflect in toolbar button
        with QSignalBlocker(self.act_hardstretch):
            self.act_hardstretch.setChecked(new_profile == "hard")

        self._log(f"Display-Stretch profile -> {new_profile.upper()}  ({sw.windowTitle()})")

    def _sync_autostretch_action(self, on: bool):
        """Sync the autostretch action's checked state."""
        from PyQt6.QtCore import QSignalBlocker
        
        if hasattr(self, "act_autostretch"):
            block = QSignalBlocker(self.act_autostretch)
            self.act_autostretch.setChecked(bool(on))

    def _edit_display_target(self):
        """Open dialog to edit display stretch target median."""
        from PyQt6.QtWidgets import QInputDialog
        
        cur = float(self.settings.value("display/target", 0.30, type=float))
        val, ok = QInputDialog.getDouble(
            self, "Target Median", "Target (0.01 - 0.90):", cur, 0.01, 0.90, 3
        )
        if not ok:
            return
        self.settings.setValue("display/target", float(val))
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(view, "set_autostretch_target"):
            view.set_autostretch_target(float(val))
        if not getattr(view, "autostretch_enabled", False):
            if hasattr(view, "set_autostretch"):
                view.set_autostretch(True)
            self._sync_autostretch_action(True)

    def _edit_display_sigma(self):
        """Open dialog to edit display stretch sigma."""
        from PyQt6.QtWidgets import QInputDialog
        
        cur = float(self.settings.value("display/sigma", 5.0, type=float))
        val, ok = QInputDialog.getDouble(
            self, "Sigma", "Sigma (0.5 - 10.0):", cur, 0.5, 10.0, 2
        )
        if not ok:
            return
        self.settings.setValue("display/sigma", float(val))
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        if hasattr(view, "set_autostretch_sigma"):
            view.set_autostretch_sigma(float(val))
        if not getattr(view, "autostretch_enabled", False):
            if hasattr(view, "set_autostretch"):
                view.set_autostretch(True)
            self._sync_autostretch_action(True)

    def _copy_active_view(self):
        """Copy the current view state (zoom/pan) for pasting to other views."""
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        self._copied_view_state = {
            "scale": getattr(view, "scale", 1.0),
            "hbar": view.scroll.horizontalScrollBar().value() if hasattr(view, "scroll") else 0,
            "vbar": view.scroll.verticalScrollBar().value() if hasattr(view, "scroll") else 0,
        }
        self._log("View state copied")

    def _paste_active_view(self):
        """Paste a previously copied view state to the active view."""
        if not getattr(self, "_copied_view_state", None):
            return
        sw = self.mdi.activeSubWindow()
        if not sw:
            return
        view = sw.widget()
        state = self._copied_view_state
        
        if hasattr(view, "set_scale"):
            view.set_scale(state.get("scale", 1.0))
        if hasattr(view, "scroll"):
            view.scroll.horizontalScrollBar().setValue(state.get("hbar", 0))
            view.scroll.verticalScrollBar().setValue(state.get("vbar", 0))
        self._log("View state pasted")
