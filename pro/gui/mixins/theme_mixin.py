# pro/gui/mixins/theme_mixin.py
"""
Theme management mixin for AstroSuiteProMainWindow.

This mixin contains all theme-related functionality: palette definitions,
theme application, and system theme detection.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

if TYPE_CHECKING:
    pass


class ThemeMixin:
    """
    Mixin for theme management.
    
    Provides methods for creating palettes, applying themes, and
    responding to system theme changes.
    """

    def _apply_workspace_theme(self):
        """Retint the QMdiArea background + viewport to current theme colors."""
        pal = QApplication.palette()
        # Use Base for light, Window for dark (looks better with your palettes)
        role = QPalette.ColorRole.Base if self._theme_mode() == "light" else QPalette.ColorRole.Window
        col = pal.color(role)

        # 1) Tell QMdiArea to use a flat color background
        try:
            self.mdi.setBackground(QBrush(col))
        except Exception:
            pass

        # 2) Also set the viewport palette (some styles ignore setBackground)
        try:
            vp = self.mdi.viewport()
            vp.setAutoFillBackground(True)
            p = vp.palette()
            p.setColor(QPalette.ColorRole.Window, col)
            vp.setPalette(p)
            vp.update()
        except Exception:
            pass

        # 3) Ensure the overlay canvas stays transparent and refreshes
        try:
            if hasattr(self, "shortcuts") and self.shortcuts and getattr(self.shortcuts, "canvas", None):
                c = self.shortcuts.canvas
                c.setStyleSheet("background: transparent;")
                c.update()
        except Exception:
            pass

    def apply_theme_from_settings(self):
        """Apply the theme based on current settings."""
        mode = self._theme_mode()
        app = QApplication.instance()
        color_scheme = app.styleHints().colorScheme()

        # Resolve "system" to dark/light
        if mode == "system":
            if color_scheme == Qt.ColorScheme.Dark:
                print("System is in Dark Mode")
                mode = "dark"
            else:
                print("System is in Light Mode")
                mode = "light"

        # Base style
        if mode in ("dark", "gray", "light", "custom"):
            app.setStyle("Fusion")
        else:
            app.setStyle(None)

        # Palettes
        if mode == "dark":
            app.setPalette(self._dark_palette())
            app.setStyleSheet(
                "QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #5a5a5a; }"
            )
        elif mode == "gray":
            app.setPalette(self._gray_palette())
            app.setStyleSheet(
                "QToolTip { color: #f0f0f0; background-color: #3a3a3a; border: 1px solid #5a5a5a; }"
            )
        elif mode == "light":
            app.setPalette(self._light_palette())
            app.setStyleSheet(
                "QToolTip { color: #141414; background-color: #ffffee; border: 1px solid #c8c8c8; }"
            )
        elif mode == "custom":
            app.setPalette(self._custom_palette())
            # Tooltips roughly matching the custom dark-ish style
            app.setStyleSheet(
                "QToolTip { color: #f0f0f0; background-color: #303030; border: 1px solid #5a5a5a; }"
            )
        else:  # system/native fallback
            app.setPalette(QApplication.style().standardPalette())
            app.setStyleSheet("")

        # Optional: apply custom font
        if mode == "custom":
            font_str = self.settings.value("ui/custom/font", "", type=str) or ""
            if font_str:
                try:
                    f = QFont()
                    if f.fromString(font_str):
                        app.setFont(f)
                except Exception:
                    pass

        # Nudge widgets to pick up role changes
        self._repolish_top_levels()
        self._apply_workspace_theme()
        self._style_mdi_titlebars()
        self._menu_view_panels = None

        try:
            vp = self.mdi.viewport()
            vp.setAutoFillBackground(True)
            vp.setPalette(QApplication.palette())
            vp.update()
        except Exception:
            pass

    def _repolish_top_levels(self):
        """Force all top-level widgets to repolish their styles."""
        app = QApplication.instance()
        for w in app.topLevelWidgets():
            w.setUpdatesEnabled(False)
            w.style().unpolish(w)
            w.style().polish(w)
            w.setUpdatesEnabled(True)

    def _style_mdi_titlebars(self):
        """Apply theme-specific styles to MDI subwindow titlebars."""
        mode = self._theme_mode()
        if mode == "dark":
            base = "#1b1b1b"   # inactive titlebar
            active = "#242424"  # active titlebar
            fg = "#dcdcdc"
        elif mode in ("gray", "custom"):
            base = "#3a3a3a"
            active = "#454545"
            fg = "#f0f0f0"
        else:
            # No override in light / system modes
            self.mdi.setStyleSheet("")
            return

        self.mdi.setStyleSheet(f"""
            QMdiSubWindow::titlebar        {{ background: {base};  color: {fg}; }}
            QMdiSubWindow::titlebar:active {{ background: {active}; color: {fg}; }}
        """)

    def _dark_palette(self) -> QPalette:
        """Create a dark theme palette."""
        p = QPalette()

        # Bases
        bg = QColor(18, 18, 18)      # editor / view backgrounds (Base)
        panel = QColor(27, 27, 27)   # window / panels (Window, Button)
        altbase = QColor(33, 33, 33)
        text = QColor(220, 220, 220)
        dis = QColor(140, 140, 140)
        hi = QColor(30, 144, 255)    # highlight (dodger blue)

        p.setColor(QPalette.ColorRole.Window, panel)
        p.setColor(QPalette.ColorRole.WindowText, text)
        p.setColor(QPalette.ColorRole.Base, bg)
        p.setColor(QPalette.ColorRole.AlternateBase, altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase, panel)
        p.setColor(QPalette.ColorRole.ToolTipText, text)
        p.setColor(QPalette.ColorRole.Text, text)
        p.setColor(QPalette.ColorRole.Button, panel)
        p.setColor(QPalette.ColorRole.ButtonText, text)
        p.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight, hi)
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        p.setColor(QPalette.ColorRole.Link, QColor(90, 160, 255))
        p.setColor(QPalette.ColorRole.LinkVisited, QColor(160, 140, 255))
        
        # Qt6: explicit placeholder color helps avoid faint-on-faint
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(160, 160, 160))
        except Exception:
            pass

        # Disabled
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(24, 24, 24))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(60, 60, 60))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(210, 210, 210))

        return p

    def _custom_palette(self) -> QPalette:
        """
        Build a QPalette from user-defined colors in QSettings.
        Falls back to a gray-ish baseline if any key is missing.
        """
        s = self.settings

        def col(key: str, default: QColor) -> QColor:
            val = s.value(key, default.name(), type=str) or default.name()
            return QColor(val)

        window = col("ui/custom/window", QColor(54, 54, 54))
        base = col("ui/custom/base", QColor(40, 40, 40))
        altbase = col("ui/custom/altbase", QColor(64, 64, 64))
        text = col("ui/custom/text", QColor(230, 230, 230))
        button = col("ui/custom/button", window)
        hi = col("ui/custom/highlight", QColor(95, 145, 230))
        link = col("ui/custom/link", QColor(120, 170, 255))
        linkv = col("ui/custom/link_visited", QColor(180, 150, 255))

        p = QPalette()

        # Core roles
        p.setColor(QPalette.ColorRole.Window, window)
        p.setColor(QPalette.ColorRole.WindowText, text)
        p.setColor(QPalette.ColorRole.Base, base)
        p.setColor(QPalette.ColorRole.AlternateBase, altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase, window)
        p.setColor(QPalette.ColorRole.ToolTipText, text)
        p.setColor(QPalette.ColorRole.Text, text)
        p.setColor(QPalette.ColorRole.Button, button)
        p.setColor(QPalette.ColorRole.ButtonText, text)
        p.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight, hi)
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        p.setColor(QPalette.ColorRole.Link, link)
        p.setColor(QPalette.ColorRole.LinkVisited, linkv)

        # Placeholder / disabled
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(170, 170, 170))
        except Exception:
            pass

        dis = QColor(150, 150, 150)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, base.darker(115))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, hi.darker(140))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(210, 210, 210))

        return p

    def _gray_palette(self) -> QPalette:
        """Create a mid-gray theme palette."""
        p = QPalette()

        # Mid-gray neutrals
        window = QColor(54, 54, 54)    # panels/docks
        base = QColor(64, 64, 64)      # editors / text fields
        altbase = QColor(72, 72, 72)   # alternating rows
        text = QColor(230, 230, 230)
        btn = window
        dis = QColor(150, 150, 150)
        link = QColor(120, 170, 255)
        linkv = QColor(180, 150, 255)
        hi = QColor(95, 145, 230)
        hitxt = QColor(255, 255, 255)

        # Core roles
        p.setColor(QPalette.ColorRole.Window, window)
        p.setColor(QPalette.ColorRole.WindowText, text)
        p.setColor(QPalette.ColorRole.Base, base)
        p.setColor(QPalette.ColorRole.AlternateBase, altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase, QColor(60, 60, 60))
        p.setColor(QPalette.ColorRole.ToolTipText, text)
        p.setColor(QPalette.ColorRole.Text, text)
        p.setColor(QPalette.ColorRole.Button, btn)
        p.setColor(QPalette.ColorRole.ButtonText, text)
        p.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight, hi)
        p.setColor(QPalette.ColorRole.HighlightedText, hitxt)
        p.setColor(QPalette.ColorRole.Link, link)
        p.setColor(QPalette.ColorRole.LinkVisited, linkv)

        # Placeholder
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(170, 170, 170))
        except Exception:
            pass

        # Disabled group
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(58, 58, 58))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(210, 210, 210))

        return p

    def _light_palette(self) -> QPalette:
        """Create a light theme palette."""
        p = QPalette()

        # Light neutrals
        window = QColor(246, 246, 246)   # panels/docks
        base = QColor(255, 255, 255)     # text fields, editors
        altbase = QColor(242, 242, 242)  # alternating rows
        text = QColor(20, 20, 20)        # primary text
        btn = QColor(246, 246, 246)      # buttons same as window
        dis = QColor(140, 140, 140)      # disabled text
        link = QColor(25, 100, 210)      # link blue
        linkv = QColor(120, 70, 200)     # visited
        hi = QColor(43, 120, 228)        # selection blue (Windows-like)
        hitxt = QColor(255, 255, 255)    # text over selection

        # Core roles
        p.setColor(QPalette.ColorRole.Window, window)
        p.setColor(QPalette.ColorRole.WindowText, text)
        p.setColor(QPalette.ColorRole.Base, base)
        p.setColor(QPalette.ColorRole.AlternateBase, altbase)
        p.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 238))  # soft yellow tooltip
        p.setColor(QPalette.ColorRole.ToolTipText, text)
        p.setColor(QPalette.ColorRole.Text, text)
        p.setColor(QPalette.ColorRole.Button, btn)
        p.setColor(QPalette.ColorRole.ButtonText, text)
        p.setColor(QPalette.ColorRole.BrightText, QColor(180, 0, 0))
        p.setColor(QPalette.ColorRole.Highlight, hi)
        p.setColor(QPalette.ColorRole.HighlightedText, hitxt)
        p.setColor(QPalette.ColorRole.Link, link)
        p.setColor(QPalette.ColorRole.LinkVisited, linkv)

        # Helps line edits/placeholders avoid too-faint gray
        try:
            p.setColor(QPalette.ColorRole.PlaceholderText, QColor(110, 110, 110))
        except Exception:
            pass

        # Disabled group (keep contrasts sane)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, dis)
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(200, 200, 200))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, QColor(120, 120, 120))
        p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(248, 248, 248))

        return p

    def _apply_theme_safely(self):
        """Apply theme with re-entrancy guard."""
        if self._theme_guard:
            return
        self._theme_guard = True
        try:
            self.apply_theme_from_settings()
        finally:
            QTimer.singleShot(0, lambda: setattr(self, "_theme_guard", False))

    def _theme_mode(self) -> str:
        """Get the current theme mode from settings."""
        return (self.settings.value("ui/theme", "system", type=str) or "system").lower()
