# src/setiastro/saspro/widgets/theme_colors.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
The colours of SASpro's own panels (the start panel, Welcome and Setup, the
setup reminder, the tip bar, the First Steps card), all from the theme's
palette, so they look like the rest of SASpro in Gray, Dark, Light and Custom
and change with it.

A panel is the theme's window colour with a thin border; tiles, buttons and
rows on it are a shade off that; the one thing to press first is the theme's
selection colour. Read the colours when building the style sheet or painting,
never once at import, and build the sheet again on a palette change:

    def changeEvent(self, ev):
        if ev.type() in PALETTE_EVENTS:
            restyle(self, self._style())
        super().changeEvent(ev)
"""
from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

CARD_RADIUS = 6           # a panel or card
CONTROL_RADIUS = 4        # a tile, button or row on one

PALETTE_EVENTS = (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange)


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """`a` moved `t` of the way to `b`."""
    return QColor(round(a.red() + (b.red() - a.red()) * t),
                  round(a.green() + (b.green() - a.green()) * t),
                  round(a.blue() + (b.blue() - a.blue()) * t))


def theme_colors() -> SimpleNamespace:
    pal = QApplication.palette()
    role = QPalette.ColorRole
    window = pal.color(role.Window)
    text = pal.color(role.WindowText)
    accent = pal.color(role.Highlight)
    on_accent = pal.color(role.HighlightedText)
    return SimpleNamespace(
        window=window,                       # the panel itself
        text=text,
        soft=mix(text, window, 0.3),         # what explains the text above it
        muted=pal.color(QPalette.ColorGroup.Disabled, role.WindowText),
        border=mix(window, text, 0.25),      # the panel's edge
        line=mix(window, text, 0.16),        # a tile's edge, a rule
        raised=mix(window, text, 0.06),      # a tile or button on the panel
        hover=mix(window, text, 0.12),
        accent=accent,                       # the one to press first; focus
        accent_hover=mix(accent, on_accent, 0.12),
        on_accent=on_accent,
        on_accent_soft=mix(on_accent, accent, 0.25),
        selected=mix(window, accent, 0.28),  # the page or choice that is current
        link=pal.color(role.Link),
        dark=window.lightness() < 128,
    )


def restyle(widget, sheet: str) -> bool:
    """Set `sheet` if it differs: setting one can itself send a PaletteChange,
    so an unconditional set from changeEvent would never stop."""
    if widget.styleSheet() == sheet:
        return False
    widget.setStyleSheet(sheet)
    return True
