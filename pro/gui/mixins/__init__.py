# pro/gui/mixins/__init__.py
"""
GUI Mixins for AstroSuiteProMainWindow.

This package contains modular mixins that separate different aspects of the main window
into logical, maintainable components.
"""
from __future__ import annotations

from .dock_mixin import DockMixin
from .menu_mixin import MenuMixin
from .toolbar_mixin import ToolbarMixin
from .file_mixin import FileMixin
from .theme_mixin import ThemeMixin
from .geometry_mixin import GeometryMixin
from .view_mixin import ViewMixin
from .header_mixin import HeaderMixin
from .mask_mixin import MaskMixin
from .update_mixin import UpdateMixin

__all__ = [
    "DockMixin",
    "MenuMixin",
    "ToolbarMixin",
    "FileMixin",
    "ThemeMixin",
    "GeometryMixin",
    "ViewMixin",
    "HeaderMixin",
    "MaskMixin",
    "UpdateMixin",
]

