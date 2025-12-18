# pro/widgets/__init__.py
"""
Shared UI widgets for Seti Astro Suite Pro.

This package contains reusable UI components to avoid code duplication.
"""

from setiastro.saspro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.widgets.preview_dialogs import ImagePreviewDialog
from setiastro.saspro.widgets.image_utils import (
    numpy_to_qimage,
    numpy_to_qpixmap,
    qimage_to_numpy,
    create_preview_image,
    normalize_image
)
from setiastro.saspro.widgets.common_utilities import (
    AboutDialog,
    ProjectSaveWorker,
    DECOR_GLYPHS,
    strip_ui_decorations,
    _strip_ui_decorations,
    install_crash_handlers,
    get_version,
    get_build_timestamp,
)

__all__ = [
    'CustomSpinBox',
    'CustomDoubleSpinBox',
    'ZoomableGraphicsView',
    'ImagePreviewDialog',
    'numpy_to_qimage',
    'numpy_to_qpixmap',
    'qimage_to_numpy',
    'create_preview_image',
    'normalize_image',
    # Common utilities
    'AboutDialog',
    'ProjectSaveWorker',
    'DECOR_GLYPHS',
    'strip_ui_decorations',
    '_strip_ui_decorations',
    'install_crash_handlers',
    'get_version',
    'get_build_timestamp',
]
