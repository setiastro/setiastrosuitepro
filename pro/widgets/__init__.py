# pro/widgets/__init__.py
"""
Shared UI widgets for Seti Astro Suite Pro.

This package contains reusable UI components to avoid code duplication.
"""

from pro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox
from pro.widgets.graphics_views import ZoomableGraphicsView
from pro.widgets.preview_dialogs import ImagePreviewDialog
from pro.widgets.image_utils import (
    numpy_to_qimage,
    numpy_to_qpixmap,
    qimage_to_numpy,
    create_preview_image,
    normalize_image
)
from pro.widgets.common_utilities import (
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
