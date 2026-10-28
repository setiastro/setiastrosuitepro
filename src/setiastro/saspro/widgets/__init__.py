#saspro/widgets/__init__.py
"""
Shared UI widgets for Seti Astro Suite Pro.

This package contains reusable UI components to avoid code duplication.

The re-exports below are LAZY (PEP 562). Every name imports exactly as it
always did -- `from setiastro.saspro.widgets import CustomSpinBox`, or
`from setiastro.saspro.widgets import *` -- and the objects are identical to
the ones in their defining submodules. The difference is that the submodule
behind a name is only loaded when that name is first touched.

WHY THIS IS NOT JUST A SPEED CHANGE
-----------------------------------
The eager imports that used to be here were one link of a real circular
import:

    imageops/__init__          -> .stretch
    imageops/stretch           -> luminancerecombine
    luminancerecombine         -> widgets.image_utils
    widgets/__init__   (HERE)  -> widgets.preview_dialogs
    widgets/preview_dialogs    -> imageops.stretch   (half-built: ImportError)

Nothing can import `widgets.image_utils` without Python executing this file
first, so the old line 10 -- `from ...widgets.preview_dialogs import
ImagePreviewDialog` -- put that whole cycle in the path of every importer of
this package, including the plate solver.

It stayed invisible for a long time because the top-level
`setiastro/saspro/__init__.py` used to import `subwindow` eagerly, and
subwindow pulls `imageops.viewport_render`, `imageops.tiled_canvas` and
`widgets.image_utils`. That warmed both packages in a benign order before any
real work started. Two consequences of that masking are worth remembering:

  * `plate_solver.py`'s `from ...imageops.stretch import stretch_mono_image,
    stretch_color_image` sits in a `try/except Exception` that sets both names
    to None. On the CLI solve path that except was already firing, silently,
    so those helpers were already None at runtime.
  * The moment the top-level __init__ stopped importing subwindow, the blind
    solver became the first thing to enter the cycle and it blew up with the
    ImportError above.

Lazy re-exports here break the cycle from every direction, because importing
`widgets.image_utils` no longer drags in `preview_dialogs`.

KEEPING IT FIXED
----------------
Add new names to `_LAZY`, never as a module-scope import in this file. A
single eager import here re-arms the cycle for the whole package.

Note for anything *inside* this package: a bare module-global reference does
NOT go through `__getattr__` (LOAD_GLOBAL checks the module dict and then
builtins only). Import what you need directly from the defining submodule.
"""
from __future__ import annotations

# name -> submodule that defines it
_LAZY = {
    "CustomSpinBox":          ".spinboxes",
    "CustomDoubleSpinBox":    ".spinboxes",
    "ZoomableGraphicsView":   ".graphics_views",
    "ImagePreviewDialog":     ".preview_dialogs",
    "numpy_to_qimage":        ".image_utils",
    "numpy_to_qpixmap":       ".image_utils",
    "qimage_to_numpy":        ".image_utils",
    "create_preview_image":   ".image_utils",
    "normalize_image":        ".image_utils",
    # Common utilities
    "AboutDialog":            ".common_utilities",
    "ProjectSaveWorker":      ".common_utilities",
    "DECOR_GLYPHS":           ".common_utilities",
    "strip_ui_decorations":   ".common_utilities",
    "_strip_ui_decorations":  ".common_utilities",
    "install_crash_handlers": ".common_utilities",
    "get_version":            ".common_utilities",
    "get_build_timestamp":    ".common_utilities",
}

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


def __getattr__(name: str):
    """Load the defining submodule on first access to one of the names above.

    ImportError from a genuinely broken submodule is allowed to propagate --
    that matches what the old eager imports did, just at first use instead of
    at package import.
    """
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value      # cache; __getattr__ is not consulted again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))