# imageops/__init__.py
"""Image operations for Seti Astro Suite Pro.

The re-exports from `.stretch` are LAZY (PEP 562): `from
setiastro.saspro.imageops import stretch_mono_image` works exactly as before
and returns the same object, but `.stretch` is only imported when a name is
first touched.

This is the other half of the circular import documented in
`widgets/__init__.py`:

    imageops/__init__  (HERE)  -> .stretch
    imageops/stretch           -> luminancerecombine
    luminancerecombine         -> widgets.image_utils
    widgets/__init__           -> widgets.preview_dialogs
    widgets/preview_dialogs    -> imageops.stretch   (half-built: ImportError)

Because `from setiastro.saspro.imageops.stretch import X` makes Python execute
this file before it executes stretch.py, the old eager `from .stretch import
(...)` meant every importer of `imageops.stretch` entered the cycle through
this module. Making both package __init__ files lazy breaks it from either
direction.

Add new names to `_LAZY`, never as a module-scope import here.
"""
from __future__ import annotations

# name -> submodule that defines it
_LAZY = {
    "stretch_mono_image":      ".stretch",
    "stretch_color_image":     ".stretch",
    "apply_curves_adjustment": ".stretch",
}

# `apply_average_neutral_scnr` used to be listed here while never being
# imported, so `from setiastro.saspro.imageops import *` raised AttributeError
# on it -- broken before any of this work, just never exercised. stretch.py
# does not define it (nothing SCNR-shaped is in its public API), so it is
# dropped rather than guessed at. If the function does exist elsewhere in the
# tree, add it back as one line in _LAZY pointing at the module that defines
# it, plus its name here.
__all__ = [
    "stretch_mono_image",
    "stretch_color_image",
    "apply_curves_adjustment",
    "stretch_color_image_linked",
    "stretch_color_image_unlinked",
]


def __getattr__(name: str):
    """Load `.stretch` on first access to one of the names above."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value      # cache; __getattr__ is not consulted again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


# --- Backward-compatible aliases (old SASv2-style names) ---
#
# These import stretch_color_image LOCALLY, and must keep doing so. A bare
# `stretch_color_image(...)` here would compile to LOAD_GLOBAL, which checks
# this module's dict and then builtins and NEVER consults __getattr__ -- so it
# would raise NameError until something else happened to touch the name first.
# The local import costs one sys.modules lookup after the first call.

def stretch_color_image_linked(img, target_median, normalize=False,
                               apply_curves=False, curves_boost=0.0):
    from .stretch import stretch_color_image
    return stretch_color_image(
        img, target_median,
        linked=True,
        normalize=normalize,
        apply_curves=apply_curves,
        curves_boost=curves_boost,
    )


def stretch_color_image_unlinked(img, target_median, normalize=False,
                                 apply_curves=False, curves_boost=0.0):
    from .stretch import stretch_color_image
    return stretch_color_image(
        img, target_median,
        linked=False,
        normalize=normalize,
        apply_curves=apply_curves,
        curves_boost=curves_boost,
    )