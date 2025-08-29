# imageops/__init__.py

from .stretch import (
    stretch_mono_image,
    stretch_color_image,
    apply_curves_adjustment,
)

# --- Backward-compatible aliases (old SASv2-style names) ---
def stretch_color_image_linked(img, target_median, normalize=False,
                               apply_curves=False, curves_boost=0.0):
    return stretch_color_image(
        img, target_median,
        linked=True,
        normalize=normalize,
        apply_curves=apply_curves,
        curves_boost=curves_boost,
    )

def stretch_color_image_unlinked(img, target_median, normalize=False,
                                 apply_curves=False, curves_boost=0.0):
    return stretch_color_image(
        img, target_median,
        linked=False,
        normalize=normalize,
        apply_curves=apply_curves,
        curves_boost=curves_boost,
    )

__all__ = [
    "stretch_mono_image",
    "stretch_color_image",
    "apply_curves_adjustment",
    "stretch_color_image_linked",
    "stretch_color_image_unlinked",
    "apply_average_neutral_scnr",
]
