from __future__ import annotations

from pathlib import Path
import sys

def syqon_parallax_model_path(variant: str = "deblur") -> Path:
    """
    Returns the expected install path for a SyQon Parallax model file.

    Variants:
        deblur      — non-stellar deblur model
        star_reduce — star reduction model
        star_abcorr — star aberration correction model
    """
    variant = str(variant or "deblur").strip().lower()

    filenames = {
        "deblur":      "parallax_deblur.pt",
        "star_reduce": "parallax_star_reduce.pt",
        "star_abcorr": "parallax_star_abcorr.pt",
    }
    fname = filenames.get(variant, f"parallax_{variant}.pt")

    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent

    return base / "syqon_parallax_model" / fname