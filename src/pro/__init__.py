"""
Compatibility shim for older SASpro scripts that imported from `pro.*`.

New canonical path is: `setiastro.saspro.*`
"""

from __future__ import annotations
import sys
import importlib

# Map the old `pro.<mod>` names to the new `setiastro.saspro.<mod>` modules.
# Add to this list as needed (these are the ones you use in ops/commands.py).
_ALIASES = [
    "function_bundle",
    "ghs_preset",
    "curves_preset",
    "abe_preset",
    "graxpert_preset",
    "backgroundneutral",
    "remove_green",
    "luminancerecombine",
    "wavescale_hdr_preset",
    "wavescalede_preset",
    "aberration_ai_preset",
    "convo_preset",
    "cosmicclarity_preset",
    "debayer",
    "linear_fit",
    "morphology",
    "remove_stars_preset",
]

def _alias(name: str) -> None:
    new_mod = importlib.import_module(f"setiastro.saspro.{name}")
    sys.modules[f"pro.{name}"] = new_mod

for _m in _ALIASES:
    try:
        _alias(_m)
    except Exception:
        # If a module doesn’t exist in a particular build, don’t crash import pro
        pass
