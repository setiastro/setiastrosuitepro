# ============================================================
#  SatChroma preset helper
#  src/setiastro/saspro/satchroma_preset.py
#
#  apply_satchroma_via_preset — used by replay_last_action_on_base
#  and _handle_command_drop, mirroring apply_levels_via_preset /
#  apply_curves_via_preset patterns.
#
#  Part of Seti Astro Suite Pro
#  Copyright © 2025 Franklin Marek  |  www.setiastro.com
# ============================================================
from __future__ import annotations

from typing import Optional


def apply_satchroma_via_preset(main_window, doc, preset: dict) -> bool:
    """
    Apply a SatChroma adjustment headlessly to *doc* using *preset*.

    preset keys:
        mode      : int   0=HSV saturation, 1=Lab chroma
        strength  : float global curve scale  (default 1.0)
        points    : list  of [hue_norm, multiplier] pairs
        use_mask  : bool  respect active mask  (default True)

    Returns True on success, False on failure.
    """
    from setiastro.saspro.satchroma_tool import apply_satchroma_headless
    return apply_satchroma_headless(doc, preset, main_window=main_window)