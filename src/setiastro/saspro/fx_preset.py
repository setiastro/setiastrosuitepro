# saspro/fx_preset.py
from __future__ import annotations


def apply_fx_via_preset(main_window, doc, preset: dict):
    """
    Adapter matching the <tool>_preset.py convention used by the shortcuts/
    command-drop system and replay-on-base (see curves_preset.py,
    satchroma_preset.py for the sibling adapters).

    Delegates the actual work to fx_module.apply_fx_headless(); raises on
    failure so callers' existing try/except + QMessageBox pattern catches it,
    matching how the Curves branch's apply_curves_ops()/apply_curves_via_preset()
    failures propagate.
    """
    from setiastro.saspro.fx_module import apply_fx_headless
    ok = apply_fx_headless(doc, preset or {}, main_window=main_window)
    if not ok:
        raise RuntimeError("apply_fx_headless() returned False")
    return ok


def fx_effect_display_name(effect_key: str) -> str:
    """Look up the human-readable effect name for logging, e.g. 'Bloom / Star Glow'."""
    from setiastro.saspro.fx_module import EFFECTS
    return next((e["name"] for e in EFFECTS if e["key"] == effect_key), effect_key)