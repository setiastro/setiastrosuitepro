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

def open_satchroma_with_preset(main_window, preset: dict | None = None):
    """
    Open the live SatChroma dialog seeded from a preset. Used by the double-click
    preset-open path (ShortcutManager.trigger_with_preset).

    Reuses the dialog's existing public load_preset() — the exact inverse of
    get_preset() (the grip's source of truth), so mode/strength/points/use_mask
    round-trip with zero drift. Resolves the active document from the active MDI
    subwindow first, matching the other tools' openers.
    """
    from setiastro.saspro.satchroma_tool import SatChromaTool

    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))

    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            w = sw.widget()
            doc = getattr(w, "document", None)
    except Exception:
        doc = None
    if doc is None and dm is not None:
        doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
               else getattr(dm, "active_document", None))
    if doc is None or getattr(doc, "image", None) is None:
        return

    # Note the constructor signature: (doc_manager, document, parent) — parent last.
    dlg = SatChromaTool(doc_manager=dm, document=doc, parent=main_window)
    try:
        from setiastro.saspro.resources import satchroma_path
        from PyQt6.QtGui import QIcon
        dlg.setWindowIcon(QIcon(satchroma_path))
    except Exception:
        pass
    # load_preset ends with _update_preview(); no on-show reset exists in this
    # dialog, so seeding after construction holds (no stash-and-defer needed).
    if preset:
        try:
            dlg.load_preset(dict(preset))
        except Exception:
            pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()