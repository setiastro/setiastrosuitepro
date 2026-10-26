#saspro/gui/menu_spec.py
"""
Declarative menu taxonomy for AstroSuiteProMainWindow.

SINGLE SOURCE OF TRUTH for:
  * which QAction attribute belongs to which top-level menu  (MENU_CONTENTS)
  * how the big menus are grouped into submenus              (SUBMENUS)
  * what the Icon Cheat Sheet enumerates                     (cheat_sheet_groups)

Add a new action to MENU_CONTENTS (and, optionally, to SUBMENUS) and it
automatically appears in the menubar, in the "All ..." flat list, and in the
Icon Cheat Sheet. No second list to keep in sync.

Notes on labels
---------------
Qt treats '&' in menu text as a mnemonic marker, so a literal ampersand in a
submenu label must be written '&&' (same convention already used for
"&Header Mods && Misc").
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Authoritative contents of every top-level menu, in menu order.
# These are attribute names on the main window, not QAction objects.
# ---------------------------------------------------------------------------

MENU_CONTENTS: dict[str, List[str]] = {
    "file": [
        "act_open", "act_create_table",
        "act_save",
        "act_save_fits", "act_save_xisf", "act_save_tiff", "act_save_png",
        "act_save_jpeg", "act_save_webp", "act_save_psb",
        "act_export_fits_bundle", "act_export_fits",
        "act_clear_views",
        "act_project_new", "act_project_save", "act_project_load",
        "act_exit",
    ],
    "edit": [
        "act_undo", "act_redo", "act_copy", "act_paste",
        "act_mono_to_rgb", "act_rgb_to_mono", "act_swap_rb",
    ],
    "display": [
        "act_autostretch", "act_hardstretch", "act_autostretch_continuous",
        "act_stretch_linked", "act_display_target", "act_display_sigma",
        "act_no_black_clip", "act_bake_display_stretch",
        "act_zoom_in", "act_zoom_out", "act_zoom_1_1", "act_zoom_fit",
        "act_auto_fit_resize",
    ],
    "functions": [
        "act_abe", "act_add_stars", "act_background_neutral", "act_blemish",
        "act_clahe", "act_clone_stamp", "act_convo", "act_cosmetic_correction",
        "act_crop", "act_curves", "act_extract_luma", "act_fx", "act_graxpert",
        "act_ghs", "act_halobgon", "act_histogram", "act_hist_transform",
        "act_linear_fit", "act_morphology", "act_nbextract", "act_pedestal",
        "act_pixelmath", "act_recombine_luma", "act_remove_green",
        "act_remove_stars", "act_rgb_combine", "act_rgb_extract",
        "act_satchroma", "act_sfcc", "act_sssc", "act_signature",
        "act_star_stretch", "act_stat_stretch", "act_texture_clarity",
        "act_wavescale_de", "act_wavescale_hdr", "act_white_balance",
    ],
    "smart_tools": [
        "actAberrationAI", "actCosmicUI", "actCosmicSat", "act_graxpert",
        "actSyQonTools", "act_remove_stars", "act_rcastro",
    ],
    "tools": [
        "act_blink", "act_contsub", "act_freqsep", "act_image_combine",
        "act_multiscale_decomp", "act_narrowband_integration",
        "act_narrowband_normalization", "act_nbtorgb", "act_ppp",
        "act_selective_color", "act_selective_lum", "act_slap_toolkit",
        "act_magnitude", "act_snr", "act_view_bundles", "act_function_bundles",
    ],
    "geometry": [
        "act_geom_invert", "act_geom_flip_h", "act_geom_flip_v",
        "act_geom_rot_cw", "act_geom_rot_ccw", "act_geom_rot_180",
        "act_geom_rot_any", "act_geom_rescale", "act_geom_resize_canvas",
        "act_debayer",
    ],
    "star_stuff": [
        "act_astrospike", "act_exo_detector", "act_dither_analysis",
        "act_image_peeker", "act_isophote", "act_live_stacking",
        "act_mosaic_master", "act_flythrough", "act_planet_projection",
        "act_planetary_stacker", "act_plate_solve", "act_psf_viewer",
        "act_rgb_align", "act_star_align", "act_star_register",
        "act_star_spikes", "act_stacking_suite", "act_supernova_hunter",
        "act_surface_mosaic", "act_unwarp",
    ],
    "masks": [
        "act_create_mask", "act_apply_mask", "act_remove_mask",
        "act_show_mask", "act_hide_mask", "act_invert_mask",
    ],
    "whats_in_my": [
        "act_whats_in_my_sky", "act_wimi", "act_finder_chart", "act_atlas",
    ],
    "header_misc": [
        "act_acv_exporter", "act_astrobin_exporter", "act_batch_convert",
        "act_batch_renamer", "act_copy_astrometry", "act_fits_batch_modifier",
        "act_fits_modifier",
    ],
    "history": [
        "act_history_explorer",
    ],
    "shortcuts": [
        "act_icon_table",
    ],
    "view": [
        "act_cascade", "act_tile", "act_tile_vert", "act_tile_horiz",
        "act_tile_grid", "act_minimize_all_views", "act_open_panel_host",
        "act_send_panels_to_host", "act_return_panels_to_main",
        "act_link_group",
    ],
    "workflows": [
        "act_workflows", "act_mini_workflows",
    ],
    "settings": [
        "act_gaia_database",
    ],
    "about": [
        "act_docs", "act_file_associations", "act_check_updates",
        "act_welcome", "act_bored",
    ],
}


# ---------------------------------------------------------------------------
# Submenu grouping for the three oversized menus.
# Any action in MENU_CONTENTS that is NOT listed here still reaches the user
# via the "Other" submenu and the "All ..." flat list -- nothing can be lost.
# ---------------------------------------------------------------------------

SUBMENUS: dict[str, List[Tuple[str, List[str]]]] = {
    "functions": [
        ("Gradient && Background", [
            "act_abe", "act_graxpert", "act_pedestal",
        ]),
        ("Color Calibration", [
            "act_background_neutral", "act_white_balance", "act_sfcc",
            "act_sssc", "act_linear_fit",
        ]),
        ("Color && Saturation", [
            "act_remove_green", "act_satchroma",
        ]),
        ("Stretching && Tone", [
            "act_stat_stretch", "act_star_stretch", "act_ghs", "act_curves",
            "act_hist_transform", "act_histogram",
        ]),
        ("Detail && Contrast", [
            "act_convo", "act_texture_clarity", "act_wavescale_hdr",
            "act_wavescale_de", "act_clahe", "act_morphology",
        ]),
        ("Stars", [
            "act_remove_stars", "act_add_stars", "act_halobgon",
        ]),
        ("Channels && Luminance", [
            "act_extract_luma", "act_recombine_luma", "act_rgb_extract",
            "act_rgb_combine", "act_nbextract",
        ]),
        ("Repair && Retouch", [
            "act_blemish", "act_clone_stamp", "act_cosmetic_correction",
            "act_crop",
        ]),
        ("Creative && Overlays", [
            "act_fx", "act_signature", "act_pixelmath",
        ]),
    ],
    "tools": [
        ("Narrowband", [
            "act_narrowband_integration", "act_narrowband_normalization",
            "act_nbtorgb", "act_ppp", "act_contsub",
        ]),
        ("Selective Correction", [
            "act_selective_color", "act_selective_lum",
        ]),
        ("Decomposition", [
            "act_freqsep", "act_multiscale_decomp",
        ]),
        ("Measurement && Inspection", [
            "act_blink", "act_magnitude", "act_snr",
        ]),
        ("Compositing", [
            "act_image_combine", "act_slap_toolkit",
        ]),
        ("Bundles", [
            "act_view_bundles", "act_function_bundles",
        ]),
    ],
    "star_stuff": [
        ("Stacking", [
            "act_stacking_suite", "act_live_stacking", "act_planetary_stacker",
        ]),
        ("Registration && Alignment", [
            "act_star_align", "act_star_register", "act_rgb_align",
            "act_unwarp",
        ]),
        ("Mosaics", [
            "act_mosaic_master", "act_surface_mosaic",
        ]),
        ("Astrometry && Analysis", [
            "act_plate_solve", "act_psf_viewer", "act_image_peeker",
            "act_dither_analysis", "act_isophote",
        ]),
        ("Discovery", [
            "act_exo_detector", "act_supernova_hunter",
        ]),
        ("3D && Projection", [
            "act_planet_projection", "act_flythrough",
        ]),
        ("Diffraction Spikes", [
            "act_star_spikes", "act_astrospike",
        ]),
    ],
}

# Label used for the flat "everything" submenu at the bottom of a grouped menu.
ALL_LABELS: dict[str, str] = {
    "functions": "All Functions",
    "tools": "All Tools",
    "star_stuff": "All Star Stuff",
}

# Human-readable names for cheat-sheet group headers.
MENU_TITLES: dict[str, str] = {
    "file": "File",
    "edit": "Edit",
    "display": "Display",
    "functions": "Functions",
    "smart_tools": "Smart Tools",
    "tools": "Tools",
    "geometry": "Geometry",
    "star_stuff": "Star Stuff",
    "masks": "Masks",
    "whats_in_my": "What's In My…",
    "header_misc": "Header & Misc",
    "history": "History",
    "shortcuts": "Shortcuts",
    "view": "View",
    "workflows": "Workflows",
    "settings": "Settings",
    "about": "About",
}

# Order the Icon Cheat Sheet walks the menus in.
CHEAT_SHEET_ORDER: List[str] = [
    "file", "edit", "display", "functions", "smart_tools", "tools",
    "geometry", "star_stuff", "masks", "whats_in_my", "header_misc",
    "workflows", "view", "history", "shortcuts", "settings", "about",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(window, attrs: Iterable[str]) -> List:
    """Map attribute names to live QActions, skipping any that don't exist."""
    out = []
    for name in attrs:
        act = getattr(window, name, None)
        if act is not None:
            out.append(act)
    return out


def _label_of(act) -> str:
    try:
        return (act.text() or "").replace("&", "").strip()
    except Exception:
        return ""


def _sorted_actions(actions: Sequence) -> List:
    return sorted(actions, key=lambda a: _label_of(a).lower())


def _tr(window, text: str) -> str:
    try:
        return window.tr(text)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Menubar construction
# ---------------------------------------------------------------------------

def add_actions(window, menu, key: str) -> None:
    """Add every action for `key` to `menu` as a flat list (menu order)."""
    for act in _resolve(window, MENU_CONTENTS.get(key, [])):
        menu.addAction(act)


def build_categorized_menu(
    window,
    menu,
    key: str,
    *,
    flat_list_inline: bool = False,
    include_all_list: bool = True,
) -> None:
    """
    Populate `menu` with category submenus, then the complete flat list.

    A QAction may live in more than one QMenu, so each tool appears both in
    its category submenu and in the full list -- muscle memory is preserved.

    flat_list_inline=True puts the full list directly in the menu below the
    categories instead of nesting it under an "All ..." submenu.
    """
    groups = SUBMENUS.get(key, [])
    all_attrs = MENU_CONTENTS.get(key, [])
    categorized: set[str] = set()

    for label, attrs in groups:
        acts = _resolve(window, attrs)
        if not acts:
            continue
        sub = menu.addMenu(_tr(window, label))
        for act in acts:
            sub.addAction(act)
        categorized.update(attrs)

    # Anything added to MENU_CONTENTS but not yet filed into a category.
    leftovers = [a for a in all_attrs if a not in categorized]
    leftover_acts = _resolve(window, leftovers)
    if leftover_acts:
        sub = menu.addMenu(_tr(window, "Other"))
        for act in _sorted_actions(leftover_acts):
            sub.addAction(act)

    if not include_all_list:
        return

    every = _sorted_actions(_resolve(window, all_attrs))
    if not every:
        return

    menu.addSeparator()
    if flat_list_inline:
        for act in every:
            menu.addAction(act)
    else:
        all_label = ALL_LABELS.get(key, "All")
        sub = menu.addMenu(_tr(window, all_label))
        for act in every:
            sub.addAction(act)


# ---------------------------------------------------------------------------
# Icon Cheat Sheet
# ---------------------------------------------------------------------------

def cheat_sheet_groups(window) -> List[Tuple[str, List]]:
    """
    Build the Icon Cheat Sheet's MENU_GROUPS from the same taxonomy that
    builds the menubar, so the two can never drift apart again.

    Menus that have submenus are emitted as one group per submenu, e.g.
    "Functions — Color Calibration", which also gives the exported PDF a far
    more useful structure.
    """
    groups: List[Tuple[str, List]] = []

    for key in CHEAT_SHEET_ORDER:
        title = MENU_TITLES.get(key, key.replace("_", " ").title())
        subs = SUBMENUS.get(key)

        if not subs:
            acts = _resolve(window, MENU_CONTENTS.get(key, []))
            if acts:
                groups.append((title, acts))
            continue

        categorized: set[str] = set()
        for label, attrs in subs:
            acts = _resolve(window, attrs)
            categorized.update(attrs)
            if acts:
                groups.append((f"{title} — {label.replace('&&', '&')}", acts))

        leftovers = [a for a in MENU_CONTENTS.get(key, []) if a not in categorized]
        leftover_acts = _resolve(window, leftovers)
        if leftover_acts:
            groups.append((f"{title} — Other", leftover_acts))

    return groups


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def audit(window) -> dict:
    """
    Report taxonomy drift. Safe to call from the diagnostics report.

    Returns:
        {
          "missing":       attrs listed here that don't exist on the window,
          "uncategorized": attrs in a submenu-backed menu with no category,
          "orphans":       act_* QActions on the window listed in no menu,
        }
    """
    from PyQt6.QtGui import QAction

    listed: set[str] = set()
    missing: List[str] = []
    for key, attrs in MENU_CONTENTS.items():
        for name in attrs:
            listed.add(name)
            if getattr(window, name, None) is None:
                missing.append(f"{key}:{name}")

    uncategorized: List[str] = []
    for key, subs in SUBMENUS.items():
        filed = {a for _label, attrs in subs for a in attrs}
        for name in MENU_CONTENTS.get(key, []):
            if name not in filed:
                uncategorized.append(f"{key}:{name}")

    orphans: List[str] = []
    for name in dir(window):
        if not (name.startswith("act_") or name.startswith("act")):
            continue
        if name in listed:
            continue
        try:
            obj = getattr(window, name, None)
        except Exception:
            continue
        if isinstance(obj, QAction):
            orphans.append(name)

    return {
        "missing": sorted(missing),
        "uncategorized": sorted(uncategorized),
        "orphans": sorted(orphans),
    }