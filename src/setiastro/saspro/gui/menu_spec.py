#saspro/gui/menu_spec.py
"""
Declarative menu taxonomy for AstroSuiteProMainWindow.

SINGLE SOURCE OF TRUTH for:
  * which QAction attribute belongs to which top-level menu  (MENU_CONTENTS)
  * how the big menus are grouped into submenus              (SUBMENUS)
  * which menus show those groups inline, between separators  (SECTIONED_MENUS)
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

import re
from typing import Iterable, List, Sequence, Tuple

from PyQt6.QtCore import QT_TRANSLATE_NOOP

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
        # History Explorer had its own top-level History menu before.
        "act_undo", "act_redo", "act_history_explorer", "act_copy", "act_paste",
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
        "act_abe", "act_add_stars", "act_background_neutral",
        "act_banding_blaster", "act_blemish",
        "act_clahe", "act_clone_stamp", "act_convo", "act_cosmetic_correction",
        "act_crop", "act_curves", "act_extract_luma", "act_fx", "act_graxpert",
        "act_ghs", "act_halobgon", "act_histogram", "act_hist_transform",
        "act_linear_fit", "act_morphology", "act_nbextract", "act_pedestal",
        "act_pixelmath", "act_recombine_luma", "act_remove_green",
        "act_remove_stars", "act_rgb_combine", "act_rgb_extract",
        "act_satchroma", "act_sfcc", "act_srgc", "act_sssc", "act_curve_manager",
        "act_signature", "act_star_spikes", "act_astrospike",
        "act_star_stretch", "act_stat_stretch", "act_texture_clarity",
        "act_wavescale_de", "act_wavescale_hdr", "act_white_balance",
    ],
    "smart_tools": [
        "actAberrationAI", "actCosmicUI", "actCosmicSat", "act_graxpert",
        "actSyQonTools", "act_remove_stars", "act_rcastro",
    ],
    "tools": [
        "act_contsub", "act_freqsep", "act_image_combine",
        "act_hdr_composition",
        "act_multiscale_decomp", "act_narrowband_integration",
        "act_narrowband_normalization", "act_nbtorgb", "act_ppp",
        "act_selective_color", "act_selective_lum", "act_slap_toolkit",
        "act_view_bundles", "act_function_bundles",
        "act_planet_projection", "act_flythrough",
        # Header Mods & Misc was a top-level menu until 2.1.
        "act_acv_exporter", "act_astrobin_exporter", "act_batch_convert",
        "act_batch_renamer", "act_copy_astrometry", "act_fits_batch_modifier",
        "act_fits_modifier",
    ],
    "geometry": [
        "act_geom_invert", "act_geom_flip_h", "act_geom_flip_v",
        "act_geom_rot_cw", "act_geom_rot_ccw", "act_geom_rot_180",
        "act_geom_rot_any", "act_geom_rescale", "act_geom_resize_canvas",
        "act_debayer",
    ],
    # "Star Stuff" until 1.24.8, split so stacking can be found by name.
    "stacking": [
        "act_stacking_suite", "act_quick_stack", "act_live_stacking", "act_planetary_stacker",
        "act_star_align", "act_star_register", "act_rgb_align",
        "act_mosaic_master", "act_surface_mosaic",
    ],
    "analysis": [
        "act_plate_solve", "act_unwarp", "act_lunar_solver", "act_blink", "act_psf_viewer",
        "act_image_peeker", "act_dither_analysis", "act_magnitude", "act_snr",
        "act_isophote", "act_exo_detector", "act_supernova_hunter",
        # What's In My... was a top-level menu until 2.1.
        "act_whats_in_my_sky", "act_wimi", "act_finder_chart", "act_atlas",
    ],
    "masks": [
        "act_create_mask", "act_apply_mask", "act_remove_mask",
        "act_show_mask", "act_hide_mask", "act_invert_mask",
    ],
    # A submenu of View since 2.1 (built in menu_mixin); kept as its own
    # Icon Cheat Sheet group.
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


# Submenus that were top-level menus until 2.1 keep their old menu titles,
# mnemonic included, so the existing translations still apply.
HEADER_MISC_LABEL = QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "&Header Mods && Misc")
WHATS_IN_MY_LABEL = QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "&What's In My...")


# ---------------------------------------------------------------------------
# Grouping for the larger menus. Functions and Tools show each group as a
# submenu; the menus in SECTIONED_MENUS are short enough to show the groups
# inline, between separators. The Icon Cheat Sheet uses the groups either way.
# Any action in MENU_CONTENTS that is NOT listed here still reaches the user
# via the "Other" submenu and the "All ..." flat list -- nothing can be lost.
# ---------------------------------------------------------------------------

SUBMENUS: dict[str, List[Tuple[str, List[str]]]] = {
    "functions": [
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Gradient && Background"), [
            "act_abe", "act_srgc", "act_graxpert", "act_pedestal",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Color Calibration"), [
            "act_background_neutral", "act_white_balance", "act_sfcc",
            "act_sssc", "act_curve_manager", "act_linear_fit",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Color && Saturation"), [
            "act_remove_green", "act_satchroma",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Stretching && Tone"), [
            "act_stat_stretch", "act_star_stretch", "act_ghs", "act_curves",
            "act_hist_transform", "act_histogram",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Detail && Contrast"), [
            "act_convo", "act_texture_clarity", "act_wavescale_hdr",
            "act_wavescale_de", "act_clahe", "act_morphology",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Stars"), [
            "act_remove_stars", "act_add_stars", "act_halobgon",
            "act_star_spikes", "act_astrospike",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Channels && Luminance"), [
            "act_extract_luma", "act_recombine_luma", "act_rgb_extract",
            "act_rgb_combine", "act_nbextract",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Repair && Retouch"), [
            "act_blemish", "act_banding_blaster", "act_clone_stamp",
            "act_cosmetic_correction", "act_crop",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Creative && Overlays"), [
            "act_fx", "act_signature", "act_pixelmath",
        ]),
    ],
    "tools": [
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Narrowband"), [
            "act_narrowband_integration", "act_narrowband_normalization",
            "act_nbtorgb", "act_ppp", "act_contsub",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Selective Correction"), [
            "act_selective_color", "act_selective_lum",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Decomposition"), [
            "act_freqsep", "act_multiscale_decomp",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Compositing"), [
            "act_image_combine", "act_hdr_composition", "act_slap_toolkit",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Visualization"), [
            "act_planet_projection", "act_flythrough",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Bundles"), [
            "act_view_bundles", "act_function_bundles",
        ]),
        (HEADER_MISC_LABEL, [
            "act_acv_exporter", "act_astrobin_exporter", "act_batch_convert",
            "act_batch_renamer", "act_copy_astrometry", "act_fits_batch_modifier",
            "act_fits_modifier",
        ]),
    ],
    "stacking": [
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Stacking"), [
            "act_stacking_suite", "act_quick_stack", "act_live_stacking", "act_planetary_stacker",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Registration && Alignment"), [
            "act_star_align", "act_star_register", "act_rgb_align",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Mosaics"), [
            "act_mosaic_master", "act_surface_mosaic",
        ]),
    ],
    "analysis": [
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Astrometry"), [
            "act_plate_solve", "act_unwarp", "act_lunar_solver",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Inspection"), [
            "act_blink", "act_psf_viewer", "act_image_peeker",
            "act_dither_analysis",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Measurement"), [
            "act_magnitude", "act_snr", "act_isophote",
        ]),
        (QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Discovery"), [
            "act_exo_detector", "act_supernova_hunter",
        ]),
        (WHATS_IN_MY_LABEL, [
            "act_whats_in_my_sky", "act_wimi", "act_finder_chart", "act_atlas",
        ]),
    ],
}

# Menus whose groups are laid out inline (separators, no submenus, no "All"
# list): with about ten tools each, every one stays a single click away.
SECTIONED_MENUS: frozenset[str] = frozenset({"stacking", "analysis"})

# Groups a sectioned menu shows as a submenu after all: former top-level menus
# folded in. A menu with one also gets an "All ..." list, so those tools are
# in a flat list too.
NESTED_GROUPS: dict[str, frozenset[str]] = {
    "analysis": frozenset({WHATS_IN_MY_LABEL}),
}

# Label used for the flat "everything" submenu at the top of a grouped menu.
ALL_LABELS: dict[str, str] = {
    "functions": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "All Functions"),
    "tools": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "All Tools"),
    "analysis": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "All Analysis Tools"),
}

# Human-readable names for cheat-sheet group headers.
MENU_TITLES: dict[str, str] = {
    "file": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "File"),
    "edit": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Edit"),
    "display": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Display"),
    "functions": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Functions"),
    "smart_tools": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Smart Tools"),
    "tools": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Tools"),
    "geometry": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Geometry"),
    "stacking": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Stacking"),
    "analysis": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Analysis"),
    "masks": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Masks"),
    "shortcuts": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Shortcuts"),
    "view": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "View"),
    "workflows": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Workflows"),
    "settings": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Settings"),
    # The menu is titled Help (it was About on Windows/Linux); the key stays.
    "about": QT_TRANSLATE_NOOP("AstroSuiteProMainWindow", "Help"),
}

# Order the Icon Cheat Sheet walks the menus in.
CHEAT_SHEET_ORDER: List[str] = [
    "file", "edit", "display", "functions", "smart_tools", "tools",
    "geometry", "stacking", "analysis", "masks",
    "workflows", "view", "shortcuts", "settings", "about",
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
        return (act.text() or "").replace("&&", "\0").replace("&", "").replace("\0", "&").strip()
    except Exception:
        return ""


def _sorted_actions(actions: Sequence) -> List:
    return sorted(actions, key=lambda a: _label_of(a).lower())


def _tr(window, text: str) -> str:
    try:
        return window.tr(text)
    except Exception:
        return text


def _plain(label: str) -> str:
    """A menu label as text: mnemonics out ("&What's", CJK "(&W)"), "&&" -> "&"."""
    label = re.sub(r"\(&[^&)]\)", "", label).replace("&&", "\0")
    return label.replace("&", "").replace("\0", "&").strip()


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
    Populate `menu` with the complete flat list, then the category submenus.

    A QAction may live in more than one QMenu, so each tool appears both in
    its category submenu and in the full list -- muscle memory is preserved.

    flat_list_inline=True puts the full list directly in the menu below the
    categories instead of nesting it under an "All ..." submenu.
    """
    groups = SUBMENUS.get(key, [])
    all_attrs = MENU_CONTENTS.get(key, [])
    categorized: set[str] = set()

    every = _sorted_actions(_resolve(window, all_attrs)) if include_all_list else []

    # --- ALL-FIRST ---
    # The "All ..." submenu is the first entry in the dropdown, above the
    # category submenus, so the flat list is one hop away from the menubar.
    if every and not flat_list_inline:
        all_label = ALL_LABELS.get(key, "All")
        sub = menu.addMenu(_tr(window, all_label))
        for act in every:
            sub.addAction(act)
        menu.addSeparator()

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

    # Inline flat list stays at the bottom of the menu.
    if every and flat_list_inline:
        menu.addSeparator()
        for act in every:
            menu.addAction(act)


def build_sectioned_menu(window, menu, key: str) -> None:
    """
    Populate `menu` with its SUBMENUS groups laid out inline, one block per
    group with a separator between blocks, for menus short enough that a
    submenu would only hide the tools. Anything in MENU_CONTENTS that no group
    lists goes in a last block, so nothing can be lost.

    A group in NESTED_GROUPS is a submenu instead, and then the menu starts
    with the "All ..." flat list as well, so its tools are in a flat list too.
    """
    groups = SUBMENUS.get(key, [])
    nested = NESTED_GROUPS.get(key, frozenset())
    filed = {a for _label, attrs in groups for a in attrs}

    if nested:
        every = _sorted_actions(_resolve(window, MENU_CONTENTS.get(key, [])))
        if every:
            sub = menu.addMenu(_tr(window, ALL_LABELS.get(key, "All")))
            for act in every:
                sub.addAction(act)
            menu.addSeparator()

    blocks = [(label if label in nested else None, _resolve(window, attrs))
              for label, attrs in groups]
    blocks.append((None, _resolve(window, [a for a in MENU_CONTENTS.get(key, [])
                                           if a not in filed])))

    first = True
    for label, acts in blocks:
        if not acts:
            continue
        if not first:
            menu.addSeparator()
        target = menu if label is None else menu.addMenu(_tr(window, label))
        for act in acts:
            target.addAction(act)
        first = False


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
        title = _tr(window, MENU_TITLES.get(key, key.replace("_", " ").title()))
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
                groups.append((f"{title} — {_plain(_tr(window, label))}", acts))

        leftovers = [a for a in MENU_CONTENTS.get(key, []) if a not in categorized]
        leftover_acts = _resolve(window, leftovers)
        if leftover_acts:
            groups.append((f"{title} — {_tr(window, 'Other')}", leftover_acts))

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