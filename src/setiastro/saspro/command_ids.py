"""
Centralized command ID normalization table.
Used by both the command drop handler and the preview command store
to ensure consistent cid strings throughout the replay system.

Scope: this table is the vocabulary for the *replay / bundle* system —
tools that produce a repeatable, headless single-image edit. Canonical cids
match the reg() names in MainWindow._create_actions.

normalize_command_id() passes any unknown id through unchanged (lower-cased),
so tools that are NOT replayable steps are intentionally omitted here:
  - viewers / analysis:  histogram, psf_viewer, image_peeker, magnitude_tool,
    snr_tool, dither_analysis, isophote, whats_in_my_sky, whats_in_my_image,
    finder_chart, atlas, exo_detector, supernova_hunter
  - capture / stacking:  blink, live_stacking, stacking_suite, planetary_stacker
  - project / view I/O:  open, save_as, checkpoint_save, undo, redo,
    project_new/save/load, view_bundles, function_bundles, zoom_1_1, autostretch
  - masks / interactive: create_mask, invert_mask, remove_mask, show_mask,
    hide_mask, blemish_blaster, clone_stamp, signature positioning, star_spikes,
    astrospike, slap, mosaic_master, surface_mosaic, planet_projection, flythrough
  - not-yet-headless single-image edits (add when they gain a headless path):
    ppp, freqsep, selective_color, selective_lum, mono_to_rgb, nbtorgb,
    narrowband_normalization, narrowband_integration, contsub, nbextract,
    sfcc, sssc, add_stars
"""

COMMAND_ID_ALIASES: dict[str, str] = {
    # ---- geometry ----
    "flip_horizontal": "geom_flip_horizontal",
    "geom_flip_h": "geom_flip_horizontal",
    "geom_flip_horizontal": "geom_flip_horizontal",
    "flip_vertical": "geom_flip_vertical",
    "geom_flip_v": "geom_flip_vertical",
    "geom_flip_vertical": "geom_flip_vertical",
    "geom_rotate_clockwise": "geom_rotate_clockwise",
    "rotate_clockwise": "geom_rotate_clockwise",
    "geom_rot_cw": "geom_rotate_clockwise",
    "rotate_counterclockwise": "geom_rotate_counterclockwise",
    "geom_rot_ccw": "geom_rotate_counterclockwise",
    "geom_rotate_counterclockwise": "geom_rotate_counterclockwise",
    "rotate_180": "geom_rotate_180",
    "geom_rot_180": "geom_rotate_180",
    "geom_rotate_180": "geom_rotate_180",
    "rotate_any": "geom_rotate_any",
    "rotate_arbitrary": "geom_rotate_any",
    "geom_rotate_any": "geom_rotate_any",
    "invert": "geom_invert",
    "geom_invert": "geom_invert",
    "rescale": "geom_rescale",
    "geom_rescale": "geom_rescale",
    "geom_resize_canvas": "geom_resize_canvas",
    "resize_canvas": "geom_resize_canvas",

    # ---- stretch / tone ----
    "stat_stretch": "stat_stretch",
    "statistical_stretch": "stat_stretch",
    "statistical stretch": "stat_stretch",
    "star_stretch": "star_stretch",
    "star stretch": "star_stretch",
    "levels": "levels",
    "histogram_transform": "levels",
    "histogram transform": "levels",
    "histogram transformation": "levels",
    # NOTE: 'histogram' (bare) is a SEPARATE viewer action in reg()
    # (act_histogram), NOT Levels. Deliberately not aliased to 'levels'
    # (it used to be — that was a mis-route for the histogram viewer).
    "ghs": "ghs",
    "hyperbolic_stretch": "ghs",
    "generalized_hyperbolic_stretch": "ghs",
    "universal_hyperbolic_stretch": "ghs",
    "uhs": "ghs",
    "hyperbolic stretch": "ghs",
    "generalized hyperbolic stretch": "ghs",
    "curves": "curves",
    "pedestal": "pedestal",
    "linear_fit": "linear_fit",
    "linear fit": "linear_fit",

    # ---- background / color ----
    "abe": "abe",
    "automatic_background_extraction": "abe",
    "graxpert": "graxpert",
    "grax": "graxpert",
    "remove_gradient_graxpert": "graxpert",
    "background_neutral": "background_neutral",
    "background neutralization": "background_neutral",
    "background neutralisation": "background_neutral",
    "white_balance": "white_balance",
    "white balance": "white_balance",
    "remove_green": "remove_green",
    "remove green": "remove_green",
    "scnr": "remove_green",
    "scnr (remove green)": "remove_green",
    "remove green (scnr)": "remove_green",
    "satchroma": "satchroma",
    "sat_chroma": "satchroma",
    "sat chroma": "satchroma",
    "fx": "fx",

    # ---- stars ----
    "remove_stars": "remove_stars",
    "star_removal": "remove_stars",
    "starnet": "remove_stars",
    "darkstar": "remove_stars",

    # ---- AI tools ----
    "aberrationai": "aberrationai",
    "aberration": "aberrationai",
    "ai_aberration": "aberrationai",
    "aberration correction (ai)": "aberrationai",
    "cosmic": "cosmic_clarity",
    "cosmicclarity": "cosmic_clarity",
    "cosmic_clarity": "cosmic_clarity",
    "cosmic clarity": "cosmic_clarity",
    "cosmic clarity – denoise": "cosmic_clarity",
    "cosmic clarity – sharpen": "cosmic_clarity",
    "cosmic clarity - denoise": "cosmic_clarity",
    "cosmic clarity - sharpen": "cosmic_clarity",
    # NOTE: cosmicclaritysat (satellite trail removal) is a DISTINCT tool —
    # keep it self-mapped so nothing collapses it into cosmic_clarity.
    "cosmicclaritysat": "cosmicclaritysat",
    "syqontools": "syqontools",
    "rcastro": "rcastro",
    "rc_astro": "rcastro",
    "rc-astro": "rcastro",
    "rc astro": "rcastro",
    "blurxterminator": "rcastro",
    "starxterminator": "rcastro",
    "noisexterminator": "rcastro",

    # ---- processing ----
    "texture_clarity": "texture_clarity",
    "texture and clarity": "texture_clarity",
    "texture clarity": "texture_clarity",
    "crop": "crop",
    "geom_crop": "crop",
    "wavescale_hdr": "wavescale_hdr",
    "wavescalehdr": "wavescale_hdr",
    "wavescale": "wavescale_hdr",
    "wavescale_dark_enhance": "wavescale_dark_enhance",
    "wavescale_dark_enhancer": "wavescale_dark_enhance",
    "wavescale dark enhancer": "wavescale_dark_enhance",
    "wsde": "wavescale_dark_enhance",
    "dark_enhancer": "wavescale_dark_enhance",
    "clahe": "clahe",
    "morphology": "morphology",
    "pixel_math": "pixel_math",
    "pixel math": "pixel_math",
    "halo_b_gon": "halo_b_gon",
    "halo-b-gon": "halo_b_gon",
    "convo": "convo",
    "convolution": "convo",
    "deconvolution": "convo",
    "convo_deconvo": "convo",
    "cosmetic_correction": "cosmetic_correction",
    "cosmetic correction": "cosmetic_correction",
    "cosmetic": "cosmetic_correction",
    "unwarp": "unwarp",
    "remove_sip": "unwarp",
    "remove sip": "unwarp",
    "multiscale_decomp": "multiscale_decomp",
    "multiscale decomposition": "multiscale_decomp",
    "multiscale": "multiscale_decomp",
    "signature_insert": "signature_insert",
    "signature / insert": "signature_insert",
    "signature": "signature_insert",
    "debayer": "debayer",

    # ---- luminance / channels ----
    "rgb_to_mono": "rgb_to_mono",
    "rgb to mono": "rgb_to_mono",
    "rgb → mono": "rgb_to_mono",
    "rgb2mono": "rgb_to_mono",
    "convert_to_mono": "rgb_to_mono",
    "mono_to_rgb": "mono_to_rgb",
    "mono to rgb": "mono_to_rgb",
    "mono → rgb": "mono_to_rgb",
    "mono2rgb": "mono_to_rgb",
    "convert_to_rgb": "mono_to_rgb",
    "swap_rb": "swap_rb",
    "swap_r_b": "swap_rb",
    "swap r/b": "swap_rb",
    "swap r ↔ b": "swap_rb",
    "extract_luminance": "extract_luminance",
    "extract luminance": "extract_luminance",
    "extract_luma": "extract_luminance",
    "recombine_luminance": "recombine_luminance",
    "recombine luminance": "recombine_luminance",
    "recombine_luma": "recombine_luminance",
    "rgb_extract": "rgb_extract",
    "rgb extract": "rgb_extract",
    "extract_rgb": "rgb_extract",
    "rgb_combine": "rgb_combine",
    "rgb combine": "rgb_combine",
    "combine_rgb": "rgb_combine",

    # ---- combine / alignment / astrometry ----
    "image_combine": "image_combine",
    "image combine": "image_combine",
    "star_alignment": "star_align",
    "align_stars": "star_align",
    "star align": "star_align",
    "star_align": "star_align",
    "star_register": "star_register",
    "stellar_registration": "star_register",
    "star registration": "star_register",
    "rgb_align": "rgb_align",
    "rgb align": "rgb_align",
    "plate_solve": "plate_solve",
    "plate solve": "plate_solve",
    "platesolve": "plate_solve",
}


def normalize_command_id(cid: str) -> str:
    """Normalize a raw command id or step name to the canonical cid string."""
    return COMMAND_ID_ALIASES.get(str(cid or "").strip().lower(), str(cid or "").strip().lower())