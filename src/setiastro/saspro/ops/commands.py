# ops/commands.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Preset / Command metadata models
# -----------------------------------------------------------------------------

@dataclass
class PresetSpec:
    """
    Describes one preset key for a command.
    """
    key: str
    type: str                    # "float" | "int" | "bool" | "str" | "enum" | "dict"
    default: Any = None
    min: float | None = None
    max: float | None = None
    enum: list[str] | None = None

    # prefer desc, but accept help= for backward-compat
    desc: str = ""               # human description
    help: str = ""               # DEPRECATED alias for desc

    optional: bool = True        # if False, UI/docs treat as required

    def __post_init__(self):
        if (not self.desc) and self.help:
            self.desc = self.help


@dataclass
class CommandSpec:
    """
    One command in SASpro.
    """
    id: str

    # prefer name/notes, but accept title/summary
    name: str = ""
    notes: str = ""

    title: str = ""              # DEPRECATED alias for name
    summary: str = ""            # DEPRECATED alias for notes

    group: str = "General"

    # how scripts should call it (preferred)
    call_style: str = "ctx.run_command"   # or "direct_import"

    # Optional: lazy callable import for headless apply
    # Use either import_path+callable_name OR ui_method/headless_method names.
    import_path: str | None = None        # e.g. "pro.whitebalance"
    callable_name: str | None = None      # e.g. "apply_white_balance_to_doc"

    # If command is handled by AstroSuiteProMainWindow methods:
    ui_method: str | None = None          # e.g. "_open_wavescale_hdr"
    headless_method: str | None = None    # e.g. "_apply_star_stretch_preset_to_doc"

    replay_apply_name: str | None = None

    presets: list[PresetSpec] = field(default_factory=list)

    examples: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    supports_mono: bool = True
    supports_rgb: bool = True
    supports_linear: bool = True
    supports_nonlinear: bool = True

    def __post_init__(self):
        # allow old style title/summary usage
        if not self.name and self.title:
            self.name = self.title
        if not self.notes and self.summary:
            self.notes = self.summary


# -----------------------------------------------------------------------------
# Registry container + helpers
# -----------------------------------------------------------------------------

COMMAND_REGISTRY: Dict[str, CommandSpec] = {}


def register(spec: CommandSpec) -> CommandSpec:
    COMMAND_REGISTRY[spec.id] = spec
    return spec


# -----------------------------------------------------------------------------
# CID normalization + aliases  (lifted from _cid_norm)
# -----------------------------------------------------------------------------

ALIASES: Dict[str, str] = {
    # geometry short ↔ long ids
    "flip_horizontal": "geom_flip_horizontal",
    "geom_flip_h": "geom_flip_horizontal",
    "geom_flip_horizontal": "geom_flip_horizontal",

    "flip_vertical": "geom_flip_vertical",
    "geom_flip_v": "geom_flip_vertical",
    "geom_rotate_clockwise": "geom_rotate_clockwise",
    "rotate_clockwise": "geom_rotate_clockwise",
    "geom_rot_cw": "geom_rotate_clockwise",

    "rotate_counterclockwise": "geom_rotate_counterclockwise",
    "geom_rot_ccw": "geom_rotate_counterclockwise",
    "geom_rotate_counterclockwise": "geom_rotate_counterclockwise",

    "rotate_180": "geom_rotate_180",
    "geom_rotate_180": "geom_rotate_180",

    "invert": "geom_invert",
    "geom_invert": "geom_invert",

    "rescale": "geom_rescale",
    "geom_rescale": "geom_rescale",

    # stretches / gradients
    "ghs": "ghs",
    "hyperbolic_stretch": "ghs",
    "universal_hyperbolic_stretch": "ghs",

    "abe": "abe",
    "automatic_background_extraction": "abe",

    "graxpert": "graxpert",
    "grax": "graxpert",
    "remove_gradient_graxpert": "graxpert",

    # star removal
    "remove_stars": "remove_stars",
    "star_removal": "remove_stars",
    "starnet": "remove_stars",
    "darkstar": "remove_stars",

    # AI tools  ✅ FIXED
    "aberrationai": "aberration_ai",
    "aberration_ai": "aberration_ai",   # explicit canonical alias
    "aberration": "aberration_ai",
    "ai_aberration": "aberration_ai",

    "cosmic": "cosmic_clarity",
    "cosmicclarity": "cosmic_clarity",
    "cosmic_clarity": "cosmic_clarity",

    # misc
    "crop": "crop",
    "geom_crop": "crop",

    "wavescale_hdr": "wavescale_hdr",
    "wavescalehdr": "wavescale_hdr",
    "wavescale": "wavescale_hdr",

    "wavescale_dark_enhance": "wavescale_dark_enhance",
    "wavescale_dark_enhancer": "wavescale_dark_enhance",
    "wsde": "wavescale_dark_enhance",
    "dark_enhancer": "wavescale_dark_enhance",

    "star_alignment": "star_align",
    "align_stars": "star_align",
    "align": "star_align",

    "convo": "convo",
    "convolution": "convo",
    "deconvolution": "convo",
    "convo_deconvo": "convo",
}


def normalize_cid(cid: str | None) -> str:
    c = (cid or "").strip().lower()
    return ALIASES.get(c, c)


def get_spec(cid: str | None) -> CommandSpec | None:
    return COMMAND_REGISTRY.get(normalize_cid(cid))


def list_commands() -> Dict[str, str]:
    return {cid: spec.name for cid, spec in COMMAND_REGISTRY.items()}


# -----------------------------------------------------------------------------
# Registry population (starter set)
# -----------------------------------------------------------------------------

# Bundles
register(CommandSpec(
    id="function_bundle",
    title="Function Bundle",
    group="Bundles",
    summary=(
        "Runs a sequence of steps from a saved Function Bundle or an inline "
        "steps list. "
        "Config: name/bundle_name='Bundle Name' OR steps=[...], "
        "optional targets, inherit_target."
    ),
    call_style="ctx.run_command",
    import_path="pro.function_bundle",          # <── important
    callable_name="run_function_bundle_command",# <── important
    notes=(
        "Use this command from scripts to run a saved Function Bundle or an "
        "inline list of steps.\n\n"
        "- For saved bundles, specify `bundle_name` (or `name`).\n"
        "- `inherit_target=True` forwards the current target (active view / ROI) "
        "into each step.\n"
        "- This is the same mechanism used by the bundled 'Run Function Bundle…' script."
    ),
    presets=[
        PresetSpec(
            key="bundle_name",
            type="str",
            desc="Name of the saved Function Bundle to run (same as shown in the Function Bundles dialog).",
            optional=True,
        ),
        PresetSpec(
            key="name",
            type="str",
            desc="Alias of `bundle_name` for backward compatibility.",
            optional=True,
        ),
        PresetSpec(
            key="steps",
            type="list",
            desc="Inline steps list (advanced). If given, this is used instead of a saved bundle.",
            optional=True,
        ),
        PresetSpec(
            key="inherit_target",
            type="bool",
            default=True,
            desc="If True, each step runs on the same target (active view / ROI) as the bundle call.",
            optional=True,
        ),
        PresetSpec(
            key="target",
            type="dict",
            desc="Optional explicit target override (advanced; normally you just let inherit_target=True).",
            optional=True,
        ),
    ],
    examples=[
        # Mirrors your run_function_bundle.py config, but simplified
        "def run(ctx):\n"
        "    cfg = {\n"
        "        'bundle_name': 'PreProcess',  # name from Function Bundles dialog\n"
        "        'inherit_target': True,\n"
        "    }\n"
        "    ctx.run_command('function_bundle', cfg)\n",

        # Inline steps example (for power users)
        "def run(ctx):\n"
        "    steps = [\n"
        "        {'id': 'stat_stretch', 'preset': {'target_median': 0.25}},\n"
        "        {'id': 'remove_green', 'preset': {'amount': 0.6}},\n"
        "    ]\n"
        "    ctx.run_command('function_bundle', {\n"
        "        'steps': steps,\n"
        "        'inherit_target': True,\n"
        "    })\n",
    ],
))


register(CommandSpec(
    id="bundle",
    title="Bundle Exec",
    group="Bundles",
    summary="Internal bundle runner. steps=[...], targets='all_open'|[doc_ptrs], stop_on_error.",
    call_style="ctx.run_command",
    import_path="pro.function_bundle",
    callable_name="run_function_bundle_command",
))


# ---------------- Stretches / tone ----------------

register(CommandSpec(
    id="stat_stretch",
    title="Statistical Stretch",
    group="Stretch",
    summary=(
        "Stretch linear data to a target median using Statistical Stretch. "
        "Supports linked/unlinked color stretch, optional normalization, "
        "and optional curves boost."
    ),
    headless_method="_apply_stat_stretch_preset_to_doc",
    ui_method="_open_statistical_stretch_with_preset",
    presets=[
        PresetSpec(
            key="target_median",
            type="float",
            default=0.25,
            min=0.0, max=1.0,
            help="Target median after stretch. Typical values 0.20–0.30."
        ),
        PresetSpec(
            key="linked",
            type="bool",
            default=True,
            help="If True, stretch RGB channels together (linked). If False, unlinked."
        ),
        PresetSpec(
            key="normalize",
            type="bool",
            default=True,
            help="If True, normalize channels/whitepoint after stretch."
        ),
        PresetSpec(
            key="apply_curves",
            type="bool",
            default=False,
            help="If True, apply the optional curves adjustment pass."
        ),
        PresetSpec(
            key="curves_boost",
            type="float",
            default=0.0,
            min=0.0, max=1.0,
            help="Curves boost strength in [0,1]. Only used if apply_curves=True."
        ),
    ],
    examples=[
        "ctx.run_command('stat_stretch', {'target_median': 0.25, 'linked': True})",
        "ctx.run_command('stat_stretch', {'target_median': 0.22, 'linked': False, "
        "'normalize': False, 'apply_curves': True, 'curves_boost': 0.15})",
    ],
    aliases=["statistical_stretch", "statstretch"],
))

register(CommandSpec(
    id="star_stretch",
    name="Star Stretch",
    group="Stretch",
    ui_method="_open_star_stretch_with_preset",
    headless_method="_apply_star_stretch_preset_to_doc",
    summary=(
        "Stretches stars on linear data using your Star Stretch kernel. "
        "Supports optional color boost and SCNR green neutralization."
    ),
    presets=[
        PresetSpec(
            key="stretch_factor",
            type="float",
            default=5.0,
            min=0.0,
            max=8.0,
            desc="Star stretch strength. Typical range: 3–6. (Aliases: stretch_amount, amount)"
        ),
        PresetSpec(
            key="color_boost",
            type="float",
            default=1.0,
            min=0.0,
            max=2.0,
            desc="Color saturation boost multiplier. 1.0 = no change. (Alias: saturation)"
        ),
        PresetSpec(
            key="scnr_green",
            type="bool",
            default=False,
            desc="If True, apply SCNR-like green neutralization. (Alias: scnr)"
        ),
    ],
    aliases=[
        "starstretch",
        "stretch_stars",
    ],
    examples=[
        "ctx.run_command('star_stretch', {'stretch_factor': 5.0})",
        "ctx.run_command('star_stretch', {'stretch_factor': 4.2, 'color_boost': 1.3})",
        "ctx.run_command('star_stretch', {'amount': 5.5, 'saturation': 1.15, 'scnr': True})",
    ],
    supports_mono=True,     # your headless path supports mono via temp RGB + collapse
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=False,  # star stretch is intended for star images / linear use
))


register(CommandSpec(
    id="ghs",
    name="Generalized Hyperbolic Stretch",
    group="Stretch",
    import_path="pro.ghs_preset",
    callable_name="apply_ghs_via_preset",
    ui_method="open_ghs_with_preset",
    summary=(
        "Universal / Generalized Hyperbolic Stretch. Builds a monotone control curve from "
        "alpha/beta/gamma with a pivot symmetry point and optional LP/HP protection. "
        "Applies to K (brightness) or individual RGB channels, with active-mask blending."
    ),
    presets=[
        PresetSpec(
            key="alpha",
            type="float",
            default=1.0,
            min=0.02,
            max=10.0,
            desc=(
                "Hyperbolic alpha (controls left/right curve rolloff). "
                "Dialog slider stores alpha*50. Range ≈0.02–10."
            )
        ),
        PresetSpec(
            key="beta",
            type="float",
            default=1.0,
            min=0.02,
            max=10.0,
            desc=(
                "Hyperbolic beta (asymmetry between shadows/highlights). "
                "Dialog slider stores beta*50. Range ≈0.02–10."
            )
        ),
        PresetSpec(
            key="gamma",
            type="float",
            default=1.0,
            min=0.01,
            max=5.0,
            desc=(
                "Gamma lift after hyperbolic mapping. "
                "Dialog slider stores gamma*100. Range ≈0.01–5."
            )
        ),
        PresetSpec(
            key="pivot",
            type="float",
            default=0.5,
            min=0.0,
            max=1.0,
            desc=(
                "Symmetry / pivot point in normalized domain. "
                "0.5 is neutral midtone pivot."
            )
        ),
        PresetSpec(
            key="lp",
            type="float",
            default=0.0,
            min=0.0,
            max=1.0,
            desc=(
                "Low-protect (LP). Blends left side toward identity y=x. "
                "Dialog slider stores lp*360."
            )
        ),
        PresetSpec(
            key="hp",
            type="float",
            default=0.0,
            min=0.0,
            max=1.0,
            desc=(
                "High-protect (HP). Blends right side toward identity y=x. "
                "Dialog slider stores hp*360."
            )
        ),
        PresetSpec(
            key="channel",
            type="enum",
            default="K (Brightness)",
            enum=["K (Brightness)", "R", "G", "B"],
            desc=(
                "Target channel. 'K (Brightness)' applies to luminance/brightness. "
                "R/G/B apply to individual color channels. "
                "Aliases accepted: k, brightness, rgb, r, g, b."
            )
        ),
    ],
    aliases=[
        "hyperbolic_stretch",
        "universal_hyperbolic_stretch",
        "uhs",
    ],
    examples=[
        # gentle linked luminance stretch
        "ctx.run_command('ghs', {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0, 'pivot': 0.5})",
        # stronger contrast with protection
        "ctx.run_command('ghs', {'alpha': 2.2, 'beta': 1.4, 'gamma': 1.1, 'pivot': 0.45, 'lp': 0.10, 'hp': 0.20})",
        # per-channel tweak
        "ctx.run_command('ghs', {'alpha': 1.6, 'beta': 1.0, 'gamma': 0.95, 'pivot': 0.5, 'channel': 'R'})",
    ],
    supports_mono=True,       # K works on mono; RGB channels are ignored/treated safely by _apply_mode_any
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


register(CommandSpec(
    id="curves",
    title="Curves",
    group="Stretch",
    import_path="pro.curves_preset",
    callable_name="apply_curves_via_preset",
    ui_method="open_curves_with_preset",
    summary=(
        "Preset schema: {mode, shape, amount, points_norm?}. "
        "mode applies to K/R/G/B or Lab/LCh-style channels. "
        "shape selects a built-in curve unless shape='custom'. "
        "amount is 0..1 intensity for non-custom shapes. "
        "custom uses points_norm (normalized 0..1). "
        "Advanced: preset may also include points_scene/handles/control_points."
    ),
    presets=[
        PresetSpec(
            key="mode",
            type="enum",
            default="K (Brightness)",
            enum=[
                "K (Brightness)",
                "R", "G", "B",
                "L*", "a*", "b*",
                "Chroma",
                "Saturation",
            ],
            help=(
                "Channel/mode to apply curve to. "
                "Aliases accepted: k, brightness, rgb -> K (Brightness); "
                "lum/l/lab_l -> L*; lab_a -> a*; lab_b -> b*; "
                "chroma -> Chroma; sat/s -> Saturation."
            ),
        ),
        PresetSpec(
            key="shape",
            type="enum",
            default="linear",
            enum=[
                "linear",
                "s_mild",
                "s_med",
                "s_strong",
                "lift_shadows",
                "crush_shadows",
                "fade_blacks",
                "rolloff_highlights",
                "flatten",
                "custom",
            ],
            help=(
                "Built-in curve shape. "
                "If 'custom', provide points_norm (or points_scene/handles)."
            ),
        ),
        PresetSpec(
            key="amount",
            type="float",
            default=0.5,
            min=0.0,
            max=1.0,
            help=(
                "Intensity for built-in shapes (0..1). "
                "Ignored when shape='custom'."
            ),
        ),
        PresetSpec(
            key="points_norm",
            type="dict",
            default=None,
            optional=True,
            help=(
                "For shape='custom': normalized control points "
                "as [[x,y], ...] with x,y in [0..1]. "
                "Example: [[0,0],[0.25,0.2],[0.5,0.5],[0.75,0.85],[1,1]]."
            ),
        ),
        PresetSpec(
            key="points_scene",
            type="dict",
            default=None,
            optional=True,
            help=(
                "Advanced custom curve in scene coords [[x,y],...], "
                "x,y in [0..360]. Synonyms accepted by engine: "
                "scene_points, handles, control_points."
            ),
        ),
    ],
    supports_mono=True,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


# ---------------- Gradient / background ----------------

register(CommandSpec(
    id="abe",
    title="Automatic Background Extraction",
    group="Background",
    import_path="pro.abe_preset",
    callable_name="apply_abe_via_preset",
    ui_method="open_abe_with_preset",   # ✅ matches your pro/abe_preset.py
    summary=(
        "Automatic Background Extraction (headless). "
        "Runs abe_run() with polynomial degree + RBF option. "
        "Headless mode does NOT support exclusion polygons (exclusion_mask=None). "
        "If an active mask is present, blends result as m*out + (1-m)*src. "
        "Optional: create a separate background document."
    ),
    presets=[
        PresetSpec(
            key="degree",
            type="int",
            default=2,
            min=0, max=6,
            help=(
                "Polynomial degree for background model. "
                "0 = RBF-only (allowed in headless). "
                "Dialog range is 0–6."
            ),
        ),
        PresetSpec(
            key="samples",
            type="int",
            default=120,
            min=20, max=100000,
            help="Number of sample points used for background fitting.",
        ),
        PresetSpec(
            key="downsample",
            type="int",
            default=6,
            min=1, max=64,
            help="Downsample factor for analysis grid (higher = faster, coarser).",
        ),
        PresetSpec(
            key="patch",
            type="int",
            default=15,
            min=5, max=151,
            help="Patch size (px) for local background sampling.",
        ),
        PresetSpec(
            key="rbf",
            type="bool",
            default=True,
            help="If True, include RBF smoothing component in the model.",
        ),
        PresetSpec(
            key="rbf_smooth",
            type="float",
            default=1.0,
            min=0.0, max=1000.0,
            help=(
                "RBF smoothness. Dialog stores rbf_smooth*100, "
                "so 1.0 here corresponds to slider=100."
            ),
        ),
        PresetSpec(
            key="make_background_doc",
            type="bool",
            default=False,
            help="If True, also open a background-only document.",
        ),
    ],
    aliases=[
        "automatic_background_extraction",
        "background_extraction",
    ],
    examples=[
        "ctx.run_command('abe', {'degree': 2, 'samples': 120, 'downsample': 6, 'patch': 15})",
        "ctx.run_command('abe', {'degree': 0, 'rbf': True, 'rbf_smooth': 0.8})",
        "ctx.run_command('abe', {'degree': 3, 'samples': 250, 'make_background_doc': True})",
    ],
    supports_mono=True,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


register(CommandSpec(
    id="graxpert",
    title="GraXpert Gradient / Denoise",
    group="Background",
    import_path="pro.graxpert_preset",
    callable_name="run_graxpert_via_preset",
    # no ui_method here unless you want to open your optional preset dialog from drops
    # ui_method="open_graxpert_with_preset",  # (only if/when you add one)
    summary=(
        "GraXpert headless runner. "
        "op='background' runs background-extraction with smoothing. "
        "op='denoise' runs denoising with strength, optional ai_version, and batch_size. "
        "Uses GPU by default, but honors preset['gpu'] or saved QSettings graxpert/use_gpu. "
        "Writes temporary float32 TIFF and runs GraXpert CLI like SASv2."
    ),
    presets=[
        PresetSpec(
            key="op",
            type="enum",
            default="background",
            enum=["background", "denoise"],
            help=(
                "Operation mode. "
                "'background' = gradient removal (background-extraction). "
                "'denoise' = GraXpert denoising."
            ),
        ),

        # --- background mode ---
        PresetSpec(
            key="smoothing",
            type="float",
            default=0.10,
            min=0.0,
            max=1.0,
            optional=True,
            help=(
                "Background mode only: smoothing value (0..1). "
                "Ignored when op='denoise'."
            ),
        ),

        # --- denoise mode ---
        PresetSpec(
            key="strength",
            type="float",
            default=0.50,
            min=0.0,
            max=1.0,
            optional=True,
            help=(
                "Denoise mode only: denoise strength (0..1). "
                "Ignored when op='background'."
            ),
        ),
        PresetSpec(
            key="ai_version",
            type="str",
            default="",
            optional=True,
            help=(
                "Denoise mode only: explicit GraXpert AI model version "
                "(e.g. '3.0.2'). Blank/omitted uses latest/auto."
            ),
        ),
        PresetSpec(
            key="batch_size",
            type="int",
            default=None,
            min=1,
            max=64,
            optional=True,
            help=(
                "Denoise mode only: CLI batch size. "
                "If omitted, runner uses 4 when gpu=True else 1."
            ),
        ),

        # --- shared ---
        PresetSpec(
            key="gpu",
            type="bool",
            default=True,
            optional=True,
            help=(
                "Use GPU if available. "
                "If omitted, defaults from QSettings 'graxpert/use_gpu' "
                "(falls back to True)."
            ),
        ),

        # --- advanced / power-user ---
        PresetSpec(
            key="exe",
            type="str",
            default="",
            optional=True,
            help=(
                "Optional explicit path to GraXpert executable. "
                "If omitted, _resolve_graxpert_exec() is used."
            ),
        ),
    ],
    aliases=[
        "grax",
        "remove_gradient_graxpert",
        "graxpert_denoise",
        "graxpert_background",
    ],
    examples=[
        # gradient removal
        "ctx.run_command('graxpert', {'op': 'background', 'smoothing': 0.12})",
        # denoise auto-model, GPU
        "ctx.run_command('graxpert', {'op': 'denoise', 'strength': 0.45, 'gpu': True})",
        # denoise specific model, CPU, explicit batch size
        "ctx.run_command('graxpert', {'op': 'denoise', 'strength': 0.6, 'ai_version': '3.0.2', 'gpu': False, 'batch_size': 1})",
    ],
    supports_mono=True,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


register(CommandSpec(
    id="background_neutral",
    name="Background Neutralization",
    group="Background",
    import_path="pro.backgroundneutral",
    callable_name="run_background_neutral_via_preset",
    summary=(
        "Neutralizes RGB background either automatically or using a user-specified "
        "normalized rectangle. Headless mode blends with active destination mask "
        "before committing."
    ),
    presets=[
        PresetSpec(
            key="mode",
            type="enum",
            default="auto",
            enum=["auto", "rect"],
            help=(
                "Neutralization mode. "
                "'auto' picks an automatic 50x50-ish dark background rect. "
                "'rect' uses rect_norm=[x,y,w,h] in normalized 0..1 coords."
            ),
            optional=True,
        ),
        PresetSpec(
            key="rect_norm",
            type="dict",
            default=None,
            optional=True,
            help=(
                "Required only if mode='rect'. "
                "Normalized rectangle [x0, y0, w, h], each in 0..1. "
                "Example: [0.10, 0.12, 0.20, 0.18]."
            ),
        ),
    ],
    aliases=[
        "background_neutralization",
        "neutralize_background",
        "bn",
    ],
    examples=[
        # auto (default)
        "ctx.run_command('background_neutral', {'mode': 'auto'})",
        # rect mode with normalized ROI
        "ctx.run_command('background_neutral', {'mode': 'rect', 'rect_norm': [0.1, 0.1, 0.2, 0.2]})",
        # mode omitted -> auto
        "ctx.run_command('background_neutral', {})",
    ],
    supports_mono=False,   # your headless code raises on non-RGB
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


# ---------------- Color / WB ----------------

register(CommandSpec(
    id="remove_green",
    name="Remove Green (SCNR)",
    group="Color",
    import_path="pro.remove_green",
    callable_name="apply_remove_green_preset_to_doc",
    ui_method="open_remove_green_dialog",
    summary=(
        "Suppresses excess green using an SCNR-style operation. "
        "Neutral comparator is derived from R/B using avg/max/min. "
        "Optionally preserves perceived lightness (Rec.709 luma) "
        "and blends with the active destination mask."
    ),
    presets=[
        PresetSpec(
            key="amount",
            type="float",
            default=1.0,
            min=0.0,
            max=1.0,
            desc=(
                "Strength of green suppression (0..1). "
                "Aliases accepted: strength, value."
            ),
        ),
        PresetSpec(
            key="mode",
            type="enum",
            default="avg",
            enum=["avg", "max", "min"],
            desc=(
                "Neutral mode for comparing green against R/B: "
                "avg = Average(R,B), max = Max(R,B), min = Min(R,B). "
                "Unknown values fall back to 'avg'."
            ),
        ),
        PresetSpec(
            key="preserve_lightness",
            type="bool",
            default=True,
            desc=(
                "If True, rescales all channels to preserve perceived "
                "lightness after green suppression (Rec.709 luma), "
                "with highlight safety."
            ),
        ),
    ],
    aliases=[
        "scnr",
        "remove_green_cast",
        "green_neutralize",
    ],
    examples=[
        "ctx.run_command('remove_green', {'amount': 1.0, 'mode': 'avg'})",
        "ctx.run_command('remove_green', {'amount': 0.7, 'mode': 'max'})",
        "ctx.run_command('remove_green', {'strength': 0.5, 'mode': 'min', 'preserve_lightness': False})",
    ],
    supports_mono=False,   # _ensure_rgb returns None on mono; headless becomes no-op
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


register(CommandSpec(
    id="white_balance",
    name="White Balance",
    group="Color",
    headless_method="_apply_white_balance_preset_to_doc",
    summary=(
        "White balance for RGB images. Modes: "
        "star (SEP-based), auto (grid/brightest region), manual (per-channel gains). "
        "Star mode falls back to Auto if detection fails."
    ),
    presets=[
        PresetSpec(
            key="mode",
            type="enum",
            default="star",
            enum=["star", "auto", "manual"],
            desc=(
                "White balance mode. "
                "'star' uses SEP star colors; "
                "'auto' uses headless auto WB; "
                "'manual' uses r/g/b gains."
            ),
        ),
        PresetSpec(
            key="threshold",
            type="float",
            default=50.0,
            min=1.0,
            max=100.0,
            optional=True,
            desc=(
                "SEP star threshold (sigma) for star mode. "
                "Higher = fewer/brighter stars."
            ),
        ),
        PresetSpec(
            key="reuse_cached_sources",
            type="bool",
            default=True,
            optional=True,
            desc=(
                "Reuse cached SEP detections for speed (star mode)."
            ),
        ),
        PresetSpec(
            key="r_gain",
            type="float",
            default=1.0,
            min=0.5,
            max=2.0,
            optional=True,
            desc="Manual red gain (manual mode).",
        ),
        PresetSpec(
            key="g_gain",
            type="float",
            default=1.0,
            min=0.5,
            max=2.0,
            optional=True,
            desc="Manual green gain (manual mode).",
        ),
        PresetSpec(
            key="b_gain",
            type="float",
            default=1.0,
            min=0.5,
            max=2.0,
            optional=True,
            desc="Manual blue gain (manual mode).",
        ),
    ],
    examples=[
        "ctx.run_command('white_balance', {'mode': 'star', 'threshold': 45})",
        "ctx.run_command('white_balance', {'mode': 'auto'})",
        "ctx.run_command('white_balance', {'mode': 'manual', 'r_gain': 1.12, 'g_gain': 1.0, 'b_gain': 0.94})",
    ],
    supports_mono=False,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


# ---------------- Luminance tools ----------------

register(CommandSpec(
    id="extract_luminance",
    name="Extract Luminance",
    group="Luminance",
    ui_method="_extract_luminance",                 # now accepts preset optionally
    headless_method="_apply_extract_luminance_preset_to_doc",
    summary=(
        "Create a new mono luminance document from RGB using selectable methods "
        "(Rec.709/601/2020, max, median, equal, or SNR-weighted)."
    ),
    presets=[
        PresetSpec(
            key="mode",
            type="enum",
            default="rec709",
            enum=["rec709", "rec601", "rec2020", "max", "snr", "equal", "median"],
            desc=(
                "Luminance extraction method. "
                "Aliases accepted: method, luma_method, nb_max→max, snr_unequal→snr."
            ),
        ),
    ],
    supports_mono=False,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


register(CommandSpec(
    id="recombine_luminance",
    name="Recombine Luminance",
    group="Luminance",
    import_path="pro.luminancerecombine",
    callable_name="run_recombine_luminance_via_preset",
    ui_method="_recombine_luminance_ui",
    notes=(
        "Replaces target RGB luminance using another open view (mono or RGB). "
        "Headless preset may specify the luminance source by title or doc_ptr. "
        "If omitted, first eligible non-target open doc is used."
    ),
    presets=[
        # ---- luminance source selection ----
        PresetSpec(
            key="source_title",
            type="str",
            default=None,
            optional=True,
            desc=(
                "Title/name of the open document to use as luminance source. "
                "Matches subwindow title or doc.display_name()."
            ),
        ),
        PresetSpec(
            key="source_doc_ptr",
            type="int",
            default=None,
            optional=True,
            desc=(
                "Doc identity pointer (id(doc)) for luminance source. "
                "Useful for replay or scripts that stored doc_ptr."
            ),
        ),

        # ---- luminance compute method ----
        PresetSpec(
            key="method",
            type="enum",
            default="rec709",
            enum=["rec709", "rec601", "rec2020", "max", "snr", "equal", "median"],
            desc=(
                "How to compute luminance if the source is RGB. "
                "rec709/601/2020 use standard weights; "
                "max uses channel max (narrowband style); "
                "snr uses noise-weighted mean; "
                "equal is average RGB; "
                "median is per-pixel median."
            ),
        ),

        # ---- optional explicit weights ----
        PresetSpec(
            key="weights",
            type="dict",
            default=None,
            optional=True,
            desc=(
                "Optional custom RGB weights as [wr,wg,wb]. "
                "Overrides method weights when provided."
            ),
        ),

        # ---- blend / protection ----
        PresetSpec(
            key="blend",
            type="float",
            default=1.0,
            min=0.0, max=1.0,
            desc="Blend factor. 1.0 = full replace, 0.0 = no change.",
        ),
        PresetSpec(
            key="soft_knee",
            type="float",
            default=0.0,
            min=0.0, max=1.0,
            desc=(
                "Highlight protection strength. "
                "0 = off, higher compresses extreme up-scaling."
            ),
        ),
    ],
    examples=[
        # simplest: auto-pick first eligible luminance view
        "ctx.run_command('recombine_luminance', {'method': 'rec709'})",
        # specify a source by title
        "ctx.run_command('recombine_luminance', {'source_title':'M33 — Luminance', 'method':'rec709'})",
        # narrowband / max-style source
        "ctx.run_command('recombine_luminance', {'source_title':'NB Stars', 'method':'max', 'blend':1.0})",
        # custom weights + mild highlight protection
        "ctx.run_command('recombine_luminance', {'weights':[0.2,0.7,0.1], 'soft_knee':0.15})",
    ],
    supports_mono=False,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))

# ---------------- WaveScale family ----------------

register(CommandSpec(
    id="wavescale_hdr",
    name="WaveScale HDR",
    group="Contrast",
    import_path="pro.wavescale_hdr_preset",
    callable_name="run_wavescale_hdr_via_preset",
    ui_method="_open_wavescale_hdr",   # or whatever your main window uses
    summary=(
        "Wavelet-based HDR compression. Builds a luminance mask and compresses "
        "coarse scales to recover dynamic range. Mask gamma controls protection."
    ),
    presets=[
        PresetSpec(
            key="n_scales",
            type="int",
            default=5,
            min=2, max=10,
            desc="Number of wavelet scales. Typical 4–6."
        ),
        PresetSpec(
            key="compression_factor",
            type="float",
            default=1.5,
            min=0.10, max=5.00,
            desc="Coarse-scale compression strength. Higher = more HDR."
        ),
        PresetSpec(
            key="mask_gamma",
            type="float",
            default=5.0,
            min=0.10, max=10.00,
            desc="Gamma shaping for the luminance protection mask."
        ),
    ],
    examples=[
        "ctx.run_command('wavescale_hdr', {'n_scales': 5, 'compression_factor': 1.5, 'mask_gamma': 5.0})",
        "ctx.run_command('wavescale_hdr', {'n_scales': 4, 'compression_factor': 2.2})",
    ],
    supports_mono=True,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))

register(CommandSpec(
    id="wavescale_dark_enhance",
    name="WaveScale Dark Enhance",
    group="Contrast",
    import_path="pro.wavescalede_preset",
    callable_name="run_wavescalede_via_preset",
    ui_method="_open_wavescale_dark_enhance",  # adjust if your main window uses a different name
    summary=(
        "Wavelet-based enhancement of dark / low-contrast structures. "
        "Builds a luminance mask, boosts dark-scale content, and optionally iterates."
    ),
    presets=[
        PresetSpec(
            key="n_scales",
            type="int",
            default=6,
            min=2, max=10,
            desc="Number of wavelet scales. Typical 5–7."
        ),
        PresetSpec(
            key="boost_factor",
            type="float",
            default=5.0,
            min=0.1, max=10.0,   # ✅ match preset dialog
            desc="Boost factor for dark structures."
        ),
        PresetSpec(
            key="mask_gamma",
            type="float",
            default=1.0,
            min=0.1, max=10.0,
            desc="Gamma shaping for the protection mask."
        ),
        PresetSpec(
            key="iterations",
            type="int",
            default=2,
            min=1, max=10,
            desc="Extra enhancement passes."
        ),
    ],
    examples=[
        "ctx.run_command('wavescale_dark_enhance', {'n_scales': 6, 'boost_factor': 5.0})",
        "ctx.run_command('wavescale_dark_enhance', {'n_scales': 7, 'boost_factor': 6.5, 'mask_gamma': 1.3, 'iterations': 3})",
    ],
    supports_mono=True,
    supports_rgb=True,
    supports_linear=True,
    supports_nonlinear=True,
))


# ---------------- Geometry (headless) ----------------

register(CommandSpec(
    id="geom_invert",
    title="Invert",
    group="Geometry",
    headless_method="_apply_geom_invert_to_doc",
))

register(CommandSpec(
    id="geom_flip_horizontal",
    title="Flip Horizontal",
    group="Geometry",
    headless_method="_apply_geom_flip_h_to_doc",
))

register(CommandSpec(
    id="geom_flip_vertical",
    title="Flip Vertical",
    group="Geometry",
    headless_method="_apply_geom_flip_v_to_doc",
))

register(CommandSpec(
    id="geom_rotate_clockwise",
    title="Rotate 90° CW",
    group="Geometry",
    headless_method="_apply_geom_rot_cw_to_doc",
))

register(CommandSpec(
    id="geom_rotate_counterclockwise",
    title="Rotate 90° CCW",
    group="Geometry",
    headless_method="_apply_geom_rot_ccw_to_doc",
))

register(CommandSpec(
    id="geom_rotate_180",
    title="Rotate 180°",
    group="Geometry",
    headless_method="_apply_geom_rot_180_to_doc",
))

register(CommandSpec(
    id="geom_rescale",
    title="Rescale",
    group="Geometry",
    headless_method="_apply_geom_rescale_preset_to_doc",
    presets=[
        PresetSpec("factor", "float", default=1.0, min=0.05, max=20.0,
                   desc="Scale multiplier."),
    ],
))

register(CommandSpec(
    id="aberration_ai",
    title="Aberration AI",
    group="Optics",
    import_path="pro.aberration_ai_preset",
    callable_name="run_aberration_ai_via_preset",
    # ui_method="open_aberration_ai_dialog",   # if you have one; otherwise omit
    presets=[
        PresetSpec(
            "model", "path", default="",
            desc="Path to .onnx model. If empty, uses QSettings('AberrationAI/model_path')."
        ),
        PresetSpec(
            "patch", "int", default=512, min=128, max=2048,
            desc="Patch size for tiled inference. CoreML is clamped to 128."
        ),
        PresetSpec(
            "overlap", "int", default=64, min=16, max=512,
            desc="Overlap between patches."
        ),
        PresetSpec(
            "border_px", "int", default=10, min=0, max=64,
            desc="Border in pixels to preserve from the original."
        ),
        PresetSpec(
            "auto_gpu", "bool", default=True,
            desc="Auto-pick best GPU provider if available (forced off on Apple Silicon)."
        ),
        PresetSpec(
            "provider", "enum", default="CPUExecutionProvider",
            enum=[
                "CPUExecutionProvider",
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
            ],
            desc="Explicit provider when auto_gpu=False."
        ),
    ],
    supports_mono=True,
    supports_rgb=True,
))

register(CommandSpec(
    id="convo",
    title="Convolution / Deconvolution",
    group="Blur & Sharpen",
    import_path="pro.convo_preset",
    callable_name="run_convo_via_preset",
    aliases=[
        "convolution",
        "deconvolution",
        "convo_deconvo",   # keep backward compat
    ],
    presets=[
        PresetSpec("op", "enum", default="convolution",
                   enum=["convolution", "deconvolution", "tv"],
                   desc="Operation type."),

        # shared strength (used by all ops)
        PresetSpec("strength", "float", default=1.0, min=0.0, max=1.0,
                   desc="Blend strength."),

        # --- convolution ---
        PresetSpec("radius", "float", default=5.0, min=0.1, max=200.0,
                   desc="Convolution PSF radius (px)."),
        PresetSpec("kurtosis", "float", default=2.0, min=0.1, max=10.0,
                   desc="Convolution PSF kurtosis (σ)."),
        PresetSpec("aspect", "float", default=1.0, min=0.1, max=10.0,
                   desc="Convolution PSF aspect ratio."),
        PresetSpec("rotation", "float", default=0.0, min=0.0, max=360.0,
                   desc="Convolution PSF rotation (deg)."),

        # --- deconvolution general ---
        PresetSpec("algo", "enum", default="Richardson-Lucy",
                   enum=["Richardson-Lucy", "Wiener", "Larson-Sekanina", "Van Cittert"],
                   desc="Deconvolution algorithm."),

        # RL/Wiener PSF
        PresetSpec("psf_radius", "float", default=3.0, min=0.1, max=100.0,
                   desc="Deconvolution PSF radius (px)."),
        PresetSpec("psf_kurtosis", "float", default=2.0, min=0.1, max=10.0,
                   desc="Deconvolution PSF kurtosis (σ)."),
        PresetSpec("psf_aspect", "float", default=1.0, min=0.1, max=10.0,
                   desc="Deconvolution PSF aspect ratio."),
        PresetSpec("psf_rotation", "float", default=0.0, min=0.0, max=360.0,
                   desc="Deconvolution PSF rotation (deg)."),

        # RL options
        PresetSpec("rl_iter", "int", default=30, min=1, max=200,
                   desc="Richardson–Lucy iterations."),
        PresetSpec("rl_reg", "enum", default="None (Plain R–L)",
                   enum=["None (Plain R–L)", "Tikhonov (L2)", "Total Variation (TV)"],
                   desc="RL regularization."),
        PresetSpec("rl_dering", "bool", default=True,
                   desc="RL de-ring bilateral pass."),
        PresetSpec("luminance_only", "bool", default=True,
                   desc="Run RL/Wiener on L* only."),

        # Wiener options
        PresetSpec("wiener_nsr", "float", default=0.01, min=0.0, max=1.0,
                   desc="Wiener NSR."),
        PresetSpec("wiener_reg", "enum", default="None (Classical Wiener)",
                   enum=["None (Classical Wiener)", "Tikhonov (L2)"],
                   desc="Wiener regularization."),
        PresetSpec("wiener_dering", "bool", default=True,
                   desc="Wiener de-ring pass."),

        # Larson–Sekanina
        PresetSpec("ls_rstep", "float", default=0.0, min=0.0, max=50.0,
                   desc="LS radial step (px)."),
        PresetSpec("ls_astep", "float", default=1.0, min=0.1, max=360.0,
                   desc="LS angular step (deg)."),
        PresetSpec("ls_operator", "enum", default="Divide",
                   enum=["Divide", "Subtract"],
                   desc="LS operator."),
        PresetSpec("ls_blend", "enum", default="SoftLight",
                   enum=["SoftLight", "Screen"],
                   desc="LS blend mode."),

        # Van Cittert
        PresetSpec("vc_iter", "int", default=10, min=1, max=1000,
                   desc="Van Cittert iterations."),
        PresetSpec("vc_relax", "float", default=0.0, min=0.0, max=1.0,
                   desc="Van Cittert relaxation."),

        # --- TV ---
        PresetSpec("tv_weight", "float", default=0.10, min=0.0, max=1.0,
                   desc="TV denoise weight."),
        PresetSpec("tv_iter", "int", default=10, min=1, max=100,
                   desc="TV iterations."),
        PresetSpec("tv_multichannel", "bool", default=True,
                   desc="TV multi-channel."),
    ],
    supports_mono=True,
    supports_rgb=True,
))

register(CommandSpec(
    id="cosmic_clarity",
    title="Cosmic Clarity",
    group="AI",
    import_path="pro.cosmicclarity_preset",
    callable_name="run_cosmicclarity_via_preset",
    presets=[
        PresetSpec("mode", "enum", default="sharpen",
                   enum=["sharpen", "denoise", "both", "superres"],
                   desc="Which CC pipeline to run."),

        PresetSpec("gpu", "bool", default=True,
                   desc="Use GPU acceleration when available."),

        PresetSpec("create_new_view", "bool", default=False,
                   desc="Create new view instead of overwriting active."),

        # --- Sharpen presets ---
        PresetSpec("sharpening_mode", "enum", default="Both",
                   enum=["Both", "Stellar Only", "Non-Stellar Only"],
                   desc="Sharpening mode."),
        PresetSpec("auto_psf", "bool", default=True,
                   desc="Auto-detect PSF for non-stellar sharpening."),
        PresetSpec("nonstellar_psf", "float", default=3.0, min=1.0, max=8.0,
                   desc="Non-stellar PSF strength (1–8)."),
        PresetSpec("stellar_amount", "float", default=0.50, min=0.0, max=1.0,
                   desc="Stellar sharpening amount."),
        PresetSpec("nonstellar_amount", "float", default=0.50, min=0.0, max=1.0,
                   desc="Non-stellar sharpening amount."),
        PresetSpec("sharpen_channels_separately", "bool", default=False,
                   desc="Sharpen R/G/B separately (RGB only)."),

        # --- Denoise presets ---
        PresetSpec("denoise_luma", "float", default=0.50, min=0.0, max=1.0,
                   desc="Luminance denoise strength."),
        PresetSpec("denoise_color", "float", default=0.50, min=0.0, max=1.0,
                   desc="Color denoise strength."),
        PresetSpec("denoise_mode", "enum", default="full",
                   enum=["full", "luminance"],
                   desc="Denoise mode."),
        PresetSpec("separate_channels", "bool", default=False,
                   desc="Denoise RGB channels separately."),

        # --- SuperRes presets ---
        PresetSpec("scale", "int", default=2, min=2, max=4,
                   desc="Super-resolution scale factor."),
    ],
    supports_mono=True,
    supports_rgb=True,
))

register(CommandSpec(
    id="debayer",
    title="Debayer",
    group="Color / CFA",
    import_path="pro.debayer",
    callable_name="run_debayer_via_preset",
    presets=[
        PresetSpec(
            "pattern", "enum", default="auto",
            enum=["auto", "RGGB", "BGGR", "GRBG", "GBRG"],
            desc="Bayer pattern to use. 'auto' tries header then scoring."
        ),
        PresetSpec(
            "method", "enum", default="auto",
            enum=["auto", "edge", "bilinear", "AHD", "DHT"],
            desc="Debayer method. Edge/Bilinear for Bayer; AHD/DHT for X-Trans."
        ),
    ],
    supports_mono=True,   # mosaic is mono input
    supports_rgb=False,   # reject RGB in apply_debayer_preset_to_doc
))

register(CommandSpec(
    id="linear_fit",
    title="Linear Fit",
    group="Calibration",
    import_path="pro.linear_fit",
    callable_name="run_linear_fit_via_preset",
    presets=[
        PresetSpec(
            "rgb_mode_idx", "int", default=0, min=0, max=4,
            desc="RGB target strategy: 0 highest median, 1 lowest, 2 R, 3 G, 4 B."
        ),
        PresetSpec(
            "rescale_mode_idx", "int", default=1, min=0, max=2,
            desc="Out-of-range handling: 0 clip, 1 normalize if needed, 2 leave as-is."
        ),
    ],
    supports_mono=True,
    supports_rgb=True,
    replay_apply_name="apply_linear_fit_to_doc",  # if your CommandSpec supports this hook
))

register(CommandSpec(
    id="morphology",
    title="Morphology",
    group="Masks & Morphology",
    import_path="pro.morphology",
    callable_name="apply_morphology_to_doc",
    presets=[
        PresetSpec(
            "operation", "enum",
            default="erosion",
            enum=["erosion", "dilation", "opening", "closing"],
            desc="Morphological operation."
        ),
        PresetSpec(
            "kernel", "int",
            default=3, min=1, max=31,
            desc="Kernel diameter (odd)."
        ),
        PresetSpec(
            "iterations", "int",
            default=1, min=1, max=10,
            desc="Number of iterations."
        ),
    ],
    supports_mono=True,
    supports_rgb=True,
    replay_apply_name="apply_morphology_to_doc",  # if your CommandSpec supports it
))

register(CommandSpec(
    id="remove_stars",
    title="Remove Stars",
    group="Star Tools",
    import_path="pro.remove_stars_preset",
    callable_name="run_remove_stars_via_preset",
    replay_apply_name="apply_remove_stars_to_doc",
    presets=[
        PresetSpec("tool", "enum", default="starnet",
                   enum=["starnet", "darkstar"],
                   desc="Which star removal engine to use."),

        # StarNet
        PresetSpec("linear", "bool", default=True,
                   desc="Temporary stretch before StarNet then unstretch."),
        PresetSpec("starnet_exe", "path", default="",
                   desc="Optional StarNet exe override; else uses QSettings."),

        # DarkStar
        PresetSpec("disable_gpu", "bool", default=False,
                   desc="Disable GPU for DarkStar."),
        PresetSpec("mode", "enum", default="unscreen",
                   enum=["unscreen", "additive"],
                   desc="DarkStar blending mode."),
        PresetSpec("show_extracted_stars", "bool", default=True,
                   desc="If DarkStar produced stars-only, open it as a new view."),
        PresetSpec("stride", "int", default=512, min=64, max=1024,
                   desc="Chunk/stride size for DarkStar."),
        PresetSpec("darkstar_exe", "path", default="",
                   desc="Optional DarkStar exe override; else uses CosmicClarity root."),
    ],
    supports_mono=True,
    supports_rgb=True,
))



# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
