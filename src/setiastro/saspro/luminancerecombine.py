#src/setiastro/saspro/luminancerecombine.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

from setiastro.saspro.headless_utils import normalize_headless_main, unwrap_docproxy
from setiastro.saspro.ops.command_runner import CommandError
import numpy as np

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    extract_mask_from_document as _active_mask_array_from_doc,
    to_float01_strict as _to_float01_strict,
)

_LUMA_REC709  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_LUMA_REC601  = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
_LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

# ---- Luma profiles (UI selectable) ----
# Key = what the UI stores in self.luma_method / preset["mode"]
# weights must be length-3 (RGB), assumed linear
LUMA_PROFILES: dict[str, dict] = {
    # --- Standard ---
    "rec709": {"method": "rec709", "weights": _LUMA_REC709, "category": "Standard", "description": "Broadband RGB (Rec.709)"},
    "rec601": {"method": "rec601", "weights": _LUMA_REC601, "category": "Standard", "description": "Rec.601"},
    "rec2020": {"method": "rec2020", "weights": _LUMA_REC2020, "category": "Standard", "description": "Rec.2020"},
    "equal": {"method": "equal", "weights": None, "category": "Standard", "description": "Equal RGB"},
    "max": {"method": "max", "weights": None, "category": "Standard", "description": "Max (Narrowband mappings)"},
    "median": {"method": "median", "weights": None, "category": "Standard", "description": "Median RGB"},
    "snr": {"method": "snr", "weights": None, "category": "Standard", "description": "Unequal Noise (SNR)"},

    # --- Sensors (examples — paste your whole list here) ---
    "sensor:Sony IMX571 (ASI2600/QHY268)": {
        "method": "custom",
        "weights": np.array([0.2944, 0.5021, 0.2035], dtype=np.float32),
        "category": "Sensors/Sony Modern BSI",
        "description": "Sony IMX571 26MP APS-C BSI (STARVIS)",
        "info": "Gold standard APS-C. Excellent balance for broadband.",
    },
    "sensor:Sony IMX533 (ASI533)": {
        "method": "custom",
        "weights": np.array([0.2910, 0.5072, 0.2018], dtype=np.float32),
        "category": "Sensors/Sony Modern BSI",
        "description": "Sony IMX533 9MP 1\" Square BSI (STARVIS)",
        "info": "Popular square format. Very low noise.",
    },
    "sensor:Sony IMX455 (ASI6200/QHY600)": {
        "weights": (0.2987, 0.5001, 0.2013),
        "description": "Sony IMX455 61MP Full Frame BSI (STARVIS)",
        "info": "Full frame reference sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX294 (ASI294)": {
        "weights": (0.3068, 0.5008, 0.1925),
        "description": "Sony IMX294 11.7MP 4/3\" BSI",
        "info": "High sensitivity 4/3 format.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX183 (ASI183)": {
        "weights": (0.2967, 0.4983, 0.2050),
        "description": "Sony IMX183 20MP 1\" BSI",
        "info": "High resolution 1-inch sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX178 (ASI178)": {
        "weights": (0.2346, 0.5206, 0.2448),
        "description": "Sony IMX178 6.4MP 1/1.8\" BSI",
        "info": "High resolution entry-level sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX224 (ASI224)": {
        "weights": (0.3402, 0.4765, 0.1833),
        "description": "Sony IMX224 1.27MP 1/3\" BSI",
        "info": "Classic planetary sensor. High Red response.",
        "category": "Sony / Modern BSI",
    },

    # --- SONY STARVIS 2 (NIR Optimized) ---
    "sensor:Sony IMX585 (ASI585) - STARVIS 2": {
        "weights": (0.3431, 0.4822, 0.1747),
        "description": "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2)",
        "info": "NIR optimized. Excellent for H-Alpha/Narrowband.",
        "category": "Sony / STARVIS 2",
    },
    "sensor:Sony IMX662 (ASI662) - STARVIS 2": {
        "weights": (0.3430, 0.4821, 0.1749),
        "description": "Sony IMX662 2.1MP 1/2.8\" BSI (STARVIS 2)",
        "info": "Planetary/Guiding. High Red/NIR sensitivity.",
        "category": "Sony / STARVIS 2",
    },
    "sensor:Sony IMX678/715 - STARVIS 2": {
        "weights": (0.3426, 0.4825, 0.1750),
        "description": "Sony IMX678/715 BSI (STARVIS 2)",
        "info": "High resolution planetary/security sensors.",
        "category": "Sony / STARVIS 2",
    },

    # --- PANASONIC / OTHERS ---
    "sensor:Panasonic MN34230 (ASI1600/QHY163)": {
        "weights": (0.2650, 0.5250, 0.2100),
        "description": "Panasonic MN34230 4/3\" CMOS",
        "info": "Classic Mono/OSC sensor. Optimized weights.",
        "category": "Panasonic",
    },

    # --- CANON DSLR (Averaged Profiles) ---
    "sensor:Canon EOS (Modern - 60D/6D/R)": {
        "weights": (0.2550, 0.5250, 0.2200),
        "description": "Canon CMOS Profile (Modern)",
        "info": "Balanced profile for most Canon EOS cameras (60D, 6D, 5D, R-series).",
        "category": "Canon",
    },
    "sensor:Canon EOS (Legacy - 300D/40D)": {
        "weights": (0.2400, 0.5400, 0.2200),
        "description": "Canon CMOS Profile (Legacy)",
        "info": "For older Canon models (Digic 2/3 era).",
        "category": "Canon",
    },

    # --- NIKON DSLR (Averaged Profiles) ---
    "sensor:Nikon DSLR (Modern - D5300/D850)": {
        "weights": (0.2600, 0.5100, 0.2300),
        "description": "Nikon CMOS Profile (Modern)",
        "info": "Balanced profile for Nikon Expeed 4+ cameras.",
        "category": "Nikon",
    },

    # --- SMART TELESCOPES ---
    "sensor:ZWO Seestar S50": {
        "weights": (0.3333, 0.4866, 0.1801),
        "description": "ZWO Seestar S50 (IMX462)",
        "info": "Specific profile for Seestar S50 smart telescope.",
        "category": "Smart Telescopes",
    },
    "sensor:ZWO Seestar S30": {
        "weights": (0.2928, 0.5053, 0.2019),
        "description": "ZWO Seestar S30",
        "info": "Specific profile for Seestar S30 smart telescope.",
        "category": "Smart Telescopes",
    },
}


# ---------- helpers ----------
def resolve_luma_profile_weights(mode: str | None):
    """
    Returns (resolved_method, weights_or_None, profile_name_or_None)

    - Standard modes return (mode, None or standard weights, None)
    - Sensor profiles return ("custom", weights, <profile display name>)
    """
    if mode is None:
        mode = "rec709"
    key = str(mode).strip()

    # common aliases
    alias = {
        "rec.709": "rec709",
        "rec-709": "rec709",
        "rgb": "rec709",
        "k": "rec709",
        "rec.601": "rec601",
        "rec-601": "rec601",
        "rec.2020": "rec2020",
        "rec-2020": "rec2020",
        "nb_max": "max",
        "narrowband": "max",
        "snr_unequal": "snr",
        "unequal_noise": "snr",
    }
    key = alias.get(key.lower(), key)

    prof = LUMA_PROFILES.get(key)
    if not prof:
        # fallback
        return ("rec709", _LUMA_REC709, None)

    method = str(prof.get("method", "rec709")).strip().lower()
    w = prof.get("weights", None)
    if w is not None:
        w = np.asarray(w, dtype=np.float32)

    if key.startswith("sensor:"):
        # Use "custom" path in compute_luminance by passing weights
        # We'll return resolved_method="rec709" (ignored) and weights=w
        # BUT to keep your API simple: return ("rec709", w, profile_name)
        profile_name = key.split("sensor:", 1)[1].strip()
        return ("rec709", w, profile_name)

    # Standard modes
    return (key, w, None)


def _estimate_noise_sigma_per_channel(img01: np.ndarray) -> np.ndarray:
    # unchanged (but call with strict input)
    a = img01
    if a.ndim == 2:
        a = a[..., None]
    a = a[::4, ::4, :].astype(np.float32, copy=False)
    med = np.median(a, axis=(0,1))
    mad = np.median(np.abs(a - med), axis=(0,1))
    sigma = 1.4826 * mad
    sigma[sigma <= 1e-12] = 1e-12
    return sigma.astype(np.float32)

# ---------- luminance compute (linear) ----------

def compute_luminance(
    img: np.ndarray,
    method: str | None = "rec709",
    weights: Optional[np.ndarray] = None,
    noise_sigma: Optional[np.ndarray] = None,
    normalize_weights: bool = True
) -> np.ndarray:
    """
    Returns 2-D linear luminance Y in [0,1] (float32).
    No per-image normalization. If custom `weights` are supplied and
    `normalize_weights=False`, their absolute sum is respected.
    """
    f = _to_float01_strict(img)

    if f.ndim == 2:
        return np.ascontiguousarray(f.astype(np.float32, copy=False))
    if f.ndim != 3:
        raise ValueError("compute_luminance: expected 2-D or 3-D array.")

    H, W, C = f.shape
    if C == 1:
        return np.ascontiguousarray(f[..., 0].astype(np.float32, copy=False))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 1 or w.size not in (C, 3):
            raise ValueError("weights must be 1-D with length equal to channel count or 3.")
        if normalize_weights:
            s = float(w.sum())
            if s != 0.0:
                w = w / s
        useC = w.size
        lum = np.tensordot(f[..., :useC], w, axes=([2], [0]))
    elif method == "equal":
        lum = f[..., :3].mean(axis=2)
    elif method == "snr":
        if noise_sigma is None:
            raise ValueError("snr method requires noise_sigma per channel.")
        ns = np.asarray(noise_sigma, dtype=np.float32)
        if ns.ndim != 1 or ns.size not in (C, 3):
            raise ValueError("noise_sigma must be 1-D with length equal to channel count or 3.")
        useC = ns.size
        w = 1.0 / (ns[:useC]**2 + 1e-12)
        w = w / w.sum()
        lum = np.tensordot(f[..., :useC], w, axes=([2],[0]))
    elif method == "max":
        lum = f.max(axis=2)
    elif method == "median":
        lum = np.median(f, axis=2)
    elif method == "rec601":
        lum = np.tensordot(f[..., :3], _LUMA_REC601, axes=([2],[0]))
    elif method == "rec2020":
        lum = np.tensordot(f[..., :3], _LUMA_REC2020, axes=([2],[0]))
    else:  # default rec709
        lum = np.tensordot(f[..., :3], _LUMA_REC709, axes=([2],[0]))

    return np.clip(lum.astype(np.float32, copy=False), 0.0, 1.0)

# ---------- luminance recombine (linear scaling) ----------

def recombine_luminance_linear_scale(
    target_rgb: np.ndarray,
    new_L: np.ndarray,
    weights: np.ndarray = _LUMA_REC709,
    eps: float = 1e-6,
    blend: float = 1.0,           # 0..1, 1=full replace
    highlight_soft_knee: float = 0.0  # 0..1, optional protection
) -> np.ndarray:
    """
    Replace linear luminance Y (w·RGB) with `new_L` by per-pixel scaling:
      s = new_L / (Y + eps);  RGB' = RGB * s
    This preserves hue/chroma in linear space and round-trips when new_L==Y.
    Optional: blend (mix with original) and highlight soft-knee protection.
    """
    rgb = _to_float01_strict(target_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Recombine Luminance requires an RGB target image.")

    H, W, _ = rgb.shape
    L = new_L.astype(np.float32)
    if L.shape[:2] != (H, W):
        L = cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR)

    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (3,):
        raise ValueError("weights must be length-3 for RGB recombine.")

    # current Y
    Y = rgb[..., 0]*w[0] + rgb[..., 1]*w[1] + rgb[..., 2]*w[2]
    s = L / (Y + eps)

    if highlight_soft_knee > 0.0:
        # compress extreme upsizing to avoid blowing out tiny Y
        # knee in [0..1], higher = more protection
        k = np.clip(highlight_soft_knee, 0.0, 1.0)
        s = s / (1.0 + k*(s - 1.0))

    out = rgb * s[..., None]
    out = np.clip(out, 0.0, 1.0)

    if 0.0 <= blend < 1.0:
        out = rgb*(1.0 - blend) + out*blend

    return out.astype(np.float32, copy=False)

def _resolve_active_doc_from(main, target_doc=None):
    doc = target_doc
    if doc is None:
        d = getattr(main, "_active_doc", None)
        doc = d() if callable(d) else d
    doc = unwrap_docproxy(doc)
    return doc


def apply_recombine_to_doc(
    target_doc,
    luminance_source_img: np.ndarray,
    method: str = "rec709",
    weights: Optional[np.ndarray] = None,
    noise_sigma: Optional[np.ndarray] = None,
    blend: float = 1.0,
    soft_knee: float = 0.0
):
    """
    Overwrite target_doc.image by recombining with luminance from source (RGB or mono).
    Uses linear scaling recombine; honors destination mask if present.
    """
    base = _to_float01_strict(np.asarray(target_doc.image))

    # Resolve profile (sensor profiles return weights w)
    resolved_method, w, profile_name = resolve_luma_profile_weights(method)

    # Caller override for weights wins (useful for custom UI / scripts)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.size != 3:
            raise ValueError("weights must be a 3-element RGB vector")
    elif w is not None:
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if w.size != 3:
            w = None  # ignore bad profile weights defensively

    # Build L (mono source passes through; RGB is weighted)
    src = _to_float01_strict(luminance_source_img)
    if src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 1):
        L = src if src.ndim == 2 else src[..., 0]
        # For mono L sources, we still want recombine weights to match the selected method/profile.
    else:
        # Noise sigma: if caller provided, use it; otherwise estimate when needed
        ns = None
        if resolved_method == "snr":
            if noise_sigma is not None:
                ns = np.asarray(noise_sigma, dtype=np.float32).reshape(-1)
            else:
                ns = _estimate_noise_sigma_per_channel(src)

        # compute_luminance respects weights override; for sensor/custom profiles w is used
        L = compute_luminance(src, method=resolved_method, weights=w, noise_sigma=ns)

    # For scaling recombine, we need an actual RGB weight vector.
    # If we don't have one from the chosen mode/profile, fall back sensibly.
    if w is not None and w.size == 3:
        recombine_w = w
    else:
        # If your resolver returns w=None for rec709/rec601/rec2020, fill explicitly here:
        if resolved_method == "rec601":
            recombine_w = _LUMA_REC601
        elif resolved_method == "rec2020":
            recombine_w = _LUMA_REC2020
        else:
            recombine_w = _LUMA_REC709

    replaced = recombine_luminance_linear_scale(
        base,
        L,
        weights=recombine_w,
        blend=float(blend),
        highlight_soft_knee=float(soft_knee),
    )

    # Metadata
    md = {"step_name": "Recombine Luminance", "luma_method": resolved_method}
    if profile_name:
        md["luma_profile"] = profile_name
    if w is not None:
        md["luma_weights"] = np.asarray(w, dtype=np.float32).tolist()

    target_doc.apply_edit(replaced.astype(np.float32, copy=False), metadata=md, step_name="Recombine Luminance")


def run_recombine_luminance_via_preset(main_or_ctx, preset=None, target_doc=None):
    """
    Headless entrypoint for recombine_luminance.

    preset supports:
      - source_doc_ptr: int (id(doc))  [highest priority]
      - source_title:  str            [next priority]
      - method, weights, blend, soft_knee (existing)
    If neither source_* is given, first eligible non-target open doc is used.
    """
    from setiastro.saspro.luminancerecombine import apply_recombine_to_doc

    p = dict(preset or {})
    main, doc, dm = normalize_headless_main(main_or_ctx, target_doc)

    # ---- Validate target ----
    if doc is None or getattr(doc, "image", None) is None:
        raise CommandError("recombine_luminance: no active RGB ImageDocument. Load an image first.")

    # ---- Collect open docs (unwrapped) ----
    open_docs = []
    if dm is not None:
        try:
            if hasattr(dm, "all_documents") and callable(dm.all_documents):
                open_docs = [unwrap_docproxy(d) for d in dm.all_documents()]
            elif hasattr(dm, "_docs"):
                open_docs = [unwrap_docproxy(d) for d in dm._docs]
        except Exception:
            open_docs = []

    # Filter to docs that look like images
    def _has_image(d):
        return d is not None and getattr(d, "image", None) is not None

    open_docs = [d for d in open_docs if _has_image(d)]

    # ---- Resolve luminance source ----
    src_doc = None

    # 1) source_doc_ptr
    src_ptr = p.get("source_doc_ptr", None)
    if src_ptr is not None:
        try:
            src_ptr = int(src_ptr)
            for d in open_docs:
                if id(d) == src_ptr:
                    src_doc = d
                    break
        except Exception:
            src_doc = None

    # 2) source_title
    if src_doc is None:
        st = p.get("source_title", None)
        if st:
            st_low = str(st).strip().lower()

            def _title_of(d):
                # prefer display_name() if available
                try:
                    if hasattr(d, "display_name") and callable(d.display_name):
                        return str(d.display_name())
                except Exception:
                    pass
                # fallback to metadata display_name or file basename
                try:
                    md = getattr(d, "metadata", {}) or {}
                    if md.get("display_name"):
                        return str(md["display_name"])
                    fp = md.get("file_path")
                    if fp:
                        import os
                        return os.path.basename(fp)
                except Exception:
                    pass
                return ""

            for d in open_docs:
                if d is doc:
                    continue
                if _title_of(d).lower() == st_low:
                    src_doc = d
                    break

    # 3) auto-pick first eligible non-target doc
    if src_doc is None:
        for d in open_docs:
            if d is doc:
                continue
            src_doc = d
            break

    if src_doc is None:
        raise CommandError(
            "recombine_luminance: no luminance source found. "
            "Open another image, or pass preset {'source_title': ...} "
            "or {'source_doc_ptr': id(doc)}."
        )

    # ---- Execute recombine ----
    src_img = np.asarray(src_doc.image)

    apply_recombine_to_doc(
        doc,
        src_img,
        method=p.get("method", "rec709"),
        weights=p.get("weights", None),
        blend=float(p.get("blend", 1.0)),
        soft_knee=float(p.get("soft_knee", 0.0)),
    )

