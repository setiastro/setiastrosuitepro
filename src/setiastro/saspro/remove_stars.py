# pro/remove_stars.py
from __future__ import annotations
import os
import platform
import shutil
import stat
import tempfile
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog,
    QDialog, QVBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QCheckBox, QSpinBox, QFormLayout, QDialogButtonBox, QWidget, QHBoxLayout
)

# use your legacy I/O functions (as requested)
from setiastro.saspro.legacy.image_manager import save_image, load_image
import glob
try:
    import cv2
except Exception:
    cv2 = None

# Shared utilities
from setiastro.saspro.widgets.image_utils import extract_mask_from_document as _active_mask_array_from_doc

_MAD_NORM = 1.4826

# --------- deterministic, invertible stretch used for StarNet ----------
# ---------- Siril-like MTF (linked) pre-stretch for StarNet ----------
def _robust_peak_sigma(gray: np.ndarray) -> tuple[float, float]:
    gray = gray.astype(np.float32, copy=False)
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    sigma = 1.4826 * mad if mad > 0 else float(gray.std())
    # optional: refine "peak" as histogram mode around median
    try:
        hist, edges = np.histogram(gray, bins=2048, range=(gray.min(), gray.max()))
        peak = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist)+1]))
    except Exception:
        peak = med
    return peak, max(sigma, 1e-8)

def _mtf_apply(x: np.ndarray, shadows: float, midtones: float, highlights: float) -> np.ndarray:
    # x in [0, +], returns [0..1]ish given s,h
    s, m, h = float(shadows), float(midtones), float(highlights)
    denom = max(h - s, 1e-8)
    xp = (x - s) / denom
    # clamp xp to avoid crazy values
    xp = np.clip(xp, 0.0, 1.0)
    num = (m - 1.0) * xp
    den = ((2.0 * m - 1.0) * xp) - m
    y = np.divide(num, den, out=np.zeros_like(xp, dtype=np.float32), where=np.abs(den) > 1e-12)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

def _mtf_inverse(y: np.ndarray, shadows: float, midtones: float, highlights: float) -> np.ndarray:
    """
    Pseudoinverse of MTF, matching Siril's MTF_pseudoinverse() implementation.

    C reference:

    float MTF_pseudoinverse(float y, struct mtf_params params) {
        return ((((params.shadows + params.highlights) * params.midtones
                - params.shadows) * y - params.shadows * params.midtones
                + params.shadows)
                / ((2 * params.midtones - 1) * y - params.midtones + 1));
    }
    """
    s = float(shadows)
    m = float(midtones)
    h = float(highlights)

    yp = np.clip(y.astype(np.float32, copy=False), 0.0, 1.0)

    num = (((s + h) * m - s) * yp - s * m + s)
    den = (2.0 * m - 1.0) * yp - m + 1.0

    x = np.divide(
        num,
        den,
        out=np.full_like(yp, s, dtype=np.float32),  # fallback ~shadows if denom≈0
        where=np.abs(den) > 1e-12
    )

    # Clamp back into [s, h] and then [0,1] for safety
    x = np.clip(x, s, h)
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

def _mtf_params_linked(img_rgb01: np.ndarray, shadowclip_sigma: float = -2.8, targetbg: float = 0.25):
    """
    Compute linked (single) MTF parameters for RGB image in [0..1].
    Returns dict(s=..., m=..., h=...).
    """
    # luminance proxy for stats
    if img_rgb01.ndim == 2:
        gray = img_rgb01
    else:
        gray = img_rgb01.mean(axis=2)
    peak, sigma = _robust_peak_sigma(gray)
    s = peak + shadowclip_sigma * sigma
    # keep [0..1) with margin
    s = float(np.clip(s, gray.min(), max(gray.max() - 1e-6, 0.0)))
    h = 1.0  # Siril effectively normalizes to <=1 before 16-bit TIFF
    # solve for midtones m so that mtf(xp(peak)) = targetbg
    x = (peak - s) / max(h - s, 1e-8)
    x = float(np.clip(x, 1e-6, 1.0 - 1e-6))
    y = float(np.clip(targetbg, 1e-6, 1.0 - 1e-6))
    denom = (2.0 * y * x) - y - x
    m = (x * (y - 1.0)) / denom if abs(denom) > 1e-12 else 0.5
    m = float(np.clip(m, 1e-4, 1.0 - 1e-4))
    return {"s": s, "m": m, "h": h}

def _apply_mtf_linked_rgb(img_rgb01: np.ndarray, p: dict) -> np.ndarray:
    if img_rgb01.ndim == 2:
        img_rgb01 = np.stack([img_rgb01]*3, axis=-1)
    y = np.empty_like(img_rgb01, dtype=np.float32)
    for c in range(3):
        y[..., c] = _mtf_apply(img_rgb01[..., c], p["s"], p["m"], p["h"])
    return np.clip(y, 0.0, 1.0)

def _invert_mtf_linked_rgb(img_rgb01: np.ndarray, p: dict) -> np.ndarray:
    y = np.empty_like(img_rgb01, dtype=np.float32)
    for c in range(3):
        y[..., c] = _mtf_inverse(img_rgb01[..., c], p["s"], p["m"], p["h"])
    return y


def _mtf_params_unlinked(img_rgb01: np.ndarray,
                         shadows_clipping: float = -2.8,
                         targetbg: float = 0.25) -> dict:
    """
    Siril-style per-channel MTF parameter estimation, matching
    find_unlinked_midtones_balance_default() / find_unlinked_midtones_balance().

    Works on float32 data assumed in [0,1].
    Returns dict with arrays: {'s': (C,), 'm': (C,), 'h': (C,)}.
    """
    """
    Siril-style per-channel MTF parameter estimation, matching
    find_unlinked_midtones_balance_default() / find_unlinked_midtones_balance().

    Works on float32 data assumed in [0,1].
    Returns dict with arrays: {'s': (C,), 'm': (C,), 'h': (C,)}.
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    
    # Analyze input shape to handle mono efficiently
    if x.ndim == 2:
        # (H, W) -> treat as single channel
        x_in = x[..., None] # Virtual 3D (H,W,1)
        C_in = 1
    elif x.ndim == 3 and x.shape[2] == 1:
        x_in = x
        C_in = 1
    else:
        x_in = x
        C_in = x.shape[2]

    # Vectorized stats calculation on actual data only
    med = np.median(x_in, axis=(0, 1)).astype(np.float32) # shape (C_in,)
    
    # MAD requires centered abs diff
    diff = np.abs(x_in - med.reshape(1, 1, C_in))
    mad_raw = np.median(diff, axis=(0, 1)).astype(np.float32) # shape (C_in,)
    
    mad = mad_raw * _MAD_NORM
    mad[mad == 0] = 0.001

    inverted_flags = (med > 0.5)
    # If mono, we just check the one channel. If RGB, we check all.
    # Logic below assumes we return 3-channel params s,m,h even for mono input (broadcasted).
    
    # To match original behavior which always returned 3-element arrays for s,m,h:
    # We will compute s_in, m_in, h_in for the input channels, then broadcast to 3.
    
    s_in = np.zeros(C_in, dtype=np.float32)
    m_in = np.zeros(C_in, dtype=np.float32)
    h_in = np.zeros(C_in, dtype=np.float32)

    # We iterate C_in times (1 or 3)
    for c in range(C_in):
        is_inv = inverted_flags[c]
        md = med[c]
        md_dev = mad[c]
        
        if not is_inv:
            # Normal
            c0 = max(md + shadows_clipping * md_dev, 0.0)
            m2 = md - c0
            
            s_in[c] = c0
            m_in[c] = float(_mtf_scalar(m2, targetbg, 0.0, 1.0))
            h_in[c] = 1.0
        else:
            # Inverted
            c1 = min(md - shadows_clipping * md_dev, 1.0)
            m2 = c1 - md
            
            s_in[c] = 0.0
            m_in[c] = 1.0 - float(_mtf_scalar(m2, targetbg, 0.0, 1.0))
            h_in[c] = c1

    # Broadcast to 3 channels if needed
    if C_in == 1:
        s = np.repeat(s_in, 3)
        m = np.repeat(m_in, 3)
        h = np.repeat(h_in, 3)
    else:
        s = s_in
        m = m_in
        h = h_in

    return {"s": s, "m": m, "h": h}


    """
    Apply per-channel MTF exactly. p from _mtf_params_unlinked.
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    h_dim, w_dim = x.shape[:2]
    
    # Output is always 3-channel to satisfy pipeline expectations
    out = np.empty((h_dim, w_dim, 3), dtype=np.float32)
    
    if x.ndim == 2:
        # 1 source channel -> 3 output channels (using per-channel p)
        for c in range(3):
            out[..., c] = _mtf_apply(x, float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    elif x.ndim == 3 and x.shape[2] == 1:
        # 1 source channel (3D) -> 3 output channels
        src = x[..., 0]
        for c in range(3):
            out[..., c] = _mtf_apply(src, float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    else:
        # 3 source channels -> 3 output channels
        # (Assuming input matches p dimension, i.e., 3)
        for c in range(3):
            out[..., c] = _mtf_apply(x[..., c], float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
            
    return np.clip(out, 0.0, 1.0)


    """
    Exact analytic inverse per channel (uses same s/m/h arrays).
    """
    y = np.asarray(img_rgb01, dtype=np.float32)
    h_dim, w_dim = y.shape[:2]
    
    # Output follows input structure logic: return 3-channel
    out = np.empty((h_dim, w_dim, 3), dtype=np.float32)

    if y.ndim == 2:
        for c in range(3):
            out[..., c] = _mtf_inverse(y, float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    elif y.ndim == 3 and y.shape[2] == 1:
        src = y[..., 0]
        for c in range(3):
            out[..., c] = _mtf_inverse(src, float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    else:
        for c in range(3):
            out[..., c] = _mtf_inverse(y[..., c], float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
            
    return np.clip(out, 0.0, 1.0)

    """
    Make sure img is RGB float32 in [0,1], stretch each channel to [0,1]
    using percentiles. Returns (stretched_img, params) where params can be
    fed to _stat_unstretch_rgb() to invert exactly.
    """
    x = img.astype(np.float32, copy=False)
    was_single = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)
    
    # Determine dims
    h_dim, w_dim = x.shape[:2]
    out = np.empty((h_dim, w_dim, 3), dtype=np.float32)
    lo_vals, hi_vals = [], []
    
    # If mono, we compute stats once but apply them conceptually to 3 channels (params replicated)
    # OR we just compute stats once and return replicated params.
    
    if was_single:
        # Use single source for stats
        if x.ndim == 2:
             src = x
        else:
             src = x[..., 0]
             
        lo = float(np.percentile(src, lo_pct))
        hi = float(np.percentile(src, hi_pct))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi - lo < 1e-6: hi = lo + 1e-6
        
        # Fill all 3 channels
        val = (src - lo) / (hi - lo)
        for c in range(3):
            out[..., c] = val
            lo_vals.append(lo)
            hi_vals.append(hi)
            
    else:
        # RGB input
        for c in range(3):
            ch = x[..., c]
            lo = float(np.percentile(ch, lo_pct))
            hi = float(np.percentile(ch, hi_pct))
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi): hi = 1.0
            if hi - lo < 1e-6: hi = lo + 1e-6
            
            lo_vals.append(lo)
            hi_vals.append(hi)
            out[..., c] = (ch - lo) / (hi - lo)

    out = np.clip(out, 0.0, 1.0)
    params = {"lo": lo_vals, "hi": hi_vals, "was_single": was_single}
    return out, params


    """
    Inverse of _stat_stretch_rgb. Expects img RGB float32 [0,1].
    """
    lo = np.asarray(params["lo"], dtype=np.float32)
    hi = np.asarray(params["hi"], dtype=np.float32)
    
    # Just work on input directly; if it matches params length (3), good.
    out = img.astype(np.float32, copy=True)
    if out.ndim == 2:
        # Should not typically happen if we maintain RGB pipeline, but safety:
        # We can't apply 3 diff lo/hi to 1 channel unambiguously. 
        # Assume channel 0 params.
        out = out * (hi[0] - lo[0]) + lo[0]
    else:
        # 3D
        range_c = min(3, out.shape[2])
        for c in range(range_c):
            out[..., c] = out[..., c] * (hi[c] - lo[c]) + lo[c]
            
    out = np.clip(out, 0.0, 1.0)
    
    # Handle the mono-restoration flag
    if params.get("was_single", False):
        if out.ndim == 3 and out.shape[2] >= 3:
            # Average to mono
            mono = out.mean(axis=2, keepdims=False)
            # Re-stack primarily because the StarNet pipeline assumes RGB passing
            # This 'repeat' is unfortunately needed if the pipeline demands 3-channel return 
            # BUT we can check if the caller actually needs it.
            # The original code did: out = np.stack([out] * 3, axis=-1)
            # We will keep this behavior for safety but it's at the very end.
            out = np.stack([mono] * 3, axis=-1)
            
    return out

def _mtf_scalar(x: float, m: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Scalar midtones transfer function matching the PixInsight / Siril spec.

    For x in [lo, hi], rescale to [0,1] and apply:

        M(x; m) = (m - 1) * xp / ((2*m - 1)*xp - m)

    with the special cases x<=lo -> 0, x>=hi -> 1.
    """
    # clamp to the input domain
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0

    denom_range = hi - lo
    if abs(denom_range) < 1e-12:
        return 0.0

    xp = (x - lo) / denom_range  # normalized x in [0,1]

    num = (m - 1.0) * xp
    den = (2.0 * m - 1.0) * xp - m

    if abs(den) < 1e-12:
        # the spec says M(m; m) = 0.5, but if we ever hit this numerically
        # just return 0.5 as a safe fallback
        return 0.5

    y = num / den
    # clamp to [0,1] as PI/Siril do
    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return float(y)


# ------------------------------------------------------------
# Settings helper
# ------------------------------------------------------------
def _get_setting_any(settings, keys: tuple[str, ...], default: str = "") -> str:
    if not settings:
        return default
    for k in keys:
        try:
            v = settings.value(k, "", type=str)
        except Exception:
            v = settings.value(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


# ================== HEADLESS, ARRAY-IN → STARLESS-ARRAY-OUT ==================

def starnet_starless_from_array(arr_rgb01: np.ndarray, settings, *, tmp_prefix="comet") -> np.ndarray:
    """
    Siril-style MTF round-trip for 32-bit data:

      1) Normalize to [0,1] (preserving overall scale separately)
      2) Compute unlinked MTF params per channel (Siril auto-stretch)
      3) Apply unlinked MTF -> 16-bit TIFF for StarNet
      4) StarNet -> read starless 16-bit TIFF
      5) Apply per-channel MTF pseudoinverse with SAME params
      6) Restore original scale if >1.0
    """
    import os
    import platform
    import subprocess
    import numpy as np

    # save_image / load_image / _get_setting_any assumed available
    arr = np.asarray(arr_rgb01, dtype=np.float32)
    was_single = (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)

    exe = _get_setting_any(settings, ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("StarNet executable not configured (settings 'paths/starnet').")

    workdir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(workdir, f"{tmp_prefix}_in.tif")
    out_path = os.path.join(workdir, f"{tmp_prefix}_out.tif")

    # --- Normalize input shape (virtual) and safe values ---
    x_in = arr
    if x_in.ndim == 3 and x_in.shape[2] == 1:
        # Treat (H,W,1) as (H,W) to avoid complications, or keep it.
        # But we DO NOT expand to 3 channels here.
        pass
    
    # We work with x_in directly; _mtf_* functions now handle mono/2d.
    x_in = np.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Preserve original numeric scale if users pass >1.0
    xmax = float(np.max(x_in)) if x_in.size else 1.0
    scale_factor = xmax if xmax > 1.01 else 1.0
    xin = (x_in / scale_factor) if scale_factor > 1.0 else x_in
    xin = np.clip(xin, 0.0, 1.0)
    xin = (x / scale_factor) if scale_factor > 1.0 else x
    xin = np.clip(xin, 0.0, 1.0)

    # --- Siril-style unlinked MTF params + pre-stretch ---
    mtf_params = _mtf_params_unlinked(xin, shadows_clipping=-2.8, targetbg=0.25)
    x_for_starnet = _apply_mtf_unlinked_rgb(xin, mtf_params).astype(np.float32, copy=False)

    # --- Write 16-bit TIFF for StarNet ---
    save_image(
        x_for_starnet, in_path,
        original_format="tif", bit_depth="16-bit",
        original_header=None, is_mono=False, image_meta=None, file_meta=None
    )

    # --- Run StarNet ---
    exe_name = os.path.basename(exe).lower()
    if platform.system() in ("Windows", "Linux"):
        cmd = [exe, in_path, out_path, "256"]
    else:
        cmd = [exe, "--input", in_path, "--output", out_path] if "starnet2" in exe_name else [exe, in_path, out_path]

    rc = subprocess.call(cmd, cwd=workdir)
    if rc != 0 or not os.path.exists(out_path):
        _safe_rm(in_path); _safe_rm(out_path)
        raise RuntimeError(f"StarNet failed rc={rc}")

    starless_s, _, _, _ = load_image(out_path)
    _safe_rm(in_path); _safe_rm(out_path)

    if starless_s.ndim == 2:
        starless_s = np.stack([starless_s] * 3, axis=-1)
    elif starless_s.ndim == 3 and starless_s.shape[2] == 1:
        starless_s = np.repeat(starless_s, 3, axis=2)
    starless_s = np.clip(starless_s.astype(np.float32, copy=False), 0.0, 1.0)

    # --- Apply Siril-style pseudoinverse MTF with SAME params ---
    starless_lin01 = _invert_mtf_unlinked_rgb(starless_s, mtf_params)

    # Restore original scale if we normalized earlier
    if scale_factor > 1.0:
        starless_lin01 *= scale_factor

    result = np.clip(starless_lin01, 0.0, 1.0).astype(np.float32, copy=False)

    # If the source was mono, return mono
    if was_single and result.ndim == 3:
        result = result.mean(axis=2)

    return result


def darkstar_starless_from_array(arr_rgb01: np.ndarray, settings, *, tmp_prefix="comet",
                                 disable_gpu=False, mode="unscreen", stride=512) -> np.ndarray:
    """
    Save arr -> run DarkStar -> load starless -> return starless RGB float32 [0..1].
    """
    exe, base = _resolve_darkstar_exe(type("dummy", (), {"settings": settings}) )
    if not exe or not base:
        raise RuntimeError("Cosmic Clarity DarkStar executable not configured.")
    arr = np.asarray(arr_rgb01, dtype=np.float32)
    was_single = (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)
    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _purge_darkstar_io(base, prefix=None, clear_input=True, clear_output=True)

    in_path = os.path.join(input_dir, f"{tmp_prefix}_in.tif")
    save_image(
        arr, in_path,
        original_format="tif", bit_depth="32-bit floating point",
        original_header=None, is_mono=was_single, image_meta=None, file_meta=None
    )

    args = []
    if disable_gpu: args.append("--disable_gpu")
    args += ["--star_removal_mode", mode, "--chunk_size", str(int(stride))]
    import subprocess
    rc = subprocess.call([exe] + args, cwd=output_dir)
    if rc != 0:
        _safe_rm(in_path); raise RuntimeError(f"DarkStar failed rc={rc}")

    starless_path = os.path.join(output_dir, "imagetoremovestars_starless.tif")
    starless, _, _, _ = load_image(starless_path)
    if starless is None:
        _safe_rm(in_path); raise RuntimeError("DarkStar produced no starless image.")
    if starless.ndim == 2 or (starless.ndim == 3 and starless.shape[2] == 1):
        starless = np.stack([starless] * 3, axis=-1)
    starless = np.clip(starless.astype(np.float32, copy=False), 0.0, 1.0)

    # If the source was mono, collapse back to single channel
    if was_single and starless.ndim == 3:
        starless = starless.mean(axis=2)

    # cleanup typical outputs
    _purge_darkstar_io(base, prefix="imagetoremovestars", clear_input=True, clear_output=True)
    return starless


# ------------------------------------------------------------
# Public entry
# ------------------------------------------------------------
def remove_stars(main, target_doc=None):
    # block interactive UI during/just-after a headless preset run
    if getattr(main, "_remove_stars_headless_running", False):
        return
    if getattr(main, "_remove_stars_guard", False):
        return    

    tool, ok = QInputDialog.getItem(
        main, "Select Star Removal Tool", "Choose a tool:",
        ["StarNet", "CosmicClarityDarkStar"], 0, False
    )
    if not ok:
        return

    # explicit doc wins; otherwise fall back to _active_doc
    doc = target_doc
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "No Image", "Please load an image before removing stars.")
        return

    if tool == "CosmicClarityDarkStar":
        _run_darkstar(main, doc)
    else:
        _run_starnet(main, doc)




def _first_nonzero_bp_per_channel(img3: np.ndarray) -> np.ndarray:
    """Per-channel minimum positive sample (0 if none)."""
    bps = np.zeros(3, dtype=np.float32)
    for c in range(3):
        ch = img3[..., c].reshape(-1)
        pos = ch[ch > 0.0]
        bps[c] = float(pos.min()) if pos.size else 0.0
    return bps


def _prepare_statstretch_input_for_starnet(img_rgb01: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Build the input to StarNet using your statistical stretch flow:
      • record per-channel first-nonzero blackpoints
      • subtract pedestals
      • record per-channel medians
      • unlinked statistical stretch to target 0.25
    Returns: (stretched_for_starnet_01, meta_dict)
    """
    import numpy as np
    from setiastro.saspro.imageops.stretch import stretch_color_image

    x = np.asarray(img_rgb01, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)

    # per-channel pedestal
    bp = _first_nonzero_bp_per_channel(x)
    xin_ped = np.clip(x - bp.reshape((1, 1, 3)), 0.0, 1.0)

    # per-channel medians (after pedestal removal)
    m0 = np.array([float(np.median(xin_ped[..., c])) for c in range(3)], dtype=np.float32)

    # unlinked stat-stretch to 0.25
    x_for_starnet = stretch_color_image(
        xin_ped, target_median=0.25, linked=False,
        normalize=False, apply_curves=False, curves_boost=0.0
    ).astype(np.float32, copy=False)

    meta = {
        "statstretch": True,
        "bp": bp,              # pedestals we subtracted (in 0..1 domain)
        "m0": m0,              # per-channel original medians (post-pedestal)
    }
    return x_for_starnet, meta


def _inverse_statstretch_from_starless(starless_s01: np.ndarray, meta: dict) -> np.ndarray:
    """
    Inverse of the stat-stretch prep:
      • per-channel stretch back to each original median m0[c]
      • add back the saved pedestal bp[c]
    Returns starless in 0..1 domain (float32).
    """
    import numpy as np
    from setiastro.saspro.imageops.stretch import stretch_mono_image

    s = np.asarray(starless_s01, dtype=np.float32)
    if s.ndim == 2:
        s = np.stack([s]*3, axis=-1)
    elif s.ndim == 3 and s.shape[2] == 1:
        s = np.repeat(s, 3, axis=2)
    s = np.clip(s, 0.0, 1.0)

    bp = np.asarray(meta.get("bp"), dtype=np.float32).reshape((1, 1, 3))
    m0 = np.asarray(meta.get("m0"), dtype=np.float32)

    out = np.empty_like(s, dtype=np.float32)
    for c in range(3):
        out[..., c] = stretch_mono_image(
            s[..., c], target_median=float(m0[c]),
            normalize=False, apply_curves=False, curves_boost=0.0
        )

    out = out + bp
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# ------------------------------------------------------------
# StarNet (SASv2-like: 16-bit TIFF in StarNet folder)
# ------------------------------------------------------------
def _run_starnet(main, doc):
    import os
    import platform
    import numpy as np
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    # --- Resolve StarNet exe, persist in settings
    exe = _get_setting_any(getattr(main, "settings", None),
                           ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        exe_path, _ = QFileDialog.getOpenFileName(main, "Select StarNet Executable", "", "Executable Files (*)")
        if not exe_path:
            return
        exe = exe_path
        s = getattr(main, "settings", None)
        if s:
            s.setValue("starnet/exe_path", exe)
            s.setValue("paths/starnet", exe)

    if platform.system() in ("Darwin", "Linux"):
        _ensure_exec_bit(exe)

    sysname = platform.system()
    if sysname not in ("Windows", "Darwin", "Linux"):
        QMessageBox.critical(main, "Unsupported OS",
                             f"The current operating system '{sysname}' is not supported.")
        return

    # --- Ask linearity (SASv2 behavior)
    reply = QMessageBox.question(
        main, "Image Linearity", "Is the current image linear?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )
    is_linear = (reply == QMessageBox.StandardButton.Yes)
    did_stretch = is_linear 
    try:
        main._last_remove_stars_params = {
            "engine": "StarNet",
            "is_linear": bool(is_linear),
            "did_stretch": bool(did_stretch),
            "label": "Remove Stars (StarNet)",
        }
    except Exception:
        pass
    # 🔁 Record headless command for Replay Last
    try:
        main._last_headless_command = {
            "command_id": "remove_stars",
            "preset": {
                "tool": "starnet",
                "linear": bool(is_linear),
            },
        }
        if hasattr(main, "_log"):
            main._log(
                f"[Replay] Recorded remove_stars (StarNet, linear="
                f"{'yes' if is_linear else 'no'})"
            )
    except Exception:
        pass    
    # --- Ensure RGB float32 in safe range (without expanding yet)
    # Starnet needs RGB eventually, but we can compute stats/normalization on mono
    src = np.asarray(doc.image)
    if src.ndim == 3 and src.shape[2] == 1:
        # standardizing shape is cheap
        processing_image = src[..., 0]
    else:
        processing_image = src
        
    processing_image = np.nan_to_num(processing_image.astype(np.float32, copy=False),
                                     nan=0.0, posinf=0.0, neginf=0.0)

    # --- Scale normalization if >1.0
    scale_factor = float(np.max(processing_image))
    if scale_factor > 1.0:
        processing_norm = processing_image / scale_factor
    else:
        processing_norm = processing_image

    # --- Build input/output paths
    starnet_dir = os.path.dirname(exe) or os.getcwd()
    input_image_path  = os.path.join(starnet_dir, "imagetoremovestars.tif")
    output_image_path = os.path.join(starnet_dir, "starless.tif")

    # --- Prepare input for StarNet (Siril-style MTF pre-stretch for linear data) ---
    img_for_starnet = processing_norm
    if is_linear:
        # Siril-style unlinked MTF params from linear normalized image
        mtf_params = _mtf_params_unlinked(processing_norm, shadows_clipping=-2.8, targetbg=0.25)
        img_for_starnet = _apply_mtf_unlinked_rgb(processing_norm, mtf_params)

        # 🔐 Stash EXACT params for inverse step later
        try:
            setattr(main, "_starnet_stat_meta", {
                "scheme": "siril_mtf",
                "s": np.asarray(mtf_params["s"], dtype=np.float32),
                "m": np.asarray(mtf_params["m"], dtype=np.float32),
                "h": np.asarray(mtf_params["h"], dtype=np.float32),
                "scale": float(scale_factor),
            })
        except Exception:
            pass
    else:
        # non-linear: do not try to invert any pre-stretch later
        if hasattr(main, "_starnet_stat_meta"):
            delattr(main, "_starnet_stat_meta")


    # --- Write TIFF for StarNet
    try:
        save_image(img_for_starnet, input_image_path,
                   original_format="tif", bit_depth="16-bit",
                   original_header=None, is_mono=False, image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to write input TIFF:\n{e}")
        return

    # --- Launch StarNet in a worker (keeps your progress dialog)
    exe_name = os.path.basename(exe).lower()
    if sysname in ("Windows", "Linux"):
        command = [exe, input_image_path, output_image_path, "256"]
    else:  # macOS
        if "starnet2" in exe_name:
            command = [exe, "--input", input_image_path, "--output", output_image_path]
        else:
            command = [exe, input_image_path, output_image_path]

    dlg = _ProcDialog(main, title="StarNet Progress")
    thr = _ProcThread(command, cwd=starnet_dir)
    thr.output_signal.connect(dlg.append_text)

    # Capture everything we need in the closure for finish handler
    thr.finished_signal.connect(
        lambda rc, ds=did_stretch: _on_starnet_finished(
            main, doc, rc, dlg, input_image_path, output_image_path, ds
        )
    )
    dlg.cancel_button.clicked.connect(thr.cancel)

    dlg.show()
    thr.start()
    dlg.exec()


def _on_starnet_finished(main, doc, return_code, dialog, input_path, output_path, did_stretch):
    import os
    import numpy as np
    from PyQt6.QtWidgets import QMessageBox
    from setiastro.saspro.imageops.stretch import stretch_mono_image  # used for statistical inverse

    def _first_nonzero_bp_per_channel(img3: np.ndarray) -> np.ndarray:
        bps = np.zeros(3, dtype=np.float32)
        for c in range(3):
            ch = img3[..., c].reshape(-1)
            pos = ch[ch > 0.0]
            bps[c] = float(pos.min()) if pos.size else 0.0
        return bps

    dialog.append_text(f"\nProcess finished with return code {return_code}.\n")
    if return_code != 0:
        QMessageBox.critical(main, "StarNet Error", f"StarNet failed with return code {return_code}.")
        _safe_rm(input_path); _safe_rm(output_path)
        dialog.close()
        return

    if not os.path.exists(output_path):
        QMessageBox.critical(main, "StarNet Error", "Starless image was not created.")
        _safe_rm(input_path)
        dialog.close()
        return

    dialog.append_text(f"Starless image found at {output_path}. Loading image...\n")
    starless_rgb, _, _, _ = load_image(output_path)
    _safe_rm(input_path); _safe_rm(output_path)

    if starless_rgb is None:
        QMessageBox.critical(main, "StarNet Error", "Failed to load starless image.")
        dialog.close()
        return

    # ensure 3ch float32 in [0..1]
    if starless_rgb.ndim == 2:
        starless_rgb = np.stack([starless_rgb] * 3, axis=-1)
    elif starless_rgb.ndim == 3 and starless_rgb.shape[2] == 1:
        starless_rgb = np.repeat(starless_rgb, 3, axis=2)
    starless_rgb = np.clip(starless_rgb.astype(np.float32, copy=False), 0.0, 1.0)

    # original image (from the doc) as 3ch float32, track if it was mono
    orig = np.asarray(doc.image)
    if orig.ndim == 2:
        original_rgb = np.stack([orig] * 3, axis=-1)
        orig_was_mono = True
    elif orig.ndim == 3 and orig.shape[2] == 1:
        original_rgb = np.repeat(orig, 3, axis=2)
        orig_was_mono = True
    else:
        original_rgb = orig
        orig_was_mono = False
    original_rgb = original_rgb.astype(np.float32, copy=False)


    # ---- Inversion back to the document’s domain ----
    if did_stretch:
        # Prefer the new Siril-style MTF meta if present
        meta = getattr(main, "_starnet_stat_meta", None)
        mtf_params_legacy = getattr(main, "_starnet_last_mtf_params", None)

        if isinstance(meta, dict) and meta.get("scheme") == "siril_mtf":
            dialog.append_text("Unstretching (Siril-style MTF pseudoinverse)...\n")
            try:
                s_vec = np.asarray(meta.get("s"), dtype=np.float32)
                m_vec = np.asarray(meta.get("m"), dtype=np.float32)
                h_vec = np.asarray(meta.get("h"), dtype=np.float32)
                scale_factor = float(meta.get("scale", 1.0))

                p = {"s": s_vec, "m": m_vec, "h": h_vec}
                inv = _invert_mtf_unlinked_rgb(starless_rgb, p)

                if scale_factor > 1.0:
                    inv = inv * scale_factor

                starless_rgb = np.clip(inv, 0.0, 1.0)
            except Exception as e:
                dialog.append_text(f"⚠️ Siril-style MTF inverse failed: {e}\n")

        elif isinstance(meta, dict) and meta.get("scheme") == "statstretch":
            # Back-compat: statistical round-trip with bp/m0
            dialog.append_text("Unstretching (statistical inverse w/ original BP/M0)...\n")

            bp_vec = np.asarray(meta.get("bp"), dtype=np.float32)
            m0_vec = np.asarray(meta.get("m0"), dtype=np.float32)
            scale_factor = float(meta.get("scale", 1.0))

            inv = np.empty_like(starless_rgb, dtype=np.float32)
            for c in range(3):
                inv[..., c] = stretch_mono_image(
                    starless_rgb[..., c],
                    target_median=float(m0_vec[c]),
                    normalize=False, apply_curves=False, curves_boost=0.0
                )

            inv += bp_vec.reshape((1, 1, 3))
            inv = np.clip(inv, 0.0, 1.0)
            if scale_factor > 1.0:
                inv *= scale_factor
            starless_rgb = np.clip(inv, 0.0, 1.0)

        elif mtf_params_legacy:
            # Very old MTF path (linked, single triple) – keep for safety
            dialog.append_text("Unstretching (legacy MTF inverse)...\n")
            try:
                starless_rgb = _invert_mtf_linked_rgb(starless_rgb, mtf_params_legacy)
                sc = float(mtf_params_legacy.get("scale", 1.0))
                if sc > 1.0:
                    starless_rgb = starless_rgb * sc
            except Exception as e:
                dialog.append_text(f"⚠️ Legacy MTF inverse failed: {e}\n")
            starless_rgb = np.clip(starless_rgb, 0.0, 1.0)

        # Clean up stashed meta so it can't leak to future ops
        try:
            if hasattr(main, "_starnet_stat_meta"):
                delattr(main, "_starnet_stat_meta")
        except Exception:
            pass



    # ---- Stars-Only = original − starless (linear-domain diff) ----
    dialog.append_text("Generating stars-only image...\n")
    stars_only = np.clip(original_rgb - starless_rgb, 0.0, 1.0)

    # apply active mask (doc-based)
    m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
    if m3 is not None:
        stars_only *= m3
        dialog.append_text("✅ Applied active mask to the stars-only image.\n")
    else:
        dialog.append_text("ℹ️ No active mask for stars-only; skipping.\n")

    # If the original doc was mono, return a mono stars-only image
    if orig_was_mono:
        stars_to_push = stars_only.mean(axis=2).astype(np.float32, copy=False)
    else:
        stars_to_push = stars_only

    # push Stars-Only as new document with suffix _stars
    _push_as_new_doc(main, doc, stars_to_push, title_suffix="_stars", source="Stars-Only (StarNet)")
    dialog.append_text("Stars-only image pushed.\n")

    # mask-blend starless with original using active mask, then overwrite current view
    dialog.append_text("Preparing to update current view with starless (mask-blend)...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)

    # If the original doc was mono, collapse back to single-channel
    if orig_was_mono:
        final_to_apply = final_starless.mean(axis=2).astype(np.float32, copy=False)
    else:
        final_to_apply = final_starless.astype(np.float32, copy=False)

    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
        }

        # 🔹 Attach replay-last metadata
        rp = getattr(main, "_last_remove_stars_params", None)
        if isinstance(rp, dict):
            replay_params = dict(rp)  # shallow copy so we don't mutate the stored one
        else:
            replay_params = {
                "engine": "StarNet",
                "is_linear": bool(did_stretch),
                "did_stretch": bool(did_stretch),
                "label": "Remove Stars (StarNet)",
            }

        replay_params.setdefault("engine", "StarNet")
        replay_params.setdefault("label", "Remove Stars (StarNet)")

        meta["replay_last"] = {
            "op": "remove_stars",
            "params": replay_params,
        }

        # Clean up the stash so it can't leak to the next unrelated op
        try:
            if hasattr(main, "_last_remove_stars_params"):
                delattr(main, "_last_remove_stars_params")
        except Exception:
            pass

        doc.apply_edit(
            final_to_apply,
            metadata=meta,
            step_name="Stars Removed"
        )
        if hasattr(main, "_log"):
            main._log("Stars Removed (StarNet)")
    except Exception as e:
        QMessageBox.critical(main, "StarNet Error", f"Failed to apply starless result:\n{e}")

    dialog.append_text("Temporary files cleaned up.\n")
    dialog.close()



# ------------------------------------------------------------
# CosmicClarityDarkStar
# ------------------------------------------------------------
def _run_darkstar(main, doc):
    exe, base = _resolve_darkstar_exe(main)
    if not exe or not base:
        QMessageBox.critical(main, "Cosmic Clarity Folder Error",
                             "Cosmic Clarity Dark Star executable not set.")
        return

    # --- Input/output folders per SASv2 ---
    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _purge_darkstar_io(base, prefix=None, clear_input=True, clear_output=True)

    # --- Config dialog (same as before) ---
    cfg = DarkStarConfigDialog(main)
    if not cfg.exec():
        return
    params = cfg.get_values()
    disable_gpu = params["disable_gpu"]
    mode = params["mode"]                         # "unscreen" or "additive"
    show_extracted_stars = params["show_extracted_stars"]
    stride = params["stride"]                     # 64..1024, default 512

    # 🔹 Ask if image is linear (so we know whether to MTF-prestretch)
    reply = QMessageBox.question(
        main, "Image Linearity", "Is the current image linear?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes
    )
    is_linear = (reply == QMessageBox.StandardButton.Yes)
    did_prestretch = is_linear

    # 🔹 Stash parameters for replay-last
    try:
        main._last_remove_stars_params = {
            "engine": "CosmicClarityDarkStar",
            "disable_gpu": bool(disable_gpu),
            "mode": mode,
            "show_extracted_stars": bool(show_extracted_stars),
            "stride": int(stride),
            "is_linear": bool(is_linear),
            "did_prestretch": bool(did_prestretch),
            "label": "Remove Stars (DarkStar)",
        }
    except Exception:
        pass

    # 🔁 Record headless command for Replay Last
    try:
        main._last_headless_command = {
            "command_id": "remove_stars",
            "preset": {
                "tool": "darkstar",
                "disable_gpu": bool(disable_gpu),
                "mode": mode,
                "show_extracted_stars": bool(show_extracted_stars),
                "stride": int(stride),
                "is_linear": bool(is_linear),
                "did_prestretch": bool(did_prestretch),
            },
        }
        if hasattr(main, "_log"):
            main._log(
                "[Replay] Recorded remove_stars (DarkStar, "
                f"mode={mode}, stride={int(stride)}, "
                f"gpu={'off' if disable_gpu else 'on'}, "
                f"stars={'on' if show_extracted_stars else 'off'}, "
                f"linear={'yes' if is_linear else 'no'})"
            )
    except Exception:
        pass

    # --- Build processing image (RGB float32, normalized) ---
    # DarkStar needs RGB, but we can delay expansion until save
    src = np.asarray(doc.image)
    if src.ndim == 3 and src.shape[2] == 1:
        processing_image = src[..., 0]
    else:
        processing_image = src

    processing_image = np.nan_to_num(
        processing_image.astype(np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    scale_factor = float(np.max(processing_image)) if processing_image.size else 1.0
    if scale_factor > 1.0:
        processing_norm = processing_image / scale_factor
    else:
        processing_norm = processing_image
    processing_norm = np.clip(processing_norm, 0.0, 1.0)

    # --- Optional Siril-style MTF pre-stretch for linear data ---
    img_for_darkstar = processing_norm
    if is_linear:
        try:
            mtf_params = _mtf_params_unlinked(
                processing_norm,
                shadows_clipping=-2.8,
                targetbg=0.25
            )
            img_for_darkstar = _apply_mtf_unlinked_rgb(processing_norm, mtf_params)

            # 🔐 Stash EXACT params for inverse step later
            setattr(main, "_darkstar_mtf_meta", {
                "s": np.asarray(mtf_params["s"], dtype=np.float32),
                "m": np.asarray(mtf_params["m"], dtype=np.float32),
                "h": np.asarray(mtf_params["h"], dtype=np.float32),
                "scale": float(scale_factor),
            })
            if hasattr(main, "_log"):
                main._log("[DarkStar] Applying Siril-style MTF pre-stretch for linear image.")
        except Exception as e:
            # If anything goes wrong, fall back to un-stretched normalized image
            img_for_darkstar = processing_norm
            try:
                if hasattr(main, "_darkstar_mtf_meta"):
                    delattr(main, "_darkstar_mtf_meta")
            except Exception:
                pass
            if hasattr(main, "_log"):
                main._log(f"[DarkStar] MTF pre-stretch failed, using normalized image only: {e}")
    else:
        # Non-linear: don't store any pre-stretch meta
        try:
            if hasattr(main, "_darkstar_mtf_meta"):
                delattr(main, "_darkstar_mtf_meta")
        except Exception:
            pass

    # --- Save pre-stretched image as 32-bit float TIFF for DarkStar ---
    in_path = os.path.join(input_dir, "imagetoremovestars.tif")
    try:
        # Check if we need to expand on-the-fly for DarkStar (it expects RGB input)
        # If img_for_darkstar is mono, save_image might save mono.
        # "is_mono=False" flag to save_image hints we want RGB.
        # If the array is 2D, save_image might still save mono unless we feed it 3D.
        # For safety with DarkStar, we create the 3D view now if needed.
        
        to_save = img_for_darkstar
        if to_save.ndim == 2:
            to_save = np.stack([to_save]*3, axis=-1)
        elif to_save.ndim == 3 and to_save.shape[2] == 1:
            to_save = np.repeat(to_save, 3, axis=2)
            
        save_image(
            to_save,
            in_path,
            original_format="tif",
            bit_depth="32-bit floating point",
            original_header=None,
            is_mono=False,  # we always send RGB to DarkStar
            image_meta=None,
            file_meta=None
        )
    except Exception as e:
        QMessageBox.critical(main, "Cosmic Clarity", f"Failed to write input TIFF:\n{e}")
        return

    # --- Build CLI exactly like SASv2 (using --chunk_size, not chunk_size) ---
    args = []
    if disable_gpu:
        args.append("--disable_gpu")
    args += ["--star_removal_mode", mode]
    if show_extracted_stars:
        args.append("--show_extracted_stars")
    args += ["--chunk_size", str(stride)]

    command = [exe] + args

    dlg = _ProcDialog(main, title="CosmicClarityDarkStar Progress")
    thr = _ProcThread(command, cwd=output_dir)
    thr.output_signal.connect(dlg.append_text)
    thr.finished_signal.connect(
        lambda rc, base=base, ds=did_prestretch: _on_darkstar_finished(
            main, doc, rc, dlg, in_path, output_dir, base, ds
        )
    )
    dlg.cancel_button.clicked.connect(thr.cancel)

    dlg.show()
    thr.start()
    dlg.exec()




def _resolve_darkstar_exe(main):
    """
    Return (exe_path, base_folder) or (None, None) on cancel/error.
    Accepts either a folder (stored) or a direct executable path.
    Saves the folder back to QSettings under 'paths/cosmic_clarity'.
    """
    settings = getattr(main, "settings", None)
    raw = _get_setting_any(settings, ("paths/cosmic_clarity", "cosmic_clarity_folder"), "")

    def _platform_exe_name():
        return "setiastrocosmicclarity_darkstar.exe" if platform.system() == "Windows" \
               else "setiastrocosmicclarity_darkstar"

    exe_name = _platform_exe_name()

    exe_path = None
    base_folder = None

    if raw:
        if os.path.isfile(raw):
            # user stored the executable path directly
            exe_path = raw
            base_folder = os.path.dirname(raw)
        elif os.path.isdir(raw):
            # user stored the parent folder
            base_folder = raw
            exe_path = os.path.join(base_folder, exe_name)

    # if missing or invalid, let user pick the executable directly
    if not exe_path or not os.path.exists(exe_path):
        picked, _ = QFileDialog.getOpenFileName(main, "Select CosmicClarityDarkStar Executable", "", "Executable Files (*)")
        if not picked:
            return None, None
        exe_path = picked
        base_folder = os.path.dirname(picked)

    # ensure exec bit on POSIX
    if platform.system() in ("Darwin", "Linux"):
        _ensure_exec_bit(exe_path)

    # persist folder (not the exe) to the canonical key
    if settings:
        settings.setValue("paths/cosmic_clarity", base_folder)
        settings.sync()

    return exe_path, base_folder


def _on_darkstar_finished(main, doc, return_code, dialog, in_path, output_dir, base_folder, did_prestretch):
    dialog.append_text(f"\nProcess finished with return code {return_code}.\n")
    if return_code != 0:
        QMessageBox.critical(main, "CosmicClarityDarkStar Error",
                             f"CosmicClarityDarkStar failed with return code {return_code}.")
        _safe_rm(in_path); dialog.close(); return

    starless_path = os.path.join(output_dir, "imagetoremovestars_starless.tif")
    if not os.path.exists(starless_path):
        QMessageBox.critical(main, "CosmicClarityDarkStar Error", "Starless image was not created.")
        _safe_rm(in_path); dialog.close(); return

    dialog.append_text(f"Loading starless image from {starless_path}...\n")
    starless, _, _, _ = load_image(starless_path)
    if starless is None:
        QMessageBox.critical(main, "CosmicClarityDarkStar Error", "Failed to load starless image.")
        _safe_rm(in_path); dialog.close(); return

    if starless.ndim == 2 or (starless.ndim == 3 and starless.shape[2] == 1):
        starless_rgb = np.stack([starless] * 3, axis=-1)
    else:
        starless_rgb = starless
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    src = np.asarray(doc.image)
    if src.ndim == 2:
        original_rgb = np.stack([src] * 3, axis=-1)
        orig_was_mono = True
    elif src.ndim == 3 and src.shape[2] == 1:
        original_rgb = np.repeat(src, 3, axis=2)
        orig_was_mono = True
    else:
        original_rgb = src
        orig_was_mono = False
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # --- Undo the MTF pre-stretch (if we did one) ---
    if did_prestretch:
        meta = getattr(main, "_darkstar_mtf_meta", None)
        if isinstance(meta, dict):
            dialog.append_text("Unstretching starless result (DarkStar MTF inverse)...\n")
            try:
                s_vec = np.asarray(meta.get("s"), dtype=np.float32)
                m_vec = np.asarray(meta.get("m"), dtype=np.float32)
                h_vec = np.asarray(meta.get("h"), dtype=np.float32)
                scale = float(meta.get("scale", 1.0))

                p = {"s": s_vec, "m": m_vec, "h": h_vec}
                inv = _invert_mtf_unlinked_rgb(starless_rgb, p)

                if scale > 1.0:
                    inv *= scale

                starless_rgb = np.clip(inv, 0.0, 1.0)
            except Exception as e:
                dialog.append_text(f"⚠️ DarkStar MTF inverse failed: {e}\n")

        # Clean up pre-stretch meta so it can't leak into another op
        try:
            if hasattr(main, "_darkstar_mtf_meta"):
                delattr(main, "_darkstar_mtf_meta")
        except Exception:
            pass

    # --- stars-only optional push (as before) ---
    stars_path = os.path.join(output_dir, "imagetoremovestars_stars_only.tif")
    if os.path.exists(stars_path):
        dialog.append_text(f"Loading stars-only image from {stars_path}...\n")
        stars_only, _, _, _ = load_image(stars_path)
        if stars_only is not None:
            if stars_only.ndim == 2 or (stars_only.ndim == 3 and stars_only.shape[2] == 1):
                stars_only = np.stack([stars_only] * 3, axis=-1)
            stars_only = stars_only.astype(np.float32, copy=False)
            m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
            if m3 is not None:
                stars_only *= m3
                dialog.append_text("✅ Applied active mask to stars-only image.\n")
            else:
                dialog.append_text("ℹ️ Mask not active for stars-only; skipping.\n")

            # If the original doc was mono, collapse stars-only back to single channel
            if orig_was_mono:
                stars_to_push = stars_only.mean(axis=2).astype(np.float32, copy=False)
            else:
                stars_to_push = stars_only

            _push_as_new_doc(main, doc, stars_to_push, title_suffix="_stars", source="Stars-Only (DarkStar)")
        else:
            dialog.append_text("Failed to load stars-only image.\n")
    else:
        dialog.append_text("No stars-only image generated.\n")

    # --- Mask-blend starless → overwrite current doc (in original domain) ---
    dialog.append_text("Mask-blending starless image before update...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)

    # If the original doc was mono, collapse back to single-channel
    if orig_was_mono:
        final_to_apply = final_starless.mean(axis=2).astype(np.float32, copy=False)
    else:
        final_to_apply = final_starless.astype(np.float32, copy=False)

    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": bool(orig_was_mono),
        }

        # 🔹 Attach replay-last metadata
        rp = getattr(main, "_last_remove_stars_params", None)
        if isinstance(rp, dict):
            replay_params = dict(rp)
        else:
            replay_params = {
                "engine": "CosmicClarityDarkStar",
                "label": "Remove Stars (DarkStar)",
            }

        replay_params.setdefault("engine", "CosmicClarityDarkStar")
        replay_params.setdefault("label", "Remove Stars (DarkStar)")

        meta["replay_last"] = {
            "op": "remove_stars",
            "params": replay_params,
        }

        # Clean up stash
        try:
            if hasattr(main, "_last_remove_stars_params"):
                delattr(main, "_last_remove_stars_params")
        except Exception:
            pass

        doc.apply_edit(
            final_to_apply,
            metadata=meta,
            step_name="Stars Removed"
        )
        if hasattr(main, "_log"):
            main._log("Stars Removed (DarkStar)")
    except Exception as e:
        QMessageBox.critical(main, "CosmicClarityDarkStar", f"Failed to apply result:\n{e}")

    # --- cleanup ---
    try:
        _safe_rm(in_path)
        _safe_rm(starless_path)
        _safe_rm(os.path.join(output_dir, "imagetoremovestars_stars_only.tif"))

        # 🔸 Final sweep: nuke any imagetoremovestars* leftovers in both dirs
        base_folder = os.path.dirname(output_dir)  # <-- derive CC base from output_dir
        _purge_darkstar_io(base_folder, prefix="imagetoremovestars", clear_input=True, clear_output=True)

        dialog.append_text("Temporary files cleaned up.\n")
    except Exception as e:
        dialog.append_text(f"Cleanup error: {e}\n")

    dialog.close()


# ------------------------------------------------------------
# Mask helpers (doc-centric)
# ------------------------------------------------------------
# _active_mask_array_from_doc is now imported from setiastro.saspro.widgets.image_utils


def _active_mask3_from_doc(doc, w, h) -> np.ndarray | None:
    """Return 3-channel mask resized to (h,w) if a doc-level mask exists; else None."""
    m = _active_mask_array_from_doc(doc)
    if m is None:
        return None
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = (np.linspace(0, m.shape[0] - 1, h)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, w)).astype(np.int32)
            m = m[yi][:, xi]
    return np.repeat(m[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def _mask_blend_with_doc_mask(doc, starless_rgb: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """Blend using mask from doc if present: result = starless*m + original*(1-m)."""
    m = _active_mask_array_from_doc(doc)
    if m is None:
        return starless_rgb
    h, w = starless_rgb.shape[:2]
    if m.shape != (h, w):
        if cv2 is not None:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            yi = (np.linspace(0, m.shape[0] - 1, h)).astype(np.int32)
            xi = (np.linspace(0, m.shape[1] - 1, w)).astype(np.int32)
            m = m[yi][:, xi]
    m3 = np.repeat(m[:, :, None], 3, axis=2)
    return np.clip(starless_rgb * m3 + original_rgb * (1.0 - m3), 0.0, 1.0).astype(np.float32, copy=False)


def _derive_view_base_title(main, doc) -> str:
    """
    Prefer the active view's title (respecting per-view rename/override),
    fallback to the document display name, then to doc.name, and finally 'Image'.
    Also strips any decorations (mask glyph, 'Active View:' prefix) if available.
    """
    # 1) Ask main for a subwindow for this document, if it exposes a helper
    try:
        if hasattr(main, "_subwindow_for_document"):
            sw = main._subwindow_for_document(doc)
            if sw:
                w = sw.widget() if hasattr(sw, "widget") else sw
                # Preferred: view's effective title (includes per-view override)
                if hasattr(w, "_effective_title"):
                    t = w._effective_title() or ""
                else:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                if hasattr(w, "_strip_decorations"):
                    t, _ = w._strip_decorations(t)
                if t.strip():
                    return t.strip()
    except Exception:
        pass

    # 2) Try scanning MDI for a subwindow whose widget holds this document
    try:
        mdi = (getattr(main, "mdi_area", None)
               or getattr(main, "mdiArea", None)
               or getattr(main, "mdi", None))
        if mdi and hasattr(mdi, "subWindowList"):
            for sw in mdi.subWindowList():
                w = sw.widget()
                if getattr(w, "document", None) is doc:
                    t = sw.windowTitle() if hasattr(sw, "windowTitle") else ""
                    if hasattr(w, "_strip_decorations"):
                        t, _ = w._strip_decorations(t)
                    if t.strip():
                        return t.strip()
    except Exception:
        pass

    # 3) Fallback to document's display name (then name, then generic)
    try:
        if hasattr(doc, "display_name"):
            t = doc.display_name()
            if t and t.strip():
                return t.strip()
    except Exception:
        pass
    return (getattr(doc, "name", "") or "Image").strip()


# ------------------------------------------------------------
# New document helper
# ------------------------------------------------------------
def _push_as_new_doc(main, doc, arr: np.ndarray, title_suffix="_stars", source="Stars-Only"):
    dm = getattr(main, "docman", None)
    if not dm or not hasattr(dm, "open_array"):
        return
    try:
        # Use the current view's title if available (respects per-view rename)
        base = _derive_view_base_title(main, doc)

        # avoid double-suffix if user already named it with the suffix
        if title_suffix and base.endswith(title_suffix):
            title = base
        else:
            title = f"{base}{title_suffix}"

        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2),
            "source": source,
        }
        newdoc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=title)
        if hasattr(main, "_spawn_subwindow_for"):
            main._spawn_subwindow_for(newdoc)
    except Exception:
        pass



# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_exec_bit(path: str):
    if platform.system() == "Windows":
        return
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def _safe_rm(p):
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except Exception:
        pass

def _safe_rm_globs(patterns: list[str]):
    for pat in patterns:
        try:
            for fp in glob.glob(pat):
                _safe_rm(fp)
        except Exception:
            pass

def _purge_darkstar_io(base_folder: str, *, prefix: str | None = None, clear_input=True, clear_output=True):
    """Delete old image-like files from CC DarkStar input/output."""
    try:
        inp = os.path.join(base_folder, "input")
        out = os.path.join(base_folder, "output")
        if clear_input and os.path.isdir(inp):
            for fn in os.listdir(inp):
                fp = os.path.join(inp, fn)
                if os.path.isfile(fp) and (prefix is None or fn.startswith(prefix)):
                    _safe_rm(fp)
        if clear_output and os.path.isdir(out):
            for fn in os.listdir(out):
                fp = os.path.join(out, fn)
                if os.path.isfile(fp) and (prefix is None or fn.startswith(prefix)):
                    _safe_rm(fp)
    except Exception:
        pass


# ------------------------------------------------------------
# Proc runner & dialog (merged stdout/stderr)
# ------------------------------------------------------------
class _ProcThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, command: list[str], cwd: str | None = None, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self.process = None

    def cancel(self):
        """Request the subprocess to stop."""
        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass


    def run(self):
        import subprocess
        import os
        env = os.environ.copy()
        for k in ("PYTHONHOME","PYTHONPATH","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","PYTHONEXECUTABLE"):
            env.pop(k, None)
        rc = -1
        try:
            self.process = subprocess.Popen(
                self.command, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, text=True, start_new_session=True, env=env
            )
            for line in iter(self.process.stdout.readline, ""):
                if not line: break
                self.output_signal.emit(line.rstrip())
            try:
                self.process.stdout.close()
            except Exception:
                pass
            rc = self.process.wait()
        except Exception as e:
            self.output_signal.emit(str(e))
            rc = -1
        finally:
            self.process = None
        self.finished_signal.emit(rc)


class _ProcDialog(QDialog):
    def __init__(self, parent, title="Process"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 420)
        lay = QVBoxLayout(self)
        self.text = QTextEdit(self); self.text.setReadOnly(True)
        lay.addWidget(self.text)
        self.cancel_button = QPushButton("Cancel", self)
        lay.addWidget(self.cancel_button)

    def append_text(self, s: str):
        try:
            self.text.append(s)
        except Exception:
            pass


class DarkStarConfigDialog(QDialog):
    """
    SASv2-style config UI:
      - Disable GPU: Yes/No (default No)
      - Star Removal Mode: unscreen | additive (default unscreen)
      - Show Extracted Stars: Yes/No (default No)
      - Stride (powers of 2): 64,128,256,512,1024 (default 512)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CosmicClarity Dark Star Settings")

        self.chk_disable_gpu = QCheckBox("Disable GPU")
        self.chk_disable_gpu.setChecked(False)  # default No (unchecked)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["unscreen", "additive"])
        self.cmb_mode.setCurrentText("unscreen")

        self.chk_show_stars = QCheckBox("Show Extracted Stars")
        self.chk_show_stars.setChecked(True)

        self.cmb_stride = QComboBox()
        for v in (64, 128, 256, 512, 1024):
            self.cmb_stride.addItem(str(v), v)
        self.cmb_stride.setCurrentText("512")  # default 512

        form = QFormLayout()
        form.addRow("Star Removal Mode:", self.cmb_mode)
        form.addRow("Stride (power of two):", self.cmb_stride)
        form.addRow("", self.chk_disable_gpu)
        form.addRow("", self.chk_show_stars)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(btns)

    def get_values(self):
        return {
            "disable_gpu": self.chk_disable_gpu.isChecked(),
            "mode": self.cmb_mode.currentText(),
            "show_extracted_stars": self.chk_show_stars.isChecked(),
            "stride": int(self.cmb_stride.currentData()),
        }
