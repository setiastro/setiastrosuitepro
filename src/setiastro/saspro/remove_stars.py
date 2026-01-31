# pro/remove_stars.py
from __future__ import annotations
import os
import platform
import shutil
import stat
import tempfile
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog,
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QProgressBar,
    QLabel, QComboBox, QCheckBox, QSpinBox, QFormLayout, QDialogButtonBox, QWidget, QHBoxLayout
)

from setiastro.saspro.cosmicclarity_engines.darkstar_engine import (
    darkstar_starremoval_rgb01,
    DarkStarParams,
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
    h = 1.0  
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
    per-channel MTF parameter estimation, matching
    find_unlinked_midtones_balance_default() / find_unlinked_midtones_balance().

    Works on float32 data assumed in [0,1].
    Returns dict with arrays: {'s': (C,), 'm': (C,), 'h': (C,)}.
    """
    """
    per-channel MTF parameter estimation, matching
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

def _mtf_scalar(x: float, m: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
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

    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return float(y)


def _apply_mtf_unlinked_rgb(img_rgb01: np.ndarray, p: dict) -> np.ndarray:
    """
    Apply per-channel MTF exactly. p from _mtf_params_unlinked.
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)

    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[2]):
        out[..., c] = _mtf_apply(x[..., c], float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    return np.clip(out, 0.0, 1.0)


def _invert_mtf_unlinked_rgb(img_rgb01: np.ndarray, p: dict) -> np.ndarray:
    """
    Exact analytic inverse per channel (uses same s/m/h arrays).
    """
    y = np.asarray(img_rgb01, dtype=np.float32)
    if y.ndim == 2:
        y = np.stack([y]*3, axis=-1)
    elif y.ndim == 3 and y.shape[2] == 1:
        y = np.repeat(y, 3, axis=2)

    out = np.empty_like(y, dtype=np.float32)
    for c in range(y.shape[2]):
        out[..., c] = _mtf_inverse(y[..., c], float(p["s"][c]), float(p["m"][c]), float(p["h"][c]))
    return np.clip(out, 0.0, 1.0)
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
      1) Normalize to [0,1] (preserving overall scale separately)
      2) Compute unlinked MTF params per channel
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
    x_in = np.asarray(arr, dtype=np.float32)

    # If (H,W,1), collapse to (H,W) so mono flows cleanly
    if x_in.ndim == 3 and x_in.shape[2] == 1:
        x_in = x_in[..., 0]

    # sanitize
    x_in = np.nan_to_num(x_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Preserve original numeric scale if users pass >1.0
    xmax = float(np.max(x_in)) if x_in.size else 1.0
    scale_factor = xmax if xmax > 1.01 else 1.0

    xin = (x_in / scale_factor) if scale_factor > 1.0 else x_in
    xin = np.clip(xin, 0.0, 1.0)


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


    starless_lin01 = _invert_mtf_unlinked_rgb(starless_s, mtf_params)

    # Restore original scale if we normalized earlier
    if scale_factor > 1.0:
        starless_lin01 *= scale_factor

    result = np.clip(starless_lin01, 0.0, 1.0).astype(np.float32, copy=False)

    # If the source was mono, return mono
    if was_single and result.ndim == 3:
        result = result.mean(axis=2)

    return result

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

def _run_starnet(main, doc):
    import os
    import platform
    import numpy as np
    import re
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    # --- Resolve StarNet exe, persist in settings
    exe = _get_setting_any(getattr(main, "settings", None),
                           ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        exe_path, _ = QFileDialog.getOpenFileName(
            main, "Select StarNet Executable", "", "Executable Files (*)"
        )
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
        QMessageBox.critical(
            main, "Unsupported OS",
            f"The current operating system '{sysname}' is not supported."
        )
        return

    # --- Ask linearity (SASv2 behavior)
    reply = QMessageBox.question(
        main, "Image Linearity", "Is the current image linear?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )
    is_linear = (reply == QMessageBox.StandardButton.Yes)
    did_stretch = is_linear

    # stash params for replay-last
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

    # --- Ensure float32 and sane values (no forced RGB expansion yet)
    src = np.asarray(doc.image)
    if src.ndim == 3 and src.shape[2] == 1:
        processing_image = src[..., 0]
    else:
        processing_image = src

    processing_image = np.nan_to_num(
        processing_image.astype(np.float32, copy=False),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    # --- Scale normalization if >1.0
    scale_factor = float(np.max(processing_image)) if processing_image.size else 1.0
    if scale_factor > 1.0:
        processing_norm = processing_image / scale_factor
    else:
        processing_norm = processing_image

    # --- Build input/output paths (StarNet folder)
    starnet_dir = os.path.dirname(exe) or os.getcwd()
    input_image_path = os.path.join(starnet_dir, "imagetoremovestars.tif")
    output_image_path = os.path.join(starnet_dir, "starless.tif")


    img_for_starnet = processing_norm
    if is_linear:
        mtf_params = _mtf_params_unlinked(processing_norm, shadows_clipping=-2.8, targetbg=0.25)
        img_for_starnet = _apply_mtf_unlinked_rgb(processing_norm, mtf_params)

        # 🔐 Stash EXACT params for inverse step later
        try:
            setattr(main, "_starnet_stat_meta", {
                "scheme": "stretch_mtf",
                "s": np.asarray(mtf_params["s"], dtype=np.float32),
                "m": np.asarray(mtf_params["m"], dtype=np.float32),
                "h": np.asarray(mtf_params["h"], dtype=np.float32),
                "scale": float(scale_factor),
            })
        except Exception:
            pass
    else:
        # non-linear: do not try to invert any pre-stretch later
        try:
            if hasattr(main, "_starnet_stat_meta"):
                delattr(main, "_starnet_stat_meta")
        except Exception:
            pass

    # --- Write TIFF for StarNet (16-bit)
    try:
        save_image(
            img_for_starnet, input_image_path,
            original_format="tif", bit_depth="16-bit",
            original_header=None, is_mono=False, image_meta=None, file_meta=None
        )
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to write input TIFF:\n{e}")
        return

    # --- Build command
    exe_name = os.path.basename(exe).lower()
    if sysname in ("Windows", "Linux"):
        command = [exe, input_image_path, output_image_path, "256"]
    else:  # macOS
        if "starnet2" in exe_name:
            command = [exe, "--input", input_image_path, "--output", output_image_path]
        else:
            command = [exe, input_image_path, output_image_path]

    # --- Progress dialog + worker
    dlg = _ProcDialog(main, title="StarNet Progress")
    dlg.reset_progress("Starting StarNet…")
    dlg.pbar.setRange(0, 100)
    dlg.pbar.setValue(0)
    dlg.pbar.setFormat("0%")
    dlg.append_text("Launching StarNet...\n")

    thr = _ProcThread(command, cwd=starnet_dir)

    # ---- Output parsing (stages + percent finished + tile count) ----
    _re_pct = re.compile(r"^\s*(\d{1,3})%\s+finished\s*$")
    _re_tiles = re.compile(r"Total number of tiles:\s*(\d+)\s*$")

    tile_total = {"n": 0}
    last_pct = {"v": -1}

    def _stage_from_line(low: str) -> str | None:
        if "reading input image" in low:
            return "Reading input image…"
        if ("bits per sample" in low) or ("samples per pixel" in low) or ("height:" in low) or ("width:" in low):
            return "Inspecting input…"
        if "restoring neural network checkpoint" in low:
            return "Loading model checkpoint…"
        if "created device" in low and "gpu" in low:
            return "Initializing GPU…"
        if "loaded cudnn version" in low:
            return "Initializing cuDNN…"
        if "total number of tiles" in low:
            return "Tiling image…"
        if "% finished" in low:
            return "Processing tiles…"
        if "writing" in low or "saving" in low:
            return "Writing output…"
        return None

    def _is_noise(low: str) -> bool:
        # Keep progress + stage parsing, but optionally suppress spam in the log box.
        # You can loosen/tighten this anytime.
        return (
            (low.startswith("202") and "tensorflow/" in low) or
            ("cpu_feature_guard" in low) or
            ("mlir" in low) or
            ("ptxas" in low) or
            ("bfc_allocator" in low) or
            ("garbage collection" in low)
        )

    def _on_out(line: str):
        low = line.lower()

        # stage updates
        st = _stage_from_line(low)
        if st:
            try:
                dlg.lbl_stage.setText(st)
            except Exception:
                pass

        # tile total
        m = _re_tiles.search(line)
        if m:
            try:
                tile_total["n"] = int(m.group(1))
            except Exception:
                tile_total["n"] = 0

        # percent updates (throttled)
        m = _re_pct.match(line)
        if m:
            try:
                pct = int(m.group(1))
                pct = max(0, min(100, pct))
                if pct != last_pct["v"]:
                    last_pct["v"] = pct
                    if tile_total["n"] > 0:
                        done = int(round(tile_total["n"] * (pct / 100.0)))
                        dlg.set_progress(done, tile_total["n"], "Processing tiles…")
                    else:
                        dlg.set_progress(pct, 100, "Processing…")
            except Exception:
                pass

        # append (optionally suppress TF spam)
        if not _is_noise(low):
            dlg.append_text(line)

    thr.output_signal.connect(_on_out)

    # finished -> apply + cleanup
    def _on_finish(rc: int):
        try:
            # snap to 100% for UX (even if StarNet ended abruptly, it will be overwritten by error handling)
            dlg.set_progress(100, 100, "StarNet finished. Loading output…")
        except Exception:
            pass
        _on_starnet_finished(main, doc, rc, dlg, input_image_path, output_image_path, did_stretch)

    thr.finished_signal.connect(_on_finish)

    # cancel kills subprocess
    dlg.cancel_button.clicked.connect(thr.cancel)

    thr.start()
    dlg.exec()

def _on_starnet_finished(main, doc, return_code, dialog, input_path, output_path, did_stretch):
    import os
    import numpy as np
    from PyQt6.QtWidgets import QMessageBox
    from setiastro.saspro.imageops.stretch import stretch_mono_image  # used for statistical inverse
    try:
        dialog.pbar.setRange(0, 100)
        dialog.set_progress(100, 100, "StarNet finished. Loading output…")
    except Exception:
        pass
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

        meta = getattr(main, "_starnet_stat_meta", None)
        mtf_params_legacy = getattr(main, "_starnet_last_mtf_params", None)

        if isinstance(meta, dict) and meta.get("scheme") == "stretch_mtf":
            dialog.append_text("Unstretching (MTF pseudoinverse)...\n")
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
                dialog.append_text(f"⚠️ MTF inverse failed: {e}\n")

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
    import numpy as np
    from PyQt6.QtWidgets import QMessageBox

    # --- Config dialog (keep as-is) ---
    cfg = DarkStarConfigDialog(main)
    if not cfg.exec():
        return
    v = cfg.get_values()

    disable_gpu = bool(v["disable_gpu"])
    mode = str(v["mode"])                         # "unscreen" | "additive"
    show_extracted_stars = bool(v["show_extracted_stars"])
    stride = int(v["stride"])                     # chunk size

    # 🔹 Stash parameters for replay-last (same structure as you had)
    try:
        main._last_remove_stars_params = {
            "engine": "DarkStar",
            "disable_gpu": bool(disable_gpu),
            "mode": mode,
            "show_extracted_stars": bool(show_extracted_stars),
            "stride": int(stride),
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
            },
        }
        if hasattr(main, "_log"):
            main._log(
                "[Replay] Recorded remove_stars (DarkStar, "
                f"mode={mode}, chunk={int(stride)}, "
                f"gpu={'off' if disable_gpu else 'on'}, "
                f"stars={'on' if show_extracted_stars else 'off'})"
            )
    except Exception:
        pass

    # --- Build input image for engine: float32 [0..1], HxWx3/1/mono ok ---
    src = np.asarray(doc.image)
    orig_was_mono = (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1)

    x = np.nan_to_num(src.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    # If your doc domain can exceed 1.0, normalize to [0..1] for the engine.
    # (Matches what you were already doing for external DarkStar.)
    scale_factor = float(np.max(x)) if x.size else 1.0
    if scale_factor > 1.0:
        x = x / scale_factor
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    params = DarkStarParams(
        use_gpu=(not disable_gpu),
        chunk_size=int(stride),
        overlap_frac=0.125,
        mode=mode,
        output_stars_only=show_extracted_stars,
    )

    dlg = _ProcDialog(main, title="Dark Star Progress")
    dlg.append_text("Starting Dark Star (engine)…\n")

    thr = _DarkStarThread(x, params, parent=dlg)

    def _on_prog(done, total, stage):
        dlg.set_progress(done, total, stage)

    def _on_done(starless, stars_only, was_mono, err):
        if err:
            QMessageBox.critical(main, "Dark Star Error", f"Dark Star failed:\n{err}")
            dlg.close()
            return

        # Engine returns float32 [0..1]; if we normalized >1, restore scale if you want.
        # BUT you later clip to [0..1] for doc anyway, so this is mostly for consistency.
        if scale_factor > 1.0:
            try:
                starless = starless * scale_factor
                if stars_only is not None:
                    stars_only = stars_only * scale_factor
            except Exception:
                pass

        # Original as RGB for blending math
        orig = np.asarray(doc.image).astype(np.float32, copy=False)
        if orig.ndim == 2:
            original_rgb = np.stack([orig]*3, axis=-1)
        elif orig.ndim == 3 and orig.shape[2] == 1:
            original_rgb = np.repeat(orig, 3, axis=2)
        else:
            original_rgb = orig

        # Starless as RGB for blending math
        if starless.ndim == 2:
            starless_rgb = np.stack([starless]*3, axis=-1)
        elif starless.ndim == 3 and starless.shape[2] == 1:
            starless_rgb = np.repeat(starless, 3, axis=2)
        else:
            starless_rgb = starless

        starless_rgb = starless_rgb.astype(np.float32, copy=False)

        # ---- Optional stars-only push ----
        if show_extracted_stars:
            if stars_only is None:
                # Safety fallback if someone changes engine params later
                stars_only_rgb = np.clip(original_rgb - starless_rgb, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                if stars_only.ndim == 2:
                    stars_only_rgb = np.stack([stars_only]*3, axis=-1)
                elif stars_only.ndim == 3 and stars_only.shape[2] == 1:
                    stars_only_rgb = np.repeat(stars_only, 3, axis=2)
                else:
                    stars_only_rgb = stars_only.astype(np.float32, copy=False)

            m3 = _active_mask3_from_doc(doc, stars_only_rgb.shape[1], stars_only_rgb.shape[0])
            if m3 is not None:
                stars_only_rgb *= m3
                dlg.append_text("✅ Applied active mask to stars-only image.\n")

            if orig_was_mono:
                stars_to_push = stars_only_rgb.mean(axis=2).astype(np.float32, copy=False)
            else:
                stars_to_push = stars_only_rgb

            _push_as_new_doc(main, doc, stars_to_push, title_suffix="_stars", source="Stars-Only (DarkStar)")
            dlg.append_text("Stars-only image pushed.\n")

        # ---- Mask-blend starless into current doc ----
        final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)

        if orig_was_mono:
            final_to_apply = final_starless.mean(axis=2).astype(np.float32, copy=False)
        else:
            final_to_apply = final_starless.astype(np.float32, copy=False)

        # Clip to [0..1] because that’s what your pipeline expects almost everywhere
        final_to_apply = np.clip(final_to_apply, 0.0, 1.0).astype(np.float32, copy=False)

        try:
            meta = {
                "step_name": "Stars Removed",
                "bit_depth": "32-bit floating point",
                "is_mono": bool(orig_was_mono),
            }

            rp = getattr(main, "_last_remove_stars_params", None)
            replay_params = dict(rp) if isinstance(rp, dict) else {"engine": "DarkStar", "label": "Remove Stars (DarkStar)"}
            replay_params.setdefault("engine", "DarkStar")
            replay_params.setdefault("label", "Remove Stars (DarkStar)")

            meta["replay_last"] = {"op": "remove_stars", "params": replay_params}

            try:
                if hasattr(main, "_last_remove_stars_params"):
                    delattr(main, "_last_remove_stars_params")
            except Exception:
                pass

            doc.apply_edit(final_to_apply, metadata=meta, step_name="Stars Removed")
            if hasattr(main, "_log"):
                main._log("Stars Removed (DarkStar)")
        except Exception as e:
            QMessageBox.critical(main, "Dark Star Error", f"Failed to apply result:\n{e}")

        dlg.append_text("Done.\n")
        dlg.close()

    thr.progress_signal.connect(_on_prog)
    thr.finished_signal.connect(_on_done)

    dlg.cancel_button.clicked.connect(lambda: dlg.append_text("Cancel not supported for in-process engine.\n"))
    dlg.show()
    thr.start()
    dlg.exec()

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
        self.setMinimumSize(600, 460)

        self._last_pct = -1  # for throttling UI updates

        lay = QVBoxLayout(self)

        # --- status line + progress bar ---
        self.lbl_stage = QLabel("", self)
        self.lbl_stage.setWordWrap(True)
        self.lbl_stage.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.lbl_stage)

        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        lay.addWidget(self.pbar)

        # --- log output ---
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)

        # --- cancel ---
        self.cancel_button = QPushButton("Cancel", self)
        lay.addWidget(self.cancel_button)

    def append_text(self, s: str):
        try:
            self.text.append(s)
        except Exception:
            pass

    def set_progress(self, done: int, total: int, stage: str = ""):
        """
        Update stage label + progress bar.
        `total<=0` puts the bar into an indeterminate-ish state (0%).
        Throttles updates when percent hasn't changed.
        """
        try:
            if stage:
                self.lbl_stage.setText(stage)

            if total and total > 0:
                pct = int(round(100.0 * float(done) / float(total)))
                pct = max(0, min(100, pct))
            else:
                pct = 0

            # throttle: only repaint if percent changed or we're at end
            if pct != self._last_pct or done == total:
                self._last_pct = pct
                self.pbar.setValue(pct)

                # keep the text helpful
                if total and total > 0:
                    self.pbar.setFormat(f"{pct}%  ({done}/{total})")
                else:
                    self.pbar.setFormat(f"{pct}%")
        except Exception:
            pass

    def reset_progress(self, stage: str = ""):
        self._last_pct = -1
        if stage:
            try:
                self.lbl_stage.setText(stage)
            except Exception:
                pass
        try:
            self.pbar.setValue(0)
            self.pbar.setFormat("0%")
        except Exception:
            pass

class _DarkStarThread(QThread):
    progress_signal = pyqtSignal(int, int, str)   # done, total, stage
    finished_signal = pyqtSignal(object, object, bool, str)  # starless, stars_only, was_mono, errstr

    def __init__(self, img_rgb01: np.ndarray, params: DarkStarParams, parent=None):
        super().__init__(parent)
        self._img = img_rgb01
        self._params = params

    def run(self):
        try:
            def prog(done, total, stage):
                self.progress_signal.emit(int(done), int(total), str(stage))

            # status_cb is optional; keep quiet or route to progress text if you want
            starless, stars_only, was_mono = darkstar_starremoval_rgb01(
                self._img,
                params=self._params,
                progress_cb=prog,
                status_cb=lambda s: None,
            )
            self.finished_signal.emit(starless, stars_only, bool(was_mono), "")
        except Exception as e:
            self.finished_signal.emit(None, None, False, str(e))


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


def darkstar_starless_from_array(
    arr_rgb01: np.ndarray,
    *,
    use_gpu: bool = True,
    chunk_size: int = 512,
    overlap_frac: float = 0.125,
    mode: str = "unscreen",              # "unscreen" | "additive"
    output_stars_only: bool = False,
    progress_cb=None,                    # (done:int, total:int, stage:str) -> None
    status_cb=None                       # (msg:str) -> None
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """
    Headless DarkStar:
      input: float32 [0..1], mono or RGB
      output: (starless, stars_only_or_None, was_mono)
    """
    x = np.asarray(arr_rgb01, dtype=np.float32)

    was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    # Normalize shape to what engine expects (it can accept mono or rgb, but keep it consistent)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[..., 0]  # collapse to (H,W) for mono

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x = np.clip(x, 0.0, 1.0)

    params = DarkStarParams(
        use_gpu=bool(use_gpu),
        chunk_size=int(chunk_size),
        overlap_frac=float(overlap_frac),
        mode=str(mode),
        output_stars_only=bool(output_stars_only),
    )

    def _prog(done, total, stage):
        if callable(progress_cb):
            try:
                progress_cb(int(done), int(total), str(stage))
            except Exception:
                pass

    def _status(msg: str):
        if callable(status_cb):
            try:
                status_cb(str(msg))
            except Exception:
                pass

    starless, stars_only, engine_was_mono = darkstar_starremoval_rgb01(
        x,
        params=params,
        progress_cb=_prog if callable(progress_cb) else None,
        status_cb=_status if callable(status_cb) else None,
    )

    # Engine should return [0..1] float32, but enforce it anyway
    if starless is not None:
        starless = np.clip(np.asarray(starless, dtype=np.float32), 0.0, 1.0)

    if stars_only is not None:
        stars_only = np.clip(np.asarray(stars_only, dtype=np.float32), 0.0, 1.0)

    # If input was mono, guarantee mono out (some paths may hand back rgb)
    if was_mono and starless is not None and starless.ndim == 3:
        starless = starless.mean(axis=2).astype(np.float32, copy=False)

    if was_mono and stars_only is not None and stars_only.ndim == 3:
        stars_only = stars_only.mean(axis=2).astype(np.float32, copy=False)

    return starless, (stars_only if output_stars_only else None), bool(was_mono)
