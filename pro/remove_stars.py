# pro/remove_stars.py
from __future__ import annotations
import os, platform, shutil, stat, tempfile
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QInputDialog, QMessageBox, QFileDialog,
    QDialog, QVBoxLayout, QTextEdit, QPushButton,
    QLabel, QComboBox, QCheckBox, QSpinBox, QFormLayout, QDialogButtonBox, QWidget, QHBoxLayout
)

# use your legacy I/O functions (as requested)
from legacy.image_manager import save_image, load_image
import glob
try:
    import cv2
except Exception:
    cv2 = None

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
    # Inverse via property: mtf^{-1}(·, m) = mtf(·, 1-m)
    s, m, h = float(shadows), float(midtones), float(highlights)
    yp = np.clip(y.astype(np.float32, copy=False), 0.0, 1.0)
    # apply with (1 - m)
    m_inv = np.clip(1.0 - m, 1e-4, 1.0 - 1e-4)
    num = (m_inv - 1.0) * yp
    den = ((2.0 * m_inv - 1.0) * yp) - m_inv
    xp = np.divide(num, den, out=np.zeros_like(yp, dtype=np.float32), where=np.abs(den) > 1e-12)
    xp = np.clip(xp, 0.0, 1.0)
    return (xp * (h - s) + s).astype(np.float32, copy=False)

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
                         shadowclip_sigma: float = -2.8,
                         targetbg: float = 0.25):
    """
    Compute per-channel MTF parameters for RGB image in [0..1].
    Returns dict with arrays: {'s': (3,), 'm': (3,), 'h': (3,)}
    Exact inverse possible with _invert_mtf_unlinked_rgb using same params.
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)

    C = x.shape[2]
    s = np.zeros(C, dtype=np.float32)
    m = np.zeros(C, dtype=np.float32)
    h = np.ones (C, dtype=np.float32)  # we normalize to ≤1 before 16-bit write

    # robust peak/sigma per channel
    for c in range(C):
        ch = x[..., c]
        # use your robust peak/sigma if available; fallback to median/MAD
        try:
            peak_c, sigma_c = _robust_peak_sigma(ch)
        except NameError:
            med = float(np.median(ch))
            mad = float(np.median(np.abs(ch - med))) + 1e-12
            peak_c  = med
            sigma_c = 1.4826 * mad

        # compute shadows s_c and clamp to [min, max-eps]
        s_c = float(peak_c + shadowclip_sigma * sigma_c)
        xmin = float(np.min(ch)) if ch.size else 0.0
        xmax = float(np.max(ch)) if ch.size else 1.0
        eps  = 1e-6
        s_c  = float(np.clip(s_c, xmin, max(xmax - eps, xmin)))

        # solve midtones m so that mtf maps peak→targetbg
        denom = max(1e-12, (h[c] - s_c))
        xnorm = (peak_c - s_c) / denom
        xnorm = float(np.clip(xnorm, 1e-6, 1.0 - 1e-6))
        y     = float(np.clip(targetbg, 1e-6, 1.0 - 1e-6))
        d     = (2.0 * y * xnorm) - y - xnorm
        m_c   = (xnorm * (y - 1.0)) / d if abs(d) > 1e-12 else 0.5
        m_c   = float(np.clip(m_c, 1e-4, 1.0 - 1e-4))

        s[c], m[c] = s_c, m_c

    return {"s": s, "m": m, "h": h}


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

def _stat_stretch_rgb(img: np.ndarray,
                      lo_pct: float = 0.25,
                      hi_pct: float = 99.75) -> tuple[np.ndarray, dict]:
    """
    Make sure img is RGB float32 in [0,1], stretch each channel to [0,1]
    using percentiles. Returns (stretched_img, params) where params can be
    fed to _stat_unstretch_rgb() to invert exactly.
    """
    was_single = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
    if was_single:
        img = np.stack([img] * 3, axis=-1)

    x = img.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    lo_vals, hi_vals = [], []

    for c in range(3):
        ch = x[..., c]
        lo = float(np.percentile(ch, lo_pct))
        hi = float(np.percentile(ch, hi_pct))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        lo_vals.append(lo); hi_vals.append(hi)
        out[..., c] = (ch - lo) / (hi - lo)

    out = np.clip(out, 0.0, 1.0)
    params = {"lo": lo_vals, "hi": hi_vals, "was_single": was_single}
    return out, params


def _stat_unstretch_rgb(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Inverse of _stat_stretch_rgb. Expects img RGB float32 [0,1].
    """
    lo = np.asarray(params["lo"], dtype=np.float32)
    hi = np.asarray(params["hi"], dtype=np.float32)
    out = img.astype(np.float32, copy=False).copy()
    for c in range(3):
        out[..., c] = out[..., c] * (hi[c] - lo[c]) + lo[c]
    out = np.clip(out, 0.0, 1.0)
    if params.get("was_single", False):
        out = out.mean(axis=2, keepdims=False)  # back to single channel if needed
        # StarNet needs RGB during processing; we keep RGB after removal for consistency.
        # If you want to return mono to the doc when the source was mono, do it at the very end.
        out = np.stack([out] * 3, axis=-1)
    return out


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
    Statistical-Stretch round-trip for 32-bit data (pedestal-safe):
      1) Record per-channel first-nonzero blackpoints (BP[c]); subtract them
      2) Record per-channel medians M0[c]
      3) Unlinked statistical stretch to target 0.25 (per-channel) -> write 16-bit TIFF
      4) StarNet -> read starless TIFF
      5) Per-channel statistical stretch back to each original median M0[c]
      6) Add back BP[c]
    Returns starless RGB float32 in [0..1] (or scaled back if you fed >1).
    """
    import os, platform, subprocess
    import numpy as np
    from imageops.stretch import stretch_color_image, stretch_mono_image
    # save_image / load_image / _get_setting_any / _safe_rm assumed available in this module
    arr = np.asarray(arr_rgb01, dtype=np.float32)
    was_single = (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)
    exe = _get_setting_any(settings, ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("StarNet executable not configured (settings 'paths/starnet').")

    workdir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(workdir, f"{tmp_prefix}_in.tif")
    out_path = os.path.join(workdir, f"{tmp_prefix}_out.tif")

    # --- Normalize input shape and safe values
    x = arr
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Preserve original numeric scale if users pass >1.0
    xmax = float(np.max(x)) if x.size else 1.0
    scale_factor = xmax if xmax > 1.01 else 1.0
    xin = (x / scale_factor) if scale_factor > 1.0 else x
    xin = np.clip(xin, 0.0, 1.0)

    H, W, _ = xin.shape

    # --- Helper: per-channel first-nonzero (minimum positive) blackpoint
    def _first_nonzero_bp_per_channel(img3):
        bps = np.zeros(3, dtype=np.float32)
        for c in range(3):
            ch = img3[..., c].reshape(-1)
            pos = ch[ch > 0.0]
            bps[c] = float(pos.min()) if pos.size else 0.0
        return bps

    # 1) Per-channel "first non-zero" blackpoints, subtract pedestals
    bp = _first_nonzero_bp_per_channel(xin)
    xin_ped = xin - bp.reshape((1, 1, 3))
    xin_ped = np.clip(xin_ped, 0.0, 1.0)

    # 2) Record per-channel medians after pedestal removal
    m0 = np.array([float(np.median(xin_ped[..., c])) for c in range(3)], dtype=np.float32)

    # 3) Unlinked statistical stretch -> target 0.25
    #    (uses your SAS math internally, per-channel)
    target_bg = 0.25
    x_for_starnet = stretch_color_image(
        xin_ped, target_median=target_bg,
        linked=False, normalize=False, apply_curves=False, curves_boost=0.0
    ).astype(np.float32, copy=False)

    # Write 16-bit TIFF for StarNet
    save_image(
        x_for_starnet, in_path,
        original_format="tif", bit_depth="16-bit",
        original_header=None, is_mono=False, image_meta=None, file_meta=None
    )

    # 4) Run StarNet
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
        starless_s = np.stack([starless_s]*3, axis=-1)
    elif starless_s.ndim == 3 and starless_s.shape[2] == 1:
        starless_s = np.repeat(starless_s, 3, axis=2)
    starless_s = np.clip(starless_s.astype(np.float32, copy=False), 0.0, 1.0)

    # 5) Per-channel statistical stretch back to original per-channel medians (M0)
    #    We use your mono stretch per channel so targets can differ by channel.
    def _per_channel_to_targets(img3, targets_rgb):
        out = np.empty_like(img3, dtype=np.float32)
        for c in range(3):
            ch = img3[..., c]
            out[..., c] = stretch_mono_image(
                ch, target_median=float(targets_rgb[c]),
                normalize=False, apply_curves=False, curves_boost=0.0
            )
        return out

    starless_back = _per_channel_to_targets(starless_s, m0)

    # 6) Add back the original pedestals
    starless_lin = starless_back + bp.reshape((1, 1, 3))

    # Restore original scale if we normalized earlier
    if scale_factor > 1.0:
        starless_lin *= scale_factor

    # Match previous function’s contract (float32, clipped to [0..1])
    result = np.clip(starless_lin, 0.0, 1.0).astype(np.float32, copy=False)

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
    from imageops.stretch import stretch_color_image

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
    from imageops.stretch import stretch_mono_image

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
    import os, platform, numpy as np
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
    # --- Ensure RGB float32 in safe range
    src = np.asarray(doc.image)
    if src.ndim == 2:
        processing_image = np.stack([src]*3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1:
        processing_image = np.repeat(src, 3, axis=2)
    else:
        processing_image = src
    processing_image = np.nan_to_num(processing_image.astype(np.float32, copy=False),
                                     nan=0.0, posinf=0.0, neginf=0.0)

    # --- Scale normalization if >1.0 (same reason as before: 16-bit export safety)
    scale_factor = float(np.max(processing_image))
    if scale_factor > 1.0:
        processing_norm = processing_image / scale_factor
    else:
        processing_norm = processing_image

    # --- Build input/output paths
    starnet_dir = os.path.dirname(exe) or os.getcwd()
    input_image_path  = os.path.join(starnet_dir, "imagetoremovestars.tif")
    output_image_path = os.path.join(starnet_dir, "starless.tif")

    # --- Prepare input for StarNet
    meta = None
    img_for_starnet = processing_norm
    if is_linear:
        # Statistical round-trip preparation (records pedestals+medians)
        img_for_starnet, meta = _prepare_statstretch_input_for_starnet(processing_norm)
        # 🔐 Stash the exact meta for inverse step later
        setattr(main, "_starnet_stat_meta", {
            "bp": meta["bp"],         # exact per-channel first-nonzero BP used
            "m0": meta["m0"],         # exact per-channel medians after pedestal removal
            "scale": scale_factor,    # pre-export normalization scale
        })
    else:
        # make sure no stale meta lingers
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
    dlg.cancel_button.clicked.connect(thr.terminate)

    dlg.show()
    thr.start()
    dlg.exec()


def _on_starnet_finished(main, doc, return_code, dialog, input_path, output_path, did_stretch):
    import os, numpy as np
    from PyQt6.QtWidgets import QMessageBox
    from imageops.stretch import stretch_mono_image  # used for statistical inverse

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
        mtf_params = getattr(main, "_starnet_last_mtf_params", None)

        if mtf_params:
            # Back-compat: old MTF path
            dialog.append_text("Unstretching (MTF inverse)...\n")
            try:
                starless_rgb = _invert_mtf_linked_rgb(starless_rgb, mtf_params)
                sc = float(mtf_params.get("scale", 1.0))
                if sc > 1.0:
                    starless_rgb = starless_rgb * sc
            except Exception as e:
                dialog.append_text(f"⚠️ MTF inverse failed: {e}\n")
            starless_rgb = np.clip(starless_rgb, 0.0, 1.0)

        else:
            # ✅ Statistical round-trip inverse using the EXACT meta captured during prep
            dialog.append_text("Unstretching (statistical inverse w/ original BP/M0)...\n")

            meta = getattr(main, "_starnet_stat_meta", None)
            if meta is None:
                # Fallback (shouldn't happen, but be safe): recompute from original
                dialog.append_text("⚠️ Missing stat meta; recomputing BP/M0 from original.\n")
                scale_factor = float(np.max(original_rgb)) if original_rgb.size else 1.0
                orig_norm = np.clip(original_rgb / scale_factor, 0.0, 1.0) if scale_factor > 1.0 else np.clip(original_rgb, 0.0, 1.0)
                bp_vec = _first_nonzero_bp_per_channel(orig_norm)
                m0_vec = np.array(
                    [float(np.median(np.clip(orig_norm[..., c] - bp_vec[c], 0.0, 1.0))) for c in range(3)],
                    dtype=np.float32
                )
            else:
                bp_vec = np.asarray(meta.get("bp"), dtype=np.float32)
                m0_vec = np.asarray(meta.get("m0"), dtype=np.float32)
                scale_factor = float(meta.get("scale", 1.0))

            # Per-channel inverse: stretch back to each original median m0[c]
            inv = np.empty_like(starless_rgb, dtype=np.float32)
            for c in range(3):
                inv[..., c] = stretch_mono_image(
                    starless_rgb[..., c],
                    target_median=float(m0_vec[c]),
                    normalize=False, apply_curves=False, curves_boost=0.0
                )

            # 🔁 Add back the EXACT original pedestals
            inv += bp_vec.reshape((1, 1, 3))
            inv = np.clip(inv, 0.0, 1.0)

            # Restore original numeric scale, if any
            if scale_factor > 1.0:
                inv = inv * scale_factor

            starless_rgb = np.clip(inv, 0.0, 1.0)

            # Clean up stashed meta so it can't leak to the next run
            if hasattr(main, "_starnet_stat_meta"):
                delattr(main, "_starnet_stat_meta")


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

    # Input/output folders per SASv2
    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _purge_darkstar_io(base, prefix=None, clear_input=True, clear_output=True)   
    # Save the current image as 32-bit float TIFF (no stretch)
    in_path = os.path.join(input_dir, "imagetoremovestars.tif")
 
    try:
        save_image(doc.image, in_path, original_format="tif",
                   bit_depth="32-bit floating point",
                   original_header=None, is_mono=False, image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "Cosmic Clarity", f"Failed to write input TIFF:\n{e}")
        return

    # Show SASv2-style config dialog
    cfg = DarkStarConfigDialog(main)
    if not cfg.exec():
        _safe_rm(in_path)
        return
    params = cfg.get_values()
    disable_gpu = params["disable_gpu"]
    mode = params["mode"]                         # "unscreen" or "additive"
    show_extracted_stars = params["show_extracted_stars"]
    stride = params["stride"]                     # 64..1024, default 512
    # 🔹 Stash parameters for replay-last
    try:
        main._last_remove_stars_params = {
            "engine": "CosmicClarityDarkStar",
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
                f"mode={mode}, stride={int(stride)}, "
                f"gpu={'off' if disable_gpu else 'on'}, "
                f"stars={'on' if show_extracted_stars else 'off'})"
            )
    except Exception:
        pass    
    # Build CLI exactly like SASv2 (using --chunk_size, not chunk_size)
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
        lambda rc, base=base: _on_darkstar_finished(main, doc, rc, dlg, in_path, output_dir, base)
            )
    dlg.cancel_button.clicked.connect(thr.terminate)

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


def _on_darkstar_finished(main, doc, return_code, dialog, in_path, output_dir, base_folder):
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


    # stars-only optional push
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

    # mask-blend starless → overwrite current doc
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


    # cleanup
    try:
        # Remove known outputs
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
def _active_mask_array_from_doc(doc) -> np.ndarray | None:
    """Return active mask (H,W) float32 in [0,1] from the document, if present."""
    try:
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 3:
            if cv2 is not None:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            else:
                a = a.mean(axis=2)
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return a
    except Exception:
        return None


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

    def run(self):
        import subprocess, os
        env = os.environ.copy()
        for k in ("PYTHONHOME","PYTHONPATH","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","PYTHONEXECUTABLE"):
            env.pop(k, None)
        rc = -1
        try:
            p = subprocess.Popen(
                self.command, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, text=True, start_new_session=True, env=env
            )
            for line in iter(p.stdout.readline, ""):
                if not line: break
                self.output_signal.emit(line.rstrip())
            try:
                p.stdout.close()
            except Exception:
                pass
            rc = p.wait()
        except Exception as e:
            self.output_signal.emit(str(e))
            rc = -1
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
