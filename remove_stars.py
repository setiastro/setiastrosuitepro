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
    Always: MTF pre-stretch -> StarNet -> inverse MTF -> rescale -> starless RGB [0..1].
    """
    exe = _get_setting_any(settings, ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("StarNet executable not configured (settings 'paths/starnet').")

    workdir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(workdir, f"{tmp_prefix}_in.tif")
    out_path = os.path.join(workdir, f"{tmp_prefix}_out.tif")

    # ensure RGB float32
    x = arr_rgb01.astype(np.float32, copy=False)
    if x.ndim == 2: x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1: x = np.repeat(x, 3, axis=2)

    # --- ALWAYS do deterministic MTF stretch like SASv2 ---
    scale_factor = float(np.max(x)) if np.isfinite(np.max(x)) else 1.0
    x_norm = x / scale_factor if scale_factor > 1.0 else x
    mtf = _mtf_params_linked(x_norm, shadowclip_sigma=-2.8, targetbg=0.25)
    x_stretched = _apply_mtf_linked_rgb(x_norm, mtf)

    # StarNet expects 16-bit TIFF in its folder
    save_image(x_stretched, in_path, original_format="tif", bit_depth="16-bit",
               original_header=None, is_mono=False, image_meta=None, file_meta=None)

    # run StarNet
    import subprocess, platform
    exe_name = os.path.basename(exe).lower()
    if platform.system() in ("Windows", "Linux"):
        cmd = [exe, in_path, out_path, "256"]
    else:  # macOS
        cmd = [exe, "--input", in_path, "--output", out_path] if "starnet2" in exe_name else [exe, in_path, out_path]

    rc = subprocess.call(cmd, cwd=workdir)
    if rc != 0 or not os.path.exists(out_path):
        _safe_rm(in_path); _safe_rm(out_path)
        raise RuntimeError(f"StarNet failed rc={rc}")

    starless, _, _, _ = load_image(out_path)
    _safe_rm(in_path); _safe_rm(out_path)

    if starless.ndim == 2: starless = np.stack([starless]*3, axis=-1)
    starless = starless.astype(np.float32, copy=False)

    # --- ALWAYS inverse the MTF & restore scale ---
    starless = _invert_mtf_linked_rgb(starless, mtf)
    if scale_factor > 1.0:
        starless *= scale_factor

    return np.clip(starless, 0.0, 1.0)



def darkstar_starless_from_array(arr_rgb01: np.ndarray, settings, *, tmp_prefix="comet",
                                 disable_gpu=False, mode="unscreen", stride=512) -> np.ndarray:
    """
    Save arr -> run DarkStar -> load starless -> return starless RGB float32 [0..1].
    """
    exe, base = _resolve_darkstar_exe(type("dummy", (), {"settings": settings}) )
    if not exe or not base:
        raise RuntimeError("Cosmic Clarity DarkStar executable not configured.")

    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    _purge_darkstar_io(base, prefix=None, clear_input=True, clear_output=True)

    in_path = os.path.join(input_dir, f"{tmp_prefix}_in.tif")
    save_image(arr_rgb01, in_path, original_format="tif", bit_depth="32-bit floating point",
               original_header=None, is_mono=False, image_meta=None, file_meta=None)

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
    if starless.ndim == 2: starless = np.stack([starless]*3, axis=-1)
    starless = starless.astype(np.float32, copy=False)

    # cleanup typical outputs
    _purge_darkstar_io(base, prefix="imagetoremovestars", clear_input=True, clear_output=True)
    return np.clip(starless, 0.0, 1.0)


# ------------------------------------------------------------
# Public entry
# ------------------------------------------------------------
def remove_stars(main):
    # block interactive UI during/just-after a headless preset run
    if getattr(main, "_remove_stars_headless_running", False):
        return
    if getattr(main, "_remove_stars_guard", False):
        return    
    """Choose StarNet or CosmicClarityDarkStar, process active doc, update starless in-place, open stars-only as new doc."""
    tool, ok = QInputDialog.getItem(
        main, "Select Star Removal Tool", "Choose a tool:",
        ["StarNet", "CosmicClarityDarkStar"], 0, False
    )
    if not ok:
        return

    # active doc
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


# ------------------------------------------------------------
# StarNet (SASv2-like: 16-bit TIFF in StarNet folder)
# ------------------------------------------------------------
def _run_starnet(main, doc):
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

    # SASv2: ask linearity
    reply = QMessageBox.question(
        main, "Image Linearity", "Is the current image linear?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )
    is_linear = (reply == QMessageBox.StandardButton.Yes)

    # ensure RGB for StarNet
    src = np.asarray(doc.image)
    if src.ndim == 2:
        processing_image = np.stack([src] * 3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 1:
        processing_image = np.repeat(src, 3, axis=2)
    else:
        processing_image = src
    processing_image = processing_image.astype(np.float32, copy=False)

    # optional stretch (SASv2)
    did_stretch = False
    if is_linear:
        did_stretch = True
        # normalize if >1.0 (Siril rescales to avoid clipping on 16-bit TIFF)
        scale_factor = float(np.max(processing_image))
        if scale_factor > 1.0:
            processing_norm = processing_image / scale_factor
        else:
            processing_norm = processing_image

        mtf_params = _mtf_params_linked(processing_norm, shadowclip_sigma=-2.8, targetbg=0.25)
        processing_image = _apply_mtf_linked_rgb(processing_norm, mtf_params)

        # stash params for inverse step
        setattr(main, "_starnet_last_mtf_params", {
            "s": mtf_params["s"], "m": mtf_params["m"], "h": mtf_params["h"],
            "scale": scale_factor
        })
    else:
        if hasattr(main, "_starnet_last_mtf_params"):
            delattr(main, "_starnet_last_mtf_params")

    # write input/output paths in StarNet folder
    starnet_dir = os.path.dirname(exe) or os.getcwd()
    input_image_path = os.path.join(starnet_dir, "imagetoremovestars.tif")
    output_image_path = os.path.join(starnet_dir, "starless.tif")

    try:
        # StarNet requires 16-bit TIFF
        save_image(processing_image, input_image_path,
                   original_format="tif", bit_depth="16-bit",
                   original_header=None, is_mono=False,
                   image_meta=None, file_meta=None)
    except Exception as e:
        QMessageBox.critical(main, "StarNet", f"Failed to write input TIFF:\n{e}")
        return

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
    thr.finished_signal.connect(
        lambda rc: _on_starnet_finished(main, doc, rc, dlg, input_image_path, output_image_path, did_stretch)
    )
    dlg.cancel_button.clicked.connect(thr.terminate)

    dlg.show()
    thr.start()
    dlg.exec()


def _on_starnet_finished(main, doc, return_code, dialog, input_path, output_path, did_stretch):
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
    if starless_rgb is None:
        QMessageBox.critical(main, "StarNet Error", "Failed to load starless image.")
        _safe_rm(input_path); _safe_rm(output_path)
        dialog.close()
        return

    # ensure 3ch
    if starless_rgb.ndim == 2 or (starless_rgb.ndim == 3 and starless_rgb.shape[2] == 1):
        starless_rgb = np.stack([starless_rgb] * 3, axis=-1)
    starless_rgb = starless_rgb.astype(np.float32, copy=False)

    # unstretch (if we stretched)
    if did_stretch:
        params = getattr(main, "_starnet_last_mtf_params", None)
        if params:
            starless_rgb = _invert_mtf_linked_rgb(starless_rgb, params)
            if float(params.get("scale", 1.0)) > 1.0:
                starless_rgb = starless_rgb * float(params["scale"])
            # keep numeric sanity
            starless_rgb = np.clip(starless_rgb, 0.0, 1.0)
            
    # original image (from the doc)
    orig = np.asarray(doc.image)
    if orig.ndim == 2:
        original_rgb = np.stack([orig] * 3, axis=-1)
    elif orig.ndim == 3 and orig.shape[2] == 1:
        original_rgb = np.repeat(orig, 3, axis=2)
    else:
        original_rgb = orig
    original_rgb = original_rgb.astype(np.float32, copy=False)

    # Stars-Only (SASv2 formula)
    dialog.append_text("Generating stars-only image...\n")
    with np.errstate(divide='ignore', invalid='ignore'):
        stars_only = (original_rgb - starless_rgb) / np.clip(1.0 - starless_rgb, 1e-6, None)
        stars_only = np.nan_to_num(stars_only, nan=0.0, posinf=0.0, neginf=0.0)
    stars_only = np.clip(stars_only, 0.0, 1.0)

    # apply active mask (doc-based)
    m3 = _active_mask3_from_doc(doc, stars_only.shape[1], stars_only.shape[0])
    if m3 is not None:
        stars_only *= m3
        dialog.append_text("✅ Applied active mask to the stars-only image.\n")
    else:
        dialog.append_text("ℹ️ No active mask for stars-only; skipping.\n")

    # push Stars-Only as new document with suffix _stars
    _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (StarNet)")
    dialog.append_text("Stars-only image pushed.\n")

    # mask-blend starless with original using active mask
    dialog.append_text("Preparing to update current view with starless (mask-blend)...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)

    # overwrite the current doc view
    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": False,
        }
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
        if hasattr(main, "_log"):
            main._log("Stars Removed (StarNet)")
    except Exception as e:
        QMessageBox.critical(main, "StarNet Error", f"Failed to apply starless result:\n{e}")

    _safe_rm(input_path); _safe_rm(output_path)
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
    elif src.ndim == 3 and src.shape[2] == 1:
        original_rgb = np.repeat(src, 3, axis=2)
    else:
        original_rgb = src
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
            _push_as_new_doc(main, doc, stars_only, title_suffix="_stars", source="Stars-Only (DarkStar)")
        else:
            dialog.append_text("Failed to load stars-only image.\n")
    else:
        dialog.append_text("No stars-only image generated.\n")

    # mask-blend starless → overwrite current doc
    dialog.append_text("Mask-blending starless image before update...\n")
    final_starless = _mask_blend_with_doc_mask(doc, starless_rgb, original_rgb)
    try:
        meta = {
            "step_name": "Stars Removed",
            "bit_depth": "32-bit floating point",
            "is_mono": False,
        }
        doc.apply_edit(final_starless.astype(np.float32, copy=False), metadata=meta, step_name="Stars Removed")
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


# ------------------------------------------------------------
# New document helper
# ------------------------------------------------------------
def _push_as_new_doc(main, doc, arr: np.ndarray, title_suffix="_stars", source="Stars-Only"):
    dm = getattr(main, "docman", None)
    if not dm or not hasattr(dm, "open_array"):
        return
    try:
        base = ""
        if hasattr(doc, "display_name") and callable(doc.display_name):
            base = doc.display_name()
        else:
            base = getattr(doc, "name", "") or "Image"
        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2),
            "source": source
        }
        newdoc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=f"{base}{title_suffix}")
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
