from __future__ import annotations

import sys
import numpy as np
from typing import Callable, Optional
from pathlib import Path
from setiastro.saspro.legacy.image_manager import load_image, save_image

from setiastro.saspro.cosmicclarity_engines.sharpen_engine import sharpen_rgb01
from setiastro.saspro.cosmicclarity_engines.denoise_engine import denoise_rgb01
from setiastro.saspro.cosmicclarity_engines.superres_engine import superres_rgb01
from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
    get_satellite_models,
    satellite_remove_image,
)

ProgressCB = Optional[Callable[[int, int], bool]]  # (done,total)->continue?


def _to_rgb01(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Returns (rgb01_float32, was_mono).
    Assumes img is already float in [0,1] OR some other dtype already normalized by load_image.
    Your load_image already tends to return float01; if not, normalize here if needed.
    """
    arr = np.asarray(img)

    was_mono = False
    if arr.ndim == 2:
        was_mono = True
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        was_mono = True
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr, was_mono


def _back_to_mono_if_needed(rgb01: np.ndarray, was_mono: bool) -> np.ndarray:
    if not was_mono:
        return rgb01
    # preserve “mono stays mono”: any channel is fine since they’re identical for true mono workflows
    return np.asarray(rgb01[..., 0], dtype=np.float32)


def run_cosmicclarity_on_array(
    img: np.ndarray,
    preset: dict,
    *,
    progress_cb: ProgressCB = None,
) -> np.ndarray:
    """
    In-process Cosmic Clarity runner using the SAME preset keys your dialog produces.
    """
    mode = str(preset.get("mode", "sharpen")).lower()
    use_gpu = bool(preset.get("gpu", True))

    # ✅ NEW: chunking (shared across engines that support it)
    chunk_size = int(preset.get("chunk_size", 256))
    overlap    = int(preset.get("overlap", 64))

    # basic safety clamps (avoid weird/negative values from old configs)
    if chunk_size < 64:
        chunk_size = 64
    if overlap < 0:
        overlap = 0
    # keep overlap from exceeding chunk_size (seams/tiling math can break)
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    rgb01, was_mono = _to_rgb01(img)
    out = rgb01

    # small helper to wrap progress_cb into stage-aware progress
    def stage_progress_adapter(stage_base: int, stage_span: int):
        def _cb(done: int, total: int):
            if progress_cb is None:
                return True
            if total <= 0:
                return progress_cb(stage_base, 100)
            pct = stage_base + int(stage_span * (done / total))
            return progress_cb(pct, 100)
        return _cb

    if mode in ("sharpen", "both"):
        prog = stage_progress_adapter(0, 50 if mode == "both" else 100)
        out = sharpen_rgb01(
            out,
            sharpening_mode=str(preset.get("sharpening_mode", "Both")),
            stellar_amount=float(preset.get("stellar_amount", 0.5)),
            nonstellar_amount=float(preset.get("nonstellar_amount", 0.5)),
            nonstellar_strength=float(preset.get("nonstellar_psf", 3.0)),
            auto_detect_psf=bool(preset.get("auto_psf", True)),
            separate_channels=bool(preset.get("sharpen_channels_separately", False)),
            use_gpu=use_gpu,
            # ✅ NEW
            chunk_size=chunk_size,
            overlap=overlap,
            progress_cb=prog,
        )

    if mode in ("denoise", "both"):
        prog = stage_progress_adapter(50 if mode == "both" else 0, 50 if mode == "both" else 100)
        out = denoise_rgb01(
            out,
            denoise_strength=float(preset.get("denoise_luma", 0.5)),
            denoise_mode=str(preset.get("denoise_mode", "full")),
            separate_channels=bool(preset.get("separate_channels", False)),
            color_denoise_strength=float(preset.get("denoise_color", 0.5)),
            use_gpu=use_gpu,
            # ✅ NEW
            chunk_size=chunk_size,
            overlap=overlap,
            progress_cb=prog,
        )

    if mode == "superres":
        prog = stage_progress_adapter(0, 100)
        out = superres_rgb01(
            out,
            scale=int(preset.get("scale", 2)),
            use_gpu=True,  # matches your UI behavior
            progress_cb=prog,
        )

    if mode == "satellite":
        models = get_satellite_models(use_gpu=use_gpu, status_cb=lambda _: None)

        def sat_prog(done: int, total: int):
            if progress_cb is None:
                return True
            if total <= 0:
                return progress_cb(0, 100)
            return progress_cb(int(100 * done / total), 100)

        out, _detected = satellite_remove_image(
            out,
            models=models,
            mode=str(preset.get("sat_mode", "full")).lower(),
            clip_trail=bool(preset.get("sat_clip_trail", True)),
            sensitivity=float(preset.get("sat_sensitivity", 0.10)),
            progress_cb=sat_prog,
        )

    out = np.asarray(out, dtype=np.float32)
    out = np.clip(out, 0.0, 1.0)

    out = _back_to_mono_if_needed(out, was_mono)
    return out

def _infer_format_from_path(p: str, fallback: str = "tif") -> str:
    ext = (Path(p).suffix or "").lower().lstrip(".")
    if ext in ("tif", "tiff"):
        return "tif"
    if ext in ("fit", "fits"):
        return "fits"
    if ext in ("jpg", "jpeg"):
        return "jpg"
    if ext in ("png", "xisf"):
        return ext
    return fallback


def run_cosmicclarity_on_file(
    input_path: str,
    output_path: str,
    preset: dict,
    *,
    progress_cb: ProgressCB = None,
) -> None:
    img, hdr, bd, mono = load_image(input_path)
    if img is None:
        raise RuntimeError(f"Failed to load: {input_path}")

    out = run_cosmicclarity_on_array(img, preset, progress_cb=progress_cb)

    out_mono = mono or (out.ndim == 2)

    # Prefer output extension; if user gave no extension, fall back to input extension
    fmt = _infer_format_from_path(output_path, fallback=_infer_format_from_path(input_path, "tif"))

    # Preserve bit depth when reasonable (TIFF honors it; FITS honors too in your save_image)
    bit_depth = bd  # e.g. "8-bit", "16-bit", "32-bit floating point", etc.

    save_image(
        out,
        output_path,
        fmt,
        bit_depth=bit_depth,
        original_header=hdr,
        is_mono=out_mono,
    )