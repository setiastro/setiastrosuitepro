#saspro/cosmicclarity_headless.py
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

def _get_darkstar():
    import os
    import sys
    # On Linux, when invoked as a subprocess from PixInsight, the inherited
    # LD_LIBRARY_PATH includes PI's Qt6 libs which clash with PyQt6.
    # Stripping PI paths from the environment before importing PyQt6 resolves
    # the undefined symbol crash.
    if sys.platform.startswith("linux"):
        lp = os.environ.get("LD_LIBRARY_PATH", "")
        filtered = ":".join(
            p for p in lp.split(":")
            if "PixInsight" not in p and "pixinsight" not in p
        )
        os.environ["LD_LIBRARY_PATH"] = filtered

    from setiastro.saspro.remove_stars import darkstar_starless_from_array
    return darkstar_starless_from_array

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

class _SkipSaveSignal(Exception):
    """Raised internally when sat_skip_save=True and no trail was detected."""
    pass

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

    chunk_size = int(preset.get("chunk_size", 256))
    overlap = int(preset.get("overlap", 64))

    if chunk_size < 64:
        chunk_size = 64
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    rgb01, was_mono = _to_rgb01(img)
    out = rgb01

    stages: list[str] = []
    if mode == "sharpen":
        stages.append("sharpen")
    elif mode == "denoise":
        stages.append("denoise")
    elif mode == "both":
        stages.extend(["sharpen", "denoise"])
    elif mode == "superres":
        stages.append("superres")
    elif mode == "satellite":
        stages.append("satellite")
    elif mode == "darkstar":
        stages.append("darkstar")
    else:
        raise RuntimeError(f"Unknown mode: {mode}")

    nstages = max(1, len(stages))

    def stage_progress_adapter(stage_index: int):
        stage_base = int(100 * stage_index / nstages)
        stage_next = int(100 * (stage_index + 1) / nstages)
        stage_span = max(1, stage_next - stage_base)

        def _cb(done: int, total: int):
            if progress_cb is None:
                return True
            if total <= 0:
                return progress_cb(stage_base, 100)
            pct = stage_base + int(stage_span * (done / total))
            return progress_cb(pct, 100)
        return _cb

    for i, stage in enumerate(stages):
        prog = stage_progress_adapter(i)

        if stage == "sharpen":
            out = sharpen_rgb01(
                out,
                sharpening_mode=str(preset.get("sharpening_mode", "Both")),
                stellar_amount=float(preset.get("stellar_amount", 0.5)),
                nonstellar_amount=float(preset.get("nonstellar_amount", 0.5)),
                nonstellar_strength=float(preset.get("nonstellar_psf", 3.0)),
                auto_detect_psf=bool(preset.get("auto_psf", True)),
                separate_channels=bool(preset.get("sharpen_channels_separately", False)),
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                overlap=overlap,
                temp_stretch=bool(preset.get("temp_stretch", False)),
                target_median=float(preset.get("target_median", 0.25)),
                stellar_correct_mode=str(preset.get("stellar_correct_mode", "sharpen_only")),
                execution_mode=str(preset.get("sharpen_execution_mode", preset.get("execution_mode", "auto"))),
                batch_size_override=int(preset.get("sharpen_batch_size_override", preset.get("batch_size_override", 0))),
                progress_cb=prog,
            )

        elif stage == "denoise":
            out = denoise_rgb01(
                out,
                denoise_strength=float(preset.get("denoise_luma", 0.5)),
                denoise_mode=str(preset.get("denoise_mode", "full")),
                separate_channels=bool(preset.get("separate_channels", False)),
                color_denoise_strength=float(preset.get("denoise_color", 0.5)),
                use_gpu=use_gpu,
                lite=bool(preset.get("denoise_lite", False)),
                walking=bool(preset.get("denoise_walking", False)),
                chunk_size=chunk_size,
                overlap=overlap,
                temp_stretch=bool(preset.get("temp_stretch", False)),
                target_median=float(preset.get("target_median", 0.25)),
                progress_cb=prog,
            )
        elif stage == "superres":
            out = superres_rgb01(
                out,
                scale=int(preset.get("scale", 2)),
                use_gpu=True,
                progress_cb=prog,
            )

        elif stage == "satellite":
            models = get_satellite_models(use_gpu=use_gpu, status_cb=lambda _: None)

            def sat_prog(done: int, total: int):
                if progress_cb is None:
                    return True
                if total <= 0:
                    return progress_cb(0, 100)
                return prog(done, total)

            out, detected = satellite_remove_image(
                out,
                models=models,
                mode=str(preset.get("sat_mode", "full")).lower(),
                clip_trail=bool(preset.get("sat_clip_trail", True)),
                sensitivity=float(preset.get("sat_sensitivity", 0.10)),
                progress_cb=sat_prog,
            )

            if bool(preset.get("sat_skip_save", False)) and not detected:
                # Signal to the file-level runner that we should skip writing output
                raise _SkipSaveSignal("No satellite trail detected — skip save requested.")

        elif stage == "darkstar":
            def ds_prog(done: int, total: int, _stage: str):
                if progress_cb is None:
                    return
                if total <= 0:
                    progress_cb(0, 100)
                    return
                prog(done, total)

            darkstar_starless_from_array = _get_darkstar()
            starless, _stars_only, _was_mono_engine = darkstar_starless_from_array(
                out,
                use_gpu=use_gpu,
                chunk_size=int(preset.get("chunk_size", preset.get("stride", 512))),
                overlap_frac=float(preset.get("overlap_frac", 0.125)),
                mode=str(preset.get("darkstar_mode", "unscreen")),
                processing_path=str(preset.get("processing_path", "hybrid_luma_color")),
                output_stars_only=bool(preset.get("show_extracted_stars", False)),
                edge_padding=int(preset.get("edge_padding", 64)),
                compatibility_mode=bool(preset.get("compatibility_mode", False)),
                progress_cb=ds_prog,
                status_cb=lambda _msg: None,
            )
            out = starless

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

    try:
        out = run_cosmicclarity_on_array(img, preset, progress_cb=progress_cb)
    except _SkipSaveSignal as e:
        print(f"[Satellite] {e} Output not written.", flush=True)
        return

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