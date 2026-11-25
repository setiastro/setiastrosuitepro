# src/setiastro/saspro/cosmicclarity_engines/warmup.py
# SetiAstro Suite Pro -- get a Cosmic Clarity engine ready while its dialog is open.
#
# Copyright (c) 2026 Franklin Marek / SetiAstro

"""
When a tool's dialog opens, queue on the GPU thread (_gpu_thread) what its
first run would otherwise do before the first tile: import torch, load the
models, build the fp16 kernels for the input shapes that run will use (the
engines' warm_up functions). The user is still choosing settings; Execute
then starts on a warm engine.

A run queued behind a warm-up stops it at the next step and does whatever was
left itself, so a warm-up never makes a run slower. Nothing here raises or
shows anything: a warm-up that cannot run just leaves the run to do the work,
and says why in the log.
"""

from __future__ import annotations

import threading
import time

from setiastro.saspro.cosmicclarity_engines import _gpu_thread

_REQUESTED: set = set()
_LOCK = threading.Lock()


def forget() -> None:
    """After the model caches are cleared: the next dialog warms up again."""
    with _LOCK:
        _REQUESTED.clear()


def _hw_and_mono(image) -> tuple[tuple[int, int], bool]:
    """image: the array the run will be handed, or just its shape."""
    shape = tuple(image.shape) if hasattr(image, "shape") else tuple(image or ())
    if len(shape) < 2:
        raise ValueError("no image")
    mono = len(shape) == 2 or shape[2] == 1
    return (int(shape[0]), int(shape[1])), mono


def _queue(key, label: str, fn, *args, **kwargs):
    """Queue one warm-up behind any run; returns its Future, or None if this
    exact warm-up was already asked for."""
    with _LOCK:
        if key in _REQUESTED:
            return None
        _REQUESTED.add(key)

    def _job():
        t0 = time.perf_counter()
        try:
            n = fn(*args, should_stop=_gpu_thread.has_waiting_work, **kwargs)
            print(f"[CC warm-up] {label}: ready, {n} input shape(s) built "
                  f"in {time.perf_counter() - t0:.1f} s")
        except Exception as e:
            with _LOCK:
                _REQUESTED.discard(key)
            print(f"[CC warm-up] {label}: skipped ({type(e).__name__}: {e})")

    return _gpu_thread.submit_background(_job)


def warm_cosmic_clarity(preset: dict, image) -> list:
    """For the Cosmic Clarity dialog: the engines its preset's mode will run.
    Returns the Futures of the warm-ups it queued."""
    futs = []
    try:
        p = dict(preset or {})
        if not bool(p.get("gpu", True)):
            return futs
        hw, mono = _hw_and_mono(image)
        from setiastro.saspro.cosmicclarity_headless import resolve_overlap_px
        chunk = int(p.get("chunk_size", 256))
        overlap = int(resolve_overlap_px(p))
        mode = str(p.get("mode", "sharpen")).lower()

        if mode in ("sharpen", "both"):
            from setiastro.saspro.cosmicclarity_engines import sharpen_engine
            kw = dict(
                use_gpu=True,
                mode=str(p.get("sharpening_mode", "Both")),
                chunk_size=chunk, overlap=overlap,
                execution_mode=str(p.get("sharpen_execution_mode", p.get("execution_mode", "auto"))),
                batch_size_override=int(p.get("sharpen_batch_size_override",
                                              p.get("batch_size_override", 0))),
                stellar_correct_mode=str(p.get("stellar_correct_mode", "sharpen_only")),
                correct_model_version=str(p.get("correct_model_version", "V2 (latest)")),
            )
            futs.append(_queue(("sharpen", hw) + tuple(sorted(kw.items())), "Sharpen",
                               sharpen_engine.warm_up, hw, **kw))

        if mode in ("denoise", "both"):
            from setiastro.saspro.cosmicclarity_engines import denoise_engine
            kw = dict(
                use_gpu=True,
                denoise_mode=str(p.get("denoise_mode", "luminance")),
                separate_channels=bool(p.get("separate_channels", False)),
                mono_input=mono,
                chunk_size=chunk, overlap=overlap,
                lite=bool(p.get("denoise_lite", False)),
                walking=bool(p.get("denoise_walking", False)),
                execution_mode=str(p.get("denoise_execution_mode", "auto")),
                batch_size_override=int(p.get("denoise_batch_size_override", 0)),
            )
            futs.append(_queue(("denoise", hw) + tuple(sorted(kw.items())), "Denoise",
                               denoise_engine.warm_up, hw, **kw))
    except Exception as e:
        print(f"[CC warm-up] not started ({type(e).__name__}: {e})")
    return [f for f in futs if f is not None]


def warm_darkstar(params, image):
    """For a star-removal dialog running Dark Star with these DarkStarParams.
    Returns the warm-up's Future, or None."""
    try:
        if not bool(getattr(params, "use_gpu", True)):
            return None
        hw, mono = _hw_and_mono(image)
        from setiastro.saspro.cosmicclarity_engines import darkstar_engine
        key = ("darkstar", hw, mono, int(params.chunk_size), float(params.overlap_frac),
               str(params.processing_path), bool(params.compatibility_mode))
        return _queue(key, "Dark Star", darkstar_engine.warm_up, hw, params=params, mono_input=mono)
    except Exception as e:
        print(f"[CC warm-up] not started ({type(e).__name__}: {e})")
        return None
