# src/setiastro/saspro/cosmicclarity_engines/_fastops.py
# SetiAstro Suite Pro -- faster, bit-identical forms of full-frame numpy steps
# the Cosmic Clarity engines repeat on every run.
#
# Copyright (c) 2026 Franklin Marek / SetiAstro

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cc-median")
    return _POOL


def _median_of_channel(a: np.ndarray, c: int):
    return np.median(np.ascontiguousarray(a[..., c]))


def channel_medians(img) -> np.ndarray:
    """
    np.median(img, axis=(0, 1)) for an H x W x C image, same values, ~4x faster.

    For the two-axis form numpy first copies the whole frame into channel-major
    order (on a 12 MP RGB frame that copy is most of the 0.9 s), then selects.
    Here each channel is copied on its own and the three selections run at
    once: np.partition releases the GIL. Same selection, same float32 mean of
    the two middle values, so the results are identical, NaN included.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 2:
        return np.median(a, axis=(0, 1))
    meds = list(_pool().map(lambda c: _median_of_channel(a, c), range(a.shape[2])))
    return np.array(meds)


def fill_batch(arr: np.ndarray, batch_size: int, total: int) -> np.ndarray:
    """
    Pad a short last batch up to batch_size with copies of its last tile, when
    the run has more than one batch. Callers drop the extra outputs.

    Under fp16 autocast cuDNN compiles kernels for every new input shape it
    meets: ~3.5 s apiece on an RTX 3070 Ti, paid again in every session. The
    short last batch was always a new shape. A tile's fp16 output does shift
    with the batch it rides in (up to ~5e-4 on a 0..1 scale, measured), which
    every batch already does today.
    """
    n = int(arr.shape[0])
    if n == 0 or n >= batch_size or total <= batch_size:
        return arr
    return np.concatenate([arr, np.repeat(arr[-1:], batch_size - n, axis=0)], axis=0)


def pad_constant(image: np.ndarray, border_size: int, values) -> np.ndarray:
    """
    np.pad(..., mode="constant") of each channel with its own value, then
    np.stack -- as one allocation and one copy instead of a padded copy per
    channel plus a stacked copy of those. values: one per channel (a scalar
    for a 2-D image). Output dtype is image.dtype, as np.pad's is.
    """
    image = np.asarray(image)
    b = int(border_size)
    H, W = image.shape[:2]
    out = np.empty((H + 2 * b, W + 2 * b) + image.shape[2:], dtype=image.dtype)
    if image.ndim == 2:
        out[...] = values
    else:
        for c, v in enumerate(values):
            out[..., c] = v
    out[b:b + H, b:b + W, ...] = image
    return out
