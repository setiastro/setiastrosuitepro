# src/setiastro/saspro/blink_measure_worker.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Standalone, GUI-free SEP core for the Blink Comparator's frame metrics.

Why this exists: SEP (1.4.x) holds the GIL for the whole of sep.Background and
sep.extract, which are ~90% of a frame's measurement. Measured on a 36-core box,
8 threads ran either call at x0.9-1.0 of serial, so Blink's thread pool used one
core whatever its worker count. Running this function in ProcessPoolExecutor
children is what lets the metrics pass use every core, the same way
stacking_measure_worker does for the stacking suite.

It is deliberately import-light (numpy + sep only): no PyQt, no pyqtgraph, no
blink_comparator_pro. A spawned child imports this module and nothing else.

The body below is the SEP section of MetricsPanel._compute_one, moved here.
The one change is how SEP's pixel stack is sized (extract_small_first), which
is a speed/memory change, not a result change. The thread path calls this
same function in-process, so both paths run identical code. (SEP 1.4.1 itself
is not perfectly repeatable: re-extracting the same frame can shift a few
deblended objects' shapes, ~0.005% on median FWHM, depending on call history.)

Returns ``(fwhm_ds, ecc, star_cnt, noise, patch, trail)`` for Grade / Pre-flag:
noise (SEP globalrms), patchiness (std/median of the RMS map), and high-end
trail (80th-percentile eccentricity).
"""
from __future__ import annotations

import numpy as np

# SEP's pixel stack is allocated -- and touched -- on every extract call, so
# its size is a per-frame cost. Blink used 20M: ~800 MB per call, 324 ms of a
# ~650 ms frame, and with one call per core at once it saturated memory
# bandwidth (36 processes ran each frame 4x slower than one did). The stack's
# size only changes a result when it OVERFLOWS, which SEP reports as an error,
# so sep_tiled.extract_small_first raises it to 1M first and, on overflow,
# retries the SAME threshold at the size of the frame, which cannot overflow.
# Same answer as an unlimited stack. It only ever RAISES the stack: it is
# process-global, and lowering it under another tool in a GUI thread would
# make that tool overflow. (A fresh worker process starts at SEP's 300k.)


def _background_noise_and_patch(bkg) -> tuple[float, float]:
    """SEP globalrms and a patchiness score from the rms map."""
    try:
        gr = float(bkg.globalrms)
    except Exception:
        try:
            gr = float(np.nanmedian(np.asarray(bkg.rms(), dtype=np.float32)))
        except Exception:
            gr = float("nan")
    noise = float(gr) if np.isfinite(gr) and gr > 0 else float("nan")

    patch = float("nan")
    try:
        rms = np.asarray(bkg.rms(), dtype=np.float32)
        finite = np.isfinite(rms) & (rms > 0)
        if finite.any():
            med = float(np.median(rms[finite]))
            if med > 0:
                patch = float(np.std(rms[finite]) / med)
    except Exception:
        patch = float("nan")
    return noise, patch


def measure_ds(ds: np.ndarray):
    """
    SEP background + star extraction on the 2x-downsampled float32 frame.

    Returns ``(fwhm_ds, ecc, star_cnt, noise, patch, trail)`` with FWHM in
    downsampled pixels. When no catalogue is found, shape fields are NaN / 0
    and noise/patch may still be set. Returns ``None`` only on total failure;
    SEP failures raise and the caller turns them into its "bad frame" values.
    """
    import sep
    from setiastro.saspro.sep_tiled import extract_small_first

    try:
        ds = np.ascontiguousarray(ds, dtype=np.float32)
        if ds.ndim != 2 or ds.size == 0:
            return None
        if not np.isfinite(ds).all():
            ds = np.nan_to_num(ds, nan=0.0, posinf=1.0, neginf=0.0).astype(
                np.float32, copy=False)

        bkg = sep.Background(ds)
        back = bkg.back()
        noise, patch = _background_noise_and_patch(bkg)
        err = noise if np.isfinite(noise) else None

        candidates = [
            (7.0, 4),
            (5.0, 4),
            (4.0, 3),
            (3.5, 2),
        ]
        sub = ds - back
        cat = None
        for thr, minarea in candidates:
            try:
                cat = extract_small_first(
                    sep, sub,
                    thresh=float(thr),
                    err=err,
                    minarea=int(minarea),
                    clean=True,
                    deblend_nthresh=32,
                )
            except Exception:
                cat = None
            if cat is not None and len(cat) > 0:
                break

        if cat is None or len(cat) == 0:
            return float("nan"), float("nan"), 0, noise, patch, float("nan")

        a = np.maximum(cat["a"].astype(np.float32, copy=False), 1e-12)
        b = np.maximum(cat["b"].astype(np.float32, copy=False), 0.0)

        sig = np.sqrt(a * b).astype(np.float32, copy=False)
        fwhm_ds = float(np.nanmedian(2.3548 * sig))

        q = np.clip(b / a, 0.0, 1.0)
        e_true = np.sqrt(np.maximum(0.0, 1.0 - q * q))
        ecc = float(np.nanmedian(e_true))
        trail = float(np.nanpercentile(e_true, 80))

        return fwhm_ds, ecc, int(len(cat)), noise, patch, trail
    except Exception:
        return None
