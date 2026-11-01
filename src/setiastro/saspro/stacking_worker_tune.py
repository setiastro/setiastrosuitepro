# ──────────────────────────────────────────────────────────────────────────────
#  src/setiastro/saspro/stacking_worker_tune.py — SetiAstroSuite Pro
#  Copyright (C) 2026 Franklin Marek / SetiAstro
#  Licensed under the GNU General Public License v3.0 or later.
# ──────────────────────────────────────────────────────────────────────────────
"""
Decide how registration should detect stars on THIS machine.

There is no safe constant. Measured on a 36-core box with an 8 GB card:

    2048x2048 frames:  8 workers 109.6 f/s, 16 workers 100.2, 32 collapses
    3000x4000 frames:  8 workers  71.6 f/s, 16 workers HUNG (killed at 20 min)

Same hardware, same code — the cliff moved because per-worker GPU working set
scales with frame width. And the ranking between backends is hardware-dependent
in the other direction too: a 36-core CPU running SEP across every core gets
within 1.6x of the GPU, so an i9 paired with a small card will genuinely be
faster on the CPU. The only way to know is to measure on the user's box.

So this module does two things:

  `recommended_workers(...)`  a cheap, VRAM-aware default that needs no
                              measurement — use it as the floor so an untuned
                              install never oversubscribes the card.

  `tune(...)`                 the real measurement: escalate worker counts for
                              each backend, with a watchdog that kills a
                              configuration the moment it goes off the rails,
                              and persist the winner to QSettings.

`tune()` takes `status_cb` and `cancel_cb` so it drops straight into a
QThread-backed settings button (the StarRegistrationWorker idiom:
`progress_update.emit` and `_is_cancelled`).

The watchdog is the part that matters. A pathological worker count does not
fail, it thrashes: the card pegs at 100% and the batch never returns. The
budget is derived from the measurement itself — a batch is abandoned once it
exceeds `ABORT_FACTOR` x the single-worker time for the same batch, because N
workers can never legitimately be slower than one. On the data above that kills
the 16-worker 3000x4000 case at ~3 s instead of 20 minutes.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

__all__ = [
    "BAND_KEY",
    "KEY_WORKERS_MANUAL",
    "KEY_BACKEND",
    "KEY_FPS",
    "KEY_MP",
    "KEY_STAMP",
    "KEY_WORKERS",
    "clear_tuning",
    "hardware_stamp",
    "read_tuning",
    "recommended_workers",
    "resolve_align_workers",
    "stamp_is_current",
    "tune",
    "tune_all",
    "tune_band_height",
]

# ── QSettings keys ───────────────────────────────────────────────────────────
# The user-facing one. 0 means "Auto": let this module decide. Anything above
# zero is an explicit choice and is honoured as-is. It lives under
# stacking/align/* like every other shared alignment pref, so the Stacking
# Suite settings page and the Star Registration window read and write the same
# value — the same arrangement stacking/align/kernel already uses.
KEY_WORKERS_MANUAL = "stacking/align/workers"

# The optimizer's own bookkeeping. Separate from the above so a measurement can
# be invalidated when the hardware changes without silently discarding a number
# the user typed in themselves.
KEY_BACKEND = "stacking/align/tuned_backend"   # "gpu" | "cpu"
KEY_WORKERS = "stacking/align/tuned_workers"   # int
KEY_FPS = "stacking/align/tuned_fps"           # float, frames/s at that setting
KEY_MP = "stacking/align/tuned_mp"             # megapixels the tuning was done at
KEY_STAMP = "stacking/align/tuned_stamp"       # hardware fingerprint

# ── watchdog / escalation policy ─────────────────────────────────────────────
ABORT_FACTOR = 1.5      # abandon a batch past this x the 1-worker time
ABORT_FLOOR_S = 4.0     # ...but never give a batch less than this
FALLOFF_STOP = 0.90     # stop climbing once throughput drops below this x best
KNEE_TOL = 0.97         # accept the fewest workers within this x the peak
KILL_GRACE_S = 3.0      # terminate -> kill grace for a wedged worker

WORKER_LADDER = (1, 2, 4, 8, 12, 16, 24, 32)


# ─────────────────────────────────────────────────────────────────────────────
# cheap default (no measurement)
# ─────────────────────────────────────────────────────────────────────────────
def _total_vram_mb() -> float | None:
    """Total VRAM on the active torch device, or None if not a GPU/unknown."""
    try:
        from setiastro.saspro import gpu_star_detect as G
        torch, dev = G._resolve()
        if torch is None or dev is None:
            return None
        if str(getattr(dev, "type", "")) != "cuda":
            return None
        _free, total = torch.cuda.mem_get_info()
        return float(total) / 2 ** 20
    except Exception:
        return None


def recommended_workers(megapixels: float = 12.0,
                        backend: str = "gpu",
                        cpu_count: int | None = None) -> int:
    """
    A safe worker count with no measurement required.

    For the CPU backend this is just the core count — SEP is pure CPU and
    scales cleanly (measured monotonic to 32 workers).

    For the GPU it is bounded by VRAM, because each spawned worker carries its
    own CUDA context plus a working set that grows with frame width. The
    coefficients are fitted to the measurements in this module's docstring:
    ~400 MB fixed per worker plus ~25 MB per megapixel, against 75% of the
    card. That yields 8 workers at 12 MP on an 8 GB card and 12 at 4 MP, both
    of which match what was measured.
    """
    nproc = int(cpu_count or os.cpu_count() or 8)
    if backend == "cpu":
        return max(1, min(nproc, 32))

    vram = _total_vram_mb()
    if not vram:
        # unknown card (ROCm / DirectML / MPS report differently): stay modest
        return max(1, min(nproc, 4))

    mp_ = max(1.0, float(megapixels))
    per_worker = 400.0 + 25.0 * mp_
    cap = int((vram * 0.75) // per_worker)
    return max(1, min(nproc, 32, cap))


# ─────────────────────────────────────────────────────────────────────────────
# hardware fingerprint — so stored tuning is discarded when the box changes
# ─────────────────────────────────────────────────────────────────────────────
def hardware_stamp() -> str:
    bits = [platform.machine(), f"cpu{os.cpu_count()}"]
    try:
        from setiastro.saspro import gpu_star_detect as G
        torch, dev = G._resolve()
        if torch is None:
            bits.append("notorch")
        else:
            bits.append(f"torch{getattr(torch, '__version__', '?')}")
            bits.append(str(dev))
            try:
                if str(getattr(dev, "type", "")) == "cuda":
                    bits.append(torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception:
        bits.append("notorch")
    return "|".join(bits)


def read_tuning(megapixels: float | None = None) -> dict | None:
    """
    Return the stored tuning if it is still applicable, else None.

    Discarded when the hardware fingerprint changed, or when the frame size now
    in use is more than 1.5x away from the size it was tuned at — the 2048 vs
    3000x4000 result above shows that a worker count safe at one size can wedge
    the card at another.
    """
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        backend = s.value(KEY_BACKEND, None)
        workers = s.value(KEY_WORKERS, None)
        if backend is None or workers is None:
            return None
        if str(s.value(KEY_STAMP, "")) != hardware_stamp():
            return None
        out = {
            "backend": str(backend),
            "workers": int(workers),
            "fps": float(s.value(KEY_FPS, 0.0) or 0.0),
            "mp": float(s.value(KEY_MP, 0.0) or 0.0),
        }
        if megapixels and out["mp"] > 0:
            r = float(megapixels) / out["mp"]
            if r > 1.5 or r < (1 / 1.5):
                return None
        return out
    except Exception:
        return None


def stamp_is_current() -> bool | None:
    """
    True/False if a stored hardware stamp matches this machine, None if none.

    Used by the UI to say "measured on different hardware — re-run Optimize"
    rather than silently overriding a number the user can see in a spin box.
    """
    try:
        from PyQt6.QtCore import QSettings
        stored = QSettings().value(KEY_STAMP, None)
        if stored in (None, ""):
            return None
        return str(stored) == hardware_stamp()
    except Exception:
        return None


def resolve_align_workers(megapixels: float | None = None,
                          *, cpu_count: int | None = None) -> tuple[int, str]:
    """
    How many processes the registration solve pass should use, and why.

    Returns (workers, source) where source is a short phrase for the status
    pane. Resolution order, most specific first:

      1. an explicit `stacking/align/workers` above zero — the user typed a
         number into one of the two dialogs, or the optimizer wrote its result
         there. Honoured as-is: an explicit setting is not second-guessed.
      2. a stored measurement whose hardware stamp and frame size still match.
      3. the VRAM-derived default, which needs no measurement.

    This is the single place that decision lives. The patch in
    star_alignment.py calls it rather than carrying the logic inline, so
    changing the policy does not mean re-patching a 7,500-line file.
    """
    # 1. explicit
    try:
        from PyQt6.QtCore import QSettings
        raw = QSettings().value(KEY_WORKERS_MANUAL, 0)
        manual = int(raw) if raw not in (None, "") else 0
    except Exception:
        manual = 0
    if manual > 0:
        return manual, "set in settings"

    # 2. measured
    t = read_tuning(megapixels)
    if t and int(t.get("workers", 0)) > 0:
        return int(t["workers"]), f"measured ({t.get('backend', '?')})"

    # 3. default
    try:
        from PyQt6.QtCore import QSettings
        v = QSettings().value("stacking/align/gpu_detect", True)
        gpu_on = str(v).strip().lower() not in ("0", "false", "no", "off")
    except Exception:
        gpu_on = True
    backend = "gpu" if gpu_on else "cpu"
    return (recommended_workers(megapixels or 12.0, backend, cpu_count),
            f"{backend} default")


def clear_tuning() -> None:
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        for k in (KEY_BACKEND, KEY_WORKERS, KEY_FPS, KEY_MP, KEY_STAMP):
            s.remove(k)
    except Exception:
        pass


def _save_tuning(backend: str, workers: int, fps: float, mp_: float) -> bool:
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        s.setValue(KEY_BACKEND, str(backend))
        s.setValue(KEY_WORKERS, int(workers))
        s.setValue(KEY_FPS, float(fps))
        s.setValue(KEY_MP, float(mp_))
        s.setValue(KEY_STAMP, hardware_stamp())
        # Keep the switch astroalign._find_sources already reads in step with
        # the verdict, so nothing else has to know about this module: a CPU
        # verdict turns the GPU detector off rather than leaving it on to be
        # measured as the loser and used anyway.
        s.setValue("stacking/align/gpu_detect", backend == "gpu")
        # Put the measured number where the user can see and edit it. Leaving
        # it in tuned_workers only would show "Auto" in both dialogs straight
        # after a run that just measured a value, which reads like the run did
        # nothing. The tuned_* keys stay as the record of WHAT was measured and
        # on which hardware, so the UI can still say "re-run Optimize" after a
        # GPU swap instead of silently overriding what is now an explicit
        # setting.
        s.setValue(KEY_WORKERS_MANUAL, int(workers))
        s.sync()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# worker side — each worker synthesises ONE frame and holds it, so the timed
# region is detection only: no disk, no per-task synthesis.
# ─────────────────────────────────────────────────────────────────────────────
_FRAME = None


def _init(root: str, h: int, w: int, seed: int):
    global _FRAME
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    rng = np.random.default_rng(seed + os.getpid())
    img = rng.normal(100.0, 3.0, size=(h, w)).astype(np.float32)
    n = max(200, int(h * w / 12000))
    ys = rng.integers(12, h - 12, n)
    xs = rng.integers(12, w - 12, n)
    amp = rng.uniform(300.0, 7000.0, n).astype(np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            wt = 1.0 if (dy or dx) == 0 else (0.6 if dy * dx == 0 else 0.36)
            np.add.at(img, (ys + dy, xs + dx), amp * wt)
    _FRAME = img


def _job_gpu(_i):
    from setiastro.saspro import gpu_star_detect as G
    t0 = time.perf_counter()
    pts = G.detect_stars(_FRAME, detection_sigma=5.0, min_area=5,
                         max_stars=120, allow_cpu=True)
    return (time.perf_counter() - t0, 0 if pts is None else len(pts),
            pts is None)


def _job_cpu(_i):
    import sep
    t0 = time.perf_counter()
    im = np.ascontiguousarray(_FRAME, dtype=np.float32)
    bkg = sep.Background(im)
    src = sep.extract(im - bkg.back(), 5.0 * bkg.globalrms, minarea=5)
    return time.perf_counter() - t0, len(src), False


def _job_warm(_i):
    """One detection to build the CUDA context before the clock starts."""
    try:
        from setiastro.saspro import gpu_star_detect as G
        G.warmup()
    except Exception:
        pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# the measurement
# ─────────────────────────────────────────────────────────────────────────────
def _kill_pool(ex) -> None:
    """
    Forcibly stop a wedged pool.

    `shutdown(wait=False, cancel_futures=True)` cannot stop a task that is
    already running, and a worker spinning on a thrashing GPU will not notice
    cancellation — so reach for the child processes. `_processes` is private
    but stable across CPython 3.9-3.13, and there is no public equivalent.
    """
    procs = list(getattr(ex, "_processes", {}) or {})
    objs = list((getattr(ex, "_processes", {}) or {}).values())
    for p in objs:
        try:
            p.terminate()
        except Exception:
            pass
    t0 = time.perf_counter()
    for p in objs:
        try:
            p.join(timeout=max(0.1, KILL_GRACE_S - (time.perf_counter() - t0)))
        except Exception:
            pass
    for p in objs:
        try:
            if p.is_alive():
                p.kill()
        except Exception:
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    del procs


def _run_batch(backend, nw, nframes, h, w, root, budget_s, cancel_cb):
    """
    Time `nframes` detections across `nw` workers.

    Returns (seconds, n_fallback, None) on success, or (None, 0, reason) when
    the batch did not complete. The reason is distinguished because the three
    cases call for different handling: "budget" means this worker count is
    pathological and higher counts will be worse, so stop climbing; "error"
    means the backend is broken and the whole sweep is pointless; "cancelled"
    means the user asked to stop.
    """
    ctx = mp.get_context("spawn")
    fn = _job_gpu if backend == "gpu" else _job_cpu
    warm_fn = _job_warm if backend == "gpu" else fn
    ex = ProcessPoolExecutor(max_workers=nw, mp_context=ctx,
                             initializer=_init,
                             initargs=(root, h, w, 1234))
    try:
        # start every worker and pay backend start-up outside the timed region
        warm = [ex.submit(warm_fn, i) for i in range(nw)]
        done, not_done = wait(warm, timeout=max(90.0, 20.0 * nw))
        if not_done:
            _kill_pool(ex)
            return None, 0, "budget"
        for f in done:
            try:
                f.result()
            except Exception as e:
                _kill_pool(ex)
                return None, 0, f"error: {type(e).__name__}: {e}"

        t0 = time.perf_counter()
        futs = [ex.submit(fn, i) for i in range(nframes)]
        pending = set(futs)
        nfb = 0
        deadline = t0 + budget_s
        while pending:
            if cancel_cb is not None and cancel_cb():
                _kill_pool(ex)
                return None, 0, "cancelled"
            if time.perf_counter() > deadline:
                _kill_pool(ex)
                return None, 0, "budget"
            got, pending = wait(pending, timeout=0.25,
                                return_when=FIRST_COMPLETED)
            for f in got:
                try:
                    _dt, _n, fb = f.result()
                    nfb += 1 if fb else 0
                except Exception as e:
                    _kill_pool(ex)
                    return None, 0, f"error: {type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
    finally:
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    return dt, nfb, None


def _sweep(backend, nframes, h, w, root, ladder, status_cb, cancel_cb):
    """Escalate worker counts until throughput falls off or a batch is killed."""
    log = status_cb or (lambda *_: None)
    rows = []
    t_one = None
    best_fps = 0.0
    best_nw = 0

    for nw in ladder:
        if cancel_cb is not None and cancel_cb():
            log("   cancelled.")
            break

        if t_one is None:
            budget = max(ABORT_FLOOR_S, 120.0)      # first run sets the scale
        else:
            budget = max(ABORT_FLOOR_S, ABORT_FACTOR * t_one)

        dt, nfb, reason = _run_batch(backend, nw, nframes, h, w, root,
                                     budget, cancel_cb)
        if reason is not None:
            if reason == "cancelled":
                log("   cancelled.")
                break
            if reason.startswith("error"):
                # the backend itself is broken; nothing to learn from more rungs
                log(f"   {backend} x{nw:<3} FAILED — {reason}")
                rows.append({"workers": nw, "fps": 0.0, "aborted": True,
                             "reason": reason})
                break
            log(f"   {backend} x{nw:<3} ABANDONED (exceeded {budget:.1f}s "
                f"budget) — not trying more workers")
            rows.append({"workers": nw, "fps": 0.0, "aborted": True,
                         "reason": reason})
            break

        fps = nframes / dt if dt > 0 else 0.0
        rows.append({"workers": nw, "seconds": dt, "fps": fps,
                     "fallback": nfb, "aborted": False})
        extra = f"  ({nfb} fell back to SEP)" if nfb else ""
        log(f"   {backend} x{nw:<3} {dt:7.2f}s   {fps:7.2f} frames/s{extra}")

        if t_one is None:
            t_one = dt
        if fps > best_fps:
            best_fps, best_nw = fps, nw
        elif fps < best_fps * FALLOFF_STOP:
            log(f"   {backend}: past the peak at {best_nw} workers — stopping")
            break

    # Take the KNEE, not the argmax. Measured on a 3070 Ti at 12 MP:
    # x4 = 45.13 f/s, x8 = 44.78 f/s — a 0.8% difference, which is noise, and
    # picking between them on noise is arbitrary. The smaller count is strictly
    # better: on the GPU it is less VRAM and one fewer ~6.5 s CUDA start-up per
    # worker, and on the CPU it hands cores back to the rest of the pipeline.
    # So: the fewest workers that reach KNEE_TOL of the best measured rate.
    good = [r for r in rows if not r.get("aborted") and r.get("fps", 0) > 0]
    if good:
        peak = max(r["fps"] for r in good)
        knee = min((r for r in good if r["fps"] >= peak * KNEE_TOL),
                   key=lambda r: r["workers"])
        if knee["workers"] < best_nw:
            log(f"   {backend}: {knee['workers']} workers is within "
                f"{(1 - KNEE_TOL) * 100:.0f}% of the peak at {best_nw} "
                f"({knee['fps']:.2f} vs {peak:.2f} frames/s) — taking the "
                f"smaller count")
            best_fps, best_nw = knee["fps"], knee["workers"]

    return rows, best_fps, best_nw


# ─────────────────────────────────────────────────────────────────────────────
# Band height — the integration / rejection knob
#
# torch_rejection and torch_calibration have NO worker count: they run in the
# parent process and torch_rejection already serialises GPU access through
# _GPU_LOCK. Their tunable is `stacking/chunk_height`, the number of full-width
# rows in one band handed to _torch_reduce_tile.
#
# stacking_suite ships _DEFAULT_BAND_ROWS = 16 with the note "16 keeps per-band
# pixel load close to the old, proven 256x256 tile (~65 KPx), so it won't
# regress any GPU; raise to 32/64 to trade more per-band GPU work for fewer,
# fatter passes." That is a deliberately safe guess. 16 rows x 4000 px is
# 64 KPx per GPU call, so a 12 MP frame becomes ~188 separate launches per
# integration pass and the card spends its time on launch overhead rather than
# arithmetic. How far it can be raised depends on VRAM, on the stack depth N,
# and on the algorithm, so it is exactly the sort of thing to measure.
#
# The sweep calls the REAL reducer (torch_rejection.torch_reduce_tile) on a
# synthetic band stack, so whatever the rejection math actually costs is what
# gets measured. The rejection math itself is not touched.
# ─────────────────────────────────────────────────────────────────────────────
BAND_LADDER = (16, 32, 64, 128, 256, 512, 1024)
BAND_KEY = "stacking/chunk_height"

# Measured on a 3070 Ti, 12 MP frames, 24 deep, winsorized sigma clipping:
#
#   rows   ms/band   ms per row   Mpx/s
#     16       6.7      0.419      9.58
#     32      13.0      0.406      9.87
#     64      22.6      0.353     11.30
#    128      43.7      0.341     11.70
#    256      89.6      0.350     11.40
#    512     174.7      0.341     11.70
#   1024     374.6      0.366     10.90
#
# Time is almost perfectly LINEAR in rows and throughput varies only 1.22x over
# a 64x range of band sizes. So integration is not launch-overhead bound the way
# detection was — it is bound by the rejection arithmetic itself (three
# iterations of nanmedian/nanstd, each sorting N values per pixel). The shipped
# default of 16 was a sound choice.
#
# A bigger band costs host RAM linearly: compute_safe_chunk sizes the prefetch
# pool as N * _PREFETCH_POOL_GUESS + workers buffers per row, so 128 rows wants
# 8x the RAM of 16 — and on a RAM-tight box compute_safe_chunk clamps it back
# anyway. So a band is only worth taking if it buys something real.
#
# The question to ask is NOT "is the whole curve flat". Two runs of the above:
#
#   rows      run1    run2    % of peak
#     16      9.58    9.46      82 / 81
#     32      9.87   11.70      84 / 100
#     64     11.30   11.70      97 / 100
#   >=128    ~11.6   ~11.6     ~99 / ~99
#
# 16 rows is the worst point in BOTH runs, consistently ~1.23x off the best, so
# there IS a real step from 16 to 32 and it costs ~6 MB of band buffer. Judging
# by total spread would have thrown that away: the spread is 1.221x and 1.237x
# on the two runs, which straddles any fixed threshold and makes the verdict a
# coin toss between runs.
#
# So "flat" means: the SMALLEST band is already within tolerance of the best.
# That is the actual condition under which keeping the default is right, and it
# has no cliff to sit on.
#
# BAND_KNEE_TOL is looser than the worker-count KNEE_TOL because band timings
# are noisier — the 32-row point moved 18.5% between the two runs above, so a
# 3% tolerance would chase noise around the plateau and land somewhere
# different every time.
BAND_KNEE_TOL = 0.90
BAND_REPS = 5


def _mark_band_schema_current(settings, log=None) -> bool:
    """
    Stop stacking_suite's one-time band migration from eating a tuned value.

    StackingSuiteDialog.__init__ runs a version-gated migration that resets
    `stacking/chunk_height` to `_DEFAULT_BAND_ROWS`, because pre-band builds
    stored a 2-D TILE height there and a stale 256 means a 256 x width band in
    band mode. It is gated so a user's later choice survives — but the gate is
    `settings_schema_version < _STACKING_SCHEMA_VERSION`, and on a fresh install
    that version is absent. So:

        fresh install -> run the CLI tuner (writes chunk_height = 32)
                      -> open Stacking Suite for the first time
                      -> migration fires, chunk_height is reset to 16, silently

    The Optimize button is not affected, because the dialog's __init__ has
    already run and bumped the version by then. Only the CLI-before-first-open
    path loses the value.

    So when the tuner writes a band height it also marks the schema current.
    The version is read from stacking_suite rather than duplicated here, so a
    future bump does not leave a stale constant behind; if that import is not
    possible the write is skipped and behaviour is exactly as before.
    """
    try:
        from setiastro.saspro.stacking_suite import _STACKING_SCHEMA_VERSION
    except Exception:
        if log:
            log("   (could not read the settings schema version; if you have "
                "never opened Stacking Suite on this install, open it once "
                "and re-check 'Rows per pass')")
        return False
    try:
        cur = int(settings.value("stacking/settings_schema_version", 0) or 0)
    except Exception:
        cur = 0
    if cur < int(_STACKING_SCHEMA_VERSION):
        settings.setValue("stacking/settings_schema_version",
                          int(_STACKING_SCHEMA_VERSION))
        if log:
            log(f"   (marked settings schema v{int(_STACKING_SCHEMA_VERSION)} "
                f"so the one-time band migration will not reset this)")
    return True


def _band_stack(n_frames, rows, width, channels, seed=11):
    """A synthetic (N, rows, width, C) band stack with outliers to reject."""
    rng = np.random.default_rng(seed)
    shape = (n_frames, rows, width, channels)
    ts = rng.normal(1000.0, 12.0, size=shape).astype(np.float32)
    # ~0.5% hot pixels so the rejection branches actually execute
    k = max(1, int(ts.size * 0.005))
    idx = rng.integers(0, ts.size, k)
    flat = ts.reshape(-1)
    flat[idx] += rng.uniform(300.0, 5000.0, k).astype(np.float32)
    return ts


def _time_band(rows, width, n_frames, channels, algo,
               reps=BAND_REPS):
    """
    Returns (seconds, megapixels_per_second, None) for one band height, or
    (None, 0.0, reason) when it did not run.

    The reason is reported rather than collapsed into "something went wrong":
    an OOM at a given band height is a useful measurement and means stop
    widening, while an import error or a bad algorithm name means the sweep
    itself is misconfigured and no band height will work. Those need different
    messages or the user is left guessing.
    """
    try:
        from setiastro.saspro.torch_rejection import torch_reduce_tile
    except Exception as e:
        return None, 0.0, f"setup: cannot import torch_reduce_tile ({e})"
    try:
        ts = _band_stack(n_frames, rows, width, channels)
        w = np.ones(n_frames, dtype=np.float32)
        kw = dict(algo_name=algo, sigma_low=2.5, sigma_high=2.5, iterations=3)

        out, _rej = torch_reduce_tile(ts, w, **kw)        # warm / compile
        if out is None:
            return None, 0.0, "reducer returned None"

        best = None
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            torch_reduce_tile(ts, w, **kw)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        # throughput in output megapixels per second (band area, not stack volume)
        mpx_s = (rows * width * channels) / 1e6 / max(best, 1e-9)
        return best, mpx_s, None
    except MemoryError as e:
        return None, 0.0, f"oom: {e}"
    except Exception as e:
        msg = str(e).lower()
        kind = "oom" if ("out of memory" in msg or "alloc" in msg) else "error"
        return None, 0.0, f"{kind}: {type(e).__name__}: {e}"


def tune_band_height(frame_shape=(3000, 4000),
                     n_frames: int = 24,
                     channels: int = 1,
                     algo: str = "Weighted Windsorized Sigma Clipping",
                     *,
                     status_cb=None,
                     cancel_cb=None,
                     save: bool = True,
                     ram_cap_rows: int | None = None) -> dict:
    """
    Find the band height that integrates fastest without blowing up.

    Returns {"rows": int, "mpx_s": float, "rows_tested": [...], "saved": bool}.

    `ram_cap_rows` lets the caller pass stacking_suite's own host-RAM ceiling
    (`compute_safe_chunk` already derives one) so tuning never recommends a
    band the reader cannot buffer. Without it the sweep is bounded only by what
    the GPU tolerates, and stacking_suite's clamp will still apply at run time.
    """
    log = status_cb or (lambda *_: None)
    h, w = int(frame_shape[0]), int(frame_shape[1])

    log(f"⚙️  Tuning integration band height for {w}x{h}, "
        f"{n_frames} frames deep, {channels} channel(s), algo={algo}.")

    ladder = [r for r in BAND_LADDER if r <= h]
    if ram_cap_rows:
        ladder = [r for r in ladder if r <= int(ram_cap_rows)]
        log(f"    host-RAM ceiling caps the sweep at {ram_cap_rows} rows.")
    if not ladder:
        ladder = [min(16, h)]

    rows_tested = []
    best_mpx = 0.0
    best_rows = 0
    t_first = None

    for rows in ladder:
        if cancel_cb is not None and cancel_cb():
            log("   cancelled.")
            break

        dt, mpx, why = _time_band(rows, w, n_frames, channels, algo)
        if why is not None:
            if why.startswith("setup"):
                # nothing to do with band height; no point sweeping further
                log(f"   band sweep unavailable — {why}")
                rows_tested.append({"rows": rows, "mpx_s": 0.0,
                                    "aborted": True, "reason": why})
                break
            log(f"   band {rows:>5} rows: {why} — not going wider")
            rows_tested.append({"rows": rows, "mpx_s": 0.0, "aborted": True,
                                "reason": why})
            break
        # Watchdog, same principle as the worker sweep: a band that is too big
        # for the card does not error, it thrashes. Per-band time must grow no
        # worse than linearly with area, so a band whose time per pixel has
        # blown out past the first measurement is pathological.
        if t_first is None:
            t_first = dt / rows
        elif (dt / rows) > t_first * ABORT_FACTOR * 2.0:
            log(f"   band {rows:>5} rows: {dt * 1000:8.1f} ms "
                f"— time per row blew out ({(dt/rows)/t_first:.1f}x); "
                f"treating as over budget, not going wider")
            rows_tested.append({"rows": rows, "mpx_s": mpx, "aborted": True})
            break

        band_mb = (n_frames * rows * w * channels * 4) / 2 ** 20
        rows_tested.append({"rows": rows, "seconds": dt, "mpx_s": mpx,
                            "band_mb": band_mb, "aborted": False})
        log(f"   band {rows:>5} rows: {dt * 1000:8.1f} ms/band   "
            f"{mpx:8.3g} Mpx/s   {band_mb:7.1f} MB/band stack")

        if mpx > best_mpx:
            best_mpx, best_rows = mpx, rows
        elif mpx < best_mpx * FALLOFF_STOP:
            log(f"   past the peak at {best_rows} rows — stopping")
            break

    good = [r for r in rows_tested if not r.get("aborted") and r["mpx_s"] > 0]
    if not good:
        log("⚠️  No usable band height measured; leaving the setting alone.")
        return {"rows": None, "mpx_s": 0.0, "rows_tested": rows_tested,
                "saved": False}

    # Knee: the smallest band within BAND_KNEE_TOL of the peak. A smaller band
    # is less VRAM and less host RAM for the prefetch buffers, so when two are
    # within noise of each other the smaller one wins.
    peak = max(r["mpx_s"] for r in good)
    floor_ = min(r["mpx_s"] for r in good)
    spread = peak / max(floor_, 1e-9)
    knee = min((r for r in good if r["mpx_s"] >= peak * BAND_KNEE_TOL),
               key=lambda r: r["rows"])
    best_rows, best_mpx = knee["rows"], knee["mpx_s"]

    # "Flat" = the smallest band is ALREADY as fast as the best, so there is
    # nothing to buy by going wider. Not "the total spread is small": a curve
    # with one real step at the bottom and a long plateau above it has a small
    # spread but a genuine, cheap gain to take.
    smallest = min(good, key=lambda r: r["rows"])
    flat = len(good) >= 2 and smallest["mpx_s"] >= peak * BAND_KNEE_TOL

    if flat:
        best_rows, best_mpx = smallest["rows"], smallest["mpx_s"]
        log(f"✅ Band height barely matters here: {best_rows} rows already "
            f"reaches {best_mpx / peak * 100:.0f}% of the best measured rate "
            f"({peak:.3g} Mpx/s), so integration is bound by the rejection "
            f"arithmetic, not by per-band overhead.")
        log(f"   Keeping {best_rows} rows — a bigger band would cost host RAM "
            f"linearly for nothing.")
    else:
        base = next((r for r in good if r["rows"] == 16), None)
        if base and base["mpx_s"] > 0 and best_rows != 16:
            log(f"✅ Band height {best_rows} rows: {best_mpx:.3g} Mpx/s — "
                f"{best_mpx / base['mpx_s']:.2f}x the shipped default of 16, "
                f"which measured {base['mpx_s'] / peak * 100:.0f}% of the best "
                f"rate here.")
        else:
            log(f"✅ Band height {best_rows} rows: {best_mpx:.3g} Mpx/s.")
        chosen = next((r for r in good if r["rows"] == best_rows), None)
        if chosen and chosen.get("band_mb"):
            log(f"   One band holds {chosen['band_mb']:.1f} MB of stack at "
                f"{n_frames} frames deep; compute_safe_chunk will clamp this "
                f"if the prefetch pool will not fit in RAM.")

    saved = False
    if save:
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings()
            s.setValue(BAND_KEY, int(best_rows))
            _mark_band_schema_current(s, log)
            s.sync()
            saved = True
        except Exception:
            saved = False
        log("   Saved." if saved else
            "   Could not write settings (no Qt?); result not persisted.")

    return {"rows": best_rows, "mpx_s": best_mpx, "spread": spread,
            "flat": flat, "rows_tested": rows_tested, "saved": saved}


def tune(frame_shape=(3000, 4000),
         nframes: int | None = None,
         *,
         root: str | None = None,
         status_cb=None,
         cancel_cb=None,
         save: bool = True,
         test_gpu: bool = True,
         test_cpu: bool = True) -> dict:
    """
    Measure the best star-detection backend and worker count on this machine.

    frame_shape : (H, W) — use the user's real sensor size; the answer depends
                  on it (16 workers was fine at 2048x2048 and wedged the same
                  card at 3000x4000).
    nframes     : detections per batch; defaults to ~3x the core count, bounded.
    status_cb   : called with progress lines (QThread: progress_update.emit).
    cancel_cb   : called between batches; return True to stop.
    save        : persist the winner to QSettings.

    Returns a dict with `backend`, `workers`, `fps`, the full `gpu`/`cpu` row
    lists, and `saved`.
    """
    log = status_cb or (lambda *_: None)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    mp_ = (h * w) / 1e6
    nproc = os.cpu_count() or 8
    if nframes is None:
        nframes = int(min(96, max(16, 3 * nproc)))
    if root is None:
        # directory containing the `setiastro` package
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    log(f"⚙️  Tuning star detection for {w}x{h} ({mp_:.1f} MP), "
        f"{nframes} detections per batch, {nproc} cores.")
    log(f"    {hardware_stamp()}")

    gpu_cap = recommended_workers(mp_, "gpu", nproc)
    cpu_cap = recommended_workers(mp_, "cpu", nproc)
    gpu_ladder = [n for n in WORKER_LADDER if n <= max(gpu_cap * 2, 2)]
    cpu_ladder = [n for n in WORKER_LADDER if n <= cpu_cap]
    log(f"    VRAM-derived GPU default is {gpu_cap}; probing up to "
        f"{gpu_ladder[-1] if gpu_ladder else 0}.")

    results = {"gpu": [], "cpu": []}
    bests = {}

    if test_gpu:
        try:
            from setiastro.saspro import gpu_star_detect as G
            # SASPRO_GPU_DETECT_ALLOW_CPU=1 means the user deliberately enabled
            # the torch path on CPU, so tuning should measure it rather than
            # skip it — CPU torch threads the work and is not automatically
            # slower than SEP.
            live = G.gpu_available() or (
                os.environ.get("SASPRO_GPU_DETECT_ALLOW_CPU") == "1"
                and G._resolve()[0] is not None
            )
        except Exception:
            live = False
        if not live:
            log("   no GPU backend available — skipping the GPU sweep.")
        else:
            log("   GPU (torch) sweep:")
            rows, fps, nw = _sweep("gpu", nframes, h, w, root, gpu_ladder,
                                   log, cancel_cb)
            results["gpu"] = rows
            if nw:
                bests["gpu"] = (fps, nw)

    if test_cpu and not (cancel_cb is not None and cancel_cb()):
        log("   CPU (SEP) sweep:")
        rows, fps, nw = _sweep("cpu", nframes, h, w, root, cpu_ladder,
                               log, cancel_cb)
        results["cpu"] = rows
        if nw:
            bests["cpu"] = (fps, nw)

    if not bests:
        log("⚠️  Tuning produced no usable result; leaving settings untouched.")
        return {"backend": None, "workers": None, "fps": 0.0,
                "gpu": results["gpu"], "cpu": results["cpu"], "saved": False}

    backend = max(bests, key=lambda k: bests[k][0])
    fps, workers = bests[backend]

    other = "cpu" if backend == "gpu" else "gpu"
    if other in bests:
        ofps, onw = bests[other]
        margin = fps / max(ofps, 1e-9)
        log(f"✅ {backend.upper()} wins: {workers} workers, {fps:.2f} frames/s "
            f"— {margin:.2f}x the best {other.upper()} "
            f"({onw} workers, {ofps:.2f} frames/s).")
        # A thin win is not worth the VRAM and the 6.5 s/worker start-up.
        if backend == "gpu" and margin < 1.15:
            log("   Margin is inside noise; preferring the CPU path, which "
                "costs no VRAM and no CUDA start-up.")
            backend, (fps, workers) = other, bests[other]
    else:
        log(f"✅ {backend.upper()}: {workers} workers, {fps:.2f} frames/s.")

    saved = _save_tuning(backend, workers, fps, mp_) if save else False
    if save:
        log("   Saved." if saved else
            "   Could not write settings (no Qt?); result not persisted.")

    return {"backend": backend, "workers": workers, "fps": fps,
            "mp": mp_, "gpu": results["gpu"], "cpu": results["cpu"],
            "saved": saved}


def tune_all(frame_shape=(3000, 4000),
             *,
             n_frames: int = 24,
             channels: int = 1,
             algo: str = "Weighted Windsorized Sigma Clipping",
             status_cb=None,
             cancel_cb=None,
             save: bool = True,
             ram_cap_rows: int | None = None) -> dict:
    """
    Both stages, in the order a settings button should run them.

    Stage 1 (registration) decides the star detector and the solve-pass worker
    count. Stage 2 (integration) decides the band height for the rejection
    reducer. They are independent: stage 1 is about how many PROCESSES may hold
    a CUDA context at once, stage 2 is about how much work goes into ONE call in
    the parent process. One number does not imply the other.
    """
    log = status_cb or (lambda *_: None)
    det = tune(frame_shape, status_cb=log, cancel_cb=cancel_cb, save=save)
    if cancel_cb is not None and cancel_cb():
        return {"detection": det, "band": None}
    log("")
    band = tune_band_height(frame_shape, n_frames, channels, algo,
                            status_cb=log, cancel_cb=cancel_cb, save=save,
                            ram_cap_rows=ram_cap_rows)
    return {"detection": det, "band": band}


# ─────────────────────────────────────────────────────────────────────────────
def _cli_print(*a):
    """Flushing printer for the CLI.

    Python block-buffers stdout when it is a pipe or a file, so `... > log.txt`
    or `| tee` shows nothing at all until the sweep ends — which looks exactly
    like the hang this tool exists to prevent. Interactive consoles are line
    buffered so it only bites when redirected, which is when someone is most
    likely to be watching for progress.
    """
    print(*a, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    _args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _flags = {a for a in sys.argv[1:] if a.startswith("--")}
    _h = int(_args[0]) if len(_args) > 0 else 3000
    _w = int(_args[1]) if len(_args) > 1 else 4000
    _n = int(_args[2]) if len(_args) > 2 else None

    if "--band" in _flags:
        out = tune_band_height((_h, _w), _n or 24, status_cb=_cli_print, save=False)
        print()
        print(f"RESULT: band_rows={out['rows']} mpx_s={out['mpx_s']:.1f}")
    elif "--detect" in _flags:
        out = tune((_h, _w), _n, status_cb=_cli_print, save=False)
        print()
        print(f"RESULT: backend={out['backend']} workers={out['workers']} "
              f"fps={out['fps']:.2f}")
    else:
        out = tune_all((_h, _w), n_frames=_n or 24, status_cb=_cli_print, save=False)
        d, b = out["detection"], out["band"]
        print()
        print(f"RESULT: backend={d['backend']} workers={d['workers']} "
              f"fps={d['fps']:.2f}"
              + (f" | band_rows={b['rows']} mpx_s={b['mpx_s']:.1f}"
                 if b else ""))