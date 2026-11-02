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
    "worker_ladder",
    "stamp_is_current",
    "tune",
    "tune_all",
    "tune_band_height",
    "tune_warp",
    "finalize_layout",
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
# The first rung has no baseline to derive a budget from, so it gets a flat
# one. It has to cover the SLOWEST configuration of the slowest backend on the
# largest sensor — single-threaded SEP on a 46.8 MP frame is minutes, and the
# old 120 s made that rung abandon every time. Overshooting here is cheap:
# a rung that would really hang is caught by the per-rung budget from the
# second rung onward.
FIRST_BUDGET_S = 420.0
FALLOFF_STOP = 0.90     # stop climbing once throughput drops below this x best
KNEE_TOL = 0.97         # accept the fewest workers within this x the peak
KILL_GRACE_S = 3.0      # terminate -> kill grace for a wedged worker

WORKER_LADDER = (1, 2, 4, 8, 12, 16, 24, 32)
# How many midpoint probes to run after the climb stops. The ladder's gaps
# (4->8, 16->24, 24->32) are wide enough to hide the real best, so when the
# sweep ends between two rungs the midpoint is worth one batch. Two steps
# closes even the 24->32 gap to within one worker; beyond that the gain is
# smaller than the run-to-run noise.
REFINE_STEPS = 2


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
        # WAIT for the workers to actually exit on the success path. Every
        # early return above has already been through _kill_pool, which
        # terminates and joins, so the only case reaching here is a batch that
        # finished cleanly and whose pool will close immediately.
        #
        # This used to be wait=False, which returns before the children are
        # gone. The sweep then spawned the next rung's pool on top of the
        # previous one's dying workers — each still holding a CUDA context.
        # With the midpoint probe added that overlap got wide enough for the
        # watchdog test to catch 14 live processes after a sweep.
        try:
            ex.shutdown(wait=True, cancel_futures=True)
        except Exception:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
    return dt, nfb, None


def worker_ladder(cpu_count: int | None = None) -> list[int]:
    """
    The worker counts a sweep will probe.

    Deliberately NOT bounded by the VRAM heuristic. That heuristic is a
    conservative guess for installs that never measure; using it to bound the
    measurement makes the measurement no better than the guess. On a 46.8 MP
    sensor it reads 3, and bounding at 2x it stopped probing at 4 workers
    while throughput was still climbing.

    What ends a sweep is the watchdog (a rung too big for the card is
    abandoned) and the falloff rule (throughput dropped past the peak).

    Exposed so tests exercise the real ladder rather than rebuilding it.
    """
    nproc = int(cpu_count or os.cpu_count() or 8)
    ladder_max = max(2, min(nproc, 32))
    return [n for n in WORKER_LADDER if n <= ladder_max]


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
            budget = max(ABORT_FLOOR_S, FIRST_BUDGET_S)
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

            rows.append({"workers": nw, "fps": 0.0, "aborted": True,
                         "reason": reason})
            if t_one is None:
                # No baseline yet, so this rung is the SLOWEST configuration
                # there is — fewer workers cannot be faster. A timeout here
                # says the batch is too big for one worker, not that the
                # backend is hopeless, so keep climbing and let a later rung
                # set the scale. Giving up here is what made a 46.8 MP CPU
                # sweep report "cpu x1 ABANDONED" and never try x2.
                log(f"   {backend} x{nw:<3} too slow for one batch within "
                    f"{budget:.0f}s — trying more workers")
                continue

            log(f"   {backend} x{nw:<3} ABANDONED (exceeded {budget:.1f}s "
                f"budget) — not trying more workers")
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

    # ── bisect the gap the sweep stopped in ─────────────────────────────────
    # The ladder is coarse: 1, 2, 4, 8, 12, 16, 24, 32. If 4 works and 8 is
    # abandoned, the usable ceiling is somewhere in between and 6 may be both
    # safe and faster — the same is true when 8 merely falls off rather than
    # aborting, because the peak can sit between two rungs. Probing the
    # midpoint costs one batch and is the difference between "4 workers" and
    # the actual best the card will take.
    for _ in range(REFINE_STEPS):
        if cancel_cb is not None and cancel_cb():
            break
        ok_rows = [r for r in rows
                   if not r.get("aborted") and r.get("fps", 0) > 0]
        if not ok_rows or t_one is None or best_nw <= 0:
            break
        # Bisect between the BEST rung and whatever ended the climb above it —
        # not between the highest *successful* rung and the next. In the
        # falloff case the rung that stopped the sweep did succeed, it was
        # just slower, so keying off the highest success leaves nothing above
        # it to bisect against and the refinement never fires at all.
        lo = int(best_nw)
        above = [r["workers"] for r in rows if r["workers"] > lo]
        if not above:
            break
        hi = min(above)
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break

        budget = max(ABORT_FLOOR_S, ABORT_FACTOR * t_one)
        dt, nfb, reason = _run_batch(backend, mid, nframes, h, w, root,
                                     budget, cancel_cb)
        if reason is not None:
            if reason == "cancelled":
                break
            rows.append({"workers": mid, "fps": 0.0, "aborted": True,
                         "reason": reason})
            log(f"   {backend} x{mid:<3} ABANDONED too (between {lo} and "
                f"{hi}) — {lo} stands")
            break

        fps = nframes / dt if dt > 0 else 0.0
        rows.append({"workers": mid, "seconds": dt, "fps": fps,
                     "fallback": nfb, "aborted": False, "refined": True})
        log(f"   {backend} x{mid:<3} {dt:7.2f}s   {fps:7.2f} frames/s"
            f"   (probing between {lo} and {hi})")
        if fps > best_fps:
            best_fps, best_nw = fps, mid
        if hi - mid < 2:
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
            behind = (1.0 - knee["fps"] / peak) * 100.0
            log(f"   {backend}: {best_nw} workers was fastest at "
                f"{peak:.2f} frames/s, but {knee['workers']} is within "
                f"{behind:.1f}% of it ({knee['fps']:.2f} frames/s) and costs "
                f"{best_nw - knee['workers']} fewer worker(s).")
            log(f"   Anything within {(1 - KNEE_TOL) * 100:.0f}% of the best "
                f"counts as a tie and the smaller count wins — on the GPU "
                f"that is less VRAM and one fewer CUDA start-up each; on the "
                f"CPU it hands cores back to the rest of the pipeline.")
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
# Starts BELOW the shipped default of 16 on purpose. At 46.8 MP and 64 frames
# deep the curve was 16 -> 5.38, 32 -> 5.11, 64 -> 4.59 Mpx/s: monotonically
# decreasing, so the peak is somewhere at or below 16 and a ladder starting at
# 16 can never see it.
BAND_LADDER = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
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

        # Say WHY, whenever the pick is not simply the fastest. Reporting only
        # the winner makes a deliberate trade look like a bug: "16 rows: 4.5
        # Mpx/s" next to a table whose best line reads 4.62 invites exactly
        # the question of why the faster one was passed over.
        fastest = max(good, key=lambda r: r["mpx_s"])
        if fastest["rows"] != best_rows:
            behind = (1.0 - best_mpx / peak) * 100.0
            msg = (f"   {fastest['rows']} rows was fastest at "
                   f"{peak:.3g} Mpx/s, but {best_rows} is within "
                   f"{behind:.1f}% of it")
            if chosen and chosen.get("band_mb") and fastest.get("band_mb"):
                msg += (f" and holds {chosen['band_mb']:.1f} MB per band "
                        f"against {fastest['band_mb']:.1f} MB")
            log(msg + ".")
            log(f"   Anything within {(1 - BAND_KNEE_TOL) * 100:.0f}% of the "
                f"best counts as a tie and the smaller band wins, because "
                f"band timings move about that much between runs. Set 'Rows "
                f"per pass' by hand for the outright fastest.")

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
        # Scale the batch with the sensor so the sweep's wall time stays
        # roughly constant. 96 frames is fine at 12 MP; at 46.8 MP the same
        # count is four times the pixels and makes the single-worker CPU rung
        # take minutes for no extra information.
        _scale = 12.0 / max(1.0, mp_)
        nframes = int(min(96, max(12, round(3 * nproc * _scale))))
    if root is None:
        # directory containing the `setiastro` package
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))

    log(f"⚙️  Tuning star detection for {w}x{h} ({mp_:.1f} MP), "
        f"{nframes} detections per batch, {nproc} cores.")
    log(f"    {hardware_stamp()}")

    gpu_cap = recommended_workers(mp_, "gpu", nproc)
    cpu_cap = recommended_workers(mp_, "cpu", nproc)

    # Probe the FULL ladder for both backends. An earlier version bounded the
    # GPU sweep at 2x the VRAM heuristic, which is wrong twice over: the
    # heuristic is a conservative guess for installs that never measure, so
    # using it to bound the measurement makes the measurement no better than
    # the guess — and on a 46.8 MP sensor it stopped probing at 4 workers
    # while throughput was still climbing (13.91 -> 15.56 -> 16.42 f/s).
    #
    # What stops the sweep is the watchdog and the falloff rule, which is
    # exactly what they are for. A rung too big for the card is abandoned
    # within ABORT_FACTOR x the single-worker time rather than being
    # pre-emptively skipped.
    gpu_ladder = worker_ladder(nproc)
    cpu_ladder = [n for n in gpu_ladder if n <= cpu_cap]
    ladder_max = gpu_ladder[-1] if gpu_ladder else 0
    log(f"    VRAM-derived GPU default is {gpu_cap} (used only when nothing "
        f"has been measured); probing up to {ladder_max}.")

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — should the anti-ring warp run on the GPU?
#
# `_finalize_write_job` runs `finalize_workers` processes, each warping with
# `cpu_count // finalize_workers` cv2 threads, and gpu_warp's slot gate lets
# exactly one of them use the card. So enabling it does not replace the pool,
# it REPLACES ONE WORKER: the other N-1 keep doing cv2 warps either way.
#
# That makes the test simple, and it is not the test the first benchmark ran.
# Comparing the card against the whole pool said "loses 5x" and was measuring
# the wrong thing. What matters is whether one GPU warp beats one CPU worker
# at the thread count that worker actually gets:
#
#     all-CPU   throughput = W / t_cpu(T)
#     hybrid    throughput = 1 / t_gpu + (W - 1) / t_cpu(T)
#
# so the card is worth using iff t_gpu < t_cpu(T), and the gain is the
# difference between those two rates — small when they are close, large on a
# strong card paired with a weak CPU, which is the configuration this exists
# for. Measured on a 3070 Ti + 36-core Xeon the two are within a few percent;
# on an RTX 5090 with eight cores they will not be.
# ─────────────────────────────────────────────────────────────────────────────
KEY_GPU_WARP = "stacking/align/gpu_warp"
KEY_GPU_WARP_SLOTS = "stacking/align/gpu_warp_slots"
WARP_MARGIN = 1.05      # the card must beat one CPU worker by this to be worth it
WARP_REPS = 3


def finalize_layout(cpu_count: int | None = None,
                    finalize_workers: int | None = None) -> tuple[int, int]:
    """(workers, cv2_threads_per_worker) exactly as `_finalize_writes` picks them.

    Mirrors star_alignment: `min(cpu_count, 8)` workers, floor 2, and
    `cpu_count // workers` OpenCV threads each. Measuring at any other thread
    count answers a question nobody asked.
    """
    ncpu = int(cpu_count or os.cpu_count() or 8)
    w = int(finalize_workers or min(ncpu, 8))
    w = max(2, w)
    return w, max(1, ncpu // w)


def _cpu_antiring_fn():
    """The shipped `_warp_antiring`, imported if possible, lifted if not.

    Lifting matters for a CLI run, where importing star_alignment drags in Qt.
    Either way the reference is the function the pipeline really calls, not a
    copy of it that can drift.
    """
    try:
        from setiastro.saspro.star_alignment import _warp_antiring
        return _warp_antiring
    except Exception:
        pass
    try:
        import ast
        import cv2
        here = os.path.dirname(os.path.abspath(__file__))
        src = open(os.path.join(here, "star_alignment.py"), encoding="utf-8").read()
        fn = next(n for n in ast.parse(src).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_warp_antiring")
        ns = {"np": np, "cv2": cv2}
        exec(compile(ast.Module([fn], []), "<lifted>", "exec"), ns)
        return ns["_warp_antiring"]
    except Exception:
        return None


def _warp_frame(h, w, seed=5):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.05, 0.004, (h, w)).astype(np.float32)
    n = max(200, (h * w) // 40000)
    yy, xx = np.mgrid[-5:6, -5:6].astype(np.float32)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.7 ** 2)).astype(np.float32)
    for y, x, a in zip(rng.integers(8, h - 8, n), rng.integers(8, w - 8, n),
                       rng.uniform(0.05, 0.98, n).astype(np.float32)):
        img[y - 5:y + 6, x - 5:x + 6] += a * g
    return np.clip(img, 0.0, None)


def tune_warp(frame_shape=(3000, 4000),
              *,
              finalize_workers: int | None = None,
              status_cb=None,
              cancel_cb=None,
              save: bool = True) -> dict:
    """
    Decide whether the anti-ring warp should use the GPU on this machine.

    Returns {"use_gpu", "t_gpu", "t_cpu", "workers", "threads", "gain",
             "slots", "saved"}.
    """
    log = status_cb or (lambda *_: None)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    W, T = finalize_layout(finalize_workers=finalize_workers)
    out = {"use_gpu": False, "t_gpu": None, "t_cpu": None,
           "workers": W, "threads": T, "gain": 1.0, "slots": 1, "saved": False}

    log(f"⚙️  Tuning the anti-ring warp for {w}x{h}: finalize runs {W} workers "
        f"x {T} cv2 thread(s).")
    log("    One GPU slot replaces ONE of those workers, so the question is "
        "whether the card beats a single worker — not the whole pool.")

    try:
        from setiastro.saspro import gpu_warp as GW
    except Exception as e:
        log(f"   gpu_warp unavailable ({type(e).__name__}); leaving it off.")
        return _save_warp(out, save, log)
    if not GW.available():
        log("   No GPU device for warping; leaving it off.")
        return _save_warp(out, save, log)
    if GW.bilinear_mode() is None:
        log("   This OpenCV build's bilinear could not be identified, so the "
            "GPU warp would decline every frame anyway; leaving it off.")
        return _save_warp(out, save, log)

    cpu_fn = _cpu_antiring_fn()
    if cpu_fn is None:
        log("   Could not reach _warp_antiring to compare against; "
            "leaving it off.")
        return _save_warp(out, save, log)

    try:
        import cv2
    except Exception:
        log("   No cv2; leaving it off.")
        return _save_warp(out, save, log)

    img = _warp_frame(h, w)
    th = math.radians(0.31)
    c, s_ = math.cos(th), math.sin(th)
    M = np.array([[c, -s_, 7.43], [s_, c, -4.17], [9e-7, -5e-7, 1.0]], np.float64)

    def _gpu():
        return GW.warp_antiring_gpu(img, M, (w, h), perspective=True)

    def _cpu():
        return cpu_fn(
            lambda: cv2.warpPerspective(img, M, (w, h),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0),
            lambda: cv2.warpPerspective(img, M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0))

    def _best(fn, reps):
        best = None
        for _ in range(reps):
            if cancel_cb is not None and cancel_cb():
                return None
            t0 = time.perf_counter()
            r = fn()
            dt = time.perf_counter() - t0
            if r is None:
                return None
            best = dt if best is None else min(best, dt)
        return best

    old_threads = None
    try:
        if _gpu() is None:
            log("   The GPU warp declined this frame; leaving it off.")
            return _save_warp(out, save, log)
        t_gpu = _best(_gpu, WARP_REPS)

        old_threads = cv2.getNumThreads()
        cv2.setNumThreads(T)        # what one finalize worker really gets
        _cpu()                      # warm
        t_cpu = _best(_cpu, WARP_REPS)
    except Exception as e:
        log(f"   Warp timing failed ({type(e).__name__}: {e}); leaving it off.")
        return _save_warp(out, save, log)
    finally:
        try:
            if old_threads is not None:
                cv2.setNumThreads(int(old_threads))
        except Exception:
            pass

    if t_gpu is None or t_cpu is None:
        log("   cancelled.")
        return _save_warp(out, save, log)

    out["t_gpu"], out["t_cpu"] = t_gpu, t_cpu
    all_cpu = W / t_cpu
    hybrid = 1.0 / t_gpu + (W - 1) / t_cpu
    out["gain"] = hybrid / max(all_cpu, 1e-9)

    log(f"   gpu_warp          {t_gpu:7.3f} s/frame")
    log(f"   cv2, {T} thread(s) {t_cpu:7.3f} s/frame   (one finalize worker)")
    log(f"   all-CPU pool: {all_cpu:6.3f} frames/s   "
        f"hybrid: {hybrid:6.3f} frames/s   ({out['gain']:.2f}x)")

    # How many slots the card could hold, reported but not probed: measuring it
    # means several processes warping at once, which is exactly the thing that
    # hangs a card that cannot take it.
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            _free, total = torch.cuda.mem_get_info()
            room = int((total / 2 ** 20) * 0.70 // max(peak + 400.0, 1.0))
            out["slots"] = max(1, min(W, room))
            if out["slots"] > 1:
                log(f"   VRAM suggests up to {out['slots']} concurrent GPU "
                    f"warps ({peak:.0f} MB each + context); the default is 1. "
                    f"Raise 'gpu_warp_slots' only if you want to test it.")
    except Exception:
        pass

    if t_gpu * WARP_MARGIN < t_cpu:
        out["use_gpu"] = True
        log(f"✅ GPU warp ON: the card does a frame in {t_gpu:.2f}s against "
            f"{t_cpu:.2f}s for one CPU worker, so it adds "
            f"{(out['gain'] - 1) * 100:.0f}% to finalize throughput.")
    else:
        out["use_gpu"] = False
        if t_gpu < t_cpu:
            log(f"✅ GPU warp OFF: the card is faster than one worker but only "
                f"by {(t_cpu / t_gpu - 1) * 100:.0f}%, which is inside noise "
                f"and not worth the VRAM and the CUDA start-up.")
        else:
            log(f"✅ GPU warp OFF: one CPU worker with {T} thread(s) does a "
                f"frame in {t_cpu:.2f}s against the card's {t_gpu:.2f}s.")
    return _save_warp(out, save, log)


def _save_warp(out, save, log):
    if not save:
        return out
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        s.setValue(KEY_GPU_WARP, bool(out["use_gpu"]))
        s.sync()
        out["saved"] = True
    except Exception:
        out["saved"] = False
    log("   Saved." if out["saved"] else
        "   Could not write settings (no Qt?); result not persisted.")
    return out


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
    reducer. Stage 3 decides whether the anti-ring warp uses the GPU. All three
    are independent: stage 1 is about how many PROCESSES may hold a CUDA
    context at once, stage 2 about how much work goes into ONE call in the
    parent process, stage 3 about whether one finalize worker is better served
    by the card than by its share of the cores. No one of them implies another
    — the same box can want the GPU detector and the CPU warp, which is in fact
    the measured answer on a 3070 Ti beside a 36-core Xeon.
    """
    log = status_cb or (lambda *_: None)
    det = tune(frame_shape, status_cb=log, cancel_cb=cancel_cb, save=save)
    if cancel_cb is not None and cancel_cb():
        return {"detection": det, "band": None, "warp": None}
    log("")
    band = tune_band_height(frame_shape, n_frames, channels, algo,
                            status_cb=log, cancel_cb=cancel_cb, save=save,
                            ram_cap_rows=ram_cap_rows)
    if cancel_cb is not None and cancel_cb():
        return {"detection": det, "band": band, "warp": None}
    log("")
    warp = tune_warp(frame_shape, status_cb=log, cancel_cb=cancel_cb,
                     save=save)
    return {"detection": det, "band": band, "warp": warp}


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

    if "--warp" in _flags:
        out = tune_warp((_h, _w), status_cb=_cli_print, save=False)
        print()
        print(f"RESULT: gpu_warp={out['use_gpu']} t_gpu={out['t_gpu']} "
              f"t_cpu={out['t_cpu']} gain={out['gain']:.3f}")
    elif "--band" in _flags:
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
        wp = out.get("warp")
        print(f"RESULT: backend={d['backend']} workers={d['workers']} "
              f"fps={d['fps']:.2f}"
              + (f" | band_rows={b['rows']} mpx_s={b['mpx_s']:.1f}"
                 if b else "")
              + (f" | gpu_warp={wp['use_gpu']} gain={wp['gain']:.2f}"
                 if wp else ""))