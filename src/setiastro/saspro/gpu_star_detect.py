# src/setiastro/saspro/gpu_star_detect.py
"""
GPU star detector for the REGISTRATION path (SASpro).

Why this exists
---------------
``astroalign._find_sources`` ran SEP on every frame, every refinement pass:

    image = np.ascontiguousarray(img.astype(np.float32))   # full-frame copy
    bkg   = sep.Background(image, mask=mask)
    sources = sep.extract(image - bkg.back(), ...)         # 2 more full-frame arrays

On a 61 MP frame that is ~1 GB of allocation churn plus single-threaded C
extraction with connected-component labelling and deblending — per frame, per
pass. It is by a wide margin the slowest step in alignment.

The key realisation is that **alignment does not need SExtractor.** It needs N
repeatable sub-pixel centroids ordered by flux. Photometric completeness and
deblending fidelity do not matter here, because the matcher in astroalign 2.0
(canonical triangle invariants -> vertex voting -> parity-aware RANSAC ->
full-set re-match) was built specifically to tolerate dropped and spurious
detections. So we replace CC-labelling with:

    sigma-clipped background mesh  ->  threshold
    3x3..(2r+1) max-pool local maxima (this IS the "labelling")
    batched 2nd-moment centroid on NxSxS stamps

which is a handful of torch ops, runs on CUDA / MPS / DirectML through the
existing runtime_torch resolver, and returns in ~10 ms on a 61 MP frame.

Deliberate deviations from SEP
------------------------------
* Background centre is a 4-iteration 3-sigma-clipped MEAN, not SExtractor's
  mode estimate (2.5*median - 1.5*mean). The mode needs a per-cell median,
  which costs a sort over every cell and is the one reducer that is flaky on
  DirectML. A clipped mean sits a few percent high in crowded fields, i.e. the
  detection floor is marginally conservative. With SASpro's default
  ``align/det_sigma`` of 20 that difference is far below the noise.
* No deblending. Two stars closer than ``min_separation`` merge into one
  detection. The vertex-voting matcher does not care; a merged pair is simply
  one fewer usable control point.
* ``min_area`` is enforced as "at least N pixels above threshold inside a
  5x5 box" rather than true connected-component area. Same job (hot-pixel
  rejection), no labelling pass.

Bonus over SEP-as-called: second moments give FWHM and ellipticity for free,
so ``align/min_fwhm`` and ``align/max_ellipticity`` can be applied *inside*
detection instead of as a post-filter — hot pixels, cosmic rays and satellite
streaks never reach the matcher.

Contract
--------
``detect_stars()`` returns exactly what ``astroalign._find_sources`` returns:
an (N, 2) float array of (x, y) positions sorted by flux DESCENDING, or
``None`` if the GPU path is unavailable/fails (caller falls back to SEP).

Nothing heavy is imported at module scope — torch is resolved lazily through
``runtime_torch`` so PyInstaller bootstrap and CPU-only users are unaffected.

SASpro / Franklin Marek | www.setiastro.com
"""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "backend_note",
    "detect_stars",
    "detect_stars_full",
    "gpu_available",
    "reset_backend",
    "warmup",
]

# ─────────────────────────────────────────────────────────────────────────────
# Lazy torch resolution (same pattern as torch_rejection / torch_calibration)
# ─────────────────────────────────────────────────────────────────────────────
_TORCH = None
_DEVICE = None
_RESOLVED = False
_RESOLVE_NOTE = "not resolved"


def reset_backend():
    """Force the next call to re-resolve torch/device (used by tests)."""
    global _TORCH, _DEVICE, _RESOLVED, _RESOLVE_NOTE
    _TORCH = None
    _DEVICE = None
    _RESOLVED = False
    _RESOLVE_NOTE = "not resolved"


def _resolve():
    """
    Resolve (torch, device) once per process. Never raises: a failure here
    means the caller falls back to SEP.

    SASpro's `runtime_torch` is the preferred resolver because it owns the
    CUDA / ROCm / XPU / DirectML / MPS preference order and the PyInstaller
    sys.path fix-up. It is only *preferred*, though: if it cannot be imported
    (running the module standalone, a trimmed build), fall back to a plain
    `import torch` rather than disabling the whole GPU path silently.
    """
    global _TORCH, _DEVICE, _RESOLVED, _RESOLVE_NOTE
    if _RESOLVED:
        return _TORCH, _DEVICE
    _RESOLVED = True

    torch = None
    dev = None

    # Try a PLAIN import first, and only fall back to runtime_torch's
    # bootstrap if torch is genuinely not importable.
    #
    # This ordering matters because of where this function now runs.
    # runtime_torch.import_torch() is a bootstrapper: it can shell out to pip
    # (_ensure_numpy -> _pip_ok -> subprocess.run) to install or re-pin numpy.
    # That is fine in the parent process, which is the only place it used to be
    # called from — torch_rejection and torch_calibration both run in-parent.
    # But registration spawns up to N worker processes and each one calls
    # _find_sources -> here, so the bootstrap version of this would have N
    # workers shelling out to pip at once on any machine whose numpy needs
    # re-pinning. A pip subprocess is not an exception, it is a HANG, so the
    # try/except below would not have caught it.
    #
    # In a spawned worker torch is already importable (the parent made it so),
    # so the plain import succeeds and no bootstrap runs. runtime_torch is
    # still asked for the DEVICE, which is where its CUDA / ROCm / XPU /
    # DirectML / MPS preference order lives, so nothing is lost.
    try:
        import torch as _t
        torch = _t
        _RESOLVE_NOTE = "plain torch"
    except Exception:
        torch = None

    if torch is None:
        try:
            from setiastro.saspro.runtime_torch import (
                import_torch, add_runtime_to_sys_path,
            )
            try:
                add_runtime_to_sys_path(lambda *_: None)
            except Exception:
                pass
            torch = import_torch(
                prefer_cuda=True, prefer_xpu=False, prefer_dml=True,
                status_cb=lambda *_: None,
            )
            _RESOLVE_NOTE = "runtime_torch bootstrap"
        except Exception as e2:
            _TORCH, _DEVICE = None, None
            _RESOLVE_NOTE = f"no torch ({type(e2).__name__})"
            return _TORCH, _DEVICE

    # Device preference still comes from runtime_torch when it is importable.
    try:
        from setiastro.saspro.runtime_torch import best_device
        dev = best_device(torch, prefer_cuda=True, prefer_dml=True)
        if dev is not None:
            _RESOLVE_NOTE += " + runtime_torch device"
    except Exception:
        dev = None

    if torch is None:
        _TORCH, _DEVICE = None, None
        _RESOLVE_NOTE = "no torch"
        return _TORCH, _DEVICE

    if dev is None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = torch.device("cuda")
            elif (getattr(getattr(torch, "backends", None), "mps", None)
                  and torch.backends.mps.is_available()):
                dev = torch.device("mps")
            elif getattr(getattr(torch, "xpu", None), "is_available", None) \
                    and torch.xpu.is_available():
                dev = torch.device("xpu")
            else:
                dev = torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")

    _TORCH, _DEVICE = torch, dev
    return _TORCH, _DEVICE


def backend_note() -> str:
    """One-line description of how torch resolved — for status logs / support."""
    torch, dev = _resolve()
    if torch is None:
        return f"gpu_star_detect: disabled — {_RESOLVE_NOTE}"
    return (f"gpu_star_detect: torch {getattr(torch, '__version__', '?')} "
            f"on {dev} via {_RESOLVE_NOTE}")


def gpu_available() -> bool:
    """True iff torch resolves to a non-CPU device we can detect on."""
    torch, dev = _resolve()
    if torch is None or dev is None:
        return False
    try:
        # privateuseone == torch-directml
        return str(dev.type) not in ("cpu",)
    except Exception:
        return False


def _inference_ctx(torch):
    """no_grad/inference_mode that will not explode on older or DML builds."""
    try:
        if getattr(torch, "inference_mode", None) is not None:
            _C = getattr(torch, "_C", None)
            if _C is not None and hasattr(_C, "_InferenceMode"):
                return torch.inference_mode()
    except Exception:
        pass
    if getattr(torch, "no_grad", None) is not None:
        return torch.no_grad()
    import contextlib
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────────────────────
# Background: sigma-clipped mean on a coarse mesh, computed on a subsampled
# copy of the frame. The mesh only needs *statistics*, so sampling every 4th
# pixel costs 16x less work and leaves ~256 samples per 64x64 cell — plenty for
# a background estimate, and it keeps the whole stage off the critical memory
# path on small cards.
# ─────────────────────────────────────────────────────────────────────────────
def _background_mesh(torch, F, x, m, mesh, subsample, n_iter, clip_sigma):
    """
    x : (1,1,H,W) float32 on device, non-finite already replaced
    m : (1,1,H,W) float32 validity mask (1 = usable)

    Returns (cell_mean (1,1,ny,nx), globalrms float).
    """
    ss = max(1, int(subsample))
    xs = x[..., ::ss, ::ss]
    ms = m[..., ::ss, ::ss]

    cs = max(2, int(round(float(mesh) / ss)))

    # Pad up to a whole number of cells. Replicate on the image, zero on the
    # mask, so padded area contributes geometry but not statistics.
    hs, ws = xs.shape[-2], xs.shape[-1]
    ph = (-hs) % cs
    pw = (-ws) % cs
    if ph or pw:
        xs = F.pad(xs, (0, pw, 0, ph), mode="replicate")
        ms = F.pad(ms, (0, pw, 0, ph), mode="constant", value=0.0)

    k2 = float(cs * cs)

    def _sum(a):
        return F.avg_pool2d(a, cs, stride=cs) * k2

    mean = None
    std = None
    for it in range(max(1, int(n_iter))):
        cnt = _sum(ms).clamp_min(1.0)
        s1 = _sum(xs * ms)
        s2 = _sum(xs * xs * ms)
        mean = s1 / cnt
        var = (s2 / cnt - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        if it + 1 >= n_iter:
            break
        mu_up = F.interpolate(mean, size=xs.shape[-2:], mode="nearest")
        sd_up = F.interpolate(std, size=xs.shape[-2:], mode="nearest").clamp_min(1e-12)
        ms = ms * ((xs - mu_up).abs() <= (clip_sigma * sd_up)).to(xs.dtype)

    # globalrms: robust across cells. Small tensor, pull to host so we never
    # depend on a device median reducer (DirectML's are unreliable).
    try:
        rms_host = std.detach().reshape(-1).to("cpu").numpy()
        rms_host = rms_host[np.isfinite(rms_host) & (rms_host > 0)]
        globalrms = float(np.median(rms_host)) if rms_host.size else 0.0
    except Exception:
        globalrms = 0.0

    return mean, globalrms


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────
def detect_stars_full(
    image2d,
    *,
    detection_sigma: float = 5.0,
    min_area: int = 5,
    mask=None,
    max_stars: int | None = None,
    min_separation: int = 3,
    min_fwhm: float = 0.0,
    max_ellipticity: float = 1.0,
    stamp: int = 11,
    mesh: int = 64,
    subsample: int = 4,
    bkg_iters: int = 4,
    bkg_clip: float = 3.0,
    band_rows: int = 2048,
    max_peaks: int = 200_000,
    status_cb=None,
    allow_cpu: bool = False,
):
    """
    Detect stars on a 2-D image using torch.

    Returns a dict with keys x, y, flux, fwhm, ecc, peak, globalrms, threshold
    (all numpy, flux-descending), or ``None`` if the GPU path is unavailable or
    anything goes wrong — callers must treat ``None`` as "use SEP".

    ``allow_cpu`` runs the same kernels on CPU torch. It is off in production
    because CPU torch is slower than SEP's C extractor, but it lets the
    algorithm be verified against SEP on a machine with no GPU (the selftest
    uses it; ``SASPRO_GPU_DETECT_ALLOW_CPU=1`` forces it on).
    """
    log = status_cb or (lambda *_: None)

    torch, dev = _resolve()
    if torch is None or dev is None:
        return None
    if str(dev.type) == "cpu":
        # Off by default on CPU, conservatively: SEP is a mature C extractor
        # and this path has far less mileage on it. Note that "CPU torch is
        # slower than SEP" turned out NOT to be true when measured — on a
        # 1000x1400 frame CPU torch ran ~200 ms against SEP's ~600 ms, because
        # torch threads the work and sep.extract is single-threaded. So this
        # gate is about trust, not speed, and SASPRO_GPU_DETECT_ALLOW_CPU=1
        # (or allow_cpu=True) is a reasonable thing to turn on deliberately.
        if not (allow_cpu or os.environ.get("SASPRO_GPU_DETECT_ALLOW_CPU") == "1"):
            return None

    try:
        import torch.nn.functional as F
    except Exception:
        return None

    a = np.asarray(image2d)
    if a.ndim != 2 or a.size == 0:
        return None
    H, W = int(a.shape[0]), int(a.shape[1])
    if H < 16 or W < 16:
        return None

    half = int(stamp) // 2
    sep_r = max(1, int(min_separation))
    halo = max(sep_r, half, 2) + 1

    try:
        with _inference_ctx(torch):
            xt = torch.from_numpy(
                np.ascontiguousarray(a, dtype=np.float32)
            ).to(dev, dtype=torch.float32, non_blocking=True).reshape(1, 1, H, W)

            # ── validity: finite pixels, minus any caller mask ───────────────
            valid = torch.isfinite(xt)
            if mask is not None:
                mk = np.asarray(mask)
                if mk.ndim == 3:
                    mk = np.logical_or.reduce(mk, axis=-1)
                if mk.shape == (H, W):
                    mt = torch.from_numpy(
                        np.ascontiguousarray(mk.astype(bool))
                    ).to(dev).reshape(1, 1, H, W)
                    valid = valid & (~mt)
                    del mt

            # Non-finite pixels must not poison the pooled sums. Satellite
            # trails arrive as NaN under SASpro's 0.0 = no-data convention.
            xt = torch.where(valid, xt, torch.zeros_like(xt))
            mvalid = valid.to(xt.dtype)

            # ── background ───────────────────────────────────────────────────
            cell_mean, globalrms = _background_mesh(
                torch, F, xt, mvalid, mesh, subsample, bkg_iters, bkg_clip
            )
            if not np.isfinite(globalrms) or globalrms <= 0.0:
                return None

            bkg_full = F.interpolate(
                cell_mean, size=(H, W), mode="bilinear", align_corners=False
            )
            del cell_mean, mvalid

            # In-place so we never hold x and bkg and data simultaneously.
            xt.sub_(bkg_full)
            del bkg_full
            xt = torch.where(valid, xt, torch.zeros_like(xt))  # data, bkg-subtracted

            thr = float(detection_sigma) * float(globalrms)

            # ── banded detection ─────────────────────────────────────────────
            # Peak VRAM tracks the band, not the frame, so a 61 MP frame still
            # fits alongside the rest of the pipeline on a small card.
            bh = int(max(64, min(int(band_rows), H)))
            cy_l, cx_l, pk_l, fl_l, fw_l, ec_l = [], [], [], [], [], []

            ma = int(max(1, min_area))
            ma_k = 5 if ma > 4 else 3

            dy_off = torch.arange(-half, half + 1, device=dev, dtype=torch.long)
            dx_off = torch.arange(-half, half + 1, device=dev, dtype=torch.long)
            yyf = dy_off.to(torch.float32).reshape(1, -1, 1)
            xxf = dx_off.to(torch.float32).reshape(1, 1, -1)

            n_found = 0
            for y0 in range(0, H, bh):
                y1 = min(y0 + bh, H)
                ry0 = max(0, y0 - halo)
                ry1 = min(H, y1 + halo)

                band = xt[:, :, ry0:ry1, :]
                bvalid = valid[:, :, ry0:ry1, :]

                det = (band > thr) & bvalid
                if ma > 1:
                    cnt = F.avg_pool2d(
                        det.to(band.dtype), ma_k, stride=1, padding=ma_k // 2
                    ) * float(ma_k * ma_k)
                    # avg_pool2d sums in float32 then scales, so an exact count
                    # of `ma` can land at 4.999999. Compare against ma - 0.5.
                    det = det & (cnt >= float(ma) - 0.5)
                    del cnt

                mx = F.max_pool2d(band, 2 * sep_r + 1, stride=1, padding=sep_r)
                peaks = det & (band >= mx)
                del det, mx

                # restrict to the band proper (halo rows belong to neighbours)
                lo = y0 - ry0
                hi = lo + (y1 - y0)
                pk2 = peaks[0, 0]
                pk2[:lo, :] = False
                if hi < pk2.shape[0]:
                    pk2[hi:, :] = False

                bw = pk2.shape[1]
                flat = torch.nonzero(pk2.reshape(-1), as_tuple=False).reshape(-1)
                del peaks, pk2
                if flat.numel() == 0:
                    del band, bvalid
                    continue

                b2 = band[0, 0]
                vals = b2.reshape(-1)[flat]

                cap = int(max_peaks)
                if max_stars:
                    cap = min(cap, max(4 * int(max_stars), 2000))
                if flat.numel() > cap:
                    keep = torch.topk(vals, cap).indices
                    flat = flat[keep]
                    vals = vals[keep]
                    del keep

                iy = torch.div(flat, bw, rounding_mode="floor")
                ix = flat - iy * bw

                # ── batched 2nd-moment centroids ─────────────────────────────
                bp = F.pad(
                    b2.reshape(1, 1, b2.shape[0], b2.shape[1]),
                    (half, half, half, half), mode="constant", value=0.0,
                )[0, 0]
                YY = iy.reshape(-1, 1, 1) + half + dy_off.reshape(1, -1, 1)
                XX = ix.reshape(-1, 1, 1) + half + dx_off.reshape(1, 1, -1)
                # Weight the moments by flux ABOVE THE DETECTION THRESHOLD, not
                # by every non-negative pixel in the stamp. SEP takes moments
                # over the detected segment; an 11x11 box around a star in a
                # crowded field also contains its neighbours' wings, and
                # weighting those pulls the centroid off by a few hundredths of
                # a pixel — which is the whole budget when drizzle wants
                # < 0.2 px. Subtracting the threshold reproduces
                # "moments over the above-threshold footprint" cheaply.
                st = (bp[YY, XX] - thr).clamp_min(0.0)
                del bp, YY, XX

                s0 = st.sum(dim=(1, 2))
                s0c = s0.clamp_min(1e-12)
                xbar = (st * xxf).sum(dim=(1, 2)) / s0c
                ybar = (st * yyf).sum(dim=(1, 2)) / s0c
                dxc = xxf - xbar.reshape(-1, 1, 1)
                dyc = yyf - ybar.reshape(-1, 1, 1)
                sxx = (st * dxc * dxc).sum(dim=(1, 2)) / s0c
                syy = (st * dyc * dyc).sum(dim=(1, 2)) / s0c
                sxy = (st * dxc * dyc).sum(dim=(1, 2)) / s0c
                del st, dxc, dyc

                tr = 0.5 * (sxx + syy)
                rad = (((sxx - syy) * 0.5) ** 2 + sxy * sxy).clamp_min(0.0).sqrt()
                a2 = (tr + rad).clamp_min(1e-12)
                b2e = (tr - rad).clamp_min(0.0)
                ecc = 1.0 - (b2e / a2).sqrt()
                fwhm = 2.3548 * tr.clamp_min(0.0).sqrt()

                keep = (s0 > 0)
                # A centroid that ran far from its peak means a blend or an
                # edge artefact; those make bad, non-repeatable control points.
                keep = keep & (xbar.abs() <= half * 0.6) & (ybar.abs() <= half * 0.6)
                if float(min_fwhm) > 0.0:
                    keep = keep & (fwhm >= float(min_fwhm))
                if float(max_ellipticity) < 1.0:
                    keep = keep & (ecc <= float(max_ellipticity))

                if not bool(keep.any()):
                    del band, bvalid, b2, flat, vals, iy, ix
                    continue

                cx = (ix.to(torch.float32) + xbar)[keep]
                cy = (iy.to(torch.float32) + ybar + float(ry0))[keep]

                cx_l.append(cx.to("cpu").numpy())
                cy_l.append(cy.to("cpu").numpy())
                pk_l.append(vals[keep].to("cpu").numpy())
                fl_l.append(s0[keep].to("cpu").numpy())
                fw_l.append(fwhm[keep].to("cpu").numpy())
                ec_l.append(ecc[keep].to("cpu").numpy())
                n_found += int(cx.numel())

                del band, bvalid, b2, flat, vals, iy, ix, s0, xbar, ybar
                del sxx, syy, sxy, tr, rad, a2, b2e, ecc, fwhm, keep, cx, cy

            del xt, valid

    except Exception as e:
        log(f"⚠️ GPU star detect failed ({type(e).__name__}: {e}); using SEP.")
        try:
            _free()
        except Exception:
            pass
        return None

    if not cx_l:
        return None

    x = np.concatenate(cx_l).astype(np.float64)
    y = np.concatenate(cy_l).astype(np.float64)
    flux = np.concatenate(fl_l).astype(np.float64)
    peak = np.concatenate(pk_l).astype(np.float64)
    fwhm = np.concatenate(fw_l).astype(np.float64)
    ecc = np.concatenate(ec_l).astype(np.float64)

    # flux-descending first, so "keep the first" means "keep the brightest".
    order = np.argsort(-flux, kind="stable")
    x, y, flux, peak, fwhm, ecc = (
        x[order], y[order], flux[order], peak[order], fwhm[order], ecc[order]
    )

    # De-duplicate within min_separation, exactly, by distance.
    #
    # The max-pool already guarantees no two peaks within sep_r *inside one
    # band*, so what is left is (a) the halo overlap between adjacent bands and
    # (b) saturated cores, where `band >= mx` is true at every pixel of the flat
    # plateau and one star emits a cluster of peaks. A rounding grid misses both
    # — a plateau wider than one cell straddles cells, and two peaks 1 px apart
    # can round into different cells. Duplicate control points are poison for
    # the triangle stage: they make degenerate triangles and split the vertex
    # vote. The point list here is small (a few thousand after the cap), so an
    # exact radius query costs microseconds and removes the whole class of bug.
    if len(x) > 1:
        try:
            from scipy.spatial import cKDTree
            pts = np.column_stack([x, y])
            tree = cKDTree(pts)
            drop = np.zeros(len(x), dtype=bool)
            # pairs come back as (i, j) with i < j, i.e. i is the brighter one.
            # query_pairs does not promise an order, so sort them: the greedy
            # pass below is order-dependent and detection must be repeatable
            # frame to frame or the matcher sees a different star list each time.
            pr = tree.query_pairs(float(sep_r), output_type="ndarray")
            if len(pr):
                pr = pr[np.lexsort((pr[:, 1], pr[:, 0]))]
                for i, j in pr:
                    if not drop[i]:
                        drop[j] = True
            keep_idx = np.flatnonzero(~drop)
        except Exception:
            # scipy missing/failed: fall back to the cheap grid rather than
            # returning unfiltered duplicates
            gy = np.rint(y / float(sep_r)).astype(np.int64)
            gx = np.rint(x / float(sep_r)).astype(np.int64)
            span = int(np.max(gx)) - int(np.min(gx)) + 3
            gkey = gy * span + (gx - int(np.min(gx)) + 1)
            _, keep_idx = np.unique(gkey, return_index=True)
            keep_idx = np.sort(keep_idx)
        x, y, flux, peak, fwhm, ecc = (
            x[keep_idx], y[keep_idx], flux[keep_idx], peak[keep_idx],
            fwhm[keep_idx], ecc[keep_idx]
        )

    if max_stars:
        n = int(max_stars)
        x, y, flux, peak, fwhm, ecc = (
            x[:n], y[:n], flux[:n], peak[:n], fwhm[:n], ecc[:n]
        )

    log(f"⭐ GPU detect: {len(x)} star(s) "
        f"(rms={globalrms:.5g}, thr={thr:.5g}, raw peaks={n_found})")

    return {
        "x": x, "y": y, "flux": flux, "peak": peak,
        "fwhm": fwhm, "ecc": ecc,
        "globalrms": float(globalrms), "threshold": float(thr),
    }


def detect_stars(image2d, **kw):
    """
    Drop-in for ``astroalign._find_sources``: (N,2) float array of (x, y)
    sorted by flux descending, or ``None`` when the GPU path is unavailable.
    """
    res = detect_stars_full(image2d, **kw)
    if res is None:
        return None
    if res["x"].size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack([res["x"], res["y"]])


def warmup(status_cb=None) -> bool:
    """
    Force CUDA context creation and kernel compilation up front.

    The first detection in a process pays the whole backend start-up cost —
    measured at ~6.5 s on a CUDA box against ~13 ms steady state. Called once
    at worker start-up, that cost lands somewhere visible instead of on the
    first frame, where it looks like a hang.

    IMPORTANT for registration: ``star_alignment._make_executor`` spawns up to
    ``max(2, min(cpu_count, 32))`` processes and each one calls
    ``find_transform`` -> ``_find_sources``, so each would build its OWN CUDA
    context — several hundred MB of VRAM and seconds of start-up apiece. With
    the GPU detector active the right worker count is far lower than with SEP,
    because the bottleneck has moved off the CPU. See tests/gpu_pool_probe.py,
    which measures the crossover on the actual card.

    Returns True if the backend is live afterwards.
    """
    log = status_cb or (lambda *_: None)
    torch, dev = _resolve()
    if torch is None or dev is None:
        log("gpu_star_detect: no torch backend; registration will use SEP.")
        return False
    try:
        rng = np.random.default_rng(0)
        img = rng.normal(100.0, 3.0, size=(256, 256)).astype(np.float32)
        img[64, 64] += 5000.0
        img[150, 90] += 3000.0
        detect_stars_full(img, detection_sigma=5.0, min_area=5,
                          allow_cpu=True, mesh=32, band_rows=256)
        log(backend_note())
        return True
    except Exception as e:
        log(f"gpu_star_detect: warmup failed ({type(e).__name__}: {e}); "
            f"registration will use SEP.")
        return False


def _free():
    """Best-effort VRAM release (matches free_torch_memory's intent)."""
    torch, _ = _resolve()
    if torch is None:
        return
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Self-check:  python -m setiastro.saspro.gpu_star_detect
# Compares GPU detections against SEP on a synthetic field, reports the
# centroid agreement for matched stars. This is the number that matters —
# alignment cares about repeatability, not completeness.
# ─────────────────────────────────────────────────────────────────────────────
def _selftest(n_stars=400, H=1200, W=1600, seed=7, sigma=5.0):
    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 3.0, size=(H, W)).astype(np.float32)

    # smooth background gradient, so the mesh has something to do
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img += 20.0 * (xx / W) + 12.0 * (yy / H)

    tx = rng.uniform(30, W - 30, n_stars)
    ty = rng.uniform(30, H - 30, n_stars)
    amp = rng.uniform(80, 4000, n_stars)
    s = 1.6
    for x0, y0, A in zip(tx, ty, amp):
        i0, i1 = int(y0) - 8, int(y0) + 9
        j0, j1 = int(x0) - 8, int(x0) + 9
        gy, gx = np.mgrid[i0:i1, j0:j1].astype(np.float32)
        img[i0:i1, j0:j1] += A * np.exp(
            -(((gx - x0) ** 2 + (gy - y0) ** 2) / (2 * s * s))
        )

    print("[selftest] " + backend_note())
    truth = np.column_stack([tx, ty])
    from scipy.spatial import cKDTree
    ttree = cKDTree(truth)

    out = detect_stars_full(img, detection_sigma=sigma, min_area=5,
                            max_stars=None, status_cb=print, allow_cpu=True)
    if out is None:
        print("[selftest] FAIL: detector returned None (no torch at all?)")
        return False

    gx_, gy_ = out["x"], out["y"]
    gpts = np.column_stack([gx_, gy_])
    d, _ = ttree.query(gpts, k=1)
    ok = d < 2.0
    print(f"[selftest] GPU found {len(gx_)} of {n_stars} injected; "
          f"{int(ok.sum())} within 2 px")
    if ok.any():
        print(f"[selftest] centroid error vs truth: "
              f"median={np.median(d[ok]):.4f} px  "
              f"p95={np.percentile(d[ok], 95):.4f} px  "
              f"max={d[ok].max():.4f} px")
    print(f"[selftest] median FWHM={np.median(out['fwhm']):.3f} px  "
          f"(injected ~{2.3548 * s:.3f})  "
          f"median ecc={np.median(out['ecc']):.4f}")

    # flux ordering must be monotonically descending — the matcher's triangle
    # stage takes the brightest N, so a wrong order changes which stars match
    fl = out["flux"]
    ordered = bool(np.all(np.diff(fl) <= 0))
    print(f"[selftest] flux descending: {ordered}")

    # ── the verdict that actually matters: do we agree with SEP? ──────────
    verdict = True
    recall = float(ok.sum()) / max(1, n_stars)
    if ok.any():
        med = float(np.median(d[ok]))
        if med > 0.20:
            print(f"[selftest] FAIL: median centroid error {med:.4f} px "
                  f"exceeds the 0.2 px drizzle budget")
            verdict = False
    if recall < 0.90:
        print(f"[selftest] FAIL: recall {recall:.1%} below 90%")
        verdict = False
    if not ordered:
        print("[selftest] FAIL: flux not sorted descending")
        verdict = False

    try:
        import sep, time
        im = np.ascontiguousarray(img.astype(np.float32))
        t0 = time.perf_counter()
        bkg = sep.Background(im)
        src = sep.extract(im - bkg.back(), sigma * bkg.globalrms, minarea=5)
        t_sep = time.perf_counter() - t0

        detect_stars_full(img, detection_sigma=sigma, min_area=5,
                          allow_cpu=True)          # warm
        t0 = time.perf_counter()
        detect_stars_full(img, detection_sigma=sigma, min_area=5,
                          allow_cpu=True)
        t_gpu = time.perf_counter() - t0

        spts = np.column_stack([src["x"], src["y"]])
        ds, _ = ttree.query(spts, k=1)
        oks = ds < 2.0
        print(f"[selftest] SEP  found {len(spts)} of {n_stars} injected; "
              f"{int(oks.sum())} within 2 px")
        if oks.any():
            print(f"[selftest] SEP centroid error vs truth: "
                  f"median={np.median(ds[oks]):.4f} px  "
                  f"p95={np.percentile(ds[oks], 95):.4f} px")

        # cross-match the two detectors against each other
        if len(spts) and len(gpts):
            dxm, _ = cKDTree(spts).query(gpts, k=1)
            agree = dxm < 1.0
            print(f"[selftest] GPU<->SEP: {int(agree.sum())}/{len(gpts)} "
                  f"within 1 px; median offset "
                  f"{np.median(dxm[agree]) if agree.any() else float('nan'):.4f} px")

        print(f"[selftest] timing: SEP {t_sep*1000:.1f} ms  |  "
              f"torch[{_DEVICE}] {t_gpu*1000:.1f} ms  "
              f"({t_sep/max(t_gpu,1e-9):.2f}x)")
        if str(getattr(_DEVICE, 'type', '')) == 'cpu':
            print("[selftest] NOTE: timing above is CPU torch, which is the "
                  "slow path by design — it proves correctness, not speed. "
                  "Run on the GPU box for the real number.")
    except Exception as e:
        print(f"[selftest] SEP comparison skipped: {type(e).__name__}: {e}")

    print(f"[selftest] {'PASS' if verdict else 'FAIL'}")
    return verdict


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if _selftest() else 1)