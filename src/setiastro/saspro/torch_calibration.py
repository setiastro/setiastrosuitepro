# setiastro/saspro/torch_calibration.py
from __future__ import annotations
import numpy as np
from typing import Optional

# Lazy torch — only imported when first needed
_torch = None
_device = None

def _get_torch_and_device():
    global _torch, _device
    if _torch is not None:
        return _torch, _device
    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(
            prefer_cuda=True,
            prefer_xpu=False,
            prefer_dml=True,
            status_cb=lambda s: None,
        )
        # Pick best available device
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (hasattr(torch, "backends")
              and hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available()):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _torch = torch
        _device = device
        return _torch, _device
    except Exception:
        return None, None


def gpu_available() -> bool:
    torch, device = _get_torch_and_device()
    if torch is None or device is None:
        return False
    return str(device.type) in ("cuda", "mps")


def _to_tensor(arr: np.ndarray, torch, device):
    """Convert numpy array to float32 torch tensor on device."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(a).to(device, dtype=torch.float32)


def _to_numpy(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def subtract_dark_gpu(
    light: np.ndarray,
    dark: np.ndarray,
    pedestal: float = 0.0,
) -> Optional[np.ndarray]:
    """
    GPU dark subtraction: light - dark + pedestal.
    Returns None if GPU unavailable (caller falls back to Numba).
    light/dark: any of (H,W), (C,H,W), (H,W,C) — matched shapes expected.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    try:
        t_light = _to_tensor(light, torch, device)
        t_dark  = _to_tensor(dark,  torch, device)

        # broadcast handles (C,H,W) light with (H,W) dark etc.
        result = t_light - t_dark + float(pedestal)
        result = torch.clamp(result, 0.0, float("inf"))

        out = _to_numpy(result)
        return out.reshape(light.shape)
    except Exception:
        return None


def divide_flat_gpu(
    light: np.ndarray,
    flat: np.ndarray,
) -> Optional[np.ndarray]:
    """
    GPU flat division. flat should already be normalized (median=1.0 per channel).
    Returns None if GPU unavailable.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    try:
        t_light = _to_tensor(light, torch, device)
        t_flat  = _to_tensor(flat,  torch, device)

        # Guard against zeros/nan in flat
        t_flat = torch.where(
            torch.isfinite(t_flat) & (t_flat > 0),
            t_flat,
            torch.ones_like(t_flat),
        )

        result = t_light / t_flat
        out = _to_numpy(result)
        return out.reshape(light.shape)
    except Exception:
        return None


def cosmetic_correction_gpu(
    image: np.ndarray,
    hot_sigma: float = 5.0,
    cold_sigma: float = 5.0,
    sat_quantile: float = 0.9995,
) -> Optional[np.ndarray]:
    """
    GPU cosmetic correction using unfold for 3×3 neighborhood stats.
    Computes ring median via sorting the 8 neighbors, then MAD sigma.
    Star guard and cold cluster guard are NOT applied here (they require
    conditional branching that's expensive on GPU) — use this for clean
    sensors where hot/cold pixels are isolated.
    Returns None if GPU unavailable or image is not 2D mono.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    # Only handle 2D for now — color is per-channel via Numba anyway
    if image.ndim not in (2, 3):
        return None

    try:
        was_2d = (image.ndim == 2)
        if was_2d:
            arr = image[None, None]  # (1,1,H,W)
        else:
            # CHW
            if image.shape[0] == 3:
                arr = image[None]    # (1,3,H,W)
            else:
                return None          # HWC — don't handle here

        t = _to_tensor(arr, torch, device)  # (1,C,H,W)
        _, C, H, W = t.shape

        # unfold: extract 3×3 patches → (1, C, H, W, 9)
        padded = torch.nn.functional.pad(t, (1,1,1,1), mode="reflect")
        patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        # patches shape: (1, C, H, W, 3, 3) → flatten last two → (1,C,H,W,9)
        patches = patches.reshape(1, C, H, W, 9)

        # center pixel index is 4 in the 3×3 (row-major: (1,1)=4)
        center = patches[..., 4:5]          # (1,C,H,W,1)

        # ring = all 8 neighbors (exclude center index 4)
        ring = torch.cat([patches[..., :4], patches[..., 5:]], dim=-1)  # (1,C,H,W,8)

        # ring median via sort
        ring_sorted, _ = torch.sort(ring, dim=-1)
        ring_median = (ring_sorted[..., 3] + ring_sorted[..., 4]) / 2.0  # median of 8

        # MAD
        mad = torch.median(torch.abs(ring - ring_median.unsqueeze(-1)), dim=-1).values
        sigma = 1.4826 * mad + 1e-8

        # saturation threshold per channel
        result = t.clone()
        for c in range(C):
            plane = t[0, c]
            sat_val = float(torch.quantile(plane.float(), sat_quantile).item())

            hot_mask  = (center[0, c, :, :, 0] > ring_median[0, c] + hot_sigma  * sigma[0, c])
            cold_mask = (center[0, c, :, :, 0] < ring_median[0, c] - cold_sigma * sigma[0, c])
            sat_mask  = (plane >= sat_val)

            # don't touch saturated pixels
            hot_mask  = hot_mask  & ~sat_mask
            cold_mask = cold_mask & ~sat_mask

            result[0, c] = torch.where(
                hot_mask | cold_mask,
                ring_median[0, c],
                plane,
            )

        out = _to_numpy(result[0])  # (C,H,W) or (1,H,W)
        if was_2d:
            return out[0]   # back to (H,W)
        return out          # (C,H,W)

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GPU Bayer demosaic (RCD-N / bilinear / sparse CFA)
#
# Vectorized torch port of the numba kernels in legacy/numba_utils.py. Every
# phase is the same stencil arithmetic expressed as shifted tensor slices, so
# the index algebra is a line-for-line mirror of the loop bodies rather than a
# reimplementation.
#
# Returns None — meaning "caller falls back to Numba" — for any method not
# ported here (vng, ahd, edge, strict_cfa), for non-2D input, and whenever the
# GPU is unavailable or anything raises. That keeps this strictly additive: a
# machine where the GPU path is slower or broken behaves exactly as before.
#
# Accuracy note: the numba kernels promote to float64 internally (a Python
# float literal times a float32 array element is float64 under numba's numpy
# promotion rules) and then store back to float32. This runs in float32 on the
# GPU, which is the whole point, so results differ from numba by a small
# relative epsilon. Pass dtype64=True to compute in float64 and reproduce the
# numba arithmetic for verification.
# ─────────────────────────────────────────────────────────────────────────────

import threading as _threading

_CFA_CHAN_T = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}

# The normalization stage debayers several frames concurrently (one per worker
# thread), but there is one GPU. Each in-flight RCD frame holds the mosaic, the
# bilinear base, the output and a handful of full-frame intermediates per phase
# -- roughly 10-12x the mosaic's bytes, so ~1.2 GB for a 26 Mpx frame. Letting
# N workers do that at once is how you turn a 8 GB card into a PCIe thrash on
# Windows (it does not raise, it pages and hangs). Serializing device work
# keeps peak VRAM at one frame's worth; the CPU threads carry on with loading,
# background matching and writing meanwhile, which is where the overlap win
# actually comes from.
_GPU_DEBAYER_LOCK = _threading.Lock()

# (The old _GPU_DEBAYER_VRAM_FACTOR lived here. It encoded "RCD needs about
# 14x the mosaic's bytes", which is the same fact _PLANES_PER_METHOD now
# states per method -- and AHD needs more than RCD, so a single global
# multiple cannot be right for both. The factor is gone; callers compute what
# they need from the planes table and pass bytes.)


def _resolve_gpu_device():
    """(torch, device) resolved the way the GPU modules that work resolve it.

    _get_torch_and_device below goes straight to runtime_torch.import_torch().
    gpu_warp._resolve() and gpu_star_detect._resolve() deliberately do not:
    they first call add_runtime_to_sys_path(), which stages SASpro's own torch
    runtime and demotes a shadowing CPU-only wheel, and only fall back to
    import_torch (which can shell out to pip) as a last resort. gpu_warp's own
    comment says what skipping that costs: "a CPU-only wheel sitting in the
    project venv wins over the CUDA build SASpro went to the trouble of
    installing ... nothing here accelerates anything while still reporting
    success."

    They also choose the device with runtime_torch.best_device(prefer_cuda,
    prefer_dml) and accept any non-CPU type, so torch-directml
    (privateuseone) counts; the resolver below knows only cuda and mps.

    That difference is exactly the state a machine is in when the anti-ring
    warp finds a card and this module reports no device.

    This deliberately NEVER calls _get_torch_and_device(), not even as a
    fallback. That function calls runtime_torch.import_torch(), which can shell
    out to pip and install packages -- observed doing exactly that while
    probing for a device. gpu_warp keeps it as a last resort for the same
    reason ("can shell out to pip and hang a spawned worker"), and the demosaic
    is reached from inside the normalization worker threads, which is the worst
    possible place for that. If the proven resolvers cannot find a non-CPU
    device, the honest answer is "no GPU" and the numba kernel runs.
    """
    import importlib
    _fallback = (None, None)
    for _name in ("gpu_star_detect", "gpu_warp"):
        try:
            _m = importlib.import_module(f"setiastro.saspro.{_name}")
            _t, _d = _m._resolve()
            if _t is None or _d is None:
                continue
            if str(_d.type) != "cpu":
                return _t, _d
            # Remember a CPU resolution so backend_note can say "+cpu build"
            # rather than the much less useful "torch unavailable".
            if _fallback == (None, None):
                _fallback = (_t, _d)
        except Exception:
            pass
    if _fallback != (None, None):
        return _fallback

    # Neither helper module was importable. Try a plain import and the shared
    # device picker; both are install-free.
    try:
        import torch as _t
    except Exception:
        return None, None
    try:
        from setiastro.saspro.runtime_torch import best_device
        _d = best_device(_t, prefer_cuda=True, prefer_dml=True)
    except Exception:
        _d = None
    if _d is None:
        try:
            if hasattr(_t, "cuda") and _t.cuda.is_available():
                _d = _t.device("cuda")
            elif (getattr(getattr(_t, "backends", None), "mps", None)
                  and _t.backends.mps.is_available()):
                _d = _t.device("mps")
            else:
                _d = _t.device("cpu")
        except Exception:
            _d = _t.device("cpu")
    return _t, _d


def _gpu_usable_bytes(torch, device):
    """Device memory we could actually use, or None when unknowable.

    Driver-free memory alone is the wrong number: torch's caching allocator
    holds blocks it has reserved but is not currently using, and it will hand
    those back to us without going to the driver. Ignoring them made the
    check refuse a frame the card had room for -- which is what happens on the
    demosaic stage, since the warp stage ran first and left a reserve.

    Only CUDA reports this. MPS and DirectML do not, so they return None and
    rely on the OOM path instead.
    """
    try:
        if str(device.type) != "cuda":
            return None
        free, _total = torch.cuda.mem_get_info(device)
        try:
            reclaimable = max(0, int(torch.cuda.memory_reserved(device))
                              - int(torch.cuda.memory_allocated(device)))
        except Exception:
            reclaimable = 0
        return int(free) + reclaimable
    except Exception:
        return None


def _gpu_has_room(torch, device, need_bytes, status_cb=None) -> bool:
    """True when the device plausibly has room for `need_bytes`.

    Releases cached blocks and asks again before refusing, so a reserve left
    by an earlier stage does not look like a full card.

    Takes the figure already worked out by the caller rather than a multiple
    of the mosaic -- what one demosaic needs depends on the method, and since
    the work is now split into row strips it is a question about one strip.
    """
    need = float(need_bytes)
    have = _gpu_usable_bytes(torch, device)
    if have is None or have >= need:
        return True
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    have = _gpu_usable_bytes(torch, device)
    if have is not None and have < need:
        if status_cb:
            status_cb(f"needs ~{need / 2**20:.0f} MB on the device, "
                      f"{have / 2**20:.0f} MB usable")
        return False
    return True

# Methods this module implements. Anything else returns None so the Numba
# kernel handles it.
#
# "bilinear" and "edge" are deliberately NOT here. Their numba kernels
# (_bilinear_interpolate_numba, _edge_aware_interpolate_numba) fill holes by
# averaging the non-zero neighbours of the array they are concurrently writing
# into, so each pixel's average is contaminated by neighbours already filled
# earlier in the same pass. The result is order-dependent by construction and
# cannot be reproduced by a gather, which is the only form a vectorized or GPU
# implementation can take. Measured against an order-independent gather over
# the same samples: 67% of pixels differ, up to 0.17 absolute. Rather than
# ship a GPU path that disagrees with the CPU path, these decline and fall
# back to numba. (RCD, VNG and AHD all build from the immutable mosaic and are
# order-independent -- hence portable. VNG is the one still unported: its
# gradient terms are table-driven per Bayer phase, which is a data-dependent
# gather rather than a stencil.)
GPU_DEBAYER_METHODS = ("rcd", "ahd")


def _bayer_masks_t(bp: str, H: int, W: int, torch, device):
    """R/G/B site masks for a Bayer phase. Mirrors _rcd_build_masks.

    Built by OR-ing row/column parity vectors rather than by indexing a colour
    table with a full-frame phase map. The table version allocated TWO
    (H,W) int64 arrays -- the phase map and colors[phase] -- which is 8 bytes
    per pixel each: 714 MB on a 46.8 Mpx frame, before any demosaic work, on a
    card that also has to hold the output and the per-phase intermediates.
    Each channel takes 1 or 2 of the 4 phases, so this peaks at two (H,W) bool
    arrays instead.
    """
    t = _CFA_CHAN_T[bp]
    ye = ((torch.arange(H, device=device) & 1) == 0).view(H, 1)
    xe = ((torch.arange(W, device=device) & 1) == 0).view(1, W)
    yo, xo = ~ye, ~xe
    out = []
    for c in range(3):
        m = None
        for p in range(4):
            if t[p] != c:
                continue
            part = (ye if p < 2 else yo) & (xe if p % 2 == 0 else xo)
            m = part if m is None else (m | part)
        if m is None:    # a pattern with no sites of this colour cannot happen
            m = torch.zeros((H, W), dtype=torch.bool, device=device)
        out.append(m)
    return out[0], out[1], out[2]


def _bilinear_prefill_t(im, masks, torch):
    """Full (H,W,3) bilinear demosaic. Mirrors _vng_bilinear_prefill.

    The numba version averages, per channel, the 3x3 neighbours that carry
    that channel, counting only in-bounds pixels, and uses the raw sample
    itself at a site of its own colour. A zero-padded 3x3 box sum over the
    masked samples divided by the same box sum over the mask is identical:
    out-of-bounds contributes 0 to both numerator and count, exactly like the
    loop's bounds check.
    """
    Fn = torch.nn.functional
    H, W = im.shape
    k = torch.ones((1, 1, 3, 3), dtype=im.dtype, device=im.device)
    # Write each channel into the final buffer as it is computed. Collecting
    # three planes in a list and then stacking them held six full-frame planes
    # at once (the three planes plus the stack's three), which is 1.1 GB at
    # 46.8 Mpx for no reason.
    out = torch.empty((H, W, 3), dtype=im.dtype, device=im.device)
    for c, m in enumerate(masks):
        mf = m.to(im.dtype)
        s = Fn.conv2d((im * mf).view(1, 1, H, W), k, padding=1).view(H, W)
        n = Fn.conv2d(mf.view(1, 1, H, W), k, padding=1).view(H, W)
        del mf
        # Where no neighbour carried this colour, s is 0 and n is 0, so
        # dividing by the clamped count yields 0 -- the same value the
        # explicit torch.where produced, without the extra buffers.
        s.div_(n.clamp_(min=1.0))
        del n
        out[..., c] = torch.where(m, im, s)
        del s
    return out


def _sparse_cfa_t(im, masks, torch):
    """Sample lay-down with no interpolation (the cfa_drizzle path)."""
    H, W = im.shape
    out = torch.zeros((H, W, 3), dtype=im.dtype, device=im.device)
    for c, m in enumerate(masks):
        out[..., c] = torch.where(m, im, torch.zeros_like(im))
    return out


def estimate_mosaic_noise_sigma_t(im, g_mask, torch) -> float:
    """Robust per-frame mosaic noise sigma for RCD-N (Laplacian MAD on green).

    Mirrors _estimate_mosaic_noise_sigma, with one deliberate difference: the
    subsample stride is forced ODD.

    The numba version walks the full index grid at stride s and keeps points
    that land on a green site. A constant EVEN stride only ever visits CFA
    phase 0 (y&1 == 0 and x&1 == 0), which is R for RGGB and B for BGGR — so
    no green sample is collected, `samples` comes back empty and the function
    returns 0.0, silently disabling the noise-floor subtraction for the whole
    frame. s = int(sqrt(total_green / 250000)) lands on an even number for
    many common sensor sizes (IMX571 bin1, IMX533, IMX294, IMX183, and most
    full-frame DSLRs). For GRBG/GBRG, whose phase 0 IS green, the same stride
    instead collects every grid point and overshoots the 250k sample target by
    ~4x. Forcing the stride odd makes both sub-lattices reachable and keeps the
    sample count near target for every pattern.
    """
    H, W = im.shape
    if H < 5 or W < 5:
        return 0.0
    lap = (4.0 * im[2:-2, 2:-2]
           - im[0:-4, 2:-2] - im[4:, 2:-2]
           - im[2:-2, 0:-4] - im[2:-2, 4:])
    gm = g_mask[2:-2, 2:-2]
    total_green = int(gm.sum().item())
    if total_green <= 0:
        return 0.0
    target = 250_000
    s = 1
    if total_green > target:
        s = int((float(total_green) / target) ** 0.5)
        if s < 1:
            s = 1
        if s % 2 == 0:
            s += 1
    v = lap[::s, ::s][gm[::s, ::s]]
    if v.numel() == 0:
        return 0.0
    med = v.median()
    mad = (v - med).abs().median()
    sigma = float(mad.item()) * 1.4826 / 4.4721
    if sigma != sigma or sigma < 0.0:   # NaN or negative
        return 0.0
    return sigma


def _rcd_green_at_rb_t(im, out, rb_mask, noise_sigma, denoise_k, torch):
    """Phase 1 — G at R/B sites. Mirrors _rcd_green_at_rb over [2:-2, 2:-2]."""
    eps = 1e-10
    noise_floor = denoise_k * (1.414 + 2.449) * noise_sigma
    c = im[2:-2, 2:-2]
    xm1 = im[2:-2, 1:-3]; xp1 = im[2:-2, 3:-1]
    xm2 = im[2:-2, 0:-4]; xp2 = im[2:-2, 4:]
    ym1 = im[1:-3, 2:-2]; yp1 = im[3:-1, 2:-2]
    ym2 = im[0:-4, 2:-2]; yp2 = im[4:, 2:-2]

    G_H = 0.5 * (xm1 + xp1) + 0.25 * (2.0 * c - xm2 - xp2)
    G_V = 0.5 * (ym1 + yp1) + 0.25 * (2.0 * c - ym2 - yp2)
    D_H = (xm1 - xp1).abs() + (xm2 - 2.0 * c + xp2).abs()
    D_V = (ym1 - yp1).abs() + (ym2 - 2.0 * c + yp2).abs()
    if noise_floor > 0.0:
        D_H = (D_H - noise_floor).clamp_min(0.0)
        D_V = (D_V - noise_floor).clamp_min(0.0)
    w_H = 1.0 / (D_H * D_H + eps)
    w_V = 1.0 / (D_V * D_V + eps)
    G = ((w_H * G_H + w_V * G_V) / (w_H + w_V)).clamp_min(0.0)

    m = rb_mask[2:-2, 2:-2]
    out[2:-2, 2:-2, 1] = torch.where(m, G, out[2:-2, 2:-2, 1])


def _rcd_rb_at_g_t(out, g_mask, r_mask, torch):
    """Phase 2 — R and B at G sites, difference form. Mirrors _rcd_rb_at_g.

    The numba loop probes r_mask[y, x-1] to learn which axis carries R; the
    same probe vectorizes to the left-shifted R mask. Both axis assignments
    are computed and selected, since a vectorized pass cannot branch.
    """
    gc = out[1:-1, 1:-1, 1]
    hL = out[1:-1, 0:-2, :]; hR = out[1:-1, 2:, :]
    vU = out[0:-2, 1:-1, :]; vD = out[2:, 1:-1, :]
    probe = r_mask[1:-1, 0:-2]

    # horizontal axis = R, vertical = B
    rA = gc + 0.5 * ((hL[..., 0] - hL[..., 1]) + (hR[..., 0] - hR[..., 1]))
    bA = gc + 0.5 * ((vU[..., 2] - vU[..., 1]) + (vD[..., 2] - vD[..., 1]))
    # horizontal axis = B, vertical = R
    rB = gc + 0.5 * ((vU[..., 0] - vU[..., 1]) + (vD[..., 0] - vD[..., 1]))
    bB = gc + 0.5 * ((hL[..., 2] - hL[..., 1]) + (hR[..., 2] - hR[..., 1]))

    r_est = torch.where(probe, rA, rB).clamp_min(0.0)
    b_est = torch.where(probe, bA, bB).clamp_min(0.0)

    m = g_mask[1:-1, 1:-1]
    out[1:-1, 1:-1, 0] = torch.where(m, r_est, out[1:-1, 1:-1, 0])
    out[1:-1, 1:-1, 2] = torch.where(m, b_est, out[1:-1, 1:-1, 2])


def _rcd_rb_at_br_t(out, r_mask, rb_mask, noise_sigma, denoise_k, torch):
    """Phase 3 — R at B sites and B at R sites. Mirrors _rcd_rb_at_br."""
    eps = 1e-10
    noise_floor = denoise_k * (1.414 + 2.449) * noise_sigma
    gc = out[2:-2, 2:-2, 1]
    nw = out[1:-3, 1:-3, :]; ne = out[1:-3, 3:-1, :]
    sw = out[3:-1, 1:-3, :]; se = out[3:-1, 3:-1, :]
    is_r = r_mask[2:-2, 2:-2]

    # The opposite colour: B (channel 2) at an R site, R (channel 0) at a B site
    nw_c = torch.where(is_r, nw[..., 2], nw[..., 0])
    ne_c = torch.where(is_r, ne[..., 2], ne[..., 0])
    sw_c = torch.where(is_r, sw[..., 2], sw[..., 0])
    se_c = torch.where(is_r, se[..., 2], se[..., 0])
    g_nw = nw[..., 1]; g_ne = ne[..., 1]
    g_sw = sw[..., 1]; g_se = se[..., 1]

    D_NESW = (ne_c - sw_c).abs() + (2.0 * gc - g_ne - g_sw).abs()
    D_NWSE = (nw_c - se_c).abs() + (2.0 * gc - g_nw - g_se).abs()
    if noise_floor > 0.0:
        D_NESW = (D_NESW - noise_floor).clamp_min(0.0)
        D_NWSE = (D_NWSE - noise_floor).clamp_min(0.0)
    w1 = 1.0 / (D_NESW * D_NESW + eps)
    w2 = 1.0 / (D_NWSE * D_NWSE + eps)
    c_NESW = gc + 0.5 * ((ne_c - g_ne) + (sw_c - g_sw))
    c_NWSE = gc + 0.5 * ((nw_c - g_nw) + (se_c - g_se))
    c_est = ((w1 * c_NESW + w2 * c_NWSE) / (w1 + w2)).clamp_min(0.0)

    m = rb_mask[2:-2, 2:-2]
    out[2:-2, 2:-2, 2] = torch.where(m & is_r, c_est, out[2:-2, 2:-2, 2])
    out[2:-2, 2:-2, 0] = torch.where(m & ~is_r, c_est, out[2:-2, 2:-2, 0])


def _rcd_clip_blend_t(im, clip_threshold, clip_feather, torch):
    """Soft RCD -> bilinear crossfade mask near clipped samples.

    Mirrors _rcd_build_clip_blend: a 7x7 dilation of (im >= threshold) then a
    Gaussian feather. max_pool2d with kernel 7 / padding 3 IS a 7x7 grey
    dilation on a 0/1 map, and the Gaussian is built with the same
    coefficients cv2.getGaussianKernel produces, applied separably with
    reflect padding to match cv2's default BORDER_REFLECT_101.
    Returns None when nothing is near clip (the numba fast path).
    """
    Fn = torch.nn.functional
    hard = (im >= clip_threshold)
    if not bool(hard.any()):
        return None
    H, W = im.shape
    x = hard.to(im.dtype).view(1, 1, H, W)
    x = Fn.max_pool2d(x, kernel_size=7, stride=1, padding=3)

    sigma = max(0.5, float(clip_feather))
    ksize = int(2 * round(3.0 * sigma) + 1)
    half = ksize // 2
    i = torch.arange(ksize, device=im.device, dtype=im.dtype) - half
    g = torch.exp(-(i * i) / (2.0 * sigma * sigma))
    g = g / g.sum()

    pad = min(half, max(H, W) - 1)
    x = Fn.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = Fn.conv2d(x, g.view(1, 1, 1, -1), padding=0)
    x = Fn.pad(x, (0, 0, pad, pad), mode="reflect")
    x = Fn.conv2d(x, g.view(1, 1, -1, 1), padding=0)
    return x.view(H, W).clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# AHD — Adaptive Homogeneity-Directed demosaic (Hirakawa & Parks 2005)
#
# Vectorized port of the _ahd_* numba kernels. AHD is portable for the same
# reason RCD is: every phase reads the immutable mosaic or an earlier phase's
# output, never its own, so a gather reproduces it exactly. (bilinear and edge
# are not portable for precisely the opposite reason.)
#
# Both of _ahd_fill_rb's loops were checked for this: the R/B-at-G loop reads
# R/B at R/B sites (prefill) and G at R/B sites (phase 1) and writes at G
# sites; the diagonal loop reads the opposite colour at diagonal R/B sites
# (prefill) and G there (phase 1) and writes at R/B sites. Disjoint, both ways.
#
# One deliberate difference from the numba version: hom_h_s / hom_v_s are
# allocated with np.empty_like there and the 3x3 box blur only writes the
# interior, so _ahd_select_winner compares UNINITIALISED memory on the 1-pixel
# border. It happens to be harmless -- nothing has written either candidate
# there, so they are identical and all three branches of the comparison give
# the same number -- but it is not deterministic. These start at zero.
# ─────────────────────────────────────────────────────────────────────────────


def _ahd_green_candidates_t(im, cand_h, cand_v, rb_mask, torch):
    """Phase 1 — Hamilton-Adams G at R/B sites, H and V separately.

    Mirrors _ahd_fill_green_candidates over [2:-2, 2:-2]. Same estimator as
    RCD phase 1 but without the gradient weighting: AHD keeps both directions
    and decides later, which is the whole point of the method.
    """
    c = im[2:-2, 2:-2]
    xm1 = im[2:-2, 1:-3]; xp1 = im[2:-2, 3:-1]
    xm2 = im[2:-2, 0:-4]; xp2 = im[2:-2, 4:]
    ym1 = im[1:-3, 2:-2]; yp1 = im[3:-1, 2:-2]
    ym2 = im[0:-4, 2:-2]; yp2 = im[4:, 2:-2]

    G_H = (0.5 * (xm1 + xp1) + 0.25 * (2.0 * c - xm2 - xp2)).clamp_min(0.0)
    G_V = (0.5 * (ym1 + yp1) + 0.25 * (2.0 * c - ym2 - yp2)).clamp_min(0.0)

    m = rb_mask[2:-2, 2:-2]
    cand_h[2:-2, 2:-2, 1] = torch.where(m, G_H, cand_h[2:-2, 2:-2, 1])
    cand_v[2:-2, 2:-2, 1] = torch.where(m, G_V, cand_v[2:-2, 2:-2, 1])


def _ahd_fill_rb_t(out, g_mask, r_mask, rb_mask, torch):
    """Phase 2 — R and B everywhere, difference form. Mirrors _ahd_fill_rb.

    Both of the numba function's loops run over [1:-1, 1:-1]. They are
    independent of each other (see the module comment above), so the order
    here does not matter either.
    """
    # --- R/B at G sites (perpendicular 4-neighbours) ---
    gc = out[1:-1, 1:-1, 1]
    hL = out[1:-1, 0:-2, :]; hR = out[1:-1, 2:, :]
    vU = out[0:-2, 1:-1, :]; vD = out[2:, 1:-1, :]
    probe = r_mask[1:-1, 0:-2]

    rA = gc + 0.5 * ((hL[..., 0] - hL[..., 1]) + (hR[..., 0] - hR[..., 1]))
    bA = gc + 0.5 * ((vU[..., 2] - vU[..., 1]) + (vD[..., 2] - vD[..., 1]))
    rB = gc + 0.5 * ((vU[..., 0] - vU[..., 1]) + (vD[..., 0] - vD[..., 1]))
    bB = gc + 0.5 * ((hL[..., 2] - hL[..., 1]) + (hR[..., 2] - hR[..., 1]))
    r_est = torch.where(probe, rA, rB).clamp_min(0.0)
    b_est = torch.where(probe, bA, bB).clamp_min(0.0)
    del rA, bA, rB, bB

    mg = g_mask[1:-1, 1:-1]
    r_new = torch.where(mg, r_est, out[1:-1, 1:-1, 0])
    b_new = torch.where(mg, b_est, out[1:-1, 1:-1, 2])
    del r_est, b_est

    # --- the opposite colour at R/B sites (4 diagonals) ---
    # Read before writing anything: the two loops are independent in the numba
    # version, and materialising these first keeps that true here.
    gc2 = out[1:-1, 1:-1, 1]
    nw = out[0:-2, 0:-2, :]; ne = out[0:-2, 2:, :]
    sw = out[2:, 0:-2, :]; se = out[2:, 2:, :]
    is_r = r_mask[1:-1, 1:-1]

    c_nw = torch.where(is_r, nw[..., 2], nw[..., 0])
    c_ne = torch.where(is_r, ne[..., 2], ne[..., 0])
    c_sw = torch.where(is_r, sw[..., 2], sw[..., 0])
    c_se = torch.where(is_r, se[..., 2], se[..., 0])
    c_est = (gc2 + 0.25 * ((c_nw - nw[..., 1]) + (c_ne - ne[..., 1])
                           + (c_sw - sw[..., 1]) + (c_se - se[..., 1]))
             ).clamp_min(0.0)
    del c_nw, c_ne, c_sw, c_se

    mrb = rb_mask[1:-1, 1:-1]
    out[1:-1, 1:-1, 0] = torch.where(mrb & ~is_r, c_est, r_new)
    out[1:-1, 1:-1, 2] = torch.where(mrb & is_r, c_est, b_new)
    del c_est, r_new, b_new


def _ahd_lab_t(rgb, torch):
    """Linear RGB (sRGB primaries, D65) -> CIELab. Mirrors
    _ahd_rgb_to_lab_linear, including feeding linear data straight in with no
    sRGB gamma inverse."""
    r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
    X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    Y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b)
    Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883

    delta = 6.0 / 29.0
    delta3 = delta * delta * delta
    three_delta2 = 3.0 * delta * delta
    four_29 = 4.0 / 29.0

    def _f(t):
        # torch.where evaluates both branches, and a cube root of a tiny
        # positive number is finite, so the clamp is only to keep pow() off
        # anything negative that round-off might produce.
        return torch.where(t > delta3,
                           t.clamp_min(0.0).pow(1.0 / 3.0),
                           t / three_delta2 + four_29)

    fx = _f(X); fy = _f(Y); fz = _f(Z)
    del X, Y, Z
    L = 116.0 * fy - 16.0
    a_ = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return torch.stack([L, a_, b_], dim=-1)


def _ahd_homogeneity_t(lab, ball, torch):
    """Phases 4 and 4b — 4-neighbour CIE76 homogeneity count, then the 3x3 box
    aggregate. Mirrors _ahd_homogeneity_map + _ahd_box_blur_3x3.

    Returns the 3x3 box SUM, deliberately NOT divided by 9.

    This matters, and it is the one place where copying the numba arithmetic
    literally gives the wrong answer. The counts are integers 0..4, so a box
    sum is an integer 0..36. Numba forms `s * (1.0/9.0)` from an exactly
    represented float32 integer, so two neighbourhoods with equal counts come
    out BIT-IDENTICAL -- an exact tie, which _ahd_select_winner resolves by
    averaging the two candidates. Ties are not a rare edge case: in any flat
    region both directions score 4/4 everywhere, so most of a calibrated sub
    ties.

    A conv2d with a 1/9 kernel accumulates in whatever order the kernel picks,
    so an equal-count tie can land 1 ULP apart. That is numerically
    negligible and behaviourally not: it turns "average the candidates" into
    "pick one", on roughly 3.5% of pixels, differing by the candidate spread
    (~1e-2) rather than by an epsilon. Measured, before this was fixed.

    Summing with a kernel of ones keeps every value an exact small integer in
    both float32 and float64, whatever order the accumulation takes, so ties
    stay exact. Dividing by 9 is then pointless: the comparison in
    _ahd_demosaic_t is monotone in the sum, and both maps are scaled alike.

    The 1-pixel border is ZERO, matching the numba homogeneity array's
    np.zeros init. Numba's SMOOTHED buffers are np.empty_like and their border
    is never written, so it reads uninitialized -- observed holding NaN. That
    is benign there only because phases 1 and 2 never touch the outer ring, so
    both candidates still hold the identical bilinear prefill and all three
    branches of the winner test return the same value. Zeros here are
    deterministic and give the same result.
    """
    Fn = torch.nn.functional
    H, W, _ = lab.shape
    ball2 = float(ball) * float(ball)
    cen = lab[1:-1, 1:-1, :]
    count = torch.zeros((H - 2, W - 2), dtype=lab.dtype, device=lab.device)
    for sl in (lab[0:-2, 1:-1, :], lab[2:, 1:-1, :],
               lab[1:-1, 0:-2, :], lab[1:-1, 2:, :]):
        d = sl - cen
        d2 = (d * d).sum(dim=-1)
        count += (d2 <= ball2).to(lab.dtype)
        del d, d2

    hom = torch.zeros((H, W), dtype=lab.dtype, device=lab.device)
    hom[1:-1, 1:-1] = count
    del count, cen

    k = torch.ones((1, 1, 3, 3), dtype=lab.dtype, device=lab.device)
    sm = Fn.conv2d(hom.view(1, 1, H, W), k, padding=0).view(H - 2, W - 2)
    out = torch.zeros((H, W), dtype=lab.dtype, device=lab.device)
    out[1:-1, 1:-1] = sm
    return out


def _ahd_demosaic_t(im, masks, ball, blend, bilinear, torch):
    """Full AHD on one slab. Mirrors _ahd_demosaic's phase order.

    `blend` and `bilinear` are passed in because the caller already builds
    them (the blend mask depends only on the mosaic, and the prefill is the
    shared starting point for both candidates AND the crossfade target).
    """
    r_mask, g_mask, b_mask = masks
    rb = r_mask | b_mask

    cand_h = bilinear.clone()
    cand_v = bilinear.clone()
    _ahd_green_candidates_t(im, cand_h, cand_v, rb, torch)
    _ahd_fill_rb_t(cand_h, g_mask, r_mask, rb, torch)
    _ahd_fill_rb_t(cand_v, g_mask, r_mask, rb, torch)

    # Lab -> homogeneity, one candidate at a time so only one Lab buffer is
    # resident. Two (H,W,3) Lab arrays at once is 6 full-frame planes, which
    # at 46.8 Mpx is a gigabyte that buys nothing.
    #
    # These are exact integer box sums, not averages -- see
    # _ahd_homogeneity_t. Comparing sums is monotonically the same test as
    # comparing sum/9, and it keeps an equal-count tie an EXACT tie, which is
    # what sends the pixel to the averaging branch below.
    hom_h = _ahd_homogeneity_t(_ahd_lab_t(cand_h, torch), ball, torch)
    hom_v = _ahd_homogeneity_t(_ahd_lab_t(cand_v, torch), ball, torch)

    gt = (hom_h > hom_v).unsqueeze(-1)
    lt = (hom_v > hom_h).unsqueeze(-1)
    del hom_h, hom_v
    out = torch.where(gt, cand_h,
                      torch.where(lt, cand_v, 0.5 * (cand_h + cand_v)))
    del gt, lt, cand_h, cand_v

    if blend is not None:
        b3 = blend.unsqueeze(-1)
        out = out * (1.0 - b3) + bilinear * b3
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strip processing
#
# Device memory for a whole-frame demosaic scales with the sensor: RCD needs
# roughly 14 full-frame float32 planes resident and AHD about 20, so a 46.8 Mpx
# frame wants ~2.5 GB for RCD and ~3.7 GB for AHD before transients. On an 8 GB
# card that also has a desktop on it, AHD whole-frame simply does not fit.
#
# Splitting into row strips with a halo caps that at the strip size instead, so
# VRAM stops scaling with the sensor. Every kernel here reads at most a few
# rows away, so a halo wider than the deepest dependency chain makes the
# central rows bit-for-bit what the whole-frame pass would have produced:
#
#   bilinear prefill    +-1
#   RCD phase 1 / 3     +-2        AHD phase 1   +-2
#   RCD phase 2         +-1        AHD phase 2   +-1
#                                  homogeneity   +-1
#                                  box smooth    +-1
#   clip blend          +-3 dilation, then a Gaussian of ksize
#                       2*round(3*sigma)+1 -> +-6 at the default sigma=2
#
# The clip blend dominates at +-9, so 12 is used, with margin. The halo is
# also where the slab's own edge effects live (the kernels treat a slab edge
# like an image edge, and the Gaussian reflects there), which is exactly why
# those rows are computed and then thrown away.
#
# Two things MUST hold or the output is wrong rather than merely different:
#
#   1. A slab must start on an EVEN row. The Bayer masks are built from row
#      parity, so starting a slab on an odd row silently swaps the pattern for
#      that slab -- red and blue exchanged across a band of the image.
#   2. RCD's noise sigma must be estimated once on the WHOLE mosaic and passed
#      in. Per-slab estimates differ from each other and from the whole-frame
#      value, which would make the noise floor vary down the image.
# ─────────────────────────────────────────────────────────────────────────────
_STRIP_HALO = 12

# A slab thinner than this is mostly halo, and a frame cut into more slabs
# than this spends more time moving data across PCIe than demosaicing. Past
# either line the numba kernel -- which is parallel and already uses every
# core -- is the better answer, so the GPU path declines rather than grinding.
#
# This is not hypothetical. Flooring at 64 rows instead of declining turned a
# 5644-row frame into 89 round trips held under the global device lock, which
# is a stacking run that goes from fast to apparently frozen with no error.
_STRIP_MIN_ROWS = 64
_STRIP_MAX_SLABS = 24

# How long a frame may wait for another frame to finish with the device.
# One demosaic is sub-second to a few seconds, and at most _STRIP_MAX_SLABS
# round trips, so a queue of workers clears in well under a minute; 120 s
# means something is genuinely wrong rather than merely busy. On expiry the
# frame declines and runs on the CPU, so a stalled device costs throughput
# instead of hanging the run.
_GPU_DEBAYER_LOCK_TIMEOUT = 120.0

# Full-frame float32 planes resident at peak, per method. Used to pick a strip
# height, and to decide whether a whole frame would fit at all.
#
# These are counted by reading the kernels -- what is alive at the deepest
# point, which for both methods is phase 2 -- not measured on a card, since
# that needs the card the code is deciding about. If the count is optimistic
# the allocation raises and the out-of-memory path below halves the strip, so
# being wrong costs one wasted attempt rather than a failed frame. The budget
# fraction also leaves room: 0.55 of usable, not 0.9.
_PLANES_PER_METHOD = {"rcd": 14, "ahd": 20}

# Fraction of usable device memory a single demosaic may plan to occupy.
# The rest is headroom for allocator fragmentation and whatever else holds the
# card during a stacking run.
_STRIP_BUDGET_FRACTION = 0.55


def _strip_rows(H, W, itemsize, planes, budget_bytes):
    """Rows per strip, H for "do it in one pass", or 0 for "do not use the GPU".

    Chosen so (strip + 2*halo) * W * itemsize * planes fits the budget, then
    rounded DOWN to an even number of rows so every slab starts on an even row.

    Returning 0 is the important case. There is a band of device memory -- too
    little for a sensible strip, still enough to pass a minimum-slab check --
    where striping technically works and is far slower than the CPU kernel.
    Grinding through it under the global device lock looks exactly like a
    hang. Declining hands the frame to numba, which is the right answer there.
    """
    if budget_bytes is None or budget_bytes <= 0:
        return H
    per_row = float(W) * itemsize * planes
    if per_row <= 0:
        return H
    rows = int(budget_bytes / per_row) - 2 * _STRIP_HALO
    if rows >= H:
        return H
    if rows < _STRIP_MIN_ROWS:
        return 0
    rows -= (rows & 1)
    if rows <= 0 or (H + rows - 1) // rows > _STRIP_MAX_SLABS:
        return 0
    return rows


def _debayer_gpu_strips(torch, device, a, bp, m, dtype64, preserve_zeros,
                        denoise_k, clip_blend, clip_threshold, clip_feather,
                        ahd_lab_ball, sigma, rows, status_cb=None):
    """Run the demosaic in row strips and assemble the result on the host.

    Every argument is forwarded to _debayer_gpu_once BY KEYWORD. That function
    takes fourteen positional parameters, debayer_gpu swallows any exception
    from it as "no GPU, use numba", and a strip is the one caller that is not
    the public entry point -- so a positional call that drifted by one slot
    here would show up as a permanent, silent fall back to the CPU rather than
    as an error. (It did, during development.)
    """
    H, W = a.shape
    out = np.empty((H, W, 3), dtype=np.float32)
    if status_cb:
        try:
            n = (H + rows - 1) // rows
            status_cb(f"{n} strip(s) of {rows} rows (+/-{_STRIP_HALO} halo)")
        except Exception:
            pass
    y = 0
    while y < H:
        y1 = min(H, y + rows)
        # halo, clipped to the frame; lo stays even so the Bayer phase of the
        # slab matches the full frame
        lo = max(0, y - _STRIP_HALO)
        lo -= (lo & 1)
        hi = min(H, y1 + _STRIP_HALO)
        slab = _debayer_gpu_once(
            torch, device, np.ascontiguousarray(a[lo:hi]), bp, m,
            interpolate=True, dtype64=dtype64, preserve_zeros=preserve_zeros,
            denoise_k=denoise_k, clip_blend=clip_blend,
            clip_threshold=clip_threshold, clip_feather=clip_feather,
            ahd_lab_ball=ahd_lab_ball, sigma=sigma)
        out[y:y1] = slab[y - lo:y1 - lo]
        del slab
        y = y1
    return out


def _debayer_gpu_once(torch, device, a, bp, m, interpolate, dtype64,
                      preserve_zeros, denoise_k, clip_blend, clip_threshold,
                      clip_feather, ahd_lab_ball, sigma):
    """One demosaic on the device, over whatever rows `a` holds.

    Called for a whole frame or for one strip -- it cannot tell the
    difference, which is the point. The caller holds _GPU_DEBAYER_LOCK, has
    already resolved the RCD-N sigma on the WHOLE mosaic, and has picked the
    clip settings for the chosen method.
    """
    dt = torch.float64 if dtype64 else torch.float32
    im = torch.from_numpy(
        np.ascontiguousarray(a, dtype=np.float64 if dtype64 else np.float32)
    ).to(device, dtype=dt)
    H, W = im.shape

    r_mask, g_mask, b_mask = _bayer_masks_t(bp, H, W, torch, device)
    masks = (r_mask, g_mask, b_mask)

    if not interpolate:
        out = _sparse_cfa_t(im, masks, torch)
    else:
        # Build the blend mask BEFORE the base image. It depends only on the
        # mosaic, and on any frame with nothing near clip it comes back None --
        # which is most calibrated subs. Knowing that first means the
        # full-frame clone of the bilinear base is only paid when it will
        # actually be used.
        blend = (_rcd_clip_blend_t(im, clip_threshold, clip_feather, torch)
                 if clip_blend else None)
        bilinear = _bilinear_prefill_t(im, masks, torch)

        if m == "ahd":
            out = _ahd_demosaic_t(im, masks, ahd_lab_ball, blend, bilinear,
                                  torch)
            del blend, bilinear, r_mask, g_mask, b_mask
        else:
            out = bilinear.clone() if blend is not None else bilinear
            rb = r_mask | b_mask
            del b_mask      # r_mask is still needed by phases 2 and 3
            _rcd_green_at_rb_t(im, out, rb, float(sigma), denoise_k, torch)
            _rcd_rb_at_g_t(out, g_mask, r_mask, torch)
            _rcd_rb_at_br_t(out, r_mask, rb, float(sigma), denoise_k, torch)
            del rb, r_mask, g_mask
            if blend is not None:
                b3 = blend.unsqueeze(-1)
                out = out * (1.0 - b3) + bilinear * b3
                del b3, blend, bilinear

    if preserve_zeros:
        zm = (im == 0)
        if bool(zm.any()):
            out = torch.where(zm.unsqueeze(-1), torch.zeros_like(out), out)
        del zm
    del im

    # .cpu() is the device sync; nothing downstream needs a tensor.
    return out.to(torch.float32).cpu().numpy()


def debayer_gpu(image_data: np.ndarray,
                bayer_pattern: str,
                *,
                cfa_drizzle: bool = False,
                method: str = "rcd",
                preserve_zeros: bool = True,
                rcd_denoise: bool = True,
                rcd_denoise_k: float = 4.0,
                rcd_clip_blend: bool = True,
                rcd_clip_threshold: float = 0.95,
                rcd_clip_feather: float = 2.0,
                ahd_lab_ball: float = 1.0,
                ahd_clip_blend: bool = True,
                ahd_clip_threshold: float = 0.95,
                ahd_clip_feather: float = 2.0,
                noise_sigma: Optional[float] = None,
                dtype64: bool = False,
                force_device=None,
                status_cb=None) -> Optional[np.ndarray]:
    """GPU Bayer demosaic. Signature mirrors numba_utils.debayer_fits_fast.

    Returns an (H,W,3) float32 array, or None if the caller should fall back
    to the Numba kernel — no GPU, an unported method, odd input, or any
    exception. Never raises.

    noise_sigma overrides the per-frame RCD-N estimate (used by the
    verification harness to isolate the demosaic from the estimator).
    dtype64 computes in float64 to reproduce numba's internal promotion.
    force_device is for testing on CPU torch.
    """
    def _no(reason):
        if status_cb:
            try:
                status_cb(reason)
            except Exception:
                pass
        return None

    torch, device = _resolve_gpu_device()
    if force_device is not None:
        device = force_device
    if torch is None or device is None:
        return _no("no torch device resolved")
    if force_device is None and str(device.type) == "cpu":
        return _no("torch resolved to the CPU")

    bp = (bayer_pattern or "").upper()
    if bp not in _CFA_CHAN_T:
        return _no(f"unsupported Bayer pattern {bayer_pattern!r}")

    a = np.asarray(image_data)
    if a.ndim != 2 or a.size == 0 or min(a.shape) < 5:
        return _no(f"not a usable 2-D mosaic (shape {getattr(a, 'shape', None)})")

    interpolate = not cfa_drizzle
    m = (method or "rcd").lower()
    if interpolate and m not in GPU_DEBAYER_METHODS:
        return _no(f"method {m!r} has no GPU implementation")

    itemsize = 8 if dtype64 else 4
    _planes = _PLANES_PER_METHOD.get(m, 14)
    _min_slab = ((_STRIP_MIN_ROWS + 2 * _STRIP_HALO)
                 * a.shape[1] * itemsize * _planes)

    # Which clip settings apply, and how deep the memory need is.
    if m == "ahd":
        _clip_blend = bool(ahd_clip_blend)
        _clip_thr = float(ahd_clip_threshold)
        _clip_feather = float(ahd_clip_feather)
    else:
        _clip_blend = bool(rcd_clip_blend)
        _clip_thr = float(rcd_clip_threshold)
        _clip_feather = float(rcd_clip_feather)

    def _run(rows, sigma):
        H = a.shape[0]
        if rows >= H:
            return _debayer_gpu_once(
                torch, device, a, bp, m, interpolate, dtype64, preserve_zeros,
                rcd_denoise_k, _clip_blend, _clip_thr, _clip_feather,
                ahd_lab_ball, sigma)
        return _debayer_gpu_strips(
            torch, device, a, bp, m, dtype64, preserve_zeros, rcd_denoise_k,
            _clip_blend, _clip_thr, _clip_feather, ahd_lab_ball, sigma, rows,
            status_cb=status_cb)

    def _plan():
        _u = _gpu_usable_bytes(torch, device)
        _budget = None if _u is None else int(_u * _STRIP_BUDGET_FRACTION)
        return _strip_rows(a.shape[0], a.shape[1], itemsize, _planes, _budget)

    # One lock, taken with a timeout, around everything that touches the
    # device -- including the headroom check, because _gpu_has_room calls
    # empty_cache() and a device-wide cudaFree issued from one worker thread
    # while another is mid-demosaic is a synchronisation point that does not
    # belong in a parallel stage.
    #
    # The timeout is the part that matters. Without it, whatever one frame
    # does to the device, every other worker inherits: the normalization pool
    # has no way back to the CPU kernel, as_completed waits forever, and the
    # run stops with no error to show for it.
    if not _GPU_DEBAYER_LOCK.acquire(timeout=_GPU_DEBAYER_LOCK_TIMEOUT):
        return _no(f"another frame has held the device for over "
                   f"{_GPU_DEBAYER_LOCK_TIMEOUT:.0f}s")
    _rows = 0       # bound before the try: the retry handler reports it
    try:
        try:
            if not _gpu_has_room(torch, device, _min_slab,
                                 status_cb=status_cb):
                return _no("not enough device memory for even a minimal strip")
            _rows = _plan()
            if _rows <= 0:
                return _no("too little device memory to strip this frame "
                           "sensibly; the numba kernel is faster here")

            # The RCD-N sigma is estimated ONCE, on the whole mosaic.
            # Per-strip estimates would differ from each other and from the
            # whole-frame value, so the noise floor would vary down the image.
            # Costs one extra upload of the mosaic, which is milliseconds.
            _sigma = 0.0
            if interpolate and m == "rcd":
                if noise_sigma is not None:
                    _sigma = float(noise_sigma)
                elif rcd_denoise and rcd_denoise_k > 0.0:
                    _full = torch.from_numpy(
                        np.ascontiguousarray(a, dtype=np.float32)
                    ).to(device, dtype=torch.float32)
                    _gm = _bayer_masks_t(bp, a.shape[0], a.shape[1],
                                         torch, device)[1]
                    _sigma = estimate_mosaic_noise_sigma_t(_full, _gm, torch)
                    del _full, _gm
            return _run(_rows, _sigma)
        except Exception as e:
            # An OOM here is usually another stage's reserve, not a card that
            # is genuinely too small: release the cached blocks and give it one
            # more go before falling back to numba. Anything else is reported
            # as-is rather than swallowed -- a silent None was how a real
            # failure got reported to the user as "cancelled".
            _oom = ("out of memory" in str(e).lower()
                    or type(e).__name__ == "OutOfMemoryError")
            if not _oom:
                return _no(f"{type(e).__name__}: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Retrying the same plan would fail the same way, so go narrower:
            # half the rows the budget suggested. If that falls under the
            # sensible-strip floor, decline rather than grind.
            try:
                _rows = _plan()
                _rows = (a.shape[0] // 2) if _rows >= a.shape[0] else (_rows // 2)
                _rows -= (_rows & 1)
                if (_rows < _STRIP_MIN_ROWS
                        or (a.shape[0] + _rows - 1) // _rows
                        > _STRIP_MAX_SLABS):
                    return _no("out of device memory, and a narrower strip "
                               "would be slower than the CPU kernel")
                if status_cb:
                    status_cb(f"out of memory; retrying in {_rows}-row strips")
                return _run(_rows, 0.0 if noise_sigma is None
                            else float(noise_sigma))
            except Exception as e2:
                return _no(f"out of device memory even in {_rows}-row strips "
                           f"({type(e2).__name__})")
    finally:
        _GPU_DEBAYER_LOCK.release()


def debayer_gpu_available() -> bool:
    """True when a GPU debayer could run at all — torch resolves to a device
    whose type is not "cpu". Says nothing about whether it is FASTER; that is
    the optimizer's job (stacking/debayer_gpu).

    The "not cpu" test rather than a cuda/mps allowlist is deliberate and
    matches gpu_warp.available() and gpu_star_detect.gpu_available():
    torch-directml reports its device type as "privateuseone", and an
    allowlist silently excludes every DirectML box.
    """
    t, d = _resolve_gpu_device()
    if t is None or d is None:
        return False
    try:
        return str(d.type) != "cpu"
    except Exception:
        return False


def debayer_backend_note() -> str:
    """How torch resolved, for the optimizer log and for support.

    Calls out a CPU-only wheel explicitly, the way gpu_warp.backend_note()
    does: such a build resolves, runs, accelerates nothing, and the only
    outward sign is "+cpu" in a version string.
    """
    t, d = _resolve_gpu_device()
    if t is None:
        return "torch unavailable"
    if d is None:
        return "no device resolved"
    ver = str(getattr(t, "__version__", "?"))
    name = str(d.type)
    try:
        if name == "cuda":
            name = f"cuda ({t.cuda.get_device_name(0)})"
    except Exception:
        pass
    note = f"torch {ver} on {name}"
    if str(d.type) == "cpu":
        note += (" — CPU-ONLY: no GPU path. SASpro's staged CUDA runtime is "
                 "absent or shadowed by a pip-installed torch in this venv"
                 if "+cpu" in ver else
                 " — no GPU device resolved; everything falls back")
        note += f" (torch at {getattr(t, '__file__', '?')})"
    return note
