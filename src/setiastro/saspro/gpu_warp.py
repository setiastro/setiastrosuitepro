# ──────────────────────────────────────────────────────────────────────────────
#  src/setiastro/saspro/gpu_warp.py — SetiAstroSuite Pro
#  Copyright (C) 2026 Franklin Marek / SetiAstro
#  Licensed under the GNU General Public License v3.0 or later.
# ──────────────────────────────────────────────────────────────────────────────
"""
The anti-ring warp, on the GPU.

Profiled on a 574-frame run of 8288x5644 frames, `_finalize_write_job` spent
58% of its time in the warp — 5.15 s per frame — against 8% in the write. It
is the single biggest cost in registration, and it is pure compute: two
full-resolution warps per frame (Lanczos for sharpness, bilinear for the
ring-free envelope) plus the star-mask morphology.

On the GPU the two warps collapse into one pass, because the bilinear result
is a 2x2 subset of the same 8x8 neighbourhood the Lanczos gather already has
to read. The morphology is min/max pooling. Nothing here is novel; it is the
same arithmetic, moved.

MATCHING OpenCV
---------------
This replaces every pixel of every aligned frame, so "close enough" is not
good enough — a user who re-stacks after upgrading must get the same result.
Four details of cv2's internals had to be reproduced to get there, each one
found by measuring rather than by reading:

  * OpenCV's Lanczos4 kernel is exactly sinc(t)*sinc(t/4) normalised to sum 1.
    Its `interpolateLanczos4` is an optimised rewriting of that; the two agree
    to 8e-16 across the fractional range.

  * The Lanczos path does NOT use the exact fractional coordinate. It
    quantises it to 1/INTER_TAB_SIZE = 1/32 and looks the coefficients up in a
    precomputed table. Skipping that quantisation leaves a median error of
    3e-3 — small-looking, and far above float32 noise.

  * `warpAffine` derives its coordinates in 10-bit fixed point (AB_SCALE =
    1024) and shifts down to the 1/32 grid, so the integer base and the table
    index both come from that pipeline rather than from the real-valued
    coordinate. `warpPerspective` has no such stage.

  * The bilinear path is NOT quantised to 1/32 (that is off by 7e-3) and it
    does not share the Lanczos neighbourhood.

With those in place the Lanczos channel — which is the output everywhere
outside the star mask, around 99% of pixels — agrees with cv2 to 5e-7, the
float32 representation floor.

WHERE IT DELIBERATELY DOES NOT MATCH
------------------------------------
cv2's float32 bilinear carries its sample coordinate as a float32. At the
right-hand edge of an 8288-wide frame one ULP is 9.8e-4 px, so cv2's own
bilinear is uncertain by about a millipixel there, and which way the last bit
falls depends on the build (scalar, SIMD or IPP). It is not reproducible
across OpenCV builds, and rounding this module's coordinates to float32 to
imitate it tracks cv2 *worse*, not better.

So the bilinear here is computed exactly, in float64 coordinates. Measured
against the exact float64 bilinear on an 8288-wide frame:

    cv2                4.2e-04
    this module        1.3e-07

The consequence, stated plainly because it is a real difference and not
noise: the blended star cores and the envelope-clamped pixels — about 1% of
the frame — can differ from the CPU path by up to ~2e-3 at full frame width,
and this module is the accurate side of that difference. 2e-3 is well under
the read noise of any single sub, and it averages down over a stack, but it
is not bit-identical and `gpu_warp_test.py` asserts accuracy against the
float64 truth rather than bug-compatibility with cv2.

`warp_antiring_gpu` returns None on any failure so the caller falls back to
the CPU path unchanged.
"""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "available",
    "bilinear_mode",
    "backend_note",
    "reset_backend",
    "warp_antiring_gpu",
]

# cv2 constants, reproduced so this module does not import cv2
INTER_BITS = 5
INTER_TAB_SIZE = 1 << INTER_BITS          # 32

_TORCH = None
_DEVICE = None
_RESOLVED = False
_NOTE = "not resolved"


def reset_backend():
    global _TORCH, _DEVICE, _RESOLVED, _NOTE
    _TORCH = _DEVICE = None
    _RESOLVED = False
    _NOTE = "not resolved"


def _resolve():
    """(torch, device) or (None, None). Mirrors gpu_star_detect's resolver."""
    global _TORCH, _DEVICE, _RESOLVED, _NOTE
    if _RESOLVED:
        return _TORCH, _DEVICE
    _RESOLVED = True
    torch = None
    staged = False

    # Stage SASpro's own torch runtime onto sys.path FIRST. This is install-
    # free — it injects the staged venv's site-packages and demotes a
    # shadowing torch — and skipping it is how a CPU-only wheel sitting in the
    # project venv wins over the CUDA build SASpro went to the trouble of
    # installing. That is not hypothetical: it is what `torch 2.13.0+cpu on
    # cpu via plain torch` means, and in that state nothing here accelerates
    # anything while still reporting success.
    #
    # `import_torch` is the one that can shell out to pip and hang a spawned
    # worker, so it stays a last resort; `add_runtime_to_sys_path` cannot.
    try:
        import sys as _sys
        from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
        _before = list(_sys.path)
        add_runtime_to_sys_path(lambda *_: None)
        # it returns None whether or not it found anything, so the only honest
        # way to report which torch this is, is to see if sys.path moved
        staged = any(p not in _before for p in _sys.path)
    except Exception:
        pass

    try:
        import torch as _t
        torch = _t
        _NOTE = "staged runtime torch" if staged else "plain torch"
    except Exception:
        pass
    if torch is None:
        try:
            from setiastro.saspro.runtime_torch import import_torch
            torch = import_torch(prefer_cuda=True, prefer_xpu=False,
                                 prefer_dml=True, status_cb=lambda *_: None)
            _NOTE = "runtime_torch bootstrap"
        except Exception as e:
            _TORCH, _DEVICE = None, None
            _NOTE = f"no torch ({type(e).__name__})"
            return _TORCH, _DEVICE
    dev = None
    try:
        from setiastro.saspro.runtime_torch import best_device
        dev = best_device(torch, prefer_cuda=True, prefer_dml=True)
    except Exception:
        dev = None
    if dev is None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = torch.device("cuda")
            elif (getattr(getattr(torch, "backends", None), "mps", None)
                  and torch.backends.mps.is_available()):
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        except Exception:
            dev = torch.device("cpu")
    _TORCH, _DEVICE = torch, dev
    return _TORCH, _DEVICE


def available() -> bool:
    """True iff a non-CPU torch device is usable for warping."""
    torch, dev = _resolve()
    if torch is None or dev is None:
        return False
    try:
        return str(dev.type) != "cpu"
    except Exception:
        return False


def backend_note() -> str:
    torch, dev = _resolve()
    if torch is None:
        return f"gpu_warp: disabled — {_NOTE}"
    ver = str(getattr(torch, "__version__", "?"))
    line = f"gpu_warp: torch {ver} on {dev} via {_NOTE}"
    # Say so loudly. A CPU-only wheel resolves and runs and accelerates
    # nothing, and the only outward sign is a version string with "+cpu" in it.
    if str(dev) == "cpu":
        if "+cpu" in ver:
            line += ("\n  ! this is a CPU-ONLY torch build — there is no GPU"
                     " path at all.\n    SASpro's staged CUDA runtime is"
                     " either absent or shadowed by a\n    pip-installed"
                     " torch in this venv.")
        else:
            line += "\n  ! no GPU device resolved; everything will fall back."
        line += f"\n    torch is at {getattr(torch, '__file__', '?')}"
    return line


# ─────────────────────────────────────────────────────────────────────────────
# Lanczos4 coefficient table — 32 fractional positions x 8 taps, exactly the
# table cv2 builds in initInterTab2D.
# ─────────────────────────────────────────────────────────────────────────────
def _lanczos4_table(np_mod=np):
    f = np_mod.arange(INTER_TAB_SIZE, dtype=np_mod.float64) / INTER_TAB_SIZE
    t = f[:, None] + 3.0 - np_mod.arange(8, dtype=np_mod.float64)[None, :]
    w = np_mod.sinc(t) * np_mod.sinc(t / 4.0)
    return w / w.sum(axis=1, keepdims=True)


def _bilinear_table(np_mod=np):
    f = np_mod.arange(INTER_TAB_SIZE, dtype=np_mod.float64) / INTER_TAB_SIZE
    return np_mod.stack([1.0 - f, f], axis=1)


_TAB_CACHE: dict = {}


def _tables(torch, dev):
    key = str(dev)
    hit = _TAB_CACHE.get(key)
    if hit is not None:
        return hit
    lan = torch.from_numpy(_lanczos4_table().astype(np.float32)).to(dev)
    bil = torch.from_numpy(_bilinear_table().astype(np.float32)).to(dev)
    _TAB_CACHE[key] = (lan, bil)
    return lan, bil


# ─────────────────────────────────────────────────────────────────────────────
# morphology
# ─────────────────────────────────────────────────────────────────────────────
def _dilate_rect(torch, F, x, r):
    """Max filter over a (2r+1) square — cv2.dilate with MORPH_RECT."""
    if r <= 0:
        return x
    return F.max_pool2d(x, 2 * r + 1, stride=1, padding=r)


def _erode_rect(torch, F, x, r):
    """Min filter over a (2r+1) square — cv2.erode with MORPH_RECT."""
    if r <= 0:
        return x
    return -F.max_pool2d(-x, 2 * r + 1, stride=1, padding=r)


def _ellipse_kernel(r: int) -> np.ndarray:
    """cv2.getStructuringElement(MORPH_ELLIPSE, (2r+1, 2r+1)), reproduced.

    OpenCV builds this row by row from the circle equation rather than from
    a distance test, so a naive (dx^2+dy^2 <= r^2) disc is NOT the same shape
    and would grow the star mask differently.
    """
    import math
    n = 2 * r + 1
    k = np.zeros((n, n), dtype=np.uint8)
    inv_r2 = 1.0 / (r * r) if r else 0.0
    for i in range(n):
        j1 = j2 = 0
        dy = i - r
        if abs(dy) <= r:
            dx = int(round(r * math.sqrt(max(0.0, 1.0 - dy * dy * inv_r2))))
            j1 = max(r - dx, 0)
            j2 = min(r + dx + 1, n)
        k[i, j1:j2] = 1
    return k


def _median_mad(torch, plane, finite_count):
    """median and MAD over the finite pixels, with numpy's tie rule.

    np.median averages the two middle values of an even-sized sample;
    torch.median returns the lower one. On a 46.8 MP frame the gap between
    adjacent order statistics is ~1e-8, so the two agree almost always — and
    "almost" is the problem. The threshold feeds a hard comparison, so a
    single pixel landing in that gap flips a mask pixel, and a flipped mask
    pixel moves by |hi - lo|, which on a star core is ~1e-2. Rare, invisible
    in a tolerance, and a real difference from the CPU path.

    Non-finite pixels are pushed to +inf rather than gathered out, so this
    stays allocation-free on a full frame.
    """
    big = torch.finfo(plane.dtype).max
    n = int(finite_count)
    # a frame with no NaN needs no copy at all, and the deviations are built
    # in place: written the obvious way this step allocated four full planes,
    # which at 46.8 MP is 700 MB of nothing
    flat = (plane.reshape(-1) if n == plane.numel()
            else torch.nan_to_num(plane.reshape(-1), nan=big, posinf=big,
                                  neginf=big))
    if n <= 0:
        z = torch.zeros((), dtype=plane.dtype, device=plane.device)
        return z, z

    def _med(v):
        if n % 2:
            return torch.kthvalue(v, n // 2 + 1).values
        lo_ = torch.kthvalue(v, n // 2).values
        hi_ = torch.kthvalue(v, n // 2 + 1).values
        return (lo_ + hi_) / 2.0

    med = _med(flat)
    dev_ = flat.sub(med).abs_()
    if n != plane.numel():
        dev_.masked_fill_(flat >= big, big)
    return med, _med(dev_)


def _band_rows(out_h, out_w, budget_bytes=192 << 20):
    """Output rows per Lanczos pass.

    The 8x8 gather keeps roughly ten (rows, out_w) planes live at once — two
    float64 coordinate grids, four int64 index grids, an accumulator and the
    tap temporaries. Full frame that is gigabytes; banded it is whatever this
    budget allows, which is what lets an 8 GB card with a display attached
    finish the frame instead of raising and falling back to the CPU.
    """
    # per output pixel, live at once: two float64 coordinate grids (16 B),
    # four int64 index grids (32 B), two 8-deep float32 weight stacks (64 B),
    # an accumulator and tap temporaries (~24 B)
    per_row = max(1, out_w * 136)
    rows = int(budget_bytes // per_row)
    return max(16, min(int(out_h), rows))


def _dilate_ellipse(torch, F, x, r):
    """Max filter with cv2's elliptical structuring element.

    Decomposed by rows rather than unfolded. cv2 builds MORPH_ELLIPSE row by
    row from the circle equation, so every row of the element is a single
    contiguous run centred on the middle column; a max over the disc is
    therefore a 1-D max filter per row followed by a max down the rows.

    The obvious F.unfold version is not an option at full resolution: a 13x13
    element over an 8288x5644 frame would materialise 169 copies of it, 31 GB,
    and the whole point of this module is to run inside an 8 GB card.
    """
    if r <= 0:
        return x
    k = _ellipse_kernel(r)
    if k.all():
        return _dilate_rect(torch, F, x, r)

    out = None
    h = x.shape[-2]
    for row in range(2 * r + 1):
        cols = np.nonzero(k[row])[0]
        if cols.size == 0:
            continue
        half = int(cols.size // 2)
        # max_pool2d pads with -inf, which is cv2's dilate border
        band = (F.max_pool2d(x, (1, 2 * half + 1), stride=1, padding=(0, half))
                if half > 0 else x)
        dy = row - r                      # element row -> source row offset
        if dy == 0:
            shifted = band
        else:
            # cv2's dilate border is -DBL_MAX, i.e. contributes nothing; the
            # dy == 0 row is always in frame so no pixel stays at the fill
            shifted = torch.full_like(band, torch.finfo(band.dtype).min)
            if dy > 0:                    # element below centre reads above
                shifted[..., :h - dy, :] = band[..., dy:, :]
            else:
                shifted[..., -dy:, :] = band[..., :h + dy, :]
        out = shifted if out is None else torch.maximum(out, shifted)
    return x if out is None else out


def _gaussian_blur(torch, F, x, ksize: int, sigma: float):
    """Separable gaussian matching cv2.GaussianBlur's kernel and BORDER_REFLECT_101."""
    if ksize <= 1 or sigma <= 0:
        return x
    if ksize % 2 == 0:
        ksize += 1
    r = ksize // 2
    # cv2.getGaussianKernel
    i = np.arange(ksize, dtype=np.float64) - r
    g = np.exp(-(i * i) / (2.0 * sigma * sigma))
    g = (g / g.sum()).astype(np.float32)
    gk = torch.from_numpy(g).to(x.device)
    xpad = F.pad(x, (r, r, 0, 0), mode="reflect")
    x1 = F.conv2d(xpad, gk.view(1, 1, 1, -1))
    xpad = F.pad(x1, (0, 0, r, r), mode="reflect")
    return F.conv2d(xpad, gk.view(1, 1, -1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# the warp
# ─────────────────────────────────────────────────────────────────────────────
AB_BITS = 10                       # cv2: MAX(10, INTER_BITS)
AB_SCALE = 1 << AB_BITS            # 1024


def _affine_fixed_point(torch, M, out_h, out_w, dev, y0=0, rows=None):
    """cv2's WarpAffineInvoker coordinate pipeline, reproduced.

    warpAffine does NOT evaluate the source coordinate in floating point. It
    builds it in 10-bit fixed point:

        adelta[x] = cvRound(Mi[0,0] * x * 1024)
        X0        = cvRound((Mi[0,1]*y + Mi[0,2]) * 1024) + 16
        X         = (X0 + adelta[x]) >> 5        -> units of 1/32
        ix        = X >> 5 ;  table index = X & 31

    so its coordinates carry about 1e-3 px of quantisation. Computing them
    exactly instead leaves the median at float32 noise but puts ~1e-2 in the
    max, because on the flank of a bright star 1e-3 px of position is 1% of
    the peak. warpPerspective has no such stage, which is why the projective
    path was already two orders of magnitude closer.

    Returns integer source positions and 1/32 table indices directly.
    """
    Mt = np.asarray(M, dtype=np.float64).reshape(2, 3)
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = Mt
    Mi = np.linalg.inv(A)                      # cv2 inverts unless WARP_INVERSE_MAP

    x = np.arange(out_w, dtype=np.float64)
    n = out_h if rows is None else int(rows)
    y = np.arange(y0, y0 + n, dtype=np.float64)
    rnd = np.rint                              # cvRound: half-to-even, as numpy
    adelta = rnd(Mi[0, 0] * x * AB_SCALE).astype(np.int64)
    bdelta = rnd(Mi[1, 0] * x * AB_SCALE).astype(np.int64)
    round_delta = AB_SCALE // INTER_TAB_SIZE // 2        # 16
    X0 = rnd((Mi[0, 1] * y + Mi[0, 2]) * AB_SCALE).astype(np.int64) + round_delta
    Y0 = rnd((Mi[1, 1] * y + Mi[1, 2]) * AB_SCALE).astype(np.int64) + round_delta

    # >> is an arithmetic shift in C, i.e. floor division for negatives too
    X = (X0[:, None] + adelta[None, :]) >> (AB_BITS - INTER_BITS)
    Y = (Y0[:, None] + bdelta[None, :]) >> (AB_BITS - INTER_BITS)
    ix = X >> INTER_BITS
    iy = Y >> INTER_BITS
    qx = X & (INTER_TAB_SIZE - 1)
    qy = Y & (INTER_TAB_SIZE - 1)

    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    # The bilinear half needs the EXACT coordinate — cv2's float bilinear is
    # not quantised — so it is computed separately rather than reconstructed
    # from the fixed-point value, which would re-introduce the quantisation
    # this function exists to reproduce for Lanczos only.
    sx_f = Mi[0, 0] * x[None, :] + Mi[0, 1] * y[:, None] + Mi[0, 2]
    sy_f = Mi[1, 0] * x[None, :] + Mi[1, 1] * y[:, None] + Mi[1, 2]
    return T(ix), T(iy), T(qx), T(qy), T(sx_f), T(sy_f)


def _source_coords(torch, M, out_h, out_w, dev, perspective: bool,
                   y0=0, rows=None):
    """Destination grid mapped back through M to source coordinates.

    M is the FORWARD transform (source -> destination), the same matrix the
    cv2 calls receive, so it is inverted here exactly as cv2 does internally
    when WARP_INVERSE_MAP is not set.
    """
    n = out_h if rows is None else int(rows)
    yy = torch.arange(y0, y0 + n, device=dev,
                      dtype=torch.float64).view(-1, 1)
    xx = torch.arange(out_w, device=dev, dtype=torch.float64).view(1, -1)
    Mt = torch.as_tensor(np.asarray(M, dtype=np.float64), device=dev)
    if perspective:
        Mi = torch.linalg.inv(Mt)
        den = Mi[2, 0] * xx + Mi[2, 1] * yy + Mi[2, 2]
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        sx = (Mi[0, 0] * xx + Mi[0, 1] * yy + Mi[0, 2]) / den
        sy = (Mi[1, 0] * xx + Mi[1, 1] * yy + Mi[1, 2]) / den
    else:
        A = torch.eye(3, dtype=torch.float64, device=dev)
        A[:2, :] = Mt.reshape(2, 3)
        Ai = torch.linalg.inv(A)
        sx = Ai[0, 0] * xx + Ai[0, 1] * yy + Ai[0, 2]
        sy = Ai[1, 0] * xx + Ai[1, 1] * yy + Ai[1, 2]
    return sx.expand(n, out_w), sy.expand(n, out_w)


# ─────────────────────────────────────────────────────────────────────────────
# Which bilinear does the local cv2 do?
# ─────────────────────────────────────────────────────────────────────────────
# This is build-dependent, and assuming either answer is wrong somewhere.
# `remapBilinear` looks its weights up in the 1/INTER_TAB_SIZE table, which is
# what the OpenCV source describes and what ships on Windows; some builds take
# a float-specific path and use the exact fractional coordinate instead. The
# two disagree by ~7e-3 on a star flank — not enough to move a star (0.015 px
# of centroid, 0.006% of flux) but far more than float32 noise, and there is
# no reason to accept it when the local answer is one small warp away.
#
# Override with SASPRO_GPU_WARP_BILINEAR=table|exact if a build ever fools it.
_BIL_MODE = None


def _fixed_point_frac_at(M, y, x):
    """cv2's affine fixed-point fraction at one output pixel, in 1/32 units."""
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = np.asarray(M, np.float64).reshape(2, 3)
    Mi = np.linalg.inv(A)
    rnd = np.rint
    rd = AB_SCALE // INTER_TAB_SIZE // 2
    X = (int(rnd((Mi[0, 1] * y + Mi[0, 2]) * AB_SCALE)) + rd
         + int(rnd(Mi[0, 0] * x * AB_SCALE))) >> (AB_BITS - INTER_BITS)
    Y = (int(rnd((Mi[1, 1] * y + Mi[1, 2]) * AB_SCALE)) + rd
         + int(rnd(Mi[1, 0] * x * AB_SCALE))) >> (AB_BITS - INTER_BITS)
    return (X >> INTER_BITS, Y >> INTER_BITS,
            X & (INTER_TAB_SIZE - 1), Y & (INTER_TAB_SIZE - 1),
            Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2],
            Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2])


def bilinear_mode():
    """'table' (1/32, like the OpenCV source) or 'exact', or None if unclear.

    Measured by warping a delta image: with one source pixel set to 1, cv2's
    output at a destination pixel IS the weight it gave that tap, so the two
    candidate models can be told apart directly instead of inferred.
    """
    global _BIL_MODE
    if _BIL_MODE is not None:
        return _BIL_MODE
    env = os.environ.get("SASPRO_GPU_WARP_BILINEAR", "").strip().lower()
    if env in ("table", "quantised", "quantized"):
        _BIL_MODE = "table"
        return _BIL_MODE
    if env in ("exact", "float"):
        _BIL_MODE = "exact"
        return _BIL_MODE
    try:
        import cv2
        n = 24
        M = np.array([[1.0, 0.0, 0.37], [0.0, 1.0, 0.21]], np.float64)
        py, px = 10, 10
        ix, iy, qx, qy, sx, sy = _fixed_point_frac_at(M, py, px)
        d = np.zeros((n, n), np.float32)
        d[iy, ix] = 1.0                      # the (0,0) tap of the 2x2
        got = float(cv2.warpAffine(d, M, (n, n), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0.0)[py, px])
        fx, fy = sx - np.floor(sx), sy - np.floor(sy)
        w_exact = float(np.float32((1.0 - fx) * (1.0 - fy)))
        w_table = float(np.float32((1.0 - qx / INTER_TAB_SIZE)
                                   * (1.0 - qy / INTER_TAB_SIZE)))
        if abs(w_exact - w_table) < 1e-4:    # the probe failed to separate them
            return None
        de, dt = abs(got - w_exact), abs(got - w_table)
        if min(de, dt) > 1e-5:               # neither model — do not guess
            return None
        _BIL_MODE = "exact" if de < dt else "table"
        return _BIL_MODE
    except Exception:
        return None


PAD = 4            # the 8x8 support reaches 3 before and 4 after the base


def _pad_const(torch, F, src, value):
    """Frame the source in `PAD` pixels of `value`.

    This is what lets the gather drop its per-tap bounds test. cv2's
    BORDER_CONSTANT says any tap outside the source contributes border_value;
    with the source framed in border_value, simply clamping the index into the
    padded array produces exactly that — a tap one pixel out lands on the pad,
    and a tap fifty pixels out clamps to the pad as well, which holds the same
    value. So the result is identical and five compares, two ands and a
    `where` per tap disappear.

    An interior-only fast path was tried first and is worthless here: a
    registration warp fills the frame and shifts by more than the support
    radius, so every band touches an edge and never qualifies.
    """
    return F.pad(src.view(1, 1, src.shape[-2], src.shape[-1]),
                 (PAD, PAD, PAD, PAD), mode="constant",
                 value=float(value)).view(src.shape[-2] + 2 * PAD,
                                          src.shape[-1] + 2 * PAD)


def _gather(torch, srcp, hw, sx, sy, tab_lan, tab_bil,
            coords=None, lanczos=True, bilinear=True, bil_mode="exact"):
    """Produce both the Lanczos and the bilinear warp of `src`.

    These cannot share one neighbourhood, however tempting it looks. cv2
    quantises the coordinate to 1/32 for Lanczos4 but not for 32-bit-float
    bilinear — feeding the bilinear the quantised table is off by 7e-3. The
    two also disagree about the integer base near a pixel boundary, so the
    bilinear 2x2 is not always the middle of the Lanczos 8x8.

    The bilinear coordinate is kept in float64 here. cv2 carries its own in
    float32, which at an 8288-wide frame's right edge is a millipixel of
    coordinate error; recovering cv2's weights from delta images shows them
    sitting 0 to 1 ULP of a float32 coordinate away from the true fraction.
    Imitating that does not help — it tracks cv2 worse — so this is exact
    instead, and the module docstring records what that costs.

    The 2x2 is four reads against sixty-four, so computing it separately costs
    about 6% and buys correctness.
    """
    H, W = int(hw[0]), int(hw[1])          # the UNPADDED source shape
    Wp = W + 2 * PAD
    dev = srcp.device

    if coords is not None:
        ix, iy, qx, qy = coords
    else:
        # perspective path: cv2 quantises the double coordinate straight to
        # 1/32, with no intermediate fixed-point stage
        qxf = torch.round(sx * INTER_TAB_SIZE)
        qyf = torch.round(sy * INTER_TAB_SIZE)
        ix = torch.div(qxf, INTER_TAB_SIZE, rounding_mode="floor").long()
        iy = torch.div(qyf, INTER_TAB_SIZE, rounding_mode="floor").long()
        qx = (qxf.long() - ix * INTER_TAB_SIZE).clamp_(0, INTER_TAB_SIZE - 1)
        qy = (qyf.long() - iy * INTER_TAB_SIZE).clamp_(0, INTER_TAB_SIZE - 1)

    # BORDER_CONSTANT comes from the padding (see _pad_const), so a tap is a
    # clamp and a gather with no bounds test.
    flat = srcp.reshape(-1)

    def _row(yy):
        return (yy + PAD).clamp_(0, H + 2 * PAD - 1) * Wp

    def _tap_at(rowb, xx):
        return flat[rowb + (xx + PAD).clamp_(0, W + 2 * PAD - 1)]

    acc_hi = None
    if lanczos:
        # The 8 weight planes per axis are materialised once for the band.
        # Full frame that would be an (H, W, 8) tensor — 1.5 GB an axis at
        # 8288x5644 — which is what made banding necessary in the first place;
        # for one band it is ~80 MB and saves re-gathering the coefficient
        # table 72 times, once per tap.
        wx = tab_lan[qx]                       # (rows, W, 8)
        wy = tab_lan[qy]

        acc_hi = torch.zeros(sx.shape, dtype=torch.float32, device=dev)
        for j in range(8):
            rowb = _row(iy + (j - 3))
            wyj = wy[..., j]
            for i in range(8):
                acc_hi += _tap_at(rowb, ix + (i - 3)) * wx[..., i] * wyj
            del rowb
        del wx, wy

    if not bilinear:
        return acc_hi, None

    if bil_mode == "table":
        # cv2's remapBilinear looks its weights up in the same 1/32 table the
        # Lanczos path uses, so the 2x2 IS the centre of the 8x8 — same base,
        # same fraction, four taps already loaded. Cheaper than the exact
        # version, not a concession to it.
        #
        # The weights are formed the way cv2 forms them: the two 1-D pairs
        # first, then their four products, then the dot product in that order.
        # k/32 is exact in float32, so each is a single rounding and the
        # result matches bit for bit rather than to within an ULP or two.
        wx0, wx1 = tab_bil[qx, 0], tab_bil[qx, 1]
        wy0, wy1 = tab_bil[qy, 0], tab_bil[qy, 1]
        r0 = _row(iy)
        r1 = _row(iy + 1)
        acc_lo = (_tap_at(r0, ix) * (wy0 * wx0)
                  + _tap_at(r0, ix + 1) * (wy0 * wx1)
                  + _tap_at(r1, ix) * (wy1 * wx0)
                  + _tap_at(r1, ix + 1) * (wy1 * wx1))
        return acc_hi, acc_lo

    # bilinear: exact fractional weights, its own integer base
    ixb = torch.floor(sx)
    iyb = torch.floor(sy)
    fxb = (sx - ixb).to(torch.float32)
    fyb = (sy - iyb).to(torch.float32)
    ixb = ixb.long()
    iyb = iyb.long()
    r0 = _row(iyb)
    r1 = _row(iyb + 1)
    acc_lo = (
        _tap_at(r0, ixb) * (1 - fxb) * (1 - fyb)
        + _tap_at(r0, ixb + 1) * fxb * (1 - fyb)
        + _tap_at(r1, ixb) * (1 - fxb) * fyb
        + _tap_at(r1, ixb + 1) * fxb * fyb
    )
    return acc_hi, acc_lo


def _warp_antiring_once(img, M, out_wh, *,
                      perspective: bool = False,
                      border_value: float = 0.0,
                      star_sigma: float = 3.0,
                      grow_px: int = 6,
                      feather_px: float = 2.0,
                      envelope: bool = True,
                      envelope_pad_px: int = 1,
                      band_rows: int | None = None,
                      allow_cpu: bool = False,
                      status_cb=None):
    """
    GPU equivalent of `star_alignment._warp_antiring` for a 2-D frame.

    img          : (H, W) float array. NaN is preserved the way the CPU path
                   preserves it — satellite-trail no-data blooms through the
                   Lanczos support and stays NaN in the output.
    M            : 2x3 affine or 3x3 homography, FORWARD (source -> dest).
    out_wh       : (W, H) of the destination.
    band_rows    : output rows per Lanczos pass. None picks a size from the
                   frame width that keeps the 8x8 gather inside a few hundred
                   MB; it only affects memory, never the result.
    The remaining arguments mirror `_warp_antiring`'s defaults exactly.

    Returns a float32 (H, W) array, or None if the GPU path is unavailable or
    anything fails — the caller must treat None as "use the CPU path".
    """
    log = status_cb or (lambda *_: None)
    torch, dev = _resolve()
    if torch is None or dev is None:
        return None
    if str(dev.type) == "cpu" and not (
            allow_cpu or os.environ.get("SASPRO_GPU_WARP_ALLOW_CPU") == "1"):
        return None

    try:
        import torch.nn.functional as F
    except Exception:
        return None

    a = np.asarray(img)
    if a.ndim != 2:
        return None
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if out_w <= 0 or out_h <= 0:
        return None

    try:
        with torch.inference_mode():
            src = torch.from_numpy(
                np.ascontiguousarray(a, dtype=np.float32)).to(dev)

            # NaN must not poison the weighted sums of its neighbours beyond
            # where the CPU path spreads it, so carry it as a separate plane
            # and re-apply: a destination pixel is NaN iff any tap it reads
            # was NaN, which is what the CPU warp produces.
            nan_mask = torch.isnan(src)
            has_nan = bool(nan_mask.any())
            nan_plane = None
            if has_nan:
                src = torch.where(nan_mask, torch.zeros_like(src), src)
                nan_plane = _pad_const(torch, F, nan_mask.to(torch.float32), 0.0)
            del nan_mask
            shw = (src.shape[-2], src.shape[-1])
            src = _pad_const(torch, F, src, border_value)

            tab_lan, tab_bil = _tables(torch, dev)
            bmode = bilinear_mode()
            if bmode is None:
                # cv2 here does neither model; a silent mismatch on every
                # pixel of every frame is not worth a speed-up
                log("⚠️ GPU warp: could not identify this OpenCV build's "
                    "bilinear; using CPU. Override with "
                    "SASPRO_GPU_WARP_BILINEAR=table or =exact.")
                return None
            nrows = (_band_rows(out_h, out_w) if band_rows is None
                     else max(1, min(int(out_h), int(band_rows))))

            def _coords(y0, n):
                """Coordinates and, for affine, cv2's fixed-point indices."""
                if perspective:
                    sx, sy = _source_coords(torch, M, out_h, out_w, dev, True,
                                            y0, n)
                    return sx, sy, None
                ix, iy, qx, qy, sx, sy = _affine_fixed_point(
                    torch, M, out_h, out_w, dev, y0, n)
                return sx, sy, (ix, iy, qx, qy)

            # ── pass 1: the bilinear plane, in full ──────────────────────
            # It has to exist in full before any Lanczos row can be written:
            # the star threshold is a median over the whole of it and the
            # envelope clamp reads a 3x3 neighbourhood of it. Four taps per
            # pixel, so one float32 plane is the whole cost.
            lo = torch.empty((out_h, out_w), dtype=torch.float32, device=dev)
            bloom_lo = None
            if has_nan:
                bloom_lo = torch.zeros((out_h, out_w), dtype=torch.bool,
                                       device=dev)
            for y0 in range(0, out_h, nrows):
                n = min(nrows, out_h - y0)
                sx, sy, co = _coords(y0, n)
                _, lo_b = _gather(torch, src, shw, sx, sy, tab_lan, tab_bil,
                                  co, lanczos=False, bil_mode=bmode)
                lo[y0:y0 + n] = lo_b
                if has_nan:
                    _, nb = _gather(torch, nan_plane, shw, sx, sy, tab_lan,
                                    tab_bil, co, lanczos=False, bil_mode=bmode)
                    bloom_lo[y0:y0 + n] = nb.abs() > 1e-12
                del sx, sy, co, lo_b
            if has_nan:
                lo = torch.where(bloom_lo, torch.full_like(lo, float("nan")), lo)

            lo4 = lo.view(1, 1, out_h, out_w)

            # ── the star threshold: the one genuinely frame-wide step ────
            n_finite = (out_h * out_w if not has_nan
                        else int(torch.isfinite(lo).sum()))
            med, mad = _median_mad(torch, lo, n_finite)
            thr = med + float(star_sigma) * 1.4826 * mad
            # whether ANY pixel is a star is also global, but it needs only a
            # bool plane, not a float one plus its morphology
            blend = bool((lo4 > thr).any())

            # Everything else about the mask and the envelope is local, so it
            # runs per band off a slice of `lo` with a halo. Done frame-wide
            # it cost 3.8 GB peak — fine alone, fatal with several finalize
            # workers sharing one card.
            r_env = int(envelope_pad_px) if (envelope and envelope_pad_px > 0) else 0
            r_fea = (int(round(feather_px)) if (blend and feather_px
                                                and feather_px > 0) else 0)
            r_grow = int(grow_px) if (blend and grow_px > 0) else 0
            halo = max(r_env, r_grow + r_fea)
            big = torch.finfo(lo4.dtype).max

            # ── pass 2: the Lanczos plane, band by band ──────────────────
            # This is the expensive half — 64 taps a pixel — and the only
            # part that has to be banded. Each band is finished and copied
            # out before the next one allocates, so peak memory is `src`,
            # `lo` and one band's working set rather than the whole gather.
            res = np.empty((out_h, out_w), dtype=np.float32)
            for y0 in range(0, out_h, nrows):
                n = min(nrows, out_h - y0)
                sx, sy, co = _coords(y0, n)
                hi_b, _ = _gather(torch, src, shw, sx, sy, tab_lan, tab_bil,
                                  co, bilinear=False)
                bloom = None
                if has_nan:
                    nb, _ = _gather(torch, nan_plane, shw, sx, sy, tab_lan,
                                    tab_bil, co, bilinear=False)
                    bloom = nb.abs() > 1e-12
                    del nb
                hi4 = hi_b.view(1, 1, n, out_w)

                # the halo slice. Pooling it with padding=halo pads both of
                # its ends, which is right exactly where the slice end IS a
                # frame end; everywhere else the padded rows are trimmed off
                # before they can reach the output.
                y_a = max(0, y0 - halo)
                y_b = min(out_h, y0 + n + halo)
                ext = lo4[:, :, y_a:y_b]
                off = y0 - y_a

                # 1. envelope clamp — Lanczos held inside the bilinear range
                if r_env:
                    # cv2.erode/dilate ignore NaN (they are min/max over the
                    # element); max_pool2d would propagate it, so substitute
                    # the identity for each direction.
                    lo_min = _erode_rect(
                        torch, F, torch.nan_to_num(ext, nan=big), r_env)
                    lo_max = _dilate_rect(
                        torch, F, torch.nan_to_num(ext, nan=-big), r_env)
                    hi4 = torch.minimum(
                        torch.maximum(hi4, lo_min[:, :, off:off + n]),
                        lo_max[:, :, off:off + n])
                    del lo_min, lo_max

                # 2. star-disc blend
                if blend:
                    mb = (ext > thr).to(torch.float32)
                    if r_grow:
                        mb = _dilate_ellipse(torch, F, mb, r_grow)
                    if r_fea:
                        mb = _gaussian_blur(torch, F, mb, 2 * r_fea + 1,
                                            float(feather_px))
                    mb = mb[:, :, off:off + n]
                    out_b = hi4 * (1.0 - mb) + ext[:, :, off:off + n] * mb
                    del mb
                else:
                    out_b = hi4

                # 3. zero floor
                out_b = torch.clamp(out_b, min=0.0).view(n, out_w)
                if has_nan:
                    out_b = torch.where(bloom,
                                        torch.full_like(out_b, float("nan")),
                                        out_b)
                res[y0:y0 + n] = out_b.detach().to("cpu").numpy()
                del sx, sy, co, hi_b, hi4, out_b, bloom, ext

        return res
    except Exception as e:
        # Running out of VRAM is the one failure worth another attempt, so it
        # is re-raised as MemoryError for the wrapper to catch; everything
        # else means the GPU path cannot do this frame and the CPU should.
        txt = f"{type(e).__name__}: {e}"
        oom = isinstance(e, MemoryError) or "out of memory" in txt.lower() \
            or "OutOfMemory" in txt or "CUDA_ERROR_OUT_OF_MEMORY" in txt
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if oom:
            raise MemoryError(txt) from e
        log(f"⚠️ GPU warp failed ({txt}); using CPU.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# One card, many finalize workers
# ─────────────────────────────────────────────────────────────────────────────
# `_finalize_write_job` runs in a process pool — the log line reads
# "Finalize: 8 workers x 4 cv2 threads" — and every one of those processes
# would otherwise build its own CUDA context and ask for ~1.2 GB of working
# set. On an 8 GB card that is roughly 13 GB wanted, and on Windows/WDDM an
# oversubscribed card does NOT raise: the driver pages to system memory over
# PCIe and the run appears to hang with the GPU pegged. No exception means the
# out-of-memory retry never fires, so the only protection is not to let it
# happen.
#
# So a bounded number of processes may use the card, enforced with an
# OS-level file lock — the kernel drops it if a worker dies, which a lock file
# or a counter would not. A process that loses the race is marked denied for
# good and uses cv2 from then on: it never allocates, so it never builds a
# context, which is what actually bounds VRAM. The other workers keep their
# cores busy with cv2 warps rather than queueing, so the pool stays a hybrid
# instead of serialising behind the card.
#
# Slots: QSettings "stacking/align/gpu_warp_slots" or SASPRO_GPU_WARP_SLOTS.
# Default 1. Raise it only if the bench says a second concurrent warp helps,
# which needs the VRAM headroom to exist.
_SLOT_FH = None
_SLOT_DENIED = False


def _gpu_slots() -> int:
    env = os.environ.get("SASPRO_GPU_WARP_SLOTS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    try:
        from PyQt6.QtCore import QSettings as _QS
        return max(1, int(_QS().value("stacking/align/gpu_warp_slots", 1)))
    except Exception:
        return 1


def _claim_gpu_slot() -> bool:
    """Claim one of the card's slots for the lifetime of this process.

    Held, not acquired per frame: releasing between frames would let a second
    process build a context later and defeat the point.
    """
    global _SLOT_FH, _SLOT_DENIED
    if _SLOT_FH is not None:
        return True
    if _SLOT_DENIED:
        return False
    import tempfile
    try:
        import getpass
        who = getpass.getuser()
    except Exception:
        who = "user"
    for i in range(_gpu_slots()):
        path = os.path.join(tempfile.gettempdir(),
                            f"saspro_gpu_warp_{who}_{i}.lock")
        try:
            fh = open(path, "a+b")
            fh.write(b"\0")
            fh.flush()
            fh.seek(0)
        except Exception:
            continue
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _SLOT_FH = fh
            return True
        except Exception:
            try:
                fh.close()
            except Exception:
                pass
    _SLOT_DENIED = True
    return False


def warp_antiring_gpu(img, M, out_wh, *, band_rows=None, status_cb=None, **kw):
    """GPU anti-ring warp, falling back to smaller bands before giving up.

    The finalize pass runs a process pool, so several workers can be warping
    full frames on the same card at once and VRAM is not ours alone. Free-VRAM
    readings are no help here — `mem_get_info` does not aggregate across
    processes on Windows/WDDM, which this project already measured while
    sizing the detection pool — so the only reliable signal is the allocation
    failing. On an out-of-memory error, retry with a quarter of the band
    before conceding the frame to the CPU: a narrower band is slower, still
    far cheaper than 5 s of cv2.

    Returns float32 (H, W), or None meaning "use the CPU path".
    """
    log = status_cb or (lambda *_: None)
    torch, dev = _resolve()
    if torch is None or dev is None or str(dev.type) == "cpu":
        # nothing to contend for; let the usual checks below decline
        if not (kw.get("allow_cpu")
                or os.environ.get("SASPRO_GPU_WARP_ALLOW_CPU") == "1"):
            return None
    elif not _claim_gpu_slot():
        # another finalize worker owns the card; this one stays on cv2 and
        # keeps its core busy rather than queueing behind it
        return None
    rows = band_rows
    for attempt in range(3):
        try:
            return _warp_antiring_once(img, M, out_wh, band_rows=rows,
                                       status_cb=status_cb, **kw)
        except MemoryError as e:
            # only a memory failure is retried; a None return means the GPU
            # path declined this frame outright and retrying cannot help
            if rows is None:
                rows = _band_rows(int(out_wh[1]), int(out_wh[0]))
            rows //= 4
            if attempt == 2 or rows < 16:
                log(f"⚠️ GPU warp out of memory ({e}); using CPU.")
                return None
            log(f"GPU warp out of memory; retrying with band_rows={rows}")
    return None


if __name__ == "__main__":
    print(backend_note())