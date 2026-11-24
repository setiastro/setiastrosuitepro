# src/setiastro/saspro/quick_stack_gpu.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Quick Stack on an NVIDIA GPU.

    CPU workers   read a frame into a shared slot
    GPU process   upload -> dark / flat -> debayer -> bin -> star detection
    CPU workers   match the star list to the reference's
    GPU process   warp + accumulate -> trimmed mean

The GPU process owns the card for the whole run: VRAM is released when it
exits, and the app's own process never loads torch for this. It and the
workers are kept warm in a GpuSession (starting torch takes several seconds),
so the window can start them before the first frame is stacked.

Warping and accumulating is one compiled CUDA kernel. Ordinary torch ops moved
six times the memory and were ten times slower; this does the same arithmetic
as quick_stack_worker._accumulate_band and gives bit-identical sums (checked
once at start-up, on this machine's card, before any frame is trusted to it).
torch.cuda._compile_kernel is private, so everything here fails soft: no CUDA
card, an older torch, too little VRAM or a crash raises GpuUnavailable and the
caller stacks on the CPU.
"""
from __future__ import annotations

import contextlib
import itertools
import os
import queue
import threading
import time
from collections import deque

import numpy as np

from setiastro.saspro import quick_stack_worker as engine


class GpuUnavailable(RuntimeError):
    """This stack cannot run on the GPU; run it on the CPU."""


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────
KERNELS = r'''
/* raw int16 (value - 32768) -> calibrated float32 [0,1]: (v - dark) / flat. A zero sample is no
   data and stays 0; a result that lands on 0 is nudged off it. quick_stack_worker._calibrate_plane. */
extern "C" __global__ void qs_cal(const short* __restrict__ raw, const float* __restrict__ dark, const float* __restrict__ flat,
   float* __restrict__ out, const int* __restrict__ D) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long)D[0] * D[1]) return;
  const int u = (int)raw[i] + 32768; if (u == 0) { out[i] = 0.0f; return; }
  float v = (float)u / 65535.0f;
  if (D[2]) v = v - dark[i];
  if (D[3]) v = v / flat[i];
  if (v == 0.0f) v = 1e-9f;
  out[i] = v;
}

/* mosaic -> (H,W,3): mean of the same-colour samples in the 3x3, zeros (no data) left out; a zero
   sample is no data in every channel. The same rule as numba_utils._bilinear_from_mosaic. */
extern "C" __global__ void qs_deb(const float* __restrict__ m, float* __restrict__ out, const int* __restrict__ D) {
  const int H = D[0], W = D[1];
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long)H * W) return;
  float* o = out + i * 3; const float v = m[i];
  if (v == 0.0f) { o[0] = o[1] = o[2] = 0.0f; return; }
  const int y = (int)(i / W), x = (int)(i - (long)y * W), own = D[3 + ((y & 1) << 1 | (x & 1))];
  for (int c = 0; c < 3; ++c) {
    if (c == own) { o[c] = v; continue; }
    double s = 0.0; int n = 0;
    for (int dy = -1; dy <= 1; ++dy) { const int yy = y + dy; if (yy < 0 || yy >= H) continue;
      for (int dx = -1; dx <= 1; ++dx) { const int xx = x + dx; if (xx < 0 || xx >= W) continue;
        if (D[3 + ((yy & 1) << 1 | (xx & 1))] != c) continue;
        const float q = m[(long)yy * W + xx]; if (q != 0.0f) { s += q; ++n; } } }
    o[c] = n ? (float)(s / n) : 0.0f;
  }
}

/* mean over channels, then over exact ds x ds blocks: the grid stars are found on */
extern "C" __global__ void qs_bin(const float* __restrict__ src, float* __restrict__ out, const int* __restrict__ D) {
  const int W = D[1], C = D[2], ds = D[3], h = D[4], w = D[5];
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long)h * w) return;
  const int y = (int)(i / w), x = (int)(i - (long)y * w); double s = 0.0;
  for (int dy = 0; dy < ds; ++dy) for (int dx = 0; dx < ds; ++dx) {
    const float* p = src + ((long)(y * ds + dy) * W + (x * ds + dx)) * C; double g = 0.0;
    for (int c = 0; c < C; ++c) g += p[c]; s += g / C; }
  out[i] = (float)(s / (double)(ds * ds));
}

/* bilinear inverse warp straight into the running sum / count / two highest / two lowest */
extern "C" __global__ void qs_acc(const float* __restrict__ src, const double* __restrict__ P, const float* __restrict__ off, const int* __restrict__ D,
   float* __restrict__ sumv, unsigned short* __restrict__ cnt, float* __restrict__ hi0, float* __restrict__ hi1, float* __restrict__ lo0, float* __restrict__ lo1) {
  const int H = D[0], W = D[1], C = D[2], sH = D[3], sW = D[4];
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long)H * W) return;
  const int y = (int)(i / W), x = (int)(i - (long)y * W);
  const double sx = P[1] * y + P[2], sy = P[4] * y + P[5];
  const double xs = sx + P[0] * x, ys = sy + P[3] * x;
  if (xs < 0.0 || ys < 0.0) return;
  const int ix = (int)xs, iy = (int)ys; if (ix >= sW - 1 || iy >= sH - 1) return;
  const float fx = (float)(xs - ix), fy = (float)(ys - iy);
  const long b0 = ((long)iy * sW + ix) * C, b1 = b0 + (long)sW * C;
  for (int c = 0; c < C; ++c) {
    const float a = src[b0 + c], b = src[b0 + C + c], d = src[b1 + c], e = src[b1 + C + c];
    if (a == 0.0f || b == 0.0f || d == 0.0f || e == 0.0f) continue;
    const float t = a + (b - a) * fx, u = d + (e - d) * fx, v = t + (u - t) * fy;
    if (!(fabsf(v) < 1e30f)) continue;
    const float r = v - off[c]; const long k = i * C + c;
    sumv[k] += r; cnt[k] += 1;
    if (r > hi0[k]) { hi1[k] = hi0[k]; hi0[k] = r; } else if (r > hi1[k]) hi1[k] = r;
    if (r < lo0[k]) { lo1[k] = lo0[k]; lo0[k] = r; } else if (r < lo1[k]) lo1[k] = r;
  }
}

/* trimmed mean, the rule of quick_stack_worker._finish */
extern "C" __global__ void qs_fin(const float* __restrict__ sumv, const unsigned short* __restrict__ cnt, const float* __restrict__ hi0, const float* __restrict__ hi1,
   const float* __restrict__ lo0, const float* __restrict__ lo1, float* __restrict__ out, const int* __restrict__ D) {
  const long N = (long)D[0] * D[1] * D[2], k = (long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= N) return;
  const int n = cnt[k]; if (!n) { out[k] = 0.0f; return; }
  double s = sumv[k]; int dr = 0;
  if (n >= D[3]) { s -= hi0[k]; s -= lo0[k]; dr = 2; if (n >= D[4]) { s -= hi1[k]; s -= lo1[k]; dr = 4; } }
  out[k] = (float)(s / (n - dr));
}
'''
KERNEL_NAMES = ("qs_cal", "qs_deb", "qs_bin", "qs_acc", "qs_fin")
_CFA = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}
VRAM_RESERVE = 0.8e9          # star detection work space and whatever else is on the card
MAX_HELD = 12                 # frames on the card waiting for their match


def _compile(torch) -> dict:
    """{name: kernel}. --fmad=false keeps the arithmetic the CPU kernel's."""
    if not hasattr(torch.cuda, "_compile_kernel"):
        raise GpuUnavailable(f"this PyTorch ({torch.__version__}) cannot compile CUDA kernels")
    opts = ["--fmad=false"]
    try:                                # all four in one compile
        from torch.cuda import _utils as cu
        ptx, _ = cu._nvrtc_compile(KERNELS, KERNEL_NAMES[0], None, None, opts)
        mod = cu._cuda_load_module(ptx, list(KERNEL_NAMES))
        return {n: mod[n] for n in KERNEL_NAMES}
    except Exception:
        return {n: torch.cuda._compile_kernel(KERNELS, n, nvcc_options=opts)
                for n in KERNEL_NAMES}


def _grid(n: int):
    return ((n + 255) // 256, 1, 1)


class _Run:
    """One stack on the card."""

    def __init__(self, torch, dev, K, gd, sa, token, p):
        self.torch, self.dev, self.K, self.gd, self.sa = torch, dev, K, gd, sa
        self.token, self.p = token, p
        H, W, C = self.H, self.W, self.C = p["H"], p["W"], p["nch"]
        ds = self.ds = p["ds"]
        self.h, self.w = H // ds, W // ds
        N = H * W * C
        free, _total = torch.cuda.mem_get_info()
        fixed = N * (4 + 2 + 16) + N * 4                          # accumulators + scratch frame
        fixed += H * W * 4 * (len(p["masters"] or {}) + (1 if C == 3 else 0))     # masters, mosaic
        held = max(H * W * 2, N * 4 if p["any_f32"] else 0) + self.h * self.w * 4
        self.max_held = int(min(MAX_HELD, (free - fixed - VRAM_RESERVE) // held))
        if self.max_held < 2:
            raise GpuUnavailable(f"needs about {(fixed + 2 * held + VRAM_RESERVE) / 1e9:.1f} GB of "
                                 f"video memory, {free / 1e9:.1f} GB is free")
        shape = (H, W, C)
        inf = float("inf")
        full = lambda v: torch.full(shape, v, dtype=torch.float32, device=dev)   # noqa: E731
        self.sumv = full(0.0)
        self.cnt = torch.zeros(shape, dtype=torch.int16, device=dev)
        self.hi0, self.hi1, self.lo0, self.lo1 = full(-inf), full(-inf), full(inf), full(inf)
        self.scratch = torch.empty(shape, dtype=torch.float32, device=dev)
        ints = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)          # noqa: E731
        self.D_deb = {bp: ints(H, W, C, *codes) for bp, codes in _CFA.items()}
        # The card calibrates raw 16-bit mosaics and mono planes only. Masters
        # fitted to frames that are colour already go on in the loaders.
        masters = {w: d for w, d in (p["masters"] or {}).items()
                   if len(d["shape"]) == 2 and not d.get("colour")}
        self.D_cal = ints(H, W, int("dark" in masters), int("flat" in masters))
        self.dark, self.flat = (self._master(masters.get(w), torch) for w in ("dark", "flat"))
        self.mosaic = torch.empty((H, W), dtype=torch.float32, device=dev) if C == 3 else None
        self.D_bin = ints(H, W, C, ds, self.h, self.w)
        self.D_acc = ints(H, W, C, H, W)
        self.D_fin = ints(H, W, C, engine.MIN_TRIM_FRAMES, engine.FULL_TRIM_FRAMES)
        self.held: dict = {}                 # idx -> (meta, tensor, sky, gray)
        self.ref_sky = None
        self.ladder = p["ladder"]
        self.slots = engine.SharedBlock.attach(p["slots"])
        self.out = engine.SharedBlock.attach(p["out"])
        self.cap = p["cap"]
        # Page-lock the slots: uploads become direct DMA (about twice as fast).
        self.registered = None
        with contextlib.suppress(Exception):
            a = self.slots.view((self.slots.nbytes,), np.uint8)
            rc = torch.cuda.cudart().cudaHostRegister(int(a.ctypes.data), int(a.nbytes), 0)
            if int(getattr(rc, "value", rc)) == 0:
                self.registered = int(a.ctypes.data)

    # -- pixels ------------------------------------------------------------
    def _master(self, desc, torch):
        """A calibration plane on the card (a one-pixel stand-in when there is none)."""
        if not desc:
            return torch.zeros((1,), dtype=torch.float32, device=self.dev)
        b = engine.SharedBlock.attach(desc)
        try:
            if tuple(desc["shape"]) != (self.H, self.W):
                raise GpuUnavailable("the master's size is not the frames' size")
            return torch.from_numpy(b.view((self.H, self.W), np.float32).copy()).to(self.dev)
        finally:
            b.close()

    def _to_hwc(self, meta, t):
        """The frame as (H,W,C) float32 on the card, calibrated."""
        if meta["kind"] == "f32":
            return t
        g = _grid(self.H * self.W)
        if self.C == 1:
            self.K["qs_cal"](grid=g, block=(256, 1, 1),
                             args=[t, self.dark, self.flat, self.scratch, self.D_cal])
        else:
            self.K["qs_cal"](grid=g, block=(256, 1, 1),
                             args=[t, self.dark, self.flat, self.mosaic, self.D_cal])
            self.K["qs_deb"](grid=g, block=(256, 1, 1),
                             args=[self.mosaic, self.scratch, self.D_deb[meta["bayer"]]])
        return self.scratch

    def _sky(self, x):
        torch = self.torch
        s = x[::8, ::8]
        out = []
        for c in range(self.C):
            v = s[..., c]
            v = v[(v != 0) & torch.isfinite(v)]
            out.append(float(v.median()) if v.numel() else 0.0)
        return np.asarray(out, np.float32)

    def _points(self, gray, level):
        cfg = self.p["cfg"]
        r = self.gd.detect_stars_full(
            gray, detection_sigma=self.ladder[level], min_area=cfg["minarea"],
            max_stars=20000, min_fwhm=cfg["min_fwhm"], max_ellipticity=cfg["max_ellipticity"])
        if r is None or len(r["x"]) == 0:
            return np.empty((0, 2), np.float32)
        return self.sa._grid_balance_uniform(
            np.asarray(r["x"], np.float32), np.asarray(r["y"], np.float32),
            self.h, self.w, (4, 4), 25, cfg["limit_stars"])

    def frame(self, conn, idx, slot, meta, is_ref):
        torch = self.torch
        base = slot * self.cap
        if meta["kind"] == "u16":
            src = torch.from_numpy(self.slots.view((self.H, self.W), np.int16, base))
        else:
            src = torch.from_numpy(self.slots.view((self.H, self.W, self.C), np.float32, base))
        t = src.to(self.dev, non_blocking=True)
        torch.cuda.synchronize()                                  # the slot is free once copied
        conn.send(("free", self.token, slot))
        x = self._to_hwc(meta, t)
        gray = torch.empty((self.h, self.w), dtype=torch.float32, device=self.dev)
        self.K["qs_bin"](grid=_grid(self.h * self.w), block=(256, 1, 1), args=[x, gray, self.D_bin])
        sky = self._sky(x)
        if is_ref:
            levels = [self._points(gray, k) for k in range(len(self.ladder))]
            self.ref_sky = sky
            if not self.p["ref_only"]:
                self._accumulate(meta, t, sky, np.array([[1.0, 0, 0], [0, 1.0, 0]]))
            conn.send(("ref", self.token, levels))
            conn.send(("acc_done", self.token, idx, 0.0))
            return
        self.held[idx] = (meta, t, sky, gray)
        conn.send(("pts", self.token, idx, 0, self._points(gray, 0)))

    def redetect(self, conn, idx, level):
        conn.send(("pts", self.token, idx, level, self._points(self.held[idx][3], level)))

    def _accumulate(self, meta, t, sky, T):
        inv = np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2]
        torch = self.torch
        P = torch.tensor(inv.astype(np.float64).ravel(), device=self.dev)
        off = torch.tensor((sky - self.ref_sky).astype(np.float32), device=self.dev)
        self.K["qs_acc"](grid=_grid(self.H * self.W), block=(256, 1, 1),
                         args=[self._to_hwc(meta, t), P, off, self.D_acc, self.sumv, self.cnt,
                               self.hi0, self.hi1, self.lo0, self.lo1])
        return off

    def accumulate(self, conn, idx, T):
        meta, t, sky, _gray = self.held.pop(idx)
        self._accumulate(meta, t, sky, T)
        conn.send(("acc_done", self.token, idx, float((sky - self.ref_sky).mean())))

    def finish(self, conn):
        torch = self.torch
        self.held.clear()
        self.scratch = self.mosaic = None
        torch.cuda.empty_cache()
        out = torch.empty((self.H, self.W, self.C), dtype=torch.float32, device=self.dev)
        self.K["qs_fin"](grid=_grid(self.H * self.W * self.C), block=(256, 1, 1),
                         args=[self.sumv, self.cnt, self.hi0, self.hi1, self.lo0, self.lo1,
                               out, self.D_fin])
        cov = int(self.cnt.max())
        dst = torch.from_numpy(self.out.view((self.H, self.W, self.C), np.float32))
        dst.copy_(out)
        conn.send(("result", self.token, cov))

    def close(self):
        if self.registered is not None:
            with contextlib.suppress(Exception):
                self.torch.cuda.synchronize()
                self.torch.cuda.cudart().cudaHostUnregister(self.registered)
        self.registered = None
        self.held = {}
        for n in ("sumv", "cnt", "hi0", "hi1", "lo0", "lo1", "scratch", "mosaic", "dark", "flat"):
            self.__dict__.pop(n, None)
        self.slots.close()
        self.out.close()
        with contextlib.suppress(Exception):
            self.torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# The GPU process
# ─────────────────────────────────────────────────────────────────────────────
def _self_test(torch, dev, K) -> None:
    """The kernels against the CPU code, on this card: exact or not used."""
    rng = np.random.default_rng(7)
    H, W = 96, 131
    C = 3
    raw = rng.integers(0, 65535, (H, W), dtype=np.uint16)
    raw[5:9, 20:40] = 0
    stored = (raw.astype(np.int32) - 32768).astype(np.int16)
    dark = rng.normal(0.02, 0.004, (H, W)).astype(np.float32)
    flat = rng.uniform(0.4, 1.3, (H, W)).astype(np.float32)
    raw01 = raw.astype(np.float32) / 65535.0
    # A sample equal to its dark must not read as a hole.
    dark[50, 60], dark[51, 61] = raw01[50, 60], raw01[51, 61]
    from setiastro.saspro.legacy.numba_utils import debayer_fits_fast
    t_raw, t_dark, t_flat = (torch.from_numpy(v).to(dev) for v in (stored, dark, flat))
    none = torch.zeros((1,), dtype=torch.float32, device=dev)
    for use_dark, use_flat in ((0, 0), (1, 0), (0, 1), (1, 1)):
        m = torch.empty((H, W), dtype=torch.float32, device=dev)
        K["qs_cal"](grid=_grid(H * W), block=(256, 1, 1),
                    args=[t_raw, t_dark if use_dark else none, t_flat if use_flat else none, m,
                          torch.tensor([H, W, use_dark, use_flat], dtype=torch.int32, device=dev)])
        want = engine.calibrate_image(raw01, dark if use_dark else None,
                                      flat if use_flat else None)
        if not np.array_equal(m.cpu().numpy(), want):
            raise GpuUnavailable("the calibration self-test failed on this card")
        for bp in _CFA:
            x = torch.empty((H, W, C), dtype=torch.float32, device=dev)
            K["qs_deb"](grid=_grid(H * W), block=(256, 1, 1),
                        args=[m, x, torch.tensor([H, W, C, *_CFA[bp]], dtype=torch.int32,
                                                 device=dev)])
            cpu = debayer_fits_fast(want, bp, method="bilinear")
            if np.abs(x.cpu().numpy() - cpu).max() > 1e-6:
                raise GpuUnavailable("the debayer self-test failed on this card")

    for ds in (1, 3):
        h, w = H // ds, W // ds
        src = rng.normal(0.2, 0.1, (H, W, C)).astype(np.float32)
        binned = torch.empty((h, w), dtype=torch.float32, device=dev)
        K["qs_bin"](grid=_grid(h * w), block=(256, 1, 1),
                    args=[torch.from_numpy(src).to(dev), binned,
                          torch.tensor([H, W, C, ds, h, w], dtype=torch.int32, device=dev)])
        want = src[:h * ds, :w * ds].astype(np.float64).mean(2).reshape(h, ds, w, ds).mean((1, 3))
        if np.abs(binned.cpu().numpy() - want).max() > 1e-6:
            raise GpuUnavailable("the binning self-test failed on this card")

    # Enough frames, with values on both sides of zero, that every branch of
    # the accumulator runs (highest, second highest, lowest, second lowest),
    # over shifts, a small rotation and a half turn; then the trim itself.
    A = {n: np.zeros((H, W, C), d) for n, d in engine.SharedArrays.SPEC}
    A["hi0"][...] = A["hi1"][...] = -np.inf
    A["lo"][...] = A["lo1"][...] = np.inf
    mk = lambda v: torch.full((H, W, C), v, dtype=torch.float32, device=dev)      # noqa: E731
    inf = float("inf")
    g = [mk(0.0), torch.zeros((H, W, C), dtype=torch.int16, device=dev), mk(-inf), mk(-inf),
         mk(inf), mk(inf)]
    D_acc = torch.tensor([H, W, C, H, W], dtype=torch.int32, device=dev)
    n_frames = engine.FULL_TRIM_FRAMES + 3
    for k in range(n_frames):
        f = rng.normal(0.0, 0.05, (H, W, C)).astype(np.float32) + np.float32(0.02)
        f[10:20, 30:50] = 0.0
        f[40, 40] = np.nan
        th = np.radians((0.0, 11.0, 180.0, -0.6)[k % 4])
        c, n = np.cos(th), np.sin(th)
        tx, ty = ((0.4, -0.7), (14.2, -9.7), (W - 1.3, H - 2.6), (3.1, 0.2))[k % 4]
        inv = np.array([[c, -n, tx], [n, c, ty]])
        off = np.array([0.011, -0.003, 0.002], np.float32) * np.float32(k % 3)
        for b in range(engine.N_BANDS):
            engine._accumulate_band(f, inv, off, H * b // engine.N_BANDS,
                                    H * (b + 1) // engine.N_BANDS, A["sumv"], A["cnt"],
                                    A["hi0"], A["hi1"], A["lo"], A["lo1"])
        K["qs_acc"](grid=_grid(H * W), block=(256, 1, 1),
                    args=[torch.from_numpy(f).to(dev), torch.tensor(inv.ravel(), device=dev),
                          torch.tensor(off, device=dev), D_acc, *g])
    for name, t in zip(("sumv", "cnt", "hi0", "hi1", "lo", "lo1"), g):
        x1, x2 = A[name], t.cpu().numpy()
        if name == "cnt":
            x2 = x2.astype(np.uint16)
        if not np.array_equal(x1, x2, equal_nan=True):
            raise GpuUnavailable(f"the accumulator self-test failed on this card ({name})")
    cpu = np.empty((H, W, C), np.float32)
    engine._finish(A["sumv"], A["cnt"], A["hi0"], A["hi1"], A["lo"], A["lo1"], cpu,
                   engine.MIN_TRIM_FRAMES, engine.FULL_TRIM_FRAMES)
    out = torch.empty((H, W, C), dtype=torch.float32, device=dev)
    K["qs_fin"](grid=_grid(H * W * C), block=(256, 1, 1),
                args=[*g, out, torch.tensor([H, W, C, engine.MIN_TRIM_FRAMES,
                                             engine.FULL_TRIM_FRAMES],
                                            dtype=torch.int32, device=dev)])
    if not np.array_equal(cpu, out.cpu().numpy()):
        raise GpuUnavailable("the trimmed-mean self-test failed on this card")


def _boot():
    from setiastro.saspro.runtime_torch import import_torch
    try:
        torch = import_torch(prefer_cuda=True, status_cb=lambda *_: None, allow_install=False)
    except Exception as e:
        raise GpuUnavailable(f"PyTorch is not available ({type(e).__name__})")
    if not torch.cuda.is_available():
        raise GpuUnavailable("no NVIDIA GPU is available to PyTorch")
    from setiastro.saspro import gpu_star_detect as gd
    from setiastro.saspro import star_alignment as sa
    dev = torch.device("cuda")
    gd.reset_backend()
    gd._resolve()
    K = _compile(torch)
    _self_test(torch, dev, K)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return torch, dev, K, gd, sa, {"name": torch.cuda.get_device_name(0),
                                   "torch": str(torch.__version__), "free": free, "total": total}


# Accumulating and dropping free memory, so they jump the queue.
_FIRST = ("acc", "drop", "cancel", "finish", "quit", "detect")


def _server_main(conn) -> None:
    engine.die_with_parent()
    try:
        torch, dev, K, gd, sa, info = _boot()
    except Exception as e:
        with contextlib.suppress(Exception):
            conn.send(("fail", str(e) if isinstance(e, GpuUnavailable)
                       else f"{type(e).__name__}: {e}"))
        return
    conn.send(("ready", info))
    run = None
    todo: deque = deque()
    while True:
        try:
            while not todo or conn.poll(0):
                todo.append(conn.recv())
        except (EOFError, OSError):
            break
        k = next((i for i, m in enumerate(todo) if m[0] in _FIRST), 0)
        m = todo[k]
        del todo[k]
        try:
            if m[0] == "quit":
                break
            if m[0] == "begin":
                if run is not None:
                    run.close()
                run = None
                run = _Run(torch, dev, K, gd, sa, m[1], m[2])
                conn.send(("begun", run.token, run.max_held))
            elif run is None or m[1] != run.token:
                if m[0] == "cancel":
                    conn.send(("closed", m[1]))
            elif m[0] == "frame":
                run.frame(conn, m[2], m[3], m[4], m[5])
            elif m[0] == "detect":
                run.redetect(conn, m[2], m[3])
            elif m[0] == "acc":
                run.accumulate(conn, m[2], m[3])
            elif m[0] == "drop":
                run.held.pop(m[2], None)
            elif m[0] == "finish":
                run.finish(conn)
            elif m[0] == "cancel":
                tok = run.token
                run.close()
                run = None
                conn.send(("closed", tok))
        except Exception as e:
            why = ("ran out of video memory" if "out of memory" in str(e).lower()
                   else str(e) if isinstance(e, GpuUnavailable) else f"{type(e).__name__}: {e}")
            tok = m[1] if len(m) > 1 else None
            if run is not None:
                with contextlib.suppress(Exception):
                    run.close()
                run = None
            with contextlib.suppress(Exception):
                conn.send(("fatal", tok, why))
    with contextlib.suppress(Exception):
        if run is not None:
            run.close()


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────
def hardware_acceleration_on() -> bool:
    """Stacking Settings -> "Use hardware acceleration if available"."""
    try:
        from PyQt6.QtCore import QSettings
        v = QSettings().value("stacking/use_hardware_accel", True)
        return v if isinstance(v, bool) else str(v).strip().lower() not in ("0", "false", "no", "off", "")
    except Exception:
        return True


def nvidia_driver_present() -> bool:
    """Is there an NVIDIA driver at all? Only then is the GPU process worth starting."""
    import sys
    if sys.platform == "win32":
        return os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                           "System32", "nvcuda.dll"))
    if sys.platform.startswith("linux"):
        if os.path.exists("/proc/driver/nvidia/version"):
            return True
        from ctypes.util import find_library
        return bool(find_library("cuda"))
    return False


def _session_workers() -> int:
    """Loader processes: half the cores, 12 at most, fewer when memory is short (each
    commits about 1 GB, mostly for libraries, and holds about 0.1 GB of its own)."""
    n = max(2, min(12, (os.cpu_count() or 4) // 2))
    try:
        import psutil
        return max(2, min(n, int(psutil.virtual_memory().available * 0.6 // 1e9)))
    except Exception:
        return min(n, 4)


class GpuSession:
    """The GPU process and the CPU workers, kept warm between stacks.

    state: "new", "starting", "ready", "failed" (reason says why), "closed".
    start() returns at once; the first stack waits for whatever is left."""

    def __init__(self, workers: int = 0):
        self.state, self.reason, self.info = "new", "", {}
        self.workers = workers or _session_workers()
        self._lock = threading.RLock()
        self._ready = threading.Event()
        self.run_lock = threading.Lock()
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self._proc = self._conn = self._pool = None
        self._send_lock = threading.Lock()
        self._tokens = itertools.count(1)

    # -- life --------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self.state != "new":
                return
            why = (None if hardware_acceleration_on()
                   else "hardware acceleration is turned off in Stacking Settings")
            if why is None and not nvidia_driver_present():
                why = "no NVIDIA GPU"
            if why is not None:
                self.state, self.reason = "failed", why
                self._ready.set()
                return
            self.state = "starting"
        threading.Thread(target=self._start, name="quick-stack-gpu-start", daemon=True).start()

    def _start(self) -> None:
        try:
            if engine._executor_kind() != "process":
                raise GpuUnavailable("worker processes are not available")
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            parent, child = ctx.Pipe(duplex=True)
            proc = ctx.Process(target=_server_main, args=(child,), daemon=True)
            proc.start()
            child.close()
            with self._lock:
                if self.state == "closed":
                    proc.kill()
                    return
                self._proc, self._conn = proc, parent
            self._warm_pool()
            threading.Thread(target=self._reader, args=(parent,), name="quick-stack-gpu-read",
                             daemon=True).start()
        except Exception as e:
            self._fail(str(e) or type(e).__name__)

    def _warm_pool(self) -> None:
        p = self.pool()
        for _ in range(self.workers):
            p.submit(engine.warm_worker)        # each submit starts a process
            time.sleep(0.25)                    # let the window breathe between them

    def pool(self):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        with self._lock:
            if self.state == "closed":
                raise GpuUnavailable("the session is closed")
            if self._pool is None:
                self._pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"),
                                                 initializer=engine.init_session_worker)
            return self._pool

    def drop_pool(self) -> None:
        """Forget a pool that broke; the next use makes a new one."""
        with self._lock:
            p, self._pool = self._pool, None
        if p is not None:
            engine._shutdown(p, "process", hard=True)

    def _reader(self, conn) -> None:
        try:
            while True:
                m = conn.recv()
                if m[0] == "ready":
                    with self._lock:
                        if self.state != "closed":
                            self.state, self.info = "ready", m[1]
                    self._ready.set()
                elif m[0] == "fail":
                    self._fail(m[1])
                else:
                    self.events.put(m)
        except (EOFError, OSError):
            pass
        except Exception as e:                                    # a torn message
            self._fail(f"{type(e).__name__}: {e}")
        if self.state in ("starting", "ready"):
            self._fail("the GPU process stopped unexpectedly")
        self.events.put(("dead", None))

    def _fail(self, why: str) -> None:
        with self._lock:
            if self.state != "closed":
                self.state, self.reason = "failed", why
        self._ready.set()

    def wait_ready(self, should_stop=lambda: False, timeout: float = 90.0) -> bool:
        self.start()
        end = time.monotonic() + timeout
        while not self._ready.wait(0.2):
            if should_stop():
                raise InterruptedError("Quick Stack was cancelled.")
            if time.monotonic() > end:
                self._fail("the GPU took too long to start")
        return self.state == "ready"

    def send(self, *m) -> None:
        with self._send_lock:
            try:
                self._conn.send(m)
            except (OSError, ValueError, AttributeError) as e:
                raise GpuUnavailable("the GPU process stopped unexpectedly") from e

    def new_token(self) -> int:
        return next(self._tokens)

    def close(self) -> None:
        """Stop everything now. Safe to call from any thread, more than once."""
        with self._lock:
            if self.state == "closed":
                return
            self.state = "closed"
            proc, conn, pool = self._proc, self._conn, self._pool
            self._proc = self._conn = self._pool = None
        self._ready.set()
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.send(("quit",))
            with contextlib.suppress(Exception):
                conn.close()
        if proc is not None:
            def reap(p=proc):
                p.join(3)
                if p.is_alive():
                    p.kill()
            threading.Thread(target=reap, daemon=True).start()
        if pool is not None:
            engine._shutdown(pool, "process", hard=True)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration (runs in the app's Quick Stack thread)
# ─────────────────────────────────────────────────────────────────────────────
def stack_frames_gpu(paths, session: GpuSession, *, ref_path=None, log=None, progress=None,
                     should_stop=None, masters=None, mdesc=None) -> engine.StackResult:
    """engine.stack_frames on the GPU. Raises GpuUnavailable if it cannot.

    mdesc: the descriptor of a MasterSet the caller loaded (and closes) for
    `masters`; without it the masters are loaded here."""
    say = log or (lambda s: None)
    step = progress or (lambda done, total, msg: None)
    stopped = should_stop or (lambda: False)
    t0 = time.perf_counter()
    step(0, 0, "Starting the GPU…")
    if not session.wait_ready(stopped):
        raise GpuUnavailable(session.reason or "the GPU is not ready")
    with session.run_lock:
        return _run(paths, session, ref_path, say, step, stopped, t0, masters, mdesc)


def _run(paths, session, ref_path, say, step, stopped, t0, masters=None, mdesc=None):
    if not masters or mdesc:
        return _stack(paths, session, ref_path, say, step, stopped, t0,
                      masters if mdesc else None, mdesc or None)
    step(0, 0, "Loading the masters…")
    mset = engine.MasterSet.load(masters, paths, say)
    try:
        used = mset.choice if mset.blocks else None
        return _stack(paths, session, ref_path, say, step, stopped, t0, used,
                      mset.descriptor() if used else None)
    finally:
        mset.close()


def _stack(paths, session, ref_path, say, step, stopped, t0, masters, mdesc):
    from concurrent.futures import ThreadPoolExecutor
    paths = [os.path.normpath(p) for p in paths]
    cfg = engine.align_settings()
    (H, W), nch, ref_hdr, ref_path, ds, lines = _reference(session, paths, ref_path, cfg, stopped,
                                                           mdesc)
    cfg["downsample"] = ds
    for line in lines:
        say(line)
    frames = paths
    ref_i = engine.index_of(paths, ref_path)
    ref_only = ref_i < 0                            # aligned to, not stacked
    if ref_only:
        paths, ref_i = [ref_path] + paths, 0
    with ThreadPoolExecutor(8) as ex:
        as_u16 = list(ex.map(engine.quick_u16, paths))
    if any(d.get("colour") for d in (mdesc or {}).values()):
        # Masters fitted to colour frames: a raw 16-bit frame among them does
        # not fit, and the loaders say so and skip it.
        as_u16 = [False] * len(as_u16)
    any_f32 = not all(as_u16)
    cap = H * W * 2 if not any_f32 else max(H * W * 2, H * W * nch * 4)
    n_slots = session.workers + 4
    with contextlib.suppress(Exception):
        import psutil
        n_slots = min(n_slots, max(4, int(psutil.virtual_memory().available * 0.25 // cap)))
    say(f"{len(frames)} frames, {W} x {H}{f' x {nch}' if nch > 1 else ''}, reference "
        f"{os.path.basename(ref_path)}{' (aligned to, not stacked)' if ref_only else ''}, "
        f"{session.info.get('name', 'GPU')} with {session.workers} loader processes")

    token = session.new_token()
    slots = out = None
    futures: set = set()
    result = None
    try:
        slots = engine.SharedBlock.create(n_slots * cap)
        out = engine.SharedBlock.create(H * W * nch * 4)
        while True:                                               # stale messages of an old run
            try:
                session.events.get_nowait()
            except queue.Empty:
                break
        session.send("begin", token, {
            "H": H, "W": W, "nch": nch, "ds": ds, "any_f32": any_f32, "cap": cap,
            "cfg": cfg, "ladder": engine._sigma_ladder(cfg["det_sigma"]), "masters": mdesc,
            "ref_only": ref_only,
            "slots": slots.descriptor(), "out": out.descriptor()})
        result = _loop(paths, as_u16, ref_i, ref_only, (H, W), nch, ds, cfg, cap, n_slots, mdesc, session,
                       token, slots, out, futures, step, stopped, say)
        image, cov, oks, skipped = result
    finally:
        for f in futures:
            f.cancel()
        # The process lets go of the shared memory before it goes away.
        try:
            session.send("cancel", token)
            _await(session, token, "closed", 15.0)
        except Exception:
            pass
        if slots is not None:
            slots.close(unlink=True)
        if out is not None:
            out.close(unlink=True)
        # The idle workers still have the slots mapped: let that memory go.
        with contextlib.suppress(Exception):
            for _ in range(session.workers):
                session.pool().submit(engine.forget_blocks)
    engine.require_stack(oks, skipped, frames)
    return engine.make_result(image, ref_hdr, nch, oks, skipped, frames, cov, t0, say, "GPU",
                              masters, ref_path)


def _reference(session, paths, ref_path, cfg, stopped, mdesc):
    """engine.reference_job on a worker, with the window alive and Cancel working."""
    from concurrent.futures import BrokenExecutor, TimeoutError as Waiting

    def broken(e):
        session.drop_pool()
        return GpuUnavailable(f"the worker processes stopped ({type(e).__name__})")
    try:
        f = session.pool().submit(engine.reference_job, paths, ref_path, cfg, mdesc)
    except GpuUnavailable:
        raise
    except Exception as e:
        raise broken(e) from e
    while True:
        try:
            return f.result(timeout=0.2)
        except Waiting:
            if stopped():
                f.cancel()
                raise InterruptedError("Quick Stack was cancelled.")
        except BrokenExecutor as e:                 # (it is a RuntimeError)
            raise broken(e) from e
        except (RuntimeError, ValueError):          # no usable reference, or masters that do not fit:
            raise                                   # the CPU would say the same
        except Exception as e:
            raise broken(e) from e


def _await(session, token, kind, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            m = session.events.get(timeout=0.25)
        except queue.Empty:
            continue
        if m[0] == "dead":
            raise GpuUnavailable("the GPU process stopped unexpectedly")
        if m[0] == "fut" or m[1] != token:
            continue
        if m[0] == "fatal":
            raise GpuUnavailable(m[2])
        if m[0] == kind:
            return m
    raise GpuUnavailable("the GPU did not answer")


STALL_SECONDS = 120.0


def _loop(paths, as_u16, ref_i, ref_only, hw, nch, ds, cfg, cap, n_slots, mdesc, session, token,
          slots, out, futures, step, stopped, say):
    from setiastro.saspro import star_alignment as sa
    S = sa._S(ds)                                   # the GPU bins exact ds x ds blocks
    desc = slots.descriptor()
    H, W = hw
    grid = (H // ds, W // ds)
    total = len(paths) - int(ref_only)
    order = [ref_i] + [i for i in range(len(paths)) if i != ref_i]
    todo = deque(order)
    free = deque(range(n_slots))
    waiting: deque = deque()                        # loaded, waiting for the card
    slot_of: dict = {}
    held: set = set()                               # on the card, waiting for a match
    levels = None                                   # reference star lists, one per sigma
    oks, skipped = [], []
    max_held, done, finishing = 0, 0, False
    info_of: dict = {}

    pool = session.pool()
    ev = session.events

    def submit(tag, fn, *a):
        try:
            f = pool.submit(fn, *a)
        except Exception as e:
            session.drop_pool()
            raise GpuUnavailable(f"the worker processes stopped ({type(e).__name__})") from e
        futures.add(f)

        def cb(f=f, tag=tag):
            ev.put(("fut", token, tag, f))
        f.add_done_callback(cb)

    def finish_frame():
        nonlocal done
        done += 1
        step(done, total, f"Stacking frame {done} of {total}…")

    def skip(idx, why):
        skipped.append((paths[idx], why))
        say(f"Skipped {os.path.basename(paths[idx])}: {why}")
        finish_frame()

    def match(idx, level, pts):
        info_of[idx] = (level, pts)
        submit(("match", idx, level), engine.match_job, pts, levels[level], grid, cfg)

    def pump():
        # The reference is first in line; the rest may load meanwhile but are
        # not sent to the card until its star lists are known.
        while todo and free:
            idx = todo.popleft()
            s = free.popleft()
            slot_of[idx] = s
            submit(("load", idx), engine.load_slot_job, idx, paths[idx], as_u16[idx], desc, s,
                   cap, hw, nch, mdesc)
        while waiting and levels is not None and len(held) < max_held:
            idx, meta = waiting.popleft()
            held.add(idx)
            session.send("frame", token, idx, slot_of[idx], meta, False)

    m = _await(session, token, "begun", 60.0) if not stopped() else None
    if m is None:
        raise InterruptedError("Quick Stack was cancelled.")
    max_held = m[2]
    pump()
    last = time.monotonic()
    step(0, total, "Stacking…")
    while True:
        if stopped():
            raise InterruptedError("Quick Stack was cancelled.")
        if done >= total and levels is not None and not finishing:
            finishing = True
            session.send("finish", token)
        try:
            m = ev.get(timeout=0.25)
        except queue.Empty:
            if time.monotonic() - last > STALL_SECONDS:
                raise GpuUnavailable("the GPU stopped responding")
            continue
        last = time.monotonic()
        kind = m[0]
        if kind == "dead":
            raise GpuUnavailable(session.reason or "the GPU process stopped unexpectedly")
        if m[1] != token:
            continue
        if kind == "fatal":
            raise GpuUnavailable(m[2])
        if kind == "fut":
            tag, f = m[2], m[3]
            futures.discard(f)
            try:
                r = f.result()
            except Exception as e:
                session.drop_pool()
                raise GpuUnavailable(f"the worker processes stopped ({type(e).__name__})") from e
            idx = tag[1]
            if tag[0] == "load":
                if r[0] == "skip":
                    free.append(slot_of.pop(idx))
                    if idx == ref_i:
                        raise GpuUnavailable(f"the reference could not be read ({r[2]})")
                    skip(idx, r[2])
                elif idx == ref_i:
                    session.send("frame", token, idx, slot_of[idx], r[2], True)
                else:
                    waiting.append((idx, r[2]))
            else:
                level = tag[2]
                T, info = r
                if T is not None:
                    T_full, bad = engine.lift_transform(T, ds, S)
                    if T_full is None:
                        T, info = None, bad
                if T is None and level + 1 < len(levels):
                    session.send("detect", token, idx, level + 1)
                elif T is None:
                    session.send("drop", token, idx)
                    held.discard(idx)
                    skip(idx, info)
                else:
                    info_of[idx] = {"angle": engine.rotation_deg(T_full), "inliers": info["inliers"],
                                    "rms": info["rms"] * ds}
                    session.send("acc", token, idx, T_full)
        elif kind == "free":
            s = m[2]
            for i, v in list(slot_of.items()):
                if v == s:
                    del slot_of[i]
            free.append(s)
        elif kind == "ref":
            levels = m[2]
        elif kind == "pts":
            match(m[2], m[3], m[4])
        elif kind == "acc_done":
            idx = m[2]
            held.discard(idx)
            if ref_only and idx == ref_i:
                continue
            d = info_of.pop(idx, None)
            if not isinstance(d, dict):              # the reference itself
                d = {"angle": 0.0, "inliers": 0, "rms": 0.0}
            oks.append((paths[idx], {**d, "offset": m[3]}))
            finish_frame()
        elif kind == "result":
            return np.array(out.view((H, W, nch), np.float32))[..., 0] if nch == 1 else \
                np.array(out.view((H, W, nch), np.float32)), m[2], oks, skipped
        pump()
