# src/setiastro/saspro/quick_stack_gpu.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Quick Stack on a GPU.

    CPU workers   read a frame into a shared slot
    GPU process   upload -> dark / flat -> debayer -> bin -> star detection
    CPU workers   match the star list to the reference's
    GPU process   warp + accumulate (weighted by the star count) -> trimmed mean

Any GPU SASpro's hardware acceleration installs, in SASpro's order: CUDA
(NVIDIA) or ROCm (AMD), Intel XPU, DirectML (any DirectX 12 card on Windows),
Apple's Metal (MPS).

The GPU process owns the card for the whole run: VRAM is released when it
exits, and the app's own process never loads torch for this. It and the
workers are kept warm in a GpuSession (starting torch takes several seconds),
so the window can start them before the first frame is stacked.

On CUDA and ROCm, warping and accumulating is one compiled kernel (NVRTC or
hipRTC, _Fused). Ordinary torch ops moved six times the memory and were ten
times slower; the kernels do the same arithmetic as
quick_stack_worker._accumulate_band and give bit-identical sums (checked once at
start-up, on this machine's card, before any frame is trusted to it). The other
backends have no kernel compiler: they run the same steps as ordinary torch
operations in row bands (_Portable), checked at start-up against the CPU code to
float32's reach; so does a CUDA or ROCm card whose compile fails. The compilers
are private torch API, so everything here fails soft: no GPU, an older torch,
too little VRAM or a crash raises GpuUnavailable and the caller stacks on the CPU.
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

/* bilinear inverse warp straight into the running sum / count / two highest / two lowest.
   D[5] (weighted): the frame's samples are scaled by its weight wtab[Q[0]], the weights are summed,
   and each extreme keeps the frame's code Q[0]; quick_stack_worker._accumulate_band. Without it the
   weight is 1 and the weight arrays (one-element stand-ins then) are never touched. */
extern "C" __global__ void qs_acc(const float* __restrict__ src, const double* __restrict__ P, const float* __restrict__ off, const int* __restrict__ D,
   const float* __restrict__ wtab, const int* __restrict__ Q,
   float* __restrict__ sumv, unsigned short* __restrict__ cnt, float* __restrict__ wsum,
   float* __restrict__ hi0, float* __restrict__ hi1, float* __restrict__ lo0, float* __restrict__ lo1,
   unsigned char* __restrict__ qh0, unsigned char* __restrict__ qh1, unsigned char* __restrict__ ql0, unsigned char* __restrict__ ql1) {
  const int H = D[0], W = D[1], C = D[2], sH = D[3], sW = D[4], wt = D[5];
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (long)H * W) return;
  const int y = (int)(i / W), x = (int)(i - (long)y * W);
  const double sx = P[1] * y + P[2], sy = P[4] * y + P[5];
  const double xs = sx + P[0] * x, ys = sy + P[3] * x;
  if (xs < 0.0 || ys < 0.0) return;
  const int ix = (int)xs, iy = (int)ys; if (ix >= sW - 1 || iy >= sH - 1) return;
  const float fx = (float)(xs - ix), fy = (float)(ys - iy);
  const long b0 = ((long)iy * sW + ix) * C, b1 = b0 + (long)sW * C;
  const unsigned char q = (unsigned char)Q[0]; const float w = wt ? wtab[q] : 1.0f;
  for (int c = 0; c < C; ++c) {
    const float a = src[b0 + c], b = src[b0 + C + c], d = src[b1 + c], e = src[b1 + C + c];
    if (a == 0.0f || b == 0.0f || d == 0.0f || e == 0.0f) continue;
    const float t = a + (b - a) * fx, u = d + (e - d) * fx, v = t + (u - t) * fy;
    if (!(fabsf(v) < 1e30f)) continue;
    const float r = v - off[c]; const long k = i * C + c;
    sumv[k] += w * r; cnt[k] += 1;
    if (wt) {
      wsum[k] += w;
      if (r > hi0[k]) { hi1[k] = hi0[k]; qh1[k] = qh0[k]; hi0[k] = r; qh0[k] = q; } else if (r > hi1[k]) { hi1[k] = r; qh1[k] = q; }
      if (r < lo0[k]) { lo1[k] = lo0[k]; ql1[k] = ql0[k]; lo0[k] = r; ql0[k] = q; } else if (r < lo1[k]) { lo1[k] = r; ql1[k] = q; }
    } else {
      if (r > hi0[k]) { hi1[k] = hi0[k]; hi0[k] = r; } else if (r > hi1[k]) hi1[k] = r;
      if (r < lo0[k]) { lo1[k] = lo0[k]; lo0[k] = r; } else if (r < lo1[k]) lo1[k] = r;
    }
  }
}

/* trimmed mean, the rule of quick_stack_worker._finish: with D[5] each dropped extreme takes its own
   frame's weight out of the sum and the total weight; without, every frame weighs 1 */
extern "C" __global__ void qs_fin(const float* __restrict__ sumv, const unsigned short* __restrict__ cnt, const float* __restrict__ wsum,
   const float* __restrict__ hi0, const float* __restrict__ hi1, const float* __restrict__ lo0, const float* __restrict__ lo1,
   const unsigned char* __restrict__ qh0, const unsigned char* __restrict__ qh1, const unsigned char* __restrict__ ql0, const unsigned char* __restrict__ ql1,
   const float* __restrict__ wtab, float* __restrict__ out, const int* __restrict__ D) {
  const long N = (long)D[0] * D[1] * D[2], k = (long)blockIdx.x * blockDim.x + threadIdx.x; if (k >= N) return;
  const int n = cnt[k]; if (!n) { out[k] = 0.0f; return; }
  double s = sumv[k];
  if (D[5]) {
    double ws = wsum[k], w;
    if (n >= D[3]) {
      w = wtab[qh0[k]]; s -= w * (double)hi0[k]; ws -= w;
      w = wtab[ql0[k]]; s -= w * (double)lo0[k]; ws -= w;
      if (n >= D[4]) {
        w = wtab[qh1[k]]; s -= w * (double)hi1[k]; ws -= w;
        w = wtab[ql1[k]]; s -= w * (double)lo1[k]; ws -= w;
      }
    }
    out[k] = ws > 0.0 ? (float)(s / ws) : 0.0f;
    return;
  }
  int dr = 0;
  if (n >= D[3]) { s -= hi0[k]; s -= lo0[k]; dr = 2; if (n >= D[4]) { s -= hi1[k]; s -= lo1[k]; dr = 4; } }
  out[k] = (float)(s / (n - dr));
}
'''
KERNEL_NAMES = ("qs_cal", "qs_deb", "qs_bin", "qs_acc", "qs_fin")
_CFA = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}
VRAM_RESERVE = 0.8e9          # star detection work space and whatever else is on the card
MAX_HELD = 12                 # frames on the card waiting for their match


@contextlib.contextmanager
def _no_toolkit_headers():
    """Compile without the CUDA toolkit installed.

    NVRTC comes with PyTorch and these kernels include no headers, but torch
    adds the toolkit's include folder to every compile and raises OSError
    ("CUDA_HOME environment variable is not set") on a PC without the toolkit,
    which is most users' PCs. That folder is left out there."""
    try:
        import torch.utils.cpp_extension as ce
        real = ce.include_paths
    except Exception:
        ce = None
    if ce is not None:
        def include_paths(device_type="cpu", *args, **kwargs):
            try:
                return real(device_type, *args, **kwargs)
            except OSError:
                return real("cpu", *args, **kwargs)
        ce.include_paths = include_paths
    try:
        yield
    finally:
        if ce is not None:
            ce.include_paths = real


def _compile(torch) -> dict:
    """{name: kernel}. --fmad=false keeps the arithmetic the CPU kernel's."""
    if not hasattr(torch.cuda, "_compile_kernel"):
        raise GpuUnavailable(f"this PyTorch ({torch.__version__}) cannot compile CUDA kernels")
    opts = ["-ffp-contract=off"] if getattr(torch.version, "hip", None) else ["--fmad=false"]
    with _no_toolkit_headers():
        try:                            # all four in one compile
            from torch.cuda import _utils as cu
            ptx, _ = cu._nvrtc_compile(KERNELS, KERNEL_NAMES[0], None, None, opts)
            mod = cu._cuda_load_module(ptx, list(KERNEL_NAMES))
            return {n: mod[n] for n in KERNEL_NAMES}
        except Exception:
            return {n: torch.cuda._compile_kernel(KERNELS, n, nvcc_options=opts)
                    for n in KERNEL_NAMES}


def _grid(n: int):
    return ((n + 255) // 256, 1, 1)


ACC_BYTES = 4 + 2 + 16       # per sample: sum, count, two highest, two lowest
WEIGHT_BYTES = 4 + 4         # and for the weights: their sum, a code beside each extreme


def plan_memory(free: int, n_samples: int, other: int, held: int):
    """(frames the card can hold waiting for their match, weighted).

    The weights' accumulators are what is left out when the card cannot also
    take them: the frames then weigh the same, and the stack stays on the
    GPU. other: the scratch frame, masters and mosaic; held: one waiting frame."""
    fixed = n_samples * ACC_BYTES + other
    for weighted in (True, False):
        need = fixed + (n_samples * WEIGHT_BYTES if weighted else 0)
        n = int(min(MAX_HELD, (free - need - VRAM_RESERVE) // held))
        if n >= 2:
            return n, weighted
    raise GpuUnavailable(f"needs about {(fixed + 2 * held + VRAM_RESERVE) / 1e9:.1f} GB of "
                         f"video memory, {free / 1e9:.1f} GB is free")


# ─────────────────────────────────────────────────────────────────────────────
# The card, whichever it is
# ─────────────────────────────────────────────────────────────────────────────
def _kind(torch, dev) -> str:
    """"cuda", "rocm", "xpu", "directml" or "mps"."""
    if dev.type == "cuda":
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if dev.type == "privateuseone":
        return "directml"
    return str(dev.type)


_BACKEND_LABEL = {"cuda": "", "rocm": "ROCm", "xpu": "Intel XPU", "directml": "DirectML", "mps": "Metal"}


def _empty_cache(torch, dev) -> None:
    mod = {"cuda": "cuda", "xpu": "xpu", "mps": "mps"}.get(dev.type)
    if mod:
        with contextlib.suppress(Exception):
            getattr(torch, mod).empty_cache()


def _device_name(torch, dev) -> str:
    kind = _kind(torch, dev)
    name = ""
    with contextlib.suppress(Exception):
        if kind in ("cuda", "rocm"):
            name = torch.cuda.get_device_name(dev.index or 0)
        elif kind == "xpu":
            name = torch.xpu.get_device_name(dev.index or 0)
        elif kind == "directml":
            import torch_directml
            name = torch_directml.device_name(dev.index or 0)
    name = str(name or "").split(chr(0))[0].strip()          # DirectML's name ends in a NUL
    name = name or ("Apple GPU" if kind == "mps" else "GPU")
    label = _BACKEND_LABEL.get(kind, kind)
    return f"{name} ({label})" if label else name


def _dxgi_memory(name: str):
    """(free, total) for a DirectML card: the video memory budget Windows gives
    this process on the adapter called `name`, less what the process already
    uses. None when DXGI cannot say."""
    import ctypes
    from ctypes import POINTER, byref, c_long, c_uint, c_void_p

    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16),
                    ("d", ctypes.c_ubyte * 8)]

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", c_uint), ("DeviceId", c_uint),
                    ("SubSysId", c_uint), ("Revision", c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("LuidLow", ctypes.c_uint32), ("LuidHigh", ctypes.c_int32), ("Flags", c_uint)]

    class MEMINFO(ctypes.Structure):
        _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                    ("AvailableForReservation", ctypes.c_uint64), ("CurrentReservation", ctypes.c_uint64)]

    def guid(a, b, c, *d):
        return GUID(a, b, c, (ctypes.c_ubyte * 8)(*d))

    def method(obj, index, *argtypes):
        vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        return ctypes.WINFUNCTYPE(c_long, c_void_p, *argtypes)(vtbl[index])

    iid_factory1 = guid(0x770AAE78, 0xF26F, 0x4DBA, 0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
    iid_adapter3 = guid(0x645967A4, 0x1392, 0x4310, 0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD)
    try:
        create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create.argtypes, create.restype = [POINTER(GUID), POINTER(c_void_p)], c_long
        factory = c_void_p()
        if create(byref(iid_factory1), byref(factory)) != 0 or not factory.value:
            return None
        found = []
        try:
            for i in range(16):
                adapter = c_void_p()
                if method(factory, 12, c_uint, POINTER(c_void_p))(factory, i, byref(adapter)) != 0:
                    break                                         # DXGI_ERROR_NOT_FOUND: no more
                try:
                    desc = DESC1()
                    if method(adapter, 10, POINTER(DESC1))(adapter, byref(desc)) != 0 or desc.Flags & 2:
                        continue                                  # 2: the software renderer
                    a3 = c_void_p()
                    if method(adapter, 0, POINTER(GUID), POINTER(c_void_p))(
                            adapter, byref(iid_adapter3), byref(a3)) != 0:
                        continue
                    try:
                        mi = MEMINFO()                            # node 0, the local segment group
                        if method(a3, 14, c_uint, c_uint, POINTER(MEMINFO))(a3, 0, 0, byref(mi)) == 0:
                            found.append((desc.Description.strip(), int(mi.Budget), int(mi.CurrentUsage)))
                    finally:
                        method(a3, 2)(a3)
                finally:
                    method(adapter, 2)(adapter)
        finally:
            method(factory, 2)(factory)
    except Exception:
        return None
    if not found:
        return None
    pick = next((f for f in found if f[0] == (name or "").strip()), None) or max(found, key=lambda f: f[1])
    return max(0, pick[1] - pick[2]), pick[1]


def _memory(torch, dev) -> tuple:
    """(free, total) bytes of video memory this process can use."""
    kind = _kind(torch, dev)
    if kind in ("cuda", "rocm"):
        return torch.cuda.mem_get_info()
    if kind == "xpu":
        with contextlib.suppress(Exception):
            free, total = torch.xpu.mem_get_info(dev.index or 0)
            return int(free), int(total)
        total = int(torch.xpu.get_device_properties(dev.index or 0).total_memory)
        return total - int(torch.xpu.memory_reserved()), total
    if kind == "mps":                                    # unified memory: what Metal recommends
        with contextlib.suppress(Exception):
            total = int(torch.mps.recommended_max_memory())
            return max(0, total - int(torch.mps.driver_allocated_memory())), total
        import psutil
        total = int(psutil.virtual_memory().available * 0.6)
        return total, total
    if kind == "directml":
        import torch_directml
        got = _dxgi_memory(torch_directml.device_name(dev.index or 0))
        if got:
            return got
    raise GpuUnavailable("the free video memory on this card is not known")


def _f64_ok(torch, dev) -> bool:
    """Can the card do float64? DirectML and Metal are never asked."""
    if _kind(torch, dev) in ("directml", "mps"):
        return False
    try:
        t = torch.tensor([1.25, 3.0], dtype=torch.float64, device=dev)
        return float((t * t).sum().item()) == 10.5625
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# The pixel steps, two ways
# ─────────────────────────────────────────────────────────────────────────────
ACC_NAMES = ("sumv", "cnt", "wsum", "hi0", "hi1", "lo0", "lo1", "qh0", "qh1", "ql0", "ql1")


class _Fused:
    """The compiled kernels: CUDA through NVRTC, ROCm through hipRTC. They do
    the CPU code's arithmetic, so the self-test asks for exact sums."""

    exact = True
    pinned_uploads = True          # the slots are page-locked: uploads are direct DMA
    scratch_bytes = 0
    label = "compiled kernels"

    def __init__(self, torch, dev, K):
        self.torch, self.dev, self.K = torch, dev, K
        self._D: dict = {}

    def _ints(self, *v):
        t = self._D.get(v)
        if t is None:
            t = self._D[v] = self.torch.tensor(v, dtype=self.torch.int32, device=self.dev)
        return t

    def upload(self, view):
        t = self.torch.from_numpy(view).to(self.dev, non_blocking=True)
        self.torch.cuda.synchronize()                             # the slot is free once copied
        return t

    def download(self, src, dst):
        dst.copy_(src)

    def cal(self, raw, dark, flat, out, use_dark, use_flat):
        H, W = out.shape
        self.K["qs_cal"](grid=_grid(H * W), block=(256, 1, 1),
                         args=[raw, dark, flat, out, self._ints(H, W, int(use_dark), int(use_flat))])

    def deb(self, m, out, bayer):
        H, W = m.shape
        self.K["qs_deb"](grid=_grid(H * W), block=(256, 1, 1),
                         args=[m, out, self._ints(H, W, 3, *_CFA[bayer])])

    def bin(self, src, out, ds):
        H, W, C = src.shape
        h, w = out.shape
        self.K["qs_bin"](grid=_grid(h * w), block=(256, 1, 1), args=[src, out, self._ints(H, W, C, ds, h, w)])

    def sky(self, x):
        torch = self.torch
        s = x[::8, ::8]
        out = []
        for c in range(x.shape[2]):
            v = s[..., c]
            v = v[(v != 0) & torch.isfinite(v)]
            out.append(float(v.median()) if v.numel() else 0.0)
        return np.asarray(out, np.float32)

    def acc(self, src, inv, off, q, weighted, A, wtab):
        torch, dev = self.torch, self.dev
        H, W, C = A["sumv"].shape
        P = torch.tensor(np.asarray(inv, np.float64).ravel(), device=dev)
        o = torch.tensor(np.asarray(off, np.float32), device=dev)
        Q = torch.tensor([int(q)], dtype=torch.int32, device=dev)
        self.K["qs_acc"](grid=_grid(H * W), block=(256, 1, 1),
                         args=[src, P, o, self._ints(H, W, C, int(src.shape[0]), int(src.shape[1]),
                                                     int(weighted)),
                               wtab, Q, *(A[n] for n in ACC_NAMES)])

    def fin(self, A, weighted, out, wtab):
        H, W, C = out.shape
        self.K["qs_fin"](grid=_grid(H * W * C), block=(256, 1, 1),
                         args=[*(A[n] for n in ACC_NAMES), wtab, out,
                               self._ints(H, W, C, engine.MIN_TRIM_FRAMES, engine.FULL_TRIM_FRAMES,
                                          int(weighted))])


def _split(v: np.ndarray):
    """float64 coordinates as whole pixels (int32) and the fraction (float32, [0, 1))."""
    i = np.floor(v)
    f = (v - i).astype(np.float32)
    up = f >= 1.0                       # a fraction just under 1 can round up to 1.0
    f[up] = 0.0
    return (i + up).astype(np.int32), f


class _Portable:
    """The same steps as ordinary torch operations, for every card without a
    kernel compiler (DirectML, Intel XPU, Apple Metal) and for a CUDA or ROCm
    card whose compile failed. Run in row bands, so the work space stays small.

    float32 throughout (DirectML and Metal have no float64; the trimmed mean
    uses it where the card has it). A coordinate is split on the CPU into whole
    pixels and a fraction, so the fraction is within about 2e-7 px of the CPU
    engine's. Some cards round a division differently in the last bit. The
    self-test allows for both and for nothing more.

    Lookups are index_select, never t[index]: on DirectML indexing with a
    tensor took 26 ms where index_select took 0.6 ms for the same 2 MP gather.
    A mask becomes a number through torch.where, never .to(): on DirectML the
    conversions were most of the accumulate (355 -> 161 ms per 46.8 MP frame).
    DirectML also does integer arithmetic in float32 (16777217 + 1 gave
    16777216), so there an index is formed per block of rows and no number
    on the card passes 2^24 (_gather); the self-test has a frame past it."""

    exact = False
    pinned_uploads = False
    SCRATCH = 256 << 20                 # bytes of temporaries per band
    scratch_bytes = 2 * SCRATCH
    label = "torch operations"

    def __init__(self, torch, dev):
        import torch.nn.functional as F
        self.torch, self.dev, self.F = torch, dev, F
        self.f64 = _f64_ok(torch, dev)
        self.max_index = (1 << 24) if _kind(torch, dev) == "directml" else None
        self._k: dict = {}

    def _c(self, value, dtype):
        """A one-element constant on the card, kept."""
        t = self._k.get((value, dtype))
        if t is None:
            t = self._k[(value, dtype)] = self.torch.tensor(value, dtype=dtype, device=self.dev)
        return t

    def _num(self, mask, dtype, value=1):
        """mask as value / 0 in dtype."""
        return self.torch.where(mask, self._c(value, dtype), self._c(0, dtype))

    def _bands(self, rows, width, bytes_per_px):
        n = max(1, int(self.SCRATCH // max(1, width * bytes_per_px)))
        return [(y, min(rows, y + n)) for y in range(0, rows, n)]

    def upload(self, view):
        # A private copy first: when this returns the slot is reused, whatever
        # the backend does with a copy from host memory.
        return self.torch.from_numpy(np.array(view)).to(self.dev)

    def download(self, src, dst):
        """Card -> a CPU tensor. Through .cpu(): DirectML's copy_ into a CPU
        tensor writes nothing and raises nothing."""
        H = int(src.shape[0])
        for y0, y1 in self._bands(H, int(np.prod(src.shape[1:])), 8):
            dst[y0:y1].copy_(src[y0:y1].cpu())

    def cal(self, raw, dark, flat, out, use_dark, use_flat):
        torch = self.torch
        H, W = out.shape
        for y0, y1 in self._bands(H, W, 24):
            u = raw[y0:y1].to(torch.int32) + 32768
            v = u.to(torch.float32) / 65535.0
            if use_dark:
                v = v - dark[y0:y1]
            if use_flat:
                v = v / flat[y0:y1]
            v = v.masked_fill(v == 0, 1e-9).masked_fill(u == 0, 0.0)
            out[y0:y1].copy_(v)

    def deb(self, m, out, bayer):
        torch, dev = self.torch, self.dev
        H, W = m.shape
        codes = _CFA[bayer]
        luts = [torch.tensor([int(codes[k] == c) for k in range(4)], dtype=torch.uint8, device=dev)
                for c in range(3)]
        rx = torch.arange(-1, W + 1, device=dev) % 2
        for y0, y1 in self._bands(H, W, 96):
            hb = y1 - y0
            a, b = max(0, y0 - 1), min(H, y1 + 1)
            mp = self.F.pad(m[a:b], (1, 1, 1 if y0 == 0 else 0, 1 if y1 == H else 0))    # rows y0-1 .. y1
            ry = torch.arange(y0 - 1, y1 + 1, device=dev) % 2
            site = (ry[:, None] * 2 + rx[None, :]).long().reshape(-1)
            nz = mp != 0
            v = mp[1:-1, 1:-1]
            chans = []
            for c in range(3):
                colour = torch.index_select(luts[c], 0, site).view(mp.shape) != 0
                mc = colour & nz                                  # same colour, with data
                vals = mp.masked_fill(~mc, 0.0)
                cnt = self._num(mc, torch.float32, 1.0)
                s = torch.zeros((hb, W), dtype=torch.float32, device=dev)
                n = torch.zeros_like(s)
                for dy in range(3):
                    for dx in range(3):
                        s = s + vals[dy:dy + hb, dx:dx + W]
                        n = n + cnt[dy:dy + hb, dx:dx + W]
                est = (s / n.clamp_min(1.0)).masked_fill(n == 0, 0.0)
                chans.append(torch.where(colour[1:-1, 1:-1], v, est))
            out[y0:y1].copy_(torch.stack(chans, -1).masked_fill((v == 0).unsqueeze(-1), 0.0))

    def bin(self, src, out, ds):
        H, W, C = src.shape
        h, w = out.shape
        for b0, b1 in self._bands(h, w * ds * ds, 8 * C + 8):
            x = src[b0 * ds:b1 * ds, :w * ds]
            g = x.sum(-1) / C if C > 1 else x[..., 0]
            s = g.reshape(b1 - b0, ds, w, ds).sum(3).sum(1)
            out[b0:b1].copy_(s / float(ds * ds))

    def sky(self, x):
        """Per channel, the lower median of the data samples on an 8 px grid
        (what torch.median gives on the card)."""
        s = x[::8, ::8].contiguous().cpu().numpy()
        out = []
        for c in range(s.shape[2]):
            v = s[..., c].ravel()
            v = v[(v != 0) & np.isfinite(v)]
            out.append(float(np.partition(v, (v.size - 1) // 2)[(v.size - 1) // 2]) if v.size else 0.0)
        return np.asarray(out, np.float32)

    def _coord(self, ri, rf, ci, cf):
        f = rf[:, None] + cf[None, :]                             # [0, 2)
        k = f >= 1
        i = ri[:, None] + ci[None, :]
        return self.torch.where(k, i + 1, i), self.torch.where(k, f - 1.0, f)   # exact: f - 1 in [0, 1)

    def _gather(self, flat, iy, ix, sW, shape, lo, hi):
        """flat[iy * sW + ix] for every pixel (flat: the source, row-major, C
        values per pixel); every iy is in [lo, hi]. Per block of rows where
        integers stop at 2^24, skipping the blocks no pixel reads."""
        torch = self.torch
        rows = flat.shape[0] // sW
        R = rows if self.max_index is None else max(1, self.max_index // sW - 1)
        out = None
        for r0 in range(0, rows, R):
            r1 = min(rows, r0 + R)
            if r1 <= lo or r0 > hi:
                continue
            k = ((iy - r0).clamp(0, r1 - r0 - 1) * sW + ix).long().reshape(-1)
            g = torch.index_select(flat[r0 * sW:r1 * sW], 0, k).view(shape)
            out = g if out is None else torch.where(((iy >= r0) & (iy < r1)).unsqueeze(-1), g, out)
        return out

    def _extremes(self, r, good, x0, x1, q0, q1, qq, high):
        torch = self.torch
        g0 = good & ((r > x0) if high else (r < x0))
        g1 = good & ~g0 & ((r > x1) if high else (r < x1))
        if q0 is not None:
            q1.copy_(torch.where(g0, q0, torch.where(g1, qq, q1)))
            q0.copy_(torch.where(g0, qq, q0))
        x1.copy_(torch.where(g0, x0, torch.where(g1, r, x1)))
        x0.copy_(torch.where(g0, r, x0))

    def acc(self, src, inv, off, q, weighted, A, wtab):
        torch, dev = self.torch, self.dev
        H, W, C = A["sumv"].shape
        sH, sW = int(src.shape[0]), int(src.shape[1])
        inv = np.asarray(inv, np.float64)
        cols = np.arange(W, dtype=np.float64)
        up = lambda a: torch.from_numpy(a).to(dev)                # noqa: E731
        cxi, cxf = (up(a) for a in _split(inv[0, 0] * cols))
        cyi_h, cyf_h = _split(inv[1, 0] * cols)
        cyi, cyf = up(cyi_h), up(cyf_h)
        flat = src.reshape(-1, C)
        o = up(np.asarray(off, np.float32).reshape(C))
        w = float(engine.WEIGHT_TABLE[q]) if weighted else 1.0
        qq = torch.tensor(int(q), dtype=torch.uint8, device=dev)
        for y0, y1 in self._bands(H, W, 56 * C + 64):
            rows = np.arange(y0, y1, dtype=np.float64)
            rxi, rxf = (up(a) for a in _split(inv[0, 1] * rows + inv[0, 2]))
            ryi_h, ryf_h = _split(inv[1, 1] * rows + inv[1, 2])
            ryi, ryf = up(ryi_h), up(ryf_h)
            # the source rows this band can read (whole parts, one carry, clamped as below)
            lo = min(max(int(ryi_h.min()) + int(cyi_h.min()), 0), sH - 2)
            hi = min(max(int(ryi_h.max()) + int(cyi_h.max()) + 1, 0), sH - 2)
            ix, fx = self._coord(rxi, rxf, cxi, cxf)
            iy, fy = self._coord(ryi, ryf, cyi, cyf)
            ok = (ix >= 0) & (iy >= 0) & (ix < sW - 1) & (iy < sH - 1)
            iy, ix = iy.clamp(0, sH - 2), ix.clamp(0, sW - 2)
            shape = (y1 - y0, W, C)
            a, b = (self._gather(flat, iy, x, sW, shape, lo, hi) for x in (ix, ix + 1))
            iy = iy + 1
            d, e = (self._gather(flat, iy, x, sW, shape, lo + 1, hi + 1) for x in (ix, ix + 1))
            del ix, iy
            # Spelled out per channel: DirectML broadcasts a chained expression 7x slower.
            fx, fy = (f.unsqueeze(-1).expand(-1, -1, C).contiguous() for f in (fx, fy))
            t = a + (b - a) * fx
            u = d + (e - d) * fx
            v = t + (u - t) * fy
            good = ok.unsqueeze(-1) & (a != 0) & (b != 0) & (d != 0) & (e != 0) & (v.abs() < 1e30)
            del a, b, d, e, t, u, ok
            r = v - o
            del v
            sl = slice(y0, y1)
            A["sumv"][sl].add_((r if w == 1.0 else r * w).masked_fill(~good, 0.0))
            A["cnt"][sl].add_(self._num(good, A["cnt"].dtype))
            if weighted:
                A["wsum"][sl].add_(self._num(good, torch.float32, w))
            qa = (lambda n: A[n][sl]) if weighted else (lambda n: None)     # noqa: E731
            self._extremes(r, good, A["hi0"][sl], A["hi1"][sl], qa("qh0"), qa("qh1"), qq, True)
            self._extremes(r, good, A["lo0"][sl], A["lo1"][sl], qa("ql0"), qa("ql1"), qq, False)

    def fin(self, A, weighted, out, wtab):
        """quick_stack_worker._finish: each dropped extreme takes its own frame's
        weight out of the sum and the total weight; without weights every frame
        weighs 1."""
        torch = self.torch
        H, W, C = out.shape
        ft = torch.float64 if self.f64 else torch.float32
        wt = wtab.to(ft)
        for y0, y1 in self._bands(H, W * C, 96):
            sl = slice(y0, y1)
            n = A["cnt"][sl].to(torch.int32)
            n = torch.where(n < 0, n + 65536, n)                  # the count is unsigned 16-bit
            m1, m2 = n >= engine.MIN_TRIM_FRAMES, n >= engine.FULL_TRIM_FRAMES
            s = A["sumv"][sl].to(ft)
            ws = A["wsum"][sl].to(ft) if weighted else n.to(ft)
            for x, qn, m in (("hi0", "qh0", m1), ("lo0", "ql0", m1), ("hi1", "qh1", m2), ("lo1", "ql1", m2)):
                val = A[x][sl].to(ft)
                if weighted:
                    qv = A[qn][sl]
                    wq = torch.index_select(wt, 0, qv.reshape(-1).long()).view(qv.shape)
                    s = s - (wq * val).masked_fill(~m, 0.0)
                    ws = ws - wq.masked_fill(~m, 0.0)
                else:
                    s = s - val.masked_fill(~m, 0.0)
                    ws = ws - self._num(m, ft, 1.0)
            ok = (n > 0) & (ws > 0)
            res = s / torch.where(ok, ws, torch.ones_like(ws))
            out[sl].copy_(res.masked_fill(~ok, 0.0).to(torch.float32))


class _Run:
    """One stack on the card."""

    def __init__(self, torch, dev, ops, gd, sa, token, p):
        self.torch, self.dev, self.ops, self.gd, self.sa = torch, dev, ops, gd, sa
        self.token, self.p = token, p
        H, W, C = self.H, self.W, self.C = p["H"], p["W"], p["nch"]
        ds = self.ds = p["ds"]
        self.h, self.w = H // ds, W // ds
        N = H * W * C
        free, _total = _memory(torch, dev)
        other = N * 4 + ops.scratch_bytes                         # scratch frame, the steps' work space
        other += H * W * 4 * (len(p["masters"] or {}) + (1 if C == 3 else 0))     # masters, mosaic
        held = max(H * W * 2, N * 4 if p["any_f32"] else 0) + self.h * self.w * 4
        self.max_held, self.weighted = plan_memory(free, N, other, held)
        shape = (H, W, C)
        inf = float("inf")
        full = lambda v: torch.full(shape, v, dtype=torch.float32, device=dev)   # noqa: E731
        wshape = shape if self.weighted else (1,)                 # one-element stand-ins, never touched
        self.A = {"sumv": full(0.0), "cnt": torch.zeros(shape, dtype=torch.int16, device=dev),
                  "wsum": torch.zeros(wshape, dtype=torch.float32, device=dev),
                  "hi0": full(-inf), "hi1": full(-inf), "lo0": full(inf), "lo1": full(inf),
                  **{n: torch.zeros(wshape, dtype=torch.uint8, device=dev) for n in ACC_NAMES[7:]}}
        self.wtab = torch.from_numpy(engine.WEIGHT_TABLE).to(dev)
        self.ref_stars = 0
        self.weigh = False                   # by star count: decided by the reference's count
        self.scratch = torch.empty(shape, dtype=torch.float32, device=dev)
        # The card calibrates raw 16-bit mosaics and mono planes only. Masters
        # fitted to frames that are colour already go on in the loaders.
        masters = {w: d for w, d in (p["masters"] or {}).items()
                   if len(d["shape"]) == 2 and not d.get("colour")}
        self.cal_flags = (int("dark" in masters), int("flat" in masters))
        self.dark, self.flat = (self._master(masters.get(w), torch) for w in ("dark", "flat"))
        self.mosaic = torch.empty((H, W), dtype=torch.float32, device=dev) if C == 3 else None
        self.held: dict = {}                 # idx -> (meta, tensor, sky, gray, stars)
        self.ref_sky = None
        self.ladder = p["ladder"]
        self.slots = engine.SharedBlock.attach(p["slots"])
        self.out = engine.SharedBlock.attach(p["out"])
        self.cap = p["cap"]
        # Page-lock the slots: uploads become direct DMA (about twice as fast).
        self.registered = None
        if ops.pinned_uploads:
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
        use_dark, use_flat = self.cal_flags
        if self.C == 1:
            self.ops.cal(t, self.dark, self.flat, self.scratch.view(self.H, self.W), use_dark, use_flat)
        else:
            self.ops.cal(t, self.dark, self.flat, self.mosaic, use_dark, use_flat)
            self.ops.deb(self.mosaic, self.scratch, meta["bayer"])
        return self.scratch

    def _points(self, gray, level):
        """(points to match, stars found): the count is every star the detector
        kept, before the matcher's cap of 25 per cell."""
        cfg = self.p["cfg"]
        r = self.gd.detect_stars_full(
            gray, detection_sigma=self.ladder[level], min_area=cfg["minarea"],
            max_stars=20000, min_fwhm=cfg["min_fwhm"], max_ellipticity=cfg["max_ellipticity"])
        if r is None or len(r["x"]) == 0:
            return np.empty((0, 2), np.float32), 0
        x = np.asarray(r["x"], np.float32)
        return self.sa._grid_balance_uniform(
            x, np.asarray(r["y"], np.float32),
            self.h, self.w, (4, 4), 25, cfg["limit_stars"]), int(x.size)

    def frame(self, conn, idx, slot, meta, is_ref):
        torch = self.torch
        base = slot * self.cap
        if meta["kind"] == "u16":
            view = self.slots.view((self.H, self.W), np.int16, base)
        else:
            view = self.slots.view((self.H, self.W, self.C), np.float32, base)
        t = self.ops.upload(view)
        conn.send(("free", self.token, slot))
        x = self._to_hwc(meta, t)
        gray = torch.empty((self.h, self.w), dtype=torch.float32, device=self.dev)
        self.ops.bin(x, gray, self.ds)
        sky = self.ops.sky(x)
        if is_ref:
            found = [self._points(gray, k) for k in range(len(self.ladder))]
            self.ref_stars = found[0][1]
            self.weigh = self.weighted and engine.run_weighting(self.ref_stars)
            self.ref_sky = sky
            code = engine.frame_code(self.ref_stars, self.weigh)
            if not self.p["ref_only"]:
                self._accumulate(meta, t, sky, np.array([[1.0, 0, 0], [0, 1.0, 0]]), code)
            conn.send(("ref", self.token, [pts for pts, _n in found], self.ref_stars))
            conn.send(("acc_done", self.token, idx, 0.0, self.ref_stars,
                       float(engine.WEIGHT_TABLE[code])))
            return
        pts, stars = self._points(gray, 0)
        self.held[idx] = (meta, t, sky, gray, stars)
        conn.send(("pts", self.token, idx, 0, pts))

    def redetect(self, conn, idx, level):
        conn.send(("pts", self.token, idx, level, self._points(self.held[idx][3], level)[0]))

    def _accumulate(self, meta, t, sky, T, code):
        inv = np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2]
        off = (sky - self.ref_sky).astype(np.float32)
        self.ops.acc(self._to_hwc(meta, t), inv, off, int(code), self.weighted, self.A, self.wtab)
        return off

    def accumulate(self, conn, idx, T):
        meta, t, sky, _gray, stars = self.held.pop(idx)
        code = engine.frame_code(stars, self.weigh)
        self._accumulate(meta, t, sky, T, code)
        conn.send(("acc_done", self.token, idx, float((sky - self.ref_sky).mean()), stars,
                   float(engine.WEIGHT_TABLE[code])))

    def finish(self, conn):
        torch = self.torch
        self.held.clear()
        self.scratch = self.mosaic = None
        _empty_cache(torch, self.dev)
        out = torch.empty((self.H, self.W, self.C), dtype=torch.float32, device=self.dev)
        self.ops.fin(self.A, self.weighted, out, self.wtab)
        cov = int(self.A["cnt"].max())
        dst = torch.from_numpy(self.out.view((self.H, self.W, self.C), np.float32))
        self.ops.download(out, dst)
        conn.send(("result", self.token, cov))

    def close(self):
        if self.registered is not None:
            with contextlib.suppress(Exception):
                self.torch.cuda.synchronize()
                self.torch.cuda.cudart().cudaHostUnregister(self.registered)
        self.registered = None
        self.held = {}
        self.A = {}
        for n in ("wtab", "scratch", "mosaic", "dark", "flat"):
            self.__dict__.pop(n, None)
        self.slots.close()
        self.out.close()
        _empty_cache(self.torch, self.dev)


# ─────────────────────────────────────────────────────────────────────────────
# The GPU process
# ─────────────────────────────────────────────────────────────────────────────
_CPU_NAMES = ("sumv", "cnt", "wsum", "hi0", "hi1", "lo", "lo1", "qhi0", "qhi1", "qlo", "qlo1")


def _self_test(torch, dev, ops) -> None:
    """The pixel steps against the CPU code, on this card: exact for the
    compiled kernels, within float32's reach for torch operations (see
    _Portable). Not used if they disagree."""
    def check(got, want, tol, what, scaled=False):
        """scaled: a weighted sum carries its weights' size (thousands), and its
        float32 rounding with it; the trimmed mean is held to tol itself."""
        if tol is None:
            ok = np.array_equal(got, want, equal_nan=True)
        else:
            atol = tol * (max(1.0, float(np.abs(want).max())) if scaled else 1.0)
            ok = got.shape == want.shape and np.allclose(got, want, rtol=tol, atol=atol, equal_nan=True)
        if not ok:
            raise GpuUnavailable(f"the {what} self-test failed on this card")

    near = None if ops.exact else 1e-5            # sums, extremes, the trimmed mean
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
        ops.cal(t_raw, t_dark if use_dark else none, t_flat if use_flat else none, m, use_dark, use_flat)
        want = engine.calibrate_image(raw01, dark if use_dark else None,
                                      flat if use_flat else None)
        check(m.cpu().numpy(), want, None if ops.exact else 2e-6, "calibration")
        for bp in _CFA:
            x = torch.empty((H, W, C), dtype=torch.float32, device=dev)
            ops.deb(m, x, bp)
            cpu = debayer_fits_fast(want, bp, method="bilinear")
            if np.abs(x.cpu().numpy() - cpu).max() > 1e-6:
                raise GpuUnavailable("the debayer self-test failed on this card")

    for ds in (1, 3):
        h, w = H // ds, W // ds
        src = rng.normal(0.2, 0.1, (H, W, C)).astype(np.float32)
        binned = torch.empty((h, w), dtype=torch.float32, device=dev)
        ops.bin(torch.from_numpy(src).to(dev), binned, ds)
        want = src[:h * ds, :w * ds].astype(np.float64).mean(2).reshape(h, ds, w, ds).mean((1, 3))
        if np.abs(binned.cpu().numpy() - want).max() > 1e-6:
            raise GpuUnavailable("the binning self-test failed on this card")

    # Enough frames, with values on both sides of zero, that every branch of
    # the accumulator runs (highest, second highest, lowest, second lowest),
    # over shifts, a small rotation and a half turn, each frame with its own
    # weight; then the trim itself. Both ways the card runs: weighted, and with
    # every frame weighing 1 (no room for the weights), which must be the CPU
    # with every frame on the equal code.
    n_frames = engine.FULL_TRIM_FRAMES + 3
    frames = []
    for k in range(n_frames):
        f = rng.normal(0.0, 0.05, (H, W, C)).astype(np.float32) + np.float32(0.02)
        f[10:20, 30:50] = 0.0
        f[40, 40] = np.nan
        th = np.radians((0.0, 11.0, 180.0, -0.6)[k % 4])
        c, n = np.cos(th), np.sin(th)
        tx, ty = ((0.4, -0.7), (14.2, -9.7), (W - 1.3, H - 2.6), (3.1, 0.2))[k % 4]
        inv = np.array([[c, -n, tx], [n, c, ty]])
        off = np.array([0.011, -0.003, 0.002], np.float32) * np.float32(k % 3)
        frames.append((f, inv, off, 1 + (k * 97) % 255))           # codes 1 .. 255
    wtab = torch.from_numpy(engine.WEIGHT_TABLE).to(dev)
    inf = float("inf")
    for weighted in (1, 0):
        A = engine.SharedArrays.blank({n: np.empty((H, W, C), d) for n, d in engine.SharedArrays.SPEC})
        mk = lambda v: torch.full((H, W, C), v, dtype=torch.float32, device=dev)      # noqa: E731
        ws = (H, W, C) if weighted else (1,)
        g = {"sumv": mk(0.0), "cnt": torch.zeros((H, W, C), dtype=torch.int16, device=dev),
             "wsum": torch.zeros(ws, dtype=torch.float32, device=dev),
             "hi0": mk(-inf), "hi1": mk(-inf), "lo0": mk(inf), "lo1": mk(inf),
             **{n: torch.zeros(ws, dtype=torch.uint8, device=dev) for n in ACC_NAMES[7:]}}
        for f, inv, off, code in frames:
            q = code if weighted else engine.EQUAL_WEIGHT_CODE
            for b in range(engine.N_BANDS):
                engine.accumulate(A, f, inv, off, q, H * b // engine.N_BANDS,
                                  H * (b + 1) // engine.N_BANDS)
            ops.acc(torch.from_numpy(f).to(dev), inv, off, q, weighted, g, wtab)
        for gname, cname in zip(ACC_NAMES, _CPU_NAMES):
            if not weighted and gname not in ("sumv", "cnt", "hi0", "hi1", "lo0", "lo1"):
                continue
            x2 = g[gname].cpu().numpy()
            if gname == "cnt":
                x2 = x2.astype(np.uint16)
            exact = gname == "cnt" or gname.startswith("q")
            check(x2, A[cname], None if exact else near, f"accumulator ({gname})",
                  scaled=gname in ("sumv", "wsum"))
        cpu = np.empty((H, W, C), np.float32)
        engine.finish(A, cpu)
        out = torch.empty((H, W, C), dtype=torch.float32, device=dev)
        ops.fin(g, weighted, out, wtab)
        check(out.cpu().numpy(), cpu, near, "trimmed-mean")

    # Indices past 2^24, which a 96 x 131 frame never reaches: a 17 MP source
    # turned half round, so the first 64 output rows read its last rows.
    sH, sW = 2100, 8193
    f = ((((np.arange(sH)[:, None] * 7 + np.arange(sW)[None, :] * 13) % 9973) / 9973.0)
         + 0.1).astype(np.float32)[..., None]
    inv = np.array([[-1.0, 0.0, sW - 1.6], [0.0, -1.0, sH - 1.3]])
    off = np.zeros(1, np.float32)
    A = engine.SharedArrays.blank({n: np.empty((64, sW, 1), d) for n, d in engine.SharedArrays.SPEC})
    engine.accumulate(A, f, inv, off, engine.EQUAL_WEIGHT_CODE, 0, 64)
    g = {"sumv": torch.zeros((64, sW, 1), dtype=torch.float32, device=dev),
         "cnt": torch.zeros((64, sW, 1), dtype=torch.int16, device=dev),
         **{n: torch.full((64, sW, 1), v, dtype=torch.float32, device=dev)
            for n, v in (("hi0", -inf), ("hi1", -inf), ("lo0", inf), ("lo1", inf))},
         **{n: torch.zeros((1,), dtype=torch.float32 if n == "wsum" else torch.uint8, device=dev)
            for n in ("wsum",) + ACC_NAMES[7:]}}
    ops.acc(torch.from_numpy(f).to(dev), inv, off, engine.EQUAL_WEIGHT_CODE, 0, g, wtab)
    check(g["cnt"].cpu().numpy().astype(np.uint16), A["cnt"], None, "large-frame accumulator (cnt)")
    check(g["sumv"].cpu().numpy(), A["sumv"], near, "large-frame accumulator (sumv)")


def _boot():
    from setiastro.saspro.runtime_torch import import_torch, best_device
    try:
        torch = import_torch(prefer_cuda=True, status_cb=lambda *_: None, allow_install=False)
    except Exception as e:
        raise GpuUnavailable(f"PyTorch is not available ({type(e).__name__})")
    # SASpro's order: CUDA (NVIDIA) or ROCm (AMD), Intel XPU, DirectML, Apple Metal.
    try:
        dev = best_device(torch, prefer_cuda=True, prefer_xpu=True, prefer_dml=True)
    except Exception as e:
        raise GpuUnavailable(f"no GPU is available to PyTorch ({type(e).__name__})")
    if dev is None or dev.type == "cpu":
        raise GpuUnavailable("no GPU is available to PyTorch")
    from setiastro.saspro import gpu_star_detect as gd
    from setiastro.saspro import star_alignment as sa
    gd.use_device(torch, dev)                    # stars are found on the card that stacks
    ops, note = None, ""
    if dev.type == "cuda":
        try:
            ops = _Fused(torch, dev, _compile(torch))
            _self_test(torch, dev, ops)
        except Exception as e:
            note = str(e) if isinstance(e, GpuUnavailable) else f"{type(e).__name__}: {e}"
            ops = None
    if ops is None:
        ops = _Portable(torch, dev)
        _self_test(torch, dev, ops)
    _empty_cache(torch, dev)
    free, total = _memory(torch, dev)
    return torch, dev, ops, gd, sa, {"name": _device_name(torch, dev), "backend": _kind(torch, dev),
                                     "engine": ops.label, "note": note,
                                     "torch": str(torch.__version__), "free": free, "total": total}


# Accumulating and dropping free memory, so they jump the queue.
_FIRST = ("acc", "drop", "cancel", "finish", "quit", "detect")


def _server_main(conn) -> None:
    engine.die_with_parent()
    try:
        torch, dev, ops, gd, sa, info = _boot()
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
                run = _Run(torch, dev, ops, gd, sa, m[1], m[2])
                conn.send(("begun", run.token, run.max_held, run.weighted))
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


def gpu_possible():
    """Why the GPU process is not worth starting, or None. Cheap: it looks at
    the runtime folder and never imports torch. Every GPU build SASpro installs
    counts: CUDA, ROCm, Intel XPU, DirectML and Apple's Metal. When it cannot
    tell, it says yes and the GPU process finds out."""
    import glob
    import platform
    import sys
    if sys.platform == "darwin":
        if platform.machine().lower() in ("arm64", "aarch64"):
            return None
        return "PyTorch has no GPU it can use on an Intel Mac"
    if nvidia_driver_present():
        return None
    try:
        from setiastro.saspro.runtime_torch import _runtime_base_dir
        base = str(_runtime_base_dir())
    except Exception:
        return None
    sites = (glob.glob(os.path.join(base, "py3*", "venv", "Lib", "site-packages"))
             + glob.glob(os.path.join(base, "py3*", "venv", "lib", "python3*", "site-packages")))
    builds = []
    for sp in sites:
        builds += [os.path.basename(d)[len("torch-"):-len(".dist-info")].lower()
                   for d in glob.glob(os.path.join(sp, "torch-*.dist-info"))]
        if sys.platform == "win32" and glob.glob(os.path.join(sp, "torch_directml*")):
            builds.append("directml")
    if any(b == "directml" or "rocm" in b or "xpu" in b for b in builds):
        return None
    if not builds:
        if sys.platform.startswith("linux") and os.path.exists("/dev/kfd"):
            return None                       # an AMD ROCm driver; the torch may be elsewhere
        return "hardware acceleration is not installed"
    return "no GPU build of PyTorch is installed for this computer's graphics card"


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
            if why is None:
                why = gpu_possible()
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
    if session.info.get("note"):
        say(f"Compiled kernels not used ({session.info['note']}); the card runs torch operations")

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
        image, cov, oks, skipped, ref_stars, room = result
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
    # room: the card had the memory for the weights (else _loop said so).
    return engine.make_result(image, ref_hdr, nch, oks, skipped, frames, cov, t0, say, "GPU",
                              masters, ref_path, room and engine.run_weighting(ref_stars),
                              ref_stars if room else None)


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
    ref_stars = 0                                   # and its star count at the first
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
    max_held, room = m[2], m[3]
    if not room:
        say("GPU: not enough video memory to weight the frames by star count as well, "
            "so they weigh the same.")
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
            levels, ref_stars = m[2], m[3]
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
            oks.append((paths[idx], {**d, "offset": m[3], "stars": m[4], "weight": m[5]}))
            finish_frame()
        elif kind == "result":
            return (np.array(out.view((H, W, nch), np.float32))[..., 0] if nch == 1 else
                    np.array(out.view((H, W, nch), np.float32)), m[2], oks, skipped, ref_stars, room)
        pump()
