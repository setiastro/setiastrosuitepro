# src/setiastro/saspro/multiscale_decomp.py
from __future__ import annotations
import numpy as np
import cv2
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from PyQt6.QtCore import Qt, QTimer, QRect, QRectF, QSettings, QByteArray, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QIcon, QMovie
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QWidget, QLabel, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QToolButton, QSlider, QSplitter,
    QProgressDialog, QApplication
)
from contextlib import contextmanager
from functools import partial
from setiastro.saspro.thread_keeper import start_owned_thread
from setiastro.saspro.resources import get_resources, multiscale_decomp_path
from setiastro.saspro.widgets.before_after import (
    BeforeAfterCompare, display_stretch_params, apply_display_stretch,
)
# Preset drag grip (shortcuts imports this module lazily, so this top-level
# import does not create a cycle — same pattern as star_stretch/stat_stretch).
from setiastro.saspro.shortcuts import PresetDragHandle
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)  # 0 = let OpenCV decide
except Exception:
    pass
import math
from setiastro.saspro.help_support import make_help_button

_N_WORKERS = max(2, (os.cpu_count() or 4))

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

class _ZoomPanView(QGraphicsView):
    """
    QGraphicsView that supports wheel-zoom and click-drag panning.
    Calls on_view_changed() whenever viewport position/scale changes.
    """
    def __init__(self, *args, on_view_changed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._panning = False
        self._pan_start = None
        self._on_view_changed = on_view_changed  # callable or None

    def _notify(self):
        cb = self._on_view_changed
        if callable(cb):
            cb()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        ev.accept()
        self._notify()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_start is not None:
            delta = ev.pos() - self._pan_start
            self._pan_start = ev.pos()

            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            ev.accept()
            # scrollbars will trigger _notify via their signals too, but harmless:
            self._notify()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)



# ─────────────────────────────────────────────
# Core math (your backbone)
# ─────────────────────────────────────────────

_NOTED: set = set()


def _note_once(msg: str) -> None:
    """Print a fallback notice once per process, not once per blur call."""
    if msg in _NOTED or len(_NOTED) > 64:
        return
    _NOTED.add(msg)
    print(msg, flush=True)


def _blur_gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    H, W = img.shape[:2]
    k_sz = int(max(3, 2 * round(3 * sigma) + 1)) | 1

    # Small images — not worth any overhead
    if H * W < 256 * 256 or sigma < 0.5:
        return cv2.GaussianBlur(img, (k_sz, k_sz), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)

    # Try torch path first — uses GPU if available, CPU otherwise.
    # Both are genuinely parallel and don't fight with ThreadPoolExecutor.
    try:
        result = _blur_gaussian_torch(img, sigma, k_sz)
        return result
    except Exception as e:
        _note_once(f"[BLUR] torch path failed ({e}), falling back to tiled")

    # Torch unavailable — fall back to tiled scipy/cv2
    return _blur_gaussian_tiled(img, sigma, k_sz)


def _blur_gaussian_torch(img: np.ndarray, sigma: float, k_sz: int) -> np.ndarray:
    """
    Gaussian blur via torch conv2d — uses whatever device torch has available.
    Separable: 1D horiz kernel then 1D vert kernel, much faster than 2D.
    """
    try:
        # Import torch the SASpro way — already loaded, zero overhead
        import torch
    except ImportError:
        raise RuntimeError("torch not available")

    # Build 1D Gaussian kernel matching cv2's kernel exactly
    k1d = torch.tensor(
        cv2.getGaussianKernel(k_sz, sigma).flatten(),
        dtype=torch.float32
    )

    # Pick device — CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k1d = k1d.to(device)

    # Input tensor: (1, 1, H, W) for conv2d
    H, W = img.shape[:2]
    is_rgb = (img.ndim == 3)

    if is_rgb:
        # (H, W, C) -> (C, 1, H, W)
        t = torch.from_numpy(
            np.ascontiguousarray(img.transpose(2, 0, 1))
        ).float().unsqueeze(1).to(device)
        C = img.shape[2]
    else:
        # (H, W) -> (1, 1, H, W)
        t = torch.from_numpy(
            np.ascontiguousarray(img)
        ).float().unsqueeze(0).unsqueeze(0).to(device)
        C = 1

    pad = k_sz // 2

    # Horizontal pass: kernel shape (C, 1, 1, k_sz)
    kh = k1d.view(1, 1, 1, k_sz).expand(C, 1, 1, k_sz).contiguous()
    t = torch.nn.functional.pad(t, (pad, pad, 0, 0), mode='reflect')
    t = torch.nn.functional.conv2d(t, kh, groups=C)

    # Vertical pass: kernel shape (C, 1, k_sz, 1)
    kv = k1d.view(1, 1, k_sz, 1).expand(C, 1, k_sz, 1).contiguous()
    t = torch.nn.functional.pad(t, (0, 0, pad, pad), mode='reflect')
    t = torch.nn.functional.conv2d(t, kv, groups=C)

    # Back to numpy
    if is_rgb:
        # (C, 1, H, W) -> (H, W, C)
        result = t.squeeze(1).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    else:
        result = t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    return result


def _blur_gaussian_tiled(img: np.ndarray, sigma: float, k_sz: int) -> np.ndarray:
    """Tiled scipy fallback for when torch is unavailable."""
    if not _HAS_SCIPY:
        return cv2.GaussianBlur(img, (k_sz, k_sz), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)

    H, W = img.shape[:2]
    overlap = int(math.ceil(4.0 * sigma)) + 4
    n_tiles_target = _N_WORKERS * 2

    if H >= W:
        n_row = max(1, int(round(math.sqrt(n_tiles_target * H / max(W, 1)))))
        n_col = max(1, int(round(n_tiles_target / max(n_row, 1))))
    else:
        n_col = max(1, int(round(math.sqrt(n_tiles_target * W / max(H, 1)))))
        n_row = max(1, int(round(n_tiles_target / max(n_col, 1))))

    min_core = max(overlap * 2, 64)
    n_row = min(n_row, max(1, H // min_core))
    n_col = min(n_col, max(1, W // min_core))

    if n_row <= 1 and n_col <= 1:
        if img.ndim == 2:
            return _scipy_gaussian(img, sigma=sigma, mode='reflect').astype(np.float32)
        return _scipy_gaussian(img, sigma=(sigma, sigma, 0), mode='reflect').astype(np.float32)

    out = np.empty_like(img)
    row_edges = [int(round(H * i / n_row)) for i in range(n_row + 1)]
    col_edges = [int(round(W * j / n_col)) for j in range(n_col + 1)]

    def _process(rc):
        r, c = rc
        ry0, ry1 = row_edges[r], row_edges[r + 1]
        cx0, cx1 = col_edges[c], col_edges[c + 1]
        if ry1 <= ry0 or cx1 <= cx0:
            return
        py0 = max(0, ry0 - overlap); py1 = min(H, ry1 + overlap)
        px0 = max(0, cx0 - overlap); px1 = min(W, cx1 + overlap)
        crop = img[py0:py1, px0:px1]
        if crop.ndim == 2:
            blurred = _scipy_gaussian(crop, sigma=sigma, mode='reflect').astype(np.float32)
        else:
            blurred = _scipy_gaussian(crop, sigma=(sigma, sigma, 0), mode='reflect').astype(np.float32)
        oy0 = ry0 - py0; oy1 = oy0 + (ry1 - ry0)
        ox0 = cx0 - px0; ox1 = ox0 + (cx1 - cx0)
        if out.ndim == 2:
            out[ry0:ry1, cx0:cx1] = blurred[oy0:oy1, ox0:ox1]
        else:
            out[ry0:ry1, cx0:cx1, :] = blurred[oy0:oy1, ox0:ox1, :]

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        list(ex.map(_process, [(r, c) for r in range(n_row) for c in range(n_col)]))

    return out

class _Cancelled(Exception):
    """A superseded background job stopping between pyramid layers."""


def _check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise _Cancelled()


def multiscale_decompose(img01: np.ndarray, layers: int, base_sigma: float = 1.0,
                         cancelled=None):
    return _decompose_pyramid(img01, layers, base_sigma, cancelled)


def _decompose_channel(img2d: np.ndarray, layers: int, base_sigma: float,
                       cancelled=None):
    """Serial pyramid for one channel — parallel blur within each layer."""
    c = img2d.astype(np.float32, copy=False)
    details = []
    for k in range(layers):
        _check_cancel(cancelled)
        c_next = _blur_gaussian(c, base_sigma * (2 ** k))
        details.append(c - c_next)
        c = c_next
    return details, c


def _decompose_pyramid(img01: np.ndarray, layers: int, base_sigma: float,
                       cancelled=None):
    """(details, residual) of the a-trous pyramid, colour as HWC.

    Same values as running _decompose_channel on every plane (what this module
    always did). With torch the running coarse image stays on the device from
    one layer to the next instead of a host round trip per layer and plane;
    without torch, or whenever the device path does not apply, it is that
    per-plane path itself.
    """
    img = img01.astype(np.float32, copy=False)
    layers = int(layers)
    if img.ndim == 2:
        planes = [img]
    elif img.ndim == 3 and img.shape[2] >= 3:
        planes = [img[:, :, ch] for ch in range(img.shape[2])]
    else:
        return _decompose_channel(img, layers, base_sigma, cancelled)
    out = _decompose_planes_torch(img, planes, layers, base_sigma, cancelled)
    if out is not None:
        return out
    return _decompose_planes_host(img, planes, layers, base_sigma, cancelled)


def _decompose_planes_host(img, planes, layers, base_sigma, cancelled=None):
    """_decompose_channel per plane (colour planes on threads), as HWC."""
    if img.ndim == 2:
        return _decompose_channel(img, layers, base_sigma, cancelled)
    with ThreadPoolExecutor(max_workers=min(len(planes), _N_WORKERS)) as ex:
        results = list(ex.map(
            lambda p: _decompose_channel(p, layers, base_sigma, cancelled), planes))
    details = []
    for k in range(layers):
        details.append(np.stack([r[0][k] for r in results], axis=2))
        for r in results:
            r[0][k] = None                  # each plane goes once it is interleaved
    residual = np.stack([r[1] for r in results], axis=2)
    return details, residual


def _torch_for_blur():
    """(torch, device) exactly as _blur_gaussian_torch picks them, or (None, None)."""
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None, None
    return torch, dev


def _blur_plane_torch(torch, t, sigma: float, k_sz: int, device):
    """One blur of a (1, 1, H, W) float32 tensor already on `device`.

    The op sequence of _blur_gaussian_torch on a single plane (same kernel,
    pads, convolution shapes and groups), so the values are the same; only
    the host round trip around it is gone.
    """
    k1d = torch.tensor(
        cv2.getGaussianKernel(k_sz, sigma).flatten(),
        dtype=torch.float32
    ).to(device)
    pad = k_sz // 2
    kh = k1d.view(1, 1, 1, k_sz).expand(1, 1, 1, k_sz).contiguous()
    t = torch.nn.functional.pad(t, (pad, pad, 0, 0), mode='reflect')
    t = torch.nn.functional.conv2d(t, kh, groups=1)
    kv = k1d.view(1, 1, k_sz, 1).expand(1, 1, k_sz, 1).contiguous()
    t = torch.nn.functional.pad(t, (0, 0, pad, pad), mode='reflect')
    return torch.nn.functional.conv2d(t, kv, groups=1)


def _vram_ok(torch, dev, plane_bytes: int, n_planes: int) -> bool:
    """Room for the device-resident pyramid: running image, next level, one
    layer's detail and the blur temporaries. Otherwise the per-call path (one
    plane at a time) runs, as before."""
    try:
        free, _total = torch.cuda.mem_get_info(dev)
        free += max(0, torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev))
    except Exception:
        return True
    return free > plane_bytes * (3 * n_planes + 4) * 1.15


def _planes_to_device(torch, dev, hwc_img, planes):
    H, W = planes[0].shape
    if hwc_img is not None and hwc_img.ndim == 3:
        whole = torch.from_numpy(np.ascontiguousarray(hwc_img)).to(dev)   # one upload
        return [whole[:, :, ch].contiguous().view(1, 1, H, W)
                for ch in range(whole.shape[2])]
    return [torch.from_numpy(np.ascontiguousarray(p)).view(1, 1, H, W).to(dev)
            for p in planes]


def _pyramid_on_device(torch, dev, img, planes, sigmas, ks, cancelled):
    H, W = planes[0].shape
    hwc = img.ndim == 3
    host = planes            # host planes while the running image is on the host
    on_dev = None            # (1, 1, H, W) device tensors once it is on the device
    details = []
    for sigma, k_sz in zip(sigmas, ks):
        _check_cancel(cancelled)
        if sigma < 0.5:
            # _blur_gaussian hands sigma < 0.5 to cv2: same call, on the host.
            if host is None:
                host = [p.view(H, W).cpu().numpy() for p in on_dev]
                on_dev = None
            nxt = [_blur_gaussian(p, sigma) for p in host]
            det = [p - q for p, q in zip(host, nxt)]
            details.append(np.stack(det, axis=2) if hwc else det[0])
            host = nxt
            continue
        if on_dev is None:
            on_dev = _planes_to_device(torch, dev, img if host is planes else None, host)
            host = None
        nxt = [_blur_plane_torch(torch, p, sigma, k_sz, dev) for p in on_dev]
        if hwc:
            # The colour interleave happens on the device: one download per layer.
            d = torch.empty((H, W, len(nxt)), dtype=torch.float32, device=dev)
            for ch, (p, q) in enumerate(zip(on_dev, nxt)):
                d[:, :, ch] = (p - q).view(H, W)
            details.append(d.cpu().numpy())
            del d
        else:
            details.append((on_dev[0] - nxt[0]).view(H, W).cpu().numpy())
        on_dev = nxt
    if on_dev is None:
        residual = np.stack(host, axis=2) if hwc else host[0]
    elif hwc:
        residual = torch.stack([p.view(H, W) for p in on_dev], dim=2).cpu().numpy()
    else:
        residual = on_dev[0].view(H, W).cpu().numpy()
    return details, residual


def _decompose_planes_torch(img, planes, layers, base_sigma, cancelled):
    """Device-resident pyramid, or None to run the per-call path instead.

    None whenever _blur_gaussian would not take the torch path for every
    layer it could (small image, reflect pad wider than the image), when
    torch is missing, or when the device runs out of room — the per-call
    path then produces exactly what it always did.
    """
    H, W = planes[0].shape
    if layers <= 0 or H * W < 256 * 256:
        return None
    sigmas = [base_sigma * (2 ** k) for k in range(layers)]
    ks = [int(max(3, 2 * round(3 * s) + 1)) | 1 for s in sigmas]
    if any(s >= 0.5 and (k // 2) >= min(H, W) for s, k in zip(sigmas, ks)):
        return None
    torch, dev = _torch_for_blur()
    if torch is None:
        return None
    if dev.type == "cuda" and not _vram_ok(torch, dev, H * W * 4, len(planes)):
        return None
    try:
        return _pyramid_on_device(torch, dev, img, planes, sigmas, ks, cancelled)
    except _Cancelled:
        raise
    except Exception as e:
        _note_once(f"[MSD] device pyramid unavailable ({type(e).__name__}: {e}); "
                   f"using the per-call path")
        if dev.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return None


def _full_decomposition(work: np.ndarray, layers: int, base_sigma: float,
                        cancelled=None):
    """The dialog's whole-image pyramid: (details, residual, layer noise).

    Noise is measured on channel 0 of a colour layer and on the whole plane
    for mono, as the dialog always has.
    """
    img = work.astype(np.float32, copy=False)
    details, residual = _decompose_pyramid(img, layers, base_sigma, cancelled)
    # Finite input gives finite layers (normalised blurs of [0, 1] data and
    # their differences), which lets _robust_sigma sample without copying.
    finite = bool(np.isfinite(img).all())

    def _sigma_for_layer(w):
        if not w.size:
            return 1e-6
        if w.ndim == 3:
            return _robust_sigma(w[:, :, 0], assume_finite=finite)
        return _robust_sigma(w, assume_finite=finite)

    noise = [_sigma_for_layer(w) for w in details]
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()     # hand the pyramid's VRAM back
        except Exception:
            pass
    return details, residual, noise


def multiscale_reconstruct(details, residual):
    out = residual.astype(np.float32, copy=True)
    for w in details:
        out += w
    return out

def detail_only_vis(out_raw, tuned, gain: float = 4.0):
    """
    Mid-grey visualization of a detail-only reconstruction (residual off).

    Detail layers are signed and centred near zero. Summing several of them
    (multiscale_reconstruct on the enabled/tuned layers) stacks their
    amplitude, so a fixed gain blows out to pure white/black once more than
    one layer contributes. We divide the summed structure by the number of
    CONTRIBUTING layers (those not identically zero — disabled layers are
    passed in as zeros) before the mid-grey shift, i.e. we show the MEAN of
    the active layers' structure times `gain`. That keeps the display in
    range no matter how many layers are enabled, and reduces to the original
    behaviour for a single layer.
    """
    n = 0
    for w in tuned:
        if w is not None and w.any():
            n += 1
    return _detail_vis_n(out_raw, n, gain)


def _detail_vis_n(out_raw, n: int, gain: float = 4.0):
    """detail_only_vis given the number of contributing layers."""
    n = max(1, n)
    d = out_raw.astype(np.float32, copy=False) / float(n)
    return np.clip(0.5 + d * float(gain), 0.0, 1.0).astype(np.float32, copy=False)

def soft_threshold(x: np.ndarray, t: float):
    a = np.abs(x)
    return np.sign(x) * np.maximum(0.0, a - t)

def apply_layer_ops(
    w: np.ndarray,
    bias_gain: float,
    thr_sigma: float,
    amount: float,
    denoise_strength: float = 0.0,
    sigma: float | np.ndarray | None = None,
    layer_index: int | None = None,
    *,
    mode: str = "μ–σ Thresholding",
):
    w2 = w

    m = (mode or "").strip().lower()
    is_linear = m.startswith("linear")
    if is_linear:
        return w * bias_gain if abs(bias_gain - 1.0) > 1e-6 else w

    # 1) Noise reduction step (MMT-style NR)
    if denoise_strength > 0.0:
        if sigma is None:
            sigma = _robust_sigma(w2)
        sigma_f = float(sigma)

        i = int(layer_index or 0)

        # --- SMOOTH scaling option (pick ONE) ---
        # Option A: linear growth (very controllable)
        # scale = 1.0 + 0.75 * i

        # Option B: sqrt growth of 2^i (gentle, "natural")
        scale = (2.0 ** i) ** 0.5   # 1, 1.41, 2, 2.83, 4, ...

        # Base: 3σ at denoise=1 for layer 0, increases by scale
        t_dn = denoise_strength * 3.0 * scale * sigma_f

        if t_dn > 0.0:
            w_dn = soft_threshold(w2, t_dn)
            w2 = (1.0 - denoise_strength) * w2 + denoise_strength * w_dn

    # 2) Threshold in σ units + bias shaping
    if thr_sigma > 0.0:
        if sigma is None:
            sigma = _robust_sigma(w2)
        sigma_f = float(sigma)
        t = thr_sigma * sigma_f
        if t > 0.0:
            wt = soft_threshold(w2, t)
            w2 = (1.0 - amount) * w2 + amount * wt

    if abs(bias_gain - 1.0) > 1e-6:
        w2 = w2 * bias_gain
    return w2

def _robust_sigma(arr: np.ndarray, assume_finite: bool = False) -> float:
    MAX_SAMPLES = 100_000
    a0 = np.asarray(arr, dtype=np.float32)
    if assume_finite and a0.size > MAX_SAMPLES:
        # The same grid sample as below, gathered straight from the layer:
        # with nothing non-finite to drop, element i of the filtered ravel is
        # element i of the ravel, so the full-size ravel/isfinite/compress
        # copies are not needed.
        step = a0.size // MAX_SAMPLES
        idx = np.arange(0, a0.size, step)
        if a0.flags.c_contiguous:
            a = a0.reshape(-1)[idx]
        else:
            a = a0[np.unravel_index(idx, a0.shape)]
    else:
        a = a0.ravel()
        # Remove non-finite values
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 1e-6

        # For large arrays, use a grid subsample instead of random.choice
        # Grid subsample is O(1) index math vs O(N) random permutation
        if a.size > MAX_SAMPLES:
            step = a.size // MAX_SAMPLES
            a = a[::step]

    med = np.median(a)
    mad = np.median(np.abs(a - med))
    if mad <= 0:
        s = float(np.std(a))
        return s if s > 0 else 1e-6

    return float(1.4826 * mad)


# ─────────────────────────────────────────────
# Layer config
# ─────────────────────────────────────────────


@dataclass
class LayerCfg:
    enabled: bool = True
    bias_gain: float = 1.0        # 1.0 = unchanged
    thr: float = 0.0              # soft threshold in detail domain
    amount: float = 0.0           # 0..1 blend toward thresholded
    denoise: float = 0.0          # 0..1 additional noise reduction


# ─────────────────────────────────────────────
# Row-block evaluation and background jobs
# ─────────────────────────────────────────────

# Elementwise float32 work is cut into row blocks of about this many values:
# the whole op chain then runs in cache and on several cores. Every op is
# elementwise, so the result is bit-identical to the whole-array evaluation.
_BLOCK_ELEMS = 1 << 17
_N_BLOCK_WORKERS = max(2, min(8, os.cpu_count() or 4))


def _row_spans(shape) -> list:
    rows = int(shape[0]) if len(shape) else 0
    per_row = 1
    for s in shape[1:]:
        per_row *= int(s)
    step = max(1, _BLOCK_ELEMS // max(1, per_row))
    return [(r, min(rows, r + step)) for r in range(0, rows, step)]


def _run_spans(fn, spans) -> list:
    if len(spans) <= 1:
        return [fn(sp) for sp in spans]
    with ThreadPoolExecutor(max_workers=min(_N_BLOCK_WORKERS, len(spans))) as ex:
        return list(ex.map(fn, spans))


def _rep3(a: np.ndarray) -> np.ndarray:
    """np.repeat(a[:, :, None], 3, axis=2) for a 2-D a, in row blocks: np.repeat
    holds the GIL for its whole run (~0.3 s at 26 MP), which froze the GUI
    even with the work on a thread. Same values."""
    out = np.empty(a.shape[:2] + (3,), dtype=a.dtype)

    def one(sp):
        r0, r1 = sp
        out[r0:r1] = a[r0:r1, :, None]

    _run_spans(one, _row_spans(out.shape))
    return out


def _to_u8(img: np.ndarray) -> np.ndarray:
    """np.clip(img * 255.0, 0, 255).astype(np.uint8), in row blocks."""
    a = np.asarray(img)
    out = np.empty(a.shape, dtype=np.uint8)

    def one(sp):
        r0, r1 = sp
        out[r0:r1] = np.clip(a[r0:r1] * 255.0, 0, 255).astype(np.uint8)

    _run_spans(one, _row_spans(a.shape))
    return out


def _rgb01_to_qimage(img: np.ndarray) -> QImage:
    """The dialog's float [0, 1] -> 8-bit RGB display conversion, ending at a
    detached QImage, so it can run off the GUI thread (a QPixmap cannot).
    Same pixels, already in RGB32: QPixmap.fromImage would otherwise convert
    RGB888 on the GUI thread (~90 ms for 22 MP)."""
    arr = _to_u8(img)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = _rep3(arr)
    arr = np.ascontiguousarray(arr)
    q = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.convertToFormat(QImage.Format.Format_RGB32)


def _base_qimage(img: np.ndarray, cancelled=None) -> QImage:
    return _rgb01_to_qimage(img)


def _ops_noop(cfg: LayerCfg, mode: str) -> bool:
    """apply_layer_ops would hand the layer back untouched."""
    gain_off = abs(cfg.bias_gain - 1.0) <= 1e-6
    if (mode or "").strip().lower().startswith("linear"):
        return gain_off
    return gain_off and not (cfg.denoise > 0.0) and not (cfg.thr > 0.0)


def _tune_one(w, cfg: LayerCfg, sigma, i: int, mode: str, pass_index: bool):
    """One layer through apply_layer_ops exactly as the dialog calls it.
    pass_index=False reproduces Send/Split, which never passed layer_index."""
    if not cfg.enabled:
        return np.zeros_like(w)
    if pass_index:
        return apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount, cfg.denoise,
                               sigma, layer_index=i, mode=mode)
    return apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount, cfg.denoise,
                           sigma, mode=mode)


def _blockable(w, cfg: LayerCfg, sigma, mode: str) -> bool:
    """Row blocks give the whole-layer result unless apply_layer_ops has to
    measure sigma itself (that is a whole-layer statistic)."""
    if w.ndim == 0 or w.shape[0] < 2:
        return False
    if sigma is not None or not cfg.enabled:
        return True
    if (mode or "").strip().lower().startswith("linear"):
        return True
    return not (cfg.denoise > 0.0) and not (cfg.thr > 0.0)


def _tune_layer(w, cfg: LayerCfg, sigma, i: int, mode: str, pass_index: bool = True):
    """_tune_one for a whole layer, in row blocks on threads."""
    if not cfg.enabled:
        return np.zeros_like(w)
    if _ops_noop(cfg, mode) or not _blockable(w, cfg, sigma, mode):
        return _tune_one(w, cfg, sigma, i, mode, pass_index)
    out = np.empty(w.shape, dtype=np.float32)

    def one(sp):
        r0, r1 = sp
        out[r0:r1] = _tune_one(w[r0:r1], cfg, sigma, i, mode, pass_index)

    _run_spans(one, _row_spans(w.shape))
    return out


def _tuned_reconstruct(details, residual, cfgs, layer_noise, mode: str,
                       residual_enabled: bool, *, pass_index: bool = True,
                       clip01: bool = False):
    """multiscale_reconstruct(tuned, residual or zeros) — optionally clipped
    to [0, 1] — plus which tuned layers are non-zero (detail_only_vis's count),
    without holding the tuned layers: row blocks on threads.

    Bit-identical to tuning whole layers and summing them: every step is
    elementwise float32, the sum keeps the same order (residual, then layer
    0, 1, ...), and each layer keeps the sigma measured on the whole layer.
    """
    n = len(details)
    sig = [layer_noise[i] if layer_noise and i < len(layer_noise) else None
           for i in range(n)]
    if not all(_blockable(details[i], cfgs[i], sig[i], mode) for i in range(n)):
        tuned = [_tune_one(details[i], cfgs[i], sig[i], i, mode, pass_index)
                 for i in range(n)]
        res = residual if residual_enabled else np.zeros_like(residual)
        out = multiscale_reconstruct(tuned, res)
        flags = [bool(t is not None and t.any()) for t in tuned]
        if clip01:
            out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        return out, flags

    out = np.empty(residual.shape, dtype=np.float32)

    def one(sp):
        r0, r1 = sp
        r = residual[r0:r1]
        o = (r if residual_enabled else np.zeros_like(r)).astype(np.float32, copy=True)
        hit = []
        for i in range(n):
            t = _tune_one(details[i][r0:r1], cfgs[i], sig[i], i, mode, pass_index)
            o += t
            hit.append(bool(t.any()))
        if clip01:
            o = np.clip(o, 0.0, 1.0)
        out[r0:r1] = o
        return hit

    hits = _run_spans(one, _row_spans(residual.shape))
    flags = [any(h[i] for h in hits) for i in range(n)]
    return out, flags


def _blend_masked(base: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standard SASpro mask blend: base * (1 - m) + out * m (see
    MultiscaleDecompDialog._blend_with_mask)."""
    if out.ndim == 3:
        # (H, W, 1) and (H, W, 3) both need an extra axis for broadcast.
        m = mask[..., None]
    else:
        m = mask
    result = (base.astype(np.float32) * (1.0 - m)
              + out.astype(np.float32) * m)
    if not np.isfinite(result).all():
        return out
    return result.astype(np.float32)


def _restore_and_mask(out: np.ndarray, s: dict) -> np.ndarray:
    """Back to the document's shape, then the active-mask blend (Apply and
    Send to New Document)."""
    if s["orig_mono"]:
        if out.ndim == 3:
            out2d = out[:, :, 0]
        else:
            out2d = out
        if len(s["orig_shape"]) == 2:
            out_final = out2d.astype(np.float32, copy=False)
        else:
            out_final = out2d[:, :, None].astype(np.float32, copy=False)
    else:
        if out.ndim == 2:
            out_final = np.repeat(out[:, :, None], 3, axis=2).astype(np.float32, copy=False)
        else:
            out_final = out[:, :, :3].astype(np.float32, copy=False)

    # Base is the image captured at dialog open — the exact source the
    # decomposition was computed from.
    mask2d = s["mask"]
    if mask2d is not None:
        base_src = s["orig_for_mask"].astype(np.float32, copy=False)
        if out_final.ndim == 2:
            base_match = base_src[..., 0]
        elif out_final.ndim == 3 and out_final.shape[2] == 1:
            base_match = base_src[..., 0:1]
        else:
            base_match = base_src
        out_final = _blend_masked(base_match, out_final, mask2d)
    return out_final


def _split_outputs(s: dict):
    """Split Layers to Documents: ([(title, image) per detail layer], (title, residual))."""
    details, residual, noise = s["details"], s["residual"], s["noise"]
    orig_mono, orig_shape = s["orig_mono"], s["orig_shape"]
    mask2d = s["mask"]
    entries = []
    for i, w in enumerate(details):
        sigma = noise[i] if noise is not None and i < len(noise) else None
        d = _tune_layer(w, s["cfgs"][i], sigma, i, s["mode"], pass_index=False)
        d = d.astype(np.float32, copy=False)
        vis = np.clip(0.5 + d * 4.0, 0.0, 1.0).astype(np.float32, copy=False)
        del d
        if orig_mono:
            # multiscale_decompose returns 2D detail layers for mono input.
            mono = vis[..., 0] if vis.ndim == 3 else vis
            if len(orig_shape) == 3 and orig_shape[2] == 1:
                out_final = mono[:, :, None].astype(np.float32, copy=False)
            else:
                out_final = mono.astype(np.float32, copy=False)
        else:
            out_final = vis
        # Outside the mask: flat 0.5 grey, the "zero detail" baseline.
        if mask2d is not None:
            neutral = np.full_like(out_final, 0.5, dtype=np.float32)
            out_final = _blend_masked(neutral, out_final, mask2d)
        entries.append((f"Multiscale Detail Layer {i+1}", out_final))

    res = residual.astype(np.float32, copy=False)
    res_img = np.clip(res, 0.0, 1.0)
    if orig_mono:
        mono = res_img[..., 0] if res_img.ndim == 3 else res_img
        if len(orig_shape) == 3 and orig_shape[2] == 1:
            res_final = mono[:, :, None].astype(np.float32, copy=False)
        else:
            res_final = mono.astype(np.float32, copy=False)
    else:
        res_final = res_img
    # Outside the mask the ORIGINAL image, so detail_1 + ... + residual still
    # recombines to the source there.
    if mask2d is not None:
        base_src = s["orig_for_mask"].astype(np.float32, copy=False)
        if res_final.ndim == 2:
            base_match = base_src[..., 0]
        elif res_final.ndim == 3 and res_final.shape[2] == 1:
            base_match = base_src[..., 0:1]
        else:
            base_match = base_src
        res_final = _blend_masked(base_match, res_final, mask2d)
    return entries, ("Multiscale Residual Layer", res_final)


def _action_compute(s: dict, cancelled=None):
    """The numbers behind Apply / Send to New Document / Split Layers."""
    kind = s["kind"]
    if kind == "split":
        return _split_outputs(s)
    # Apply passes layer_index (denoise grows with the layer); Send never did.
    by_index = (kind == "apply")
    if s["residual_enabled"]:
        out, _ = _tuned_reconstruct(s["details"], s["residual"], s["cfgs"], s["noise"],
                                    s["mode"], True, pass_index=by_index, clip01=True)
    else:
        out_raw, flags = _tuned_reconstruct(s["details"], s["residual"], s["cfgs"],
                                            s["noise"], s["mode"], False,
                                            pass_index=by_index)
        out = _detail_vis_n(out_raw, sum(flags))
    return _restore_and_mask(out, s)


def _preview_stretch_params(s: dict):
    p = s.get("stretch")
    if p is None:
        p = display_stretch_params(s["work"], target_median=0.25, linked=False,
                                   blackpoint_sigma=3.0, ignore_zeros=True)
        s["stretch"] = p
        s["stretch_new"] = True
    return p


def _stretch_for_preview(img: np.ndarray, s: dict) -> np.ndarray:
    """MultiscaleDecompDialog._apply_preview_stretch on a snapshot."""
    if img.ndim == 2:
        img3 = _rep3(img)
    else:
        img3 = img
    if not s["autostretch"]:
        return img3
    try:
        return apply_display_stretch(img3, _preview_stretch_params(s))
    except Exception:
        return img3


def _roi_preview_image(s: dict, cancelled=None):
    """Fast ROI preview of the visible area: (float RGB crop, rect) or (None, None)."""
    vis = s.get("vis")
    if vis is None:
        return None, None
    x0, y0, x1, y1 = vis
    H, W = s["image_shape"][:2]

    MAX = 1400
    w = x1 - x0
    h = y1 - y0
    if w > MAX:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - MAX // 2)
        x1 = min(W, x0 + MAX)
    if h > MAX:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - MAX // 2)
        y1 = min(H, y0 + MAX)

    layers = s["layers"]
    base_sigma = s["base_sigma"]
    if layers <= 0:
        return None, None

    sigma_max = base_sigma * (2 ** (layers - 1))
    pad = int(np.ceil(3.0 * sigma_max)) + 2
    px0 = max(0, x0 - pad)
    py0 = max(0, y0 - pad)
    px1 = min(W, x1 + pad)
    py1 = min(H, y1 + pad)

    crop = s["work"][py0:py1, px0:px1].astype(np.float32, copy=False)
    details, residual = multiscale_decompose(crop, layers=layers, base_sigma=base_sigma,
                                             cancelled=cancelled)
    finite = bool(np.isfinite(crop).all())
    layer_noise = [_robust_sigma(d, assume_finite=finite) if d.size else 1e-6
                   for d in details]

    cx0 = x0 - px0
    cy0 = y0 - py0
    sel = s["sel"]
    if sel is not None and sel not in ("final", "residual"):
        # Detail layer — skip stretch, mid-gray viz
        i = int(sel)
        t = _tune_layer(details[i], s["cfgs"][i], layer_noise[i], i, s["mode"])
        vis_img = np.clip(0.5 + (t * 4.0), 0.0, 1.0).astype(np.float32, copy=False)
        if vis_img.ndim == 2:
            vis_img = _rep3(vis_img)
        return vis_img[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)], (x0, y0, x1, y1)

    if sel == "residual":
        out = np.clip(residual, 0.0, 1.0).astype(np.float32, copy=False)
        out = _stretch_for_preview(out, s)
    elif not s["residual_enabled"]:
        # Mid-grey visualisation, already display-scaled: not stretched,
        # same as the full-frame path.
        out_raw, flags = _tuned_reconstruct(details, residual, s["cfgs"], layer_noise,
                                            s["mode"], False)
        out = _detail_vis_n(out_raw, sum(flags))
        if out.ndim == 2:
            out = _rep3(out)
    else:
        out, _ = _tuned_reconstruct(details, residual, s["cfgs"], layer_noise,
                                    s["mode"], True, clip01=True)
        out = _stretch_for_preview(out, s)

    cx1 = cx0 + (x1 - x0)
    cy1 = cy0 + (y1 - y0)
    return out[cy0:cy1, cx0:cx1], (x0, y0, x1, y1)


def _full_preview_image(s: dict) -> np.ndarray:
    """Full-frame preview (Fast ROI off) from the cached whole-image pyramid."""
    details, residual, noise = s["details"], s["residual"], s["noise"]
    sel = s["sel"]
    if sel is not None and sel not in ("final", "residual"):
        # Detail layer view — skip stretch, mid-gray viz is already scaled
        i = int(sel)
        sigma = noise[i] if noise and i < len(noise) else None
        t = _tune_layer(details[i], s["cfgs"][i], sigma, i, s["mode"])
        vis = np.clip(0.5 + (t * 4.0), 0.0, 1.0).astype(np.float32, copy=False)
        if vis.ndim == 2:
            vis = _rep3(vis)
        return vis
    if sel == "residual":
        return _stretch_for_preview(np.clip(residual, 0, 1), s)
    if not s["residual_enabled"]:
        out_raw, flags = _tuned_reconstruct(details, residual, s["cfgs"], noise,
                                            s["mode"], False)
        vis = _detail_vis_n(out_raw, sum(flags))
        if vis.ndim == 2:
            vis = _rep3(vis)
        return vis
    # Autostretch applies here — on the linear reconstruct
    out, _ = _tuned_reconstruct(details, residual, s["cfgs"], noise, s["mode"], True,
                                clip01=True)
    return _stretch_for_preview(out, s)


def _preview_job(s: dict, cancelled=None) -> dict:
    """Everything a preview needs, ending at QImages for the GUI thread."""
    out = {"kind": s["kind"], "qimage": None, "rect": None,
           "before": None, "before_key": s["autostretch"], "stretch": None}
    if s["kind"] == "roi":
        img, rect = _roi_preview_image(s, cancelled)
        if img is not None:
            out["qimage"] = _rgb01_to_qimage(img)
            out["rect"] = rect
    else:
        out["qimage"] = _rgb01_to_qimage(_full_preview_image(s))
    if s["need_before"]:
        # The comparison's "before": the source through the same display stretch.
        out["before"] = _rgb01_to_qimage(_stretch_for_preview(s["image"], s))
    if s.get("stretch_new"):
        out["stretch"] = s["stretch"]
    return out


class _MSDJob(QThread):
    """One unit of Multiscale work off the GUI thread.

    Parentless and started with thread_keeper.start_owned_thread, so the
    dialog can close (and be deleted) mid-run. Results leave through `result`
    (None when cancelled), never through a shadowed `finished`.
    """
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, key=None):
        super().__init__(None)
        self._fn = fn
        self.key = key
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel or self.isInterruptionRequested()

    def run(self):
        fn, self._fn = self._fn, None
        try:
            out = fn(self._cancelled)
        except _Cancelled:
            out = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.result.emit(out)



# ─────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────

class MultiscaleDecompDialog(QDialog):
    def __init__(self, parent, doc, headless: bool = False):
        super().__init__(parent)
        # True when constructed for a headless preset apply (drop-on-image):
        # skip the modal init progress + initial preview, and don't persist geometry.
        self._headless = bool(headless)
        self.setWindowTitle(self.tr("Multiscale Decomposition"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.setMinimumSize(1050, 700)
        self.residual_enabled = True
        self._layer_noise = None  # list[float] per detail layer
        self._cached_coarse = None
        self._cached_img_id = None
        # Background work (_MSDJob): at most one job of each kind at a time.
        self._closing = False
        self._base_job = None        # source -> base/before pixmap at open
        self._decomp_job = None      # whole-image pyramid (Apply, full-frame preview)
        self._preview_job = None     # ROI or full-frame preview
        self._action_job = None      # Apply / Send / Split numbers
        self._decomp_waiters = []
        self._decomp_failed_key = None
        self._preview_again = False
        self._full_wait = False
        self._action_kind = None
        self._action_prog = None
        self._action_cursor = False
        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Document has no image.")

        # normalize to float32 [0..1] ...
        img0 = np.asarray(base)
        is_int = (img0.dtype.kind in "ui")

        img = img0.astype(np.float32, copy=False)
        if is_int:
            maxv = float(np.nanmax(img0)) or 1.0
            img = img / max(1.0, maxv)
        img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        self._orig_shape = img.shape
        self._orig_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

        # Keep a mono working image for computation, 3ch only for display.
        # `img` is private (np.clip made it) and nothing here writes into these
        # arrays, so they share it rather than holding four copies.
        if img.ndim == 2:
            self._image_work = img                 # (H, W) — used for decomp
            img3 = _rep3(img)
        elif img.ndim == 3 and img.shape[2] == 1:
            self._image_work = np.ascontiguousarray(img[:, :, 0]) # (H, W)
            img3 = _rep3(self._image_work)
        else:
            self._image_work = np.ascontiguousarray(img[:, :, :3]) # RGB stays as-is
            img3 = img[:, :, :3]

        self._image = img3             # 3ch for display only
        self._orig_for_mask = img3
        self._preview_img = img3

        # decomposition cache
        self._cached_layers = None
        self._cached_residual = None
        self._cached_key = None

        # per-layer configs
        self.layers = 4
        self.base_sigma = 1.0
        self.cfgs: list[LayerCfg] = [LayerCfg() for _ in range(self.layers)]

        # debounce preview updates
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._rebuild_preview)

        # Whole-image pyramid for Apply, rebuilt in the background once a
        # layers/sigma change has settled.
        self._prefetch_timer = QTimer(self)
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.setInterval(700)
        self._prefetch_timer.timeout.connect(self._prefetch_decomp)

        # One display stretch for before, after and every ROI crop (lazy).
        self._disp_stretch = None
        self._pm_before_key = None

        self._build_ui()
        # Reset returns to exactly what the dialog opened with.
        self._default_preset = self._multiscale_params()
        H, W = self._image.shape[:2]
        self._cmp.set_image_size(W, H)

        # Headless preset apply drives seed_from_preset()+_commit_to_doc() itself,
        # both of which recompute the pyramid on demand — so skip the interactive
        # init compute and the initial preview/fit.
        if not self._headless:
            # Layers table, preview combo and editor for the opening layer count
            # (this used to happen at the end of the synchronous pyramid build).
            self._sync_cfgs_and_ui()
            # The dialog opens at once. The source goes up as the base image —
            # with the opening settings the reconstruction is the source — and
            # the whole-image pyramid Apply needs is built off the GUI thread.
            self._start_base_job()
            self._request_decomp(None)
            QTimer.singleShot(0, self._fit_view)


    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.splitter)

        # ----- LEFT: preview -----
        left_widget = QWidget(self)
        left = QVBoxLayout(left_widget)

        self.scene = QGraphicsScene(self)

        self.view = _ZoomPanView(self.scene, on_view_changed=self._schedule_roi_preview)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Base full-image item (keeps zoom/pan working)
        self.pix_base = QGraphicsPixmapItem()
        self.pix_base.setOffset(0, 0)
        self.scene.addItem(self.pix_base)

        # ROI overlay item (updates fast)
        self.pix_roi = QGraphicsPixmapItem()
        self.pix_roi.setZValue(10)  # draw above base
        self.scene.addItem(self.pix_roi)

        # Before/after comparison: base + ROI overlay together are the "after";
        # the source image goes underneath through the same display stretch.
        self._cmp = BeforeAfterCompare(self.view, self.scene, [self.pix_base, self.pix_roi],
                                       settings_key="msdecomp/compare", host=self)
        self._cmp.fitRequested.connect(self._fit_view)
        left.addWidget(self._cmp.toolbar)

        left.addWidget(self.view)
        # Busy overlay (shown during recompute)
        self.busy_label = QLabel(self.tr("Computing…"), self.view.viewport())
        self.busy_label.setStyleSheet("""
            QLabel {
                background: rgba(0,0,0,140);
                color: white;
                padding: 6px 10px;
                border-radius: 8px;
                font-weight: 600;
            }
        """)
        self.busy_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.busy_label.hide()
        # --- Spinner (animated) ---
        self.busy_spinner = QLabel()
        self.busy_spinner.setFixedSize(20, 20)
        self.busy_spinner.setToolTip(self.tr("Computing…"))
        self.busy_spinner.setVisible(False)

        gif_path = get_resources().SPINNER_GIF  # <- canonical, works frozen/dev
        gif_path = os.path.normpath(gif_path)

        self._busy_movie = QMovie(gif_path)
        self._busy_movie.setScaledSize(self.busy_spinner.size())
        self.busy_spinner.setMovie(self._busy_movie)

        self._busy_show_timer = QTimer(self)
        self._busy_show_timer.setSingleShot(True)
        self._busy_show_timer.timeout.connect(self._show_busy_overlay)
        self._busy_depth = 0
        zoom_row = QHBoxLayout()

        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setIcon(QIcon.fromTheme("zoom-out"))
        self.zoom_out_btn.setToolTip(self.tr("Zoom Out"))

        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setIcon(QIcon.fromTheme("zoom-in"))
        self.zoom_in_btn.setToolTip(self.tr("Zoom In"))

        self.fit_btn = QToolButton()
        self.fit_btn.setIcon(QIcon.fromTheme("zoom-fit-best"))
        self.fit_btn.setToolTip(self.tr("Fit to Preview"))

        self.one_to_one_btn = QToolButton()
        self.one_to_one_btn.setIcon(QIcon.fromTheme("zoom-original"))
        self.one_to_one_btn.setToolTip("1:1")

        self.zoom_out_btn.clicked.connect(lambda: (self.view.scale(0.8, 0.8), self._schedule_roi_preview()))
        self.zoom_in_btn.clicked.connect(lambda: (self.view.scale(1.25, 1.25), self._schedule_roi_preview()))
        self.fit_btn.clicked.connect(self._fit_view)
        self.one_to_one_btn.clicked.connect(self._one_to_one)

        zoom_row.addStretch(1)
        zoom_row.addWidget(self.zoom_out_btn)
        zoom_row.addWidget(self.zoom_in_btn)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.fit_btn)
        zoom_row.addWidget(self.one_to_one_btn)
        zoom_row.addSpacing(10)
        zoom_row.addWidget(self.busy_spinner)   # <-- add here
        zoom_row.addStretch(1)

        left.addLayout(zoom_row)

        # ----- RIGHT: controls -----
        right_widget = QWidget(self)
        right = QVBoxLayout(right_widget)

        gb_global = QGroupBox(self.tr("Global"))
        form = QFormLayout(gb_global)

        self.spin_layers = QSpinBox()
        self.spin_layers.setRange(1, 10)
        self.spin_layers.setValue(self.layers)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.3, 5.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(self.base_sigma)

        self.cb_linked_rgb = QCheckBox(self.tr("Linked RGB (apply same params to all channels)"))
        self.cb_linked_rgb.setChecked(True)

        # NEW: Fast ROI preview
        self.cb_fast_roi_preview = QCheckBox(self.tr("Fast ROI preview (compute visible area only)"))
        self.cb_fast_roi_preview.setChecked(True)
        self.cb_fast_roi_preview.setToolTip(
            self.tr("When enabled, preview only computes the currently visible region (with padding for blur).\n"
            "Apply/Send-to-Doc always computes the full image.")
        )
        self.cb_autostretch = QCheckBox(self.tr("Auto-stretch preview"))
        self.cb_autostretch.setChecked(False)
        self.cb_autostretch.setToolTip(
            self.tr("Stretch the preview for visibility. Does not affect Apply or Send to Document — "
            "those always use linear data.")
        )
        form.addRow(self.cb_autostretch)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["μ–σ Thresholding", "Linear"])
        self.combo_mode.setCurrentText("μ–σ Thresholding")
        self.combo_mode.setToolTip(
            self.tr("Multiscale mode:\n"
            "• μ–σ Thresholding: σ-based thresholding + denoise and gain (nonlinear).\n"
            "• Linear: strictly linear multiscale transform; only Gain is applied.")
        )

        self.combo_preview = QComboBox()
        self._refresh_preview_combo()

        form.addRow(self.tr("Layers:"), self.spin_layers)
        form.addRow(self.tr("Base sigma:"), self.spin_sigma)
        form.addRow(self.cb_linked_rgb)
        form.addRow(self.cb_fast_roi_preview)
        form.addRow(self.tr("Mode:"), self.combo_mode)
        form.addRow(self.tr("Layer preview:"), self.combo_preview)

        right.addWidget(gb_global)

        # Layers table
        gb_layers = QGroupBox(self.tr("Layers"))
        v = QVBoxLayout(gb_layers)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [self.tr("On"), self.tr("Layer"), self.tr("Scale"), self.tr("Gain"), self.tr("Thr (σ)"), self.tr("Amt"), self.tr("NR"), self.tr("Type")]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(self.table.SelectionMode.SingleSelection)
        v.addWidget(self.table)
        right.addWidget(gb_layers, stretch=1)

        # Per-layer editor...
        gb_edit = QGroupBox(self.tr("Selected Layer"))
        ef = QFormLayout(gb_edit)
        self.lbl_sel = QLabel(self.tr("Layer: —"))

        # --- Spin boxes ---
        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 10.0)
        self.spin_gain.setSingleStep(0.05)
        self.spin_gain.setValue(1.0)
        self.spin_gain.setToolTip(
            self.tr("Gain: multiplies the detail coefficients on this layer.\n"
            "1.0 = unchanged, >1.0 boosts detail, <1.0 reduces it.")
        )

        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 10.0)       # N·σ
        self.spin_thr.setSingleStep(0.1)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setToolTip(
            self.tr("Threshold (σ): soft threshold level in units of this layer's noise σ.\n"
            "0 = no thresholding; 1–3 ≈ mild to strong suppression of small coefficients.")
        )

        self.spin_amt = QDoubleSpinBox()
        self.spin_amt.setRange(0.0, 1.0)
        self.spin_amt.setSingleStep(0.05)
        self.spin_amt.setToolTip(
            self.tr("Amount: blend factor toward the thresholded version of the layer.\n"
            "0 = ignore thresholding, 1 = fully use the thresholded layer.")
        )

        self.spin_denoise = QDoubleSpinBox()
        self.spin_denoise.setRange(0.0, 1.0)
        self.spin_denoise.setSingleStep(0.05)
        self.spin_denoise.setValue(0.0)
        self.spin_denoise.setToolTip(
            self.tr("Denoise: extra multiscale noise reduction on this layer.\n"
            "0 = off, 1 = strong NR (≈3σ soft threshold blended in).")
        )

        # --- Sliders (int ranges, mapped to spins) ---
        self.slider_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_gain.setRange(0, 1000)         # 0..10.00
        self.slider_gain.setToolTip(self.spin_gain.toolTip())

        self.slider_thr = QSlider(Qt.Orientation.Horizontal)
        self.slider_thr.setRange(0, 1000)         # 0..10.00 σ (×0.01)
        self.slider_thr.setToolTip(self.spin_thr.toolTip())

        self.slider_amt = QSlider(Qt.Orientation.Horizontal)
        self.slider_amt.setRange(0, 100)          # 0..1.00
        self.slider_amt.setToolTip(self.spin_amt.toolTip())

        self.slider_denoise = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise.setRange(0, 100)      # 0..1.00
        self.slider_denoise.setToolTip(self.spin_denoise.toolTip())

        # Layout rows: label -> [slider | spinbox]
        ef.addRow(self.lbl_sel)

        gain_row = QHBoxLayout()
        gain_row.addWidget(self.slider_gain)
        gain_row.addWidget(self.spin_gain)
        ef.addRow(self.tr("Gain:"), gain_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(self.slider_thr)
        thr_row.addWidget(self.spin_thr)
        ef.addRow(self.tr("Threshold (σ):"), thr_row)

        amt_row = QHBoxLayout()
        amt_row.addWidget(self.slider_amt)
        amt_row.addWidget(self.spin_amt)
        ef.addRow(self.tr("Amount:"), amt_row)

        dn_row = QHBoxLayout()
        dn_row.addWidget(self.slider_denoise)
        dn_row.addWidget(self.spin_denoise)
        ef.addRow(self.tr("Denoise:"), dn_row)

        right.addWidget(gb_edit)

        # Buttons...
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply to Document"))
        self.btn_detail_new = QPushButton(self.tr("Send to New Document"))
        self.btn_split_layers = QPushButton(self.tr("Split Layers to Documents"))
        self.btn_reset = QPushButton(self.tr("Reset"))
        self.btn_reset.setToolTip(
            self.tr("Restore layer count, base sigma, mode, Linked RGB and every layer's\n"
            "settings to the defaults. Preview options are left as they are.")
        )
        self.btn_close = QPushButton(self.tr("Close"))

        # ── Drag-to-canvas grip (PI-style "new instance") ─────────────────
        # Captures the live UI via _multiscale_params(). Dropped on empty
        # canvas → a shortcut carrying these settings; dropped on an image →
        # runs the tool headlessly with them.
        self.preset_drag_handle = PresetDragHandle(
            "multiscale_decomp",
            self._multiscale_params,
            icon=QIcon(multiscale_decomp_path),
            tooltip=(
                self.tr("Drag to the canvas to create a Multiscale Decomposition shortcut\n"
                "with these exact settings.\n"
                "Drop directly on an image to apply them headlessly.")
            ),
            parent=self,
        )
        btn_row.addWidget(self.preset_drag_handle)

        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_detail_new)
        btn_row.addWidget(self.btn_split_layers)
        btn_row.addWidget(self.btn_reset)
        btn_row.addWidget(self.btn_close)
        btn_row.addStretch(1)
        btn_row.addWidget(self.preset_drag_handle)
        btn_row.addWidget(make_help_button("multiscale_decomp", self))
        right.addLayout(btn_row)

        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        # ----- Signals -----
        self.spin_layers.valueChanged.connect(self._on_layers_changed)
        self.spin_sigma.valueChanged.connect(self._on_global_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.combo_preview.currentIndexChanged.connect(self._schedule_preview)
        self.cb_fast_roi_preview.toggled.connect(self._schedule_roi_preview)
        self.cb_autostretch.toggled.connect(self._schedule_roi_preview)
        self.table.itemSelectionChanged.connect(self._on_table_select)

        self.spin_gain.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_thr.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_amt.valueChanged.connect(self._on_layer_editor_changed)
        self.spin_denoise.valueChanged.connect(self._on_layer_editor_changed)

        self.slider_gain.valueChanged.connect(self._on_gain_slider_changed)
        self.slider_thr.valueChanged.connect(self._on_thr_slider_changed)
        self.slider_amt.valueChanged.connect(self._on_amt_slider_changed)
        self.slider_denoise.valueChanged.connect(self._on_dn_slider_changed)

        self.btn_apply.clicked.connect(self._commit_to_doc)
        self.btn_detail_new.clicked.connect(self._send_detail_to_new_doc)
        self.btn_split_layers.clicked.connect(self._split_layers_to_docs)
        self.btn_reset.clicked.connect(self._reset_params)
        self.btn_close.clicked.connect(self._on_close_clicked)

        # Connect viewport scroll changes
        self._connect_viewport_signals()

    # ---------- preset (grip / headless) ----------
    def _multiscale_params(self) -> dict:
        """Canonical preset — single source of truth for the drag grip and the
        headless runner. Schema matches _MultiScaleDecompPresetDialog.result_dict()
        plus two live-only extras (mode, residual) that the preset editor doesn't
        expose yet; seed_from_preset() honors them when present.
        """
        layers_cfg = []
        for i in range(self.layers):
            c = self.cfgs[i] if i < len(self.cfgs) else LayerCfg()
            layers_cfg.append({
                "enabled": bool(c.enabled),
                "gain":    float(c.bias_gain),
                "thr":     float(c.thr),
                "amount":  float(c.amount),
                "denoise": float(c.denoise),
            })
        return {
            "layers":     int(self.layers),
            "base_sigma": float(self.base_sigma),
            "linked_rgb": bool(self.cb_linked_rgb.isChecked()),
            "mode":       self.combo_mode.currentText(),
            "residual":   bool(self.residual_enabled),
            "layers_cfg": layers_cfg,
        }

    def seed_from_preset(self, p: dict | None) -> None:
        """Public: load a preset dict (same schema as _multiscale_params) into the
        live controls, so a double-clicked shortcut opens seeded and a headless
        drop applies these exact settings. Signals are blocked during the push;
        the table/editor/caches are refreshed explicitly afterward.
        """
        if not p:
            return

        layers = int(p.get("layers", self.layers))
        layers = max(1, min(10, layers))
        base_sigma = float(p.get("base_sigma", self.base_sigma))

        # Rebuild per-layer configs from the preset (fill missing with defaults).
        cfgs_in = p.get("layers_cfg", []) or []
        new_cfgs: list[LayerCfg] = []
        for i in range(layers):
            d = cfgs_in[i] if i < len(cfgs_in) else {}
            new_cfgs.append(LayerCfg(
                enabled   = bool(d.get("enabled", True)),
                bias_gain = float(d.get("gain",    1.0)),
                thr       = float(d.get("thr",     0.0)),
                amount    = float(d.get("amount",  0.0)),
                denoise   = float(d.get("denoise", 0.0)),
            ))

        self.layers = layers
        self.base_sigma = base_sigma
        self.cfgs = new_cfgs

        if "residual" in p:
            self.residual_enabled = bool(p["residual"])

        seeded_widgets = (self.spin_layers, self.spin_sigma,
                          self.cb_linked_rgb, self.combo_mode)
        for w in seeded_widgets:
            try:
                w.blockSignals(True)
            except Exception:
                pass
        try:
            self.spin_layers.setValue(layers)
            self.spin_sigma.setValue(base_sigma)
            self.cb_linked_rgb.setChecked(
                bool(p.get("linked_rgb", self.cb_linked_rgb.isChecked()))
            )
            mode = p.get("mode")
            if mode:
                idx = self.combo_mode.findText(str(mode))
                if idx >= 0:
                    self.combo_mode.setCurrentIndex(idx)
        finally:
            for w in seeded_widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

        # Rebuild table/combos to the seeded layer count, then refresh the editor.
        self._sync_cfgs_and_ui()
        if self.layers > 0:
            try:
                self.table.selectRow(0)
            except Exception:
                pass
            self._load_layer_into_editor(0)

        # Seeded params invalidate any cached pyramid; the next compute rebuilds it.
        self._invalidate_full_decomp_cache()
        if not self._headless:
            self._schedule_preview()

    def _reset_params(self):
        """Reset: back to the settings the dialog opened with (captured at
        construction, before any preset seeding). Preview options — layer
        preview, fast ROI, autostretch — are display choices and stay put."""
        self.seed_from_preset(self._default_preset)

    def _active_mask_id(self) -> str | None:
        return getattr(self._doc, "active_mask_id", None) or None

    def _active_mask_array(self) -> np.ndarray | None:
        """
        Fetch the active document mask as a 2D float32 array in [0, 1],
        resized to match the current document image shape and
        multiplied by the layer's opacity. Returns None if there is no
        active mask or anything about it is unusable.

        Mirrors Statistical Stretch's _active_mask_array so masks
        behave consistently across the suite — crucially, this handles
        both (H,W,1) and (H,W,3) stored masks (collapsing the latter
        via Rec.709 luma) and respects layer opacity, both of which the
        older _blend_with_mask here missed.
        """
        try:
            doc = self._doc
            mid = getattr(doc, "active_mask_id", None)
            if not mid:
                return None
            layer = getattr(doc, "masks", {}).get(mid)
            if layer is None:
                return None
            m = np.asarray(getattr(layer, "data", None))
            if m is None or m.size == 0:
                return None

            # Collapse to 2D if the mask was stored as (H,W,1) or (H,W,3).
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[..., 0]
            elif m.ndim == 3:
                m = (0.2126 * m[..., 0]
                     + 0.7152 * m[..., 1]
                     + 0.0722 * m[..., 2])

            orig = m
            if orig.dtype.kind in "ui":
                m = orig.astype(np.float32) / float(np.iinfo(orig.dtype).max)
            else:
                m = orig.astype(np.float32, copy=False)
            m = np.clip(m, 0.0, 1.0)

            # Resize to current document image size.
            base_img = getattr(doc, "image", None)
            if base_img is None:
                return None
            th, tw = base_img.shape[:2]
            sh, sw = m.shape[:2]
            if (sh, sw) != (th, tw):
                yi = np.linspace(0, sh - 1, th).astype(np.int32)
                xi = np.linspace(0, sw - 1, tw).astype(np.int32)
                m = m[yi][:, xi]

            opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
            if opacity < 1.0:
                m = m * opacity

            return m
        except Exception:
            return None

    def _blend_with_mask(self, base: np.ndarray, out: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
        """
        Standard SASpro mask blend: base * (1 - m) + out * m.

        Signature matches Statistical Stretch: caller supplies base
        (unprocessed source), out (processed result), and the 2D mask
        in [0, 1]. For 3-channel outputs the mask is broadcast across
        channels. Base and out must share the same shape; the caller
        is responsible for shape reconciliation.
        """
        return _blend_masked(base, out, mask)

    # ---------- Preview plumbing ----------
    def _spinner_on(self):
        if getattr(self, "_closing", False):
            return
        try:
            sp = getattr(self, "busy_spinner", None)
            if sp is None:
                return
            sp.setVisible(True)
            mv = getattr(self, "_busy_movie", None)
            if mv is not None and mv.state() != QMovie.MovieState.Running:
                mv.start()
        except RuntimeError:
            return

    def _spinner_off(self):
        try:
            sp = getattr(self, "busy_spinner", None)
            mv = getattr(self, "_busy_movie", None)
            if mv is not None:
                mv.stop()
            if sp is not None:
                sp.setVisible(False)
        except RuntimeError:
            return


    def _show_busy_overlay(self):
        try:
            self.busy_label.adjustSize()
            self.busy_label.move(12, 12)
            self.busy_label.show()
        except Exception:
            pass

    def _begin_busy(self):
        self._busy_depth += 1
        if self._busy_depth == 1:
            # show only if compute isn't instant
            self._busy_show_timer.start(120)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _end_busy(self):
        self._busy_depth = max(0, self._busy_depth - 1)
        if self._busy_depth == 0:
            self._busy_show_timer.stop()
            self.busy_label.hide()
            QApplication.restoreOverrideCursor()


    def _on_mode_changed(self, idx: int):
        # Re-enable/disable controls as needed
        self._update_param_widgets_for_mode()
        self._schedule_preview()

    def _schedule_preview(self):
        if getattr(self, "_closing", False):
            return
        self._preview_timer.start(60)

    def _schedule_roi_preview(self):
        if getattr(self, "_closing", False):
            return
        self._preview_timer.start(60)

    def _connect_viewport_signals(self):
        """
        Any pan/scroll should schedule ROI preview recompute.
        """
        try:
            self.view.horizontalScrollBar().valueChanged.connect(self._schedule_roi_preview)
            self.view.verticalScrollBar().valueChanged.connect(self._schedule_roi_preview)
        except Exception:
            pass

    def _decomp_key(self):
        return (int(self.spin_layers.value()), float(self.spin_sigma.value()), id(self._image))

    def _decomp_ready(self) -> bool:
        return (self._cached_layers is not None and self._cached_residual is not None
                and self._cached_key == self._decomp_key())

    def _store_decomp(self, key, details, residual, noise):
        self._cached_layers = details
        self._cached_residual = residual
        self._layer_noise = noise
        self._cached_key = key
        # Never consumed: the shrink path that read it could not be reached
        # (a layer change always rebuilds), and summing it cost ~1.5 s.
        self._cached_coarse = None

    def _recompute_decomp(self, force: bool = False):
        """Whole-image pyramid on the calling thread (headless Apply). The
        interactive dialog builds it off the GUI thread: _request_decomp()."""
        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())
        self.layers = layers
        self.base_sigma = base_sigma
        if force or not self._decomp_ready():
            details, residual, noise = _full_decomposition(self._image_work, layers, base_sigma)
            self._store_decomp(self._decomp_key(), details, residual, noise)
        self._sync_cfgs_and_ui()

    # ---------- background work ----------
    def _start_job(self, attr: str, fn, on_result, on_failed, key=None):
        job = _MSDJob(fn, key)
        job.result.connect(on_result)
        job.failed.connect(on_failed)
        setattr(self, attr, job)
        start_owned_thread(job)
        self._update_busy()
        return job

    def _update_busy(self):
        if getattr(self, "_closing", False):
            return
        jobs = (self._base_job, self._decomp_job, self._preview_job, self._action_job)
        if self._full_wait or any(j is not None for j in jobs):
            self._spinner_on()
        else:
            self._spinner_off()
        # The overlay only while the view itself is waiting (no image yet, or a
        # full-frame preview); ROI previews and the background pyramid just spin.
        if self._base_job is not None or self._full_wait:
            if not self._busy_show_timer.isActive() and not self.busy_label.isVisible():
                self._busy_show_timer.start(120)
        else:
            self._busy_show_timer.stop()
            self.busy_label.hide()

    def _stop_jobs(self):
        """Closing: stop listening to background work and cancel what can be.
        A running job ends on its own thread; thread_keeper holds it until then."""
        self._closing = True
        for name in ("_prefetch_timer", "_preview_timer", "_busy_show_timer"):
            tm = getattr(self, name, None)
            if tm is not None:
                try:
                    tm.stop()
                except RuntimeError:
                    pass
        for attr in ("_base_job", "_decomp_job", "_preview_job", "_action_job"):
            job = getattr(self, attr, None)
            setattr(self, attr, None)
            if job is None:
                continue
            for sig in (job.result, job.failed):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
            job.cancel()
        self._decomp_waiters = []
        self._action_kind = None
        self._close_action_progress()
        self._spinner_off()

    def _start_base_job(self):
        self._start_job("_base_job", partial(_base_qimage, self._image),
                        self._on_base_result, self._on_base_failed)

    def _on_base_result(self, qimg):
        self._base_job = None
        if self._closing:
            return
        if qimg is not None:
            self._show_base(QPixmap.fromImage(qimg))
        self._update_busy()

    def _on_base_failed(self, msg: str):
        self._base_job = None
        if self._closing:
            return
        print(f"[MSD] base image failed off-thread ({msg}); building it here", flush=True)
        try:
            self._show_base(self._np_to_qpix(self._image))
        except Exception:
            pass
        self._update_busy()

    def _show_base(self, pm: QPixmap):
        # With the opening settings (every layer on at gain 1, no threshold,
        # residual on) the reconstruction is the source, so the source is the
        # opening base image. A full-frame preview that landed first is newer.
        if self.pix_base.pixmap().isNull():
            self.pix_base.setPixmap(pm)
            self.pix_base.setOffset(0, 0)
        if self._pm_before_key is None:
            # Opening "before": the source without autostretch — this image.
            self._pm_before_key = False
            self._cmp.set_before(pm)
        H, W = self._image.shape[:2]
        self._cmp.set_image_size(W, H)
        self._fit_view()            # also schedules the first ROI preview

    def _request_decomp(self, then=None):
        """Have the whole-image pyramid for the current layers/sigma, building
        it off the GUI thread. then() runs on the GUI thread once it is there,
        or once building it failed (then() checks _decomp_ready())."""
        if self._closing:
            return
        if self._decomp_ready():
            if then is not None:
                then()
            return
        if then is not None and then not in self._decomp_waiters:
            self._decomp_waiters.append(then)
        key = self._decomp_key()
        job = self._decomp_job
        if job is not None:
            if job.key != key:
                job.cancel()        # superseded; its end starts the current one
            return
        self._start_job("_decomp_job",
                        partial(_full_decomposition, self._image_work, key[0], key[1]),
                        self._on_decomp_result, self._on_decomp_failed, key=key)

    def _prefetch_decomp(self):
        """Build the pyramid in the background, so Apply finds it ready."""
        if self._closing or self._headless:
            return
        if self._decomp_failed_key == self._decomp_key():
            return
        self._request_decomp(None)

    def _on_decomp_result(self, payload):
        job, self._decomp_job = self._decomp_job, None
        if self._closing:
            return
        if payload is not None and job is not None and job.key == self._decomp_key():
            details, residual, noise = payload
            self._store_decomp(job.key, details, residual, noise)
        self._after_decomp(failed=False)

    def _on_decomp_failed(self, msg: str):
        job, self._decomp_job = self._decomp_job, None
        if self._closing:
            return
        print(f"[MSD] decomposition failed: {msg}", flush=True)
        if job is not None:
            self._decomp_failed_key = job.key
        self._after_decomp(failed=True)

    def _after_decomp(self, failed: bool):
        if self._decomp_ready() or failed:
            waiters, self._decomp_waiters = self._decomp_waiters, []
            for fn in waiters:
                if not self._closing:
                    fn()
        elif self._decomp_waiters:
            self._request_decomp(None)      # settings moved on: build the current one
        else:
            self._prefetch_timer.start()
        self._update_busy()

    def _sync_cfgs_and_ui(self):
        # ensure cfg list matches layer count (your existing logic, just moved)
        if len(self.cfgs) != self.layers:
            old = self.cfgs[:]
            self.cfgs = [LayerCfg() for _ in range(self.layers)]
            for i in range(min(len(old), self.layers)):
                self.cfgs[i] = old[i]

        self._rebuild_table()
        self._refresh_preview_combo()

    def _build_tuned_layers(self):
        self._recompute_decomp(force=False)

        details = self._cached_layers
        residual = self._cached_residual
        if details is None or residual is None:
            return None, None

        mode = self.combo_mode.currentText()

        def do_one(i_w):
            i, w = i_w
            cfg = self.cfgs[i]
            if not cfg.enabled:
                return i, np.zeros_like(w)

            layer_sigma = self.base_sigma * (2 ** i)

            sigma = self._layer_noise[i] if self._layer_noise and i < len(self._layer_noise) else None
            out = apply_layer_ops(
                w,
                cfg.bias_gain,
                cfg.thr,
                cfg.amount,
                cfg.denoise,
                sigma,
                layer_index=i,
                mode=mode,
            )
            return i, out

        n = len(details)
        if n == 0:
            return [], residual

        max_workers = min(os.cpu_count() or 4, n)

        tuned = [None] * n
        # ThreadPoolExecutor is fine here because apply_layer_ops is numpy-heavy
        # (but real speed-up depends on GIL/OpenCV/BLAS behavior).
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, out in ex.map(do_one, enumerate(details)):
                tuned[i] = out

        return tuned, residual

    def _rebuild_preview(self):
        """Preview-timer slot: compute the preview off the GUI thread. A change
        that arrives while one is running queues exactly one re-run with the
        settings current at that point."""
        if getattr(self, "_closing", False):
            return
        if self._base_job is not None:
            return                  # the base image's arrival fits and previews
        if self._preview_job is not None:
            self._preview_again = True
            return
        s = self._preview_snapshot()
        if s is None:
            self._full_wait = False
            self._update_busy()
            return
        if s["kind"] == "full":
            if not self._decomp_ready():
                if self._decomp_failed_key == self._decomp_key():
                    self._full_wait = False
                    self._update_busy()
                    return
                self._full_wait = True
                self._request_decomp(self._rebuild_preview)
                self._update_busy()
                return
            s["details"] = self._cached_layers
            s["residual"] = self._cached_residual
            s["noise"] = self._layer_noise
        self._full_wait = (s["kind"] == "full")
        self._start_job("_preview_job", partial(_preview_job, s),
                        self._on_preview_result, self._on_preview_failed)

    def _preview_snapshot(self, kind: str | None = None):
        """Everything a preview reads from the widgets, read here on the GUI thread."""
        if kind is None:
            roi_ok = (
                getattr(self, "cb_fast_roi_preview", None) is not None
                and self.cb_fast_roi_preview.isChecked()
                and not self.pix_base.pixmap().isNull()
            )
            kind = "roi" if roi_ok else "full"
        autostretch = bool(self.cb_autostretch.isChecked())
        s = {
            "kind": kind,
            "work": getattr(self, '_image_work', self._image),
            "image": self._image,
            "image_shape": self._image.shape,
            "layers": int(self.spin_layers.value()),
            "base_sigma": float(self.spin_sigma.value()),
            "cfgs": [replace(c) for c in self.cfgs],
            "mode": self.combo_mode.currentText(),
            "residual_enabled": bool(self.residual_enabled),
            "sel": self.combo_preview.currentData(),
            "autostretch": autostretch,
            "stretch": self._disp_stretch,
            "need_before": self._pm_before_key != autostretch,
        }
        if kind == "roi":
            s["vis"] = self._visible_image_rect()
            if s["vis"] is None:
                return None
        return s

    def _on_preview_result(self, out):
        self._preview_job = None
        if self._closing:
            return
        if out is not None:
            if out.get("stretch") is not None and self._disp_stretch is None:
                self._disp_stretch = out["stretch"]
            if out.get("before") is not None:
                self._pm_before_key = out["before_key"]
                self._cmp.set_before(QPixmap.fromImage(out["before"]))
            qimg = out.get("qimage")
            if qimg is not None:
                pm = QPixmap.fromImage(qimg)
                if out["kind"] == "roi":
                    x0, y0 = out["rect"][:2]
                    self.pix_roi.setPixmap(pm)
                    self.pix_roi.setOffset(x0, y0)
                else:
                    self.pix_base.setPixmap(pm)
                    self.pix_base.setOffset(0, 0)
                    # Clear the ROI overlay on a full refresh
                    self.pix_roi.setPixmap(QPixmap())
                    self.pix_roi.setOffset(0, 0)
                # Keep scene bounds as full image, not ROI
                H, W = self._image.shape[:2]
                self._cmp.set_image_size(W, H)
        self._preview_done()

    def _on_preview_failed(self, msg: str):
        self._preview_job = None
        if self._closing:
            return
        print(f"[MSD] preview failed: {msg}", flush=True)
        self._preview_done()

    def _preview_done(self):
        self._full_wait = False
        if self._preview_again:
            self._preview_again = False
            self._rebuild_preview()
        self._update_busy()

    def _compute_preview_roi(self):
        """Fast ROI preview of the visible area on the calling thread: (float
        RGB crop, rect). The dialog runs the same computation off the GUI
        thread (_rebuild_preview)."""
        s = self._preview_snapshot(kind="roi")
        if s is None:
            return None, None
        roi, rect = _roi_preview_image(s)
        if s.get("stretch_new") and self._disp_stretch is None:
            self._disp_stretch = s["stretch"]
        return roi, rect


    def _update_param_widgets_for_mode(self):
        linear = (self.combo_mode.currentText() == "Linear")

        # Always allow Gain in both modes
        gain_widgets = (self.spin_gain, self.slider_gain)

        # These are only meaningful in Mean mode
        nonlin_widgets = (
            self.spin_thr, self.slider_thr,
            self.spin_amt, self.slider_amt,
            self.spin_denoise, self.slider_denoise,
        )

        # For residual row we already disable everything in _load_layer_into_editor,
        # so here we just respect the current selection.
        idx = getattr(self, "_selected_layer", None)
        if idx is None or idx == self.layers:
            # Residual – handled in _load_layer_into_editor
            return

        for w in gain_widgets:
            w.setEnabled(True)

        for w in nonlin_widgets:
            w.setEnabled(not linear)

    def _display_stretch(self):
        """Display stretch measured once, on the source image.

        Shared by the before, the full-frame after and every ROI crop.
        Stretching each on its own statistics lets the brightness drift with
        the result (and an ROI crop's statistics differ from the full
        frame's), so a before/after comparison would show differences that
        are not in the data. Same curve family as the main view's
        autostretch: 3-sigma black point, zeros (no-data) ignored, unlinked,
        target 0.25.
        """
        if self._disp_stretch is None:
            self._disp_stretch = display_stretch_params(
                self._image_work, target_median=0.25, linked=False,
                blackpoint_sigma=3.0, ignore_zeros=True)
        return self._disp_stretch

    def _apply_preview_stretch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply autostretch for display only if the toggle is on.
        img is float32 [0..1], either (H,W) 2D or (H,W,3) 3ch.
        Always returns (H,W,3) float32 suitable for QPixmap.
        """
        # Ensure 3ch for display
        if img.ndim == 2:
            img3 = np.repeat(img[:, :, None], 3, axis=2)
        else:
            img3 = img

        if not self.cb_autostretch.isChecked():
            return img3

        try:
            return apply_display_stretch(img3, self._display_stretch())
        except Exception:
            return img3

    def _sync_before_pixmap(self):
        """Give the comparison its "before": the source image through the same
        display stretch as the preview. Rebuilt only when that stretch changes."""
        key = bool(self.cb_autostretch.isChecked())
        if self._pm_before_key == key or self._image is None:
            return
        self._pm_before_key = key
        self._cmp.set_before(self._np_to_qpix(self._apply_preview_stretch(self._image)))

    def _np_to_qpix(self, img: np.ndarray) -> QPixmap:
        return QPixmap.fromImage(_rgb01_to_qimage(img))

    def _refresh_pix(self):
        pm = self._np_to_qpix(self._preview_img)
        self.pix_base.setPixmap(pm)
        self.pix_base.setOffset(0, 0)

        # Optional: clear ROI overlay on full refresh
        self.pix_roi.setPixmap(QPixmap())
        self.pix_roi.setOffset(0, 0)

        H, W = self._image.shape[:2]
        self._cmp.set_image_size(W, H)
        self._sync_before_pixmap()

    def _fast_preview_enabled(self) -> bool:
        return bool(getattr(self, "cb_fast_roi_preview", None)) and self.cb_fast_roi_preview.isChecked()

    def _invalidate_full_decomp_cache(self):
        if self._cached_layers is not None and self._cached_key == self._decomp_key():
            return              # same layers/sigma: the pyramid still holds
        self._cached_layers = None
        self._cached_coarse = None
        self._cached_residual = None
        self._cached_key = None
        self._layer_noise = None
        self._decomp_failed_key = None
        job = self._decomp_job
        if job is not None and job.key != self._decomp_key():
            job.cancel()
        if not self._headless and not getattr(self, "_closing", False):
            self._prefetch_timer.start()


    def _fit_view(self):
        if self.pix_base.pixmap().isNull():
            return
        self.view.resetTransform()
        # Scene rect, not pix_base: in side by side it spans both halves.
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._schedule_roi_preview()

    def _one_to_one(self):
        self.view.resetTransform()
        self._schedule_roi_preview()

    # ---------- Table / layer editing ----------
    def _on_gain_slider_changed(self, v: int):
        # 0..300 -> 0.00..3.00
        val = v / 100.0
        self.spin_gain.blockSignals(True)
        self.spin_gain.setValue(val)
        self.spin_gain.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_thr_slider_changed(self, v: int):
        # 0..1000 -> 0.00..10.00 σ
        val = v / 100.0
        self.spin_thr.blockSignals(True)
        self.spin_thr.setValue(val)
        self.spin_thr.blockSignals(False)
        self._on_layer_editor_changed()


    def _on_amt_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_amt.blockSignals(True)
        self.spin_amt.setValue(val)
        self.spin_amt.blockSignals(False)
        self._on_layer_editor_changed()

    def _on_dn_slider_changed(self, v: int):
        # 0..100 -> 0.00..1.00
        val = v / 100.0
        self.spin_denoise.blockSignals(True)
        self.spin_denoise.setValue(val)
        self.spin_denoise.blockSignals(False)
        self._on_layer_editor_changed()


    def _rebuild_table(self):
        self.table.blockSignals(True)
        try:
            # +1 row for residual ("R")
            self.table.setRowCount(self.layers + 1)

            # detail rows
            for i in range(self.layers):
                cfg = self.cfgs[i]

                item_on = QTableWidgetItem("")
                item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item_on.setCheckState(Qt.CheckState.Checked if cfg.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(i, 0, item_on)

                self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
                self.table.setItem(i, 2, QTableWidgetItem(f"{self.base_sigma * (2**i):.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{cfg.bias_gain:.2f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{cfg.thr:.2f}"))   # N·σ
                self.table.setItem(i, 5, QTableWidgetItem(f"{cfg.amount:.2f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{cfg.denoise:.2f}"))

                self.table.setItem(i, 7, QTableWidgetItem("D"))

            # residual row
            r = self.layers
            item_on = QTableWidgetItem("")
            item_on.setFlags(item_on.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item_on.setCheckState(
                Qt.CheckState.Checked if self.residual_enabled else Qt.CheckState.Unchecked
            )
            self.table.setItem(r, 0, item_on)

            self.table.setItem(r, 1, QTableWidgetItem("R"))
            self.table.setItem(r, 2, QTableWidgetItem("—"))
            self.table.setItem(r, 3, QTableWidgetItem("1.00"))
            self.table.setItem(r, 4, QTableWidgetItem("0.0000"))
            self.table.setItem(r, 5, QTableWidgetItem("0.00"))
            self.table.setItem(r, 6, QTableWidgetItem("0.00"))
            self.table.setItem(r, 7, QTableWidgetItem("R"))

        finally:
            self.table.blockSignals(False)

        # connect once (avoid stacking connects)
        try:
            self.table.itemChanged.disconnect(self._on_table_item_changed)
        except Exception:
            pass
        self.table.itemChanged.connect(self._on_table_item_changed)

        if self.layers > 0 and not self.table.selectedItems():
            self.table.selectRow(0)
            self._load_layer_into_editor(0)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        r, c = item.row(), item.column()

        # Residual row
        if r == self.layers:
            if c == 0:
                self.residual_enabled = (item.checkState() == Qt.CheckState.Checked)
                self._schedule_preview()
            # ignore other edits for residual
            return

        if not (0 <= r < len(self.cfgs)):
            return

        cfg = self.cfgs[r]

        if c == 0:
            # On/off
            cfg.enabled = (item.checkState() == Qt.CheckState.Checked)
            self._schedule_preview()
            return

        # numeric columns: Gain(3), Thr(4), Amt(5), NR(6)
        try:
            text = item.text().strip()
            val = float(text) if text else 0.0
        except Exception:
            return

        if c == 3:
            cfg.bias_gain = val
        elif c == 4:
            cfg.thr = val
        elif c == 5:
            cfg.amount = val
        elif c == 6:
            cfg.denoise = val
        else:
            return

        # If this row is currently selected, update editor widgets too
        if getattr(self, "_selected_layer", None) == r:
            self._load_layer_into_editor(r)

        self._schedule_preview()

    @contextmanager
    def _busy_popup(self, text: str):
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setWindowTitle(self.tr("Multiscale Decomposition"))
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()

        self._spinner_on()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            yield dlg
        finally:
            try:
                dlg.close()
            except Exception:
                pass
            QApplication.restoreOverrideCursor()
            self._spinner_off()
            QApplication.processEvents()

    def _on_table_select(self):
        rows = {it.row() for it in self.table.selectedItems()}
        if not rows:
            return
        r = min(rows)
        self._load_layer_into_editor(r)

    def _load_layer_into_editor(self, idx: int):
        self._selected_layer = idx

        if idx == self.layers:
            self.lbl_sel.setText(self.tr("Layer: R (Residual)"))
            for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                    self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
                w.setEnabled(False)
            return

        for w in (self.spin_gain, self.spin_thr, self.spin_amt, self.spin_denoise,
                self.slider_gain, self.slider_thr, self.slider_amt, self.slider_denoise):
            w.setEnabled(True)

        cfg = self.cfgs[idx]
        self.lbl_sel.setText(self.tr("Layer: {0} / {1}").format(idx+1, self.layers))

        # spins + sliders in sync
        self.spin_gain.blockSignals(True)
        self.spin_thr.blockSignals(True)
        self.spin_amt.blockSignals(True)
        self.spin_denoise.blockSignals(True)

        self.slider_gain.blockSignals(True)
        self.slider_thr.blockSignals(True)
        self.slider_amt.blockSignals(True)
        self.slider_denoise.blockSignals(True)
        try:
            self.spin_gain.setValue(cfg.bias_gain)
            self.spin_thr.setValue(cfg.thr)           # thr is N·σ now
            self.spin_amt.setValue(cfg.amount)
            self.spin_denoise.setValue(cfg.denoise)

            self.slider_gain.setValue(int(round(cfg.bias_gain * 100.0)))
            self.slider_thr.setValue(int(round(cfg.thr * 100.0)))   # N·σ → 0..1000
            self.slider_amt.setValue(int(round(cfg.amount * 100.0)))
            self.slider_denoise.setValue(int(round(cfg.denoise * 100.0)))
        finally:
            self.spin_gain.blockSignals(False)
            self.spin_thr.blockSignals(False)
            self.spin_amt.blockSignals(False)
            self.spin_denoise.blockSignals(False)
            self.slider_gain.blockSignals(False)
            self.slider_thr.blockSignals(False)
            self.slider_amt.blockSignals(False)
            self.slider_denoise.blockSignals(False)
            self._update_param_widgets_for_mode()



    def _on_layer_editor_changed(self):
        idx = getattr(self, "_selected_layer", None)
        if idx is None or not (0 <= idx < len(self.cfgs)):
            return
        cfg = self.cfgs[idx]
        cfg.bias_gain = float(self.spin_gain.value())
        cfg.thr = float(self.spin_thr.value())
        cfg.amount = float(self.spin_amt.value())
        cfg.denoise = float(self.spin_denoise.value())

        # keep table in sync
        self.table.blockSignals(True)
        try:
            self.table.item(idx, 3).setText(f"{cfg.bias_gain:.2f}")
            self.table.item(idx, 4).setText(f"{cfg.thr:.2f}")      # N·σ
            self.table.item(idx, 5).setText(f"{cfg.amount:.2f}")
            self.table.item(idx, 6).setText(f"{cfg.denoise:.2f}")

        finally:
            self.table.blockSignals(False)

        self._schedule_preview()

    def _on_layers_changed(self):
        # Always update counts/UI
        self.layers = int(self.spin_layers.value())

        # Ensure cfgs length matches new layer count and table/combos update
        self._sync_cfgs_and_ui()

        # The whole-image pyramid for the new count is built off the GUI thread:
        # by the full-frame preview when Fast ROI is off, else in the background
        # for Apply. The ROI preview computes its own.
        self._invalidate_full_decomp_cache()
        self._schedule_preview()


    def _on_global_changed(self):
        self.base_sigma = float(self.spin_sigma.value())

        # Update table scale column text (it uses self.base_sigma)
        self._sync_cfgs_and_ui()

        self._invalidate_full_decomp_cache()
        self._schedule_preview()

    def _refresh_preview_combo(self):
        self.combo_preview.blockSignals(True)
        try:
            self.combo_preview.clear()
            self.combo_preview.addItem(self.tr("Final"), userData="final")
            self.combo_preview.addItem(self.tr("R (Residual)"), userData="residual")
            for i in range(self.layers):
                self.combo_preview.addItem(self.tr("Detail Layer {0}").format(i+1), userData=i)
        finally:
            self.combo_preview.blockSignals(False)

    def _visible_image_rect(self) -> tuple[int, int, int, int] | None:
        # Use full image rect, NOT the pixmap bounds
        H, W = self._image.shape[:2]
        full_item_rect_scene = QRectF(0, 0, W, H)

        vr = self.view.viewport().rect()
        tl = self.view.mapToScene(vr.topLeft())
        br = self.view.mapToScene(vr.bottomRight())
        scene_rect = QRectF(tl, br).normalized()

        inter = scene_rect.intersected(full_item_rect_scene)
        if inter.isEmpty():
            return None

        x0 = int(np.floor(inter.left()))
        y0 = int(np.floor(inter.top()))
        x1 = int(np.ceil(inter.right()))
        y1 = int(np.ceil(inter.bottom()))

        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))

        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _np_to_qpix_roi_comp(self, img_rgb01: np.ndarray) -> QPixmap:
        """
        img_rgb01 is float32 RGB [0..1]
        """
        return QPixmap.fromImage(_rgb01_to_qimage(img_rgb01))

    def _refresh_pix_roi(self, roi_img01: np.ndarray, roi_rect: tuple[int,int,int,int]):
        x0, y0, x1, y1 = roi_rect
        pm = self._np_to_qpix_roi_comp(roi_img01)

        self.pix_roi.setPixmap(pm)
        self.pix_roi.setOffset(x0, y0)

        # Keep scene bounds as full image, not ROI
        H, W = self._image.shape[:2]
        self._cmp.set_image_size(W, H)
        self._sync_before_pixmap()


    def _build_preview_roi(self):
        vis = self._visible_image_rect()
        if vis is None:
            return None

        x0,y0,x1,y1 = vis
        layers = int(self.spin_layers.value())
        base_sigma = float(self.spin_sigma.value())

        if layers <= 0:
            return None

        sigma_max = base_sigma * (2 ** (layers - 1))
        pad = int(np.ceil(3.0 * sigma_max)) + 2

        H, W = self._image.shape[:2]
        px0 = max(0, x0 - pad); py0 = max(0, y0 - pad)
        px1 = min(W, x1 + pad); py1 = min(H, y1 + pad)

        crop = self._image[py0:py1, px0:px1].astype(np.float32, copy=False)

        # Decompose crop
        details, residual = multiscale_decompose(crop, layers=layers, base_sigma=base_sigma)

        # noise per layer (crop-based) — good enough for preview
        layer_noise = [_robust_sigma(w) if w.size else 1e-6 for w in details]

        # Apply tuning per layer (can thread this like we discussed)
        mode = self.combo_mode.currentText()
        tuned = []
        for i,w in enumerate(details):
            cfg = self.cfgs[i]
            if not cfg.enabled:
                tuned.append(np.zeros_like(w))
            else:
                tuned.append(apply_layer_ops(w, cfg.bias_gain, cfg.thr, cfg.amount, cfg.denoise,
                                            layer_noise[i], mode=mode))

        res = residual if self.residual_enabled else np.zeros_like(residual)
        out_raw = multiscale_reconstruct(tuned, res)

        # Match your preview rules
        if not self.residual_enabled:
            out = detail_only_vis(out_raw, tuned)
        else:
            out = np.clip(out_raw, 0.0, 1.0).astype(np.float32, copy=False)

        # Crop back from padded-crop coords to visible ROI coords
        cx0 = x0 - px0; cy0 = y0 - py0
        cx1 = cx0 + (x1 - x0); cy1 = cy0 + (y1 - y0)
        return out[cy0:cy1, cx0:cx1], (x0,y0,x1,y1)


    # ---------- Apply to doc ----------
    def _commit_to_doc(self):
        if not self._headless:
            try: self._save_window_state()
            except Exception: pass
            self._start_action("apply", self.tr("Applying multiscale result to document…"))
            return
        # Headless preset apply: the caller disposes of the dialog right after,
        # so the work happens here.
        with self._busy_popup(self.tr("Applying multiscale result to document…")):
            self._recompute_decomp(force=False)
            if self._cached_layers is None or self._cached_residual is None:
                return
            self._finish_commit(_action_compute(self._action_snapshot("apply")))

    def _finish_commit(self, out_final: np.ndarray):
        mid = self._active_mask_id()
        meta = {
            "step_name": "Multiscale Decomposition",
            "command_id": "multiscale_decomp",
            "preset": dict(self._multiscale_params()),
            "masked": bool(mid),
            "mask_id": mid,
            "mask_blend": "m*out+(1-m)*src",
        }

        try:
            if hasattr(self._doc, "apply_edit"):
                self._doc.apply_edit(out_final, meta, step_name="Multiscale Decomposition")
            elif hasattr(self._doc, "set_image"):
                self._doc.set_image(out_final, step_name="Multiscale Decomposition")
            else:
                self._doc.image = out_final
        except Exception as e:
            QMessageBox.critical(self, self.tr("Multiscale Decomposition"), self.tr("Failed to write to document:\n{0}").format(e))
            return

        if hasattr(self.parent(), "_refresh_active_view"):
            try:
                self.parent()._refresh_active_view()
            except Exception:
                pass

        # 🔁 Record as last headless-style command for Replay (best-effort;
        # supports both spellings used across the suite).
        try:
            mw = self.parent()
            preset = self._multiscale_params()
            for attr in ("_remember_last_headless_command",
                         "remember_last_headless_command"):
                fn = getattr(mw, attr, None)
                if callable(fn):
                    try:
                        fn("multiscale_decomp", preset,
                           description="Multiscale Decomposition")
                    except TypeError:
                        fn(command_id="multiscale_decomp", preset=preset,
                           description="Multiscale Decomposition")
                    break
        except Exception:
            pass

        self.accept()

    def _action_snapshot(self, kind: str) -> dict:
        """What Apply / Send / Split read from the dialog, read on the GUI thread.
        The active mask is the one on the SOURCE document (Send and Split honour
        it too); its base is the image captured at dialog open."""
        return {
            "kind": kind,
            "details": self._cached_layers,
            "residual": self._cached_residual,
            "noise": self._layer_noise,
            "cfgs": [replace(c) for c in self.cfgs],
            "mode": self.combo_mode.currentText(),
            "residual_enabled": bool(self.residual_enabled),
            "orig_mono": self._orig_mono,
            "orig_shape": self._orig_shape,
            "mask": self._active_mask_array(),
            "orig_for_mask": self._orig_for_mask,
        }

    def _start_action(self, kind: str, text: str):
        """Apply / Send / Split: the numbers are computed off the GUI thread
        behind a busy dialog that keeps animating; documents are written here."""
        if getattr(self, "_closing", False) or self._action_kind is not None:
            return
        if kind in ("send", "split") and self._get_doc_manager() is None:
            msg = (self.tr("No DocManager available to create a new document.") if kind == "send"
                   else self.tr("No DocManager available to create new documents."))
            QMessageBox.warning(self, self.tr("Multiscale Decomposition"), msg)
            return
        self._action_kind = kind
        self._open_action_progress(text)
        self._request_decomp(self._start_action_job)

    def _open_action_progress(self, text: str):
        dlg = QProgressDialog(text, "", 0, 0, self)
        dlg.setWindowTitle(self.tr("Multiscale Decomposition"))
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        self._action_prog = dlg
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._action_cursor = True

    def _close_action_progress(self):
        dlg, self._action_prog = getattr(self, "_action_prog", None), None
        if dlg is not None:
            try:
                dlg.close()
                dlg.deleteLater()
            except RuntimeError:
                pass
        if getattr(self, "_action_cursor", False):
            self._action_cursor = False
            QApplication.restoreOverrideCursor()

    def _end_action(self):
        self._action_kind = None
        self._close_action_progress()
        self._update_busy()

    def _start_action_job(self):
        kind = self._action_kind
        if kind is None or getattr(self, "_closing", False):
            return
        if not self._decomp_ready():
            if self._decomp_failed_key == self._decomp_key():
                self._end_action()
                QMessageBox.critical(self, self.tr("Multiscale Decomposition"),
                                     self.tr("Could not compute the multiscale decomposition."))
            else:
                self._request_decomp(self._start_action_job)
            return
        self._start_job("_action_job", partial(_action_compute, self._action_snapshot(kind)),
                        self._on_action_result, self._on_action_failed)

    def _on_action_result(self, out):
        self._action_job = None
        kind = self._action_kind
        if self._closing or kind is None:
            return
        try:
            if out is not None:
                if kind == "apply":
                    self._finish_commit(out)
                elif kind == "send":
                    self._finish_send(out)
                elif kind == "split":
                    self._finish_split(out)
        finally:
            self._end_action()

    def _on_action_failed(self, msg: str):
        self._action_job = None
        if self._closing:
            return
        self._end_action()
        QMessageBox.critical(self, self.tr("Multiscale Decomposition"),
                             self.tr("Failed to compute the multiscale result:\n{0}").format(msg))

    def _send_detail_to_new_doc(self):
        self._start_action("send", self.tr("Creating new document from multiscale result…"))

    def _finish_send(self, out_final: np.ndarray):
        dm = self._get_doc_manager()
        if dm is None:
            QMessageBox.warning(self, self.tr("Multiscale Decomposition"),
                                self.tr("No DocManager available to create a new document."))
            return
        title = "Multiscale Result"
        meta = self._build_new_doc_metadata(title, out_final)
        try:
            dm.create_document(out_final, metadata=meta, name=title)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Multiscale Decomposition"),
                                 self.tr("Failed to create new document:\n{0}").format(e))

    def _split_layers_to_docs(self):
        """
        Create a new document for each tuned detail layer *and* the residual.

        - Detail layers use the same mid-gray visualization as the per-layer preview:
              vis = 0.5 + layer*4.0
        - Residual layer is just the residual itself (0..1 clipped).
        """
        self._start_action("split", self.tr("Splitting layers into documents…"))

    def _finish_split(self, result):
        entries, (r_title, res_final) = result
        dm = self._get_doc_manager()
        if dm is None:
            QMessageBox.warning(
                self,
                self.tr("Multiscale Decomposition"),
                self.tr("No DocManager available to create new documents.")
            )
            return

        # ---- 1) Detail layers ------------------------------------------
        for i, (title, out_final) in enumerate(entries):
            meta = self._build_new_doc_metadata(title, out_final)
            try:
                dm.create_document(out_final, metadata=meta, name=title)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    self.tr("Multiscale Decomposition"),
                    self.tr("Failed to create document for layer {0}:\n{1}").format(i+1, e)
                )
                # right now we stop on first hard failure.
                return

        # ---- 2) Residual layer -----------------------------------------
        try:
            r_meta = self._build_new_doc_metadata(r_title, res_final)
            dm.create_document(res_final, metadata=r_meta, name=r_title)
        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("Multiscale Decomposition"),
                self.tr("Failed to create residual-layer document:\n{0}").format(e)
            )



    def _get_doc_manager(self):
        """
        Best-effort: find the DocManager that owns the source document.
        Prefer the doc's own _doc_manager; fall back to parent.doc_manager.
        """
        doc = getattr(self, "_doc", None)
        dm = getattr(doc, "_doc_manager", None) if doc is not None else None

        if dm is None:
            parent = self.parent()
            dm = getattr(parent, "doc_manager", None) if parent is not None else None

        return dm

    def _build_new_doc_metadata(self, title: str, img: np.ndarray) -> dict:
        """
        Clone the source document's metadata and sanitize it for a brand-new doc.
        """
        base_doc = getattr(self, "_doc", None)
        base_meta = getattr(base_doc, "metadata", {}) or {}
        meta = dict(base_meta)

        # New display name
        if title:
            meta["display_name"] = title

        # Drop things that make it look linked/preview/ROI
        imi = dict(meta.get("image_meta") or {})
        for k in ("readonly", "view_kind", "derived_from", "layer", "layer_index", "linked"):
            imi.pop(k, None)
        meta["image_meta"] = imi

        # Remove any ROI-ish keys
        for k in list(meta.keys()):
            if k.startswith("_roi_") or k.endswith("_roi") or k == "roi":
                meta.pop(k, None)

        # For a brand-new doc, don't keep the original file_path
        meta.pop("file_path", None)

        # Normalize mono flag
        if isinstance(img, np.ndarray):
            meta["is_mono"] = (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))

        # Keep bit depth / headers / WCS as-is; DocManager.open_array() will
        # ensure bit_depth etc. are sane.
        return meta

    def _restore_window_state(self):
        try:
            s = QSettings()
            g = s.value("msdecomp/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)

            # splitter
            sp = s.value("msdecomp/splitter_state", None)
            if sp is not None and hasattr(self, "splitter"):
                self.splitter.restoreState(sp)

            # table header (column widths/order)
            hdr = self.table.horizontalHeader() if hasattr(self, "table") else None
            hs = s.value("msdecomp/table_header_state", None)
            if hs is not None and hdr is not None:
                hdr.restoreState(hs)

        except Exception:
            pass


    def _save_window_state(self):
        try:
            s = QSettings()
            s.setValue("msdecomp/window_geometry", self.saveGeometry())

            if hasattr(self, "splitter"):
                s.setValue("msdecomp/splitter_state", self.splitter.saveState())

            hdr = self.table.horizontalHeader() if hasattr(self, "table") else None
            if hdr is not None:
                s.setValue("msdecomp/table_header_state", hdr.saveState())
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_geom_restored", False):
            return
        self._geom_restored = True

        def _after():
            self._restore_window_state()
            # keep your existing behavior:
            self._fit_view()
        QTimer.singleShot(0, _after)

    def done(self, r):
        # accept()/reject() end here without a closeEvent (Qt >= 6.3).
        self._stop_jobs()
        super().done(r)

    def closeEvent(self, ev):
        self._stop_jobs()
        try:
            self._save_window_state()
        except Exception:
            pass

        try:
            if hasattr(self, "_preview_timer"):
                self._preview_timer.stop()
            if hasattr(self, "_busy_show_timer"):
                self._busy_show_timer.stop()
            try:
                self.view.horizontalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
                self.view.verticalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
            except Exception:
                pass
        except Exception:
            pass

        try:
            self._cmp.detach()      # before _release_memory clears the scene
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass
        super().closeEvent(ev)

    def _release_memory(self):
        """Drop working/preview layers and scene pixmaps on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self._image_work = None
        self._image = None
        self._orig_for_mask = None
        self._preview_img = None
        self._cached_layers = None
        self._cached_residual = None
        self._cached_coarse = None
        self._layer_noise = None
        self._doc = None

        try:
            self.pix_base.setPixmap(QPixmap())
            self.pix_roi.setPixmap(QPixmap())
        except Exception:
            pass
        try:
            self.scene.clear()
        except Exception:
            pass

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

    def _on_close_clicked(self):
        try: self._save_window_state()
        except Exception: pass
        self.reject()

class _MultiScaleDecompPresetDialog(QDialog):
    """
    Preset editor for Multiscale Decomposition (headless + shortcuts).
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Multiscale Decomposition — Preset"))
        init = dict(initial or {})

        v = QVBoxLayout(self)

        # ---- Global ----
        gb = QGroupBox(self.tr("Global"))
        form = QFormLayout(gb)

        self.sp_layers = QSpinBox()
        self.sp_layers.setRange(1, 10)
        self.sp_layers.setValue(int(init.get("layers", 4)))

        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.3, 5.0)
        self.sp_sigma.setDecimals(2)
        self.sp_sigma.setSingleStep(0.1)
        self.sp_sigma.setValue(float(init.get("base_sigma", 1.0)))

        self.cb_linked = QCheckBox(self.tr("Linked RGB channels"))
        self.cb_linked.setChecked(bool(init.get("linked_rgb", True)))

        form.addRow(self.tr("Layers:"), self.sp_layers)
        form.addRow(self.tr("Base sigma:"), self.sp_sigma)
        form.addRow("", self.cb_linked)

        v.addWidget(gb)

        # ---- Layers ----
        gb_layers = QGroupBox(self.tr("Per-Layer Settings"))
        lv = QVBoxLayout(gb_layers)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [self.tr("On"), self.tr("Layer"), self.tr("Gain"), self.tr("Thr (σ)"), self.tr("Amount"), self.tr("Denoise")]
        )

        self.table.verticalHeader().setVisible(False)
        lv.addWidget(self.table)

        v.addWidget(gb_layers)

        # ---- Buttons ----
        btns = QHBoxLayout()
        ok = QPushButton(self.tr("OK"))
        cancel = QPushButton(self.tr("Cancel"))
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        self._populate_table(init)

    def _populate_table(self, init: dict):
        layers = int(self.sp_layers.value())
        cfgs = init.get("layers_cfg", [])

        self.table.setRowCount(layers)

        for i in range(layers):
            cfg = cfgs[i] if i < len(cfgs) else {}

            # Enabled
            chk = QTableWidgetItem("")
            chk.setFlags(chk.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            chk.setCheckState(
                Qt.CheckState.Checked if cfg.get("enabled", True)
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(i, 0, chk)

            self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{float(cfg.get('gain',   1.0)):.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{float(cfg.get('thr',    0.0)):.2f}"))  # N·σ
            self.table.setItem(i, 4, QTableWidgetItem(f"{float(cfg.get('amount', 0.0)):.2f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{float(cfg.get('denoise',0.0)):.2f}"))



    def result_dict(self) -> dict:
        layers = int(self.sp_layers.value())
        out_layers = []

        for r in range(layers):
            enabled = self.table.item(r, 0).checkState() == Qt.CheckState.Checked
            gain = float(self.table.item(r, 2).text())
            thr = float(self.table.item(r, 3).text())
            amt = float(self.table.item(r, 4).text())
            try:
                dn = float(self.table.item(r, 5).text())
            except Exception:
                dn = 0.0

            out_layers.append({
                "enabled": enabled,
                "gain": gain,
                "thr": thr,
                "amount": amt,
                "denoise": dn,
            })


        return {
            "layers": layers,
            "base_sigma": float(self.sp_sigma.value()),
            "linked_rgb": bool(self.cb_linked.isChecked()),
            "layers_cfg": out_layers,
        }
def closeEvent(self, ev):
    self._closing = True
    try:
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        if hasattr(self, "_busy_show_timer"):
            self._busy_show_timer.stop()
        # Optional: disconnect scrollbars to stop ROI scheduling
        try:
            self.view.horizontalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
            self.view.verticalScrollBar().valueChanged.disconnect(self._schedule_roi_preview)
        except Exception:
            pass
    except Exception:
        pass
    super().closeEvent(ev)


# ──────────────────────────────────────────────────────────────────────────────
# Preset entry points (double-click opener + headless drop runner)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_active_doc(main_window):
    """Resolve the active document the same way the tool's toolbar opener does:
    the active MDI subwindow first (so we seed onto the visible image), then the
    document manager as a fallback.
    """
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    return doc


def open_multiscale_decomp_with_preset(main_window, preset: dict | None = None):
    """Open the LIVE Multiscale Decomposition dialog seeded from a preset. Used by
    the double-click preset-open path (ShortcutManager.trigger_with_preset) and
    registered in shortcuts._preset_opener_for_command.
    """
    doc = _resolve_active_doc(main_window)
    if doc is None or getattr(doc, "image", None) is None:
        return

    dlg = MultiscaleDecompDialog(main_window, doc)
    try:
        dlg.setWindowIcon(QIcon(multiscale_decomp_path))
    except Exception:
        pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()


def apply_multiscale_decomp_headless(main_window, preset: dict | None = None, doc=None):
    """Headless runner: apply a Multiscale Decomposition preset to a document
    without showing the live dialog (drop-on-image path). Builds the dialog in
    headless mode so it reuses the exact, tested pipeline
    (decompose → apply_layer_ops → reconstruct → shape restore → mask blend →
    apply_edit) rather than re-implementing it, then disposes of it.

    Returns True on success, False if there was no usable document.
    """
    if doc is None:
        doc = _resolve_active_doc(main_window)
    if doc is None or getattr(doc, "image", None) is None:
        return False

    dlg = MultiscaleDecompDialog(main_window, doc, headless=True)
    try:
        dlg.seed_from_preset(preset or {})
        dlg._commit_to_doc()
    finally:
        try:
            dlg.deleteLater()
        except Exception:
            pass
    return True