# pro/mfdeconv.py non-sport normal version
from __future__ import annotations
import os, sys
import math
import re
import numpy as np
import time
from astropy.io import fits
from PyQt6.QtCore import QObject, pyqtSignal
from pro.psf_utils import compute_psf_kernel_for_image
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread
import contextlib
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
_USE_PROCESS_POOL_FOR_ASSETS = not getattr(sys, "frozen", False)
import gc
try:
    import sep
except Exception:
    sep = None
from pro.free_torch_memory import _free_torch_memory
from pro.mfdeconv_earlystop import EarlyStopper
torch = None        # filled by runtime loader if available
TORCH_OK = False
NO_GRAD = contextlib.nullcontext  # fallback
_XISF_READERS = []
try:
    # e.g. your legacy module
    from legacy import xisf as _legacy_xisf
    if hasattr(_legacy_xisf, "read"):
        _XISF_READERS.append(lambda p: _legacy_xisf.read(p))
    elif hasattr(_legacy_xisf, "open"):
        _XISF_READERS.append(lambda p: _legacy_xisf.open(p)[0])
except Exception:
    pass
try:
    # sometimes projects expose a generic load_image
    from legacy.image_manager import load_image as _generic_load_image  # adjust if needed
    _XISF_READERS.append(lambda p: _generic_load_image(p)[0])
except Exception:
    pass

from pathlib import Path

# at top of file with the other imports

from queue import SimpleQueue
from pro.memory_utils import LRUDict

# ── XISF decode cache → memmap on disk ─────────────────────────────────
import tempfile
import threading
import uuid
import atexit
_XISF_CACHE = LRUDict(50)
_XISF_LOCK  = threading.Lock()
_XISF_TMPFILES = []

from collections import OrderedDict

# ── CHW LRU (float32) built on top of FITS memmap & XISF memmap ────────────────
class _FrameCHWLRU:
    def __init__(self, capacity=8):
        self.cap = int(max(1, capacity))
        self.od = OrderedDict()

    def clear(self):
        self.od.clear()

    def get(self, path, Ht, Wt, color_mode):
        key = (path, Ht, Wt, str(color_mode).lower())
        hit = self.od.get(key)
        if hit is not None:
            self.od.move_to_end(key)
            return hit

        # Load backing array cheaply (memmap for FITS, cached memmap for XISF)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".xisf":
            a = _xisf_cached_array(path)  # float32, HW/HWC/CHW
        else:
            # FITS path: use astropy memmap (no data copy)
            with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
                arr = None
                for h in hdul:
                    if getattr(h, "data", None) is not None:
                        arr = h.data
                        break
                if arr is None:
                    raise ValueError(f"No image data in {path}")
                a = np.asarray(arr)
                # dtype normalize once; keep float32
                if a.dtype.kind in "ui":
                    a = a.astype(np.float32) / (float(np.iinfo(a.dtype).max) or 1.0)
                else:
                    a = a.astype(np.float32, copy=False)

        # Center-crop to (Ht, Wt) and convert to CHW
        a = np.asarray(a)  # float32
        a = _center_crop(a, Ht, Wt)

        # Respect color_mode: “luma” → 1×H×W, “PerChannel” → 3×H×W if RGB present
        cm = str(color_mode).lower()
        if cm == "luma":
            a_chw = _as_chw(_to_luma_local(a)).astype(np.float32, copy=False)
        else:
            a_chw = _as_chw(a).astype(np.float32, copy=False)
            if a_chw.shape[0] == 1 and cm != "luma":
                # still OK (mono data)
                pass

        # LRU insert
        self.od[key] = a_chw
        if len(self.od) > self.cap:
            self.od.popitem(last=False)
        return a_chw

_FRAME_LRU = _FrameCHWLRU(capacity=8)  # tune if you like

def _clear_all_caches():
    try: _clear_xisf_cache()
    except Exception as e:
        import logging
        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
    try: _FRAME_LRU.clear()
    except Exception as e:
        import logging
        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")


def _normalize_to_float32(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind in "ui":
        return (a.astype(np.float32) / (float(np.iinfo(a.dtype).max) or 1.0))
    if a.dtype == np.float32:
        return a
    return a.astype(np.float32, copy=False)

def _xisf_cached_array(path: str) -> np.memmap:
    """
    Decode an XISF image exactly once and back it by a read-only float32 memmap.
    Returns a memmap that can be sliced cheaply for tiles.
    """
    with _XISF_LOCK:
        hit = _XISF_CACHE.get(path)
        if hit is not None:
            fn, shape = hit
            return np.memmap(fn, dtype=np.float32, mode="r", shape=shape)

        # Decode once
        arr, _ = _load_image_array(path)  # your existing loader
        if arr is None:
            raise ValueError(f"XISF loader returned None for {path}")
        arr = np.asarray(arr)
        arrf = _normalize_to_float32(arr)

        # Create a temp file-backed memmap
        tmpdir = tempfile.gettempdir()
        fn = os.path.join(tmpdir, f"xisf_cache_{uuid.uuid4().hex}.mmap")
        mm = np.memmap(fn, dtype=np.float32, mode="w+", shape=arrf.shape)
        mm[...] = arrf[...]
        mm.flush()
        del mm  # close writer handle; re-open below as read-only

        _XISF_CACHE[path] = (fn, arrf.shape)
        _XISF_TMPFILES.append(fn)
        return np.memmap(fn, dtype=np.float32, mode="r", shape=arrf.shape)

def _clear_xisf_cache():
    with _XISF_LOCK:
        for fn in _XISF_TMPFILES:
            try: os.remove(fn)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        _XISF_CACHE.clear()
        _XISF_TMPFILES.clear()

atexit.register(_clear_xisf_cache)


def _is_xisf(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".xisf"

def _read_xisf_numpy(path: str) -> np.ndarray:
    if not _XISF_READERS:
        raise RuntimeError(
            "No XISF readers registered. Ensure one of "
            "legacy.xisf.read/open or *.image_io.load_image is importable."
        )
    last_err = None
    for fn in _XISF_READERS:
        try:
            arr = fn(path)
            if isinstance(arr, tuple):
                arr = arr[0]
            return np.asarray(arr)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All XISF readers failed for {path}: {last_err}")

def _fits_open_data(path: str):
    # ignore_missing_simple=True lets us open headers missing SIMPLE
    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        hdu = hdul[0]
        if hdu.data is None:
            # find first image HDU if primary is header-only
            for h in hdul[1:]:
                if getattr(h, "data", None) is not None:
                    hdu = h
                    break
        data = np.asanyarray(hdu.data)
        hdr  = hdu.header
        return data, hdr

def _load_image_array(path: str) -> tuple[np.ndarray, "fits.Header | None"]:
    """
    Return (numpy array, fits.Header or None). Color-last if 3D.
    dtype left as-is; callers cast to float32. Array is C-contig & writeable.
    """
    if _is_xisf(path):
        arr = _read_xisf_numpy(path)
        hdr = None
    else:
        arr, hdr = _fits_open_data(path)

    a = np.asarray(arr)
    # Move color axis to last if 3D with a leading channel axis
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
    # Ensure contiguous, writeable float32 decisions happen later; here we just ensure writeable
    if (not a.flags.c_contiguous) or (not a.flags.writeable):
        a = np.array(a, copy=True)
    return a, hdr

def _probe_hw(path: str) -> tuple[int, int, int | None]:
    """
    Returns (H, W, C_or_None) without changing data. Moves color to last if needed.
    """
    a, _ = _load_image_array(path)
    if a.ndim == 2:
        return a.shape[0], a.shape[1], None
    if a.ndim == 3:
        h, w, c = a.shape
        # treat mono-3D as (H,W,1)
        if c not in (1, 3) and a.shape[0] in (1, 3):
            a = np.moveaxis(a, 0, -1)
            h, w, c = a.shape
        return h, w, c if c in (1, 3) else None
    raise ValueError(f"Unsupported ndim={a.ndim} for {path}")

def _common_hw_from_paths(paths: list[str]) -> tuple[int, int]:
    """
    Replacement for the old FITS-only version: min(H), min(W) across files.
    """
    Hs, Ws = [], []
    for p in paths:
        h, w, _ = _probe_hw(p)
        Hs.append(int(h)); Ws.append(int(w))
    return int(min(Hs)), int(min(Ws))

def _to_chw_float32(img: np.ndarray, color_mode: str) -> np.ndarray:
    """
    Convert to CHW float32:
      - mono → (1,H,W)
      - RGB → (3,H,W) if 'PerChannel'; (1,H,W) if 'luma'
    """
    x = np.asarray(img)
    if x.ndim == 2:
        y = x.astype(np.float32, copy=False)[None, ...]  # (1,H,W)
        return y
    if x.ndim == 3:
        # color-last (H,W,C) expected
        if x.shape[-1] == 1:
            return x[..., 0].astype(np.float32, copy=False)[None, ...]
        if x.shape[-1] == 3:
            if str(color_mode).lower() in ("perchannel", "per_channel", "perchannelrgb"):
                r, g, b = x[..., 0], x[..., 1], x[..., 2]
                return np.stack([r.astype(np.float32, copy=False),
                                 g.astype(np.float32, copy=False),
                                 b.astype(np.float32, copy=False)], axis=0)
            # luma
            r, g, b = x[..., 0].astype(np.float32, copy=False), x[..., 1].astype(np.float32, copy=False), x[..., 2].astype(np.float32, copy=False)
            L = 0.2126*r + 0.7152*g + 0.0722*b
            return L[None, ...]
        # rare mono-3D
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.moveaxis(x, 0, -1)
            return _to_chw_float32(x, color_mode)
    raise ValueError(f"Unsupported image shape {x.shape}")

def _center_crop_hw(img: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = max(0, (h - Ht)//2); x0 = max(0, (w - Wt)//2)
    return img[y0:y0+Ht, x0:x0+Wt, ...].copy() if (Ht < h or Wt < w) else img

def _stack_loader_memmap(paths: list[str], Ht: int, Wt: int, color_mode: str):
    """
    Drop-in replacement of the old FITS-only helper.
    Returns (ys, hdrs):
      ys   : list of CHW float32 arrays cropped to (Ht,Wt)
      hdrs : list of fits.Header or None (XISF)
    """
    ys, hdrs = [], []
    for p in paths:
        arr, hdr = _load_image_array(p)
        arr = _center_crop_hw(arr, Ht, Wt)
        # normalize integer data to [0,1] like the rest of your code
        if arr.dtype.kind in "ui":
            mx = np.float32(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32, copy=False) / (mx if mx > 0 else 1.0)
        elif arr.dtype.kind == "f":
            arr = arr.astype(np.float32, copy=False)
        else:
            arr = arr.astype(np.float32, copy=False)

        y = _to_chw_float32(arr, color_mode)
        if (not y.flags.c_contiguous) or (not y.flags.writeable):
            y = np.ascontiguousarray(y.astype(np.float32, copy=True))
        ys.append(y)
        hdrs.append(hdr if isinstance(hdr, fits.Header) else None)
    return ys, hdrs

def _safe_primary_header(path: str) -> fits.Header:
    if _is_xisf(path):
        # best-effort synthetic header
        h = fits.Header()
        h["SIMPLE"]  = (True, "created by MFDeconv")
        h["BITPIX"]  = -32
        h["NAXIS"]   = 2
        return h
    try:
        return fits.getheader(path, ext=0, ignore_missing_simple=True)
    except Exception:
        return fits.Header()

# --- CUDA busy/unavailable detector (runtime fallback helper) ---
def _is_cuda_busy_error(e: Exception) -> bool:
    """
    Return True for 'device busy/unavailable' style CUDA errors that can pop up
    mid-run on shared Linux systems. We *exclude* OOM here (handled elsewhere).
    """
    em = str(e).lower()
    if "out of memory" in em:
        return False  # OOM handled by your batch size backoff
    return (
        ("cuda" in em and ("busy" in em or "unavailable" in em))
        or ("device-side" in em and "assert" in em)
        or ("driver shutting down" in em)
    )


def _compute_frame_assets(i, arr, hdr, *, make_masks, make_varmaps,
                          star_mask_cfg, varmap_cfg, status_sink=lambda s: None):
    """
    Worker function: compute PSF and optional star mask / varmap for one frame.
    Returns (index, psf, mask_or_None, var_or_None, log_lines)
    """
    logs = []
    def log(s): logs.append(s)

    # --- PSF sizing by FWHM ---
    f_hdr = _estimate_fwhm_from_header(hdr)
    f_img = _estimate_fwhm_from_image(arr)
    f_whm = f_hdr if (np.isfinite(f_hdr)) else f_img
    if not np.isfinite(f_whm) or f_whm <= 0:
        f_whm = 2.5
    k_auto = _auto_ksize_from_fwhm(f_whm)

    # --- Star-derived PSF with retries ---
    tried, psf = [], None
    for k_try in [k_auto, max(k_auto - 4, 11), 21, 17, 15, 13, 11]:
        if k_try in tried: continue
        tried.append(k_try)
        try:
            out = compute_psf_kernel_for_image(arr, ksize=k_try, det_sigma=6.0, max_stars=80)
            psf_try = out[0] if (isinstance(out, tuple) and len(out) >= 1) else out
            if psf_try is not None:
                psf = psf_try
                break
        except Exception:
            psf = None
    if psf is None:
        psf = _gaussian_psf(f_whm, ksize=k_auto)
    psf = _soften_psf(_normalize_psf(psf.astype(np.float32, copy=False)), sigma_px=0.0)

    mask = None
    var  = None

    if make_masks or make_varmaps:
        # one background per frame (reused by both)
        luma = _to_luma_local(arr)
        vmc = (varmap_cfg or {})
        sky_map, rms_map, err_scalar = _sep_background_precompute(
            luma, bw=int(vmc.get("bw", 64)), bh=int(vmc.get("bh", 64))
        )

        if make_masks:
            smc = star_mask_cfg or {}
            mask = _star_mask_from_precomputed(
                luma, sky_map, err_scalar,
                thresh_sigma = smc.get("thresh_sigma", THRESHOLD_SIGMA),
                max_objs     = smc.get("max_objs", STAR_MASK_MAXOBJS),
                grow_px      = smc.get("grow_px", GROW_PX),
                ellipse_scale= smc.get("ellipse_scale", ELLIPSE_SCALE),
                soft_sigma   = smc.get("soft_sigma", SOFT_SIGMA),
                max_radius_px= smc.get("max_radius_px", MAX_STAR_RADIUS),
                keep_floor   = smc.get("keep_floor", KEEP_FLOOR),
                max_side     = smc.get("max_side", STAR_MASK_MAXSIDE),
                status_cb    = log,
            )

        if make_varmaps:
            vmc = varmap_cfg or {}
            var = _variance_map_from_precomputed(
                luma, sky_map, rms_map, hdr,
                smooth_sigma = vmc.get("smooth_sigma", 1.0),
                floor        = vmc.get("floor", 1e-8),
                status_cb    = log,
            )

    # small per-frame summary
    fwhm_est = _psf_fwhm_px(psf)
    logs.insert(0, f"MFDeconv: PSF{i}: ksize={psf.shape[0]} | FWHM≈{fwhm_est:.2f}px")

    return i, psf, mask, var, logs

def _compute_one_worker(args):
    """
    Top-level picklable worker for ProcessPoolExecutor.
    args: (i, path, make_masks_in_worker, make_varmaps, star_mask_cfg, varmap_cfg)
    Returns (i, psf, mask, var, logs)
    """
    (i, path, make_masks_in_worker, make_varmaps, star_mask_cfg, varmap_cfg) = args
    # avoid BLAS/OMP storm inside each process
    with threadpool_limits(limits=1):
        arr, hdr = _load_image_array(path)           # FITS or XISF
        arr = np.asarray(arr, dtype=np.float32, order="C")
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        if not isinstance(hdr, fits.Header):         # synthesize FITS-like header for XISF
            hdr = _safe_primary_header(path)
        return _compute_frame_assets(
            i, arr, hdr,
            make_masks=bool(make_masks_in_worker),
            make_varmaps=bool(make_varmaps),
            star_mask_cfg=star_mask_cfg,
            varmap_cfg=varmap_cfg,
        )


def _build_psf_and_assets(
    paths,                      # list[str]
    make_masks=False,
    make_varmaps=False,
    status_cb=lambda s: None,
    save_dir: str | None = None,
    star_mask_cfg: dict | None = None,
    varmap_cfg: dict | None = None,
    max_workers: int | None = None,
    star_mask_ref_path: str | None = None,   # build one mask from this frame if provided
    # NEW (passed from multiframe_deconv so we don’t re-probe/convert):
    Ht: int | None = None,
    Wt: int | None = None,
    color_mode: str = "luma",
):
    """
    Parallel PSF + (optional) star mask + variance map per frame.

    Changes from the original:
      • Reuses the decoded frame cache (_FRAME_LRU) for FITS/XISF so we never re-decode.
      • Automatically switches to threads for XISF (so memmaps are shared across workers).
      • Builds a single reference star mask (if requested) from the cached frame and
        center-pads/crops it for all frames (no extra I/O).
      • Preserves return order and streams worker logs back to the UI.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n = len(paths)

    # Resolve target intersection size if caller didn't pass it
    if Ht is None or Wt is None:
        Ht, Wt = _common_hw_from_paths(paths)

    # Sensible default worker count (cap at 8)
    if max_workers is None:
        try:
            hw = os.cpu_count() or 4
        except Exception:
            hw = 4
        max_workers = max(1, min(8, hw))

    # Decide executor: for any XISF, prefer threads so the memmap/cache is shared
    any_xisf = any(os.path.splitext(p)[1].lower() == ".xisf" for p in paths)
    use_proc_pool = (not any_xisf) and _USE_PROCESS_POOL_FOR_ASSETS
    Executor = ProcessPoolExecutor if use_proc_pool else ThreadPoolExecutor
    pool_kind = "process" if use_proc_pool else "thread"
    status_cb(f"MFDeconv: measuring PSFs/masks/varmaps with {max_workers} {pool_kind}s…")

    # ---- helper: pad-or-crop a 2D array to (Ht,Wt), centered ----
    def _center_pad_or_crop_2d(a2d: np.ndarray, Ht: int, Wt: int, fill: float = 1.0) -> np.ndarray:
        a2d = np.asarray(a2d, dtype=np.float32)
        H, W = int(a2d.shape[0]), int(a2d.shape[1])
        # crop first if bigger
        y0 = max(0, (H - Ht) // 2); x0 = max(0, (W - Wt) // 2)
        y1 = min(H, y0 + Ht);       x1 = min(W, x0 + Wt)
        cropped = a2d[y0:y1, x0:x1]
        ch, cw = cropped.shape
        if ch == Ht and cw == Wt:
            return np.ascontiguousarray(cropped, dtype=np.float32)
        # pad if smaller
        out = np.full((Ht, Wt), float(fill), dtype=np.float32)
        oy = (Ht - ch) // 2; ox = (Wt - cw) // 2
        out[oy:oy+ch, ox:ox+cw] = cropped
        return out

    # ---- optional: build one mask from the reference frame and reuse ----
    base_ref_mask = None
    if make_masks and star_mask_ref_path:
        try:
            status_cb(f"Star mask: using reference frame for all masks → {os.path.basename(star_mask_ref_path)}")
            # Pull from the shared frame cache as luma on (Ht,Wt)
            ref_chw = _FRAME_LRU.get(star_mask_ref_path, Ht, Wt, "luma")  # (1,H,W) or (H,W)
            L = ref_chw[0] if (ref_chw.ndim == 3) else ref_chw           # 2D float32

            vmc = (varmap_cfg or {})
            sky_map, rms_map, err_scalar = _sep_background_precompute(
                L, bw=int(vmc.get("bw", 64)), bh=int(vmc.get("bh", 64))
            )
            smc = (star_mask_cfg or {})
            base_ref_mask = _star_mask_from_precomputed(
                L, sky_map, err_scalar,
                thresh_sigma = smc.get("thresh_sigma", THRESHOLD_SIGMA),
                max_objs     = smc.get("max_objs", STAR_MASK_MAXOBJS),
                grow_px      = smc.get("grow_px", GROW_PX),
                ellipse_scale= smc.get("ellipse_scale", ELLIPSE_SCALE),
                soft_sigma   = smc.get("soft_sigma", SOFT_SIGMA),
                max_radius_px= smc.get("max_radius_px", MAX_STAR_RADIUS),
                keep_floor   = smc.get("keep_floor", KEEP_FLOOR),
                max_side     = smc.get("max_side", STAR_MASK_MAXSIDE),
                status_cb    = status_cb,
            )
        except Exception as e:
            status_cb(f"⚠️ Star mask (reference) failed: {e}. Falling back to per-frame masks.")
            base_ref_mask = None

    # for GUI safety, queue logs from workers and flush in the main thread
    log_queue: SimpleQueue = SimpleQueue()

    def enqueue_logs(lines):
        for s in lines:
            log_queue.put(s)

    psfs  = [None] * n
    masks = ([None] * n) if make_masks else None
    vars_ = ([None] * n) if make_varmaps else None
    make_masks_in_worker = bool(make_masks and (base_ref_mask is None))

    # --- thread worker: get frame from cache and compute assets ---
    def _compute_one(i: int, path: str):
        # avoid heavy BLAS oversubscription inside each worker
        with threadpool_limits(limits=1):
            # Pull frame from cache honoring color_mode & target (Ht,Wt)
            img_chw = _FRAME_LRU.get(path, Ht, Wt, color_mode)  # (C,H,W) float32
            # For PSF/mask/varmap we operate on a 2D plane (luma/mono)
            arr2d = img_chw[0] if (img_chw.ndim == 3) else img_chw  # (H,W) float32

            # Header: synthesize a safe FITS-like header (works for XISF too)
            try:
                hdr = _safe_primary_header(path)
            except Exception:
                hdr = fits.Header()

            return _compute_frame_assets(
                i, arr2d, hdr,
                make_masks=bool(make_masks_in_worker),
                make_varmaps=bool(make_varmaps),
                star_mask_cfg=star_mask_cfg,
                varmap_cfg=varmap_cfg,
            )

    # --- submit jobs ---
    with Executor(max_workers=max_workers) as ex:
        futs = []
        for i, p in enumerate(paths, start=1):
            status_cb(f"MFDeconv: measuring PSF {i}/{n} …")
            if use_proc_pool:
                # Process-safe path: worker re-loads inside the subprocess
                futs.append(ex.submit(
                    _compute_one_worker,
                    (i, p, bool(make_masks_in_worker), bool(make_varmaps), star_mask_cfg, varmap_cfg)
                ))
            else:
                # Thread path: hits the shared cache (fast path for XISF/FITS)
                futs.append(ex.submit(_compute_one, i, p))

        done_cnt = 0
        for fut in as_completed(futs):
            i, psf, m, v, logs = fut.result()
            idx = i - 1
            psfs[idx] = psf
            if masks is not None:
                masks[idx] = m
            if vars_ is not None:
                vars_[idx] = v
            enqueue_logs(logs)

            done_cnt += 1
            if (done_cnt % 4) == 0 or done_cnt == n:
                while not log_queue.empty():
                    try:
                        status_cb(log_queue.get_nowait())
                    except Exception:
                        break

    # If we built a single reference mask, apply it to every frame (center pad/crop)
    if base_ref_mask is not None and masks is not None:
        for idx in range(n):
            masks[idx] = _center_pad_or_crop_2d(base_ref_mask, int(Ht), int(Wt), fill=1.0)

    # final flush of any remaining logs
    while not log_queue.empty():
        try:
            status_cb(log_queue.get_nowait())
        except Exception:
            break

    # save PSFs if requested
    if save_dir:
        for i, k in enumerate(psfs, start=1):
            if k is not None:
                fits.PrimaryHDU(k.astype(np.float32, copy=False)).writeto(
                    os.path.join(save_dir, f"psf_{i:03d}.fit"), overwrite=True
                )

    return psfs, masks, vars_


_ALLOWED = re.compile(r"[^A-Za-z0-9_-]+")

# known FITS-style multi-extensions (rightmost-first match)
_KNOWN_EXTS = [
    ".fits.fz", ".fit.fz", ".fits.gz", ".fit.gz",
    ".fz", ".gz",
    ".fits", ".fit"
]

def _sanitize_token(s: str) -> str:
    s = _ALLOWED.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _split_known_exts(p: Path) -> tuple[str, str]:
    """
    Return (name_body, full_ext) where full_ext is a REAL extension block
    (e.g. '.fits.fz'). Any junk like '.0s (1310x880)_MFDeconv' stays in body.
    """
    name = p.name
    for ext in _KNOWN_EXTS:
        if name.lower().endswith(ext):
            body = name[:-len(ext)]
            return body, ext
    # fallback: single suffix
    return p.stem, "".join(p.suffixes)

_SIZE_RE = re.compile(r"\(?\s*(\d{2,5})x(\d{2,5})\s*\)?", re.IGNORECASE)
_EXP_RE  = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
_RX_RE   = re.compile(r"(?<![A-Za-z0-9])(\d+)x\b", re.IGNORECASE)

def _extract_size(body: str) -> str | None:
    m = _SIZE_RE.search(body)
    return f"{m.group(1)}x{m.group(2)}" if m else None

def _extract_exposure_secs(body: str) -> str | None:
    m = _EXP_RE.search(body)
    if not m:
        return None
    secs = int(round(float(m.group(1))))
    return f"{secs}s"

def _strip_metadata_from_base(body: str) -> str:
    s = body

    # normalize common separators first
    s = s.replace(" - ", "_")

    # remove known trailing marker '_MFDeconv'
    s = re.sub(r"(?i)[\s_]+MFDeconv$", "", s)

    # remove parenthetical copy counters e.g. '(1)'
    s = re.sub(r"\(\s*\d+\s*\)$", "", s)

    # remove size (with or without parens) anywhere
    s = _SIZE_RE.sub("", s)

    # remove exposures like '0s', '0.5s', ' 45 s' (even if preceded by a dot)
    s = _EXP_RE.sub("", s)

    # remove any _#x tokens
    s = _RX_RE.sub("", s)

    # collapse whitespace/underscores and sanitize
    s = re.sub(r"[\s]+", "_", s)
    s = _sanitize_token(s)
    return s or "output"

def _canonical_out_name_prefix(base: str, r: int, size: str | None,
                               exposure_secs: str | None, tag: str = "MFDeconv") -> str:
    parts = [_sanitize_token(tag), _sanitize_token(base)]
    if size:
        parts.append(_sanitize_token(size))
    if exposure_secs:
        parts.append(_sanitize_token(exposure_secs))
    if int(max(1, r)) > 1:
        parts.append(f"{int(r)}x")
    return "_".join(parts)

def _sr_out_path(out_path: str, r: int) -> Path:
    """
    Build: MFDeconv_<base>[_<HxW>][_<secs>s][_2x], preserving REAL extensions.
    """
    p = Path(out_path)
    body, real_ext = _split_known_exts(p)

    # harvest metadata from the whole body (not Path.stem)
    size   = _extract_size(body)
    ex_sec = _extract_exposure_secs(body)

    # clean base
    base = _strip_metadata_from_base(body)

    new_stem = _canonical_out_name_prefix(base, r=int(max(1, r)), size=size, exposure_secs=ex_sec, tag="MFDeconv")
    return p.with_name(f"{new_stem}{real_ext}")

def _nonclobber_path(path: str) -> str:
    """
    Version collisions as '_v2', '_v3', ... (no spaces/parentheses).
    """
    p = Path(path)
    if not p.exists():
        return str(p)

    # keep the true extension(s)
    body, real_ext = _split_known_exts(p)

    # if already has _vN, bump it
    m = re.search(r"(.*)_v(\d+)$", body)
    if m:
        base = m.group(1); n = int(m.group(2)) + 1
    else:
        base = body; n = 2

    while True:
        candidate = p.with_name(f"{base}_v{n}{real_ext}")
        if not candidate.exists():
            return str(candidate)
        n += 1

def _iter_folder(basefile: str) -> str:
    d, fname = os.path.split(basefile)
    root, ext = os.path.splitext(fname)
    tgt = os.path.join(d, f"{root}.iters")
    if not os.path.exists(tgt):
        try:
            os.makedirs(tgt, exist_ok=True)
        except Exception:
            # last resort: suffix (n)
            n = 1
            while True:
                cand = os.path.join(d, f"{root}.iters ({n})")
                try:
                    os.makedirs(cand, exist_ok=True)
                    return cand
                except Exception:
                    n += 1
    return tgt

def _save_iter_image(arr, hdr_base, folder, tag, color_mode):
    """
    arr: numpy array (H,W) or (C,H,W) float32
    tag: 'seed' or 'iter_###'
    """
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    hdr = fits.Header(hdr_base) if isinstance(hdr_base, fits.Header) else fits.Header()
    hdr['MF_PART'] = (str(tag), 'MFDeconv intermediate (seed/iter)')
    hdr['MF_COLOR'] = (str(color_mode), 'Color mode used')
    path = os.path.join(folder, f"{tag}.fit")
    # overwrite allowed inside the dedicated folder
    fits.PrimaryHDU(data=arr.astype(np.float32, copy=False), header=hdr).writeto(path, overwrite=True)
    return path


def _process_gui_events_safely():
    app = QApplication.instance()
    if app and QThread.currentThread() is app.thread():
        app.processEvents()

EPS = 1e-6

# -----------------------------
# Helpers: image prep / shapes
# -----------------------------

# new: lightweight loader that yields one frame at a time
def _iter_fits(paths):
    for p in paths:
        with fits.open(p, memmap=False) as hdul:  # ⬅ False
            arr = np.array(hdul[0].data, dtype=np.float32, copy=True)  # ⬅ copy
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            hdr = hdul[0].header.copy()
        yield arr, hdr

def _to_luma_local(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return a
    # (H,W,3) or (3,H,W)
    if a.ndim == 3 and a.shape[-1] == 3:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[0] == 3:
        r, g, b = a[0], a[1], a[2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    return a.mean(axis=-1).astype(np.float32, copy=False)

def _stack_loader(paths):
    ys, hdrs = [], []
    for p in paths:
        with fits.open(p, memmap=False) as hdul:  # ⬅ False
            arr = np.array(hdul[0].data, dtype=np.float32, copy=True)  # ⬅ copy inside with
            hdr = hdul[0].header.copy()
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        ys.append(arr)
        hdrs.append(hdr)
    return ys, hdrs

def _normalize_layout_single(a, color_mode):
    """
    Coerce to:
      - 'luma'       -> (H, W)
      - 'perchannel' -> (C, H, W); mono stays (1,H,W), RGB → (3,H,W)
    Accepts (H,W), (H,W,3), or (3,H,W).
    """
    a = np.asarray(a, dtype=np.float32)

    if color_mode == "luma":
        return _to_luma_local(a)  # returns (H,W)

    # perchannel
    if a.ndim == 2:
        return a[None, ...]                     # (1,H,W)  ← keep mono as 1 channel
    if a.ndim == 3 and a.shape[-1] == 3:
        return np.moveaxis(a, -1, 0)            # (3,H,W)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        return a                                 # already (1,H,W) or (3,H,W)
    # fallback: average any weird shape into luma 1×H×W
    l = _to_luma_local(a)
    return l[None, ...]


def _normalize_layout_batch(arrs, color_mode):
    return [_normalize_layout_single(a, color_mode) for a in arrs]

def _common_hw(data_list):
    """Return minimal (H,W) across items; items are (H,W) or (C,H,W)."""
    Hs, Ws = [], []
    for a in data_list:
        if a.ndim == 2:
            H, W = a.shape
        else:
            _, H, W = a.shape
        Hs.append(H); Ws.append(W)
    return int(min(Hs)), int(min(Ws))

def _center_crop(arr, Ht, Wt):
    """Center-crop arr (H,W) or (C,H,W) to (Ht,Wt)."""
    if arr.ndim == 2:
        H, W = arr.shape
        if H == Ht and W == Wt:
            return arr
        y0 = max(0, (H - Ht) // 2)
        x0 = max(0, (W - Wt) // 2)
        return arr[y0:y0+Ht, x0:x0+Wt]
    else:
        C, H, W = arr.shape
        if H == Ht and W == Wt:
            return arr
        y0 = max(0, (H - Ht) // 2)
        x0 = max(0, (W - Wt) // 2)
        return arr[:, y0:y0+Ht, x0:x0+Wt]

def _sanitize_numeric(a):
    """Replace NaN/Inf, clip negatives, make contiguous float32."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, None).astype(np.float32, copy=False)
    return np.ascontiguousarray(a)

# -----------------------------
# PSF utilities
# -----------------------------

def _gaussian_psf(fwhm_px: float, ksize: int) -> np.ndarray:
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = (ksize - 1) / 2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    g /= (np.sum(g) + EPS)
    return g.astype(np.float32, copy=False)

def _estimate_fwhm_from_header(hdr) -> float:
    for key in ("FWHM", "FWHM_PIX", "PSF_FWHM"):
        if key in hdr:
            try:
                val = float(hdr[key])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return float("nan")

def _estimate_fwhm_from_image(arr) -> float:
    """Fast FWHM estimate from SEP 'a','b' parameters (≈ sigma in px)."""
    if sep is None:
        return float("nan")
    try:
        img = _contig(_to_luma_local(arr))             # ← ensure C-contig float32
        bkg = sep.Background(img)
        data = _contig(img - bkg.back())               # ← ensure data is C-contig
        try:
            err = bkg.globalrms
        except Exception:
            err = float(np.median(bkg.rms()))
        sources = sep.extract(data, 6.0, err=err)
        if sources is None or len(sources) == 0:
            return float("nan")
        a = np.asarray(sources["a"], dtype=np.float32)
        b = np.asarray(sources["b"], dtype=np.float32)
        ab = (a + b) * 0.5
        sigma = float(np.median(ab[np.isfinite(ab) & (ab > 0)]))
        if not np.isfinite(sigma) or sigma <= 0:
            return float("nan")
        return 2.3548 * sigma
    except Exception:
        return float("nan")

def _auto_ksize_from_fwhm(fwhm_px: float, kmin: int = 11, kmax: int = 51) -> int:
    """
    Choose odd kernel size to cover about ±4σ.
    """
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = int(math.ceil(4.0 * sigma))
    k = 2 * r + 1
    k = max(kmin, min(k, kmax))
    if (k % 2) == 0:
        k += 1
    return k

def _flip_kernel(psf):
    # PyTorch dislikes negative strides; make it contiguous.
    return np.flip(np.flip(psf, -1), -2).copy()

def _conv_same_np(img, psf):
    """
    NumPy FFT-based SAME convolution for (H,W) or (C,H,W).
    IMPORTANT: ifftshift the PSF so its peak is at [0,0] before FFT.
    """
    import numpy as _np
    import numpy.fft as _fft

    kh, kw = psf.shape

    def fftconv2(a, k):
        # a is (1,H,W); k is (kh,kw)
        H, W = a.shape[-2:]
        fftH, fftW = _fftshape_same(H, W, kh, kw)
        A  = _fft.rfftn(a, s=(fftH, fftW), axes=(-2, -1))
        K  = _fft.rfftn(_np.fft.ifftshift(k), s=(fftH, fftW), axes=(-2, -1))
        y  = _fft.irfftn(A * K, s=(fftH, fftW), axes=(-2, -1))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        return y[..., sh:sh+H, sw:sw+W]

    if img.ndim == 2:
        return fftconv2(img[None], psf)[0]
    else:
        # per-channel
        return _np.stack([fftconv2(img[c:c+1], psf)[0] for c in range(img.shape[0])], axis=0)

def _normalize_psf(psf):
    psf = np.maximum(psf, 0.0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if not np.isfinite(s) or s <= 1e-6:
        return psf
    return (psf / s).astype(np.float32, copy=False)

def _soften_psf(psf, sigma_px=0.25):
    if sigma_px <= 0:
        return psf
    r = int(max(1, round(3 * sigma_px)))
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2 * sigma_px * sigma_px)).astype(np.float32)
    g /= g.sum() + 1e-6
    return _conv_same_np(psf[None], g)[0]

def _psf_fwhm_px(psf: np.ndarray) -> float:
    """Approximate FWHM (pixels) from second moments of a normalized kernel."""
    psf = np.maximum(psf, 0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if s <= EPS:
        return float("nan")
    k = psf.shape[0]
    y, x = np.mgrid[:k, :k].astype(np.float32)
    cy = float((psf * y).sum() / s)
    cx = float((psf * x).sum() / s)
    var_y = float((psf * (y - cy) ** 2).sum() / s)
    var_x = float((psf * (x - cx) ** 2).sum() / s)
    sigma = math.sqrt(max(0.0, 0.5 * (var_x + var_y)))
    return 2.3548 * sigma  # FWHM≈2.355σ

STAR_MASK_MAXSIDE   = 2048
STAR_MASK_MAXOBJS   = 2000       # cap number of objects
VARMAP_SAMPLE_STRIDE = 8         # (kept for compat; currently unused internally)
THRESHOLD_SIGMA     = 2.0
KEEP_FLOOR          = 0.20
GROW_PX             = 8
MAX_STAR_RADIUS     = 16
SOFT_SIGMA          = 2.0
ELLIPSE_SCALE       = 1.2

def _sep_background_precompute(img_2d: np.ndarray, bw: int = 64, bh: int = 64):
    """One-time SEP background build; returns (sky_map, rms_map, err_scalar)."""
    if sep is None:
        # robust fallback
        med = float(np.median(img_2d))
        mad = float(np.median(np.abs(img_2d - med))) + 1e-6
        sky  = np.full_like(img_2d, med, dtype=np.float32)
        rmsm = np.full_like(img_2d, 1.4826 * mad, dtype=np.float32)
        return sky, rmsm, float(np.median(rmsm))

    a = np.ascontiguousarray(img_2d.astype(np.float32))
    b = sep.Background(a, bw=int(bw), bh=int(bh), fw=3, fh=3)
    sky  = np.asarray(b.back(), dtype=np.float32)
    try:
        rmsm = np.asarray(b.rms(), dtype=np.float32)
        err  = float(b.globalrms)
    except Exception:
        rmsm = np.full_like(a, float(np.median(b.rms())), dtype=np.float32)
        err  = float(np.median(rmsm))
    return sky, rmsm, err


def _auto_star_mask_sep(
    img_2d: np.ndarray,
    thresh_sigma: float = THRESHOLD_SIGMA,
    grow_px: int = GROW_PX,
    max_objs: int = STAR_MASK_MAXOBJS,
    max_side: int = STAR_MASK_MAXSIDE,
    ellipse_scale: float = ELLIPSE_SCALE,
    soft_sigma: float = SOFT_SIGMA,
    max_semiaxis_px: float | None = None,      # kept for API compat; unused
    max_area_px2: float | None = None,         # kept for API compat; unused
    max_radius_px: int = MAX_STAR_RADIUS,
    keep_floor: float = KEEP_FLOOR,
    status_cb=lambda s: None
) -> np.ndarray:
    """
    Build a KEEP weight map (float32 in [0,1]) using SEP detections only.
    **Never writes to img_2d** and draws only into a fresh mask buffer.
    """
    if sep is None:
        return np.ones_like(img_2d, dtype=np.float32, order="C")

    # Optional OpenCV path for fast draw/blur
    try:
        import cv2 as _cv2
        _HAS_CV2 = True
    except Exception:
        _HAS_CV2 = False
        _cv2 = None  # type: ignore

    h, w = map(int, img_2d.shape)

    # Background / residual on our own contiguous buffer
    data = np.ascontiguousarray(img_2d.astype(np.float32))
    bkg = sep.Background(data)
    data_sub = np.ascontiguousarray(data - bkg.back(), dtype=np.float32)
    try:
        err_scalar = float(bkg.globalrms)
    except Exception:
        err_scalar = float(np.median(np.asarray(bkg.rms(), dtype=np.float32)))

    # Downscale for detection only
    det = data_sub
    scale = 1.0
    if max_side and max(h, w) > int(max_side):
        scale = float(max(h, w)) / float(max_side)
        if _HAS_CV2:
            det = _cv2.resize(
                det,
                (max(1, int(round(w / scale))), max(1, int(round(h / scale)))),
                interpolation=_cv2.INTER_AREA
            )
        else:
            s = int(max(1, round(scale)))
            det = det[:(h // s) * s, :(w // s) * s].reshape(h // s, s, w // s, s).mean(axis=(1, 3))
            scale = float(s)

    # When averaging down by 'scale', per-pixel noise scales ~ 1/scale
    err_det = float(err_scalar) / float(max(1.0, scale))

    thresholds = [thresh_sigma, thresh_sigma*2, thresh_sigma*4,
                  thresh_sigma*8, thresh_sigma*16]
    objs = None; used = float("nan"); raw = 0
    for t in thresholds:
        try:
            cand = sep.extract(det, thresh=float(t), err=err_det)
        except Exception:
            cand = None
        n = 0 if cand is None else len(cand)
        if n == 0:          continue
        if n > max_objs*12: continue
        objs, raw, used = cand, n, float(t)
        break

    if objs is None or len(objs) == 0:
        try:
            cand = sep.extract(det, thresh=float(thresholds[-1]), err=err_det, minarea=9)
        except Exception:
            cand = None
        if cand is None or len(cand) == 0:
            status_cb("Star mask: no sources found (mask disabled for this frame).")
            return np.ones((h, w), dtype=np.float32, order="C")
        objs, raw, used = cand, len(cand), float(thresholds[-1])

    # Keep brightest max_objs
    if "flux" in objs.dtype.names:
        idx = np.argsort(objs["flux"])[-int(max_objs):]
        objs = objs[idx]
    else:
        objs = objs[:int(max_objs)]
    kept_after_cap = int(len(objs))

    # Draw on full-res fresh buffer
    mask_u8 = np.zeros((h, w), dtype=np.uint8, order="C")
    s_back = float(scale)
    MR = int(max(1, max_radius_px))
    G  = int(max(0, grow_px))
    ES = float(max(0.1, ellipse_scale))

    drawn = 0
    if _HAS_CV2:
        for o in objs:
            x = int(round(float(o["x"]) * s_back))
            y = int(round(float(o["y"]) * s_back))
            if not (0 <= x < w and 0 <= y < h):
                continue
            a = float(o["a"]) * s_back
            b = float(o["b"]) * s_back
            r = int(math.ceil(ES * max(a, b)))
            r = min(max(r, 0) + G, MR)
            if r <= 0:
                continue
            _cv2.circle(mask_u8, (x, y), r, 1, thickness=-1, lineType=_cv2.LINE_8)
            drawn += 1
    else:
        yy, xx = np.ogrid[:h, :w]
        for o in objs:
            x = int(round(float(o["x"]) * s_back))
            y = int(round(float(o["y"]) * s_back))
            if not (0 <= x < w and 0 <= y < h):
                continue
            a = float(o["a"]) * s_back
            b = float(o["b"]) * s_back
            r = int(math.ceil(ES * max(a, b)))
            r = min(max(r, 0) + G, MR)
            if r <= 0:
                continue
            y0 = max(0, y - r); y1 = min(h, y + r + 1)
            x0 = max(0, x - r); x1 = min(w, x + r + 1)
            ys = yy[y0:y1] - y
            xs = xx[x0:x1] - x
            disk = (ys * ys + xs * xs) <= (r * r)
            mask_u8[y0:y1, x0:x1][disk] = 1
            drawn += 1

    masked_px_hard = int(mask_u8.sum())

    # Feather and convert to KEEP weights in [0,1]
    m = mask_u8.astype(np.float32, copy=False)
    if soft_sigma and soft_sigma > 0.0:
        try:
            if _HAS_CV2:
                k = int(max(1, math.ceil(soft_sigma * 3)) * 2 + 1)
                m = _cv2.GaussianBlur(m, (k, k), float(soft_sigma),
                                      borderType=_cv2.BORDER_REFLECT)
            else:
                from scipy.ndimage import gaussian_filter
                m = gaussian_filter(m, sigma=float(soft_sigma), mode="reflect")
        except Exception:
            pass
        np.clip(m, 0.0, 1.0, out=m)

    keep = 1.0 - m
    kf = float(max(0.0, min(0.99, keep_floor)))
    keep = kf + (1.0 - kf) * keep
    np.clip(keep, 0.0, 1.0, out=keep)

    status_cb(
        f"Star mask: thresh={used:.3g} | detected={raw} | kept={kept_after_cap} | "
        f"drawn={drawn} | masked_px={masked_px_hard} | grow_px={G} | soft_sigma={soft_sigma} | keep_floor={keep_floor}"
    )
    return np.ascontiguousarray(keep, dtype=np.float32)



def _auto_variance_map(
    img_2d: np.ndarray,
    hdr,
    status_cb=lambda s: None,
    sample_stride: int = VARMAP_SAMPLE_STRIDE,  # kept for signature compat; not used
    bw: int = 64,   # SEP background box width  (pixels)
    bh: int = 64,   # SEP background box height (pixels)
    smooth_sigma: float = 1.0,  # Gaussian sigma (px) to smooth the variance map
    floor: float = 1e-8,        # hard floor to prevent blow-up in 1/var
) -> np.ndarray:
    """
    Build a per-pixel variance map in DN^2:

      var_DN ≈ (object_only_DN)/gain  +  var_bg_DN^2

    where:
      - object_only_DN = max(img - sky_DN, 0)
      - var_bg_DN^2 comes from SEP's local background rms (Poisson(sky)+readnoise)
      - if GAIN is missing, estimate 1/gain ≈ median(var_bg)/median(sky)

    Returns float32 array, clipped below by `floor`, optionally smoothed with a
    small Gaussian to stabilize weights. Emits a summary status line.
    """
    img = np.clip(np.asarray(img_2d, dtype=np.float32), 0.0, None)

    # --- Parse header for camera params (optional) ---
    gain = None
    for k in ("EGAIN", "GAIN", "GAIN1", "GAIN2"):
        if k in hdr:
            try:
                g = float(hdr[k])
                if np.isfinite(g) and g > 0:
                    gain = g
                    break
            except Exception:
                pass

    readnoise = None
    for k in ("RDNOISE", "READNOISE", "RN"):
        if k in hdr:
            try:
                rn = float(hdr[k])
                if np.isfinite(rn) and rn >= 0:
                    readnoise = rn
                    break
            except Exception:
                pass

    # --- Local background (full-res) ---
    if sep is not None:
        try:
            b = sep.Background(img, bw=int(bw), bh=int(bh), fw=3, fh=3)
            sky_dn_map = np.asarray(b.back(), dtype=np.float32)
            try:
                rms_dn_map = np.asarray(b.rms(), dtype=np.float32)
            except Exception:
                rms_dn_map = np.full_like(img, float(np.median(b.rms())), dtype=np.float32)
        except Exception:
            sky_dn_map = np.full_like(img, float(np.median(img)), dtype=np.float32)
            med = float(np.median(img))
            mad = float(np.median(np.abs(img - med))) + 1e-6
            rms_dn_map = np.full_like(img, float(1.4826 * mad), dtype=np.float32)
    else:
        sky_dn_map = np.full_like(img, float(np.median(img)), dtype=np.float32)
        med = float(np.median(img))
        mad = float(np.median(np.abs(img - med))) + 1e-6
        rms_dn_map = np.full_like(img, float(1.4826 * mad), dtype=np.float32)

    # Background variance in DN^2
    var_bg_dn2 = np.maximum(rms_dn_map, 1e-6) ** 2

    # Object-only DN
    obj_dn = np.clip(img - sky_dn_map, 0.0, None)

    # Shot-noise coefficient
    if gain is not None and np.isfinite(gain) and gain > 0:
        a_shot = 1.0 / gain
    else:
        sky_med = float(np.median(sky_dn_map))
        varbg_med = float(np.median(var_bg_dn2))
        if sky_med > 1e-6:
            a_shot = np.clip(varbg_med / sky_med, 0.0, 10.0)  # ~ 1/gain estimate
        else:
            a_shot = 0.0

    # Total variance: background + shot noise from object-only flux
    v = var_bg_dn2 + a_shot * obj_dn
    v_raw = v.copy()

    # Optional mild smoothing
    if smooth_sigma and smooth_sigma > 0:
        try:
            import cv2 as _cv2
            k = int(max(1, int(round(3 * float(smooth_sigma)))) * 2 + 1)
            v = _cv2.GaussianBlur(v, (k, k), float(smooth_sigma), borderType=_cv2.BORDER_REFLECT)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter
                v = gaussian_filter(v, sigma=float(smooth_sigma), mode="reflect")
            except Exception:
                r = int(max(1, round(3 * float(smooth_sigma))))
                yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
                gk = np.exp(-(xx*xx + yy*yy) / (2.0 * float(smooth_sigma) * float(smooth_sigma))).astype(np.float32)
                gk /= (gk.sum() + EPS)
                v = _conv_same_np(v, gk)

    # Clip to avoid zero/negative variances
    v = np.clip(v, float(floor), None).astype(np.float32, copy=False)

    # Emit telemetry
    try:
        sky_med  = float(np.median(sky_dn_map))
        rms_med  = float(np.median(np.sqrt(var_bg_dn2)))
        floor_pct = float((v <= floor).mean() * 100.0)
        status_cb(
            "Variance map: "
            f"sky_med={sky_med:.3g} DN | rms_med={rms_med:.3g} DN | "
            f"gain={(gain if gain is not None else 'NA')} | rn={(readnoise if readnoise is not None else 'NA')} | "
            f"smooth_sigma={smooth_sigma} | floor={floor} ({floor_pct:.2f}% at floor)"
        )
    except Exception:
        pass

    return v


def _star_mask_from_precomputed(
    img_2d: np.ndarray,
    sky_map: np.ndarray,
    err_scalar: float,
    *,
    thresh_sigma: float,
    max_objs: int,
    grow_px: int,
    ellipse_scale: float,
    soft_sigma: float,
    max_radius_px: int,
    keep_floor: float,
    max_side: int,
    status_cb=lambda s: None
) -> np.ndarray:
    """
    Build a KEEP weight map using a *downscaled detection / full-res draw* path.
    **Never writes to img_2d**; all drawing happens in a fresh `mask_u8`.
    """
    # Optional OpenCV fast path
    try:
        import cv2 as _cv2
        _HAS_CV2 = True
    except Exception:
        _HAS_CV2 = False
        _cv2 = None  # type: ignore

    H, W = map(int, img_2d.shape)

    # Residual for detection (contiguous, separate buffer)
    data_sub = np.ascontiguousarray((img_2d - sky_map).astype(np.float32))

    # Downscale *detection only* to speed up, never the draw step
    det = data_sub
    scale = 1.0
    if max_side and max(H, W) > int(max_side):
        scale = float(max(H, W)) / float(max_side)
        if _HAS_CV2:
            det = _cv2.resize(
                det,
                (max(1, int(round(W / scale))), max(1, int(round(H / scale)))),
                interpolation=_cv2.INTER_AREA
            )
        else:
            s = int(max(1, round(scale)))
            det = det[:(H // s) * s, :(W // s) * s].reshape(H // s, s, W // s, s).mean(axis=(1, 3))
            scale = float(s)

    # Threshold ladder
    thresholds = [thresh_sigma, thresh_sigma*2, thresh_sigma*4,
                  thresh_sigma*8, thresh_sigma*16]
    objs = None; used = float("nan"); raw = 0
    for t in thresholds:
        cand = sep.extract(det, thresh=float(t), err=float(err_scalar))
        n = 0 if cand is None else len(cand)
        if n == 0:          continue
        if n > max_objs*12: continue
        objs, raw, used = cand, n, float(t)
        break

    if objs is None or len(objs) == 0:
        try:
            cand = sep.extract(det, thresh=thresholds[-1], err=float(err_scalar), minarea=9)
        except Exception:
            cand = None
        if cand is None or len(cand) == 0:
            status_cb("Star mask: no sources found (mask disabled for this frame).")
            return np.ones((H, W), dtype=np.float32, order="C")
        objs, raw, used = cand, len(cand), float(thresholds[-1])

    # Brightest max_objs
    if "flux" in objs.dtype.names:
        idx = np.argsort(objs["flux"])[-int(max_objs):]
        objs = objs[idx]
    else:
        objs = objs[:int(max_objs)]
    kept = len(objs)

    # ---- draw back on full-res into a brand-new buffer ----
    mask_u8 = np.zeros((H, W), dtype=np.uint8, order="C")
    s_back = float(scale)
    MR = int(max(1, max_radius_px))
    G  = int(max(0, grow_px))
    ES = float(max(0.1, ellipse_scale))

    drawn = 0
    if _HAS_CV2:
        for o in objs:
            x = int(round(float(o["x"]) * s_back))
            y = int(round(float(o["y"]) * s_back))
            if not (0 <= x < W and 0 <= y < H):
                continue
            a = float(o["a"]) * s_back
            b = float(o["b"]) * s_back
            r = int(math.ceil(ES * max(a, b)))
            r = min(max(r, 0) + G, MR)
            if r <= 0:
                continue
            _cv2.circle(mask_u8, (x, y), r, 1, thickness=-1, lineType=_cv2.LINE_8)
            drawn += 1
    else:
        for o in objs:
            x = int(round(float(o["x"]) * s_back))
            y = int(round(float(o["y"]) * s_back))
            if not (0 <= x < W and 0 <= y < H):
                continue
            a = float(o["a"]) * s_back
            b = float(o["b"]) * s_back
            r = int(math.ceil(ES * max(a, b)))
            r = min(max(r, 0) + G, MR)
            if r <= 0:
                continue
            y0 = max(0, y - r); y1 = min(H, y + r + 1)
            x0 = max(0, x - r); x1 = min(W, x + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (yy - y)*(yy - y) + (xx - x)*(xx - x) <= r*r
            mask_u8[y0:y1, x0:x1][disk] = 1
            drawn += 1

    # Feather + convert to keep weights
    m = mask_u8.astype(np.float32, copy=False)
    if soft_sigma > 0:
        try:
            if _HAS_CV2:
                k = int(max(1, int(round(3*soft_sigma)))*2 + 1)
                m = _cv2.GaussianBlur(m, (k, k), float(soft_sigma),
                                      borderType=_cv2.BORDER_REFLECT)
            else:
                from scipy.ndimage import gaussian_filter
                m = gaussian_filter(m, sigma=float(soft_sigma), mode="reflect")
        except Exception:
            pass
        np.clip(m, 0.0, 1.0, out=m)

    keep = 1.0 - m
    kf = float(max(0.0, min(0.99, keep_floor)))
    keep = kf + (1.0 - kf) * keep
    np.clip(keep, 0.0, 1.0, out=keep)

    status_cb(f"Star mask: thresh={used:.3g} | detected={raw} | kept={kept} | drawn={drawn} | keep_floor={keep_floor}")
    return np.ascontiguousarray(keep, dtype=np.float32)


def _variance_map_from_precomputed(
    img_2d: np.ndarray,
    sky_map: np.ndarray,
    rms_map: np.ndarray,
    hdr,
    *,
    smooth_sigma: float,
    floor: float,
    status_cb=lambda s: None
) -> np.ndarray:
    img = np.clip(np.asarray(img_2d, dtype=np.float32), 0.0, None)
    var_bg_dn2 = np.maximum(rms_map, 1e-6) ** 2
    obj_dn = np.clip(img - sky_map, 0.0, None)

    gain = None
    for k in ("EGAIN", "GAIN", "GAIN1", "GAIN2"):
        if k in hdr:
            try:
                g = float(hdr[k]);  gain = g if (np.isfinite(g) and g > 0) else None
                if gain is not None: break
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    if gain is not None:
        a_shot = 1.0 / gain
    else:
        sky_med  = float(np.median(sky_map))
        varbg_med= float(np.median(var_bg_dn2))
        a_shot = (varbg_med / sky_med) if sky_med > 1e-6 else 0.0
        a_shot = float(np.clip(a_shot, 0.0, 10.0))

    v = var_bg_dn2 + a_shot * obj_dn
    if smooth_sigma > 0:
        try:
            import cv2 as _cv2
            k = int(max(1, int(round(3*smooth_sigma)))*2 + 1)
            v = _cv2.GaussianBlur(v, (k,k), float(smooth_sigma), borderType=_cv2.BORDER_REFLECT)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter
                v = gaussian_filter(v, sigma=float(smooth_sigma), mode="reflect")
            except Exception:
                pass

    np.clip(v, float(floor), None, out=v)
    try:
        rms_med = float(np.median(np.sqrt(var_bg_dn2)))
        status_cb(f"Variance map: sky_med={float(np.median(sky_map)):.3g} DN | rms_med={rms_med:.3g} DN | smooth_sigma={smooth_sigma} | floor={floor}")
    except Exception:
        pass
    return v.astype(np.float32, copy=False)



# -----------------------------
# Robust weighting (Huber)
# -----------------------------

EPS = 1e-6

def _estimate_scalar_variance(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return float((1.4826 * mad) ** 2)

def _weight_map(y, pred, huber_delta, var_map=None, mask=None):
    """
    W = [psi(r)/r] * 1/(var + eps) * mask, psi=Huber
    If huber_delta<0, delta = (-huber_delta) * RMS(residual) via MAD.
    y,pred: (H,W) or (C,H,W). var_map/mask are 2D; broadcast if needed.
    """
    r = y - pred
    # auto delta?
    if huber_delta < 0:
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-6
        rms = 1.4826 * mad
        delta = (-huber_delta) * max(rms, 1e-6)
    else:
        delta = huber_delta

    absr = np.abs(r)
    if float(delta) > 0:
        psi_over_r = np.where(absr <= delta, 1.0, delta / (absr + EPS)).astype(np.float32)
    else:
        psi_over_r = np.ones_like(r, dtype=np.float32)

    if var_map is None:
        v = _estimate_scalar_variance(r)
    else:
        v = var_map
        if v.ndim == 2 and r.ndim == 3:
            v = v[None, ...]
    w = psi_over_r / (v + EPS)
    if mask is not None:
        m = mask if mask.ndim == w.ndim else (mask[None, ...] if w.ndim == 3 else mask)
        w = w * m
    return w


# -----------------------------
# Torch / conv
# -----------------------------

def _fftshape_same(H, W, kh, kw):
    return H + kh - 1, W + kw - 1

# ---------- Torch FFT helpers (FIXED: carry padH/padW) ----------
def _precompute_torch_psf_ffts(psfs, flip_psf, H, W, device, dtype):
    """
    Pack (Kf,padH,padW,kh,kw) so the conv can crop correctly to SAME.
    Kernel is ifftshifted before padding.
    """
    tfft = torch.fft
    psf_fft, psfT_fft = [], []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        padH, padW = _fftshape_same(H, W, kh, kw)
        k_small  = torch.as_tensor(np.fft.ifftshift(k),  device=device, dtype=dtype)
        kT_small = torch.as_tensor(np.fft.ifftshift(kT), device=device, dtype=dtype)
        Kf  = tfft.rfftn(k_small,  s=(padH, padW))
        KTf = tfft.rfftn(kT_small, s=(padH, padW))
        psf_fft.append((Kf,  padH, padW, kh, kw))
        psfT_fft.append((KTf, padH, padW, kh, kw))
    return psf_fft, psfT_fft


def _fft_conv_same_torch(x, Kf_pack, out_spatial):
    tfft = torch.fft
    Kf, padH, padW, kh, kw = Kf_pack
    H, W = x.shape[-2], x.shape[-1]
    if x.ndim == 2:
        X = tfft.rfftn(x, s=(padH, padW))
        y = tfft.irfftn(X * Kf, s=(padH, padW))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        out_spatial.copy_(y[sh:sh+H, sw:sw+W])
        return out_spatial
    else:
        X = tfft.rfftn(x, s=(padH, padW), dim=(-2,-1))
        y = tfft.irfftn(X * Kf, s=(padH, padW), dim=(-2,-1))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        out_spatial.copy_(y[..., sh:sh+H, sw:sw+W])
        return out_spatial

# ---------- NumPy FFT helpers ----------
def _precompute_np_psf_ffts(psfs, flip_psf, H, W):
    import numpy.fft as fft
    meta, Kfs, KTfs = [], [], []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        fftH, fftW = _fftshape_same(H, W, kh, kw)
        Kfs.append( fft.rfftn(np.fft.ifftshift(k),  s=(fftH, fftW)) )
        KTfs.append(fft.rfftn(np.fft.ifftshift(kT), s=(fftH, fftW)) )
        meta.append((kh, kw, fftH, fftW))
    return Kfs, KTfs, meta

def _fft_conv_same_np(a, Kf, kh, kw, fftH, fftW, out):
    import numpy.fft as fft
    if a.ndim == 2:
        A = fft.rfftn(a, s=(fftH, fftW))
        y = fft.irfftn(A * Kf, s=(fftH, fftW))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        out[...] = y[sh:sh+a.shape[0], sw:sw+a.shape[1]]
        return out
    else:
        C, H, W = a.shape
        acc = []
        for c in range(C):
            A = fft.rfftn(a[c], s=(fftH, fftW))
            y = fft.irfftn(A * Kf, s=(fftH, fftW))
            sh, sw = (kh - 1)//2, (kw - 1)//2
            acc.append(y[sh:sh+H, sw:sw+W])
        out[...] = np.stack(acc, 0)
        return out



def _torch_device():
    if TORCH_OK and (torch is not None):
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # DirectML: we passed dml_device from outer scope; keep a module-global
        if globals().get("dml_ok", False) and globals().get("dml_device", None) is not None:
            return globals()["dml_device"]
    return torch.device("cpu")

def _to_t(x: np.ndarray):
    if not (TORCH_OK and (torch is not None)):
        raise RuntimeError("Torch path requested but torch is unavailable")
    device = _torch_device()
    t = torch.from_numpy(x)
    # DirectML wants explicit .to(device)
    return t.to(device, non_blocking=True) if str(device) != "cpu" else t

def _contig(x):
    return np.ascontiguousarray(x, dtype=np.float32)

def _conv_same_torch(img_t, psf_t):
    """
    img_t: torch tensor on DEVICE, (H,W) or (C,H,W)
    psf_t: torch tensor on DEVICE, (1,1,kh,kw)  (single kernel)
    Pads with 'reflect' to avoid zero-padding ringing.
    """
    kh, kw = psf_t.shape[-2:]
    pad = (kw // 2, kw - kw // 2 - 1,  # left, right
           kh // 2, kh - kh // 2 - 1)  # top, bottom

    if img_t.ndim == 2:
        x = img_t[None, None]
        x = torch.nn.functional.pad(x, pad, mode="reflect")
        y = torch.nn.functional.conv2d(x, psf_t, padding=0)
        return y[0, 0]
    else:
        C = img_t.shape[0]
        x = img_t[None]
        x = torch.nn.functional.pad(x, pad, mode="reflect")
        w = psf_t.repeat(C, 1, 1, 1)
        y = torch.nn.functional.conv2d(x, w, padding=0, groups=C)
        return y[0]

def _safe_inference_context():
    """
    Return a valid, working no-grad context:
      - prefer torch.inference_mode() if it exists *and* can be entered,
      - otherwise fall back to torch.no_grad(),
      - if torch is unavailable, return NO_GRAD.
    """
    if not (TORCH_OK and (torch is not None)):
        return NO_GRAD

    cm = getattr(torch, "inference_mode", None)
    if cm is None:
        return torch.no_grad

    # Probe inference_mode once; if it explodes on this build, fall back.
    try:
        with cm():
            pass
        return cm
    except Exception:
        return torch.no_grad

def _ensure_mask_list(masks, data):
    # 1s where valid, 0s where invalid (soft edges allowed)
    if masks is None:
        return [np.ones_like(a if a.ndim==2 else a[0], dtype=np.float32) for a in data]
    out = []
    for a, m in zip(data, masks):
        base = a if a.ndim==2 else a[0]      # mask is 2D; shared across channels
        if m is None:
            out.append(np.ones_like(base, dtype=np.float32))
        else:
            mm = np.asarray(m, dtype=np.float32)
            if mm.ndim == 3:   # tolerate (1,H,W) or (C,H,W)
                mm = mm[0]
            if mm.shape != base.shape:
                # center crop to match (common intersection already applied)
                Ht, Wt = base.shape
                mm = _center_crop(mm, Ht, Wt)
            # keep as float weights in [0,1] (do not threshold!)
            out.append(np.clip(mm.astype(np.float32, copy=False), 0.0, 1.0))
    return out

def _ensure_var_list(variances, data):
    # If None, we’ll estimate a robust scalar per frame on-the-fly.
    if variances is None:
        return [None]*len(data)
    out = []
    for a, v in zip(data, variances):
        if v is None:
            out.append(None)
        else:
            vv = np.asarray(v, dtype=np.float32)
            if vv.ndim == 3:
                vv = vv[0]
            base = a if a.ndim==2 else a[0]
            if vv.shape != base.shape:
                Ht, Wt = base.shape
                vv = _center_crop(vv, Ht, Wt)
            # clip tiny/negatives
            vv = np.nan_to_num(vv, nan=1e-8, posinf=1e8, neginf=1e8, copy=False)
            vv = np.clip(vv, 1e-8, None).astype(np.float32, copy=False)
            out.append(vv)
    return out

# ---- SR operators (downsample / upsample-sum) ----
def _downsample_avg(img, r: int):
    """Average-pool over non-overlapping r×r blocks. Works for (H,W) or (C,H,W)."""
    if r <= 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        H, W = a.shape
        Hs, Ws = (H // r) * r, (W // r) * r
        a = a[:Hs, :Ws].reshape(Hs//r, r, Ws//r, r).mean(axis=(1,3))
        return a
    else:
        C, H, W = a.shape
        Hs, Ws = (H // r) * r, (W // r) * r
        a = a[:, :Hs, :Ws].reshape(C, Hs//r, r, Ws//r, r).mean(axis=(2,4))
        return a

def _upsample_sum(img, r: int, target_hw: tuple[int,int] | None = None):
    """Adjoint of average-pooling: replicate-sum each pixel into an r×r block.
       For (H,W) or (C,H,W). If target_hw given, center-crop/pad to that size.
    """
    if r <= 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        H, W = a.shape
        out = np.kron(a, np.ones((r, r), dtype=np.float32))
    else:
        C, H, W = a.shape
        out = np.stack([np.kron(a[c], np.ones((r, r), dtype=np.float32)) for c in range(C)], axis=0)
    if target_hw is not None:
        Ht, Wt = target_hw
        out = _center_crop(out, Ht, Wt)
    return out

def _gaussian2d(ksize: int, sigma: float) -> np.ndarray:
    r = (ksize - 1) // 2
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    g = np.exp(-(x*x + y*y)/(2.0*sigma*sigma)).astype(np.float32)
    g /= g.sum() + EPS
    return g

def _conv2_same_np(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    # lightweight wrap for 2D conv on (H,W) or (C,H,W) with same-size output
    return _conv_same_np(a if a.ndim==3 else a[None], k)[0] if a.ndim==2 else _conv_same_np(a, k)

def _solve_super_psf_from_native(f_native: np.ndarray, r: int, sigma: float = 1.1,
                                 iters: int = 500, lr: float = 0.1) -> np.ndarray:
    """
    Solve: h*  = argmin_h || f_native - (D(h) * g_sigma) ||_2^2,
    where h is (r*k)×(r*k) if f_native is k×k. Returns normalized h (sum=1).
    """
    f = np.asarray(f_native, dtype=np.float32)

    # NEW: sanitize to 2D odd square before anything else
    if f.ndim != 2:
        f = np.squeeze(f)
        if f.ndim != 2:
            raise ValueError(f"PSF must be 2D, got shape {f.shape}")

    H, W = int(f.shape[0]), int(f.shape[1])
    k_sq = min(H, W)
    # center-crop to square if needed
    if H != W:
        y0 = (H - k_sq) // 2
        x0 = (W - k_sq) // 2
        f = f[y0:y0 + k_sq, x0:x0 + k_sq]
        H = W = k_sq

    # enforce odd size (required by SAME padding math)
    if (H % 2) == 0:
        # drop one pixel border to make it odd (centered)
        f = f[1:, 1:]
        H = W = f.shape[0]

    k = int(H)                     # k is now odd and square
    kr = int(k * r)


    g = _gaussian2d(k, max(sigma, 1e-3)).astype(np.float32)

    h0 = np.zeros((kr, kr), dtype=np.float32)
    h0[::r, ::r] = f
    h0 = _normalize_psf(h0)

    if TORCH_OK:
        import torch.nn.functional as F
        dev = _torch_device()

        # (1) Make sure Gaussian kernel is odd-sized for SAME conv padding
        g_pad = g
        if (g.shape[-1] % 2) == 0:
            # ensure odd + renormalize
            gg = _pad_kernel_to(g, g.shape[-1] + 1)
            g_pad = gg.astype(np.float32, copy=False)

        t   = torch.tensor(h0, device=dev, dtype=torch.float32, requires_grad=True)
        f_t = torch.tensor(f,  device=dev, dtype=torch.float32)
        g_t = torch.tensor(g_pad, device=dev, dtype=torch.float32)
        opt = torch.optim.Adam([t], lr=lr)

        # Helpful assertion avoids silent shape traps
        H, W = t.shape
        assert (H % r) == 0 and (W % r) == 0, f"h shape {t.shape} not divisible by r={r}"
        Hr, Wr = H // r, W // r

        try:
            for _ in range(max(10, iters)):
                opt.zero_grad(set_to_none=True)

                # (2) Downsample with avg_pool2d instead of reshape/mean
                blk = t.narrow(0, 0, Hr * r).narrow(1, 0, Wr * r).contiguous()
                th  = F.avg_pool2d(blk[None, None], kernel_size=r, stride=r)[0, 0]  # (k,k)

                # (3) Native-space blur with guaranteed-odd g_t
                pad = g_t.shape[-1] // 2
                conv = F.conv2d(th[None, None], g_t[None, None], padding=pad)[0, 0]

                loss = torch.mean((conv - f_t) ** 2)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    t.clamp_(min=0.0)
                    t /= (t.sum() + 1e-8)
            h = t.detach().cpu().numpy().astype(np.float32)
        except Exception:
            # (4) Conservative safety net: if a backend balks (commonly at r=2),
            #     fall back to the NumPy solver *just for this kernel*.
            h = None

    if not TORCH_OK or h is None:
        # NumPy fallback (unchanged)
        h = h0.copy()
        eta = float(lr)
        for _ in range(max(50, iters)):
            Dh    = _downsample_avg(h, r)
            conv  = _conv2_same_np(Dh, g)
            resid = (conv - f)
            grad_Dh = _conv2_same_np(resid, np.flip(np.flip(g, 0), 1))
            grad_h  = _upsample_sum(grad_Dh, r, target_hw=h.shape)
            h = np.clip(h - eta * grad_h, 0.0, None)
            s = float(h.sum()); h /= (s + 1e-8)
            eta *= 0.995

    return _normalize_psf(h)


def _downsample_avg_t(x, r: int):
    if r <= 1:
        return x
    if x.ndim == 2:
        H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x
        x2 = x[:Hr, :Wr]
        # ❌ .view → ✅ .reshape
        return x2.reshape(Hr // r, r, Wr // r, r).mean(dim=(1, 3))
    else:
        C, H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x
        x2 = x[:, :Hr, :Wr]
        # ❌ .view → ✅ .reshape
        return x2.reshape(C, Hr // r, r, Wr // r, r).mean(dim=(2, 4))

def _upsample_sum_t(x, r: int):
    if r <= 1:
        return x
    if x.ndim == 2:
        return x.repeat_interleave(r, dim=0).repeat_interleave(r, dim=1)
    else:
        return x.repeat_interleave(r, dim=-2).repeat_interleave(r, dim=-1)

def _sep_bg_rms(frames):
    """Return a robust background RMS using SEP's background model on the first frame."""
    if sep is None or not frames:
        return None
    try:
        y0 = frames[0] if frames[0].ndim == 2 else frames[0][0]  # use luma/first channel
        a = np.ascontiguousarray(y0, dtype=np.float32)
        b = sep.Background(a, bw=64, bh=64, fw=3, fh=3)
        try:
            rms_val = float(b.globalrms)
        except Exception:
            # some SEP builds don’t expose globalrms; fall back to the map’s median
            rms_val = float(np.median(np.asarray(b.rms(), dtype=np.float32)))
        return rms_val
    except Exception:
        return None

# =========================
# Memory/streaming helpers
# =========================

def _approx_bytes(arr_like_shape, dtype=np.float32):
    """Rough byte estimator for a given shape/dtype."""
    return int(np.prod(arr_like_shape)) * np.dtype(dtype).itemsize

def _mem_model(
    grid_hw: tuple[int,int],
    r: int,
    ksize: int,
    channels: int,
    mem_target_mb: int,
    prefer_tiles: bool = False,
    min_tile: int = 256,
    max_tile: int = 2048,
) -> dict:
    """
    Pick a batch size (#frames) and optional tile size (HxW) given a memory budget.
    Very conservative — aims to bound peak working-set on CPU/GPU.
    """
    Hs, Ws = grid_hw
    halo = (ksize // 2) * max(1, r)        # SR grid halo if r>1
    C    = max(1, channels)

    # working-set per *full-frame* conv scratch (num/den/tmp/etc.)
    per_frame_fft_like = 3 * _approx_bytes((C, Hs, Ws))  # tmp/pred + in/out buffers
    global_accum = 2 * _approx_bytes((C, Hs, Ws))        # num + den

    budget = int(mem_target_mb * 1024 * 1024)

    # Try to stay in full-frame mode first unless prefer_tiles
    B_full = max(1, (budget - global_accum) // max(per_frame_fft_like, 1))
    use_tiles = prefer_tiles or (B_full < 1)

    if not use_tiles:
        return dict(batch_frames=int(B_full), tiles=None, halo=int(halo), ksize=int(ksize))

    # Tile mode: pick a square tile side t that fits
    # scratch per tile ~ 3*C*(t+2h)^2 + accum(core) ~ small
    # try descending from max_tile
    t = int(min(max_tile, max(min_tile, 1 << int(np.floor(np.log2(min(Hs, Ws)))))))
    while t >= min_tile:
        th = t + 2 * halo
        per_tile = 3 * _approx_bytes((C, th, th))
        B_tile   = max(1, (budget - global_accum) // max(per_tile, 1))
        if B_tile >= 1:
            return dict(batch_frames=int(B_tile), tiles=(t, t), halo=int(halo), ksize=int(ksize))
        t //= 2

    # Worst case: 1 frame, minimal tile
    return dict(batch_frames=1, tiles=(min_tile, min_tile), halo=int(halo), ksize=int(ksize))

def _build_seed_running_mu_sigma_from_paths(
    paths, Ht, Wt, color_mode,
    *, bootstrap_frames=24, clip_sigma=3.5,  # clip_sigma used for streaming updates
    status_cb=lambda s: None, progress_cb=None
):
    """
    Seed:
      1) Load first B frames -> mean0
      2) MAD around mean0 -> ±4·MAD mask -> masked-mean seed (one mini-iteration)
      3) Stream remaining frames with σ-clipped Welford updates (unchanged behavior)
    Returns float32 image in (H,W) or (C,H,W) matching color_mode.
    """
    def p(frac, msg):
        if progress_cb:
            progress_cb(float(max(0.0, min(1.0, frac))), msg)

    n_total = len(paths)
    B = int(max(1, min(int(bootstrap_frames), n_total)))
    status_cb(f"MFDeconv: Seed bootstrap {B} frame(s) with ±4·MAD clip on the average…")
    p(0.00, f"bootstrap load 0/{B}")

    # ---------- load first B frames ----------
    boot = []
    for i, pth in enumerate(paths[:B], start=1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        boot.append(ys[0].astype(np.float32, copy=False))
        if (i == B) or (i % 4 == 0):
            p(0.25 * (i / float(B)), f"bootstrap load {i}/{B}")

    stack = np.stack(boot, axis=0)  # (B,H,W) or (B,C,H,W)
    del boot

    # ---------- mean0 ----------
    mean0 = np.mean(stack, axis=0, dtype=np.float32)
    p(0.28, "bootstrap mean computed")

    # ---------- ±4·MAD clip around mean0, then masked mean (one pass) ----------
    # MAD per-pixel: median(|x - mean0|)
    abs_dev = np.abs(stack - mean0[None, ...])
    mad = np.median(abs_dev, axis=0).astype(np.float32, copy=False)

    thr = 4.0 * mad + EPS
    mask = (abs_dev <= thr)

    # masked mean with fallback to mean0 where all rejected
    m = mask.astype(np.float32, copy=False)
    sum_acc = np.sum(stack * m, axis=0, dtype=np.float32)
    cnt_acc = np.sum(m, axis=0, dtype=np.float32)
    seed = mean0.copy()
    np.divide(sum_acc, np.maximum(cnt_acc, 1.0), out=seed, where=(cnt_acc > 0.5))
    p(0.36, "±4·MAD masked mean computed")

    # ---------- initialize Welford state from seed ----------
    # Start μ=seed, set an initial variance envelope from the bootstrap dispersion
    dif = stack - seed[None, ...]
    M2 = np.sum(dif * dif, axis=0, dtype=np.float32)
    cnt = np.full_like(seed, float(B), dtype=np.float32)
    mu  = seed.astype(np.float32, copy=False)
    del stack, abs_dev, mad, m, sum_acc, cnt_acc, dif

    p(0.40, "seed initialized; streaming refinements…")

    # ---------- stream remaining frames with σ-clipped Welford updates ----------
    remain = n_total - B
    if remain > 0:
        status_cb(f"MFDeconv: Seed μ–σ clipping {remain} remaining frame(s) (k={clip_sigma:.2f})…")

    k = float(clip_sigma)
    for j, pth in enumerate(paths[B:], start=1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        x = ys[0].astype(np.float32, copy=False)

        var  = M2 / np.maximum(cnt - 1.0, 1.0)
        sigma = np.sqrt(np.maximum(var, 1e-12, dtype=np.float32))

        accept = (np.abs(x - mu) <= (k * sigma))
        acc = accept.astype(np.float32, copy=False)

        n_new = cnt + acc
        delta = x - mu
        mu_n  = mu + (acc * delta) / np.maximum(n_new, 1.0)
        M2    = M2 + acc * delta * (x - mu_n)

        mu, cnt = mu_n, n_new

        if (j == remain) or (j % 8 == 0):
            p(0.40 + 0.60 * (j / float(remain)), f"μ–σ refine {j}/{remain}")

    p(1.0, "seed ready")
    return np.clip(mu, 0.0, None).astype(np.float32, copy=False)


def _chunk(seq, n):
    """Yield chunks of size n from seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _read_shape_fast(path) -> tuple[int,int,int]:
    if _is_xisf(path):
        a, _ = _load_image_array(path)
        if a is None:
            raise ValueError(f"No data in {path}")
        a = np.asarray(a)
    else:
        with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
            a = hdul[0].data
            if a is None:
                raise ValueError(f"No data in {path}")

    # common logic for both XISF and FITS
    if a.ndim == 2:
        H, W = a.shape
        return (1, int(H), int(W))
    if a.ndim == 3:
        if a.shape[-1] in (1, 3):      # HWC
            C = int(a.shape[-1]); H = int(a.shape[0]); W = int(a.shape[1])
            return (1 if C == 1 else 3, H, W)
        if a.shape[0] in (1, 3):       # CHW
            return (int(a.shape[0]), int(a.shape[1]), int(a.shape[2]))
    s = tuple(map(int, a.shape))
    H, W = s[-2], s[-1]
    return (1, H, W)




def _tiles_of(hw: tuple[int,int], tile_hw: tuple[int,int], halo: int):
    """
    Yield tiles as dicts: {y0,y1,x0,x1,yc0,yc1,xc0,xc1}
    (outer region includes halo; core (yc0:yc1, xc0:xc1) excludes halo).
    """
    H, W = hw
    th, tw = tile_hw
    th = max(1, int(th)); tw = max(1, int(tw))
    for y in range(0, H, th):
        for x in range(0, W, tw):
            yc0 = y; yc1 = min(y + th, H)
            xc0 = x; xc1 = min(x + tw, W)
            y0  = max(0, yc0 - halo); y1 = min(H, yc1 + halo)
            x0  = max(0, xc0 - halo); x1 = min(W, xc1 + halo)
            yield dict(y0=y0, y1=y1, x0=x0, x1=x1, yc0=yc0, yc1=yc1, xc0=xc0, xc1=xc1)

def _extract_with_halo(a, tile):
    """
    Slice 'a' ((H,W) or (C,H,W)) to [y0:y1, x0:x1] with channel kept.
    """
    y0,y1,x0,x1 = tile["y0"], tile["y1"], tile["x0"], tile["x1"]
    if a.ndim == 2:
        return a[y0:y1, x0:x1]
    else:
        return a[:, y0:y1, x0:x1]

def _add_core(accum, tile_val, tile):
    """
    Add tile_val core into accum at (yc0:yc1, xc0:xc1).
    Shapes match (2D) or (C,H,W).
    """
    yc0,yc1,xc0,xc1 = tile["yc0"], tile["yc1"], tile["xc0"], tile["xc1"]
    if accum.ndim == 2:
        h0 = yc0 - tile["y0"]; h1 = h0 + (yc1 - yc0)
        w0 = xc0 - tile["x0"]; w1 = w0 + (xc1 - xc0)
        accum[yc0:yc1, xc0:xc1] += tile_val[h0:h1, w0:w1]
    else:
        h0 = yc0 - tile["y0"]; h1 = h0 + (yc1 - yc0)
        w0 = xc0 - tile["x0"]; w1 = w0 + (xc1 - xc0)
        accum[:, yc0:yc1, xc0:xc1] += tile_val[:, h0:h1, w0:w1]

def _prepare_np_fft_packs_batch(psfs, flip_psf, Hs, Ws):
    """Precompute rFFT packs on current grid for NumPy path; returns lists aligned to batch psfs."""
    Kfs, KTfs, meta = [], [], []
    import numpy.fft as fft
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        fftH, fftW = _fftshape_same(Hs, Ws, kh, kw)
        Kfs.append(fft.rfftn(np.fft.ifftshift(k),  s=(fftH, fftW)))
        KTfs.append(fft.rfftn(np.fft.ifftshift(kT), s=(fftH, fftW)))
        meta.append((kh, kw, fftH, fftW))
    return Kfs, KTfs, meta

def _prepare_torch_fft_packs_batch(psfs, flip_psf, Hs, Ws, device, dtype):
    """Torch FFT packs per PSF on current grid; mirrors your existing packer."""
    return _precompute_torch_psf_ffts(psfs, flip_psf, Hs, Ws, device, dtype)

def _as_chw(np_img: np.ndarray) -> np.ndarray:
    x = np.asarray(np_img, dtype=np.float32, order="C")
    if x.size == 0:
        raise RuntimeError(f"Empty image array after load; raw shape={np_img.shape}")
    if x.ndim == 2:
        return x[None, ...]  # 1,H,W
    if x.ndim == 3 and x.shape[0] in (1, 3):
        if x.shape[0] == 0:
            raise RuntimeError(f"Zero channels in CHW array; shape={x.shape}")
        return x
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        if x.shape[-1] == 0:
            raise RuntimeError(f"Zero channels in HWC array; shape={x.shape}")
        return np.moveaxis(x, -1, 0)
    # last resort: treat first dim as channels, but reject zero
    if x.shape[0] == 0:
        raise RuntimeError(f"Zero channels in array; shape={x.shape}")
    return x



def _conv_same_np_spatial(a: np.ndarray, k: np.ndarray, out: np.ndarray | None = None):
    try:
        import cv2
    except Exception:
        return None  # no opencv -> caller falls back to FFT

    # cv2 wants HxW single-channel float32
    kf = np.ascontiguousarray(k.astype(np.float32))
    kf = np.flip(np.flip(kf, 0), 1)  # OpenCV uses correlation; flip to emulate conv

    if a.ndim == 2:
        y = cv2.filter2D(a, -1, kf, borderType=cv2.BORDER_REFLECT)
        if out is None: return y
        out[...] = y; return out
    else:
        C, H, W = a.shape
        if out is None:
            out = np.empty_like(a)
        for c in range(C):
            out[c] = cv2.filter2D(a[c], -1, kf, borderType=cv2.BORDER_REFLECT)
        return out

def _grouped_conv_same_torch_per_sample(x_bc_hw, w_b1kk, B, C):
    F = torch.nn.functional
    x_bc_hw = x_bc_hw.to(memory_format=torch.contiguous_format).contiguous()
    w_b1kk  = w_b1kk.to(memory_format=torch.contiguous_format).contiguous()

    kh, kw = int(w_b1kk.shape[-2]), int(w_b1kk.shape[-1])
    pad = (kw // 2, kw - kw // 2 - 1,  kh // 2, kh - kh // 2 - 1)

    # unified path (CUDA/CPU/MPS): one grouped conv with G=B*C
    G = int(B * C)
    x_1ghw = x_bc_hw.reshape(1, G, x_bc_hw.shape[-2], x_bc_hw.shape[-1])
    x_1ghw = F.pad(x_1ghw, pad, mode="reflect")
    w_g1kk = w_b1kk.repeat_interleave(C, dim=0)        # (G,1,kh,kw)
    y_1ghw = F.conv2d(x_1ghw, w_g1kk, padding=0, groups=G)
    return y_1ghw.reshape(B, C, y_1ghw.shape[-2], y_1ghw.shape[-1]).contiguous()


# put near other small helpers
def _robust_med_mad_t(x, max_elems_per_sample: int = 2_000_000):
    """
    x: (B, C, H, W) tensor on device.
    Returns (median[B,1,1,1], mad[B,1,1,1]) computed on a strided subsample
    to avoid 'quantile() input tensor is too large'.
    """
    import math
    import torch
    B = x.shape[0]
    flat = x.reshape(B, -1)
    N = flat.shape[1]
    if N > max_elems_per_sample:
        stride = int(math.ceil(N / float(max_elems_per_sample)))
        flat = flat[:, ::stride]  # strided subsample
    med = torch.quantile(flat, 0.5, dim=1, keepdim=True)
    mad = torch.quantile((flat - med).abs(), 0.5, dim=1, keepdim=True) + 1e-6
    return med.view(B,1,1,1), mad.view(B,1,1,1)

def _torch_should_use_spatial(psf_ksize: int) -> bool:
    # Prefer spatial on non-CUDA backends and for modest kernels.
    try:
        dev = _torch_device()
        if dev.type in ("mps", "privateuseone"):  # privateuseone = DirectML
            return True
        if dev.type == "cuda":
            return psf_ksize <= 51  # typical PSF sizes; spatial is fast & stable
    except Exception:
        pass
    # Allow override via env
    import os as _os
    if _os.environ.get("MF_SPATIAL", "") == "1":
        return True
    return False

def _read_tile_fits(path: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Return a (H,W) or (H,W,3|1) tile via FITS memmap, without loading whole image."""
    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        hdu = hdul[0]
        a = hdu.data
        if a is None:
            # find first image HDU if primary is header-only
            for h in hdul[1:]:
                if getattr(h, "data", None) is not None:
                    a = h.data; break
        a = np.asarray(a)  # still lazy until sliced
        # squeeze trailing singleton if present to keep your conventions
        if a.ndim == 3 and a.shape[-1] == 1:
            a = np.squeeze(a, axis=-1)
        tile = a[y0:y1, x0:x1, ...]
        # copy so we own the buffer (we will cast/normalize)
        return np.array(tile, copy=True)

def _read_tile_fits_any(path: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """FITS/XISF-aware tile read: returns spatial tile; supports 2D, HWC, and CHW."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xisf":
        a = _xisf_cached_array(path)  # float32 memmap; cheap slicing
        # a is HW, HWC, or CHW (whatever _load_image_array returned)
        if a.ndim == 2:
            return np.array(a[y0:y1, x0:x1], copy=True)
        elif a.ndim == 3:
            if a.shape[-1] in (1, 3):            # HWC
                out = a[y0:y1, x0:x1, :]
                if out.shape[-1] == 1: out = out[..., 0]
                return np.array(out, copy=True)
            elif a.shape[0] in (1, 3):           # CHW
                out = a[:, y0:y1, x0:x1]
                if out.shape[0] == 1: out = out[0]
                return np.array(out, copy=True)
            else:
                raise ValueError(f"Unsupported XISF 3D shape {a.shape} in {path}")
        else:
            raise ValueError(f"Unsupported XISF ndim {a.ndim} in {path}")

    # FITS
    with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
        a = None
        for h in hdul:
            if getattr(h, "data", None) is not None:
                a = h.data
                break
        if a is None:
            raise ValueError(f"No image data in {path}")

        a = np.asarray(a)

        if a.ndim == 2:                           # HW
            return np.array(a[y0:y1, x0:x1], copy=True)

        if a.ndim == 3:
            if a.shape[0] in (1, 3):              # CHW (planes, rows, cols)
                out = a[:, y0:y1, x0:x1]
                if out.shape[0] == 1:
                    out = out[0]
                return np.array(out, copy=True)
            if a.shape[-1] in (1, 3):             # HWC
                out = a[y0:y1, x0:x1, :]
                if out.shape[-1] == 1:
                    out = out[..., 0]
                return np.array(out, copy=True)

        # Fallback: assume last two axes are spatial (…, H, W)
        try:
            out = a[(..., slice(y0, y1), slice(x0, x1))]
            return np.array(out, copy=True)
        except Exception:
            raise ValueError(f"Unsupported FITS data shape {a.shape} in {path}")

def _infer_channels_from_tile(p: str, Ht: int, Wt: int) -> int:
    """Look at a 1×1 tile to infer channel count; supports HW, HWC, CHW."""
    y1 = min(1, Ht); x1 = min(1, Wt)
    t = _read_tile_fits_any(p, 0, y1, 0, x1)

    if t.ndim == 2:
        return 1

    if t.ndim == 3:
        # Prefer the axis that actually carries the color planes
        ch_first  = t.shape[0]  in (1, 3)
        ch_last   = t.shape[-1] in (1, 3)

        if ch_first and not ch_last:
            return int(t.shape[0])
        if ch_last and not ch_first:
            return int(t.shape[-1])

        # Ambiguous tiny tile (e.g. CHW 3×1×1 or HWC 1×1×3):
        if t.shape[0] == 3 or t.shape[-1] == 3:
            return 3
        return 1

    return 1



def _seed_median_streaming(
    paths,
    Ht,
    Wt,
    *,
    color_mode="luma",
    tile_hw=(256, 256),
    status_cb=lambda s: None,
    progress_cb=lambda f, m="": None,
    use_torch: bool | None = None,     # auto by default
):
    """
    Exact per-pixel median via tiling; RAM-bounded.
    Now shows per-tile progress and uses Torch on GPU if available.
    Parallelizes per-tile slab reads to hide I/O and luma work.
    """

    th, tw = int(tile_hw[0]), int(tile_hw[1])
    # old: want_c = 1 if str(color_mode).lower() == "luma" else (3 if _read_shape_fast(paths[0])[0] == 3 else 1)
    if str(color_mode).lower() == "luma":
        want_c = 1
    else:
        want_c = _infer_channels_from_tile(paths[0], Ht, Wt)
    seed     = np.zeros((Ht, Wt), np.float32) if want_c == 1 else np.zeros((want_c, Ht, Wt), np.float32)
    tiles    = [(y, min(y + th, Ht), x, min(x + tw, Wt)) for y in range(0, Ht, th) for x in range(0, Wt, tw)]
    total    = len(tiles)
    n_frames = len(paths)

    # Choose a sensible number of I/O workers (bounded by frames)
    try:
        _cpu = (os.cpu_count() or 4)
    except Exception:
        _cpu = 4
    io_workers = max(1, min(8, _cpu, n_frames))

    # Torch autodetect (once)
    TORCH_OK = False
    device = None
    if use_torch is not False:
        try:
            from pro.runtime_torch import import_torch
            _t = import_torch(prefer_cuda=True, status_cb=status_cb)
            dev = None
            if hasattr(_t, "cuda") and _t.cuda.is_available():
                dev = _t.device("cuda")
            elif hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
                dev = _t.device("mps")
            else:
                dev = None  # CPU tensors slower than NumPy for median; only use if forced
            if dev is not None:
                TORCH_OK = True
                device = dev
                status_cb(f"Median seed: using Torch device {device}")
        except Exception as e:
            status_cb(f"Median seed: Torch unavailable → NumPy fallback ({e})")
            TORCH_OK = False
            device = None

    def _tile_msg(ti, tn):
        return f"median tiles {ti}/{tn}"

    done = 0

    for (y0, y1, x0, x1) in tiles:
        h, w = (y1 - y0), (x1 - x0)

        # per-tile slab reader with incremental progress (parallel)
        def _read_slab_for_channel(csel=None):
            """
            Returns slab of shape (N, h, w) float32 in [0,1]-ish (normalized if input was integer).
            If csel is None and luma is requested, computes luma.
            """
            # parallel worker returns (i, tile2d)
            def _load_one(i):
                t = _read_tile_fits_any(paths[i], y0, y1, x0, x1)

                # normalize dtype
                if t.dtype.kind in "ui":
                    t = t.astype(np.float32) / (float(np.iinfo(t.dtype).max) or 1.0)
                else:
                    t = t.astype(np.float32, copy=False)

                # luma / channel selection
                if want_c == 1:
                    if t.ndim == 3:
                        t = _to_luma_local(t)
                    elif t.ndim != 2:
                        t = _to_luma_local(t)
                else:
                    if t.ndim == 2:
                        pass
                    elif t.ndim == 3 and t.shape[-1] == 3:  # HWC
                        t = t[..., csel]
                    elif t.ndim == 3 and t.shape[0] == 3:   # CHW
                        t = t[csel]
                    else:
                        t = _to_luma_local(t)
                return i, np.ascontiguousarray(t, dtype=np.float32)

            slab = np.empty((n_frames, h, w), np.float32)
            done_local = 0
            # cap workers by n_frames so we don't spawn useless threads
            with ThreadPoolExecutor(max_workers=min(io_workers, n_frames)) as ex:
                futures = [ex.submit(_load_one, i) for i in range(n_frames)]
                for fut in as_completed(futures):
                    i, t2d = fut.result()
                    # quick sanity (avoid silent mis-shapes)
                    if t2d.shape != (h, w):
                        raise RuntimeError(
                            f"Tile read mismatch at frame {i}: got {t2d.shape}, expected {(h, w)} "
                            f"tile={(y0,y1,x0,x1)}"
                        )
                    slab[i] = t2d
                    done_local += 1
                    if (done_local & 7) == 0 or done_local == n_frames:
                        tile_base = done / total
                        tile_span = 1.0 / total
                        inner     = done_local / n_frames
                        progress_cb(tile_base + 0.8 * tile_span * inner, _tile_msg(done + 1, total))
            return slab

        try:
            if want_c == 1:
                t0 = time.perf_counter()
                slab = _read_slab_for_channel()
                t1 = time.perf_counter()
                if TORCH_OK:
                    import torch as _t
                    slab_t = _t.as_tensor(slab, device=device, dtype=_t.float32)  # one H2D
                    med_t  = slab_t.median(dim=0).values
                    med_np = med_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    # no per-tile empty_cache() — avoid forced syncs
                else:
                    med_np = np.median(slab, axis=0).astype(np.float32, copy=False)
                t2 = time.perf_counter()
                seed[y0:y1, x0:x1] = med_np
                # lightweight telemetry to confirm bottleneck
                status_cb(f"seed tile {y0}:{y1},{x0}:{x1} I/O={t1-t0:.3f}s  median={'GPU' if TORCH_OK else 'CPU'}={t2-t1:.3f}s")
            else:
                for c in range(want_c):
                    slab = _read_slab_for_channel(csel=c)
                    if TORCH_OK:
                        import torch as _t
                        slab_t = _t.as_tensor(slab, device=device, dtype=_t.float32)
                        med_t  = slab_t.median(dim=0).values
                        med_np = med_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    else:
                        med_np = np.median(slab, axis=0).astype(np.float32, copy=False)
                    seed[c, y0:y1, x0:x1] = med_np

        except RuntimeError as e:
            # Per-tile GPU OOM or device issues → fallback to NumPy for this tile
            msg = str(e).lower()
            if TORCH_OK and ("out of memory" in msg or "resource" in msg or "alloc" in msg):
                status_cb(f"Median seed: GPU OOM on tile ({h}x{w}); falling back to NumPy for this tile.")
                if want_c == 1:
                    slab = _read_slab_for_channel()
                    seed[y0:y1, x0:x1] = np.median(slab, axis=0).astype(np.float32, copy=False)
                else:
                    for c in range(want_c):
                        slab = _read_slab_for_channel(csel=c)
                        seed[c, y0:y1, x0:x1] = np.median(slab, axis=0).astype(np.float32, copy=False)
            else:
                raise

        done += 1
        # report tile completion (remaining 20% of tile span reserved for median compute)
        progress_cb(done / total, _tile_msg(done, total))

        if (done & 3) == 0:
            _process_gui_events_safely()
    status_cb(f"Median seed: want_c={want_c}, seed_shape={seed.shape}")
    return seed

def _seed_bootstrap_streaming(paths, Ht, Wt, color_mode,
                              bootstrap_frames: int = 20,
                              clip_sigma: float = 5.0,
                              status_cb=lambda s: None,
                              progress_cb=None):
    """
    Seed = average first B frames, estimate global MAD threshold, masked-mean on B,
    then stream the remaining frames with σ-clipped Welford μ–σ updates.

    Returns float32, CHW for per-channel mode, or CHW(1,...) for luma (your caller later squeezes if needed).
    """
    def p(frac, msg):
        if progress_cb:
            progress_cb(float(max(0.0, min(1.0, frac))), msg)

    n = len(paths)
    B = int(max(1, min(int(bootstrap_frames), n)))
    status_cb(f"Seed: bootstrap={B}, clip_sigma={clip_sigma}")

    # ---------- pass 1: running mean over the first B frames (Welford μ only) ----------
    cnt = None
    mu  = None
    for i, pth in enumerate(paths[:B], 1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        x = ys[0].astype(np.float32, copy=False)
        if mu is None:
            mu  = x.copy()
            cnt = np.ones_like(x, dtype=np.float32)
        else:
            delta = x - mu
            cnt  += 1.0
            mu   += delta / cnt
        if (i == B) or (i % 4) == 0:
            p(0.20 * (i / float(B)), f"bootstrap mean {i}/{B}")

    # ---------- pass 2: estimate a *global* MAD around μ using strided samples ----------
    stride = 4 if max(Ht, Wt) > 1500 else 2
    # CHW or HW → build a slice that keeps C and strides H,W
    samp_slices = (slice(None) if mu.ndim == 3 else slice(None),
                   slice(None, None, stride),
                   slice(None, None, stride)) if mu.ndim == 3 else \
                  (slice(None, None, stride), slice(None, None, stride))

    mad_samples = []
    for i, pth in enumerate(paths[:B], 1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        x = ys[0].astype(np.float32, copy=False)
        d = np.abs(x - mu)
        mad_samples.append(d[samp_slices].ravel())
        if (i == B) or (i % 8) == 0:
            p(0.35 + 0.10 * (i / float(B)), f"bootstrap MAD {i}/{B}")

    # robust MAD estimate (scalar) → 4·MAD clip band
    mad_est = float(np.median(np.concatenate(mad_samples).astype(np.float32)))
    thr = 4.0 * max(mad_est, 1e-6)

    # ---------- pass 3: masked mean over first B frames using the global threshold ----------
    sum_acc = np.zeros_like(mu, dtype=np.float32)
    cnt_acc = np.zeros_like(mu, dtype=np.float32)
    for i, pth in enumerate(paths[:B], 1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        x = ys[0].astype(np.float32, copy=False)
        m = (np.abs(x - mu) <= thr).astype(np.float32, copy=False)
        sum_acc += x * m
        cnt_acc += m
        if (i == B) or (i % 4) == 0:
            p(0.48 + 0.12 * (i / float(B)), f"masked mean {i}/{B}")

    seed = mu.copy()
    np.divide(sum_acc, np.maximum(cnt_acc, 1.0), out=seed, where=(cnt_acc > 0.5))

    # ---------- pass 4: μ–σ streaming on the remaining frames (σ-clipped Welford) ----------
    M2  = np.zeros_like(seed, dtype=np.float32)     # sum of squared diffs
    cnt = np.full_like(seed, float(B), dtype=np.float32)
    mu  = seed.astype(np.float32, copy=False)

    remain = n - B
    k = float(clip_sigma)
    for j, pth in enumerate(paths[B:], 1):
        ys, _ = _stack_loader_memmap([pth], Ht, Wt, color_mode)
        x = ys[0].astype(np.float32, copy=False)

        var   = M2 / np.maximum(cnt - 1.0, 1.0)
        sigma = np.sqrt(np.maximum(var, 1e-12, dtype=np.float32))

        acc   = (np.abs(x - mu) <= (k * sigma)).astype(np.float32, copy=False)  # {0,1} mask
        n_new = cnt + acc
        delta = x - mu
        mu_n  = mu + (acc * delta) / np.maximum(n_new, 1.0)
        M2    = M2 + acc * delta * (x - mu_n)

        mu, cnt = mu_n, n_new

        if (j == remain) or (j % 8) == 0:
            p(0.60 + 0.40 * (j / float(max(1, remain))), f"μ–σ refine {j}/{remain}")

    return np.clip(mu, 0.0, None).astype(np.float32, copy=False)


def _coerce_sr_factor(srf, *, default_on_bad=2):
    """
    Parse super-res factor robustly:
    - accepts 2, '2', '2x', ' 2 X ', 2.0
    - clamps to integers >= 1
    - if invalid/missing → returns default_on_bad (we want 2 by your request)
    """
    if srf is None:
        return int(default_on_bad)
    if isinstance(srf, (float, int)):
        r = int(round(float(srf)))
        return int(r if r >= 1 else default_on_bad)
    s = str(srf).strip().lower()
    # common GUIs pass e.g. "2x", "3×", etc.
    s = s.replace("×", "x")
    if s.endswith("x"):
        s = s[:-1]
    try:
        r = int(round(float(s)))
        return int(r if r >= 1 else default_on_bad)
    except Exception:
        return int(default_on_bad)

def _pad_kernel_to(k: np.ndarray, K: int) -> np.ndarray:
    """Pad/center an odd-sized kernel to K×K (K odd)."""
    k = np.asarray(k, dtype=np.float32)
    kh, kw = int(k.shape[0]), int(k.shape[1])
    assert (kh % 2 == 1) and (kw % 2 == 1)
    if kh == K and kw == K:
        return k
    out = np.zeros((K, K), dtype=np.float32)
    y0 = (K - kh)//2; x0 = (K - kw)//2
    out[y0:y0+kh, x0:x0+kw] = k
    s = float(out.sum())
    return out if s <= 0 else (out / s).astype(np.float32, copy=False)

# -----------------------------
# Core
# -----------------------------
def multiframe_deconv(
    paths,
    out_path,
    iters=20,
    kappa=2.0,
    color_mode="luma",
    seed_mode: str = "robust",
    huber_delta=0.0,
    masks=None,
    variances=None,
    rho="huber",
    status_cb=lambda s: None,
    min_iters: int = 3,
    use_star_masks: bool = False,
    use_variance_maps: bool = False,
    star_mask_cfg: dict | None = None,
    varmap_cfg: dict | None = None,
    save_intermediate: bool = False,
    save_every: int = 1,
    # SR options
    super_res_factor: int = 1,
    sr_sigma: float = 1.1,
    sr_psf_opt_iters: int = 250,
    sr_psf_opt_lr: float = 0.1,
    # NEW
    batch_frames: int | None = None,
    # GPU tuning (optional knobs)
    mixed_precision: bool | None = None,      # default: auto (True on CUDA/MPS)
    fft_kernel_threshold: int = 1024,           # switch to FFT if K >= this (or lower if SR)
    prefetch_batches: bool = True,            # CPU→GPU double-buffer prefetch
    use_channels_last: bool | None = None,    # default: auto (True on CUDA/MPS)
    force_cpu: bool = False,
    star_mask_ref_path: str | None = None,     
    low_mem: bool = False,
):
    """
    Streaming multi-frame deconvolution with optional SR (r>1).
    Optimized GPU path: AMP for convs, channels-last, pinned-memory prefetch, optional FFT for large kernels.
    """
    mixed_precision = False
    DEBUG_FLAT_WEIGHTS = False 
    # ---------- local helpers (kept self-contained) ----------
    def _emit_pct(pct: float, msg: str | None = None):
        pct = float(max(0.0, min(1.0, pct)))
        status_cb(f"__PROGRESS__ {pct:.4f}" + (f" {msg}" if msg else ""))

    def _pad_kernel_to(k: np.ndarray, K: int) -> np.ndarray:
        """Pad/center an odd-sized kernel to K×K (K odd)."""
        k = np.asarray(k, dtype=np.float32)
        kh, kw = int(k.shape[0]), int(k.shape[1])
        assert (kh % 2 == 1) and (kw % 2 == 1)
        if kh == K and kw == K:
            return k
        out = np.zeros((K, K), dtype=np.float32)
        y0 = (K - kh)//2; x0 = (K - kw)//2
        out[y0:y0+kh, x0:x0+kw] = k
        s = float(out.sum())
        return out if s <= 0 else (out / s).astype(np.float32, copy=False)

    max_iters = max(1, int(iters))
    min_iters = max(1, int(min_iters))
    if min_iters > max_iters:
        min_iters = max_iters

    n_frames = len(paths)
    status_cb(f"MFDeconv: scanning {n_frames} aligned frames (memmap)…")
    _emit_pct(0.02, "scanning")

    # choose common intersection size without loading full pixels
    Ht, Wt = _common_hw_from_paths(paths)
    _emit_pct(0.05, "preparing")

    # --- LOW-MEM PATCH (begin) ---
    if low_mem:
        # Cap decoded-frame LRU to keep peak RAM sane on 16 GB laptops
        try:
            _FRAME_LRU.cap = max(1, min(getattr(_FRAME_LRU, "cap", 8), 2))
        except Exception:
            pass

        # Disable CPU→GPU prefetch to avoid double-buffering allocations
        prefetch_batches = False

        # Relax SEP background grid & star detection canvas when requested
        if use_variance_maps:
            varmap_cfg = {**(varmap_cfg or {})}
            # fewer, larger tiles → fewer big temporaries
            varmap_cfg.setdefault("bw", 96)
            varmap_cfg.setdefault("bh", 96)

        if use_star_masks:
            star_mask_cfg = {**(star_mask_cfg or {})}
            # shrink detection canvas to limit temp buffers inside SEP/mask draw
            star_mask_cfg["max_side"] = int(min(1024, int(star_mask_cfg.get("max_side", 2048))))
    # --- LOW-MEM PATCH (end) ---


    if any(os.path.splitext(p)[1].lower() == ".xisf" for p in paths):
        status_cb("MFDeconv: priming XISF cache (one-time decode per frame)…")
        for i, p in enumerate(paths, 1):
            try:
                _ = _xisf_cached_array(p)  # decode once, store memmap
            except Exception as e:
                status_cb(f"XISF cache failed for {p}: {e}")
            if (i & 7) == 0 or i == len(paths):
                _process_gui_events_safely()

    # per-frame loader & sequence view (closures capture Ht/Wt/color_mode/paths)
    def _load_frame_chw(i: int):
        return _FRAME_LRU.get(paths[i], Ht, Wt, color_mode)

    class _FrameSeq:
        def __len__(self): return len(paths)
        def __getitem__(self, i): return _load_frame_chw(i)
    data = _FrameSeq()

    # ---- torch detection (optional) ----
    global torch, TORCH_OK
    torch = None
    TORCH_OK = False
    cuda_ok = mps_ok = dml_ok = False
    dml_device = None

    try:
        from pro.runtime_torch import import_torch
        torch = import_torch(prefer_cuda=True, status_cb=status_cb)
        TORCH_OK = True
        try: cuda_ok = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try: mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            import torch_directml
            dml_device = torch_directml.device()
            _ = (torch.ones(1, device=dml_device) + 1).item()
            dml_ok = True
            # NEW: expose to _torch_device()
            globals()["dml_ok"] = True
            globals()["dml_device"] = dml_device
        except Exception:
            dml_ok = False
            globals()["dml_ok"] = False
            globals()["dml_device"] = None

        if cuda_ok:
            status_cb(f"PyTorch CUDA available: True | device={torch.cuda.get_device_name(0)}")
        elif mps_ok:
            status_cb("PyTorch MPS (Apple) available: True")
        elif dml_ok:
            status_cb("PyTorch DirectML (Windows) available: True")
        else:
            status_cb("PyTorch present, using CPU backend.")
        status_cb(
            f"PyTorch {getattr(torch,'__version__','?')} backend: "
            + ("CUDA" if cuda_ok else "MPS" if mps_ok else "DirectML" if dml_ok else "CPU")
        )

        try:
            # keep cuDNN autotune on
            if getattr(torch.backends, "cudnn", None) is not None:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass

        try:
            # disable TF32 matmul shortcuts if present (CUDA-only; safe no-op elsewhere)
            if getattr(getattr(torch.backends, "cuda", None), "matmul", None) is not None:
                torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass

        try:
            # prefer highest FP32 precision on PT 2.x
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    except Exception as e:
        TORCH_OK = False
        status_cb(f"PyTorch not available → CPU path. ({e})")

    if force_cpu:
        status_cb("⚠️ CPU-only debug mode: disabling PyTorch path.")
        TORCH_OK = False

    _process_gui_events_safely()

    # ---- PSFs + optional assets (computed in parallel, streaming I/O) ----
    psfs, masks_auto, vars_auto = _build_psf_and_assets(
        paths,
        make_masks=bool(use_star_masks),
        make_varmaps=bool(use_variance_maps),
        status_cb=status_cb,
        save_dir=None,
        star_mask_cfg=star_mask_cfg,
        varmap_cfg=varmap_cfg,
        star_mask_ref_path=star_mask_ref_path,
        # NEW:
        Ht=Ht, Wt=Wt, color_mode=color_mode,
    )

    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass

    psfs_native = psfs  # keep a reference to the original native PSFs for fallback

    # ---- SR lift of PSFs if needed ----
    r_req = _coerce_sr_factor(super_res_factor, default_on_bad=2)
    status_cb(f"MFDeconv: SR factor requested={super_res_factor!r} → using r={r_req}")
    r = int(r_req)

    if r > 1:
        status_cb(f"MFDeconv: Super-resolution r={r} with σ={sr_sigma} — solving SR PSFs…")
        _process_gui_events_safely()

        def _naive_sr_from_native(f_nat: np.ndarray, r_: int) -> np.ndarray:
            f2 = np.asarray(f_nat, np.float32)
            H, W = f2.shape[:2]
            k_sq = min(H, W)
            if H != W:
                y0 = (H - k_sq) // 2; x0 = (W - k_sq) // 2
                f2 = f2[y0:y0+k_sq, x0:x0+k_sq]
            if (f2.shape[0] % 2) == 0:
                f2 = f2[1:, 1:]  # make odd
            f2 = _normalize_psf(f2)
            h0 = np.zeros((f2.shape[0]*r_, f2.shape[1]*r_), np.float32)
            h0[::r_, ::r_] = f2
            return _normalize_psf(h0)

        sr_psfs = []
        for i, k_native in enumerate(psfs, start=1):
            try:
                status_cb(f"  SR-PSF{i}: native shape={np.asarray(k_native).shape}")
                h = _solve_super_psf_from_native(
                    k_native, r=r, sigma=float(sr_sigma),
                    iters=int(sr_psf_opt_iters), lr=float(sr_psf_opt_lr)
                )
            except Exception as e:
                status_cb(f"  SR-PSF{i} failed: {e!r} → using naïve upsample")
                h = _naive_sr_from_native(k_native, r)

            # guarantee odd size for downstream SAME-padding math
            if (h.shape[0] % 2) == 0:
                h = h[:-1, :-1]
            sr_psfs.append(h.astype(np.float32, copy=False))
            status_cb(f"  SR-PSF{i}: native {np.asarray(k_native).shape[0]} → {h.shape[0]} (sum={h.sum():.6f})")
        psfs = sr_psfs



    # ---- Seed (streaming) with robust bootstrap already in file helpers ----
    _emit_pct(0.25, "Calculating Seed Image...")
    def _seed_progress(frac, msg):
        _emit_pct(0.25 + 0.15 * float(frac), f"seed: {msg}")

    seed_mode_s = str(seed_mode).lower().strip()
    if seed_mode_s not in ("robust", "median"):
        seed_mode_s = "robust"
    if seed_mode_s == "median":
        status_cb("MFDeconv: Building median seed (tiled, streaming)…")
        seed_native = _seed_median_streaming(
            paths, Ht, Wt,
            color_mode=color_mode,
            tile_hw=(256, 256),
            status_cb=status_cb,
            progress_cb=_seed_progress,
            use_torch=TORCH_OK,   # ← auto: GPU if available, else NumPy
        )
    else:
        seed_native = _seed_bootstrap_streaming(
            paths, Ht, Wt, color_mode,
            bootstrap_frames=20, clip_sigma=5,
            status_cb=status_cb, progress_cb=_seed_progress
        )

    # lift seed if SR
    if r > 1:
        target_hw = (Ht * r, Wt * r)
        if seed_native.ndim == 2:
            x = _upsample_sum(seed_native / (r*r), r, target_hw=target_hw)
        else:
            C, Hn, Wn = seed_native.shape
            x = np.stack(
                [_upsample_sum(seed_native[c] / (r*r), r, target_hw=target_hw) for c in range(C)],
                axis=0
            )
    else:
        x = seed_native

    # FINAL SHAPE CHECKS (auto-correct if a GUI sent something odd)
    if x.ndim == 2: x = x[None, ...]
    Hs, Ws = x.shape[-2], x.shape[-1]
    if r > 1:
        expected_H, expected_W = Ht * r, Wt * r
        if (Hs, Ws) != (expected_H, expected_W):
            status_cb(f"SR seed grid mismatch: got {(Hs, Ws)}, expected {(expected_H, expected_W)} → correcting")
            # Rebuild from the native mean to ensure exact SR size
            x = _upsample_sum(x if x.ndim==2 else x[0], r, target_hw=(expected_H, expected_W))
            if x.ndim == 2: x = x[None, ...]
    try: del seed_native
    except Exception as e:
        import logging
        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    try:
        import gc as _gc
        _gc.collect()
    except Exception:
        pass

    flip_psf = [_flip_kernel(k) for k in psfs]
    _emit_pct(0.20, "PSF Ready")

    # --- Harmonize seed channels with the actual frames ---
    # Probe the first frame's channel count (CHW from the cache)
    try:
        y0_probe = _load_frame_chw(0)               # CHW float32
        C_ref = 1 if y0_probe.ndim == 2 else int(y0_probe.shape[0])
    except Exception:
        # Fallback: infer from seed if probing fails
        C_ref = 1 if x.ndim == 2 else int(x.shape[0])

    # If median seed came back mono but frames are RGB, broadcast it
    if x.ndim == 2 and C_ref == 3:
        x = np.stack([x] * 3, axis=0)
    elif x.ndim == 3 and x.shape[0] == 1 and C_ref == 3:
        x = np.repeat(x, 3, axis=0)
    # If seed is RGB but frames are mono (rare), collapse to luma
    elif x.ndim == 3 and x.shape[0] == 3 and C_ref == 1:
        # ITU-R BT.709 luma (safe in float)
        x = (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).astype(np.float32)

    # Ensure CHW shape for the rest of the pipeline
    if x.ndim == 2:
        x = x[None, ...]
    elif x.ndim == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
        x = np.moveaxis(x, -1, 0)

    # Set the expected channel count from the FRAMES, not from the seed
    C_EXPECTED = int(C_ref)
    _, Hs, Ws = x.shape

    

    # ---- choose default batch size ----
    if batch_frames is None:
        px = Hs * Ws
        if px >= 16_000_000:     auto_B = 2
        elif px >= 8_000_000:    auto_B = 4
        else:                    auto_B = 8
    else:
        auto_B = int(max(1, batch_frames))

    # --- LOW-MEM PATCH: clamp batch size hard ---
    if low_mem:
        auto_B = max(1, min(auto_B, 2))

    # ---- background/MAD telemetry (first frame) ----
    status_cb("MFDeconv: Calculating Backgrounds and MADs…")
    _process_gui_events_safely()
    try:
        y0 = data[0]; y0l = y0 if y0.ndim == 2 else y0[0]
        med = float(np.median(y0l)); mad = float(np.median(np.abs(y0l - med))) + 1e-6
        bg_est = 1.4826 * mad
    except Exception:
        bg_est = 0.0
    status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g})")

    # ---- mask/variance accessors ----
    def _mask_for(i, like_img):
        src = (masks if masks is not None else masks_auto)
        if src is None:  # no masks at all
            return np.ones((like_img.shape[-2], like_img.shape[-1]), dtype=np.float32)
        m = src[i]
        if m is None:
            return np.ones((like_img.shape[-2], like_img.shape[-1]), dtype=np.float32)
        m = np.asarray(m, dtype=np.float32)
        if m.ndim == 3: m = m[0]
        return _center_crop(m, like_img.shape[-2], like_img.shape[-1]).astype(np.float32, copy=False)

    def _var_for(i, like_img):
        src = (variances if variances is not None else vars_auto)
        if src is None: return None
        v = src[i]
        if v is None: return None
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 3: v = v[0]
        v = _center_crop(v, like_img.shape[-2], like_img.shape[-1])
        return np.clip(v, 1e-8, None).astype(np.float32, copy=False)

    # ---- NumPy path conv helper (keep as-is) ----
    def _conv_np_same(a, k, out=None):
        y = _conv_same_np_spatial(a, k, out)
        if y is not None:
            return y
        # No OpenCV → always use the ifftshifted FFT path
        import numpy as _np
        import numpy.fft as _fft
        H, W = a.shape[-2:]
        kh, kw = k.shape
        fftH, fftW = _fftshape_same(H, W, kh, kw)
        Kf = _fft.rfftn(_np.fft.ifftshift(k), s=(fftH, fftW))
        if out is None:
            out = _np.empty_like(a, dtype=_np.float32)
        return _fft_conv_same_np(a, Kf, kh, kw, fftH, fftW, out)

    # ---- allocate scratch & prepare PSF tensors if torch ----
    relax = 0.7
    use_torch = bool(TORCH_OK)
    cm = _safe_inference_context() if use_torch else NO_GRAD
    rho_is_l2 = (str(rho).lower() == "l2")
    local_delta = 0.0 if rho_is_l2 else huber_delta

    if use_torch:
        F = torch.nn.functional
        device = _torch_device()

        # Force FP32 tensors
        x_t  = _to_t(_contig(x)).to(torch.float32)
        num  = torch.zeros_like(x_t, dtype=torch.float32)
        den  = torch.zeros_like(x_t, dtype=torch.float32)

        # channels-last preference kept (does not change dtype)
        if use_channels_last is None:
            use_channels_last = bool(cuda_ok)   # <- never enable on MPS
        if mps_ok:
            use_channels_last = False           # <- force NCHW on MPS

        # PSF tensors strictly FP32
        psf_t  = [_to_t(_contig(k))[None, None]  for k  in psfs]     # (1,1,kh,kw)
        psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]
        use_spatial = _torch_should_use_spatial(psf_t[0].shape[-1])

        # No mixed precision, no autocast
        use_amp = False
        amp_cm = None
        amp_kwargs = {}

        # FFT gate (worth it for large kernels / SR); still FP32
        # Decide FFT vs spatial FIRST, based on native kernel sizes (no global padding!)
        K_native_max = max(int(k.shape[0]) for k in psfs)
        use_fft = False
        if device.type == "cuda" and ((K_native_max >= int(fft_kernel_threshold)) or (r > 1 and K_native_max >= max(21, int(fft_kernel_threshold) - 4))):
            use_fft = True
            # Force NCHW for FFT branch to keep channel slices contiguous/sane
            use_channels_last = False
            # Precompute FFT packs…
            psf_fft, psfT_fft = _precompute_torch_psf_ffts(
                psfs, flip_psf, Hs, Ws, device=x_t.device, dtype=torch.float32
            )
        else:
            psf_fft = psfT_fft = None
            Kmax = K_native_max
            if (Kmax % 2) == 0:
                Kmax += 1
            if any(int(k.shape[0]) != Kmax for k in psfs):
                status_cb(f"MFDeconv: normalizing PSF sizes → {Kmax}×{Kmax}")
                psfs = [_pad_kernel_to(k, Kmax) for k in psfs]
                flip_psf = [_flip_kernel(k) for k in psfs]

        # (Re)build spatial kernels strictly as contiguous FP32 tensors
        psf_t  = [_to_t(_contig(k))[None, None].to(torch.float32).contiguous()  for k  in psfs]
        psfT_t = [_to_t(_contig(kT))[None, None].to(torch.float32).contiguous() for kT in flip_psf]
    else:
        x_t = _contig(x).astype(np.float32, copy=False)
        num = np.zeros_like(x_t, dtype=np.float32)
        den = np.zeros_like(x_t, dtype=np.float32)
        use_amp = False
        use_fft = False

    # ---- batched torch helper (grouped depthwise per-sample) ----
    if use_torch:
        # inside `if use_torch:` block in multiframe_deconv — replace the whole inner helper
        def _grouped_conv_same_torch_per_sample(x_bc_hw, w_b1kk, B, C):
            """
            x_bc_hw : (B,C,H,W), torch.float32 on device
            w_b1kk  : (B,1,kh,kw), torch.float32 on device
            Returns (B,C,H,W) contiguous (NCHW).
            """
            F = torch.nn.functional

            # Force standard NCHW contiguous tensors
            x_bc_hw = x_bc_hw.to(memory_format=torch.contiguous_format).contiguous()
            w_b1kk  = w_b1kk.to(memory_format=torch.contiguous_format).contiguous()

            kh, kw = int(w_b1kk.shape[-2]), int(w_b1kk.shape[-1])
            pad = (kw // 2, kw - kw // 2 - 1,  kh // 2, kh - kh // 2 - 1)

            if x_bc_hw.device.type == "mps":
                # Safe, slower path: convolve each channel separately, no groups
                ys = []
                for j in range(B):                      # per sample
                    xj = x_bc_hw[j:j+1]                # (1,C,H,W)
                    # reflect pad once per sample
                    xj = F.pad(xj, pad, mode="reflect")
                    cj_out = []
                    # one shared kernel per sample j: (1,1,kh,kw)
                    kj = w_b1kk[j:j+1]                 # keep shape (1,1,kh,kw)
                    for c in range(C):
                        # slice that channel as its own (1,1,H,W) tensor
                        xjc = xj[:, c:c+1, ...]
                        yjc = F.conv2d(xjc, kj, padding=0, groups=1)  # no groups
                        cj_out.append(yjc)
                    ys.append(torch.cat(cj_out, dim=1))  # (1,C,H,W)
                return torch.stack([y[0] for y in ys], 0).contiguous()


            # ---- FAST PATH (CUDA/CPU): single grouped conv with G=B*C ----
            G = int(B * C)
            x_1ghw = x_bc_hw.reshape(1, G, x_bc_hw.shape[-2], x_bc_hw.shape[-1])
            x_1ghw = F.pad(x_1ghw, pad, mode="reflect")
            w_g1kk = w_b1kk.repeat_interleave(C, dim=0)        # (G,1,kh,kw)
            y_1ghw = F.conv2d(x_1ghw, w_g1kk, padding=0, groups=G)
            return y_1ghw.reshape(B, C, y_1ghw.shape[-2], y_1ghw.shape[-1]).contiguous()


        def _downsample_avg_bt_t(x, r_):
            if r_ <= 1:
                return x
            B, C, H, W = x.shape
            Hr, Wr = (H // r_) * r_, (W // r_) * r_
            if Hr == 0 or Wr == 0:
                return x
            return x[:, :, :Hr, :Wr].reshape(B, C, Hr // r_, r_, Wr // r_, r_).mean(dim=(3, 5))

        def _upsample_sum_bt_t(x, r_):
            if r_ <= 1:
                return x
            return x.repeat_interleave(r_, dim=-2).repeat_interleave(r_, dim=-1)

    def _make_pinned_batch(idx, C_expected, to_device_dtype):
        y_list, m_list, v_list = [], [], []
        for fi in idx:
            y_chw = _load_frame_chw(fi)  # CHW float32 from cache
            if y_chw.ndim != 3:
                raise RuntimeError(f"Frame {fi}: expected CHW, got {tuple(y_chw.shape)}")
            C_here = int(y_chw.shape[0])
            if C_expected is not None and C_here != C_expected:
                raise RuntimeError(f"Mixed channel counts: expected C={C_expected}, got C={C_here} (frame {fi})")

            m2d = _mask_for(fi, y_chw)
            v2d = _var_for(fi, y_chw)
            y_list.append(y_chw); m_list.append(m2d); v_list.append(v2d)

        # CPU (NCHW) tensors
        y_cpu = torch.from_numpy(np.stack(y_list, 0)).to(torch.float32).contiguous()
        m_cpu = torch.from_numpy(np.stack(m_list, 0)).to(torch.float32).contiguous()
        have_v = all(v is not None for v in v_list)
        vb_cpu = None if not have_v else torch.from_numpy(np.stack(v_list, 0)).to(torch.float32).contiguous()

        # optional pin for faster H2D on CUDA
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            y_cpu = y_cpu.pin_memory(); m_cpu = m_cpu.pin_memory()
            if vb_cpu is not None: vb_cpu = vb_cpu.pin_memory()

        # move to device in FP32, keep NCHW contiguous format
        yb = y_cpu.to(x_t.device, dtype=torch.float32, non_blocking=True).contiguous()
        mb = m_cpu.to(x_t.device, dtype=torch.float32, non_blocking=True).contiguous()
        vb = None if vb_cpu is None else vb_cpu.to(x_t.device, dtype=torch.float32, non_blocking=True).contiguous()

        return yb, mb, vb



    # ---- intermediates folder ----
    iter_dir = None
    hdr0_seed = None
    if save_intermediate:
        iter_dir = _iter_folder(out_path)
        status_cb(f"MFDeconv: Intermediate outputs → {iter_dir}")
        try:
            hdr0 = _safe_primary_header(paths[0])
        except Exception:
            hdr0 = fits.Header()
        _save_iter_image(x, hdr0, iter_dir, "seed", color_mode)

    # ---- iterative loop ----



    auto_delta_cache = None
    if use_torch and (huber_delta < 0) and (not rho_is_l2):
        auto_delta_cache = [None] * n_frames
    early = EarlyStopper(
        tol_upd_floor=2e-4,
        tol_rel_floor=5e-4,
        early_frac=0.40,
        ema_alpha=0.5,
        patience=2,
        min_iters=min_iters,
        status_cb=status_cb
    )

    used_iters = 0
    early_stopped = False
    with cm():
        for it in range(1, max_iters + 1):
            # reset accumulators
            if use_torch:
                num.zero_(); den.zero_()
            else:
                num.fill(0.0); den.fill(0.0)

            if use_torch:
                # ---- batched GPU path (hardened with busy→CPU fallback) ----
                retry_cpu = False  # if True, we’ll redo this iteration’s accumulators on CPU

                try:
                    frame_idx = list(range(n_frames))
                    B_cur = int(max(1, auto_B))
                    ci = 0
                    Cn = None

                    # AMP context helper
                    def _maybe_amp():
                        if use_amp and (amp_cm is not None):
                            return amp_cm(**amp_kwargs)
                        from contextlib import nullcontext
                        return nullcontext()

                    # prefetch first batch (unchanged)
                    if prefetch_batches:
                        idx0 = frame_idx[ci:ci+B_cur]
                        if idx0:
                            Cn = C_EXPECTED
                            yb_next, mb_next, vb_next = _make_pinned_batch(idx0, Cn, x_t.dtype)
                        else:
                            yb_next = mb_next = vb_next = None

                    while ci < n_frames:
                        idx = frame_idx[ci:ci + B_cur]
                        B = len(idx)
                        try:
                            # ---- existing gather/conv/weights/backproject code (UNCHANGED) ----
                            if prefetch_batches and (yb_next is not None):
                                yb, mb, vb = yb_next, mb_next, vb_next
                                ci2 = ci + B_cur
                                if ci2 < n_frames:
                                    idx2 = frame_idx[ci2:ci2+B_cur]
                                    Cn = Cn or (_as_chw(_sanitize_numeric(data[idx[0]])).shape[0])
                                    yb_next, mb_next, vb_next = _make_pinned_batch(idx2, Cn, torch.float32)
                                else:
                                    yb_next = mb_next = vb_next = None
                            else:
                                Cn = C_EXPECTED
                                yb, mb, vb = _make_pinned_batch(idx, Cn, torch.float32)
                                if use_channels_last:
                                    yb = yb.contiguous(memory_format=torch.channels_last)

                            wk  = torch.cat([psf_t[fi]  for fi in idx], dim=0).to(memory_format=torch.contiguous_format).contiguous()
                            wkT = torch.cat([psfT_t[fi] for fi in idx], dim=0).to(memory_format=torch.contiguous_format).contiguous()

                            # --- predict on SR grid ---
                            x_bc_hw = x_t.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
                            if use_channels_last:
                                x_bc_hw = x_bc_hw.contiguous(memory_format=torch.channels_last)

                            if use_fft:
                                pred_list = []
                                chan_tmp  = torch.empty_like(x_t[0], dtype=torch.float32)  # (H,W) single-channel
                                for j, fi in enumerate(idx):
                                    C_here = x_bc_hw.shape[1]
                                    pred_j = torch.empty_like(x_t, dtype=torch.float32)    # (C,H,W)
                                    Kf_pack = psf_fft[fi]
                                    for c in range(C_here):
                                        _fft_conv_same_torch(x_bc_hw[j, c], Kf_pack, out_spatial=chan_tmp)  # write into chan_tmp
                                        pred_j[c].copy_(chan_tmp)                                           # copy once
                                    pred_list.append(pred_j)
                                pred_super = torch.stack(pred_list, 0).contiguous()
                            else:
                                pred_super = _grouped_conv_same_torch_per_sample(
                                    x_bc_hw, wk, B, Cn
                                ).contiguous()

                            pred_low = _downsample_avg_bt_t(pred_super, r) if r > 1 else pred_super

                            # --- robust weights (UNCHANGED) ---
                            rnat = yb - pred_low

                            # Build/estimate variance map on the native grid (per batch)
                            if vb is None:
                                # robust per-batch variance estimate
                                med, mad = _robust_med_mad_t(rnat, max_elems_per_sample=2_000_000)
                                vmap = (1.4826 * mad) ** 2
                                # repeat across channels
                                vmap = vmap.repeat(1, Cn, rnat.shape[-2], rnat.shape[-1]).contiguous()
                            else:
                                vmap = vb.unsqueeze(1).repeat(1, Cn, 1, 1).contiguous()

                            if str(rho).lower() == "l2":
                                # L2 ⇒ psi/r = 1 (no robust clipping). Weighted LS with mask/variance.
                                psi_over_r = torch.ones_like(rnat, dtype=torch.float32, device=x_t.device)
                            else:
                                # Huber ⇒ auto-delta or fixed delta
                                if huber_delta < 0:
                                    # ensure we have auto deltas
                                    if (auto_delta_cache is None) or any(auto_delta_cache[fi] is None for fi in idx) or (it % 5 == 1):
                                        Btmp, C_here, H0, W0 = rnat.shape
                                        med, mad = _robust_med_mad_t(rnat, max_elems_per_sample=2_000_000)
                                        rms = 1.4826 * mad
                                        # store per-frame deltas
                                        if auto_delta_cache is None:
                                            # should not happen because we gated creation earlier, but be safe
                                            auto_delta_cache = [None] * n_frames
                                        for j, fi in enumerate(idx):
                                            auto_delta_cache[fi] = float((-huber_delta) * torch.clamp(rms[j, 0, 0, 0], min=1e-6).item())
                                    deltas = torch.tensor([auto_delta_cache[fi] for fi in idx],
                                                        device=x_t.device, dtype=torch.float32).view(B, 1, 1, 1)
                                else:
                                    deltas = torch.tensor(float(huber_delta), device=x_t.device,
                                                        dtype=torch.float32).view(1, 1, 1, 1)

                                absr = rnat.abs()
                                psi_over_r = torch.where(absr <= deltas, torch.ones_like(absr, dtype=torch.float32),
                                                        deltas / (absr + EPS))

                            # compose weights with mask and variance
                            if DEBUG_FLAT_WEIGHTS:
                                wmap_low = mb.unsqueeze(1)  # debug: mask-only weighting
                            else:
                                m1 = mb.unsqueeze(1).repeat(1, Cn, 1, 1).contiguous()
                                wmap_low = (psi_over_r / (vmap + EPS)) * m1
                                wmap_low = torch.nan_to_num(wmap_low, nan=0.0, posinf=0.0, neginf=0.0)

                            # --- adjoint + backproject (UNCHANGED) ---
                            if r > 1:
                                up_y    = _upsample_sum_bt_t(wmap_low * yb,       r)
                                up_pred = _upsample_sum_bt_t(wmap_low * pred_low,  r)
                            else:
                                up_y, up_pred = wmap_low * yb, wmap_low * pred_low

                            if use_fft:
                                back_num_list, back_den_list = [], []
                                for j, fi in enumerate(idx):
                                    C_here = up_y.shape[1]
                                    bn_j = torch.empty_like(x_t, dtype=torch.float32)
                                    bd_j = torch.empty_like(x_t, dtype=torch.float32)
                                    KTf_pack = psfT_fft[fi]
                                    for c in range(C_here):
                                        _fft_conv_same_torch(up_y[j, c],    KTf_pack, out_spatial=bn_j[c])
                                        _fft_conv_same_torch(up_pred[j, c], KTf_pack, out_spatial=bd_j[c])
                                    back_num_list.append(bn_j)
                                    back_den_list.append(bd_j)
                                back_num = torch.stack(back_num_list, 0).sum(dim=0)
                                back_den = torch.stack(back_den_list, 0).sum(dim=0)
                            else:
                                back_num = _grouped_conv_same_torch_per_sample(up_y,    wkT, B, Cn).sum(dim=0)
                                back_den = _grouped_conv_same_torch_per_sample(up_pred, wkT, B, Cn).sum(dim=0)

                            num += back_num
                            den += back_den
                            ci += B

                            if os.environ.get("MFDECONV_DEBUG_SYNC", "0") == "1":
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass

                        except RuntimeError as e:
                            emsg = str(e).lower()
                            # existing OOM backoff stays the same
                            if ("out of memory" in emsg or "resource" in emsg or "alloc" in emsg) and B_cur > 1:
                                B_cur = max(1, B_cur // 2)
                                status_cb(f"GPU OOM: reducing batch_frames → {B_cur} and retrying this chunk.")
                                if prefetch_batches:
                                    yb_next = mb_next = vb_next = None
                                continue
                            # NEW: busy/unavailable → trigger CPU retry for this iteration
                            if _is_cuda_busy_error(e):
                                status_cb("CUDA became unavailable mid-run → retrying this iteration on CPU and switching to CPU for the rest.")
                                retry_cpu = True
                                break  # break the inner while; we'll recompute on CPU
                            raise

                    _process_gui_events_safely()

                except RuntimeError as e:
                    # Catch outer-level CUDA busy/unavailable too (e.g., from syncs)
                    if _is_cuda_busy_error(e):
                        status_cb("CUDA error indicates device busy/unavailable → retrying this iteration on CPU and switching to CPU for the rest.")
                        retry_cpu = True
                    else:
                        raise

                if retry_cpu:
                    # flip to CPU for this and subsequent iterations
                    globals()["cuda_usable"] = False
                    use_torch = False
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                    try:
                        import gc as _gc
                        _gc.collect()
                    except Exception:
                        pass

                    # Rebuild accumulators on CPU for this iteration:
                    # 1) convert x_t (seed/current estimate) to NumPy
                    x_t = x_t.detach().cpu().numpy()
                    # 2) reset accumulators as NumPy arrays
                    num = np.zeros_like(x_t, dtype=np.float32)
                    den = np.zeros_like(x_t, dtype=np.float32)

                    # 3) RUN THE EXISTING NUMPY ACCUMULATION (same as your else: block below)
                    # 3) recompute this iteration’s accumulators on CPU
                    for fi in range(n_frames):
                        # load one frame as CHW (float32, sanitized)

                        y_chw = _load_frame_chw(fi)  # CHW float32 from cache

                        # forward predict (super grid) with this frame's PSF
                        pred_super = _conv_np_same(x_t, psfs[fi])

                        # downsample to native if SR was on
                        pred_low = _downsample_avg(pred_super, r) if r > 1 else pred_super

                        # per-frame mask/variance
                        m2d = _mask_for(fi, y_chw)         # 2D, [0..1]
                        v2d = _var_for(fi, y_chw)          # 2D or None

                        # robust weights per pixel/channel
                        w = _weight_map(
                            y=y_chw, pred=pred_low,
                            huber_delta=local_delta,   # 0.0 for L2, else huber_delta
                            var_map=v2d, mask=m2d
                        ).astype(np.float32, copy=False)

                        # adjoint (backproject) on super grid
                        if r > 1:
                            up_y    = _upsample_sum(w * y_chw,     r, target_hw=x_t.shape[-2:])
                            up_pred = _upsample_sum(w * pred_low,  r, target_hw=x_t.shape[-2:])
                        else:
                            up_y, up_pred = (w * y_chw), (w * pred_low)

                        num += _conv_np_same(up_y,    flip_psf[fi])
                        den += _conv_np_same(up_pred, flip_psf[fi])

                    # ensure strictly positive denominator
                    den = np.clip(den, 1e-8, None).astype(np.float32, copy=False)

                    # switch everything to NumPy for the remainder of the run
                    psf_fft = psfT_fft = None  # (Torch packs no longer used)

            # ---- multiplicative RL/MM step with clamping ----
            if use_torch:
                ratio = num / (den + EPS)
                neutral = (den.abs() < 1e-12) & (num.abs() < 1e-12)
                ratio = torch.where(neutral, torch.ones_like(ratio), ratio)
                upd = torch.clamp(ratio, 1.0 / kappa, kappa)
                x_next = torch.clamp(x_t * upd, min=0.0)
                # Robust scalars
                upd_med = torch.median(torch.abs(upd - 1))
                rel_change = (torch.median(torch.abs(x_next - x_t)) /
                            (torch.median(torch.abs(x_t)) + 1e-8))

                um = float(upd_med.detach().item())
                rc = float(rel_change.detach().item())

                if early.step(it, max_iters, um, rc):
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    status_cb(f"MFDeconv: Iteration {it}/{max_iters} (early stop)")
                    _process_gui_events_safely()
                    break

                x_t = (1.0 - relax) * x_t + relax * x_next
            else:
                ratio = num / (den + EPS)
                neutral = (np.abs(den) < 1e-12) & (np.abs(num) < 1e-12)
                if np.any(neutral): ratio[neutral] = 1.0
                upd = np.clip(ratio, 1.0 / kappa, kappa)
                x_next = np.clip(x_t * upd, 0.0, None)

                um = float(np.median(np.abs(upd - 1.0)))
                rc = float(np.median(np.abs(x_next - x_t)) / (np.median(np.abs(x_t)) + 1e-8))

                if early.step(it, max_iters, um, rc):
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    status_cb(f"MFDeconv: Iteration {it}/{max_iters} (early stop)")
                    _process_gui_events_safely()
                    break

                x_t = (1.0 - relax) * x_t + relax * x_next

            # --- LOW-MEM CLEANUP (per-iteration) ---
            if low_mem:
                # Torch temporaries we created in the iteration (best-effort deletes)
                to_kill = [
                    "pred_super", "pred_low", "wmap_low", "yb", "mb", "vb", "wk", "wkT",
                    "back_num", "back_den", "pred_list", "back_num_list", "back_den_list",
                    "x_bc_hw", "up_y", "up_pred", "psi_over_r", "vmap", "rnat", "deltas", "chan_tmp"
                ]
                loc = locals()
                for _name in to_kill:
                    if _name in loc:
                        try:
                            del loc[_name]
                        except Exception:
                            pass

                # Proactively release CUDA cache every other iter
                if use_torch:
                    try:
                        dev = _torch_device()
                        if dev.type == "cuda" and (it % 2) == 0:
                            import torch as _t
                            _t.cuda.empty_cache()
                    except Exception:
                        pass

                # Encourage Python to return big NumPy buffers to the OS sooner
                import gc as _gc
                _gc.collect()


            # ---- save intermediates ----
            if save_intermediate and (it % int(max(1, save_every)) == 0):
                try:
                    x_np = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)
                    _save_iter_image(x_np, hdr0_seed, iter_dir, f"iter_{it:03d}", color_mode)
                except Exception as _e:
                    status_cb(f"Intermediate save failed at iter {it}: {_e}")

            frac = 0.25 + 0.70 * (it / float(max_iters))
            _emit_pct(frac, f"Iteration {it}/{max_iters}")
            status_cb(f"Iter {it}/{max_iters}")
            _process_gui_events_safely()

    if not early_stopped:
        used_iters = max_iters

    # ---- save result ----
    _emit_pct(0.97, "saving")
    x_final = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)
    if x_final.ndim == 3:
        if x_final.shape[0] not in (1, 3) and x_final.shape[-1] in (1, 3):
            x_final = np.moveaxis(x_final, -1, 0)
        if x_final.shape[0] == 1:
            x_final = x_final[0]

    try:
        hdr0 = _safe_primary_header(paths[0])
    except Exception:
        hdr0 = fits.Header()

    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution')
    hdr0['MF_COLOR'] = (str(color_mode), 'Color mode used')
    hdr0['MF_RHO']   = (str(rho), 'Loss: huber|l2')
    hdr0['MF_HDEL']  = (float(huber_delta), 'Huber delta (>0 abs, <0 autoxRMS)')
    hdr0['MF_MASK']  = (bool(use_star_masks),    'Used auto star masks')
    hdr0['MF_VAR']   = (bool(use_variance_maps), 'Used auto variance maps')
    r = int(max(1, super_res_factor))
    hdr0['MF_SR']    = (int(r), 'Super-resolution factor (1 := native)')
    if r > 1:
        hdr0['MF_SRSIG'] = (float(sr_sigma), 'Gaussian sigma for SR PSF fit (native px)')
        hdr0['MF_SRIT']  = (int(sr_psf_opt_iters), 'SR-PSF solver iters')
    hdr0['MF_ITMAX'] = (int(max_iters),  'Requested max iterations')
    hdr0['MF_ITERS'] = (int(used_iters), 'Actual iterations run')
    hdr0['MF_ESTOP'] = (bool(early_stopped), 'Early stop triggered')

    if isinstance(x_final, np.ndarray):
        if x_final.ndim == 2:
            hdr0['MF_SHAPE'] = (f"{x_final.shape[0]}x{x_final.shape[1]}", 'Saved as 2D image (HxW)')
        elif x_final.ndim == 3:
            C, H, W = x_final.shape
            hdr0['MF_SHAPE'] = (f"{C}x{H}x{W}", 'Saved as 3D cube (CxHxW)')

    save_path     = _sr_out_path(out_path, super_res_factor)
    safe_out_path = _nonclobber_path(str(save_path))
    if safe_out_path != str(save_path):
        status_cb(f"Output exists — saving as: {safe_out_path}")
    fits.PrimaryHDU(data=x_final, header=hdr0).writeto(safe_out_path, overwrite=False)

    status_cb(f"✅ MFDeconv saved: {safe_out_path}  (iters used: {used_iters}{', early stop' if early_stopped else ''})")
    _emit_pct(1.00, "done")
    _process_gui_events_safely()

    try:
        if use_torch:
            try: del num, den
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            try: del psf_t, psfT_t
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            _free_torch_memory()
    except Exception:
        pass

    try:
        _clear_all_caches()
    except Exception:
        pass

    return safe_out_path

# -----------------------------
# Worker
# -----------------------------

class MultiFrameDeconvWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, out_path

    def __init__(self, parent, aligned_paths, output_path, iters, kappa, color_mode,
                 huber_delta, min_iters, use_star_masks=False, use_variance_maps=False, rho="huber",
                 star_mask_cfg: dict | None = None, varmap_cfg: dict | None = None,
                 save_intermediate: bool = False,
                 seed_mode: str = "robust",
                 # NEW SR params
                 super_res_factor: int = 1,
                 sr_sigma: float = 1.1,
                 sr_psf_opt_iters: int = 250,
                 sr_psf_opt_lr: float = 0.1,
                 star_mask_ref_path: str | None = None):

        super().__init__(parent)
        self.aligned_paths = aligned_paths
        self.output_path = output_path
        self.iters = iters
        self.kappa = kappa
        self.color_mode = color_mode
        self.huber_delta = huber_delta
        self.min_iters = min_iters  # NEW
        self.star_mask_cfg = star_mask_cfg or {}
        self.varmap_cfg    = varmap_cfg or {}
        self.use_star_masks = use_star_masks
        self.use_variance_maps = use_variance_maps
        self.rho = rho
        self.save_intermediate = save_intermediate   
        self.super_res_factor = int(super_res_factor)
        self.sr_sigma = float(sr_sigma)
        self.sr_psf_opt_iters = int(sr_psf_opt_iters)
        self.sr_psf_opt_lr = float(sr_psf_opt_lr)
        self.star_mask_ref_path = star_mask_ref_path 
        self.seed_mode = seed_mode


    def _log(self, s): self.progress.emit(s)

    def run(self):
        try:
            out = multiframe_deconv(
                self.aligned_paths,
                self.output_path,
                iters=self.iters,
                kappa=self.kappa,
                color_mode=self.color_mode,
                seed_mode=self.seed_mode,
                huber_delta=self.huber_delta,
                use_star_masks=self.use_star_masks,
                use_variance_maps=self.use_variance_maps,
                rho=self.rho,
                min_iters=self.min_iters,
                status_cb=self._log,
                star_mask_cfg=self.star_mask_cfg,
                varmap_cfg=self.varmap_cfg,
                save_intermediate=self.save_intermediate,
                super_res_factor=self.super_res_factor,
                sr_sigma=self.sr_sigma,
                sr_psf_opt_iters=self.sr_psf_opt_iters,
                sr_psf_opt_lr=self.sr_psf_opt_lr,
                star_mask_ref_path=self.star_mask_ref_path,
            )
            self.finished.emit(True, "MF deconvolution complete.", out)
            _process_gui_events_safely()
        except Exception as e:
            self.finished.emit(False, f"MF deconvolution failed: {e}", "")
        finally:
            # Hard cleanup: drop references + free GPU memory
            try:
                # Drop big Python refs that might keep tensors alive indirectly
                self.aligned_paths = []
                self.star_mask_cfg = {}
                self.varmap_cfg = {}
            except Exception:
                pass
            try:
                _free_torch_memory()  # your helper: del tensors, gc.collect(), etc.
            except Exception:
                pass
            try:
                import torch as _t
                if hasattr(_t, "cuda") and _t.cuda.is_available():
                    _t.cuda.synchronize()
                    _t.cuda.empty_cache()
                if hasattr(_t, "mps") and getattr(_t.backends, "mps", None) and _t.backends.mps.is_available():
                    # PyTorch 2.x has this
                    if hasattr(_t.mps, "empty_cache"):
                        _t.mps.empty_cache()
                # DirectML usually frees on GC; nothing special to call.
            except Exception:
                pass