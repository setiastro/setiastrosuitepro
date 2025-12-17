# pro/mfdeconvsport.py
from __future__ import annotations
import os, sys
import math
import re
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QObject, pyqtSignal
from setiastro.saspro.psf_utils import compute_psf_kernel_for_image
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
_USE_PROCESS_POOL_FOR_ASSETS = not getattr(sys, "frozen", False)
from setiastro.saspro.mfdeconv_earlystop import EarlyStopper

import contextlib
try:
    import sep
except Exception:
    sep = None
from setiastro.saspro.free_torch_memory import _free_torch_memory
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

# at top of file with the other imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import SimpleQueue
from setiastro.saspro.memory_utils import LRUDict

# ── XISF decode cache → memmap on disk ─────────────────────────────────
import tempfile
import threading
import uuid
import atexit
_XISF_CACHE = LRUDict(50)
_XISF_LOCK  = threading.Lock()
_XISF_TMPFILES = []

from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────────────────
# Unified image I/O for MFDeconv (FITS + XISF)
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
from astropy.io import fits

from pathlib import Path


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
    psf = _soften_psf(_normalize_psf(psf.astype(np.float32, copy=False)), sigma_px=0.25)

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

def _to_luma_local(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # mono fast paths
        if a.shape[-1] == 1:         # HWC mono
            return a[..., 0].astype(np.float32, copy=False)
        if a.shape[0] == 1:          # CHW mono
            return a[0].astype(np.float32, copy=False)
        # RGB
        if a.shape[-1] == 3:         # HWC RGB
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
        if a.shape[0] == 3:          # CHW RGB
            r, g, b = a[0], a[1], a[2]
            return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    # fallback: average last axis
    return a.mean(axis=-1).astype(np.float32, copy=False)

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
    # img: (H,W) or (C,H,W) numpy
    import numpy.fft as fft
    def fftconv2(a, k):
        H, W = a.shape[-2:]
        kh, kw = k.shape
        pad_h, pad_w = H + kh - 1, W + kw - 1
        A = fft.rfftn(a, s=(pad_h, pad_w), axes=(-2, -1))
        K = fft.rfftn(k, s=(pad_h, pad_w), axes=(-2, -1))
        Y = A * K
        y = fft.irfftn(Y, s=(pad_h, pad_w), axes=(-2, -1))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        return y[..., sh:sh+H, sw:sw+W]
    if img.ndim == 2:
        return fftconv2(img[None], psf)[0]
    else:
        return np.stack([fftconv2(img[c:c+1], psf)[0] for c in range(img.shape[0])], axis=0)

def _normalize_psf(psf):
    psf = np.maximum(psf, 0.0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if not np.isfinite(s) or s <= EPS:
        return psf
    return (psf / s).astype(np.float32, copy=False)

def _soften_psf(psf, sigma_px=0.25):
    # optional tiny Gaussian soften to reduce ringing; sigma<=0 disables
    if sigma_px <= 0:
        return psf
    r = int(max(1, round(3 * sigma_px)))
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2 * sigma_px * sigma_px)).astype(np.float32)
    g /= g.sum() + EPS
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

def _estimate_scalar_variance_t(r):
    # r: tensor on device
    med = torch.median(r)
    mad = torch.median(torch.abs(r - med)) + 1e-6
    return (1.4826 * mad) ** 2

def _estimate_scalar_variance(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return float((1.4826 * mad) ** 2)

def _weight_map(y, pred, huber_delta, var_map=None, mask=None):
    """
    Robust per-pixel weights for the MM update.
    W = [psi(r)/r] * 1/(var + eps) * mask
    If huber_delta < 0, delta = (-huber_delta) * RMS(residual) (auto).
    var_map: per-pixel variance (2D); if None, fall back to robust scalar via MAD.
    mask: 2D {0,1} validity; if None, treat as ones.
    """
    r = y - pred
    eps = EPS

    # resolve Huber delta
    if huber_delta < 0:
        if TORCH_OK and isinstance(r, torch.Tensor):
            med = torch.median(r)
            mad = torch.median(torch.abs(r - med)) + 1e-6
            rms = 1.4826 * mad
            delta = (-huber_delta) * torch.clamp(rms, min=1e-6)
        else:
            med = np.median(r)
            mad = np.median(np.abs(r - med)) + 1e-6
            rms = 1.4826 * mad
            delta = (-huber_delta) * max(rms, 1e-6)
    else:
        delta = huber_delta

    # psi(r)/r
    if TORCH_OK and isinstance(r, torch.Tensor):
        absr = torch.abs(r)
        if float(delta) > 0:
            psi_over_r = torch.where(absr <= delta, torch.ones_like(r), delta / (absr + eps))
        else:
            psi_over_r = torch.ones_like(r)
        if var_map is None:
            v = _estimate_scalar_variance_t(r)
        else:
            v = var_map
            if v.ndim == 2 and r.ndim == 3:
                v = v[None, ...]  # broadcast over channels
        w = psi_over_r / (v + eps)
        if mask is not None:
            m = mask if mask.ndim == w.ndim else (mask[None, ...] if w.ndim == 3 else mask)
            w = w * m
        return w
    else:
        absr = np.abs(r)
        if float(delta) > 0:
            psi_over_r = np.where(absr <= delta, 1.0, delta / (absr + eps)).astype(np.float32)
        else:
            psi_over_r = np.ones_like(r, dtype=np.float32)
        if var_map is None:
            v = _estimate_scalar_variance(r)
        else:
            v = var_map
            if v.ndim == 2 and r.ndim == 3:
                v = v[None, ...]
        w = psi_over_r / (v + eps)
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
    tfft = torch.fft
    psf_fft, psfT_fft = [], []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        padH, padW = _fftshape_same(H, W, kh, kw)

        # shift the small kernels to the origin, then FFT into padded size
        k_small  = torch.as_tensor(np.fft.ifftshift(k),  device=device, dtype=dtype)
        kT_small = torch.as_tensor(np.fft.ifftshift(kT), device=device, dtype=dtype)

        Kf  = tfft.rfftn(k_small,  s=(padH, padW))
        KTf = tfft.rfftn(kT_small, s=(padH, padW))

        psf_fft.append((Kf,  padH, padW, kh, kw))
        psfT_fft.append((KTf, padH, padW, kh, kw))
    return psf_fft, psfT_fft



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
        sh, sw = kh // 2, kw // 2
        out[...] = y[sh:sh+a.shape[0], sw:sw+a.shape[1]]
        return out
    else:
        C, H, W = a.shape
        acc = []
        for c in range(C):
            A = fft.rfftn(a[c], s=(fftH, fftW))
            y = fft.irfftn(A * Kf, s=(fftH, fftW))
            sh, sw = kh // 2, kw // 2
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
    k = int(f.shape[0]); assert f.shape[0] == f.shape[1]
    kr = int(k * r)

    # build Gaussian pre-blur at native scale (match paper §4.2)
    g = _gaussian2d(k, max(sigma, 1e-3)).astype(np.float32)

    # init h by zero-insertion (nearest upsample of f) then deconvolving g very mildly
    h0 = np.zeros((kr, kr), dtype=np.float32)
    h0[::r, ::r] = f
    h0 = _normalize_psf(h0)

    if TORCH_OK:
        dev = _torch_device()
        t = torch.tensor(h0, device=dev, dtype=torch.float32, requires_grad=True)
        f_t = torch.tensor(f,  device=dev, dtype=torch.float32)
        g_t = torch.tensor(g,  device=dev, dtype=torch.float32)
        opt = torch.optim.Adam([t], lr=lr)
        for _ in range(max(10, iters)):
            opt.zero_grad(set_to_none=True)
            H, W = t.shape
            Hr, Wr = H//r, W//r
            th = t[:Hr*r, :Wr*r].reshape(Hr, r, Wr, r).mean(dim=(1,3))
            # conv native: (Dh) * g
            conv = torch.nn.functional.conv2d(th[None,None], g_t[None,None], padding=g_t.shape[-1]//2)[0,0]
            loss = torch.mean((conv - f_t)**2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                t.clamp_(min=0.0)
                t /= (t.sum() + 1e-8)
        h = t.detach().cpu().numpy().astype(np.float32)
    else:
        # Tiny gradient-descent fallback on numpy
        h = h0.copy()
        eta = float(lr)
        for _ in range(max(50, iters)):
            Dh = _downsample_avg(h, r)
            conv = _conv2_same_np(Dh, g)
            resid = (conv - f)
            # backprop through conv and D: grad wrt Dh is resid * g^T conv; adjoint of D is upsample-sum
            grad_Dh = _conv2_same_np(resid, np.flip(np.flip(g, 0), 1))
            grad_h  = _upsample_sum(grad_Dh, r, target_hw=h.shape)
            h = np.clip(h - eta * grad_h, 0.0, None)
            s = float(h.sum());  h /= (s + 1e-8)
            eta *= 0.995
    return _normalize_psf(h)

def _downsample_avg_t(x, r: int):
    """
    Average-pool over non-overlapping r×r blocks.
    Works for (H,W) or (C,H,W). Crops to multiples of r.
    """
    if r <= 1:
        return x
    if x.ndim == 2:
        H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x  # nothing to pool
        x2 = x[:Hr, :Wr]
        return x2.view(Hr // r, r, Wr // r, r).mean(dim=(1, 3))
    else:
        C, H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x
        x2 = x[:, :Hr, :Wr]
        return x2.view(C, Hr // r, r, Wr // r, r).mean(dim=(2, 4))

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


def _read_tile_fits_any(path: str, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """FITS/XISF-aware tile read: returns spatial tile; supports 2D, HWC, and CHW."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".xisf":
        a, _ = _load_image_array(path)  # helper returns array-like + hdr/metadata
        if a is None:
            raise ValueError(f"XISF loader returned None for {path}")
        a = np.asarray(a)
        if a.ndim == 2:  # HW
            return np.array(a[y0:y1, x0:x1], copy=True)
        elif a.ndim == 3:
            if a.shape[-1] in (1, 3):            # HWC
                out = a[y0:y1, x0:x1, :]
                if out.shape[-1] == 1:
                    out = out[..., 0]
                return np.array(out, copy=True)
            elif a.shape[0] in (1, 3):           # CHW
                out = a[:, y0:y1, x0:x1]
                if out.shape[0] == 1:
                    out = out[0]
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
                if out.shape[0] == 1: out = out[0]
                return np.array(out, copy=True)
            if a.shape[-1] in (1, 3):             # HWC
                out = a[y0:y1, x0:x1, :]
                if out.shape[-1] == 1: out = out[..., 0]
                return np.array(out, copy=True)

        # Fallback: assume last two axes are spatial (…, H, W)
        try:
            out = a[(..., slice(y0, y1), slice(x0, x1))]
            return np.array(out, copy=True)
        except Exception:
            raise ValueError(f"Unsupported FITS data shape {a.shape} in {path}")


def _seed_median_full_from_data(data_list):
    """
    data_list: list of np.ndarray each shaped either (H,W) or (C,H,W),
               already cropped/sanitized to the same size by the caller.
    Returns:   (H,W) or (C,H,W) median image in float32.
    """
    if not data_list:
        raise ValueError("Empty stack for median seed")

    a0 = data_list[0]
    if a0.ndim == 2:
        # (N, H, W) -> (H, W)
        cube = np.stack([np.asarray(a, dtype=np.float32, order="C") for a in data_list], axis=0)
        med = np.median(cube, axis=0).astype(np.float32, copy=False)
        return np.ascontiguousarray(med)
    else:
        # (N, C, H, W) -> (C, H, W)
        cube = np.stack([np.asarray(a, dtype=np.float32, order="C") for a in data_list], axis=0)
        med = np.median(cube, axis=0).astype(np.float32, copy=False)
        return np.ascontiguousarray(med)


def _build_seed_running_mu_sigma_from_paths(paths, Ht, Wt, color_mode,
                                            *, bootstrap_frames=20, clip_sigma=5.0,
                                            status_cb=lambda s: None, progress_cb=lambda f,m='': None):
    K = max(1, min(int(bootstrap_frames), len(paths)))
    def _load_chw(i):
        ys, _ = _stack_loader_memmap([paths[i]], Ht, Wt, color_mode)
        return _as_chw(ys[0]).astype(np.float32, copy=False)
    x0 = _load_chw(0).copy()
    mean = x0; m2 = np.zeros_like(mean); count = 1
    for i in range(1, K):
        x = _load_chw(i); count += 1
        d = x - mean; mean += d/count; m2 += d*(x-mean)
        progress_cb(i/K*0.5, "μ-σ bootstrap")
    var = m2 / max(1, count-1); sigma = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    lo = mean - float(clip_sigma)*sigma; hi = mean + float(clip_sigma)*sigma
    acc = np.zeros_like(mean); n=0
    for i in range(len(paths)):
        x = _load_chw(i); x = np.clip(x, lo, hi, out=x)
        acc += x; n += 1; progress_cb(0.5 + 0.5*(i+1)/len(paths), "clipped mean")
    seed = (acc/max(1,n)).astype(np.float32)
    return seed[0] if (seed.ndim==3 and seed.shape[0]==1) else seed

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
    # >>> SR options
    super_res_factor: int = 1,
    sr_sigma: float = 1.1,
    sr_psf_opt_iters: int = 250,
    sr_psf_opt_lr: float = 0.1,
    star_mask_ref_path: str | None = None,     
):
    # sanitize and clamp
    max_iters = max(1, int(iters))
    min_iters = max(1, int(min_iters))
    if min_iters > max_iters:
        min_iters = max_iters

    def _emit_pct(pct: float, msg: str | None = None):
        pct = float(max(0.0, min(1.0, pct)))
        status_cb(f"__PROGRESS__ {pct:.4f}" + (f" {msg}" if msg else ""))

    status_cb(f"MFDeconv: loading {len(paths)} aligned frames…")
    _emit_pct(0.02, "loading")

    # Use unified probe to pick a common crop without loading full images
    Ht, Wt = _common_hw_from_paths(paths)
    _emit_pct(0.05, "preparing")

    # Stream actual pixels cropped to (Ht,Wt), float32 CHW/2D + headers
    ys_raw, hdrs = _stack_loader_memmap(paths, Ht, Wt, color_mode)
    relax = 0.7
    use_torch = False
    global torch, TORCH_OK

    # -------- try to import torch from per-user runtime venv --------
    torch = None
    TORCH_OK = False
    cuda_ok = mps_ok = dml_ok = False
    dml_device = None
    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(prefer_cuda=True, status_cb=status_cb)
        TORCH_OK = True

        try: cuda_ok = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception: cuda_ok = False
        try: mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception: mps_ok = False
        try:
            import torch_directml
            dml_device = torch_directml.device()
            _ = (torch.ones(1, device=dml_device) + 1).item()
            dml_ok = True
        except Exception:
            dml_ok = False

        if cuda_ok:
            status_cb(f"PyTorch CUDA available: True | device={torch.cuda.get_device_name(0)}")
        elif mps_ok:
            status_cb("PyTorch MPS (Apple) available: True")
        elif dml_ok:
            status_cb("PyTorch DirectML (Windows) available: True")
        else:
            status_cb("PyTorch present, using CPU backend.")

        status_cb(
            f"PyTorch {getattr(torch, '__version__', '?')} backend: "
            + ("CUDA" if cuda_ok else "MPS" if mps_ok else "DirectML" if dml_ok else "CPU")
        )
    except Exception as e:
        TORCH_OK = False
        status_cb(f"PyTorch not available → CPU path. ({e})")

    use_torch = bool(TORCH_OK)
    if use_torch:
        # ----- Precision policy (Sport mode but strict FP32) -----
        try:
            # Keep autotune for speed
            torch.backends.cudnn.benchmark = True

            # Force true FP32 everywhere (no TF32 shortcuts)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

        # (optional: telemetry)
        try:
            c_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
            m_tf32 = getattr(getattr(torch.backends.cuda, "matmul", object()), "allow_tf32", None)
            status_cb(
                f"Precision: cudnn.allow_tf32={c_tf32} | "
                f"matmul.allow_tf32={m_tf32} | "
                f"benchmark={torch.backends.cudnn.benchmark}"
            )
        except Exception:
            pass

    _process_gui_events_safely()

    # PSFs (auto-size per frame) + flipped copies
    psf_out_dir = None
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

    # >>> SR: lift PSFs to super-res if requested
    r = int(max(1, super_res_factor))
    if r > 1:
        status_cb(f"MFDeconv: Super-resolution r={r} with σ={sr_sigma} — solving SR PSFs…")
        _process_gui_events_safely()
        sr_psfs = []
        for i, k_native in enumerate(psfs, start=1):
            h = _solve_super_psf_from_native(k_native, r=r, sigma=float(sr_sigma),
                                             iters=int(sr_psf_opt_iters), lr=float(sr_psf_opt_lr))
            sr_psfs.append(h)
            status_cb(f"  SR-PSF{i}: native {k_native.shape[0]} → {h.shape[0]} (sum={h.sum():.6f})")
        psfs = sr_psfs

    flip_psf = [_flip_kernel(k) for k in psfs]
    _emit_pct(0.20, "PSF Ready")

    # Normalize layout BEFORE size harmonization
    data = _normalize_layout_batch(ys_raw, color_mode)  # list of (H,W) or (3,H,W)
    if str(color_mode).lower() != "luma":
        # Force strict CHW for every frame
        data = [_as_chw(a) for a in data]
        Cs = {a.shape[0] for a in data}
        if len(Cs) != 1:
            raise ValueError(f"Inconsistent channel counts in PerChannel mode: {Cs}")
    _emit_pct(0.25, "Calculating Seed Image...")

    # Center-crop all to common intersection
    Ht, Wt = _common_hw(data)
    if any(((a.shape[-2] != Ht) or (a.shape[-1] != Wt)) for a in data):
        status_cb(f"MFDeconv: Standardizing shapes → crop to {Ht}×{Wt}")
        data = [_center_crop(a, Ht, Wt) for a in data]

    # Numeric hygiene
    data = [_sanitize_numeric(a) for a in data]

    # --- SR/native seed ---
    # --- Seed (choose robust μ-σ or median) ---
    seed_mode_s = str(seed_mode).lower().strip()
    if seed_mode_s not in ("robust","median"):
        seed_mode_s = "robust"

    if seed_mode_s == "median":
        status_cb("MFDeconv: Building median seed (in-memory)…")
        # Use already normalized, cropped, sanitized frames
        seed_native = _seed_median_full_from_data(data)
    else:
        status_cb("MFDeconv: Building robust seed (live μ-σ stacking)…")
        seed_native = _build_seed_running_mu_sigma_from_paths(
            paths, Ht, Wt, color_mode,
            bootstrap_frames=20, clip_sigma=5.0,
            status_cb=status_cb, progress_cb=lambda f,m='': None
        )
    if r > 1:
        if seed_native.ndim == 2:
            x = _upsample_sum(seed_native / (r*r), r, target_hw=(Ht*r, Wt*r))
        else:
            C, Hn, Wn = seed_native.shape
            x = np.stack(
                [_upsample_sum(seed_native[c] / (r*r), r, target_hw=(Hn*r, Wn*r)) for c in range(C)],
                axis=0
            )
    else:
        x = seed_native
    Hs, Ws = x.shape if x.ndim == 2 else x.shape[-2:]

    # masks/vars aligned to native grid (2D each)
    auto_masks = masks_auto if use_star_masks else None
    auto_vars  = vars_auto  if use_variance_maps else None
    mask_list = _ensure_mask_list(masks if masks is not None else auto_masks, data)
    var_list  = _ensure_var_list(variances if variances is not None else auto_vars, data)

    iter_dir = None
    hdr0_seed = None
    if save_intermediate:
        iter_dir = _iter_folder(out_path)
        status_cb(f"MFDeconv: Intermediate outputs → {iter_dir}")
        try:
            hdr0_seed = _safe_primary_header(paths[0])
        except Exception:
            hdr0_seed = fits.Header()
        _save_iter_image(x, hdr0_seed, iter_dir, "seed", color_mode)

    status_cb("MFDeconv: Calculating Backgrounds and MADs…")
    _process_gui_events_safely()
    bg_est = _sep_bg_rms(data) or (np.median([np.median(np.abs(y - np.median(y))) for y in (data if isinstance(data, list) else [data])]) * 1.4826)
    status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g})")
    _process_gui_events_safely()

    status_cb("Computing FFTs and Allocating Scratch…")
    _process_gui_events_safely()

    # -------- precompute and allocate scratch --------
    per_frame_logging = (r > 1)
    if use_torch:
        x_t = _to_t(_contig(x))
        num = torch.zeros_like(x_t)
        den = torch.zeros_like(x_t)

        if r > 1:
            # >>> SR path now uses SPATIAL CONV (cuDNN) to avoid huge FFT buffers
            psf_t  = [_to_t(_contig(k))[None, None]  for k  in psfs]   # (1,1,kh,kw)
            psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]
        else:
            # Native spatial (as before)
            psf_t  = [_to_t(_contig(k))[None, None]  for k  in psfs]
            psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]

    else:
        x_t = x
        # CPU path: keep your (more RAM-tolerant) FFT packs
        if r > 1:
            if x_t.ndim == 2:
                Hs, Ws = x_t.shape
            else:
                _, Hs, Ws = x_t.shape
            Kfs, KTfs, meta = _precompute_np_psf_ffts(psfs, flip_psf, Hs, Ws)
            pred_super = np.empty_like(x_t)
            tmp_out    = np.empty_like(x_t)
        else:
            Kfs, KTfs, meta = _precompute_np_psf_ffts(psfs, flip_psf, Hs, Ws)
            pred_super = np.empty_like(x_t)
            tmp_out    = np.empty_like(x_t)
        num = np.zeros_like(x_t)
        den = np.zeros_like(x_t)

    status_cb("Starting First Multiplicative Iteration…")
    _process_gui_events_safely()

    cm = _safe_inference_context() if use_torch else NO_GRAD
    rho_is_l2 = (str(rho).lower() == "l2")
    local_delta = 0.0 if rho_is_l2 else huber_delta
    used_iters = 0
    early_stopped = False

    auto_delta_cache = None
    if use_torch and (huber_delta < 0) and (not rho_is_l2):
        auto_delta_cache = [None] * len(paths)

    early = EarlyStopper(
        tol_upd_floor=2e-4,   # match new version
        tol_rel_floor=5e-4,
        early_frac=0.40,
        ema_alpha=0.5,
        patience=2,
        min_iters=min_iters,
        status_cb=status_cb
    )
    x_ndim = 2 if (np.ndim(x) == 2) else 3
    fixed = 0
    for i, a in enumerate(data):
        if a.ndim != x_ndim:
            # fix common mono cases only
            if x_ndim == 2 and a.ndim == 3 and a.shape[0] == 1:
                data[i] = a[0]; fixed += 1
            elif x_ndim == 2 and a.ndim == 3 and a.shape[-1] == 1:
                data[i] = a[..., 0]; fixed += 1

    with cm():
        for it in range(1, max_iters + 1):
            if use_torch:
                num.zero_(); den.zero_()

                if r > 1:
                    # -------- SR path (SPATIAL conv + stream) --------
                    for fidx, (wk, wkT) in enumerate(zip(psf_t, psfT_t)):
                        yt_np = data[fidx]   # CHW or HW (CPU)
                        mt_np = mask_list[fidx]
                        vt_np = var_list[fidx]

                        yt = torch.as_tensor(yt_np, dtype=x_t.dtype, device=x_t.device)
                        mt = None if mt_np is None else torch.as_tensor(mt_np, dtype=x_t.dtype, device=x_t.device)
                        vt = None if vt_np is None else torch.as_tensor(vt_np, dtype=x_t.dtype, device=x_t.device)

                        # SR conv on grid of x_t
                        pred_super = _conv_same_torch(x_t, wk)              # SR grid
                        pred_low   = _downsample_avg_t(pred_super, r)       # native grid

                        if auto_delta_cache is not None:
                            if (auto_delta_cache[fidx] is None) or (it % 5 == 1):
                                rnat = yt - pred_low
                                med = torch.median(rnat)
                                mad = torch.median(torch.abs(rnat - med)) + 1e-6
                                rms = 1.4826 * mad
                                auto_delta_cache[fidx] = float((-huber_delta) * torch.clamp(rms, min=1e-6).item())
                            wmap_low = _weight_map(yt, pred_low, auto_delta_cache[fidx], var_map=vt, mask=mt)
                        else:
                            wmap_low = _weight_map(yt, pred_low, local_delta, var_map=vt, mask=mt)

                        # lift back to SR via sum-replicate
                        up_y    = _upsample_sum_t(wmap_low * yt,       r)
                        up_pred = _upsample_sum_t(wmap_low * pred_low, r)

                        # accumulate via adjoint kernel on SR grid
                        num += _conv_same_torch(up_y,    wkT)
                        den += _conv_same_torch(up_pred, wkT)

                        # free temps as aggressively as possible
                        del yt, mt, vt, pred_super, pred_low, wmap_low, up_y, up_pred
                        if cuda_ok:
                            try: torch.cuda.empty_cache()
                            except Exception as e:
                                import logging
                                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

                        if per_frame_logging and ((fidx & 7) == 0):
                            status_cb(f"Iter {it}/{max_iters} — frame {fidx+1}/{len(paths)} (SR spatial)")

                else:
                    # -------- Native path (spatial conv, stream) --------
                    for fidx, (wk, wkT) in enumerate(zip(psf_t, psfT_t)):
                        yt_np = data[fidx]
                        mt_np = mask_list[fidx]
                        vt_np = var_list[fidx]

                        yt = torch.as_tensor(yt_np, dtype=x_t.dtype, device=x_t.device)
                        mt = None if mt_np is None else torch.as_tensor(mt_np, dtype=x_t.dtype, device=x_t.device)
                        vt = None if vt_np is None else torch.as_tensor(vt_np, dtype=x_t.dtype, device=x_t.device)

                        pred = _conv_same_torch(x_t, wk)
                        wmap = _weight_map(yt, pred, local_delta, var_map=vt, mask=mt)
                        up_y    = wmap * yt
                        up_pred = wmap * pred
                        num += _conv_same_torch(up_y,    wkT)
                        den += _conv_same_torch(up_pred, wkT)

                        del yt, mt, vt, pred, wmap, up_y, up_pred
                        if cuda_ok:
                            try: torch.cuda.empty_cache()
                            except Exception as e:
                                import logging
                                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

                ratio = num / (den + EPS)
                neutral = (den.abs() < 1e-12) & (num.abs() < 1e-12)
                ratio = torch.where(neutral, torch.ones_like(ratio), ratio)
                upd = torch.clamp(ratio, 1.0 / kappa, kappa)
                x_next = torch.clamp(x_t * upd, min=0.0)

                upd_med = torch.median(torch.abs(upd - 1))
                rel_change = (torch.median(torch.abs(x_next - x_t)) /
                              (torch.median(torch.abs(x_t)) + 1e-8))

                # candidates for convergence
                try:
                    um = float(upd_med.detach().cpu().item())
                except Exception:
                    um = float(upd_med)

                try:
                    rc = float(rel_change.detach().cpu().item())
                except Exception:
                    rc = float(rel_change)

                if early.step(it, max_iters, um, rc):
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    _process_gui_events_safely()
                    break

                x_t = (1.0 - relax) * x_t + relax * x_next

            else:
                # -------- NumPy path (unchanged) --------
                num.fill(0.0); den.fill(0.0)
                if r > 1:
                    for (Kf, KTf, (kh, kw, fftH, fftW)), m2d, v2d, y_nat in zip(
                        zip(Kfs, KTfs, meta), mask_list, var_list, data):
                        _fft_conv_same_np(x_t, Kf, kh, kw, fftH, fftW, pred_super)
                        pred_low = _downsample_avg(pred_super, r)
                        wmap_low = _weight_map(y_nat, pred_low, local_delta, var_map=v2d, mask=m2d)
                        up_y    = _upsample_sum(wmap_low * y_nat,    r, target_hw=pred_super.shape[-2:])
                        up_pred = _upsample_sum(wmap_low * pred_low, r, target_hw=pred_super.shape[-2:])
                        _fft_conv_same_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out); num += tmp_out
                        _fft_conv_same_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out); den += tmp_out
                else:
                    for (Kf, KTf, (kh, kw, fftH, fftW)), m2d, v2d, y_nat in zip(
                        zip(Kfs, KTfs, meta), mask_list, var_list, data):
                        _fft_conv_same_np(x_t, Kf, kh, kw, fftH, fftW, pred_super)
                        pred = pred_super
                        wmap = _weight_map(y_nat, pred, local_delta, var_map=v2d, mask=m2d)
                        up_y, up_pred = (wmap * y_nat), (wmap * pred)
                        _fft_conv_same_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out); num += tmp_out
                        _fft_conv_same_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out); den += tmp_out

                ratio = num / (den + EPS)
                neutral = (np.abs(den) < 1e-12) & (np.abs(num) < 1e-12)
                ratio[neutral] = 1.0
                upd = np.clip(ratio, 1.0 / kappa, kappa)
                x_next = np.clip(x_t * upd, 0.0, None)

                upd_med = np.median(np.abs(upd - 1.0))
                rel_change = (np.median(np.abs(x_next - x_t)) /
                              (np.median(np.abs(x_t)) + 1e-8))
                um = float(upd_med)
                rc = float(rel_change)

                if early.step(it, max_iters, um, rc):
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    _process_gui_events_safely()
                    break


                x_t = (1.0 - relax) * x_t + relax * x_next

            # save intermediate
            if save_intermediate and (it % int(max(1, save_every)) == 0):
                try:
                    x_np = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)
                    _save_iter_image(x_np, hdr0_seed, iter_dir, f"iter_{it:03d}", color_mode)
                except Exception as _e:
                    status_cb(f"Intermediate save failed at iter {it}: {_e}")

            frac = 0.25 + 0.70 * (it / float(max_iters))
            _emit_pct(frac, f"Iteration {it}/{max_iters}")

            _process_gui_events_safely()

    if not early_stopped:
        used_iters = max_iters

    # ----------------------------
    # Save result (keep FITS-friendly order: (C,H,W))
    # ----------------------------
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

    hdr0['MF_SR']     = (int(r), 'Super-resolution factor (1 := native)')
    if r > 1:
        hdr0['MF_SRSIG']  = (float(sr_sigma), 'Gaussian sigma for SR PSF fit (pixels at native)')
        hdr0['MF_SRIT']   = (int(sr_psf_opt_iters), 'SR-PSF solver iters')

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

    return safe_out_path



# -----------------------------
# Worker
# -----------------------------

class MultiFrameDeconvWorkerSport(QObject):
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
                # NEW SR forwards
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