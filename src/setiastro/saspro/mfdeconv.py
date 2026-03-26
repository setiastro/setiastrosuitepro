# pro/mfdeconv.py non-sport normal version
from __future__ import annotations
import os, sys
import math
import re
import numpy as np
import time
from astropy.io import fits
from PyQt6.QtCore import QObject, pyqtSignal
from setiastro.saspro.psf_utils import compute_psf_kernel_for_image
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
from setiastro.saspro.free_torch_memory import _free_torch_memory
from setiastro.saspro.mfdeconv_earlystop import EarlyStopper
torch = None        # filled by runtime loader if available
TORCH_OK = False
NO_GRAD = contextlib.nullcontext  # fallback
_XISF_READERS = []
try:
    # e.g. your legacy module
    from setiastro.saspro.legacy import xisf as _legacy_xisf
    if hasattr(_legacy_xisf, "read"):
        _XISF_READERS.append(lambda p: _legacy_xisf.read(p))
    elif hasattr(_legacy_xisf, "open"):
        _XISF_READERS.append(lambda p: _legacy_xisf.open(p)[0])
except Exception:
    pass
try:
    # sometimes projects expose a generic load_image
    from setiastro.saspro.legacy.image_manager import load_image as _generic_load_image  # adjust if needed
    _XISF_READERS.append(lambda p: _generic_load_image(p)[0])
except Exception:
    pass

from pathlib import Path

# at top of file with the other imports

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
from collections import OrderedDict


def _ensure_scratch_dir(scratch_dir: str | None) -> str:
    if scratch_dir is None or not isinstance(scratch_dir, str) or not scratch_dir.strip():
        scratch_dir = tempfile.gettempdir()
    os.makedirs(scratch_dir, exist_ok=True)
    return scratch_dir


def _mm_unique_path(scratch_dir: str, tag: str, ext: str = ".mm") -> str:
    fd, path = tempfile.mkstemp(prefix=f"sas_{tag}_", suffix=ext, dir=scratch_dir)
    try:
        os.close(fd)
    except Exception:
        pass
    return path


def _prepare_frame_stack_memmap(
    paths: list[str],
    Ht: int,
    Wt: int,
    color_mode: str = "luma",
    *,
    scratch_dir: str | None = None,
    dtype: np.dtype | str = np.float32,
    tile_hw: tuple[int, int] = (512, 512),
    status_cb=lambda s: None,
):
    """
    Create one disk-backed memmap per input frame, already cropped to (Ht,Wt)
    and normalized to float32/float16. Returns:
      frame_infos: list[dict(path, shape, dtype)]
      hdrs:        list[fits.Header]
    Each memmap stores (H,W) or (C,H,W).
    """
    scratch_dir = _ensure_scratch_dir(scratch_dir)

    if isinstance(dtype, str):
        d = dtype.lower().strip()
        out_dtype = np.float16 if d in ("float16", "fp16", "half") else np.float32
    else:
        out_dtype = np.dtype(dtype)

    th, tw = int(tile_hw[0]), int(tile_hw[1])
    infos, hdrs = [], []

    status_cb(f"Preparing {len(paths)} frame memmaps → {scratch_dir}")

    for idx, p in enumerate(paths, start=1):
        try:
            hdr = _safe_primary_header(p)
        except Exception:
            hdr = fits.Header()
        hdrs.append(hdr)

        mode = str(color_mode).lower().strip()
        if mode == "luma":
            shape = (Ht, Wt)
            c_out = 1
        else:
            c0, _, _ = _read_shape_fast(p)
            c_out = 3 if c0 == 3 else 1
            shape = (c_out, Ht, Wt)

        mm_path = _mm_unique_path(scratch_dir, tag=f"frame_{idx:04d}", ext=".mm")
        mm = np.memmap(mm_path, mode="w+", dtype=out_dtype, shape=shape)
        mm_is_3d = (mm.ndim == 3)

        tiles = [
            (y, min(y + th, Ht), x, min(x + tw, Wt))
            for y in range(0, Ht, th)
            for x in range(0, Wt, tw)
        ]

        for (y0, y1, x0, x1) in tiles:
            t = _read_tile_fits_any(p, y0, y1, x0, x1)

            if t.dtype.kind in "ui":
                t = t.astype(np.float32) / (float(np.iinfo(t.dtype).max) or 1.0)
            else:
                t = t.astype(np.float32, copy=False)

            if not mm_is_3d:
                if t.ndim == 3:
                    t = _to_luma_local(t)
                elif t.ndim != 2:
                    t = _to_luma_local(t)
                if out_dtype != np.float32:
                    t = t.astype(out_dtype, copy=False)
                mm[y0:y1, x0:x1] = t
            else:
                if c_out == 3:
                    if t.ndim == 2:
                        t = np.stack([t, t, t], axis=0)
                    elif t.ndim == 3 and t.shape[-1] == 3:
                        t = np.moveaxis(t, -1, 0)
                    elif t.ndim == 3 and t.shape[0] == 3:
                        pass
                    else:
                        t = _to_luma_local(t)
                        t = np.stack([t, t, t], axis=0)

                    if out_dtype != np.float32:
                        t = t.astype(out_dtype, copy=False)
                    mm[:, y0:y1, x0:x1] = t
                else:
                    if t.ndim == 3:
                        t = _to_luma_local(t)
                    if out_dtype != np.float32:
                        t = t.astype(out_dtype, copy=False)
                    mm[0, y0:y1, x0:x1] = t

        try:
            mm.flush()
        except Exception:
            pass
        del mm

        infos.append({
            "path": mm_path,
            "shape": tuple(shape),
            "dtype": out_dtype,
        })

        if (idx % 8) == 0 or idx == len(paths):
            status_cb(f"Frame memmaps: {idx}/{len(paths)} ready")

    return infos, hdrs


def _make_memmap_tile_loader(frame_infos, max_open=32):
    """
    Returns tile_loader(i, y0, y1, x0, x1, csel=None) that slices from each
    prepared frame memmap. Keeps a tiny LRU of opened memmaps.
    """
    opened = OrderedDict()

    def _open_mm(i):
        fi = frame_infos[i]
        mm = np.memmap(fi["path"], mode="r", dtype=fi["dtype"], shape=fi["shape"])
        opened[i] = mm
        while len(opened) > int(max_open):
            _, old = opened.popitem(last=False)
            try:
                del old
            except Exception:
                pass
        return mm

    def tile_loader(i, y0, y1, x0, x1, csel=None):
        mm = opened.get(i)
        if mm is None:
            mm = _open_mm(i)
        else:
            opened.move_to_end(i, last=True)

        a = mm
        if a.ndim == 2:
            t = a[y0:y1, x0:x1]
        else:
            cc = 0 if csel is None else int(csel)
            t = a[cc, y0:y1, x0:x1]

        return np.array(t, copy=True)

    shp = frame_infos[0]["shape"]
    tile_loader.want_c = (shp[0] if len(shp) == 3 else 1)

    def _close():
        while opened:
            _, mm = opened.popitem(last=False)
            try:
                del mm
            except Exception:
                pass

    tile_loader.close = _close
    return tile_loader
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
    Hs, Ws = [], []
    for p in paths:
        h, w, _ = _probe_hw(p)
        h = int(h); w = int(w)
        if h > 0 and w > 0:
            Hs.append(h); Ws.append(w)

    if not Hs:
        raise ValueError("Could not determine any valid frame sizes.")
    Ht = min(Hs); Wt = min(Ws)
    if Ht < 8 or Wt < 8:
        raise ValueError(f"Intersection too small: {Ht}x{Wt}")
    return Ht, Wt


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
    Returns:
      (index, psf, mask_or_None, var_or_None, var_path_or_None, log_lines)
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
    psf = None
    k_ladder = [k_auto, max(k_auto - 4, 11), 21, 17, 15, 13, 11]
    sigma_ladder = [50.0, 25.0, 12.0, 6.0]

    tried = set()
    for det_sigma in sigma_ladder:
        for k_try in k_ladder:
            if (det_sigma, k_try) in tried:
                continue
            tried.add((det_sigma, k_try))
            try:
                out = compute_psf_kernel_for_image(arr, ksize=k_try, det_sigma=det_sigma, max_stars=80)
                psf_try = out[0] if (isinstance(out, tuple) and len(out) >= 1) else out
                if psf_try is not None:
                    psf = psf_try
                    break
            except Exception:
                psf = None
        if psf is not None:
            break

    if psf is None:
        psf = _gaussian_psf(f_whm, ksize=k_auto)

    psf = _soften_psf(_normalize_psf(psf.astype(np.float32, copy=False)), sigma_px=0.25)

    mask = None
    var = None
    var_path = None

    if make_masks or make_varmaps:
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

    fwhm_est = _psf_fwhm_px(psf)
    logs.insert(0, f"MFDeconv: PSF{i}: ksize={psf.shape[0]} | FWHM≈{fwhm_est:.2f}px")

    return i, psf, mask, var, var_path, logs

def _compute_one_worker(args):
    """
    Process-safe worker wrapper.
    Args tuple: (i, path, make_masks, make_varmaps, star_mask_cfg, varmap_cfg, Ht, Wt, color_mode)
    Returns: (i, psf, mask, var_or_None, var_path_or_None, logs)
    """
    if len(args) < 9:
        raise ValueError(f"_compute_one_worker expected 9 args, got {len(args)}")

    i, path, make_masks, make_varmaps, star_mask_cfg, varmap_cfg, Ht, Wt, color_mode = args[:9]

    try:
        hdr = _safe_primary_header(path)
    except Exception:
        hdr = fits.Header()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xisf":
        arr_all, _ = _load_image_array(path)
        arr_all = np.asarray(arr_all)
    else:
        with fits.open(path, memmap=True, ignore_missing_simple=True) as hdul:
            arr_all = None
            for h in hdul:
                if getattr(h, "data", None) is not None:
                    arr_all = np.asarray(h.data)
                    break
            if arr_all is None:
                raise ValueError(f"No image data in {path}")

    if arr_all.ndim == 3:
        if arr_all.shape[0] in (1, 3):
            arr2d = arr_all[0].astype(np.float32, copy=False)
        elif arr_all.shape[-1] in (1, 3):
            arr2d = _to_luma_local(arr_all).astype(np.float32, copy=False)
        else:
            arr2d = _to_luma_local(arr_all).astype(np.float32, copy=False)
    else:
        arr2d = np.asarray(arr_all, dtype=np.float32)

    H, W = arr2d.shape
    y0 = max(0, (H - Ht) // 2)
    x0 = max(0, (W - Wt) // 2)
    y1 = min(H, y0 + Ht)
    x1 = min(W, x0 + Wt)
    arr2d = np.ascontiguousarray(arr2d[y0:y1, x0:x1], dtype=np.float32)

    if arr2d.shape != (Ht, Wt):
        out = np.zeros((Ht, Wt), dtype=np.float32)
        oy = (Ht - arr2d.shape[0]) // 2
        ox = (Wt - arr2d.shape[1]) // 2
        out[oy:oy+arr2d.shape[0], ox:ox+arr2d.shape[1]] = arr2d
        arr2d = out

    return _compute_frame_assets(
        i, arr2d, hdr,
        make_masks=bool(make_masks),
        make_varmaps=bool(make_varmaps),
        star_mask_cfg=star_mask_cfg,
        varmap_cfg=varmap_cfg,
    )

def _normalize_assets_result(res):
    """
    Normalize worker results to:
      (i, psf, mask, var_or_None, var_path_or_None, logs_list)

    Supported:
      - (i, psf, mask, var, logs)
      - (i, psf, mask, var, var_path, logs)
      - (i, psf, mask, var, var_mm, var_path, logs)
    """
    if not isinstance(res, (tuple, list)) or len(res) < 5:
        raise ValueError(
            f"Unexpected assets result: {type(res)} len={len(res) if hasattr(res,'__len__') else 'na'}"
        )

    i = res[0]
    psf = res[1]
    mask = res[2]
    logs = res[-1]

    middle = res[3:-1]
    var = None
    var_path = None

    for x in middle:
        if var is None and hasattr(x, "shape"):
            var = x
        if var_path is None and isinstance(x, str):
            var_path = x

    if len(res) == 5:
        var = middle[0] if middle else None
        var_path = None

    return i, psf, mask, var, var_path, logs

def _build_psf_and_assets(
    paths,
    make_masks=False,
    make_varmaps=False,
    status_cb=lambda s: None,
    save_dir: str | None = None,
    star_mask_cfg: dict | None = None,
    varmap_cfg: dict | None = None,
    max_workers: int | None = None,
    star_mask_ref_path: str | None = None,
    Ht: int | None = None,
    Wt: int | None = None,
    color_mode: str = "luma",
):
    """
    Returns:
      (psfs, masks, vars_, var_paths)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n = len(paths)

    if Ht is None or Wt is None:
        Ht, Wt = _common_hw_from_paths(paths)

    if max_workers is None:
        try:
            hw = os.cpu_count() or 4
        except Exception:
            hw = 4
        max_workers = max(1, min(4, hw // 2))

    any_xisf = any(os.path.splitext(p)[1].lower() == ".xisf" for p in paths)
    use_proc_pool = (not any_xisf) and _USE_PROCESS_POOL_FOR_ASSETS
    Executor = ProcessPoolExecutor if use_proc_pool else ThreadPoolExecutor
    pool_kind = "process" if use_proc_pool else "thread"
    status_cb(f"MFDeconv: measuring PSFs/masks/varmaps with {max_workers} {pool_kind}s…")

    def _center_pad_or_crop_2d(a2d: np.ndarray, Ht: int, Wt: int, fill: float = 1.0) -> np.ndarray:
        a2d = np.asarray(a2d, dtype=np.float32)
        H, W = int(a2d.shape[0]), int(a2d.shape[1])
        y0 = max(0, (H - Ht) // 2)
        x0 = max(0, (W - Wt) // 2)
        y1 = min(H, y0 + Ht)
        x1 = min(W, x0 + Wt)
        cropped = a2d[y0:y1, x0:x1]
        ch, cw = cropped.shape
        if ch == Ht and cw == Wt:
            return np.ascontiguousarray(cropped, dtype=np.float32)
        out = np.full((Ht, Wt), float(fill), dtype=np.float32)
        oy = (Ht - ch) // 2
        ox = (Wt - cw) // 2
        out[oy:oy+ch, ox:ox+cw] = cropped
        return out

    base_ref_mask = None
    if make_masks and star_mask_ref_path:
        try:
            status_cb(f"Star mask: using reference frame for all masks → {os.path.basename(star_mask_ref_path)}")
            ref_chw = _FRAME_LRU.get(star_mask_ref_path, Ht, Wt, "luma")
            L = ref_chw[0] if (ref_chw.ndim == 3) else ref_chw

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
            if base_ref_mask is not None and base_ref_mask.dtype != np.uint8:
                base_ref_mask = (base_ref_mask > 0.5).astype(np.uint8, copy=False)

            del L, sky_map, rms_map
            gc.collect()
        except Exception as e:
            status_cb(f"⚠️ Star mask (reference) failed: {e}. Falling back to per-frame masks.")
            base_ref_mask = None

    log_queue: SimpleQueue = SimpleQueue()

    def enqueue_logs(lines):
        for s in lines:
            log_queue.put(s)

    psfs = [None] * n
    masks = ([None] * n) if make_masks else None
    vars_ = ([None] * n) if make_varmaps else None
    var_paths = ([None] * n) if make_varmaps else None

    make_masks_in_worker = bool(make_masks and (base_ref_mask is None))

    def _compute_one(i: int, path: str):
        with threadpool_limits(limits=1):
            img_chw = _FRAME_LRU.get(path, Ht, Wt, color_mode)
            arr2d = img_chw[0] if (img_chw.ndim == 3) else img_chw
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

    with Executor(max_workers=max_workers) as ex:
        futs = []
        for i, p in enumerate(paths, start=1):
            status_cb(f"MFDeconv: measuring PSF {i}/{n} …")
            if use_proc_pool:
                futs.append(ex.submit(
                    _compute_one_worker,
                    (i, p, bool(make_masks_in_worker), bool(make_varmaps), star_mask_cfg, varmap_cfg, Ht, Wt, color_mode)
                ))
            else:
                futs.append(ex.submit(_compute_one, i, p))

        done_cnt = 0
        for fut in as_completed(futs):
            res = fut.result()
            i, psf, m, v, vpath, logs = _normalize_assets_result(res)

            idx = i - 1
            psfs[idx] = psf
            if masks is not None:
                masks[idx] = m
            if vars_ is not None:
                vars_[idx] = v
            if var_paths is not None:
                var_paths[idx] = vpath

            enqueue_logs(logs)

            done_cnt += 1
            while not log_queue.empty():
                try:
                    status_cb(log_queue.get_nowait())
                except Exception:
                    break

            if (done_cnt % 8) == 0:
                gc.collect()

    if base_ref_mask is not None and masks is not None:
        for idx in range(n):
            masks[idx] = _center_pad_or_crop_2d(base_ref_mask, int(Ht), int(Wt), fill=1.0)

    while not log_queue.empty():
        try:
            status_cb(log_queue.get_nowait())
        except Exception:
            break

    if save_dir:
        for i, k in enumerate(psfs, start=1):
            if k is not None:
                fits.PrimaryHDU(k.astype(np.float32, copy=False)).writeto(
                    os.path.join(save_dir, f"psf_{i:03d}.fit"), overwrite=True
                )

    return psfs, masks, vars_, var_paths
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

def _to_luma_local(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return a
    # (H,W,3) or (3,H,W)
    if a.ndim == 3 and a.shape[-1] == 3:
        try:
            import cv2
            return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32, copy=False)
        except Exception:
            pass
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[0] == 3:
        r, g, b = a[0], a[1], a[2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
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
    """
    One-time SEP background build; returns (sky_map, rms_map, err_scalar).

    Guarantees:
      - Always returns a 3-tuple (sky, rms, err)
      - sky/rms are float32 and same shape as img_2d
      - Robust to sep missing, sep errors, NaNs/Infs, and tiny frames
    """
    a = np.asarray(img_2d, dtype=np.float32)
    if a.ndim != 2:
        # be strict; callers expect 2D
        raise ValueError(f"_sep_background_precompute expects 2D, got shape={a.shape}")

    H, W = int(a.shape[0]), int(a.shape[1])
    if H == 0 or W == 0:
        # should never happen, but don't return empty tuple
        sky = np.zeros((H, W), dtype=np.float32)
        rms = np.ones((H, W), dtype=np.float32)
        return sky, rms, 1.0

    # --- robust fallback builder (works for any input) ---
    def _fallback():
        # Use finite-only stats if possible
        finite = np.isfinite(a)
        if finite.any():
            vals = a[finite]
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-6
        else:
            med = 0.0
            mad = 1.0
        sky = np.full((H, W), med, dtype=np.float32)
        rms = np.full((H, W), 1.4826 * mad, dtype=np.float32)
        err = float(np.median(rms))
        return sky, rms, err

    # If sep isn't available, always fallback
    if sep is None:
        return _fallback()

    # SEP is present: sanitize input and clamp tile sizes
    # sep can choke on NaNs/Infs
    if not np.isfinite(a).all():
        # replace non-finite with median of finite values (or 0)
        finite = np.isfinite(a)
        fill = float(np.median(a[finite])) if finite.any() else 0.0
        a = np.where(finite, a, fill).astype(np.float32, copy=False)

    a = np.ascontiguousarray(a, dtype=np.float32)

    # Clamp bw/bh to image size; SEP doesn't like bw/bh > dims
    bw = int(max(8, min(int(bw), W)))
    bh = int(max(8, min(int(bh), H)))

    try:
        b = sep.Background(a, bw=bw, bh=bh, fw=3, fh=3)

        sky = np.asarray(b.back(), dtype=np.float32)
        rms = np.asarray(b.rms(), dtype=np.float32)

        # Ensure shape sanity (SEP should match, but be paranoid)
        if sky.shape != a.shape or rms.shape != a.shape:
            return _fallback()

        # globalrms sometimes isn't available depending on SEP build
        err = float(getattr(b, "globalrms", np.nan))
        if not np.isfinite(err) or err <= 0:
            # robust scalar: median rms
            err = float(np.median(rms)) if rms.size else 1.0

        return sky, rms, err

    except Exception:
        # If SEP blows up for any reason, degrade gracefully
        return _fallback()



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

def _load_frame_from_info(frame_info) -> np.ndarray:
    """
    Read one prepared frame memmap back as float32 numpy.
    Returns (H,W) for luma or (C,H,W) for per-channel.
    """
    mm = np.memmap(
        frame_info["path"],
        mode="r",
        dtype=frame_info["dtype"],
        shape=tuple(frame_info["shape"]),
    )
    arr = np.asarray(mm)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    else:
        arr = np.array(arr, copy=True)  # detach from memmap handle
    return np.ascontiguousarray(arr, dtype=np.float32)


def _load_varmap_from_path(var_path: str | None, shape_2d: tuple[int, int]) -> np.ndarray | None:
    """
    Reopen an on-disk variance map lazily as float32.
    Assumes 2D map stored as float32 memmap.
    """
    if not var_path:
        return None
    H, W = map(int, shape_2d)
    mm = np.memmap(var_path, mode="r", dtype=np.float32, shape=(H, W))
    return np.array(mm, dtype=np.float32, copy=True)

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

    Supports both NumPy arrays and torch tensors.
    """
    is_torch = ("torch" in globals()) and (torch is not None) and torch.is_tensor(y)

    if is_torch:
        r = y - pred

        # auto delta
        if huber_delta < 0:
            med = torch.median(r)
            mad = torch.median(torch.abs(r - med)) + 1e-6
            rms = 1.4826 * mad
            delta = float((-huber_delta) * torch.clamp(rms, min=1e-6).item())
        else:
            delta = float(huber_delta)

        absr = torch.abs(r)

        if delta > 0.0:
            delta_t = torch.tensor(delta, dtype=r.dtype, device=r.device)
            psi_over_r = torch.where(
                absr <= delta_t,
                torch.ones_like(r, dtype=r.dtype),
                delta_t / (absr + EPS)
            )
        else:
            psi_over_r = torch.ones_like(r, dtype=r.dtype)

        if var_map is None:
            # scalar variance estimate from residual
            medv = torch.median(r)
            madv = torch.median(torch.abs(r - medv)) + 1e-6
            rmsv = 1.4826 * madv
            v = torch.clamp(rmsv * rmsv, min=1e-8)
            v = torch.full_like(r, v)
        else:
            v = var_map
            if v.ndim == 2 and r.ndim == 3:
                v = v.unsqueeze(0).expand_as(r)
            v = torch.clamp(v, min=1e-8)

        w = psi_over_r / (v + EPS)

        if mask is not None:
            m = mask if mask.ndim == w.ndim else (mask.unsqueeze(0) if w.ndim == 3 else mask)
            w = w * m

        return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- NumPy path ----------------
    r = y - pred

    if huber_delta < 0:
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-6
        rms = 1.4826 * mad
        delta = (-huber_delta) * max(rms, 1e-6)
    else:
        delta = huber_delta

    absr = np.abs(r)
    if float(delta) > 0:
        psi_over_r = np.where(absr <= delta, 1.0, delta / (absr + EPS)).astype(np.float32, copy=False)
    else:
        psi_over_r = np.ones_like(r, dtype=np.float32)

    if var_map is None:
        v = _estimate_scalar_variance(r)
    else:
        v = var_map
        if v.ndim == 2 and r.ndim == 3:
            v = v[None, ...]
        v = np.clip(v, 1e-8, None).astype(np.float32, copy=False)

    w = psi_over_r / (v + EPS)

    if mask is not None:
        m = mask if mask.ndim == w.ndim else (mask[None, ...] if w.ndim == 3 else mask)
        w = w * m

    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
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

# ── Drop-in replacement for _fft_conv_same_np ─────────────────────────────────
# Takes a pre-allocated complex scratch buffer to avoid creating A and A*Kf
# as new allocations on every call.  The scratch buffer is allocated once
# per solver run in _run_solver_core and reused across all frames/iterations.
#
# scratch shape must be (fftH, fftW//2+1) for 2D or (C, fftH, fftW//2+1) for CHW.
# Pass scratch=None to fall back to the original allocating behaviour.
 
def _fft_conv_same_np_scratch(a, Kf, kh, kw, fftH, fftW, out, scratch=None):
    import numpy.fft as fft
 
    if a.ndim == 2:
        H, W = a.shape
        A = fft.rfftn(a, s=(fftH, fftW))   # complex, shape (fftH, fftW//2+1)
        if scratch is not None and scratch.shape == A.shape:
            np.multiply(A, Kf, out=scratch)  # scratch = A * Kf  (in-place)
            y = fft.irfftn(scratch, s=(fftH, fftW))
        else:
            y = fft.irfftn(A * Kf, s=(fftH, fftW))
        sh, sw = kh // 2, kw // 2
        out[...] = y[sh:sh + H, sw:sw + W]
        return out
 
    else:  # CHW
        C, H, W = a.shape
        acc = []
        for c in range(C):
            A = fft.rfftn(a[c], s=(fftH, fftW))
            if scratch is not None and scratch.shape == A.shape:
                np.multiply(A, Kf, out=scratch)
                y = fft.irfftn(scratch, s=(fftH, fftW))
            else:
                y = fft.irfftn(A * Kf, s=(fftH, fftW))
            sh, sw = kh // 2, kw // 2
            acc.append(y[sh:sh + H, sw:sw + W])
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
    io_workers: int | None = None,     # NEW
    tile_loader=None,                  # NEW
):
    """
    Exact per-pixel median via tiling; RAM-bounded.
    Supports optional tile_loader(i,y0,y1,x0,x1,csel=None) so normal mode
    can read from prepared memmaps instead of reopening source files.
    """

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    th, tw = int(tile_hw[0]), int(tile_hw[1])

    # channel count
    if str(color_mode).lower() == "luma":
        want_c = 1
    else:
        if tile_loader is not None and hasattr(tile_loader, "want_c"):
            want_c = int(tile_loader.want_c)
        else:
            want_c = _infer_channels_from_tile(paths[0], Ht, Wt)

    seed = np.zeros((Ht, Wt), np.float32) if want_c == 1 else np.zeros((want_c, Ht, Wt), np.float32)
    tiles = [(y, min(y + th, Ht), x, min(x + tw, Wt)) for y in range(0, Ht, th) for x in range(0, Wt, tw)]
    total = len(tiles)
    n_frames = len(paths)

    # worker count
    if io_workers is None:
        try:
            _cpu = (os.cpu_count() or 4)
        except Exception:
            _cpu = 4
        io_workers = max(1, min(8, _cpu, n_frames))
    else:
        io_workers = max(1, min(int(io_workers), n_frames))

    # Torch autodetect
    TORCH_OK = False
    device = None
    if use_torch is not False:
        try:
            from setiastro.saspro.runtime_torch import import_torch
            _t = import_torch(prefer_cuda=True, status_cb=status_cb)
            dev = None
            if hasattr(_t, "cuda") and _t.cuda.is_available():
                dev = _t.device("cuda")
            elif hasattr(_t.backends, "mps") and _t.backends.mps.is_available():
                dev = _t.device("mps")
            else:
                dev = None
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

        def _read_slab_for_channel(csel=None):
            """
            Returns slab of shape (N, h, w) float32.
            Uses tile_loader if provided, otherwise falls back to source-file tile reads.
            """
            def _load_one(i):
                if tile_loader is not None:
                    t = tile_loader(i, y0, y1, x0, x1, csel=csel)
                else:
                    t = _read_tile_fits_any(paths[i], y0, y1, x0, x1)

                    if t.dtype.kind in "ui":
                        t = t.astype(np.float32) / (float(np.iinfo(t.dtype).max) or 1.0)
                    else:
                        t = t.astype(np.float32, copy=False)

                    if want_c == 1:
                        if t.ndim == 3:
                            t = _to_luma_local(t)
                        elif t.ndim != 2:
                            t = _to_luma_local(t)
                    else:
                        if t.ndim == 2:
                            pass
                        elif t.ndim == 3 and t.shape[-1] == 3:
                            t = t[..., csel]
                        elif t.ndim == 3 and t.shape[0] == 3:
                            t = t[csel]
                        else:
                            t = _to_luma_local(t)

                t = np.ascontiguousarray(t, dtype=np.float32)
                return i, t

            slab = np.empty((n_frames, h, w), np.float32)
            done_local = 0

            with ThreadPoolExecutor(max_workers=min(io_workers, n_frames)) as ex:
                futures = [ex.submit(_load_one, i) for i in range(n_frames)]
                for fut in as_completed(futures):
                    i, t2d = fut.result()

                    if t2d.shape != (h, w):
                        raise RuntimeError(
                            f"Tile read mismatch at frame {i}: got {t2d.shape}, expected {(h, w)} "
                            f"tile={(y0, y1, x0, x1)}"
                        )

                    slab[i] = t2d
                    done_local += 1

                    if (done_local & 7) == 0 or done_local == n_frames:
                        tile_base = done / total
                        tile_span = 1.0 / total
                        inner = done_local / n_frames
                        progress_cb(tile_base + 0.8 * tile_span * inner, _tile_msg(done + 1, total))

            return slab

        try:
            if want_c == 1:
                t0 = time.perf_counter()
                slab = _read_slab_for_channel()
                t1 = time.perf_counter()

                if TORCH_OK:
                    import torch as _t
                    slab_t = _t.as_tensor(slab, device=device, dtype=_t.float32)
                    med_t = slab_t.median(dim=0).values
                    med_np = med_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    del slab_t, med_t
                else:
                    med_np = np.median(slab, axis=0).astype(np.float32, copy=False)

                t2 = time.perf_counter()
                seed[y0:y1, x0:x1] = med_np
                status_cb(
                    f"seed tile {y0}:{y1},{x0}:{x1} I/O={t1-t0:.3f}s  "
                    f"median={'GPU' if TORCH_OK else 'CPU'}={t2-t1:.3f}s"
                )

            else:
                for c in range(want_c):
                    slab = _read_slab_for_channel(csel=c)

                    if TORCH_OK:
                        import torch as _t
                        slab_t = _t.as_tensor(slab, device=device, dtype=_t.float32)
                        med_t = slab_t.median(dim=0).values
                        med_np = med_t.detach().cpu().numpy().astype(np.float32, copy=False)
                        del slab_t, med_t
                    else:
                        med_np = np.median(slab, axis=0).astype(np.float32, copy=False)

                    seed[c, y0:y1, x0:x1] = med_np

        except RuntimeError as e:
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
        progress_cb(done / total, _tile_msg(done, total))

        try:
            del slab
        except Exception:
            pass
        try:
            del med_np
        except Exception:
            pass

        if (done & 3) == 0:
            _process_gui_events_safely()
            gc.collect()

    status_cb(f"Median seed: want_c={want_c}, seed_shape={seed.shape}")
    return seed


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

class _ShapeProbe:
    """
    Minimal shape-only stand-in for ndarray objects used by
    _ensure_mask_list() / _ensure_var_list().

    Supports:
      - .shape
      - .ndim
      - a[0] for CHW-like probes, returning a 2D probe
    """
    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)

    def __getitem__(self, idx):
        # Only behavior we actually need: a[0] on 3D CHW probes
        if self.ndim == 3 and idx == 0:
            return _ShapeProbe(self.shape[1:])
        raise TypeError(f"_ShapeProbe only supports [0] on 3D shapes, got idx={idx!r} for shape={self.shape}")

    def __repr__(self):
        return f"_ShapeProbe(shape={self.shape})"

def _write_temp_array_mm(arr, tag="mfblend"):
    path = _mm_unique_path(tempfile.gettempdir(), tag=tag, ext=".mm")
    mm = np.memmap(path, mode="w+", dtype=np.float32, shape=arr.shape)
    mm[...] = arr.astype(np.float32, copy=False)
    mm.flush()
    del mm
    return path        
# -----------------------------
# Core
# -----------------------------
def multiframe_deconv_normal_rebuild(
    paths,
    out_path,
    iters=20,
    kappa=2.0,
    color_mode="luma",
    seed_mode: str = "robust",
    seed_image=None,
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
    rejection_strength: float = 1.0,
    # SR options
    super_res_factor: int = 1,
    sr_sigma: float = 1.1,
    sr_psf_opt_iters: int = 250,
    sr_psf_opt_lr: float = 0.1,
    # normal-mode memory knobs retained
    force_cpu: bool = False,
    star_mask_ref_path: str | None = None,
    low_mem: bool = False,
):
    """
    NORMAL rebuild, OOM-safe version:
      - no full-frame stack kept in RAM
      - per-frame memmaps used for seed + solver
      - median seed built tile-by-tile from memmaps
      - solver streams one frame at a time
      - optional variance maps reopened lazily from disk
      - rejection/mask behavior matches sport
    """

    max_iters = max(1, int(iters))
    min_iters = max(1, int(min_iters))
    if min_iters > max_iters:
        min_iters = max_iters

    rejection_strength = float(max(0.0, min(1.0, rejection_strength)))

    def _emit_pct(pct: float, msg: str | None = None):
        pct = float(max(0.0, min(1.0, pct)))
        status_cb(f"__PROGRESS__ {pct:.4f}" + (f" {msg}" if msg else ""))

    def _bg_rms_from_first_frame():
        try:
            y0 = _load_frame_native(0)
            y0l = y0 if y0.ndim == 2 else y0[0]
            med = float(np.median(y0l))
            mad = float(np.median(np.abs(y0l - med))) + 1e-6
            return 1.4826 * mad
        except Exception:
            return 0.0

    def _seed_bootstrap_streaming_from_infos(
        frame_infos_local,
        *,
        bootstrap_frames=20,
        clip_sigma=5.0,
        progress_cb=lambda f, m="": None,
    ):
        K = max(1, min(int(bootstrap_frames), len(frame_infos_local)))

        x0 = _load_frame_from_info(frame_infos_local[0]).astype(np.float32, copy=False)
        mean = x0.copy()
        m2 = np.zeros_like(mean, dtype=np.float32)
        count = 1

        for i in range(1, K):
            x = _load_frame_from_info(frame_infos_local[i]).astype(np.float32, copy=False)
            count += 1
            d = x - mean
            mean += d / count
            m2 += d * (x - mean)
            progress_cb(i / max(1, K), "μ-σ bootstrap")

            del x
            if low_mem:
                gc.collect()

        var = m2 / max(1, count - 1)
        sigma = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32, copy=False)
        lo = mean - float(clip_sigma) * sigma
        hi = mean + float(clip_sigma) * sigma

        acc = np.zeros_like(mean, dtype=np.float32)
        n = 0

        for i in range(len(frame_infos_local)):
            x = _load_frame_from_info(frame_infos_local[i]).astype(np.float32, copy=False)
            x = np.clip(x, lo, hi, out=x)
            acc += x
            n += 1
            progress_cb(0.5 + 0.5 * (i + 1) / len(frame_infos_local), "clipped mean")

            del x
            if low_mem and ((i & 3) == 0):
                gc.collect()

        seed = (acc / max(1, n)).astype(np.float32, copy=False)
        return seed[0] if (seed.ndim == 3 and seed.shape[0] == 1) else seed

    status_cb(f"MFDeconv[NORMAL-REBUILD]: preparing {len(paths)} aligned frames…")
    _emit_pct(0.02, "preparing")

    Ht, Wt = _common_hw_from_paths(paths)
    _emit_pct(0.05, "probing")

    if low_mem:
        try:
            _FRAME_LRU.cap = 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Build disk-backed frame memmaps ONCE
    # ------------------------------------------------------------------
    frame_infos, hdrs = _prepare_frame_stack_memmap(
        paths,
        Ht,
        Wt,
        color_mode=color_mode,
        scratch_dir=None,
        dtype=(np.float16 if low_mem else np.float32),
        tile_hw=(512, 512),
        status_cb=status_cb,
    )

    tile_loader = _make_memmap_tile_loader(frame_infos, max_open=(2 if low_mem else 8))

    def _load_frame_native(i: int) -> np.ndarray:
        return _load_frame_from_info(frame_infos[i])

    # lightweight shape probes only, not full data arrays
    data_probe = [_ShapeProbe(fi["shape"]) for fi in frame_infos]

    relax = 0.7
    global torch, TORCH_OK
    torch = None
    TORCH_OK = False
    cuda_ok = mps_ok = dml_ok = False
    dml_device = None

    # ------------------------------------------------------------------
    # torch import / backend detect
    # ------------------------------------------------------------------
    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(prefer_cuda=True, status_cb=status_cb)
        TORCH_OK = True

        try:
            cuda_ok = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        try:
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            mps_ok = False
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

    if force_cpu:
        status_cb("⚠️ CPU-only debug mode: disabling PyTorch path.")
        TORCH_OK = False

    use_torch = bool(TORCH_OK)

    if use_torch:
        try:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = False
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    _process_gui_events_safely()
    status_cb(f"MFDeconv[NORMAL-REBUILD]: rejection_strength={rejection_strength:.3f}")

    # ------------------------------------------------------------------
    # PSFs + optional assets
    # ------------------------------------------------------------------
    psfs, masks_auto, vars_auto, var_paths_auto = _build_psf_and_assets(
        paths,
        make_masks=bool(use_star_masks),
        make_varmaps=bool(use_variance_maps),
        status_cb=status_cb,
        save_dir=None,
        star_mask_cfg=star_mask_cfg,
        varmap_cfg=varmap_cfg,
        star_mask_ref_path=star_mask_ref_path,
        Ht=Ht, Wt=Wt, color_mode=color_mode,
    )

    gc.collect()

    # ------------------------------------------------------------------
    # SR lift of PSFs if needed
    # ------------------------------------------------------------------
    r = int(max(1, super_res_factor))
    if r > 1:
        status_cb(f"MFDeconv: Super-resolution r={r} with σ={sr_sigma} — solving SR PSFs…")
        _process_gui_events_safely()
        sr_psfs = []
        for i, k_native in enumerate(psfs, start=1):
            h = _solve_super_psf_from_native(
                k_native,
                r=r,
                sigma=float(sr_sigma),
                iters=int(sr_psf_opt_iters),
                lr=float(sr_psf_opt_lr),
            )
            sr_psfs.append(h)
            status_cb(f"  SR-PSF{i}: native {k_native.shape[0]} → {h.shape[0]} (sum={h.sum():.6f})")
        psfs = sr_psfs

    flip_psf = [_flip_kernel(k) for k in psfs]
    _emit_pct(0.20, "PSF Ready")

    # ------------------------------------------------------------------
    # seed creation
    # ------------------------------------------------------------------
    def _seed_progress(frac, msg):
        _emit_pct(0.25 + 0.15 * float(frac), f"seed: {msg}")

    seed_mode_s = str(seed_mode).lower().strip()
    if seed_mode_s not in ("integrated", "robust", "median"):
        seed_mode_s = "robust"

    if seed_mode_s == "integrated":
        if seed_image is None:
            raise ValueError("seed_mode='integrated' requested but no seed_image was provided.")
        status_cb("MFDeconv: Using integrated prepass seed image…")
        seed_native = np.asarray(seed_image, dtype=np.float32)

        if str(color_mode).lower() == "luma":
            if seed_native.ndim == 3:
                if seed_native.shape[0] == 1:
                    seed_native = seed_native[0]
                elif seed_native.shape[-1] == 1:
                    seed_native = seed_native[..., 0]
                elif seed_native.shape[0] == 3:
                    seed_native = np.mean(seed_native, axis=0, dtype=np.float32)
                elif seed_native.shape[-1] == 3:
                    seed_native = np.mean(seed_native, axis=-1, dtype=np.float32)
        else:
            if seed_native.ndim == 2:
                seed_native = seed_native[None, ...]
            elif (
                seed_native.ndim == 3
                and seed_native.shape[0] not in (1, 3)
                and seed_native.shape[-1] in (1, 3)
            ):
                seed_native = np.moveaxis(seed_native, -1, 0)

        if seed_native.shape[-2:] != (Ht, Wt):
            seed_native = _center_crop(seed_native, Ht, Wt)

        seed_native = _sanitize_numeric(seed_native)

    elif seed_mode_s == "median":
        status_cb("MFDeconv: Building median seed (tiled, streaming)…")
        seed_native = _seed_median_streaming(
            paths,
            Ht,
            Wt,
            color_mode=color_mode,
            tile_hw=(256, 256),
            status_cb=status_cb,
            progress_cb=_seed_progress,
            io_workers=(2 if low_mem else 4),
            tile_loader=tile_loader,
        )

    else:
        status_cb("MFDeconv: Building robust seed (live μ-σ streaming)…")
        seed_native = _seed_bootstrap_streaming_from_infos(
            frame_infos,
            bootstrap_frames=(8 if low_mem else 20),
            clip_sigma=5.0,
            progress_cb=_seed_progress,
        )

    if r > 1:
        if seed_native.ndim == 2:
            x_seed = _upsample_sum(seed_native / (r * r), r, target_hw=(Ht * r, Wt * r))
        else:
            C, Hn, Wn = seed_native.shape
            x_seed = np.stack(
                [_upsample_sum(seed_native[c] / (r * r), r, target_hw=(Hn * r, Wn * r)) for c in range(C)],
                axis=0
            )
    else:
        x_seed = seed_native

    if str(color_mode).lower() != "luma" and x_seed.ndim == 2:
        x_seed = x_seed[None, ...]

    # Release native seed if it is no longer needed separately
    if seed_native is not x_seed:
        try:
            del seed_native
        except Exception:
            pass
    gc.collect()

    # ------------------------------------------------------------------
    # masks / variances
    # ------------------------------------------------------------------
    auto_masks = masks_auto if use_star_masks else None

    star_mask_list_base = _ensure_mask_list(auto_masks, data_probe) if auto_masks is not None else None
    rej_mask_list_base = _ensure_mask_list(masks, data_probe) if masks is not None else None

    # user-provided variances still supported
    if variances is not None:
        var_list_base = _ensure_var_list(variances, data_probe)
        var_path_list_base = [None] * len(paths)
    else:
        var_list_base = _ensure_var_list(vars_auto, data_probe) if vars_auto is not None else [None] * len(paths)
        var_path_list_base = list(var_paths_auto) if (var_paths_auto is not None) else [None] * len(paths)

    def _prep_star_mask_for_mf(m):
        if m is None:
            return None
        mm = np.asarray(m, dtype=np.float32)
        if mm.ndim == 3:
            mm = mm[0]
        return np.clip(mm, 0.0, 1.0).astype(np.float32, copy=False)

    if star_mask_list_base is not None:
        star_mask_list_base = [_prep_star_mask_for_mf(m) for m in star_mask_list_base]

    def _build_effective_mask_list(rejection_mode: str):
        if rejection_mode not in ("none", "full"):
            raise ValueError(f"Unknown rejection_mode: {rejection_mode}")

        if rej_mask_list_base is not None:
            rej_mask_list = []
            for m in rej_mask_list_base:
                if m is None:
                    rej_mask_list.append(None)
                    continue
                mm = np.asarray(m, dtype=np.float32)
                if mm.ndim == 3:
                    mm = mm[0]

                if rejection_mode == "none":
                    mm = np.ones_like(mm, dtype=np.float32)
                else:
                    # incoming 1=valid, 0=rejected
                    # convert to 0=valid, 1=rejected
                    mm = np.where(mm > 0.0, 0.0, 1.0).astype(np.float32, copy=False)

                rej_mask_list.append(mm.astype(np.float32, copy=False))
        else:
            rej_mask_list = None

        star_mask_list = star_mask_list_base

        if rej_mask_list is None and star_mask_list is None:
            mask_list = _ensure_mask_list(None, data_probe)
        elif rej_mask_list is None:
            mask_list = star_mask_list
        elif star_mask_list is None:
            mask_list = rej_mask_list
        else:
            mask_list = []
            for rm, sm in zip(rej_mask_list, star_mask_list):
                if rm is None and sm is None:
                    mask_list.append(None)
                elif rm is None:
                    mask_list.append(sm)
                elif sm is None:
                    mask_list.append(rm)
                else:
                    mask_list.append(np.clip(rm * sm, 0.0, 1.0).astype(np.float32, copy=False))

        if mask_list is not None:
            cleaned = []
            for i, m in enumerate(mask_list):
                if m is None:
                    cleaned.append(None)
                    continue
                mm = np.asarray(m, dtype=np.float32)
                if mm.ndim == 3:
                    mm = mm[0]
                mm = np.clip(mm, 0.0, 1.0).astype(np.float32, copy=False)
                cleaned.append(mm)
            mask_list = cleaned

        return mask_list

    def _mask_for_run(i, like_img, mask_list):
        if mask_list is None:
            return np.ones((like_img.shape[-2], like_img.shape[-1]), dtype=np.float32)
        try:
            m = mask_list[i]
        except Exception:
            m = None
        if m is None:
            return np.ones((like_img.shape[-2], like_img.shape[-1]), dtype=np.float32)
        mm = np.asarray(m, dtype=np.float32)
        if mm.ndim == 3:
            mm = mm[0]
        mm = _center_crop(mm, like_img.shape[-2], like_img.shape[-1]).astype(np.float32, copy=False)
        return np.clip(mm, 0.0, 1.0)

    def _var_for_run(i, like_img):
        # first prefer in-memory variance supplied by caller
        try:
            v = var_list_base[i]
        except Exception:
            v = None

        if v is not None:
            vv = np.asarray(v, dtype=np.float32)
            if vv.ndim == 3:
                vv = vv[0]
            vv = _center_crop(vv, like_img.shape[-2], like_img.shape[-1])
            return np.clip(vv, 1e-8, None).astype(np.float32, copy=False)

        # then lazy-load from varmap memmap path if available
        try:
            vp = var_path_list_base[i]
        except Exception:
            vp = None

        if vp:
            vv = _load_varmap_from_path(vp, (like_img.shape[-2], like_img.shape[-1]))
            if vv is not None:
                return np.clip(vv, 1e-8, None).astype(np.float32, copy=False)

        return None

    # ------------------------------------------------------------------
    # solver core
    # ------------------------------------------------------------------

    def _run_solver_core(mask_list, run_label: str, save_intermediate_this_run: bool):
        iter_dir = None
        hdr0_seed = None
 
        x = np.array(x_seed, dtype=np.float32, copy=True)
 
        if save_intermediate_this_run:
            iter_dir = _iter_folder(out_path)
            if rejection_strength not in (0.0, 1.0):
                iter_dir = iter_dir + f"_{run_label}"
            status_cb(f"MFDeconv: Intermediate outputs ({run_label}) → {iter_dir}")
            try:
                hdr0_seed = _safe_primary_header(paths[0])
            except Exception:
                hdr0_seed = fits.Header()
            _save_iter_image(x, hdr0_seed, iter_dir, "seed", color_mode)
 
        bg_est = _bg_rms_from_first_frame()
        status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g}) [{run_label}]")
        _process_gui_events_safely()
 
        import numpy.fft as _fft
 
        if use_torch:
            x_t    = _to_t(_contig(x))
            num    = torch.zeros_like(x_t)
            den    = torch.zeros_like(x_t)
            psf_t  = [_to_t(_contig(k))[None, None]  for k in psfs]
            psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]
            fft_scratch = None   # not used on torch path
        else:
            x_t = x.astype(np.float32, copy=False)
 
            # FIX 1: metadata only, no FFTs yet
            np_meta = []
            max_fftH = max_fftW = 0
            for k in psfs:
                kh, kw = k.shape
                fftH, fftW = _fftshape_same(x_t.shape[-2], x_t.shape[-1], kh, kw)
                np_meta.append((kh, kw, fftH, fftW))
                max_fftH = max(max_fftH, fftH)
                max_fftW = max(max_fftW, fftW)
 
            # FIX 2: fixed scratch arrays allocated once
            num      = np.zeros_like(x_t)
            den      = np.zeros_like(x_t)
            pred_buf = np.empty_like(x_t)
            tmp_out  = np.empty_like(x_t)
 
            # FIX 3 (NEW): one complex scratch for FFT multiply step.
            # Shape: (max_fftH, max_fftW//2+1) — covers all PSF sizes.
            # dtype complex64 (not complex128) halves the scratch size:
            # for 11600×8700 + ksize=21 → (11620, 5861) complex64 ≈ 272MB
            # vs complex128 ≈ 544MB.  numpy rfftn returns complex128 by default;
            # we cast A to complex64 before the multiply to use this buffer.
            fft_scratch_shape = (max_fftH, max_fftW // 2 + 1)
            fft_scratch = np.empty(fft_scratch_shape, dtype=np.complex64)
            status_cb(
                f"MFDeconv: FFT scratch buffer {fft_scratch_shape} "
                f"complex64 ≈ {fft_scratch.nbytes // 1024 // 1024} MB"
            )
 
        status_cb(f"Starting First Multiplicative Iteration… ({run_label})")
        _process_gui_events_safely()
 
        cm = _safe_inference_context() if use_torch else NO_GRAD
        rho_is_l2 = (str(rho).lower() == "l2")
        local_delta = 0.0 if rho_is_l2 else huber_delta
        used_iters_local = 0
        early_stopped_local = False
 
        auto_delta_cache = None
        if use_torch and (huber_delta < 0) and (not rho_is_l2):
            auto_delta_cache = [None] * len(paths)
 
        early = EarlyStopper(
            tol_upd_floor=2e-4,
            tol_rel_floor=5e-4,
            early_frac=0.40,
            ema_alpha=0.5,
            patience=2,
            min_iters=min_iters,
            status_cb=status_cb
        )
 
        # helper: conv using scratch buffer
        def _conv_np(a, Kf, kh, kw, fftH, fftW, out):
            """FFT conv reusing fft_scratch for the A*Kf intermediate."""
            import numpy.fft as fft
            if a.ndim == 2:
                H, W = a.shape
                A = fft.rfftn(a, s=(fftH, fftW)).astype(np.complex64, copy=False)
                s = fft_scratch[:fftH, :fftW // 2 + 1]
                np.multiply(A, Kf, out=s)
                y = fft.irfftn(s, s=(fftH, fftW))
                sh, sw = kh // 2, kw // 2
                out[...] = y[sh:sh + H, sw:sw + W]
                return out
            else:
                C, H, W = a.shape
                acc = []
                for c in range(C):
                    A = fft.rfftn(a[c], s=(fftH, fftW)).astype(np.complex64, copy=False)
                    s = fft_scratch[:fftH, :fftW // 2 + 1]
                    np.multiply(A, Kf, out=s)
                    y = fft.irfftn(s, s=(fftH, fftW))
                    sh, sw = kh // 2, kw // 2
                    acc.append(y[sh:sh + H, sw:sw + W])
                out[...] = np.stack(acc, 0)
                return out
 
        with cm():
            for it in range(1, max_iters + 1):
 
                # ── TORCH ITERATION ───────────────────────────────────────────
                if use_torch:
                    num.zero_()
                    den.zero_()
 
                    for fidx, (wk, wkT) in enumerate(zip(psf_t, psfT_t)):
                        y_nat = _load_frame_native(fidx)
                        m2d   = _mask_for_run(fidx, y_nat, mask_list)
                        v2d   = _var_for_run(fidx, y_nat)
 
                        yt = torch.as_tensor(y_nat, dtype=x_t.dtype, device=x_t.device)
                        mt = None if m2d is None else torch.as_tensor(m2d, dtype=x_t.dtype, device=x_t.device)
                        vt = None if v2d is None else torch.as_tensor(v2d, dtype=x_t.dtype, device=x_t.device)
 
                        if r > 1:
                            pred_super = _conv_same_torch(x_t, wk)
                            pred_low   = _downsample_avg_t(pred_super, r)
                            delta_use  = local_delta
                            if auto_delta_cache is not None:
                                if (auto_delta_cache[fidx] is None) or (it % 5 == 1):
                                    rnat = yt - pred_low
                                    med  = torch.median(rnat)
                                    mad  = torch.median(torch.abs(rnat - med)) + 1e-6
                                    auto_delta_cache[fidx] = float(
                                        (-huber_delta) * torch.clamp(1.4826 * mad, min=1e-6).item()
                                    )
                                delta_use = auto_delta_cache[fidx]
                            wmap_low = _weight_map(yt, pred_low, delta_use, var_map=vt, mask=mt)
                            up_y     = _upsample_sum_t(wmap_low * yt,       r)
                            up_pred  = _upsample_sum_t(wmap_low * pred_low, r)
                            num += _conv_same_torch(up_y,    wkT)
                            den += _conv_same_torch(up_pred, wkT)
                            del pred_super, pred_low, wmap_low, up_y, up_pred
                        else:
                            pred      = _conv_same_torch(x_t, wk)
                            delta_use = local_delta
                            if auto_delta_cache is not None:
                                if (auto_delta_cache[fidx] is None) or (it % 5 == 1):
                                    rnat = yt - pred
                                    med  = torch.median(rnat)
                                    mad  = torch.median(torch.abs(rnat - med)) + 1e-6
                                    auto_delta_cache[fidx] = float(
                                        (-huber_delta) * torch.clamp(1.4826 * mad, min=1e-6).item()
                                    )
                                delta_use = auto_delta_cache[fidx]
                            wmap    = _weight_map(yt, pred, delta_use, var_map=vt, mask=mt)
                            num    += _conv_same_torch(wmap * yt,   wkT)
                            den    += _conv_same_torch(wmap * pred, wkT)
                            del pred, wmap
 
                        del yt, mt, vt, y_nat, m2d, v2d
 
                        if low_mem and cuda_ok:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            gc.collect()
 
                    # in-place update
                    neutral = (den.abs() < 1e-12) & (num.abs() < 1e-12)
                    den.add_(EPS)
                    num.div_(den)
                    num[neutral] = 1.0
                    num.clamp_(1.0 / kappa, kappa)
 
                    upd_med = torch.median(torch.abs(num - 1.0))
                    um      = float(upd_med.detach().cpu().item())
 
                    x_next     = x_t.mul(num).clamp_(min=0.0)
                    rel_change = (
                        torch.median(torch.abs(x_next - x_t)) /
                        (torch.median(torch.abs(x_t)) + 1e-8)
                    )
                    rc = float(rel_change.detach().cpu().item())
 
                    if early.step(it, max_iters, um, rc):
                        x_t = x_next
                        used_iters_local  = it
                        early_stopped_local = True
                        _process_gui_events_safely()
                        break
 
                    x_t.mul_(1.0 - relax)
                    x_t.add_(relax * x_next)
                    del x_next
 
                # ── NUMPY ITERATION ───────────────────────────────────────────
                else:
                    num.fill(0.0)
                    den.fill(0.0)
 
                    for fidx, (psf_k, psf_kT, (kh, kw, fftH, fftW)) in enumerate(
                        zip(psfs, flip_psf, np_meta)
                    ):
                        # on-demand FFTs cast to complex64 to match scratch buffer
                        Kf  = _fft.rfftn(np.fft.ifftshift(psf_k),  s=(fftH, fftW)).astype(np.complex64, copy=False)
                        KTf = _fft.rfftn(np.fft.ifftshift(psf_kT), s=(fftH, fftW)).astype(np.complex64, copy=False)
 
                        y_nat = _load_frame_native(fidx)
                        m2d   = _mask_for_run(fidx, y_nat, mask_list)
                        v2d   = _var_for_run(fidx, y_nat)
 
                        if r > 1:
                            _conv_np(x_t, Kf, kh, kw, fftH, fftW, pred_buf)
                            pred_low = _downsample_avg(pred_buf, r)
                            wmap_low = _weight_map(y_nat, pred_low, local_delta,
                                                   var_map=v2d, mask=m2d)
                            up_y    = _upsample_sum(wmap_low * y_nat,    r,
                                                    target_hw=pred_buf.shape[-2:])
                            up_pred = _upsample_sum(wmap_low * pred_low, r,
                                                    target_hw=pred_buf.shape[-2:])
                            _conv_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out)
                            num += tmp_out
                            _conv_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out)
                            den += tmp_out
                            del pred_low, wmap_low, up_y, up_pred
                        else:
                            _conv_np(x_t, Kf, kh, kw, fftH, fftW, pred_buf)
                            wmap    = _weight_map(y_nat, pred_buf, local_delta,
                                                  var_map=v2d, mask=m2d)
                            up_y    = wmap * y_nat
                            up_pred = wmap * pred_buf
                            _conv_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out)
                            num += tmp_out
                            _conv_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out)
                            den += tmp_out
                            del wmap, up_y, up_pred
 
                        del Kf, KTf, y_nat, m2d, v2d
 
                        if low_mem and ((fidx & 3) == 0):
                            gc.collect()
 
                    # in-place update
                    den      += EPS
                    neutral   = (np.abs(den) < 1e-12 + EPS) & (np.abs(num) < 1e-12)
                    np.divide(num, den, out=num)
                    num[neutral] = 1.0
                    np.clip(num, 1.0 / kappa, kappa, out=num)
 
                    um = float(np.median(np.abs(num - 1.0)))
 
                    np.multiply(x_t, num, out=tmp_out)
                    np.clip(tmp_out, 0.0, None, out=tmp_out)
 
                    rc = float(
                        np.median(np.abs(tmp_out - x_t)) /
                        (np.median(np.abs(x_t)) + 1e-8)
                    )
 
                    if early.step(it, max_iters, um, rc):
                        np.copyto(x_t, tmp_out)
                        used_iters_local  = it
                        early_stopped_local = True
                        _process_gui_events_safely()
                        break
 
                    x_t  *= (1.0 - relax)
                    x_t  += relax * tmp_out
 
                # ── shared post-iteration bookkeeping ────────────────────────
                if save_intermediate_this_run and (it % int(max(1, save_every)) == 0):
                    try:
                        x_np = (x_t.detach().cpu().numpy().astype(np.float32)
                                if use_torch else x_t.astype(np.float32))
                        _save_iter_image(x_np, hdr0_seed, iter_dir, f"iter_{it:03d}", color_mode)
                    except Exception as _e:
                        status_cb(f"Intermediate save failed at iter {it}: {_e}")
 
                frac = 0.25 + 0.70 * (it / float(max_iters))
                _emit_pct(frac, f"Iteration {it}/{max_iters} [{run_label}]")
                _process_gui_events_safely()
 
                if low_mem:
                    gc.collect()
 
        if not early_stopped_local:
            used_iters_local = max_iters
 
        x_final_local = (x_t.detach().cpu().numpy().astype(np.float32)
                         if use_torch else x_t.astype(np.float32))
 
        if x_final_local.ndim == 3:
            if x_final_local.shape[0] not in (1, 3) and x_final_local.shape[-1] in (1, 3):
                x_final_local = np.moveaxis(x_final_local, -1, 0)
            if x_final_local.shape[0] == 1:
                x_final_local = x_final_local[0]
 
        try:
            if use_torch:
                try:
                    del num, den
                except Exception:
                    pass
                try:
                    del psf_t, psfT_t
                except Exception:
                    pass
                _free_torch_memory()
        except Exception:
            pass
 
        return x_final_local, used_iters_local, early_stopped_local
    # ------------------------------------------------------------------
    # endpoint strategy exactly like sport
    # ------------------------------------------------------------------
    if rejection_strength <= 0.0:
        status_cb("MFDeconv: running NO-REJECTION solve.")
        mask_list_none = _build_effective_mask_list("none")
        x_final, used_iters, early_stopped = _run_solver_core(
            mask_list_none,
            run_label="no_rejection",
            save_intermediate_this_run=save_intermediate,
        )

    elif rejection_strength >= 1.0:
        status_cb("MFDeconv: running FULL-REJECTION solve.")
        mask_list_full = _build_effective_mask_list("full")
        x_final, used_iters, early_stopped = _run_solver_core(
            mask_list_full,
            run_label="full_rejection",
            save_intermediate_this_run=save_intermediate,
        )

    else:
        status_cb("MFDeconv: running blended rejection solve (two endpoint passes).")

        mask_list_none = _build_effective_mask_list("none")
        x0, used_iters0, early_stopped0 = _run_solver_core(
            mask_list_none,
            run_label="no_rejection",
            save_intermediate_this_run=save_intermediate,
        )

        x0_mm_path = _write_temp_array_mm(x0, tag="mf_no_rej")
        x0_shape = x0.shape
        del x0, mask_list_none
        gc.collect()

        if use_torch and cuda_ok:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        mask_list_full = _build_effective_mask_list("full")
        x1, used_iters1, early_stopped1 = _run_solver_core(
            mask_list_full,
            run_label="full_rejection",
            save_intermediate_this_run=save_intermediate,
        )

        x0_mm = np.memmap(x0_mm_path, mode="r", dtype=np.float32, shape=x0_shape)
        x_final = (
            x0_mm.astype(np.float32, copy=False) * (1.0 - rejection_strength)
            + x1.astype(np.float32, copy=False) * rejection_strength
        ).astype(np.float32, copy=False)

        del x0_mm, x1, mask_list_full
        gc.collect()
        try:
            os.remove(x0_mm_path)
        except Exception:
            pass

        used_iters = max(int(used_iters0), int(used_iters1))
        early_stopped = bool(early_stopped0 and early_stopped1)

        status_cb(
            f"MFDeconv: blended final result using rejection_strength={rejection_strength:.3f} "
            f"(no_rejection * {1.0 - rejection_strength:.3f} + full_rejection * {rejection_strength:.3f})"
        )

    # ------------------------------------------------------------------
    # save result
    # ------------------------------------------------------------------
    _emit_pct(0.97, "saving")

    try:
        hdr0 = _safe_primary_header(paths[0])
    except Exception:
        hdr0 = fits.Header()

    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution')
    hdr0['MF_COLOR'] = (str(color_mode), 'Color mode used')
    hdr0['MF_RHO'] = (str(rho), 'Loss: huber|l2')
    hdr0['MF_HDEL'] = (float(huber_delta), 'Huber delta (>0 abs, <0 autoxRMS)')
    hdr0['MF_MASK'] = (bool(use_star_masks), 'Used auto star masks')
    hdr0['MF_VAR'] = (bool(use_variance_maps), 'Used auto variance maps')
    hdr0['MF_RSTR'] = (float(rejection_strength), 'Rejection-map blend strength')

    hdr0['MF_SR'] = (int(r), 'Super-resolution factor (1 := native)')
    if r > 1:
        hdr0['MF_SRSIG'] = (float(sr_sigma), 'Gaussian sigma for SR PSF fit (pixels at native)')
        hdr0['MF_SRIT'] = (int(sr_psf_opt_iters), 'SR-PSF solver iters')

    hdr0['MF_ITMAX'] = (int(max_iters), 'Requested max iterations')
    hdr0['MF_ITERS'] = (int(used_iters), 'Actual iterations run')
    hdr0['MF_ESTOP'] = (bool(early_stopped), 'Early stop triggered')

    if isinstance(x_final, np.ndarray):
        if x_final.ndim == 2:
            hdr0['MF_SHAPE'] = (f"{x_final.shape[0]}x{x_final.shape[1]}", 'Saved as 2D image (HxW)')
        elif x_final.ndim == 3:
            C, H, W = x_final.shape
            hdr0['MF_SHAPE'] = (f"{C}x{H}x{W}", 'Saved as 3D cube (CxHxW)')

    save_path = _sr_out_path(out_path, super_res_factor)
    safe_out_path = _nonclobber_path(str(save_path))
    if safe_out_path != str(save_path):
        status_cb(f"Output exists — saving as: {safe_out_path}")

    fits.PrimaryHDU(data=x_final, header=hdr0).writeto(safe_out_path, overwrite=False)

    status_cb(f"✅ MFDeconv saved: {safe_out_path}  (iters used: {used_iters}{', early stop' if early_stopped else ''})")
    _emit_pct(1.00, "done")
    _process_gui_events_safely()

    try:
        tile_loader.close()
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
                 seed_image=None,
                 rejection_map=None,
                 rejection_group_maps=None,
                 prepass_payload=None,
                 reference_header=None,
                 rejection_strength: float = 1.0,
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
        self.min_iters = min_iters
        self.star_mask_cfg = star_mask_cfg or {}
        self.varmap_cfg = varmap_cfg or {}
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
        self.seed_image = None if seed_image is None else np.asarray(seed_image, dtype=np.float32, copy=True)

        self.rejection_map = rejection_map
        self.rejection_group_maps = rejection_group_maps or {}
        self.prepass_payload = prepass_payload or {}
        self.reference_header = reference_header
        self.rejection_strength = float(rejection_strength)

    def _log(self, s):
        self.progress.emit(s)

    def _get_rejection_payload(self):
        return {
            "rejection_map": self.rejection_map,
            "rejection_group_maps": self.rejection_group_maps,
            "prepass_payload": self.prepass_payload,
            "reference_header": self.reference_header,
        }

    def _frame_key_candidates(self, path: str):
        import os
        p = str(path)
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        return [p, os.path.normpath(p), base, stem]

    def _rejection_entry_for_path(self, path: str):
        import os

        rej = self.rejection_map
        if rej is None or not isinstance(rej, dict):
            return None

        cands = self._frame_key_candidates(path)

        for k in cands:
            if k in rej:
                return rej[k]

        try:
            norm_map = {os.path.normpath(str(k)): v for k, v in rej.items()}
            for k in cands:
                nk = os.path.normpath(str(k))
                if nk in norm_map:
                    return norm_map[nk]
        except Exception:
            pass

        try:
            for ck in cands:
                sck = str(ck)
                for rk, rv in rej.items():
                    srk = str(rk)
                    if srk == sck or srk.endswith(sck):
                        return rv
        except Exception:
            pass

        return None

    def _entry_to_valid_mask(self, entry, path: str):
        import numpy as np
        import os

        if entry is None:
            self._log(f"MFDeconv mask build: no entry for {os.path.basename(path)}")
            return None

        try:
            shp = tuple(_read_shape_fast(path))
        except Exception as e:
            self._log(f"MFDeconv mask build: shape read failed for {os.path.basename(path)}: {e}")
            return None

        if len(shp) == 2:
            H, W = int(shp[0]), int(shp[1])
        elif len(shp) == 3:
            if shp[0] in (1, 3, 4):
                _, H, W = int(shp[0]), int(shp[1]), int(shp[2])
            else:
                H, W, _ = int(shp[0]), int(shp[1]), int(shp[2])
        else:
            self._log(f"MFDeconv mask build: unsupported shape {shp} for {os.path.basename(path)}")
            return None

        valid = np.ones((H, W), dtype=np.float32)

        if not isinstance(entry, (list, tuple)) or len(entry) == 0:
            self._log(f"MFDeconv mask build: empty entry for {os.path.basename(path)}")
            return valid

        first = entry[0]

        if isinstance(first, (list, tuple)) and len(first) == 2:
            rej_count = 0
            for it in entry:
                try:
                    x, y = int(it[0]), int(it[1])
                except Exception:
                    continue
                if 0 <= y < H and 0 <= x < W:
                    if valid[y, x] != 0.0:
                        valid[y, x] = 0.0
                        rej_count += 1

            self._log(
                f"MFDeconv XY mask: {os.path.basename(path)} "
                f"points={len(entry)} rejected_px={rej_count}"
            )
            return valid

        if isinstance(first, (list, tuple)) and len(first) == 3:
            total_rej = 0

            for it in entry:
                try:
                    x0, y0, m = int(it[0]), int(it[1]), it[2]
                except Exception:
                    continue

                if m is None:
                    continue

                mm = np.asarray(m)
                if mm.ndim != 2:
                    mm = np.squeeze(mm)
                    if mm.ndim != 2:
                        continue

                th, tw = mm.shape
                if th <= 0 or tw <= 0:
                    continue

                mm_rej = (mm != 0)

                xs0 = max(0, x0)
                ys0 = max(0, y0)
                xs1 = min(W, x0 + tw)
                ys1 = min(H, y0 + th)

                if xs1 <= xs0 or ys1 <= ys0:
                    continue

                mx0 = xs0 - x0
                my0 = ys0 - y0
                mx1 = mx0 + (xs1 - xs0)
                my1 = my0 + (ys1 - ys0)

                tile_rej = mm_rej[my0:my1, mx0:mx1]
                if tile_rej.size == 0:
                    continue

                    # noqa: never reached in original logic? keep structure simple

                region = valid[ys0:ys1, xs0:xs1]
                total_rej += int(np.count_nonzero(tile_rej & (region > 0.0)))
                region[tile_rej] = 0.0

            self._log(
                f"MFDeconv tile mask: {os.path.basename(path)} "
                f"tiles={len(entry)} rejected_px={total_rej}"
            )
            return valid

        self._log(f"MFDeconv mask build: unknown entry format for {os.path.basename(path)}")
        return None

    def _build_rejection_masks_for_paths(self):
        import numpy as np
        import os

        masks = []
        found_any = False
        total_rejected = 0

        if self.rejection_map is None:
            self._log("MFDeconv rejection masks: no rejection_map payload")
            return None

        for p in self.aligned_paths:
            entry = self._rejection_entry_for_path(p)
            m = self._entry_to_valid_mask(entry, p)

            if m is not None:
                found_any = True
                rej_px = int(np.count_nonzero(m == 0.0))
                total_rejected += rej_px
                self._log(
                    f"MFDeconv rejection mask: {os.path.basename(p)} "
                    f"rejected_px={rej_px} shape={m.shape}"
                )
            else:
                self._log(
                    f"MFDeconv rejection mask: {os.path.basename(p)} "
                    f"no usable per-frame rejection entry"
                )

            masks.append(m)

        if not found_any:
            self._log("MFDeconv rejection masks: none built")
            return None

        self._log(f"MFDeconv rejection masks: built {len(masks)} mask(s), total_rejected_px={total_rejected}")
        return masks

    def run(self):
        try:
            rej_masks = self._build_rejection_masks_for_paths()

            try:
                self.progress.emit(
                    "MFDeconv prepass payload: "
                    f"rejection_map={'yes' if self.rejection_map is not None else 'no'}, "
                    f"group_maps={'yes' if bool(self.rejection_group_maps) else 'no'}, "
                    f"usable_masks={'yes' if rej_masks is not None else 'no'}"
                )
            except Exception:
                pass

            self._log(f"MFDeconv rejection strength={self.rejection_strength:.3f}")

            out = multiframe_deconv_normal_rebuild(
                self.aligned_paths,
                self.output_path,
                iters=self.iters,
                kappa=self.kappa,
                color_mode=self.color_mode,
                seed_mode=self.seed_mode,
                seed_image=self.seed_image,
                huber_delta=self.huber_delta,
                masks=rej_masks,
                use_star_masks=self.use_star_masks,
                use_variance_maps=self.use_variance_maps,
                rho=self.rho,
                min_iters=self.min_iters,
                status_cb=self._log,
                star_mask_cfg=self.star_mask_cfg,
                varmap_cfg=self.varmap_cfg,
                save_intermediate=self.save_intermediate,
                rejection_strength=self.rejection_strength,
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