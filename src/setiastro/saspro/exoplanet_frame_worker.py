# src/setiastro/saspro/exoplanet_frame_worker.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
GUI-free workers for the exoplanet / transient tool.

Frame load + software bin + saturation map (+ SEP catalogue), per-frame
aperture photometry, and a single SEP catalogue -- each runnable in a
ProcessPoolExecutor child (spawn) so every core works.  Threads can't do this:
sep.extract and sep.sum_circle hold the GIL for their whole call, so a thread
pool runs them one at a time.  Import-light on purpose: no PyQt, no
exoplanet_detector, so spawning a worker stays cheap.

Binned frames travel through a FrameStore -- one float32 file mapped by every
process.  A worker writes its frame's slot, the parent maps the same file:
no 50 MB arrays pickled through a pipe, and the frame set no longer has to
fit in RAM (the OS pages it; with RAM to spare it simply stays cached).
"""
from __future__ import annotations

import glob
import os
import tempfile
import uuid

import numpy as np

STORE_PREFIX = "saspro_exo_"


# ═══════════════════════════════ frame store ══════════════════════════════

class FrameStore:
    """n binned frames of one shape in a temp file, float32, mapped r/w."""

    def __init__(self, n: int, frame_shape, directory: str | None = None):
        self.directory = directory or tempfile.gettempdir()
        self.shape = (int(n),) + tuple(int(s) for s in frame_shape)
        self.dtype = np.dtype(np.float32)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.path = os.path.join(
            self.directory, f"{STORE_PREFIX}{os.getpid()}_{uuid.uuid4().hex[:10]}.f32")
        _allocate(self.path, self.nbytes)
        self.mm = np.memmap(self.path, dtype=self.dtype, mode="r+", shape=self.shape)

    def desc(self) -> dict:
        return {"path": self.path, "shape": self.shape, "dtype": self.dtype.str}

    def frame(self, i: int) -> np.ndarray:
        return self.mm[int(i)]

    def close(self) -> bool:
        """Unmap and delete.  False if something still holds a view (Windows
        refuses to delete a mapped file); sweep_stale() retries later."""
        mm, self.mm = self.mm, None
        own = _MM.pop(self.path, None)       # this process's task-side mapping
        for m in (mm, own):
            if m is not None:
                try:
                    m._mmap.close()
                except Exception:
                    pass
        del mm, own
        try:
            os.remove(self.path)
            return True
        except OSError:
            _PENDING.add(self.path)
            return False

    @staticmethod
    def fits(n: int, frame_shape, directory: str | None = None) -> bool:
        """Enough free disk for the store, with 2 GB to spare."""
        import shutil
        need = int(n) * int(np.prod(frame_shape)) * 4
        try:
            free = shutil.disk_usage(directory or tempfile.gettempdir()).free
        except OSError:
            return False
        return free > need * 1.05 + 2 * 1024 ** 3


def _allocate(path: str, nbytes: int):
    """
    Create `path` at `nbytes` without writing them.  On Windows,
    file.truncate() goes through the C runtime's _chsize, which extends a file
    by WRITING every zero: 22.7 GB for 485 binned 47 MP frames, measured at
    58 s of disk writes on the GUI thread before the first worker started.  A
    sparse file extended with SetEndOfFile takes ~1 ms; ranges never written
    read back as zeros.  POSIX ftruncate is sparse already.
    """
    with open(path, "wb") as fh:
        if os.name == "nt":
            try:
                if _win_sparse_extend(fh, int(nbytes)):
                    return
            except Exception:
                pass
        fh.truncate(int(nbytes))


def _win_sparse_extend(fh, nbytes: int) -> bool:
    """Mark fh's file sparse and set its end; False if the volume can't
    (FAT/exFAT have no sparse files) -- the caller then truncates."""
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                    ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    k32.DeviceIoControl.restype = wintypes.BOOL
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                     ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
    k32.SetFilePointerEx.restype = wintypes.BOOL
    k32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    k32.SetEndOfFile.restype = wintypes.BOOL
    h = wintypes.HANDLE(msvcrt.get_osfhandle(fh.fileno()))
    ret = wintypes.DWORD()
    FSCTL_SET_SPARSE = 0x000900C4
    if not k32.DeviceIoControl(h, FSCTL_SET_SPARSE, None, 0, None, 0, ctypes.byref(ret), None):
        return False
    return bool(k32.SetFilePointerEx(h, nbytes, None, 0) and k32.SetEndOfFile(h))


_PENDING: set = set()


def sweep_stale(directory: str | None = None):
    """Delete store files left by crashed sessions or by an earlier failed
    close of our own.  A file whose owning process is still alive (another
    SASpro window) is left alone."""
    for p in list(_PENDING):
        try:
            os.remove(p)
            _PENDING.discard(p)
        except OSError:
            pass
    try:
        import psutil
        alive = psutil.pid_exists
    except Exception:
        alive = None
    me = os.getpid()
    for p in glob.glob(os.path.join(directory or tempfile.gettempdir(), STORE_PREFIX + "*.f32")):
        try:
            pid = int(os.path.basename(p)[len(STORE_PREFIX):].split("_", 1)[0])
        except ValueError:
            continue
        if pid == me or alive is None or alive(pid):
            continue
        try:
            os.remove(p)
        except OSError:
            pass


_MM: dict = {}


def _store_view(store: dict):
    """The store mapped in this process (one mapping per store file)."""
    mm = _MM.get(store["path"])
    if mm is None:
        for k in list(_MM):            # a new store: drop maps of old ones
            _MM.pop(k, None)
        mm = np.memmap(store["path"], dtype=np.dtype(store["dtype"]), mode="r+",
                       shape=tuple(store["shape"]))
        _MM[store["path"]] = mm
    return mm


# ═══════════════════════════════ helpers ══════════════════════════════════

def bin_frame(img: np.ndarray, b: int) -> np.ndarray:
    """
    Average b x b blocks, float32.  Rows / columns that don't fill a whole
    block are dropped -- the same convention as bin2x2_numba.
    """
    a = np.asarray(img, dtype=np.float32)
    b = int(b)
    if b <= 1:
        return np.ascontiguousarray(a)
    H, W = a.shape[:2]
    h, w = H // b, W // b
    a = a[: h * b, : w * b]
    if a.ndim == 2 or a.shape[2] <= 4:
        # INTER_AREA at a whole-number factor is the exact block mean
        # (float32 rounding), ~6x faster than the numpy reshape-mean.
        try:
            import cv2
            out = cv2.resize(np.ascontiguousarray(a), (w, h), interpolation=cv2.INTER_AREA)
            if out.ndim == a.ndim:
                return out
        except Exception:
            pass
    if a.ndim == 2:
        return a.reshape(h, b, w, b).mean(axis=(1, 3), dtype=np.float32)
    return a.reshape(h, b, w, b, a.shape[2]).mean(axis=(1, 3), dtype=np.float32)


def adc_shift(img: np.ndarray) -> int | None:
    """
    How many low bits are zero in every pixel of a 16-bit integer frame that
    load_image scaled by 1/65535: 4 for a 12-bit ADC shifted into 16 bits
    (every value a multiple of 16), 2 for 14-bit, 0 for a true 16-bit ADC.
    Camera gain charts and EGAIN are in the ADC's own units, so this is the
    factor between them and the file's.  None when the values don't come
    back as integers (rescaled or debayered data).
    """
    a = np.asarray(img)
    s = (a[::7, ::7] if a.ndim == 2 else a[::7, ::7, 0]).astype(np.float64) * 65535.0
    q = np.rint(s)
    if s.size == 0 or float(np.max(np.abs(s - q))) > 0.01:
        return None
    q = q.astype(np.int64)
    q = q[(q > 0) & (q < 65535)]        # 0 = no data, 65535 = a software clip
    if q.size < 1000:
        return None
    low = int(np.bitwise_or.reduce(q & 0xFF))
    if low == 0:
        return 8
    return (low & -low).bit_length() - 1


def frame_time(path: str, hdr):
    """(jd_mid_utc, exptime_s, how) from a frame's header, or None."""
    from setiastro.saspro.exoplanet_transit import frame_mid_time_utc
    keys = ("UT-OBS", "DATE-OBS", "DATE-END", "MJD-OBS", "DATE-AVG", "MJD-AVG",
            "TIME-OBS", "JD", "EXPTIME", "EXPOSURE")
    try:
        if os.path.splitext(path)[1].lower() == ".xisf":
            kw = {}
            if isinstance(hdr, dict):
                kw = ((hdr.get("image_meta") or {}).get("FITSKeywords")
                      or hdr.get("FITSKeywords") or {})
            if not kw:
                from setiastro.saspro.legacy.xisf import XISF
                kw = XISF(path).get_images_metadata()[0].get("FITSKeywords", {}) or {}
            like = {k: kw[k][0].get("value") for k in keys if k in kw and kw[k]}
            t = frame_mid_time_utc(like)
        elif hdr is not None:
            t = frame_mid_time_utc(hdr)
        else:
            t = None
    except Exception:
        t = None
    return t if (t and t[0] is not None) else None


_THREADS_SET = False


def _worker_threads():
    """One OpenCV thread per worker process -- the pool owns the cores.  Never
    in the GUI process (thread fallback), where the setting is process-wide."""
    global _THREADS_SET
    if _THREADS_SET:
        return
    _THREADS_SET = True
    try:
        import multiprocessing as _mp
        if _mp.current_process().name == "MainProcess":
            return
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


# ═══════════════════════════════ tasks ════════════════════════════════════

def load_task(i: int, path: str, bin_factor: int, store, catalog_sigma) -> dict:
    """
    Load one frame, find its saturated pixels on the UNBINNED data (binning
    averages a clipped core away), bin it, write it to store slot i.

    Returns i, ok, err, shape, hdr, bit_depth, sat, time, mean, cat (SEP
    catalogue when catalog_sigma is given), adc_shift (16-bit integer frames
    only, see adc_shift()), and arr -- the binned frame
    itself, only when it couldn't go to the store (no store, or a frame of a
    different size).
    """
    out = {"i": int(i), "ok": False, "err": "", "path": path}
    _worker_threads()
    try:
        from setiastro.saspro.legacy.image_manager import load_image
        from setiastro.saspro import exoplanet_photometry as xphot
        img, hdr, bd, _mono = load_image(path)
        if img is None:
            out["err"] = "could not be read"
            return out
        img = np.asarray(img, dtype=np.float32)
        if str(bd or "").startswith("16-bit"):
            try:
                out["adc_shift"] = adc_shift(img)
            except Exception:
                pass
        try:
            sl, _how = xphot.saturation_level(img, hdr, bd)
            sat = xphot.saturated_coords_binned(img, sl, bin_factor=bin_factor)
        except Exception:
            sat = np.zeros((0, 2), np.float32)
        binned = bin_frame(img, bin_factor)
        del img
        plane = binned.mean(axis=2) if binned.ndim == 3 else binned
        out.update(ok=True, shape=tuple(binned.shape), hdr=hdr, bit_depth=bd, sat=sat,
                   time=frame_time(path, hdr), mean=float(np.mean(plane)), cat=None)
        if catalog_sigma:
            out["cat"] = xphot.extract_catalog(plane, float(catalog_sigma))
        if store is not None and tuple(binned.shape) == tuple(store["shape"][1:]):
            _store_view(store)[int(i)] = binned
        else:
            out["arr"] = binned
    except Exception as e:
        out["ok"] = False
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def _plane_of(i, store, arr):
    a = arr if arr is not None else _store_view(store)[int(i)]
    return a.mean(axis=2) if a.ndim == 3 else a


def catalog_task(i: int, store, arr, sigma: float):
    """SEP catalogue of one stored frame."""
    from setiastro.saspro import exoplanet_photometry as xphot
    return int(i), xphot.extract_catalog(np.ascontiguousarray(_plane_of(i, store, arr)),
                                         float(sigma))


def photometry_task(i: int, store, arr, M, ref_xy, ap, gain, sat):
    """Aperture photometry of every reference star on one stored frame."""
    from setiastro.saspro import exoplanet_photometry as xphot
    plane = _plane_of(i, store, arr)
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    if M is None:
        pred = ref_xy
    else:
        M = np.asarray(M, dtype=np.float64).reshape(2, 3)
        pred = ref_xy @ M[:, :2].T + M[:, 2]
    return int(i), xphot.measure_frame(plane, pred[:, 0], pred[:, 1], ap,
                                       gain=gain, sat_xy=sat)
