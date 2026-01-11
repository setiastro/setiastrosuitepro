# src/setiastro/saspro/imageops/serloader.py
from __future__ import annotations

import os
import io
import mmap
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Sequence, Union
from collections import OrderedDict

import numpy as np


import cv2


from PIL import Image


# ---------------------------------------------------------------------
# SER format notes (commonly used by FireCapture / SharpCap / etc.)
# - Header is 178 bytes (SER v3 style) and begins with ASCII signature
#   typically "LUCAM-RECORDER" padded to 14 bytes.
# - Most fields are little-endian; header contains an "Endian" flag.
# - Frame data follows immediately after header, then optional timestamps
#   (8 bytes per frame) at end.
# ---------------------------------------------------------------------

SER_HEADER_SIZE = 178
SER_SIGNATURE_LEN = 14

# Common SER color IDs (seen in the wild)
SER_COLOR = {
    0: "MONO",       # mono
    8: "RGB",        # RGB24/RGB48 depending pixel depth
    9: "BGR",        # BGR24/BGR48
    10: "RGBA",
    11: "BGRA",
    12: "BAYER_RGGB",
    13: "BAYER_GRBG",
    14: "BAYER_GBRG",
    15: "BAYER_BGGR",
}

BAYER_NAMES = {"BAYER_RGGB", "BAYER_GRBG", "BAYER_GBRG", "BAYER_BGGR"}


@dataclass
class SerMeta:
    path: str
    width: int
    height: int
    frames: int
    pixel_depth: int            # bits per sample (8/16 typically)
    color_id: int
    color_name: str
    little_endian: bool
    data_offset: int
    frame_bytes: int
    has_timestamps: bool

    observer: str = ""
    instrument: str = ""
    telescope: str = ""


def _decode_cstr(b: bytes) -> str:
    try:
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _bytes_per_sample(pixel_depth_bits: int) -> int:
    return 1 if int(pixel_depth_bits) <= 8 else 2


def _is_bayer(color_name: str) -> bool:
    return color_name in BAYER_NAMES


def _is_rgb(color_name: str) -> bool:
    return color_name in {"RGB", "BGR", "RGBA", "BGRA"}


def _roi_evenize_for_bayer(x: int, y: int) -> Tuple[int, int]:
    """Ensure ROI origin is even-even so Bayer phase doesn't flip."""
    if x & 1:
        x -= 1
    if y & 1:
        y -= 1
    return max(0, x), max(0, y)


def _cv2_debayer(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """
    mosaic: uint8/uint16, shape (H,W)
    returns: RGB uint8/uint16, shape (H,W,3)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not available for debayer fallback.")

    code_map = {
        "BAYER_RGGB": cv2.COLOR_BayerRG2RGB,
        "BAYER_BGGR": cv2.COLOR_BayerBG2RGB,
        "BAYER_GBRG": cv2.COLOR_BayerGB2RGB,
        "BAYER_GRBG": cv2.COLOR_BayerGR2RGB,
    }
    code = code_map.get(pattern)
    if code is None:
        raise ValueError(f"Unknown Bayer pattern: {pattern}")
    return cv2.cvtColor(mosaic, code)


def _try_numba_debayer(mosaic: np.ndarray, pattern: str) -> Optional[np.ndarray]:
    """
    Try to use SASpro's fast debayer if available.
    Expected functions (from your memory):
      - debayer_raw_fast / debayer_fits_fast (names may differ in your tree)
    We keep this very defensive; if not found, return None.
    """
    # Try a few likely import locations without hard failing
    candidates = [
        ("setiastro.saspro.imageops.debayer", "debayer_raw_fast"),
        ("setiastro.saspro.imageops.debayer", "debayer_fits_fast"),
        ("setiastro.saspro.imageops.debayer_fast", "debayer_raw_fast"),
        ("setiastro.saspro.imageops.debayer_fast", "debayer_fits_fast"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue

            # Many fast debayers accept a mosaic and a bayer string or enum.
            # We'll try a couple calling conventions.
            try:
                out = fn(mosaic, pattern)  # type: ignore
                if out is not None:
                    return out
            except Exception:
                pass

            try:
                out = fn(mosaic)  # type: ignore
                if out is not None:
                    return out
            except Exception:
                pass
        except Exception:
            continue
    return None


class _LRUCache:
    """Tiny LRU cache for decoded frames."""
    def __init__(self, max_items: int = 8):
        self.max_items = int(max_items)
        self._d: "OrderedDict[Tuple, np.ndarray]" = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value: np.ndarray):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


class SERReader:
    """
    Memory-mapped SER reader with:
    - header parsing (common v3 layout)
    - random frame access
    - optional ROI (with Bayer parity protection)
    - optional debayer
    - tiny LRU cache for smooth preview scrubbing
    """

    def __init__(self, path: str, *, cache_items: int = 10):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

        self.meta = self._parse_header(self._mm)
        self.meta.path = self.path
        self._cache = _LRUCache(max_items=cache_items)

    def close(self):
        try:
            self._cache.clear()
        except Exception:
            pass
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------- header parsing ----------------

    @staticmethod
    def _parse_header(mm: mmap.mmap) -> SerMeta:
        if mm.size() < SER_HEADER_SIZE:
            raise ValueError("File too small to be a SER file.")

        hdr = mm[:SER_HEADER_SIZE]

        sig = hdr[:SER_SIGNATURE_LEN]
        sig_txt = _decode_cstr(sig)

        # Be permissive: many SERs start with LUCAM-RECORDER
        if "LUCAM" not in sig_txt.upper():
            # still try parsing; some writers differ, but fields often match
            pass

        # Layout (little-endian) commonly:
        # 0: 14 bytes signature
        # 14: uint32 LuID
        # 18: uint32 ColorID
        # 22: uint32 LittleEndian (0/1)
        # 26: uint32 ImageWidth
        # 30: uint32 ImageHeight
        # 34: uint32 PixelDepth
        # 38: uint32 FrameCount
        # 42: char[40] Observer
        # 82: char[40] Instrument
        # 122: char[40] Telescope
        # 162: uint64 DateTime
        # 170: uint64 DateTimeUTC
        try:
            (lu_id, color_id, little_endian_u32,
             w, h, pixel_depth, frames) = struct.unpack_from("<7I", hdr, SER_SIGNATURE_LEN)
        except Exception as e:
            raise ValueError(f"Failed to parse SER header fields: {e}")

        little_endian = bool(little_endian_u32)

        observer = _decode_cstr(hdr[42:82])
        instrument = _decode_cstr(hdr[82:122])
        telescope = _decode_cstr(hdr[122:162])

        color_name = SER_COLOR.get(int(color_id), f"UNKNOWN({color_id})")

        bps = _bytes_per_sample(int(pixel_depth))

        # channels per pixel:
        # - MONO or BAYER: 1 sample per pixel
        # - RGB/BGR: 3
        # - RGBA/BGRA: 4 (rare in SER)
        if color_name in {"RGB", "BGR"}:
            channels = 3
        elif color_name in {"RGBA", "BGRA"}:
            channels = 4
        else:
            channels = 1

        frame_bytes = int(w) * int(h) * int(channels) * int(bps)
        data_offset = SER_HEADER_SIZE

        # timestamps detection
        expected_no_ts = data_offset + frames * frame_bytes
        expected_with_ts = expected_no_ts + frames * 8
        size = mm.size()
        has_ts = (size == expected_with_ts)

        return SerMeta(
            path="",
            width=int(w),
            height=int(h),
            frames=int(frames),
            pixel_depth=int(pixel_depth),
            color_id=int(color_id),
            color_name=color_name,
            little_endian=little_endian,
            data_offset=data_offset,
            frame_bytes=int(frame_bytes),
            has_timestamps=bool(has_ts),
            observer=observer,
            instrument=instrument,
            telescope=telescope,
        )

    # ---------------- core access ----------------

    def frame_offset(self, i: int) -> int:
        i = int(i)
        if i < 0 or i >= self.meta.frames:
            raise IndexError(f"Frame index {i} out of range (0..{self.meta.frames-1})")
        return self.meta.data_offset + i * self.meta.frame_bytes

    def get_frame(
        self,
        i: int,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,  # x,y,w,h
        debayer: bool = True,
        to_float01: bool = False,
        force_rgb: bool = False,
    ) -> np.ndarray:
        """
        Returns:
          - MONO: (H,W) uint8/uint16 or float32 [0..1]
          - RGB:  (H,W,3) uint8/uint16 or float32 [0..1]

        roi is applied before debayer (and ROI origin evenized for Bayer).
        """
        meta = self.meta

        # Cache key includes ROI + flags
        roi_key = None if roi is None else tuple(int(v) for v in roi)
        key = (int(i), roi_key, bool(debayer), bool(to_float01), bool(force_rgb))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        off = self.frame_offset(i)
        buf = self._mm[off:off + meta.frame_bytes]

        bps = _bytes_per_sample(meta.pixel_depth)
        if bps == 1:
            dtype = np.uint8
        else:
            dtype = np.uint16

        # Determine channels stored
        color_name = meta.color_name
        if color_name in {"RGB", "BGR"}:
            ch = 3
        elif color_name in {"RGBA", "BGRA"}:
            ch = 4
        else:
            ch = 1

        arr = np.frombuffer(buf, dtype=dtype)

        # byteswap if big-endian storage (rare, but spec supports it)
        if (dtype == np.uint16) and (not meta.little_endian):
            arr = arr.byteswap()

        if ch == 1:
            img = arr.reshape(meta.height, meta.width)
        else:
            img = arr.reshape(meta.height, meta.width, ch)

        # ROI (apply before debayer; for Bayer enforce even-even origin)
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            x = max(0, min(meta.width - 1, x))
            y = max(0, min(meta.height - 1, y))
            w = max(1, min(meta.width - x, w))
            h = max(1, min(meta.height - y, h))

            if _is_bayer(color_name) and debayer:
                x, y = _roi_evenize_for_bayer(x, y)
                w = max(1, min(meta.width - x, w))
                h = max(1, min(meta.height - y, h))

            img = img[y:y + h, x:x + w]

        # Convert BGR->RGB if needed
        if color_name == "BGR" and img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., ::-1].copy()

        # Debayer if needed
        if _is_bayer(color_name):
            if debayer:
                mosaic = img if img.ndim == 2 else img[..., 0]
                out = _try_numba_debayer(mosaic, color_name)
                if out is None:
                    out = _cv2_debayer(mosaic, color_name)
                img = out
            else:
                # keep mosaic as mono
                img = img if img.ndim == 2 else img[..., 0]

        # Force RGB for mono (useful for consistent preview pipeline)
        if force_rgb and img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # Normalize to float01
        if to_float01:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)
                img = np.clip(img, 0.0, 1.0)

        self._cache.put(key, img)
        return img

    def get_timestamp_ns(self, i: int) -> Optional[int]:
        """
        If timestamps exist, returns the 64-bit timestamp value for frame i.
        (Interpretation depends on writer; often 100ns ticks or nanoseconds.)
        """
        meta = self.meta
        if not meta.has_timestamps:
            return None
        i = int(i)
        if i < 0 or i >= meta.frames:
            return None
        ts_base = meta.data_offset + meta.frames * meta.frame_bytes
        off = ts_base + i * 8
        b = self._mm[off:off + 8]
        if len(b) != 8:
            return None
        (v,) = struct.unpack("<Q", b)
        return int(v)



# -----------------------------
# Common reader interface/meta
# -----------------------------

@dataclass
class PlanetaryMeta:
    """
    Common metadata shape used by SERViewer / stacker.
    """
    path: str
    width: int
    height: int
    frames: int
    pixel_depth: int              # 8/16 typical (AVI usually 8)
    color_name: str               # "MONO", "RGB", "BGR", "BAYER_*", etc
    has_timestamps: bool = False
    source_kind: str = "unknown"  # "ser" / "avi" / "sequence"
    file_list: Optional[List[str]] = None


class PlanetaryFrameSource:
    """
    Minimal protocol-like base. (Duck-typed by viewer/stacker)
    """
    meta: PlanetaryMeta
    path: str

    def close(self) -> None:
        raise NotImplementedError

    def get_frame(
        self,
        i: int,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        debayer: bool = True,
        to_float01: bool = False,
        force_rgb: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError


# -----------------------------
# AVI reader (OpenCV)
# -----------------------------

class AVIReader(PlanetaryFrameSource):
    """
    Frame-accurate random access using cv2.VideoCapture.
    Notes:
      - Many codecs only support approximate seeking; good enough for preview/scrub.
      - Frames come out as BGR uint8 by default.
    """
    def __init__(self, path: str, *, cache_items: int = 10):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required to read AVI files.")
        self.path = os.fspath(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {self.path}")

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # AVI decoded frames are almost always 8-bit
        self.meta = PlanetaryMeta(
            path=self.path,
            width=w,
            height=h,
            frames=max(0, n),
            pixel_depth=8,
            color_name="BGR",
            has_timestamps=False,
            source_kind="avi",
        )

        self._cache = _LRUCache(max_items=cache_items)

    def close(self):
        try:
            self._cache.clear()
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _read_raw_frame_bgr(self, i: int) -> np.ndarray:
        i = int(i)
        if i < 0 or (self.meta.frames > 0 and i >= self.meta.frames):
            raise IndexError(f"Frame index {i} out of range")

        # Seek
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to read frame {i}")

        # frame is BGR uint8, shape (H,W,3)
        return frame

    def get_frame(
        self,
        i: int,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        debayer: bool = True,
        to_float01: bool = False,
        force_rgb: bool = False,
    ) -> np.ndarray:
        roi_key = None if roi is None else tuple(int(v) for v in roi)
        key = ("avi", int(i), roi_key, bool(to_float01), bool(force_rgb))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        bgr = self._read_raw_frame_bgr(i)

        # ROI
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            H, W = bgr.shape[:2]
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            w = max(1, min(W - x, w))
            h = max(1, min(H - y, h))
            bgr = bgr[y:y + h, x:x + w]

        # BGR -> RGB
        rgb = bgr[..., ::-1].copy()

        img: np.ndarray = rgb

        # force_rgb no-op (already rgb)
        if to_float01:
            img = img.astype(np.float32) / 255.0

        self._cache.put(key, img)
        return img


# -----------------------------
# Image-sequence reader
# -----------------------------

def _imread_any(path: str) -> np.ndarray:
    """
    Read PNG/JPG/TIF/etc into numpy.
    Tries cv2 first (fast), falls back to PIL.
    Returns:
      - grayscale: (H,W) uint8/uint16
      - color:     (H,W,3) uint8/uint16 in RGB (we normalize to RGB)
    """
    p = os.fspath(path)

    # Prefer cv2 if available
    if cv2 is not None:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None:
            # cv2 gives:
            # - gray: HxW
            # - color: HxWx3 (BGR)
            # - sometimes HxWx4 (BGRA)
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., :3]  # drop alpha if present
                img = img[..., ::-1].copy()  # BGR -> RGB
            return img

    # PIL fallback
    if Image is None:
        raise RuntimeError("Neither OpenCV nor PIL are available to read images.")
    im = Image.open(p)
    # Preserve 16-bit if possible; PIL handles many TIFFs.
    if im.mode in ("I;16", "I;16B", "I"):
        arr = np.array(im)
        return arr
    if im.mode in ("L",):
        return np.array(im)
    im = im.convert("RGB")
    return np.array(im)


def _infer_bit_depth(arr: np.ndarray) -> int:
    if arr.dtype == np.uint16:
        return 16
    if arr.dtype == np.uint8:
        return 8
    # if float, assume 32 for “depth”
    if arr.dtype in (np.float32, np.float64):
        return 32
    return 8


class ImageSequenceReader(PlanetaryFrameSource):
    """
    Reads a list of image files as frames.
    Supports random access; caches decoded frames for smooth scrubbing.
    """
    def __init__(self, files: Sequence[str], *, cache_items: int = 10):
        flist = [os.fspath(f) for f in files]
        if not flist:
            raise ValueError("Empty image sequence.")
        self.files = flist
        self.path = flist[0]

        # Probe first frame
        first = _imread_any(flist[0])
        h, w = first.shape[:2]
        depth = _infer_bit_depth(first)
        if first.ndim == 2:
            cname = "MONO"
        else:
            cname = "RGB"

        self.meta = PlanetaryMeta(
            path=self.path,
            width=int(w),
            height=int(h),
            frames=len(flist),
            pixel_depth=int(depth),
            color_name=cname,
            has_timestamps=False,
            source_kind="sequence",
            file_list=list(flist),
        )

        self._cache = _LRUCache(max_items=cache_items)

    def close(self):
        try:
            self._cache.clear()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def get_frame(
        self,
        i: int,
        *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        debayer: bool = True,
        to_float01: bool = False,
        force_rgb: bool = False,
    ) -> np.ndarray:
        i = int(i)
        if i < 0 or i >= self.meta.frames:
            raise IndexError(f"Frame index {i} out of range (0..{self.meta.frames-1})")

        roi_key = None if roi is None else tuple(int(v) for v in roi)
        key = ("seq", i, roi_key, bool(to_float01), bool(force_rgb))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        img = _imread_any(self.files[i])

        # Basic consistency checks (don’t hard fail; some sequences have slight differences)
        # If sizes differ, we’ll just use whatever comes back for that frame.
        H, W = img.shape[:2]

        # ROI
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            w = max(1, min(W - x, w))
            h = max(1, min(H - y, h))
            img = img[y:y + h, x:x + w]

        # Force RGB for mono
        if force_rgb and img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # Normalize to float01
        if to_float01:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)
                img = np.clip(img, 0.0, 1.0)

        self._cache.put(key, img)
        return img


# -----------------------------
# Factory
# -----------------------------

def open_planetary_source(
    path_or_files: Union[str, Sequence[str]],
    *,
    cache_items: int = 10,
) -> PlanetaryFrameSource:
    """
    Open SER / AVI / image sequence under one API.
    """
    # Sequence
    if not isinstance(path_or_files, (str, os.PathLike)):
        return ImageSequenceReader(path_or_files, cache_items=cache_items)

    path = os.fspath(path_or_files)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".ser":
        r = SERReader(path, cache_items=cache_items)
        # ---- SER tweak: ensure meta.path is set ----
        try:
            r.meta.path = path  # type: ignore
        except Exception:
            pass
        return r

    if ext in (".avi", ".mp4", ".mov", ".mkv"):
        return AVIReader(path, cache_items=cache_items)

    # If user passes a single image, treat it as a 1-frame sequence
    if ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"):
        return ImageSequenceReader([path], cache_items=cache_items)

    raise ValueError(f"Unsupported input: {path}")