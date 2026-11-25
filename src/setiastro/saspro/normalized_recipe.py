# src/setiastro/saspro/normalized_recipe.py
# ──────────────────────────────────────────────────────────────────────────────
#  SetiAstroSuite Pro — virtual normalized frames
#  Copyright (C) 2026 Franklin Marek / SetiAstro
#  Licensed under the GNU General Public License v3.0 or later.
# ──────────────────────────────────────────────────────────────────────────────
"""
Virtual normalized frames (SASPRO_VIRTUAL_NORM_V1).

For a MONO light, normalization is a short, replayable recipe applied to its
Calibrated/ input:

    fast float-FITS read (incl. ROWORDER BOTTOM-UP -> top-down)
    -> optional 180-degree turn (meridian flip, decided from the headers)
    -> one float32 additive background offset (measured once, stored)
    -> big-endian float32, the header astropy would emit

so instead of writing Normalized_Images/<frame>_n.fit (187 MB at 46.8 MP),
normalization can write a few-KB recipe beside where that file would be, and
every reader rebuilds the identical pixels and header on demand.

The rules this module keeps:

  * Every data-dependent number is STORED as it was computed (the background
    offset as its float32 bits, the flip decision, the canvas). Regeneration
    only applies them with elementwise/geometric ops, so thread counts or the
    GPU cannot change the result.
  * The header stored is the exact header block astropy writes for the frame,
    so a reader gets the same cards, card images and order as from the file.
  * Staleness is loud: the Calibrated source's size and mtime are recorded,
    and the regenerated big-endian pixels must match a CRC32 taken when the
    recipe was written. Any mismatch raises StaleRecipeError; nothing here
    ever falls back to some other frame.

Import-light on purpose (numpy + astropy; cv2 and the legacy loader lazily):
registration's spawned workers import it, and they must never pull in the
Stacking Suite dialog.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import zlib

import numpy as np

RECIPE_SUFFIX = ".sasnorm.json"
RECIPE_VERSION = 1


class StaleRecipeError(RuntimeError):
    """A virtual normalized frame can no longer be rebuilt exactly as written."""


def recipe_path(frame_path: str) -> str:
    """Where the recipe for the normalized frame `frame_path` lives."""
    return os.path.splitext(frame_path)[0] + RECIPE_SUFFIX


def has_recipe(frame_path: str) -> bool:
    try:
        return os.path.isfile(recipe_path(frame_path))
    except Exception:
        return False


def is_virtual(frame_path: str) -> bool:
    """True when the frame exists only as a recipe (no file on disk)."""
    try:
        return (not os.path.exists(frame_path)) and has_recipe(frame_path)
    except Exception:
        return False


def exists(frame_path: str) -> bool:
    """The frame is readable: a real file, or a recipe for it."""
    return os.path.isfile(frame_path) or has_recipe(frame_path)


def remove_recipe(frame_path: str) -> None:
    try:
        os.remove(recipe_path(frame_path))
    except OSError:
        pass


# ── the Calibrated/ loader normalization uses ────────────────────────────────
def fast_fits_read(fp: str):
    """Fast FITS reader for the normalization hot path: raw float32 pixels
    + header, NaN preserved, row-order normalized -- WITHOUT load_image's
    general overhead (second header open via get_valid_header, WCS attach,
    per-frame prints, redundant copies). Returns (img, hdr) on the float
    fast path, or (None, None) to tell the caller to use the full loader
    (int/scaled/odd FITS, so bit-depth handling stays byte-identical).

    The one definition: StackingSuiteDialog._fast_fits_read delegates here,
    so normalization and regeneration cannot read a frame differently."""
    from astropy.io import fits
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdu = None
            for _x in hdul:
                _d = getattr(_x, "data", None)
                if _d is not None and getattr(_d, "ndim", 0) >= 2:
                    hdu = _x; break
            if hdu is None:
                return None, None
            data = np.asarray(hdu.data)
            hdr = hdu.header.copy()
        # Fast path only for float FITS (calibrated/normalized frames).
        if data.dtype.kind != "f":
            return None, None
        if data.dtype.byteorder not in ("=", "|"):
            data = data.astype(data.dtype.newbyteorder("="))
        img = np.asarray(data, dtype=np.float32)
        # preserve NaN (satellite no-data); scrub only +/-inf -- matches
        # _finalize_loaded_image under the NaN-preserve context. The
        # unconditional nan_to_num was a full allocate-and-copy of the
        # frame on every load to fix pixels that are almost never there;
        # probe first and fix in place only when it matters.
        if np.isinf(img).any():
            img = np.nan_to_num(img, copy=False, nan=np.nan,
                                posinf=1.0, neginf=0.0)
        img = np.squeeze(img)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] > 1 and img.shape[2] > 1:
            img = np.transpose(img, (1, 2, 0))   # CHW -> HWC
        elif img.ndim == 3 and img.shape[-1] not in (1, 3):
            return None, None   # unusual layout -> full loader
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = np.ascontiguousarray(img, dtype=np.float32)
        try:
            from setiastro.saspro.legacy.image_manager import _apply_roworder_flip
            img, hdr = _apply_roworder_flip(img, hdr)
        except Exception:
            pass
        return img, hdr
    except Exception:
        return None, None


def to_writable_f32(arr):
    """stacking_suite._to_writable_f32, verbatim."""
    a = np.asarray(arr)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    # Make sure we can write into it (mmap/XISF often returns read-only views)
    if (not a.flags.writeable) or (not a.flags.c_contiguous):
        a = np.ascontiguousarray(a.copy())
    return a


def rot180(img):
    """The rotation StackingSuiteDialog._maybe_rot180 applies, verbatim:
    cv2.flip(-1) (bit-exact data movement), numpy if OpenCV declines."""
    try:
        import cv2
        out = cv2.flip(img, -1)
        if out is not None and out.shape == img.shape:
            return np.ascontiguousarray(out)
    except Exception:
        pass
    return np.ascontiguousarray(np.rot90(img, 2))


# ── the header exactly as astropy writes it ──────────────────────────────────
class _HeaderSink(io.RawIOBase):
    """A write target that keeps the primary header block and counts the rest."""

    def __init__(self):
        super().__init__()
        self.head = bytearray()
        self.total = 0
        self.done = False

    def writable(self):
        return True

    def write(self, b):
        mv = memoryview(b).cast("B")
        n = mv.nbytes
        if not self.done:
            room = max(0, 2880 * 400 - len(self.head))
            self.head += mv[:room]
            i = 0
            while i + 80 <= len(self.head):
                if self.head[i:i + 3] == b"END" and self.head[i + 3:i + 80].strip() == b"":
                    end = ((i + 80 + 2879) // 2880) * 2880
                    if len(self.head) >= end:
                        del self.head[end:]
                        self.done = True
                    break
                i += 80
        self.total += n
        return n


def header_block(data_be, header) -> bytes:
    """The header block `fits.PrimaryHDU(data=data_be, header=header).writeto()`
    emits -- the same call normalization makes for a real _n.fit -- without
    writing the pixels anywhere. Verified against the bytes it accounts for."""
    from astropy.io import fits
    sink = _HeaderSink()
    fits.PrimaryHDU(data=data_be, header=header).writeto(sink)
    if not sink.done:
        raise RuntimeError("could not capture the FITS header block")
    pad = (-int(data_be.nbytes)) % 2880
    if sink.total != len(sink.head) + int(data_be.nbytes) + pad:
        raise RuntimeError("unexpected FITS layout while capturing the header")
    return bytes(sink.head)


# ── recipes ──────────────────────────────────────────────────────────────────
def write_recipe(frame_path: str, *, source: str, data_be, header,
                 rot180: bool, bg_offset) -> int:
    """Write the recipe that rebuilds `frame_path` from `source`.

    `data_be` is the big-endian array normalization would have written and
    `header` the header it would have written with it. Returns the recipe's
    size in bytes. Raises on failure: the caller then writes the real file.
    """
    a = np.asarray(data_be)
    if a.dtype != np.dtype(">f4") or a.ndim != 2 or not a.flags.c_contiguous:
        raise ValueError("virtual normalized frames are C-ordered mono '>f4'")
    hb = header_block(a, header)
    st = os.stat(source)
    off = np.asarray(bg_offset, dtype="<f4").reshape(())
    out = recipe_path(frame_path)
    # Also where the source sits relative to the recipe, so a stacking folder
    # that is moved or copied elsewhere still finds its Calibrated/ frames
    # (resolve_source). None when they are on different drives.
    try:
        source_rel = os.path.relpath(os.path.abspath(source),
                                     os.path.dirname(os.path.abspath(out)))
    except ValueError:
        source_rel = None
    rec = {
        "version": RECIPE_VERSION,
        "kind": "mono_add",
        "source": os.path.abspath(source),
        "source_rel": source_rel,
        "source_size": int(st.st_size),
        "source_mtime_ns": int(st.st_mtime_ns),
        "shape": [int(a.shape[0]), int(a.shape[1])],
        "rot180": bool(rot180),
        "bg_offset_f32": off.tobytes().hex(),
        "bg_offset": float(off),          # readable copy; the hex is what is used
        "crc32": int(zlib.crc32(memoryview(a).cast("B")) & 0xFFFFFFFF),
        "header": hb.decode("ascii"),
    }
    tmp = out + ".part"
    with open(tmp, "w", encoding="ascii") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    return os.path.getsize(out)


def read_recipe(frame_path: str) -> dict:
    p = recipe_path(frame_path)
    try:
        with open(p, "r", encoding="ascii") as f:
            rec = json.load(f)
    except FileNotFoundError:
        raise StaleRecipeError(
            f"{os.path.basename(frame_path)}: the normalized frame is not on disk "
            f"and has no recipe ({os.path.basename(p)}).") from None
    if int(rec.get("version", -1)) != RECIPE_VERSION or rec.get("kind") != "mono_add":
        raise StaleRecipeError(
            f"{os.path.basename(p)}: unsupported recipe (version {rec.get('version')!r}, "
            f"kind {rec.get('kind')!r}). Re-run normalization.")
    return rec


def recipe_hw(frame_path: str):
    """(H, W) of a virtual frame from its recipe alone, or None."""
    try:
        h, w = read_recipe(frame_path)["shape"]
        return int(h), int(w)
    except Exception:
        return None


def resolve_source(rec: dict, frame_path: str = "") -> str:
    """The calibrated file to rebuild from: the recorded path, or -- when that
    is gone because the stacking folder was moved or copied -- the same place
    relative to the recipe. Whichever it is must still pass check_source (size
    and mtime) and the pixel CRC, so this can never substitute a different
    frame; it only stops a moved folder from looking like changed frames."""
    src = rec.get("source", "")
    if src and os.path.exists(src):
        return src
    rel = rec.get("source_rel")
    if rel and frame_path:
        cand = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(recipe_path(frame_path))), rel))
        if os.path.exists(cand):
            return cand
    return src


def check_source(rec: dict, frame_path: str = "") -> None:
    """Raise StaleRecipeError unless the Calibrated source is the one recorded."""
    src = resolve_source(rec, frame_path)
    name = os.path.basename(frame_path) or os.path.basename(src)
    try:
        st = os.stat(src)
    except OSError:
        raise StaleRecipeError(
            f"{name}: its calibrated source {src} is missing, so the virtual "
            f"normalized frame cannot be rebuilt. Re-run registration "
            f"(normalization) with the calibrated frames in place.") from None
    if int(st.st_size) != int(rec["source_size"]) or int(st.st_mtime_ns) != int(rec["source_mtime_ns"]):
        raise StaleRecipeError(
            f"{name}: its calibrated source {src} changed after normalization "
            f"(size/mtime differ), so the virtual normalized frame would not be "
            f"the one registration used. Re-run registration (normalization).")


def regenerate(frame_path: str):
    """`(data, header)` of a virtual normalized frame, exactly as reading the
    _n.fit would give them: a C-ordered big-endian float32 array and the header
    parsed from the very block astropy would have written. Raises
    StaleRecipeError if that cannot be guaranteed."""
    from astropy.io import fits
    rec = read_recipe(frame_path)
    check_source(rec, frame_path)
    src = resolve_source(rec, frame_path)
    img, _hdr = fast_fits_read(src)
    if img is None:
        raise StaleRecipeError(
            f"{os.path.basename(frame_path)}: {src} no longer reads as a "
            f"float FITS frame.")
    img = to_writable_f32(img)
    if img.ndim != 2:
        raise StaleRecipeError(f"{os.path.basename(frame_path)}: source is not mono.")
    if rec["rot180"]:
        img = rot180(img)
    if (not img.flags.writeable) or (not img.flags.c_contiguous):
        img = np.ascontiguousarray(img, dtype=np.float32)
    off = np.frombuffer(bytes.fromhex(rec["bg_offset_f32"]), dtype="<f4")[0]
    img += np.float32(off)
    if tuple(img.shape) != tuple(rec["shape"]):
        raise StaleRecipeError(
            f"{os.path.basename(frame_path)}: rebuilt shape {img.shape} != recorded "
            f"{tuple(rec['shape'])}.")
    data = np.asarray(img, dtype=">f4")
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    crc = int(zlib.crc32(memoryview(data).cast("B")) & 0xFFFFFFFF)
    if crc != int(rec["crc32"]):
        raise StaleRecipeError(
            f"{os.path.basename(frame_path)}: rebuilt pixels do not match what "
            f"normalization produced (CRC {crc:08x} != {int(rec['crc32']):08x}). "
            f"Re-run registration (normalization).")
    header = fits.Header.fromstring(rec["header"])
    return data, header


def write_frame_bytes(frame_path: str, out_path: str) -> None:
    """Materialize a virtual frame as the exact _n.fit bytes normalization would
    have written: the recorded header block, the pixels, zero padding."""
    rec = read_recipe(frame_path)
    data, _header = regenerate(frame_path)
    hb = rec["header"].encode("ascii")
    pad = (-int(data.nbytes)) % 2880
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(hb)
        f.write(memoryview(data).cast("B"))
        if pad:
            f.write(b"\x00" * pad)
    os.replace(tmp, out_path)


@contextlib.contextmanager
def materialized(frame_path: str, directory: str | None = None):
    """Yield a path to a temporary, byte-exact copy of the frame (for readers
    that need a file, e.g. load_image); removed afterwards. A real file is
    yielded as-is."""
    if os.path.exists(frame_path):
        yield frame_path
        return
    fd, tmp = tempfile.mkstemp(prefix="sasnorm_", suffix="_n.fit", dir=directory)
    os.close(fd)
    try:
        write_frame_bytes(frame_path, tmp)
        yield tmp
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def load_pixels(frame_path: str):
    """The pixels `load_image(frame_path)[0]` gives for this normalized frame,
    real or virtual, without writing a file for a virtual one.

    A virtual frame is rebuilt in memory (regenerate: source check + CRC) and
    then given load_image's own float-FITS treatment, step for step: native
    float32, squeeze, _finalize_loaded_image (which reads the CALLING thread's
    NaN-preserve setting, so call this where load_image would have been
    called), then the ROWORDER flip on the header get_valid_header would have
    returned. Bit-identical to load_image on the file normalization would have
    written; that file used to be materialized in the temp folder only to be
    read straight back. Raises StaleRecipeError like regenerate."""
    from setiastro.saspro.legacy.image_manager import (
        load_image, _finalize_loaded_image, _apply_roworder_flip, _drop_invalid_cards)
    if not is_virtual(frame_path):
        return load_image(frame_path)[0]
    data, header = regenerate(frame_path)
    img = np.squeeze(data.astype(np.float32))   # native byte order, own copy
    data = None
    img = _finalize_loaded_image(img)
    img, _hdr = _apply_roworder_flip(img, _drop_invalid_cards(header))
    return img
