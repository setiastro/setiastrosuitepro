# src/setiastro/saspro/imageops/syq_io.py
"""
SYQ Core 1 reader/writer for SASpro.

`.syq` is SyQon Studio's native scientific container (see
https://syqon.eu/develop → "04 / .syq scientific format"). It is an open,
versioned, *tiled* container:

    [ 128-byte superblock ]
    [ object payloads … ]         (image descriptor = CBOR; tiles = raw planar)
    [ directory: u64 count + N × 96-byte entries ]
    [ 80-byte commit footer ]

Every object carries a SHA-256 over its *logical* (uncompressed) bytes, and the
superblock and footer carry their own digests, so a misparse fails loudly
instead of returning corrupt pixels.

──────────────────────────────────────────────────────────────────────────────
INTEROP STATUS — VALIDATED
──────────────────────────────────────────────────────────────────────────────
This implementation follows the authoritative SYQ Core 1 binary specification
and has been checked against a real SyQon-Studio-written file (test.syq):
superblock / footer / directory packing, the integer `sample` enum
(1 UInt8, 2 UInt16, 3 Float32, 4 Float64), the descriptor schema, planar tile
layout and Zstandard tiles all read correctly. `SYQ_STRICT_INTEROP` is on.

Reading real Studio files requires a Zstandard backend, because Studio stores
tiles with codec 1. The reader tries `zstandard` → `pyzstd` → the system
libzstd (via ctypes) and raises a clear message if none is present — so add
`zstandard` to SASpro's dependencies (or ship libzstd). Writing defaults to
uncompressed tiles (codec 0), which is spec-valid and needs no dependency;
pass `compress=True` to use Zstandard when a backend is available.

Core 1 is still an "implementation candidate", so pin the version and re-check
on SyQon releases. Optional objects (WCS, ICC, Studio session, process graph,
original-source archive, preview) are skipped per the spec; full extraction of
those is a later addition once we choose to support them.
"""
from __future__ import annotations

import io
import struct
import hashlib
import uuid as _uuid
import time
from typing import Optional, Tuple, Any, Dict, List

import numpy as np

SYQ_CORE_MAJOR = 1
SYQ_CORE_MINOR = 0
SYQ_MAGIC = bytes((0x89, 0x53, 0x59, 0x51, 0x0D, 0x0A, 0x1A, 0x0A))
SYQ_FOOTER_MAGIC = b"SYQEND01"
SYQ_MIME = "application/x-syq"

# Published numeric limits (guard before allocating).
MAX_OBJECT_BYTES = 64 * 1024 * 1024        # 64 MiB per object
MAX_DIRECTORY_BYTES = 128 * 1024 * 1024     # 128 MiB directory
MAX_CHANNELS = 1024
MAX_TILE_EDGE = 1024
MAX_TILES = 1_000_000

DEFAULT_TILE_SIZE = 512

# Confirmed against the SYQ Core 1 specification AND a real Studio-written file
# (test.syq): superblock/footer/directory packing, the integer `sample` enum,
# and the descriptor schema all validated. Strict mode requires the documented
# required keys to be present.
SYQ_STRICT_INTEROP = True


# ──────────────────────────────────────────────────────────────────────────
# WIRE — every byte-layout assumption the marketing page leaves ambiguous.
# Change values HERE (not in the code below) when the reference kit arrives.
# ──────────────────────────────────────────────────────────────────────────
class WIRE:
    # Superblock (128 bytes). Spans are (start, end) half-open, little-endian.
    SB_MAGIC = (0, 8)
    SB_MAJOR = (8, 10)          # u16
    SB_MINOR = (10, 12)         # u16
    SB_HEADER_SIZE = (12, 16)   # u32 == 128
    SB_REQUIRED_FEATURES = (16, 24)   # u64 bitmask (0 == none required)
    # 24..32 reserved gap
    SB_DIR_OFFSET = (32, 40)    # u64  ┐ "initial directory geometry + primary"
    SB_DIR_LENGTH = (40, 48)    # u64  │
    SB_PRIMARY_ID = (48, 56)    # u64  ┘
    SB_UUID = (56, 72)          # 16 bytes
    SB_CREATED_NS = (72, 80)    # u64
    # 80..96 reserved gap
    SB_DIGEST = (96, 128)       # SHA-256 of bytes [0:96]
    SB_DIGEST_COVERS = (0, 96)
    SB_SIZE = 128

    # Directory entry (96 bytes).
    DE_OBJECT_ID = (0, 8)       # u64
    DE_PARENT_ID = (8, 16)      # u64
    DE_TYPE = (16, 20)          # u32   ┐ "type + flags"
    DE_FLAGS = (20, 24)         # u32   ┘
    DE_OFFSET = (24, 32)        # u64
    DE_STORED_LEN = (32, 40)    # u64
    DE_LOGICAL_LEN = (40, 48)   # u64
    DE_CODEC = (48, 52)         # u32   ┐ "codec + checksum algorithm"
    DE_CKSUM_ALG = (52, 56)     # u32   ┘
    DE_UNCOMPRESSED_SHA = (56, 88)   # 32 bytes
    # 88..96 reserved
    DE_SIZE = 96

    # Commit footer (80 bytes).
    FT_MAGIC = (0, 8)
    FT_DIR_OFFSET = (8, 16)     # u64
    FT_DIR_LENGTH = (16, 24)    # u64
    FT_PRIMARY_ID = (24, 32)    # u64
    FT_FILE_SIZE = (32, 40)     # u64
    FT_DIR_SHA = (40, 72)       # 32 bytes (SHA-256 of the directory block)
    FT_FOOTER_SHA8 = (72, 80)   # first 8 bytes of SHA-256 over footer [0:72]
    FT_SHA_COVERS = (0, 72)
    FT_SIZE = 80

    # Object type registry.
    TYPE_IMAGE = 1
    TYPE_TILE = 2
    TYPE_METADATA = 3

    # Codec / checksum ids.
    CODEC_NONE = 0
    CODEC_ZSTD = 1
    CKSUM_SHA256 = 1

    # Image-descriptor CBOR keys (best guess from the C++ field names).
    K_WIDTH = "width"
    K_HEIGHT = "height"
    K_CHANNELS = "channels"
    K_TILE_SIZE = "tile_size"
    K_SAMPLE = "sample"
    K_TRANSFER = "transfer"
    K_STATE = "state"
    K_ROLE = "role"
    K_METADATA = "metadata"
    K_TILES = "tiles"           # ordered array of tile object IDs
    K_UUID = "uuid"             # image UUID string, no braces (required key)

    # Sample-type encoding — AUTHORITATIVE per the SYQ Core 1 spec:
    #   1 UInt8, 2 UInt16, 3 Float32, 4 Float64
    # Float16 / UInt32 / signed ints are explicitly future extensions a reader
    # must NOT guess, so any other code fails loudly.
    SAMPLE_U8 = "UInt8"
    SAMPLE_U16 = "UInt16"
    SAMPLE_F32 = "Float32"
    SAMPLE_F64 = "Float64"
    SAMPLE_INT = {SAMPLE_U8: 1, SAMPLE_U16: 2, SAMPLE_F32: 3, SAMPLE_F64: 4}
    SAMPLE_BY_INT = {1: SAMPLE_U8, 2: SAMPLE_U16, 3: SAMPLE_F32, 4: SAMPLE_F64}


_SAMPLE_DTYPE = {
    WIRE.SAMPLE_U8: np.dtype("<u1"),
    WIRE.SAMPLE_U16: np.dtype("<u2"),
    WIRE.SAMPLE_F32: np.dtype("<f4"),
    WIRE.SAMPLE_F64: np.dtype("<f8"),
}
_SAMPLE_IS_INT = {WIRE.SAMPLE_U8: True, WIRE.SAMPLE_U16: True,
                  WIRE.SAMPLE_F32: False, WIRE.SAMPLE_F64: False}
_SAMPLE_MAXVAL = {WIRE.SAMPLE_U8: 255.0, WIRE.SAMPLE_U16: 65535.0}

# SASpro bit-depth vocabulary ↔ SYQ sample.
_DEPTH_TO_SAMPLE = {
    "8-bit": WIRE.SAMPLE_U8,
    "16-bit": WIRE.SAMPLE_U16,
    "32-bit floating point": WIRE.SAMPLE_F32,
}
_SAMPLE_TO_DEPTH = {
    WIRE.SAMPLE_U8: "8-bit",
    WIRE.SAMPLE_U16: "16-bit",
    WIRE.SAMPLE_F32: "32-bit floating point",
    WIRE.SAMPLE_F64: "32-bit floating point",   # SASpro is f32-internal; f64 downcasts
}


class SyqError(Exception):
    """Raised on any structural/validation failure while reading a .syq."""


# ──────────────────────────────────────────────────────────────────────────
# Minimal CBOR (definite-length subset: uint/negint/bytes/text/array/map/
# float64/bool/null). Enough for the image descriptor + namespaced metadata,
# with no third-party dependency. Prefers `cbor2` if the app ships it.
# ──────────────────────────────────────────────────────────────────────────
# Minimal CBOR (definite-length: uint/negint/bytes/text/array/map/float16-32-64/
# bool/null/simple; tags decoded transparently). Enough for the image descriptor
# and namespaced metadata with no third-party dependency. The builtin is always
# defined (so it can be unit-tested); `cbor2` is used only if the app ships it.
# ──────────────────────────────────────────────────────────────────────────
def _cbor_encode_builtin(obj) -> bytes:
    out = bytearray()
    _cbor_enc(obj, out)
    return bytes(out)


def _cbor_enc(obj, out: bytearray):
    if obj is None:
        out.append(0xF6)
    elif obj is True:
        out.append(0xF5)
    elif obj is False:
        out.append(0xF4)
    elif isinstance(obj, bool):
        out.append(0xF5 if obj else 0xF4)
    elif isinstance(obj, int):
        if obj >= 0:
            _cbor_head(0, obj, out)
        else:
            _cbor_head(1, -1 - obj, out)
    elif isinstance(obj, float):
        out.append(0xFB)
        out += struct.pack(">d", obj)
    elif isinstance(obj, (bytes, bytearray)):
        _cbor_head(2, len(obj), out)
        out += bytes(obj)
    elif isinstance(obj, str):
        b = obj.encode("utf-8")
        _cbor_head(3, len(b), out)
        out += b
    elif isinstance(obj, (list, tuple)):
        _cbor_head(4, len(obj), out)
        for it in obj:
            _cbor_enc(it, out)
    elif isinstance(obj, dict):
        # canonical: sort keys by encoded bytes for deterministic output
        enc = [(_cbor_encode_builtin(k), v) for k, v in obj.items()]
        enc.sort(key=lambda t: t[0])
        _cbor_head(5, len(enc), out)
        for kb, v in enc:
            out += kb
            _cbor_enc(v, out)
    else:
        raise SyqError(f"CBOR: unsupported type {type(obj).__name__}")


def _cbor_head(major: int, val: int, out: bytearray):
    mt = major << 5
    if val < 24:
        out.append(mt | val)
    elif val < 0x100:
        out.append(mt | 24); out.append(val)
    elif val < 0x10000:
        out.append(mt | 25); out += struct.pack(">H", val)
    elif val < 0x100000000:
        out.append(mt | 26); out += struct.pack(">I", val)
    else:
        out.append(mt | 27); out += struct.pack(">Q", val)


def _float16_to_float(h: int) -> float:
    sign = -1.0 if (h >> 15) & 1 else 1.0
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        return sign * (frac / 1024.0) * (2.0 ** -14)
    if exp == 0x1F:
        return sign * (float("inf") if frac == 0 else float("nan"))
    return sign * (1.0 + frac / 1024.0) * (2.0 ** (exp - 15))


def _cbor_decode_builtin(buf: bytes):
    val, off = _cbor_dec(memoryview(buf), 0)
    return val


def _cbor_dec(mv: memoryview, off: int):
    ib = mv[off]; off += 1
    major = ib >> 5
    minor = ib & 0x1F

    # Major 7 = simple values and IEEE floats. Its "minor" is NOT an integer
    # length argument, so it must be handled before the generic decode below
    # (this is the bug that raised "unsupported major 7" on real descriptors).
    if major == 7:
        if minor < 20:
            return minor, off               # simple value
        if minor == 20:
            return False, off
        if minor == 21:
            return True, off
        if minor in (22, 23):
            return None, off                # null / undefined
        if minor == 24:
            sv = mv[off]; return sv, off + 1
        if minor == 25:
            h = struct.unpack_from(">H", mv, off)[0]; return _float16_to_float(h), off + 2
        if minor == 26:
            return struct.unpack_from(">f", mv, off)[0], off + 4
        if minor == 27:
            return struct.unpack_from(">d", mv, off)[0], off + 8
        raise SyqError(f"CBOR: bad simple/float info {minor}")

    # Majors 0..6 carry an unsigned integer argument in the additional info.
    if minor < 24:
        val = minor
    elif minor == 24:
        val = mv[off]; off += 1
    elif minor == 25:
        val = struct.unpack_from(">H", mv, off)[0]; off += 2
    elif minor == 26:
        val = struct.unpack_from(">I", mv, off)[0]; off += 4
    elif minor == 27:
        val = struct.unpack_from(">Q", mv, off)[0]; off += 8
    else:
        raise SyqError(f"CBOR: bad additional info {minor} for major {major}")

    if major == 0:
        return val, off
    if major == 1:
        return -1 - val, off
    if major == 2:
        return bytes(mv[off:off + val]), off + val
    if major == 3:
        return bytes(mv[off:off + val]).decode("utf-8"), off + val
    if major == 4:
        arr = []
        for _ in range(val):
            item, off = _cbor_dec(mv, off)
            arr.append(item)
        return arr, off
    if major == 5:
        d = {}
        for _ in range(val):
            k, off = _cbor_dec(mv, off)
            v, off = _cbor_dec(mv, off)
            d[k] = v
        return d, off
    if major == 6:
        # Core 1 does not use tags; if one appears, decode and return the inner
        # item transparently rather than failing.
        inner, off = _cbor_dec(mv, off)
        return inner, off
    raise SyqError(f"CBOR: unsupported major {major}")


try:
    import cbor2 as _cbor2

    def _cbor_encode(obj) -> bytes:
        return _cbor2.dumps(obj, canonical=True)

    def _cbor_decode(buf: bytes):
        return _cbor2.loads(buf)
except Exception:
    _cbor_encode = _cbor_encode_builtin
    _cbor_decode = _cbor_decode_builtin




# ──────────────────────────────────────────────────────────────────────────
# zstd (lazy). Writer defaults to CODEC_NONE so writing never needs a dep;
# reading a compressed tile imports a backend only when one is encountered.
# ──────────────────────────────────────────────────────────────────────────
def _zstd_decompress(data: bytes, expected_len: int) -> bytes:
    try:
        import zstandard as _z
        return _z.ZstdDecompressor().decompress(data, max_output_size=expected_len)
    except Exception:
        pass
    try:
        import pyzstd as _p
        return _p.decompress(data)
    except Exception:
        pass
    try:  # system libzstd via ctypes (what the reference reader uses)
        import ctypes, ctypes.util
        lib = ctypes.util.find_library("zstd") or "libzstd.so.1"
        z = ctypes.CDLL(lib)
        z.ZSTD_decompress.restype = ctypes.c_size_t
        z.ZSTD_isError.restype = ctypes.c_uint
        out = ctypes.create_string_buffer(expected_len)
        n = z.ZSTD_decompress(out, ctypes.c_size_t(expected_len),
                              data, ctypes.c_size_t(len(data)))
        if z.ZSTD_isError(n):
            raise SyqError("libzstd reported a decompression error")
        return out.raw[:n]
    except SyqError:
        raise
    except Exception as e:
        raise SyqError(
            "This .syq uses Zstandard-compressed tiles but no zstd backend is "
            "available. Install `zstandard` (pip) or ensure libzstd is on the "
            f"system. ({e})"
        )


def _zstd_compress(data: bytes) -> Optional[bytes]:
    try:
        import zstandard as _z
        return _z.ZstdCompressor(level=10).compress(data)
    except Exception:
        return None


# ── small pack/unpack helpers keyed off WIRE spans ─────────────────────────
def _u(buf: bytes, span) -> int:
    a, b = span
    return int.from_bytes(buf[a:b], "little", signed=False)


def _setu(buf: bytearray, span, value: int):
    a, b = span
    buf[a:b] = int(value).to_bytes(b - a, "little", signed=False)


def _sample_from_descriptor(raw) -> str:
    if isinstance(raw, str):
        if raw in _SAMPLE_DTYPE:
            return raw
        low = {"uint8": WIRE.SAMPLE_U8, "u8": WIRE.SAMPLE_U8,
               "uint16": WIRE.SAMPLE_U16, "u16": WIRE.SAMPLE_U16,
               "float32": WIRE.SAMPLE_F32, "f32": WIRE.SAMPLE_F32,
               "float64": WIRE.SAMPLE_F64, "f64": WIRE.SAMPLE_F64}.get(raw.lower())
        if low:
            return low
    if isinstance(raw, int) and raw in WIRE.SAMPLE_BY_INT:
        return WIRE.SAMPLE_BY_INT[raw]
    raise SyqError(f"Unrecognized SYQ sample type: {raw!r}")


def _desc_get(desc: dict, key: str, *aliases):
    if key in desc:
        return desc[key]
    for a in aliases:
        if a in desc:
            return desc[a]
    if SYQ_STRICT_INTEROP:
        raise SyqError(f"Image descriptor missing required key {key!r}")
    return None


# ──────────────────────────────────────────────────────────────────────────
# READ
# ──────────────────────────────────────────────────────────────────────────
def _read_container(path: str):
    """
    Parse and validate the superblock, footer and directory. Returns
    (blob, entries: {id: entry}, primary_id, major). Raises SyqError on any
    structural/integrity failure so a misparse never yields garbage.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    n = len(blob)
    if n < WIRE.SB_SIZE + WIRE.FT_SIZE:
        raise SyqError("File too small to be a SYQ container.")

    sb = blob[:WIRE.SB_SIZE]
    if sb[WIRE.SB_MAGIC[0]:WIRE.SB_MAGIC[1]] != SYQ_MAGIC:
        raise SyqError("Not a SYQ file (bad magic).")
    major = _u(sb, WIRE.SB_MAJOR)
    if major != SYQ_CORE_MAJOR:
        raise SyqError(f"Unsupported SYQ Core major version {major}.")
    if hashlib.sha256(sb[WIRE.SB_DIGEST_COVERS[0]:WIRE.SB_DIGEST_COVERS[1]]).digest() \
            != sb[WIRE.SB_DIGEST[0]:WIRE.SB_DIGEST[1]]:
        raise SyqError("Superblock digest mismatch (corrupt header).")
    req = _u(sb, WIRE.SB_REQUIRED_FEATURES)
    if req != 0:
        raise SyqError(f"SYQ file requires unsupported features (bitmask {req:#x}).")

    ft = blob[n - WIRE.FT_SIZE:]
    if ft[WIRE.FT_MAGIC[0]:WIRE.FT_MAGIC[1]] != SYQ_FOOTER_MAGIC:
        raise SyqError("Missing/invalid SYQ commit footer.")
    if hashlib.sha256(ft[WIRE.FT_SHA_COVERS[0]:WIRE.FT_SHA_COVERS[1]]).digest()[:8] \
            != ft[WIRE.FT_FOOTER_SHA8[0]:WIRE.FT_FOOTER_SHA8[1]]:
        raise SyqError("Footer digest mismatch (corrupt trailer).")
    dir_off = _u(ft, WIRE.FT_DIR_OFFSET)
    dir_len = _u(ft, WIRE.FT_DIR_LENGTH)
    primary_id = _u(ft, WIRE.FT_PRIMARY_ID)
    committed = _u(ft, WIRE.FT_FILE_SIZE)
    if committed != n:
        raise SyqError("Committed file size does not match actual size.")
    if dir_len > MAX_DIRECTORY_BYTES or dir_off + dir_len > n:
        raise SyqError("Directory geometry out of bounds.")

    dblock = blob[dir_off:dir_off + dir_len]
    if hashlib.sha256(dblock).digest() != ft[WIRE.FT_DIR_SHA[0]:WIRE.FT_DIR_SHA[1]]:
        raise SyqError("Directory digest mismatch (corrupt directory).")
    count = int.from_bytes(dblock[:8], "little")
    if count * WIRE.DE_SIZE + 8 > len(dblock):
        raise SyqError("Directory entry count exceeds directory block.")

    entries: Dict[int, dict] = {}
    for i in range(count):
        base = 8 + i * WIRE.DE_SIZE
        e = dblock[base:base + WIRE.DE_SIZE]
        oid = _u(e, WIRE.DE_OBJECT_ID)
        ent = {
            "id": oid,
            "parent": _u(e, WIRE.DE_PARENT_ID),
            "type": _u(e, WIRE.DE_TYPE),
            "flags": _u(e, WIRE.DE_FLAGS),
            "offset": _u(e, WIRE.DE_OFFSET),
            "stored": _u(e, WIRE.DE_STORED_LEN),
            "logical": _u(e, WIRE.DE_LOGICAL_LEN),
            "codec": _u(e, WIRE.DE_CODEC),
            "cksum_alg": _u(e, WIRE.DE_CKSUM_ALG),
            "sha": e[WIRE.DE_UNCOMPRESSED_SHA[0]:WIRE.DE_UNCOMPRESSED_SHA[1]],
        }
        if oid in entries:
            raise SyqError(f"Duplicate object ID {oid} in directory.")
        if ent["offset"] + ent["stored"] > n:
            raise SyqError(f"Object {oid} payload out of bounds.")
        if ent["logical"] > MAX_OBJECT_BYTES:
            raise SyqError(f"Object {oid} exceeds the per-object size budget.")
        entries[oid] = ent

    return blob, entries, primary_id, major


def _object_payload(blob: bytes, ent: dict) -> bytes:
    raw = blob[ent["offset"]:ent["offset"] + ent["stored"]]
    if ent["codec"] == WIRE.CODEC_ZSTD:
        data = _zstd_decompress(raw, ent["logical"])
    elif ent["codec"] == WIRE.CODEC_NONE:
        data = raw
    else:
        raise SyqError(f"Object {ent['id']} uses unsupported codec {ent['codec']}.")
    if len(data) != ent["logical"]:
        raise SyqError(f"Object {ent['id']} logical length mismatch.")
    if ent["cksum_alg"] == WIRE.CKSUM_SHA256:
        if hashlib.sha256(data).digest() != ent["sha"]:
            raise SyqError(f"Object {ent['id']} SHA-256 mismatch (corrupt).")
    return data


def _decode_descriptor(blob: bytes, entries: dict, image_id: int) -> dict:
    ent = entries.get(image_id)
    if ent is None:
        raise SyqError(f"Image object {image_id} not found.")
    if ent["type"] != WIRE.TYPE_IMAGE:
        raise SyqError(f"Object {image_id} is not an image descriptor.")
    desc = _cbor_decode(_object_payload(blob, ent))
    if not isinstance(desc, dict):
        raise SyqError("Image descriptor is not a CBOR map.")
    return desc


def syq_metadata_to_fits_header(md: dict):
    """
    Best-effort astropy fits.Header from a SYQ descriptor's `metadata` map.

    The Studio adapter stores the source file's original keyword/value cards —
    FITS included, with the WCS/astrometric solution among them — under
    `syq.source.metadata`. We replay those into a fits.Header so SASpro's
    existing WCS plumbing (attach_wcs_to_metadata) resolves the solution exactly
    as it does for FITS/XISF. FITS history/comments and the source image name are
    carried too. Returns None when nothing usable is present.
    """
    try:
        from astropy.io import fits
    except Exception:
        return None
    if not isinstance(md, dict):
        return None

    hdr = fits.Header()
    added = False

    def _coerce(v):
        if isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            s = v.strip()
            low = s.lower()
            if low in ("true", "false"):
                return low == "true"
            if s and s.lstrip("+-").isdigit():
                try:
                    return int(s)
                except Exception:
                    return v
            try:
                return float(s)
            except Exception:
                return v
        return v

    def _put(key, value):
        nonlocal added
        k = str(key).strip()
        if not k or isinstance(value, (dict, list, bytes, bytearray)):
            return
        val = _coerce(value)
        try:
            if len(k) <= 8 and k == k.upper() and all(c.isalnum() or c in "-_" for c in k):
                hdr[k] = val               # standard FITS keyword (incl. WCS)
            else:
                hdr[f"HIERARCH {k}"] = val if isinstance(val, (int, float, bool, str)) else str(val)
            added = True
        except Exception:
            try:
                hdr[f"HIERARCH {k}"] = str(value)[:68]
                added = True
            except Exception:
                pass

    src = md.get("syq.source.metadata")
    if isinstance(src, dict):
        for k, v in src.items():
            _put(k, v)

    name = md.get("syq.image.name")
    if isinstance(name, str) and name:
        _put("SYQNAME", name[:68])

    for c in (md.get("syq.fits.comments") or []):
        try:
            hdr.add_comment(str(c)); added = True
        except Exception:
            pass
    for h in (md.get("syq.fits.history") or []):
        try:
            hdr.add_history(str(h)); added = True
        except Exception:
            pass

    return hdr if added else None


def list_syq_images(path: str) -> List[dict]:
    """
    Enumerate the image *layers* in a .syq — every IMAGE_DESCRIPTOR object, the
    SYQ analog of FITS image HDUs / XISF images. The footer's primary image is
    returned first and flagged `is_primary`. Each entry carries geometry, role/
    state/transfer, the source image name and the raw `metadata` map (so callers
    can synthesize a header / attach WCS without a second read). Cheap: reads
    only the small descriptors, not tile pixels.
    """
    blob, entries, primary_id, _major = _read_container(path)
    out: List[dict] = []
    for oid, ent in entries.items():
        if ent["type"] != WIRE.TYPE_IMAGE:
            continue
        try:
            desc = _decode_descriptor(blob, entries, oid)
        except SyqError:
            continue
        md = _desc_get(desc, WIRE.K_METADATA) or {}
        try:
            sample = _sample_from_descriptor(_desc_get(desc, WIRE.K_SAMPLE))
        except SyqError:
            sample = None
        out.append({
            "id": oid,
            "is_primary": (oid == primary_id),
            "role": _desc_get(desc, WIRE.K_ROLE) or "",
            "state": _desc_get(desc, WIRE.K_STATE) or "",
            "transfer": _desc_get(desc, WIRE.K_TRANSFER) or "",
            "width": int(_desc_get(desc, WIRE.K_WIDTH) or 0),
            "height": int(_desc_get(desc, WIRE.K_HEIGHT) or 0),
            "channels": int(_desc_get(desc, WIRE.K_CHANNELS) or 0),
            "sample": sample,
            "bit_depth": _SAMPLE_TO_DEPTH.get(sample, "32-bit floating point"),
            "name": md.get("syq.image.name") if isinstance(md, dict) else None,
            "metadata": md if isinstance(md, dict) else {},
        })
    # primary first, then by object ID for stable order
    out.sort(key=lambda d: (not d["is_primary"], d["id"]))
    return out


def read_syq(path: str, image_id: Optional[int] = None):
    """
    Read one image layer from a .syq.

    image_id : the IMAGE_DESCRIPTOR object ID to read (from list_syq_images).
               None → the footer's primary image.

    Returns (image, header, bit_depth, is_mono) to match load_image():
      image      : HxW (mono) or HxWxC. Integer samples are normalized to [0,1];
                   Float32/Float64 samples are returned as float32 WITHOUT
                   clamping (per spec — HDR/negative values are preserved).
      header     : an astropy fits.Header synthesized from the layer's metadata
                   (source FITS cards + WCS), or None if none is available.
      bit_depth  : SASpro bit-depth string.
      is_mono    : True when channels == 1.
    """
    blob, entries, primary_id, _major = _read_container(path)
    target = primary_id if image_id is None else int(image_id)

    desc = _decode_descriptor(blob, entries, target)
    width = int(_desc_get(desc, WIRE.K_WIDTH) or 0)
    height = int(_desc_get(desc, WIRE.K_HEIGHT) or 0)
    channels = int(_desc_get(desc, WIRE.K_CHANNELS) or 0)
    tile_size = int(_desc_get(desc, WIRE.K_TILE_SIZE) or 0)
    sample = _sample_from_descriptor(_desc_get(desc, WIRE.K_SAMPLE))
    tile_ids = _desc_get(desc, WIRE.K_TILES) or []

    if not (0 < width and 0 < height and 0 < channels <= MAX_CHANNELS):
        raise SyqError(f"Impossible geometry {width}x{height}x{channels}.")
    if not (0 < tile_size <= MAX_TILE_EDGE):
        raise SyqError(f"Bad tile size {tile_size}.")

    nx = (width + tile_size - 1) // tile_size
    ny = (height + tile_size - 1) // tile_size
    if len(tile_ids) != nx * ny:
        raise SyqError(f"Descriptor lists {len(tile_ids)} tiles; expected {nx*ny}.")
    if nx * ny > MAX_TILES:
        raise SyqError("Tile count exceeds budget.")

    dt = _SAMPLE_DTYPE[sample]
    out = np.zeros((height, width, channels), dtype=dt)

    for idx, tid in enumerate(tile_ids):
        ent = entries.get(int(tid))
        if ent is None or ent["type"] != WIRE.TYPE_TILE:
            raise SyqError(f"Tile object {tid} missing or wrong type.")
        if ent["parent"] != target:
            raise SyqError(f"Tile {tid} parent is not image {target}.")
        ty, tx = divmod(idx, nx)
        x0, y0 = tx * tile_size, ty * tile_size
        tw = min(tile_size, width - x0)
        th = min(tile_size, height - y0)
        need = tw * th * channels * dt.itemsize
        buf = _object_payload(blob, ent)
        if len(buf) != need:
            raise SyqError(f"Tile {tid} size {len(buf)} != expected {need}.")
        tile = np.frombuffer(buf, dtype=dt).reshape((channels, th, tw))
        out[y0:y0 + th, x0:x0 + tw, :] = np.transpose(tile, (1, 2, 0))

    if _SAMPLE_IS_INT[sample]:
        img = out.astype(np.float32) / _SAMPLE_MAXVAL[sample]
    else:
        img = out.astype(np.float32, copy=False)   # no clamp: preserve HDR/negatives

    is_mono = (channels == 1)
    if is_mono:
        img = img[..., 0]

    md = _desc_get(desc, WIRE.K_METADATA) or {}
    header = syq_metadata_to_fits_header(md if isinstance(md, dict) else {})
    return img, header, _SAMPLE_TO_DEPTH[sample], is_mono


# ──────────────────────────────────────────────────────────────────────────
# WRITE
# ──────────────────────────────────────────────────────────────────────────
def write_syq(image: np.ndarray, path: str, *,
              bit_depth: str = "32-bit floating point",
              tile_size: int = DEFAULT_TILE_SIZE,
              metadata: Optional[dict] = None,
              transfer: str = "LINEAR",
              state: str = "PROCESSED_LINEAR",
              compress: bool = False) -> None:
    """
    Write `image` (HxW or HxWxC, float in [0,1] for integer depths, or raw
    float for the floating depth) as a single atomic-commit .syq.

    `compress=True` uses Zstandard tiles if a backend is available, else falls
    back to uncompressed (codec None) so writing never hard-fails on a missing
    dependency.
    """
    sample = _DEPTH_TO_SAMPLE.get(bit_depth)
    if sample is None:
        raise SyqError(
            f"SYQ does not support bit depth {bit_depth!r}. "
            f"Use one of: {sorted(_DEPTH_TO_SAMPLE)}."
        )
    dt = _SAMPLE_DTYPE[sample]

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim != 3:
        raise SyqError("Image must be 2-D (mono) or 3-D (HxWxC).")
    height, width, channels = arr.shape
    if not (0 < channels <= MAX_CHANNELS):
        raise SyqError(f"Unsupported channel count {channels}.")
    if not (0 < tile_size <= MAX_TILE_EDGE):
        raise SyqError(f"Tile size must be 1..{MAX_TILE_EDGE}.")

    # quantize
    if _SAMPLE_IS_INT[sample]:
        q = np.clip(arr.astype(np.float32), 0.0, 1.0)
        q = np.rint(q * _SAMPLE_MAXVAL[sample]).astype(dt)
    else:
        q = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(dt, copy=False)

    nx = (width + tile_size - 1) // tile_size
    ny = (height + tile_size - 1) // tile_size

    # object IDs: primary image = 1, tiles = 2..(1+ntiles)
    primary_id = 1
    tile_ids: List[int] = list(range(2, 2 + nx * ny))

    # descriptor (must reference tile IDs)
    md = dict(metadata or {})
    md.setdefault("syq.processing.software", "SASpro")
    desc = {
        WIRE.K_WIDTH: int(width),
        WIRE.K_HEIGHT: int(height),
        WIRE.K_CHANNELS: int(channels),
        WIRE.K_TILE_SIZE: int(tile_size),
        WIRE.K_SAMPLE: WIRE.SAMPLE_INT[sample],   # authoritative integer enum
        WIRE.K_UUID: str(_uuid.uuid4()),          # required key, no braces
        WIRE.K_ROLE: "MAIN_IMAGE",
        WIRE.K_TRANSFER: transfer,
        WIRE.K_STATE: state,
        WIRE.K_METADATA: md,
        WIRE.K_TILES: tile_ids,
    }
    desc_bytes = _cbor_encode(desc)

    # object list: (id, type, parent, logical_bytes)
    objects: List[Tuple[int, int, int, bytes]] = [
        (primary_id, WIRE.TYPE_IMAGE, 0, desc_bytes)
    ]
    for idx, tid in enumerate(tile_ids):
        ty, tx = divmod(idx, nx)
        x0, y0 = tx * tile_size, ty * tile_size
        tw = min(tile_size, width - x0)
        th = min(tile_size, height - y0)
        tile = q[y0:y0 + th, x0:x0 + tw, :]          # (th, tw, C)
        planar = np.ascontiguousarray(np.transpose(tile, (2, 0, 1)))  # (C, th, tw)
        objects.append((tid, WIRE.TYPE_TILE, primary_id, planar.tobytes()))

    # lay out payloads after the superblock
    body = bytearray()
    dir_entries: List[bytes] = []
    pos = WIRE.SB_SIZE
    for oid, otype, parent, logical in objects:
        stored = logical
        codec = WIRE.CODEC_NONE
        if compress and otype == WIRE.TYPE_TILE:   # never bother compressing the tiny descriptor
            z = _zstd_compress(logical)
            if z is not None and len(z) < len(logical):
                stored, codec = z, WIRE.CODEC_ZSTD
        offset = pos
        body += stored
        pos += len(stored)

        e = bytearray(WIRE.DE_SIZE)
        _setu(e, WIRE.DE_OBJECT_ID, oid)
        _setu(e, WIRE.DE_PARENT_ID, parent)
        _setu(e, WIRE.DE_TYPE, otype)
        _setu(e, WIRE.DE_FLAGS, 1)   # bit0 = required (IMAGE + TILE are Core)
        _setu(e, WIRE.DE_OFFSET, offset)
        _setu(e, WIRE.DE_STORED_LEN, len(stored))
        _setu(e, WIRE.DE_LOGICAL_LEN, len(logical))
        _setu(e, WIRE.DE_CODEC, codec)
        _setu(e, WIRE.DE_CKSUM_ALG, WIRE.CKSUM_SHA256)
        e[WIRE.DE_UNCOMPRESSED_SHA[0]:WIRE.DE_UNCOMPRESSED_SHA[1]] = hashlib.sha256(logical).digest()
        dir_entries.append(bytes(e))

    # directory
    dblock = bytearray()
    dblock += len(objects).to_bytes(8, "little")
    for e in dir_entries:
        dblock += e
    dir_offset = pos
    dir_length = len(dblock)
    pos += dir_length

    committed_size = pos + WIRE.FT_SIZE

    # superblock (dir geometry mirrors the footer for this single commit)
    sb = bytearray(WIRE.SB_SIZE)
    sb[WIRE.SB_MAGIC[0]:WIRE.SB_MAGIC[1]] = SYQ_MAGIC
    _setu(sb, WIRE.SB_MAJOR, SYQ_CORE_MAJOR)
    _setu(sb, WIRE.SB_MINOR, SYQ_CORE_MINOR)
    _setu(sb, WIRE.SB_HEADER_SIZE, WIRE.SB_SIZE)
    _setu(sb, WIRE.SB_REQUIRED_FEATURES, 0)
    _setu(sb, WIRE.SB_DIR_OFFSET, dir_offset)
    _setu(sb, WIRE.SB_DIR_LENGTH, dir_length)
    _setu(sb, WIRE.SB_PRIMARY_ID, primary_id)
    sb[WIRE.SB_UUID[0]:WIRE.SB_UUID[1]] = _uuid.uuid4().bytes
    _setu(sb, WIRE.SB_CREATED_NS, int(time.time() * 1e9))
    sb[WIRE.SB_DIGEST[0]:WIRE.SB_DIGEST[1]] = hashlib.sha256(
        bytes(sb[WIRE.SB_DIGEST_COVERS[0]:WIRE.SB_DIGEST_COVERS[1]])
    ).digest()

    # footer
    ft = bytearray(WIRE.FT_SIZE)
    ft[WIRE.FT_MAGIC[0]:WIRE.FT_MAGIC[1]] = SYQ_FOOTER_MAGIC
    _setu(ft, WIRE.FT_DIR_OFFSET, dir_offset)
    _setu(ft, WIRE.FT_DIR_LENGTH, dir_length)
    _setu(ft, WIRE.FT_PRIMARY_ID, primary_id)
    _setu(ft, WIRE.FT_FILE_SIZE, committed_size)
    ft[WIRE.FT_DIR_SHA[0]:WIRE.FT_DIR_SHA[1]] = hashlib.sha256(bytes(dblock)).digest()
    ft[WIRE.FT_FOOTER_SHA8[0]:WIRE.FT_FOOTER_SHA8[1]] = hashlib.sha256(
        bytes(ft[WIRE.FT_SHA_COVERS[0]:WIRE.FT_SHA_COVERS[1]])
    ).digest()[:8]

    # atomic replacement: write to a temp sibling then rename
    import os
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(bytes(sb))
        fh.write(bytes(body))
        fh.write(bytes(dblock))
        fh.write(bytes(ft))
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────
# Self-test: SASpro ↔ .syq round-trip (internal consistency only).
# ──────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import tempfile, os
    rng = np.random.default_rng(0)
    failures = 0
    cases = [
        ("mono f32", rng.random((300, 500)).astype(np.float32), "32-bit floating point"),
        ("rgb  f32", rng.random((513, 517, 3)).astype(np.float32), "32-bit floating point"),
        ("rgb  u16", rng.random((256, 256, 3)).astype(np.float32), "16-bit"),
        ("rgb  u8 ", rng.random((100, 640, 3)).astype(np.float32), "8-bit"),
        ("edge f32", rng.random((1000, 100, 3)).astype(np.float32), "32-bit floating point"),
    ]
    for name, img, depth in cases:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.syq")
            write_syq(img, p, bit_depth=depth, tile_size=256)
            back, hdr, bd, mono = read_syq(p)
            exp = img if img.ndim == 2 else img
            if depth == "32-bit floating point":
                ok = np.allclose(back, exp, atol=1e-6)
            elif depth == "16-bit":
                ok = np.max(np.abs(back - exp)) <= (1.0 / 65535) + 1e-6
            else:
                ok = np.max(np.abs(back - exp)) <= (1.0 / 255) + 1e-6
            same_shape = (back.shape == exp.shape)
            status = "ok" if (ok and same_shape) else "FAIL"
            if not (ok and same_shape):
                failures += 1
            print(f"  [{status}] {name:9s} depth={bd:22s} mono={mono} "
                  f"shape={back.shape} maxerr={float(np.max(np.abs(back-exp))):.2e}")
    print("SELFTEST:", "PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return failures


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    print(__doc__)