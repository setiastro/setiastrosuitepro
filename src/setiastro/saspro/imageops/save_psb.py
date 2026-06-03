# src/setiastro/saspro/imageops/save_psb.py
# SASpro — Photoshop Large Document (.psb) writer
# Copyright (c) 2026 Franklin Marek / SetiAstro
#
# Writes a minimal but valid PSB file directly from a numpy array.
# Supports:
#   - 32-bit float (mono or RGB)  — depth=32
#   - 16-bit integer (mono or RGB) — depth=16
#
# PSB vs PSD: identical structure except section lengths use 8 bytes instead
# of 4 bytes. This lifts the per-channel size limit from 2GB to ~2EB.
#
# References:
#   https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/
#   Section "Large Document Format (PSB)"

from __future__ import annotations

import struct
import numpy as np
from pathlib import Path
from typing import Union


def _pack_psb_header(
    n_channels: int,
    height: int,
    width: int,
    depth: int,        # bits per channel: 16 or 32
    color_mode: int,   # 1=grayscale, 3=RGB
) -> bytes:
    """
    PSB file header (26 bytes).
    Signature: '8BPS'
    Version: 2 (PSB), 1 = PSD
    """
    return struct.pack(
        ">4sH6xHIIHH",
        b"8BPS",      # signature
        2,            # version: 2 = PSB
        n_channels,   # number of channels (1 or 3; alpha would add 1)
        height,
        width,
        depth,        # bits per channel
        color_mode,   # 1=grayscale, 3=RGB
    )


def _empty_section() -> bytes:
    """A PSB length-prefixed empty section (8-byte length for PSB)."""
    return struct.pack(">Q", 0)


def _psb_section(data: bytes) -> bytes:
    """Wrap data in a PSB 8-byte length-prefixed section."""
    return struct.pack(">Q", len(data)) + data


def _image_data_raw_compression(channels: list[np.ndarray], depth: int) -> bytes:
    """
    Write image data section using compression mode 0 (raw / no compression).

    For 32-bit float, each pixel value must be written as big-endian IEEE 754.
    For 16-bit int, each pixel as big-endian uint16.

    PSB image data layout:
        2 bytes : compression type (0 = raw)
        then for each channel, scanline by scanline, big-endian values
    """
    out = bytearray()
    out += struct.pack(">H", 0)  # compression = 0 (raw)

    for ch in channels:
        arr = np.asarray(ch, dtype=ch.dtype)
        # PSB always big-endian
        if arr.dtype.byteorder not in (">", "=") or (
            arr.dtype.byteorder == "=" and np.dtype(arr.dtype).byteorder == "<"
        ):
            arr = arr.byteswap()
        # Ensure C-contiguous for tobytes()
        out += np.ascontiguousarray(arr).tobytes()

    return bytes(out)


def save_psb(
    path: Union[str, Path],
    image: np.ndarray,
    *,
    depth: int = 32,
) -> None:
    """
    Save a numpy array as a Photoshop Large Document (.psb).

    Parameters
    ----------
    path   : output file path (should end in .psb)
    image  : numpy array
               - HxW        → mono
               - HxWx1      → mono
               - HxWx3      → RGB
               - HxWx4      → RGBA (alpha channel included)
    depth  : output bit depth, 16 or 32 (default 32)
               32 = 32-bit float (best for astrophotography)
               16 = 16-bit integer (better PS compatibility for older workflows)

    Raises
    ------
    ValueError  : unsupported image shape or depth
    IOError     : write failure
    """
    path = Path(path)

    if depth not in (16, 32):
        raise ValueError(f"depth must be 16 or 32, got {depth}")

    arr = np.asarray(image, dtype=np.float32)

    # Normalise shape
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    H, W, C = arr.shape

    if C == 1:
        color_mode = 1   # grayscale
        n_channels = 1
    elif C == 3:
        color_mode = 3   # RGB
        n_channels = 3
    elif C == 4:
        color_mode = 3   # RGB + alpha
        n_channels = 4
    else:
        raise ValueError(f"Unsupported channel count: {C}. Must be 1, 3, or 4.")

    # ── Convert to target dtype ────────────────────────────────────────────
    if depth == 32:
        # PSB 32-bit float: values in [0,1] map directly
        channels = [
            arr[:, :, c].astype(np.float32)
            for c in range(C)
        ]
    else:
        # PSB 16-bit: scale [0,1] float → [0, 65535] uint16
        scaled = np.clip(arr, 0.0, 1.0)
        channels = [
            (scaled[:, :, c] * 65535.0).astype(np.uint16)
            for c in range(C)
        ]

    # ── Endian conversion (PSB = big-endian) ──────────────────────────────
    be_channels = []
    for ch in channels:
        if depth == 32:
            be_channels.append(ch.astype(">f4"))
        else:
            be_channels.append(ch.astype(">u2"))

    # ── Assemble file ─────────────────────────────────────────────────────
    header      = _pack_psb_header(n_channels, H, W, depth, color_mode)
    color_data  = _empty_section()   # Color Mode Data (empty for RGB/grayscale)
    image_res   = _empty_section()   # Image Resources (empty — no metadata)
    layer_info  = _empty_section()   # Layer and Mask Info (empty — flat image)
    image_data  = _image_data_raw_compression(be_channels, depth)

    with open(path, "wb") as f:
        f.write(header)
        f.write(color_data)
        f.write(image_res)
        f.write(layer_info)
        f.write(image_data)


def can_save_psb() -> bool:
    """Always True — no external dependencies required."""
    return True

def load_psb(path: Union[str, Path]) -> tuple[np.ndarray, str, bool]:
    """
    Load a flat (no layers) Photoshop Large Document (.psb) as float32 [0,1].

    Returns
    -------
    (image, bit_depth, is_mono)
        image     : float32 ndarray, shape (H,W) mono or (H,W,3) RGB
        bit_depth : "16-bit" or "32-bit floating point"
        is_mono   : True if single channel
    """
    path = Path(path)

    with open(path, "rb") as f:
        # ── Header (26 bytes) ────────────────────────────────────────────
        sig     = f.read(4)
        if sig != b"8BPS":
            raise ValueError(f"Not a PSD/PSB file: {path}")
        version = struct.unpack(">H", f.read(2))[0]
        if version not in (1, 2):
            raise ValueError(f"Unsupported PSD/PSB version: {version}")
        f.read(6)   # reserved
        n_channels = struct.unpack(">H", f.read(2))[0]
        height     = struct.unpack(">I", f.read(4))[0]
        width      = struct.unpack(">I", f.read(4))[0]
        depth      = struct.unpack(">H", f.read(2))[0]   # bits per channel
        color_mode = struct.unpack(">H", f.read(2))[0]   # 1=grayscale, 3=RGB

        if color_mode not in (1, 3):
            raise ValueError(f"Unsupported PSB color mode: {color_mode}. Only grayscale (1) and RGB (3) supported.")
        if depth not in (16, 32):
            raise ValueError(f"Unsupported PSB bit depth: {depth}. Only 16 and 32 supported.")

        # ── Section lengths: 4 bytes for PSD (v1), 8 bytes for PSB (v2) ──
        len_size = 8 if version == 2 else 4
        len_fmt  = ">Q" if version == 2 else ">I"

        def _skip_section():
            n = struct.unpack(len_fmt, f.read(len_size))[0]
            if n:
                f.seek(n, 1)

        _skip_section()   # Color Mode Data
        _skip_section()   # Image Resources
        _skip_section()   # Layer and Mask Info

        # ── Image Data ───────────────────────────────────────────────────
        compression = struct.unpack(">H", f.read(2))[0]

        if compression == 0:
            # Raw: channels written sequentially, big-endian
            data = f.read()
        elif compression == 1:
            # PackBits RLE — skip the row byte-count table first, then unpack
            if version == 2:
                # PSB: 4 bytes per row per channel
                total_rows = n_channels * height
                f.read(total_rows * 4)
            else:
                # PSD: 2 bytes per row per channel
                total_rows = n_channels * height
                f.read(total_rows * 2)
            data = _unpack_packbits(f.read())
        else:
            raise ValueError(f"Unsupported PSB compression: {compression}. Only raw (0) and PackBits (1) supported.")

        # ── Decode channels ──────────────────────────────────────────────
        bytes_per_sample = depth // 8
        samples_per_channel = height * width
        channel_byte_size = samples_per_channel * bytes_per_sample

        channels = []
        for c in range(min(n_channels, 3)):   # skip alpha if present
            raw = data[c * channel_byte_size : (c + 1) * channel_byte_size]
            if len(raw) < channel_byte_size:
                raise ValueError(f"PSB channel {c} data truncated: got {len(raw)}, expected {channel_byte_size}")

            arr = np.frombuffer(raw, dtype=np.dtype(f">{'f' if depth == 32 else 'u' + str(bytes_per_sample)}"))
            arr = arr.astype(np.float32)

            if depth == 16:
                arr /= 65535.0

            channels.append(arr.reshape(height, width))

        # ── Assemble ─────────────────────────────────────────────────────
        n_image_channels = min(n_channels, 3)
        is_mono = (color_mode == 1 or n_image_channels == 1)

        if is_mono:
            image = channels[0]
        else:
            image = np.stack(channels, axis=-1)

        bit_depth_str = "32-bit floating point" if depth == 32 else "16-bit"
        return np.clip(image, 0.0, 1.0).astype(np.float32), bit_depth_str, is_mono


def _unpack_packbits(data: bytes) -> bytes:
    """Decompress PackBits RLE as used by PSD/PSB."""
    out = bytearray()
    i = 0
    while i < len(data):
        header = data[i]
        i += 1
        if header == 128:
            pass   # nop
        elif header < 128:
            # copy (header + 1) literal bytes
            count = header + 1
            out.extend(data[i:i + count])
            i += count
        else:
            # repeat next byte (256 - header + 1) times
            count = 256 - header + 1
            out.extend([data[i]] * count)
            i += 1
    return bytes(out)