#src/setiastro/saspro/blink_fast_read.py
"""Strided reader for Blink's 1:4 / 1:8 load modes -- FITS and XISF.

Blink's resolution buttons save memory but not time: ``_load_one_image`` reads
the whole frame through ``load_image``, debayers it at full resolution, and
only then ``cv2.resize``s it down. At 1:8 that is 100% of the I/O and 100% of
the debayer for 1.5% of the pixels.

This reads only the rows the downsampled frame needs, through a plain
per-thread file handle. Deliberately NOT memmap: legacy/image_manager.py
switched the FITS loader to ``memmap=False`` precisely because faulting mmap
pages from Blink's parallel loader corrupts the heap, and a seek/read handle
shares no state at all. The FITS header probe goes under ``_FITS_IO_LOCK``
because astropy's open path is the part that is not thread-safe; the XISF
probe needs no lock (it is ElementTree over a header string) and neither does
any pixel read here.

Both formats store pixels the same way once you have the offset: uncompressed,
row-major, one contiguous plane per channel. So the probe differs per format
and the reader does not.

Scope is narrow on purpose. Anything whose normalization this module cannot
reproduce exactly as ``load_image`` would -- tile-compressed or gzipped FITS,
compressed or inline XISF blocks, the FITS dtypes ``load_image`` min-max
normalizes -- returns ``None``, and the caller falls through.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from astropy.io import fits

from setiastro.saspro.legacy.image_manager import (
    _FITS_IO_LOCK,
    _apply_roworder_flip,
    _drop_invalid_cards,
    _finalize_loaded_image,
    promote_xisf_metadata_to_header,
    read_embedded_metadata_header,
    xisf_fits_header_from_meta,
)
from setiastro.saspro.legacy.xisf import XISF

# Only plain, uncompressed containers. .gz and .fz have no contiguous pixel
# block to seek into, so they go the slow way.
PLAIN_FITS_SUFFIXES = (".fits", ".fit", ".fts")
XISF_SUFFIXES = (".xisf",)

# FITS pixel data is big-endian on disk, always. BITPIX 32 and 64 are absent
# on purpose -- see _fits_plan().
_BITPIX_DTYPE = {8: ">u1", 16: ">i2", -32: ">f4", -64: ">f8"}

# 1:2 is handled by the existing path, not here. The box below needs two rows
# per cell, so at step 2 it skips nothing: it transfers the whole pixel block
# in thousands of small reads and loses to one big read plus cv2.INTER_AREA.
# Measured, uint16 FITS, current path vs this one:
#
#     frame          1:2     1:4     1:8
#     1392 x 1040    0.31x   1.01x   2.12x
#     3008 x 2048    0.60x   1.72x   5.16x
#     6000 x 4000    0.66x   2.11x   8.17x
#     9576 x 6388    0.82x   2.62x  12.45x
#
# Reading whole cells contiguously instead was tried and is also slower than
# the current path at 1:2 (0.55-0.66x): it transfers the same bytes and its
# reduction is slower than cv2's. There is simply no win at 1:2.
MIN_STEP = 4


@dataclass(frozen=True)
class _Source:
    """Everything the reader needs, with the format already abstracted away."""
    header: fits.Header
    offsets: Tuple[int, ...]      # byte offset of each channel plane
    width: int
    height: int
    dtype: np.dtype               # on-disk dtype, byte order included
    to_float01: Callable          # raw samples -> load_image's float32 [0,1]
    bit_depth: str                # the label load_image would report
    cfa: bool
    roworder_applies: bool        # load_image only flips FITS, never XISF


# ---------------------------------------------------------------------------
# FITS
# ---------------------------------------------------------------------------

def _fits_plan(hdr):
    """``(to_float01, bit_depth_label)`` matching ``load_image``, or None.

    ``load_image`` min-max normalizes BITPIX 32, BITPIX 64 and plain signed
    16-bit. A min/max taken over a subsample is not the frame's min/max, so
    the same file would normalize differently at 1:1 and 1:4 and every BG
    reading and weighted score would shift with the zoom button. Those return
    None rather than quietly disagree with themselves.

    Everything accepted here is affine, and affine maps commute with the box
    average below -- so scaling raw samples after averaging gives the same
    answer as load_image's scale-then-resize. The one exception is the clip in
    the 16-bit branch, a no-op for the standard BZERO=32768 case.
    """
    try:
        bitpix = int(hdr["BITPIX"])
    except Exception:
        return None

    try:
        bzero = float(hdr.get("BZERO", 0.0) or 0.0)
        bscale = float(hdr.get("BSCALE", 1.0) or 1.0)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(bzero) and np.isfinite(bscale)) or bscale == 0.0:
        return None

    if bitpix == 8 and bzero == 0.0 and bscale == 1.0:
        return (lambda a: a * (1.0 / 255.0)), "8-bit"

    if bitpix == 16:
        if bzero == 0.0 and bscale == 1.0:
            return None          # true signed int16 -> load_image min-max

        # The universal CMOS/CCD case is BZERO=32768, BSCALE=1: stored
        # s = p - 32768, so p = s + 32768. astropy hands load_image a uint16
        # array for this and load_image divides by 65535; the affine below is
        # the same number taken from the raw signed samples.
        label = ("16-bit" if (bzero == 32768.0 and bscale == 1.0)
                 else "16-bit signed")

        def _affine(a, _s=bscale, _z=bzero):
            return np.clip((a * _s + _z) * (1.0 / 65535.0), 0.0, 1.0)

        return _affine, label

    if bitpix in (-32, -64) and bzero == 0.0 and bscale == 1.0:
        # load_image ignores BSCALE/BZERO on floats entirely; rather than
        # reproduce that, non-trivial scaling falls back.
        label = ("32-bit floating point" if bitpix == -32
                 else "64-bit floating point")
        return (lambda a: a), label

    return None


def _probe_fits(path):
    """Build a _Source for a plain FITS file, or None.

    Header-only: astropy never touches ``.data`` here, so this costs a couple
    of 2880-byte blocks.
    """
    with _FITS_IO_LOCK:
        with fits.open(path, memmap=False, lazy_load_hdus=True,
                       do_not_scale_image_data=True) as hdul:
            target = None
            for hdu in hdul:
                if isinstance(hdu, fits.CompImageHDU):
                    return None      # tile-compressed: no contiguous plane
                if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)):
                    continue
                h = hdu.header
                naxis = int(h.get("NAXIS", 0))
                if naxis not in (2, 3):
                    continue
                if naxis == 3 and int(h.get("NAXIS3", 0)) != 3:
                    continue         # only planar RGB; anything else falls back
                if min(int(h.get("NAXIS1", 0)), int(h.get("NAXIS2", 0))) < 8:
                    continue
                target = hdu
                break
            if target is None:
                return None

            info = target.fileinfo() or {}
            dat_loc = info.get("datLoc")
            if dat_loc is None:
                return None

            hdr = hdul[0].header.copy()
            hdr.update(target.header)
            if "BAYERPAT" not in hdr:    # mirror get_valid_header
                for other in hdul:
                    if "BAYERPAT" in other.header:
                        hdr["BAYERPAT"] = other.header["BAYERPAT"]
                        break
            hdr = _drop_invalid_cards(hdr)

    plan = _fits_plan(hdr)
    if plan is None:
        return None
    to_float01, bit_depth = plan

    bitpix = int(hdr["BITPIX"])
    if bitpix not in _BITPIX_DTYPE:
        return None
    dtype = np.dtype(_BITPIX_DTYPE[bitpix])
    w, h = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    nchan = int(hdr.get("NAXIS3", 1)) if int(hdr.get("NAXIS", 2)) == 3 else 1
    plane = w * h * dtype.itemsize
    offsets = tuple(int(dat_loc) + c * plane for c in range(nchan))

    return _Source(
        header=hdr, offsets=offsets, width=w, height=h, dtype=dtype,
        to_float01=to_float01, bit_depth=bit_depth,
        cfa=bool(str(hdr.get("BAYERPAT", "")).strip()),
        roworder_applies=True,
    )


# ---------------------------------------------------------------------------
# XISF
# ---------------------------------------------------------------------------

# load_image's XISF branch scales by dtype alone -- no min-max anywhere -- so
# every sample format it supports is servable here.
_XISF_PLAN = {
    "uint8":   (1.0 / 255.0,        "8-bit"),
    "uint16":  (1.0 / 65535.0,      "16-bit"),
    "uint32":  (1.0 / 4294967295.0, "32-bit unsigned"),
    # load_image labels float64 "32-bit floating point" too, and casts without
    # rescaling. Matched deliberately.
    "float32": (None,               "32-bit floating point"),
    "float64": (None,               "32-bit floating point"),
}


def _probe_xisf(path):
    """Build a _Source for an uncompressed, attached-block XISF file, or None.

    Reads the XML header only; pixel blocks and PixInsight's lazy property
    blocks are left on disk.
    """
    xisf = XISF(path)                       # header parse only
    metas = xisf.get_images_metadata()
    if not metas:
        return None
    meta = metas[0]

    # Attachment + uncompressed is exactly what a strided read needs.
    try:
        if not xisf.can_partial_read_image(0):
            return None
    except Exception:
        return None
    if str(meta.get("byteOrder", "little")).lower() == "big":
        return None                          # XISF's rare big-endian variant

    try:
        w, h, nchan = meta["geometry"]
    except Exception:
        return None
    if nchan not in (1, 3) or min(w, h) < 8:
        return None

    dtype = np.dtype(meta["dtype"]).newbyteorder("<")
    plan = _XISF_PLAN.get(dtype.name)
    if plan is None:
        return None
    scale, bit_depth = plan
    to_float01 = (lambda a: a) if scale is None else (lambda a, _s=scale: a * _s)

    # The Blink-relevant header: OBJECT / FILTER / EXPTIME for the tree,
    # CCD-TEMP / GAIN / OFFSET for the metrics, BAYERPAT for the debayer.
    # All of those come from FITSKeywords, which cost nothing to read.
    #
    # Deliberately NOT load_image's full XISF header build: that one fits a
    # SIP polynomial to PixInsight's astrometric grids, which resolves
    # multi-megabyte attached blocks and runs several least-squares fits per
    # file. The resulting WCS is invalid for a downsampled frame anyway -- the
    # existing cv2.resize path does not rescale it either -- so a Blink
    # preview pays nothing for it here.
    hdr = xisf_fits_header_from_meta(meta, xisf.get_file_metadata())
    try:
        hdr = promote_xisf_metadata_to_header(hdr, meta, xisf.get_file_metadata())
    except Exception:
        pass
    hdr = _drop_invalid_cards(hdr)
    try:
        hdr.add_history("SASpro Blink: fast strided XISF read; "
                        "astrometric solution not parsed")
    except Exception:
        pass

    _, pos, _size = meta["location"]
    plane = w * h * dtype.itemsize
    offsets = tuple(int(pos) + c * plane for c in range(nchan))

    return _Source(
        header=hdr, offsets=offsets, width=int(w), height=int(h), dtype=dtype,
        to_float01=to_float01, bit_depth=bit_depth,
        cfa=bool(str(hdr.get("BAYERPAT", "")).strip()),
        # load_image's XISF branch never calls _apply_roworder_flip: XISF is
        # top-down by specification.
        roworder_applies=False,
    )


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

_FLIP_PROBE = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)


def _needs_vertical_flip(hdr):
    """Would image_manager flip this frame? Asked, not reimplemented.

    Runs the real ``_apply_roworder_flip`` over a two-row sentinel, so the
    ROWORDER keyword handling and the Siril/IRIS writer-stamp inference stay
    in one place instead of two that drift apart.
    """
    probe, _ = _apply_roworder_flip(_FLIP_PROBE.copy(), hdr.copy())
    return bool(probe[0, 0] == 1.0)


def _read_plane_box(f, base, src, step, ny, nx, scrub):
    """One plane, each output pixel the mean of a 2 x step box in its cell."""
    row_bytes = src.width * src.dtype.itemsize
    usable = nx * step
    off = step // 2 - 1             # centre the 2-row box inside its cell
    out = np.empty((ny, nx), dtype=np.float32)
    for j in range(ny):
        f.seek(base + (j * step + off) * row_bytes)
        buf = f.read(2 * row_bytes)
        if len(buf) < 2 * row_bytes:
            raise ValueError(f"short read on pixel data at row {j}")
        pair = np.frombuffer(buf, dtype=src.dtype).reshape(2, src.width)[:, :usable]
        out[j] = scrub(pair.astype(np.float32)).reshape(2, nx, step).mean(axis=(0, 2))
    return out


def _read_plane_mosaic(f, base, src, step, nqy, nqx, scrub):
    """One CFA plane, kept as a mosaic: one intact quad per 2*step cell.

    The row pair must start on an even row and the column span must cover
    whole 2-column periods, or the output quad is no longer in the input's
    CFA phase.
    """
    row_bytes = src.width * src.dtype.itemsize
    usable = nqx * 2 * step
    off = step                      # even, just past the middle of the cell
    out = np.empty((2 * nqy, 2 * nqx), dtype=np.float32)
    for m in range(nqy):
        f.seek(base + (m * 2 * step + off) * row_bytes)
        buf = f.read(2 * row_bytes)
        if len(buf) < 2 * row_bytes:
            raise ValueError(f"short read on pixel data at quad row {m}")
        pair = np.frombuffer(buf, dtype=src.dtype).reshape(2, src.width)[:, :usable]
        # (row parity, quad column, column group, column parity)
        quads = scrub(pair.astype(np.float32)).reshape(2, nqx, step, 2).mean(axis=2)
        out[2 * m:2 * m + 2] = quads.reshape(2, 2 * nqx)
    return out


def read_downsampled(path, step, *, cfa_mode="color"):
    """Read a FITS or XISF frame downsampled by ``step``, or None to fall back.

    Returns ``(image_float01, header, bit_depth, needs_debayer)``. When
    ``needs_debayer`` is True the array is a *mosaic* and the caller must
    debayer it; otherwise it is finished (2-D mono, or H x W x 3 for a
    three-channel source). Either way the frame the caller ends up with is
    ``height // step`` by ``width // step``, the same size
    ``_load_one_image``'s ``cv2.resize`` produces today.

    Only ``step >= MIN_STEP`` is served; 1:2 returns None and keeps today's
    behaviour exactly.

    ``cfa_mode`` decides what happens to a frame carrying BAYERPAT:

    * ``"color"`` (default) -- read two rows out of every ``2*step``, both
      quad-aligned, and average each CFA channel separately over the
      ``2 x 2*step`` box. That yields one intact quad per cell, so the result
      is a genuine mosaic at the target size with BAYERPAT unchanged, and the
      caller's existing debayer turns it into colour. Transfers ``1/step`` of
      the pixel block.

    * ``"luminance"`` -- average whole quads into a single grey value. No
      mosaic survives, so BAYERPAT is dropped and no debayer runs at all.
      Transfers ``2/step``.

    * ``"fallback"`` -- CFA frames return None and go down the old path.

    Mono and RGB frames ignore ``cfa_mode`` and always take the box geometry.

    Noise, measured against the current debayer-then-INTER_AREA path on a
    noisy synthetic CFA flat (ratio > 1 means noisier than today):

                     R      G      B
      1:4 colour    0.93   1.26   0.92
      1:8 colour    1.18   1.66   1.16

    Red and blue come out at or near parity because colour mode spends its
    whole sample budget on real CFA samples instead of interpolated ones.
    Green is the weak channel: the sensor gives it twice the density, and the
    old path's full-cell average exploits that while this one takes ``step``
    samples per green position like any other. Averaging G1 and G2 together
    into both green slots would recover most of that at the cost of hiding a
    real G1/G2 imbalance -- a reasonable trade for a preview, left off here
    because it makes the mosaic no longer faithful to the sensor.

    On a mono plane the box gives sigma/sqrt(2*step) where INTER_AREA gives
    sigma/step, so a 1:8 preview is about 2x noisier (1:4 about 1.4x).
    Reading all ``step`` rows would match it exactly and save no I/O at all,
    which is the entire point of this module.

    Colour mode declines BOTTOM-UP FITS frames. Flipping a mosaic vertically
    changes its CFA phase, and whether SASpro already compensates for that
    lives in bayer_utils, so those keep the proven full-resolution path. XISF
    is top-down by specification and is never flipped.

    Raises on a structurally bad file (truncated pixel block); the caller
    should catch and fall back.
    """
    step = int(step)
    if step < MIN_STEP or step % 2:
        return None                  # odd steps would unbalance the CFA box
    if cfa_mode not in ("color", "luminance", "fallback"):
        raise ValueError(
            f"cfa_mode must be color/luminance/fallback, got {cfa_mode!r}")

    low = str(path).lower()
    if low.endswith(PLAIN_FITS_SUFFIXES):
        src = _probe_fits(path)
    elif low.endswith(XISF_SUFFIXES):
        src = _probe_xisf(path)
    else:
        return None
    if src is None:
        return None

    if src.cfa and cfa_mode == "fallback":
        return None
    keep_mosaic = src.cfa and cfa_mode == "color"
    if keep_mosaic and src.roworder_applies and _needs_vertical_flip(src.header):
        return None                  # see docstring: flip changes CFA phase

    w, h = src.width, src.height
    row_bytes = w * src.dtype.itemsize
    need = max(src.offsets) + h * row_bytes
    have = os.path.getsize(path)
    if need > have:
        raise ValueError(
            f"truncated pixel data: need {need} bytes, file is {have}")

    is_float = src.dtype.kind == "f"

    def _scrub(a):
        if is_float:
            # load_image scrubs these before any resize; do the same so a
            # single bad sample cannot poison a whole output cell.
            np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        return a

    if keep_mosaic:
        nqy, nqx = h // (2 * step), w // (2 * step)
        if nqy < 2 or nqx < 2:
            return None
        with open(path, "rb", buffering=0) as f:
            out = _read_plane_mosaic(f, src.offsets[0], src, step, nqy, nqx, _scrub)
    else:
        ny, nx = h // step, w // step
        if ny < 4 or nx < 4:
            return None              # too small for the seeks to pay off
        # buffering=0: we read whole rows at a time, so Python's buffer would
        # only add a copy.
        with open(path, "rb", buffering=0) as f:
            planes = [_read_plane_box(f, base, src, step, ny, nx, _scrub)
                      for base in src.offsets]
        out = planes[0] if len(planes) == 1 else np.stack(planes, axis=-1)

    out = src.to_float01(out)
    out = _finalize_loaded_image(out)

    hdr = src.header
    if not keep_mosaic and src.roworder_applies:
        # Same BOTTOM-UP handling the full loader applies. Note this updates
        # CRPIX2 against the REDUCED height, so the header's WCS is wrong by
        # the scale factor -- exactly as wrong as it already is today, since
        # the existing cv2.resize path never rescales the WCS either.
        out, hdr = _apply_roworder_flip(out, hdr)

    if src.cfa and not keep_mosaic:
        # The box averaged whole quads, so this plane is luminance, not a
        # mosaic -- there is no CFA phase left in it to recover. Drop BAYERPAT
        # so nothing downstream sees a 2-D plane still tagged as Bayer and
        # debayers it a second time: detect_bayer_pattern() reads the header,
        # and push_to_docs hands this header to the document manager as the
        # frame's original_header.
        try:
            del hdr["BAYERPAT"]
            hdr.add_history(
                f"SASpro Blink: CFA averaged to luminance at 1:{step}; "
                "BAYERPAT removed (plane is no longer a mosaic)")
        except Exception:
            pass

    return out, hdr, src.bit_depth, keep_mosaic


# ---------------------------------------------------------------------------
# Header-only probe, for Stream mode's initial scan
# ---------------------------------------------------------------------------

_COMPRESSED_FITS_SUFFIXES = (".fits.gz", ".fit.gz", ".fts.gz", ".fz")
_EMBEDDED_META_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp")


def probe_header(path):
    """``(header, (width, height))`` without decoding pixels, for Stream mode.

    Stream mode builds the Object / Filter / Exposure tree before any frame is
    decoded, so it needs the header on its own. Either element may be None:

    * FITS (including .gz and .fz) and XISF give both -- these are the formats
      people point a 500-sub folder at, and both keep their metadata in a
      header block that costs a few KB to read.
    * TIFF / PNG / JPEG / WebP give a header only if SASpro wrote one into
      them (``read_embedded_metadata_header``), and never a size.
    * RAW and anything else give nothing. Those frames land under
      "Object: Unknown" until they are first viewed, at which point the real
      header replaces the placeholder. Reading EXIF for them would work but
      is not worth a second decode path here.

    Never raises: a file that cannot be probed is simply unknown, and Stream
    mode still lists it.
    """
    low = str(path).lower()
    try:
        if low.endswith(_COMPRESSED_FITS_SUFFIXES) or low.endswith(PLAIN_FITS_SUFFIXES):
            # NOT get_valid_header: that one tests `hdu.data is not None` to
            # find the image HDU, and with memmap=False astropy materializes
            # the whole pixel block to answer it -- 39.6 ms against 0.5 ms for
            # a header read, on a 48 MB frame. Streaming a folder through that
            # would decode every file during the scan, which is the one thing
            # Stream mode exists to avoid.
            with _FITS_IO_LOCK:
                with fits.open(path, memmap=False, lazy_load_hdus=True,
                               do_not_scale_image_data=True) as hdul:
                    hdr = None
                    for hdu in hdul:
                        h = hdu.header
                        naxis = int(h.get("NAXIS", 0) or 0)
                        if naxis < 2:
                            continue
                        if min(int(h.get("NAXIS1", 0) or 0),
                               int(h.get("NAXIS2", 0) or 0)) < 1:
                            continue
                        hdr = hdul[0].header.copy()
                        hdr.update(h)
                        break
                    if hdr is None:
                        return None, None
                    if "BAYERPAT" not in hdr:
                        for other in hdul:
                            if "BAYERPAT" in other.header:
                                hdr["BAYERPAT"] = other.header["BAYERPAT"]
                                break
                    hdr = _drop_invalid_cards(hdr)
            try:
                wh = (int(hdr["NAXIS1"]), int(hdr["NAXIS2"]))
            except Exception:
                wh = None
            return hdr, wh

        if low.endswith(XISF_SUFFIXES):
            xisf = XISF(path)
            metas = xisf.get_images_metadata()
            if not metas:
                return None, None
            meta = metas[0]
            file_meta = xisf.get_file_metadata()
            hdr = xisf_fits_header_from_meta(meta, file_meta)
            try:
                hdr = promote_xisf_metadata_to_header(hdr, meta, file_meta)
            except Exception:
                pass
            hdr = _drop_invalid_cards(hdr)
            try:
                w, h, _c = meta["geometry"]
                wh = (int(w), int(h))
            except Exception:
                wh = None
            return hdr, wh

        if low.endswith(_EMBEDDED_META_SUFFIXES):
            return read_embedded_metadata_header(path), None
    except Exception as e:
        print(f"[Blink] header probe failed for {path}: {e}")
    return None, None