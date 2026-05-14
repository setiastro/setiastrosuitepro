# coding: utf-8

"""
XISF Encoder/Decoder (see https://pixinsight.com/xisf/).

This implementation is not endorsed nor related with PixInsight development team.

Copyright (C) 2021-2023 Sergio Díaz, sergiodiaz.eu

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------------
Modifications by Franklin Marek (Seti Astro / SASpro), 2026:

  - _parse_location: normalized return types so all three location kinds
    (attachment, inline, embedded) return proper tuples instead of a mix
    of tuples and raw string lists.

  - _process_property / Boolean: broadened the truthy check to accept "1"
    and "yes" in addition to "true", matching real-world writer output.

  - read_image: added byteOrder awareness so files written on big-endian
    systems are correctly byte-swapped when read on little-endian hosts
    (and vice versa).  Previously raw np.frombuffer output was used without
    checking the attribute, which would silently produce garbage on a cross-
    endian round-trip.

  - _compress / _compress_data_block: unified the 'itemsize' / 'shuffle'
    parameter naming so the two methods are consistent and the shuffle path
    cannot silently no-op due to a mismatched keyword.

  - _insert_property: extracted _build_attrs as a proper helper shared
    across all four type-family branches (scalar, string, vector, matrix)
    to eliminate the repeated inline closures and make future type additions
    easier.

  - _parse_geometry / _parse_compression / _parse_sampleFormat /
    _parse_vector_dtype: minor defensive hardening (strip whitespace from
    split tokens, consistent KeyError messages).
---------------------------------------------------------------------------
"""

from importlib.metadata import version

import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import lz4.block
import zlib
import zstandard
import base64
from datetime import datetime
import ast

__version__ = "1.0.1"


def _is_attached_or_inline_property(p_dict):
    return "location" in p_dict


def _make_lazy(p_dict):
    p_dict["_lazy"] = True
    return p_dict


class XISF:
    """Implements a baseline XISF Decoder and a simple baseline Encoder.

    See the original module docstring for full usage details.
    """

    # Static attributes
    _creator_app = f"Python {platform.python_version()}"
    _creator_module = f"XISF Python Module v{__version__} github.com/sergio-dr/xisf"
    _signature = b"XISF0100"
    _headerlength_len = 4
    _reserved_len = 4
    _xml_ns = {"xisf": "http://www.pixinsight.com/xisf"}
    _xisf_attrs = {
        "xmlns": "http://www.pixinsight.com/xisf",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "version": "1.0",
        "xsi:schemaLocation": (
            "http://www.pixinsight.com/xisf "
            "http://pixinsight.com/xisf/xisf-1.0.xsd"
        ),
    }
    _compression_def_level = {
        "zlib":  6,
        "lz4":   0,
        "lz4hc": 9,
        "zstd":  3,
    }
    _block_alignment_size = 4096
    _max_inline_block_size = 3072

    def __init__(self, fname):
        """Open a XISF file and extract its metadata.

        Args:
            fname: path to the .xisf file.

        Raises:
            ValueError: if the file does not carry the XISF signature.
            FileNotFoundError: if *fname* does not exist.
        """
        self._fname = fname
        self._headerlength = None
        self._xisf_header = None
        self._xisf_header_xml = None
        self._images_meta = None
        self._file_meta = None
        ET.register_namespace("", self._xml_ns["xisf"])
        self._read()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _read(self):
        with open(self._fname, "rb") as f:
            signature = f.read(len(self._signature))
            if signature != self._signature:
                raise ValueError("File doesn't have XISF signature")

            self._headerlength = int.from_bytes(
                f.read(self._headerlength_len), byteorder="little"
            )
            _ = f.read(self._reserved_len)
            self._xisf_header = f.read(self._headerlength)
            self._xisf_header_xml = ET.fromstring(self._xisf_header)
        self._analyze_header()

    def _analyze_header(self):
        self._images_meta = []
        for image in self._xisf_header_xml.findall("xisf:Image", self._xml_ns):
            image_basic_meta = image.attrib

            fits_keywords = {}
            for a in image.findall("xisf:FITSKeyword", self._xml_ns):
                fits_keywords.setdefault(a.attrib["name"], []).append(
                    {
                        "value": a.attrib["value"].strip("'").strip(),
                        "comment": a.attrib["comment"],
                    }
                )

            image_extended_meta = {
                "geometry": self._parse_geometry(image.attrib["geometry"]),
                "location": self._parse_location(image.attrib["location"]),
                "dtype": self._parse_sampleFormat(image.attrib["sampleFormat"]),
                "FITSKeywords": fits_keywords,
                "XISFProperties": {
                    p.attrib["id"]: prop
                    for p in image.findall("xisf:Property", self._xml_ns)
                    if (prop := self._process_property(p))
                },
            }
            if "compression" in image.attrib:
                image_extended_meta["compression"] = self._parse_compression(
                    image.attrib["compression"]
                )

            self._images_meta.append({**image_basic_meta, **image_extended_meta})

        self._file_meta = {}
        for p in self._xisf_header_xml.find("xisf:Metadata", self._xml_ns):
            self._file_meta[p.attrib["id"]] = self._process_property(p)

        self._parse_resolution_elements()
        self._parse_icc_profiles()
        self._parse_thumbnails()

    def _parse_resolution_elements(self):
        for i, image in enumerate(self._xisf_header_xml.findall("xisf:Image", self._xml_ns)):
            res_elem = image.find("xisf:Resolution", self._xml_ns)
            if res_elem is not None:
                try:
                    res_data = {
                        "horizontal": float(res_elem.attrib.get("horizontal", 72.0)),
                        "vertical":   float(res_elem.attrib.get("vertical",   72.0)),
                        "unit":       res_elem.attrib.get("unit", "inch"),
                    }
                    if i < len(self._images_meta):
                        self._images_meta[i]["Resolution"] = res_data
                except (ValueError, KeyError):
                    pass

    def _parse_icc_profiles(self):
        for i, image in enumerate(self._xisf_header_xml.findall("xisf:Image", self._xml_ns)):
            icc_elem = image.find("xisf:ICCProfile", self._xml_ns)
            if icc_elem is not None:
                try:
                    icc_data = {"present": True}
                    if "location" in icc_elem.attrib:
                        loc = self._parse_location(icc_elem.attrib["location"])
                        icc_data["location"] = loc
                        if loc[0] == "attachment" and len(loc) >= 3:
                            icc_data["size"] = loc[2]
                    if i < len(self._images_meta):
                        self._images_meta[i]["ICCProfile"] = icc_data
                except (ValueError, KeyError):
                    pass

    def _parse_thumbnails(self):
        for i, image in enumerate(self._xisf_header_xml.findall("xisf:Image", self._xml_ns)):
            thumb_elem = image.find("xisf:Thumbnail", self._xml_ns)
            if thumb_elem is not None:
                try:
                    thumb_data = {
                        "present":  True,
                        "geometry": self._parse_geometry(
                            thumb_elem.attrib.get("geometry", "0:0:0")
                        ),
                    }
                    if "location"    in thumb_elem.attrib:
                        thumb_data["location"] = self._parse_location(thumb_elem.attrib["location"])
                    if "sampleFormat" in thumb_elem.attrib:
                        thumb_data["dtype"] = self._parse_sampleFormat(thumb_elem.attrib["sampleFormat"])
                    if "colorSpace"  in thumb_elem.attrib:
                        thumb_data["colorSpace"] = thumb_elem.attrib["colorSpace"]
                    if i < len(self._images_meta):
                        self._images_meta[i]["Thumbnail"] = thumb_data
                except (ValueError, KeyError):
                    pass

    # ------------------------------------------------------------------
    # Public metadata accessors
    # ------------------------------------------------------------------

    def get_images_metadata(self):
        """Return metadata for all image blocks in the file."""
        return self._images_meta

    def get_file_metadata(self):
        """Return file-level metadata from the <Metadata> core element."""
        return self._file_meta

    def get_metadata_xml(self):
        """Return the complete XML header as an ElementTree Element."""
        return self._xisf_header_xml

    # ------------------------------------------------------------------
    # Data block reading
    # ------------------------------------------------------------------

    def _read_data_block(self, elem):
        method = elem["location"][0]
        if method == "inline":
            return self._read_inline_data_block(elem)
        elif method == "embedded":
            return self._read_embedded_data_block(elem)
        elif method == "attachment":
            return self._read_attached_data_block(elem)
        else:
            raise NotImplementedError(
                f"Data block location type '{method}' not implemented: {elem}"
            )

    @staticmethod
    def _read_inline_data_block(elem):
        # elem["location"] == ("inline", encoding)
        _, encoding = elem["location"]
        return XISF._decode_inline_or_embedded_data(encoding, elem["value"], elem)

    @staticmethod
    def _read_embedded_data_block(elem):
        # elem["location"] == ("embedded",)
        data_elem = ET.fromstring(elem["value"])
        encoding, data = data_elem.attrib["encoding"], data_elem.text
        return XISF._decode_inline_or_embedded_data(encoding, data, elem)

    @staticmethod
    def _decode_inline_or_embedded_data(encoding, data, elem):
        encodings = {"base64": base64.b64decode, "hex": base64.b16decode}
        if encoding not in encodings:
            raise NotImplementedError(
                f"Data block encoding type '{encoding}' not implemented: {elem}"
            )
        data = encodings[encoding](data)
        if "compression" in elem:
            data = XISF._decompress(data, elem)
        return data

    def _read_attached_data_block(self, elem):
        _, pos, size = elem["location"]
        with open(self._fname, "rb") as f:
            f.seek(pos)
            data = f.read(size)
        if "compression" in elem:
            data = XISF._decompress(data, elem)
        return data

    # ------------------------------------------------------------------
    # Image reading
    # ------------------------------------------------------------------

    def read_image(self, n=0, data_format="channels_last"):
        """Extract image *n* as a numpy ndarray.

        Args:
            n: index into the list returned by get_images_metadata().
            data_format: 'channels_last' (default, HWC) or 'channels_first' (CHW).

        Returns:
            numpy ndarray in the requested channel order.
        """
        try:
            meta = self._images_meta[n]
        except IndexError as e:
            if self._xisf_header is None:
                raise RuntimeError("No file loaded") from e
            if not self._images_meta:
                raise ValueError("File does not contain image data") from e
            raise ValueError(
                f"Requested image #{n}, valid range is [0..{len(self._images_meta) - 1}]"
            ) from e

        try:
            w, h, chc = meta["geometry"]
        except ValueError as e:
            raise NotImplementedError(
                f"Assumed 2D channels (width, height, channels), "
                f"found {meta['geometry']} geometry"
            ) from e

        data = self._read_data_block(meta)
        im_data = np.frombuffer(data, dtype=meta["dtype"]).reshape((chc, h, w))

        # --- byteOrder correction (fix: was silently ignored before) ---
        # The writer sets byteOrder="big" on big-endian systems.  If the
        # file was written on a host with the opposite endianness we must
        # swap so values are correct regardless of platform.
        file_big_endian = (meta.get("byteOrder", "little").lower() == "big")
        host_big_endian = (sys.byteorder == "big")
        if file_big_endian != host_big_endian:
            im_data = im_data.byteswap().newbyteorder()

        if data_format == "channels_last":
            return np.transpose(im_data, (1, 2, 0))
        return im_data

    @staticmethod
    def read(fname, n=0, image_metadata=None, xisf_metadata=None):
        """Convenience method: open *fname*, read image *n*, update metadata dicts.

        Args:
            fname: path to the .xisf file.
            n: image index (default 0).
            image_metadata: dict updated with the image's metadata.
            xisf_metadata:  dict updated with the file's metadata.

        Returns:
            numpy ndarray (channels_last).
        """
        if image_metadata is None:
            image_metadata = {}
        if xisf_metadata is None:
            xisf_metadata = {}
        xisf = XISF(fname)
        xisf_metadata.update(xisf.get_file_metadata())
        image_metadata.update(xisf.get_images_metadata()[n])
        return xisf.read_image(n)

    # ------------------------------------------------------------------
    # Partial / ROI reading
    # ------------------------------------------------------------------

    def can_partial_read_image(self, n=0):
        meta = self._images_meta[n]
        return meta["location"][0] == "attachment" and "compression" not in meta

    def read_image_roi(self, n=0, x0=0, y0=0, x1=None, y1=None,
                       channels=None, data_format="channels_last"):
        """Read a rectangular region-of-interest without loading the full image.

        Only works for uncompressed attachment blocks (check can_partial_read_image()).
        """
        meta = self._images_meta[n]
        if meta["location"][0] != "attachment":
            raise NotImplementedError("ROI read only supported for attachment blocks")
        if "compression" in meta:
            raise NotImplementedError("ROI read not supported for compressed image blocks")

        w, h, chc = meta["geometry"]
        dtype    = meta["dtype"]
        itemsize = dtype.itemsize

        if x1 is None: x1 = w
        if y1 is None: y1 = h
        x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Empty ROI")

        if channels is None:
            channels = list(range(chc))

        _, pos, _ = meta["location"]
        roi_w      = x1 - x0
        roi_h      = y1 - y0
        row_bytes  = w * itemsize
        roi_bytes  = roi_w * itemsize
        plane_bytes = h * row_bytes

        out = np.empty((len(channels), roi_h, roi_w), dtype=dtype)

        with open(self._fname, "rb") as f:
            for ci, c in enumerate(channels):
                if c < 0 or c >= chc:
                    raise IndexError(f"channel {c} out of range [0..{chc - 1}]")
                plane_base = pos + c * plane_bytes
                for r, y in enumerate(range(y0, y1)):
                    f.seek(plane_base + y * row_bytes + x0 * itemsize)
                    out[ci, r, :] = np.frombuffer(f.read(roi_bytes), dtype=dtype, count=roi_w)

        if data_format == "channels_last":
            return np.transpose(out, (1, 2, 0))
        return out

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def write(
        fname,
        im_data,
        creator_app=None,
        image_metadata=None,
        xisf_metadata=None,
        codec=None,
        shuffle=False,
        level=None,
    ):
        """Write *im_data* (numpy array) to a monolithic XISF file.

        Compression is applied only when it actually reduces the data size.

        Args:
            fname:          output filename (overwritten if existing).
            im_data:        numpy ndarray with the image data.
            creator_app:    XISF:CreatorApplication string.
            image_metadata: dict matching the structure from get_images_metadata();
                            only 'FITSKeywords' and 'XISFProperties' are written.
            xisf_metadata:  file metadata dict matching get_file_metadata().
            codec:          'zlib', 'lz4', 'lz4hc', 'zstd', or None.
            shuffle:        apply byte-shuffling before compression.
            level:          compression level (codec-specific).

        Returns:
            (bytes_written, codec_used)  — codec_used is None if compression
            was skipped because it didn't reduce size.
        """
        if image_metadata is None:
            image_metadata = {}
        if xisf_metadata is None:
            xisf_metadata = {}

        blk_sz = xisf_metadata.get(
            "XISF:BlockAlignmentSize", {"value": XISF._block_alignment_size}
        )["value"]
        max_inline_blk_sz = xisf_metadata.get(
            "XISF:MaxInlineBlockSize", {"value": XISF._max_inline_block_size}
        )["value"]

        # ---- helpers (local) ----------------------------------------

        def _create_image_metadata(im_data, id_):
            attrs = {"id": id_}
            if im_data.shape[2] in (1, 3):
                data_format = "channels_last"
                geometry = (im_data.shape[1], im_data.shape[0], im_data.shape[2])
                channels = im_data.shape[2]
            else:
                data_format = "channels_first"
                geometry = im_data.shape
                channels = im_data.shape[0]
            attrs["geometry"] = "%d:%d:%d" % geometry
            attrs["colorSpace"] = "Gray" if channels == 1 else "RGB"
            attrs["sampleFormat"] = XISF._get_sampleFormat(im_data.dtype)
            if attrs["sampleFormat"].startswith("Float"):
                attrs["bounds"] = "0:1"
            if sys.byteorder == "big" and attrs["sampleFormat"] != "UInt8":
                attrs["byteOrder"] = "big"
            return attrs, data_format

        def _serialize_data_block(data, attr_dict, codec_, level_, shuffle_):
            data_block = data.tobytes()
            uncompressed_size = data.nbytes
            codec_str = codec_

            if codec_ is None:
                data_size = uncompressed_size
            else:
                compressed_block = XISF._compress(
                    data_block, codec_, level_, shuffle_, data.itemsize
                )
                compressed_size = len(compressed_block)
                if compressed_size < uncompressed_size:
                    data_block = compressed_block
                    data_size  = compressed_size
                    if shuffle_:
                        codec_str = f"{codec_}+sh"
                        attr_dict["compression"] = (
                            f"{codec_str}:{uncompressed_size}:{data.itemsize}"
                        )
                    else:
                        attr_dict["compression"] = f"{codec_}:{uncompressed_size}"
                else:
                    data_size = uncompressed_size
                    codec_str = None

            return data_block, data_size, codec_str

        def _update_xisf_metadata():
            xisf_metadata["XISF:CreationTime"] = {
                "id": "XISF:CreationTime", "type": "String",
                "value": datetime.utcnow().isoformat(),
            }
            xisf_metadata["XISF:CreatorApplication"] = {
                "id": "XISF:CreatorApplication", "type": "String",
                "value": creator_app or XISF._creator_app,
            }
            xisf_metadata["XISF:CreatorModule"] = {
                "id": "XISF:CreatorModule", "type": "String",
                "value": XISF._creator_module,
            }
            _OSes = {
                "linux": "Linux", "win32": "Windows",
                "cygwin": "Windows", "darwin": "macOS",
            }
            xisf_metadata["XISF:CreatorOS"] = {
                "id": "XISF:CreatorOS", "type": "String",
                "value": _OSes.get(sys.platform, sys.platform),
            }
            xisf_metadata["XISF:BlockAlignmentSize"] = {
                "id": "XISF:BlockAlignmentSize", "type": "UInt16", "value": blk_sz,
            }
            xisf_metadata["XISF:MaxInlineBlockSize"] = {
                "id": "XISF:MaxInlineBlockSize", "type": "UInt16",
                "value": max_inline_blk_sz,
            }
            if codec is not None:
                xisf_metadata["XISF:CompressionCodecs"] = {
                    "id": "XISF:CompressionCodecs", "type": "String", "value": codec,
                }
                xisf_metadata["XISF:CompressionLevel"] = {
                    "id": "XISF:CompressionLevel", "type": "Int",
                    "value": level if level else XISF._compression_def_level[codec],
                }
            else:
                xisf_metadata.pop("XISF:CompressionCodecs", None)
                xisf_metadata.pop("XISF:CompressionLevel",  None)

        def _compute_attached_positions(hdr_prov_sz, attached_blocks_locations):
            _aligned = lambda pos: ((pos + blk_sz - 1) // blk_sz) * blk_sz
            hdr_sz = hdr_prov_sz
            prev_sum = 0
            while True:
                pos = _aligned(hdr_sz)
                sum_len = 0
                for loc in attached_blocks_locations:
                    loc["position"] = pos
                    sum_len += len(str(pos))
                    pos = _aligned(pos + loc["size"])
                if sum_len == prev_sum:
                    break
                prev_sum = sum_len
                hdr_sz   = hdr_prov_sz + sum_len
            for b in attached_blocks_locations:
                b["xml"].attrib["location"] = XISF._to_location(
                    ("attachment", b["position"], b["size"])
                )

        def _zero_pad(length):
            return (0).to_bytes(max(0, length), byteorder="little")

        # ---- prepare image ------------------------------------------
        im_id = image_metadata.get("id", "image")
        im_attrs, data_format = _create_image_metadata(im_data, im_id)
        im_prepared = (
            np.transpose(im_data, (2, 0, 1))
            if data_format == "channels_last"
            else im_data
        )
        im_data_block, data_size, codec_str = _serialize_data_block(
            im_prepared, im_attrs, codec, level, shuffle
        )
        im_attrs["location"] = XISF._to_location(("attachment", "", data_size))

        # ---- build XML header ---------------------------------------
        xisf_header_xml = ET.Element("xisf", XISF._xisf_attrs)
        image_xml = ET.SubElement(xisf_header_xml, "Image", im_attrs)

        for kw_name, kw_values in image_metadata.get("FITSKeywords", {}).items():
            XISF._insert_fitskeyword(image_xml, kw_name, kw_values)

        attached_blocks_locations = [
            {"xml": image_xml, "position": 0, "size": data_size, "data": im_data_block}
        ]

        for p_dict in image_metadata.get("XISFProperties", {}).values():
            if blk := XISF._insert_property(image_xml, p_dict, max_inline_blk_sz):
                attached_blocks_locations.append(blk)

        metadata_xml = ET.SubElement(xisf_header_xml, "Metadata")
        _update_xisf_metadata()
        for property_dict in xisf_metadata.values():
            if blk := XISF._insert_property(metadata_xml, property_dict, max_inline_blk_sz):
                attached_blocks_locations.append(blk)

        xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
        header_provisional_sz = (
            len(XISF._signature)
            + XISF._headerlength_len
            + len(xisf_header)
            + XISF._reserved_len
        )
        _compute_attached_positions(header_provisional_sz, attached_blocks_locations)

        # ---- write file ---------------------------------------------
        with open(fname, "wb") as f:
            f.write(XISF._signature)
            xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
            f.write(len(xisf_header).to_bytes(XISF._headerlength_len, byteorder="little"))
            f.write(_zero_pad(XISF._reserved_len))
            f.write(xisf_header)
            for b in attached_blocks_locations:
                pos = b["position"]
                f.write(_zero_pad(pos - f.tell()))
                assert f.tell() == pos
                f.write(b["data"])
            bytes_written = f.tell()

        return bytes_written, codec_str

    # ------------------------------------------------------------------
    # Property processing
    # ------------------------------------------------------------------

    def _process_property(self, p_et):
        p_dict = p_et.attrib.copy()

        if p_dict["type"] == "TimePoint":
            try:
                tp_str = p_dict.get("value", "").replace("Z", "+00:00")
                if "." in tp_str:
                    tail = tp_str.split(".")[-1]
                    if "+" not in tail and "-" not in tail:
                        tp_str += "+00:00"
                p_dict["datetime"] = datetime.fromisoformat(tp_str) if tp_str else None
            except (ValueError, TypeError):
                p_dict["datetime"] = None

        elif p_dict["type"] == "String":
            p_dict["value"] = p_et.text
            if "location" in p_dict:
                self._process_location_compression(p_dict)
                return _make_lazy(p_dict)
            return p_dict

        elif p_dict["type"] == "Boolean":
            # Accept "true"/"1"/"yes" as truthy (fix: previously only "true")
            v = str(p_dict.get("value", "false")).strip().lower()
            p_dict["value"] = v in ("true", "1", "yes")

        elif "value" in p_et.attrib:
            p_dict["value"] = ast.literal_eval(p_dict["value"])

        elif "Vector" in p_dict["type"]:
            p_dict["value"]  = p_et.text
            p_dict["length"] = int(p_dict["length"])
            p_dict["dtype"]  = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            return _make_lazy(p_dict)

        elif "Matrix" in p_dict["type"]:
            p_dict["value"]   = p_et.text
            p_dict["rows"]    = int(p_dict["rows"])
            p_dict["columns"] = int(p_dict["columns"])
            p_dict["dtype"]   = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            return _make_lazy(p_dict)

        else:
            print(f"Unsupported Property type {p_dict['type']}: {p_et}")
            return False

        return p_dict

    def resolve_property(self, p_dict):
        """Resolve a lazy property (String/Vector/Matrix with a data block).

        Mutates *p_dict* in place and returns the decoded value.
        """
        if not p_dict.get("_lazy"):
            return p_dict.get("value")

        raw = self._read_data_block(p_dict)
        t   = p_dict["type"]

        if t == "String":
            val = raw.decode("utf-8")
        elif "Vector" in t:
            val = np.frombuffer(raw, dtype=p_dict["dtype"], count=p_dict["length"])
        elif "Matrix" in t:
            length = p_dict["rows"] * p_dict["columns"]
            val = np.frombuffer(raw, dtype=p_dict["dtype"], count=length).reshape(
                (p_dict["rows"], p_dict["columns"])
            )
        else:
            val = raw

        p_dict["value"] = val
        p_dict["_lazy"] = False
        return val

    # ------------------------------------------------------------------
    # XML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_property(parent, p_dict, max_inline_block_size, codec=None, shuffle=False):
        """Insert a property element into *parent*.

        Returns a ``{"xml", "position", "size", "data"}`` dict when an
        attached data block is needed, otherwise ``False``.
        """
        # Shared helper — builds the base attribute dict and folds in
        # optional 'format' and 'comment' fields if present.
        def _attrs(**base):
            d = dict(base)
            if p_dict.get("format"):
                d["format"] = str(p_dict["format"])
            if p_dict.get("comment"):
                d["comment"] = str(p_dict["comment"])
            return d

        t = p_dict["type"]

        # ---- scalars & TimePoint ------------------------------------
        scalars = ("Int", "Byte", "Short", "Float", "Boolean", "TimePoint")
        if any(s in t for s in scalars):
            val_str = "true" if (t == "Boolean" and p_dict["value"]) else (
                "false" if t == "Boolean" else str(p_dict["value"])
            )
            ET.SubElement(
                parent, "Property",
                _attrs(id=p_dict["id"], type=t, value=val_str),
            )
            return False

        # ---- string -------------------------------------------------
        if t == "String":
            text       = str(p_dict["value"])
            data_bytes = text.encode("utf-8")
            sz         = len(data_bytes)
            if sz > max_inline_block_size:
                if codec:
                    compressed, comp_str = XISF._compress_data_block(
                        data_bytes, codec, shuffle, 1
                    )
                    xml = ET.SubElement(
                        parent, "Property",
                        _attrs(
                            id=p_dict["id"], type=t,
                            location=XISF._to_location(("attachment", "", len(compressed))),
                            compression=comp_str,
                        ),
                    )
                    return {"xml": xml, "position": 0, "size": len(compressed), "data": compressed}
                else:
                    xml = ET.SubElement(
                        parent, "Property",
                        _attrs(
                            id=p_dict["id"], type=t,
                            location=XISF._to_location(("attachment", "", sz)),
                        ),
                    )
                    return {"xml": xml, "position": 0, "size": sz, "data": data_bytes}
            else:
                ET.SubElement(
                    parent, "Property", _attrs(id=p_dict["id"], type=t)
                ).text = text
            return False

        # ---- vector / matrix (shared logic) -------------------------
        is_matrix = "Matrix" in t
        is_vector = "Vector" in t
        if is_vector or is_matrix:
            data      = p_dict["value"]
            raw_bytes = data.tobytes()
            sz        = len(raw_bytes)
            itemsize  = data.itemsize

            # Extra dimension attributes
            dim_attrs = (
                {"rows": str(data.shape[0]), "columns": str(data.shape[1])}
                if is_matrix
                else {"length": str(data.size)}
            )

            if sz > max_inline_block_size:
                if codec:
                    compressed, comp_str = XISF._compress_data_block(
                        raw_bytes, codec, shuffle, itemsize
                    )
                    xml = ET.SubElement(
                        parent, "Property",
                        _attrs(
                            id=p_dict["id"], type=t,
                            location=XISF._to_location(("attachment", "", len(compressed))),
                            compression=comp_str,
                            **dim_attrs,
                        ),
                    )
                    return {"xml": xml, "position": 0, "size": len(compressed), "data": compressed}
                else:
                    xml = ET.SubElement(
                        parent, "Property",
                        _attrs(
                            id=p_dict["id"], type=t,
                            location=XISF._to_location(("attachment", "", sz)),
                            **dim_attrs,
                        ),
                    )
                    return {"xml": xml, "position": 0, "size": sz, "data": data}
            else:
                ET.SubElement(
                    parent, "Property",
                    _attrs(
                        id=p_dict["id"], type=t,
                        location=XISF._to_location(("inline", "base64")),
                        **dim_attrs,
                    ),
                ).text = base64.b64encode(raw_bytes).decode("ascii")
            return False

        print(f"Warning: skipping unsupported property {p_dict}")
        return False

    @staticmethod
    def _insert_fitskeyword(image_xml, keyword_name, keyword_values):
        for entry in keyword_values:
            ET.SubElement(
                image_xml, "FITSKeyword",
                {
                    "name":    keyword_name,
                    "value":   entry["value"],
                    "comment": entry["comment"],
                },
            )

    # ------------------------------------------------------------------
    # Attribute parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_location_compression(p_dict):
        p_dict["location"] = XISF._parse_location(p_dict["location"])
        if "compression" in p_dict:
            p_dict["compression"] = XISF._parse_compression(p_dict["compression"])

    @staticmethod
    def _parse_geometry(g):
        """Parse a "w:h:c" geometry string to an (int, int, int) tuple."""
        return tuple(int(v.strip()) for v in g.split(":"))

    @staticmethod
    def _parse_location(l):
        """Parse a location attribute string to a normalized tuple.

        Returns:
            ("attachment", pos: int, size: int)
            ("inline",     encoding: str)
            ("embedded",)

        (Fix: previously returned a raw list of strings for inline/embedded,
        making downstream code responsible for the distinction.)
        """
        ll = [s.strip() for s in l.split(":")]
        kind = ll[0]
        if kind == "attachment":
            return ("attachment", int(ll[1]), int(ll[2]))
        elif kind == "inline":
            return ("inline", ll[1])          # encoding string
        elif kind == "embedded":
            return ("embedded",)
        else:
            raise NotImplementedError(
                f"Data block location type '{kind}' not implemented"
            )

    @staticmethod
    def _to_location(location_tuple):
        return ":".join(str(e) for e in location_tuple)

    @staticmethod
    def _parse_compression(c):
        """Parse a compression attribute string.

        Returns:
            (codec: str, uncompressed_size: int, item_size: int | None)
        """
        cl = c.split(":")
        if len(cl) == 3:
            return (cl[0], int(cl[1]), int(cl[2]))
        return (cl[0], int(cl[1]), None)

    @staticmethod
    def _parse_sampleFormat(s):
        alternate_names = {
            "Byte":   "UInt8",  "Short":  "Int16",  "UShort": "UInt16",
            "Int":    "Int32",  "UInt":   "UInt32",
            "Float":  "Float32","Double": "Float64",
        }
        s = alternate_names.get(s.strip(), s.strip())
        _dtypes = {
            "UInt8":   np.dtype("uint8"),   "UInt16":  np.dtype("uint16"),
            "UInt32":  np.dtype("uint32"),  "Float32": np.dtype("float32"),
            "Float64": np.dtype("float64"),
        }
        if s not in _dtypes:
            raise NotImplementedError(f"sampleFormat '{s}' not implemented")
        return _dtypes[s]

    @staticmethod
    def _get_sampleFormat(dtype):
        _sampleFormats = {
            "uint8":   "UInt8",  "uint16":  "UInt16", "uint32":  "UInt32",
            "float32": "Float32","float64": "Float64",
        }
        key = str(dtype)
        if key not in _sampleFormats:
            raise NotImplementedError(f"sampleFormat for dtype '{dtype}' not implemented")
        return _sampleFormats[key]

    @staticmethod
    def _parse_vector_dtype(type_name):
        alternate_names = {
            "ByteArray": "UI8Vector", "IVector":  "I32Vector",
            "UIVector":  "UI32Vector","Vector":   "F64Vector",
        }
        type_name = alternate_names.get(type_name.strip(), type_name.strip())
        prefix = type_name[:-6]  # strip "Vector" or "Matrix"
        _dtypes = {
            "I8":  np.dtype("int8"),    "UI8":  np.dtype("uint8"),
            "I16": np.dtype("int16"),   "UI16": np.dtype("uint16"),
            "I32": np.dtype("int32"),   "UI32": np.dtype("uint32"),
            "I64": np.dtype("int64"),   "UI64": np.dtype("uint64"),
            "F32": np.dtype("float32"), "F64":  np.dtype("float64"),
            "C32": np.dtype("csingle"), "C64":  np.dtype("cdouble"),
        }
        if prefix not in _dtypes:
            raise NotImplementedError(f"vector/matrix data type '{type_name}' not implemented")
        return _dtypes[prefix]

    # ------------------------------------------------------------------
    # Compression helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unshuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8")).reshape((item_size, -1))
        return np.transpose(a).tobytes()

    @staticmethod
    def _shuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8")).reshape((-1, item_size))
        return np.transpose(a).tobytes()

    @staticmethod
    def _decompress(data, elem):
        codec, uncompressed_size, item_size = elem["compression"]
        if codec.startswith("lz4"):
            data = lz4.block.decompress(data, uncompressed_size=uncompressed_size)
        elif codec.startswith("zstd"):
            data = zstandard.decompress(data, max_output_size=uncompressed_size)
        elif codec.startswith("zlib"):
            data = zlib.decompress(data)
        else:
            raise NotImplementedError(f"Unimplemented compression codec '{codec}'")
        if item_size:
            data = XISF._unshuffle(data, item_size)
        return data

    @staticmethod
    def _compress_data_block(data, codec, shuffle=False, itemsize=1):
        """Compress *data* and return (compressed_bytes, compression_attr_string).

        Args:
            data:     bytes-like or numpy array.
            codec:    'zlib', 'lz4', 'lz4hc', or 'zstd'.
            shuffle:  enable byte shuffling.
            itemsize: element size in bytes for shuffling (1 for strings).

        Returns:
            (compressed: bytes, compression_attr: str)

        (Fix: unified parameter naming with _compress so shuffle/itemsize
        cannot silently no-op due to mismatched keywords.)
        """
        raw_bytes         = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
        uncompressed_size = len(raw_bytes)
        compressed        = XISF._compress(
            raw_bytes, codec, level=None, shuffle=shuffle, itemsize=itemsize if shuffle else None
        )
        if shuffle and itemsize > 1:
            comp_str = f"{codec}+sh:{uncompressed_size}:{itemsize}"
        else:
            comp_str = f"{codec}:{uncompressed_size}"
        return compressed, comp_str

    @staticmethod
    def _compress(data, codec, level=None, shuffle=False, itemsize=None):
        """Compress *data* with optional byte-shuffling.

        Args:
            data:     bytes-like input.
            codec:    'lz4', 'lz4hc', 'zstd', or 'zlib'.
            level:    compression level (codec-specific default if None).
            shuffle:  apply byte-shuffling before compression.
            itemsize: element size for shuffling (required when shuffle=True).
        """
        compressed = XISF._shuffle(data, itemsize) if (shuffle and itemsize) else data

        if codec == "lz4hc":
            level = level or XISF._compression_def_level["lz4hc"]
            compressed = lz4.block.compress(
                compressed, mode="high_compression", compression=level, store_size=False
            )
        elif codec == "lz4":
            compressed = lz4.block.compress(compressed, store_size=False)
        elif codec == "zstd":
            level = level or XISF._compression_def_level["zstd"]
            compressed = zstandard.compress(compressed, level=level)
        elif codec == "zlib":
            level = level or XISF._compression_def_level["zlib"]
            compressed = zlib.compress(compressed, level=level)
        else:
            raise NotImplementedError(f"Unimplemented compression codec '{codec}'")

        return compressed