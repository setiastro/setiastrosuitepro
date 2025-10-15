#legacy.xisf.py
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
"""

from importlib.metadata import version


import platform
import xml.etree.ElementTree as ET
import numpy as np
import lz4.block  # https://python-lz4.readthedocs.io/en/stable/lz4.block.html
import zlib  # https://docs.python.org/3/library/zlib.html
import zstandard  # https://python-zstandard.readthedocs.io/en/stable/
import base64
import sys
from datetime import datetime
import ast

__version__ = "1.0.0"

class XISF:
    """Implements an baseline XISF Decoder and a simple baseline Encoder.
    It parses metadata from Image and Metadata XISF core elements. Image data is returned as a numpy ndarray
    (using the "channels-last" convention by default).

    What's supported:
    - Monolithic XISF files only
        - XISF data blocks with attachment, inline or embedded block locations
        - Planar pixel storage models, *however it assumes 2D images only* (with multiple channels)
        - UInt8/16/32 and Float32/64 pixel sample formats
        - Grayscale and RGB color spaces
    - Decoding:
        - multiple Image core elements from a monolithic XISF file
        - Support all standard compression codecs defined in this specification for decompression
          (zlib/lz4[hc]/zstd + byte shuffling)
    - Encoding:
        - Single image core element with an attached data block
        - Support all standard compression codecs defined in this specification for decompression
          (zlib/lz4[hc]/zstd + byte shuffling)
    - "Atomic" properties (scalar types, String, TimePoint), Vector and Matrix (e.g. astrometric
      solutions)
    - Metadata and FITSKeyword core elements

    What's not supported (at least by now):
    - Read pixel data in the normal pixel storage models
    - Read pixel data in the planar pixel storage models other than 2D images
    - Complex and Table properties
    - Any other not explicitly supported core elements (Resolution, Thumbnail, ICCProfile, etc.)

    Usage example:
    ```
    from xisf import XISF
    import matplotlib.pyplot as plt
    xisf = XISF("file.xisf")
    file_meta = xisf.get_file_metadata()
    file_meta
    ims_meta = xisf.get_images_metadata()
    ims_meta
    im_data = xisf.read_image(0)
    plt.imshow(im_data)
    plt.show()
    XISF.write(
        "output.xisf", im_data,
        creator_app="My script v1.0", image_metadata=ims_meta[0], xisf_metadata=file_meta,
        codec='lz4hc', shuffle=True
    )
    ```

    If the file is not huge and it contains only an image (or you're interested just in one of the
    images inside the file), there is a convenience method for reading the data and the metadata:
    ```
    from xisf import XISF
    import matplotlib.pyplot as plt
    im_data = XISF.read("file.xisf")
    plt.imshow(im_data)
    plt.show()
    ```

    The XISF format specification is available at https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
    """

    # Static attributes
    _creator_app = f"Python {platform.python_version()}"
    _creator_module = f"XISF Python Module v{__version__} github.com/sergio-dr/xisf"
    _signature = b"XISF0100"  # Monolithic
    _headerlength_len = 4
    _reserved_len = 4
    _xml_ns = {"xisf": "http://www.pixinsight.com/xisf"}
    _xisf_attrs = {
        "xmlns": "http://www.pixinsight.com/xisf",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "version": "1.0",
        "xsi:schemaLocation": "http://www.pixinsight.com/xisf http://pixinsight.com/xisf/xisf-1.0.xsd",
    }
    _compression_def_level = {
        "zlib": 6,  # 1..9, default: 6 as indicated in https://docs.python.org/3/library/zlib.html
        "lz4": 0,  # no other values, as indicated in https://python-lz4.readthedocs.io/en/stable/lz4.block.html
        "lz4hc": 9,  # 1..12, (4-9 recommended), default: 9 as indicated in https://python-lz4.readthedocs.io/en/stable/lz4.block.html
        "zstd": 3,  # 1..22, (3-9 recommended), default: 3 as indicated in https://facebook.github.io/zstd/zstd_manual.html
    }
    _block_alignment_size = 4096
    _max_inline_block_size = 3072

    def __init__(self, fname):
        """Opens a XISF file and extract its metadata. To get the metadata and the images, see get_file_metadata(),
        get_images_metadata() and read_image().
        Args:
            fname: filename

        Returns:
            XISF object.
        """
        self._fname = fname
        self._headerlength = None
        self._xisf_header = None
        self._xisf_header_xml = None
        self._images_meta = None
        self._file_meta = None
        ET.register_namespace("", self._xml_ns["xisf"])

        self._read()

    def _read(self):
        with open(self._fname, "rb") as f:
            # Check XISF signature
            signature = f.read(len(self._signature))
            if signature != self._signature:
                raise ValueError("File doesn't have XISF signature")

            # Get header length
            self._headerlength = int.from_bytes(f.read(self._headerlength_len), byteorder="little")
            # Equivalent:
            # self._headerlength = np.fromfile(f, dtype=np.uint32, count=1)[0]

            # Skip reserved field
            _ = f.read(self._reserved_len)

            # Get XISF (XML) Header
            self._xisf_header = f.read(self._headerlength)
            self._xisf_header_xml = ET.fromstring(self._xisf_header)
        self._analyze_header()

    def _analyze_header(self):
        # Analyze header to get Data Blocks position and length
        self._images_meta = []
        for image in self._xisf_header_xml.findall("xisf:Image", self._xml_ns):
            image_basic_meta = image.attrib

            # Parse and replace geometry and location with tuples,
            # parses and translates sampleFormat to numpy dtypes,
            # and extend with metadata from children entities (FITSKeywords, XISFProperties)

            #   The same FITS keyword can appear multiple times, so we have to
            #   prepare a dict of lists. Each element in the list is a dict
            #   that hold the value and the comment associated with the keyword.
            #   Not as clear as I would like.
            fits_keywords = {}
            for a in image.findall("xisf:FITSKeyword", self._xml_ns):
                fits_keywords.setdefault(a.attrib["name"], []).append(
                    {
                        "value": a.attrib["value"].strip("'").strip(" "),
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
            # Also parses compression attribute if present, converting it to a tuple
            if "compression" in image.attrib:
                image_extended_meta["compression"] = self._parse_compression(
                    image.attrib["compression"]
                )

            # Merge basic and extended metadata in a dict
            image_meta = {**image_basic_meta, **image_extended_meta}

            # Append the image metadata to the list
            self._images_meta.append(image_meta)

        # Analyze header for file metadata
        self._file_meta = {}
        for p in self._xisf_header_xml.find("xisf:Metadata", self._xml_ns):
            self._file_meta[p.attrib["id"]] = self._process_property(p)

        # TODO: rest of XISF core elements: Resolution, ICCProfile, Thumbnail, ...

    def get_images_metadata(self):
        """Provides the metadata of all image blocks contained in the XISF File, extracted from
        the header (<Image> core elements). To get the actual image data, see read_image().

        It outputs a dictionary m_i for each image, with the following structure:
        ```
        m_i = {
            'geometry': (width, height, channels), # only 2D images (with multiple channels) are supported
            'location': (pos, size), # used internally in read_image()
            'dtype': np.dtype('...'), # derived from sampleFormat argument
            'compression': (codec, uncompressed_size, item_size), # optional
            'key': 'value', # other <Image> attributes are simply copied
            ...,
            'FITSKeywords': { <fits_keyword>: fits_keyword_values_list, ... },
            'XISFProperties': { <xisf_property_name>: property_dict, ... }
        }

        where:

        fits_keyword_values_list = [ {'value': <value>, 'comment': <comment> }, ...]
        property_dict = {'id': <xisf_property_name>, 'type': <xisf_type>, 'value': property_value, ...}
        ```

        Returns:
            list [ m_0, m_1, ..., m_{n-1} ] where m_i is a dict as described above.

        """
        return self._images_meta

    def get_file_metadata(self):
        """Provides the metadata from the header of the XISF File (<Metadata> core elements).

        Returns:
            dictionary with one entry per property: { <xisf_property_name>: property_dict, ... }
            where:
            ```
            property_dict = {'id': <xisf_property_name>, 'type': <xisf_type>, 'value': property_value, ...}
            ```

        """
        return self._file_meta

    def get_metadata_xml(self):
        """Returns the complete XML header as a xml.etree.ElementTree.Element object.

        Returns:
            xml.etree.ElementTree.Element: complete XML XISF header
        """
        return self._xisf_header_xml

    def _read_data_block(self, elem):
        method = elem["location"][0]
        if method == "inline":
            return self._read_inline_data_block(elem)
        elif method == "embedded":
            return self._read_embedded_data_block(elem)
        elif method == "attachment":
            return self._read_attached_data_block(elem)
        else:
            raise NotImplementedError(f"Data block location type '{method}' not implemented: {elem}")

    @staticmethod
    def _read_inline_data_block(elem):
        method, encoding = elem["location"]
        assert method == "inline"
        return XISF._decode_inline_or_embedded_data(encoding, elem["value"], elem)

    @staticmethod
    def _read_embedded_data_block(elem):
        assert elem["location"][0] == "embedded"
        data_elem = ET.fromstring(elem["value"])
        encoding, data = data.attrib["encoding"], data_elem.text
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
        # Position and size of the Data Block containing the image data
        method, pos, size = elem["location"]

        assert method == "attachment"

        with open(self._fname, "rb") as f:
            f.seek(pos)
            data = f.read(size)

        if "compression" in elem:
            data = XISF._decompress(data, elem)

        return data

    def read_image(self, n=0, data_format="channels_last"):
        """Extracts an image from a XISF object.

        Args:
            n: index of the image to extract in the list returned by get_images_metadata()
            data_format: channels axis can be 'channels_first' or 'channels_last' (as used in
            keras/tensorflow, pyplot's imshow, etc.), 0 by default.

        Returns:
            Numpy ndarray with the image data, in the requested format (channels_first or channels_last).

        """
        try:
            meta = self._images_meta[n]
        except IndexError as e:
            if self._xisf_header is None:
                raise RuntimeError("No file loaded") from e
            elif not self._images_meta:
                raise ValueError("File does not contain image data") from e
            else:
                raise ValueError(
                    f"Requested image #{n}, valid range is [0..{len(self._images_meta) - 1}]"
                ) from e

        try:
            # Assumes *two*-dimensional images (chc=channel count)
            w, h, chc = meta["geometry"]
        except ValueError as e:
            raise NotImplementedError(
                f"Assumed 2D channels (width, height, channels), found {meta['geometry']} geometry"
            )

        data = self._read_data_block(meta)
        im_data = np.frombuffer(data, dtype=meta["dtype"])
        im_data = im_data.reshape((chc, h, w))
        return np.transpose(im_data, (1, 2, 0)) if data_format == "channels_last" else im_data

    @staticmethod
    def read(fname, n=0, image_metadata={}, xisf_metadata={}):
        """Convenience method for reading a file containing a single image.

        Args:
            fname (string): filename
            n (int, optional): index of the image to extract (in the list returned by get_images_metadata()). Defaults to 0.
            image_metadata (dict, optional): dictionary that will be updated with the metadata of the image.
            xisf_metadata (dict, optional): dictionary that will be updated with the metadata of the file.

        Returns:
            [np.ndarray]: Numpy ndarray with the image data, in the requested format (channels_first or channels_last).
        """
        xisf = XISF(fname)
        xisf_metadata.update(xisf.get_file_metadata())
        image_metadata.update(xisf.get_images_metadata()[n])
        return xisf.read_image(n)

    # if 'colorSpace' is not specified, im_data.shape[2] dictates if colorSpace is 'Gray' or 'RGB'
    # For float sample formats, bounds="0:1" is assumed
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
        """Writes an image (numpy array) to a XISF file. Compression may be requested but it only
        will be used if it actually reduces the data size.

        Args:
            fname: filename (will overwrite if existing)
            im_data: numpy ndarray with the image data
            creator_app: string for XISF:CreatorApplication file property (defaults to python version in None provided)
            image_metadata: dict with the same structure described for m_i in get_images_metadata().
              Only 'FITSKeywords' and 'XISFProperties' keys are actually written, the rest are derived from im_data.
            xisf_metadata: file metadata, dict with the same structure returned by get_file_metadata()
            codec: compression codec ('zlib', 'lz4', 'lz4hc' or 'zstd'), or None to disable compression
            shuffle: whether to apply byte-shuffling before compression (ignored if codec is None). Recommended
              for 'lz4' ,'lz4hc' and 'zstd' compression algorithms.
            level: for zlib, 1..9 (default: 6); for lz4hc, 1..12 (default: 9); for zstd, 1..22 (default: 3).
              Higher means more compression.
        Returns:
            bytes_written: the total number of bytes written into the output file.
            codec: The codec actually used, i.e., None if compression did not reduce the data block size so
            compression was not finally used.

        """
        if image_metadata is None:
            image_metadata = {}

        if xisf_metadata is None:
            xisf_metadata = {}

        # Data block alignment
        blk_sz = xisf_metadata.get("XISF:BlockAlignmentSize", {"value": XISF._block_alignment_size})[
            "value"
        ]
        # Maximum inline block size (larger will be attached instead)
        max_inline_blk_sz = xisf_metadata.get(
            "XISF:MaxInlineBlockSize", {"value": XISF._max_inline_block_size}
        )["value"]

        # Prepare basic image metadata
        def _create_image_metadata(im_data, id):
            image_attrs = {"id": id}
            if im_data.shape[2] == 3 or im_data.shape[2] == 1:
                data_format = "channels_last"
                geometry = (im_data.shape[1], im_data.shape[0], im_data.shape[2])
                channels = im_data.shape[2]
            else:
                data_format = "channels_first"
                geometry = im_data.shape
                channels = im_data.shape[0]
            image_attrs["geometry"] = "%d:%d:%d" % geometry
            image_attrs["colorSpace"] = "Gray" if channels == 1 else "RGB"
            image_attrs["sampleFormat"] = XISF._get_sampleFormat(im_data.dtype)
            if image_attrs["sampleFormat"].startswith("Float"):
                image_attrs["bounds"] = "0:1"  # Assumed
            if sys.byteorder == "big" and image_attrs["sampleFormat"] != "UInt8":
                image_attrs["byteOrder"] = "big"
            return image_attrs, data_format

        # Rearrange ndarray for data_format and serialize to bytes
        def _prepare_image_data_block(im_data, data_format):
            return np.transpose(im_data, (2, 0, 1)) if data_format == "channels_last" else im_data

        # Serialize a data block, with optional compression (i.e., when codec is not None)
        # Compression will be only applied if effectively reduces size
        def _serialize_data_block(data, attr_dict, codec, level, shuffle):
            data_block = data.tobytes()
            uncompressed_size = data.nbytes
            codec_str = codec

            if codec is None:
                data_size = uncompressed_size
            else:
                compressed_block = XISF._compress(data_block, codec, level, shuffle, data.itemsize)
                compressed_size = len(compressed_block)

                if compressed_size < uncompressed_size:
                    # The ideal situation, compressing actually reduces size
                    data_block, data_size = compressed_block, compressed_size

                    # Add 'compression' image attribute: (codec:uncompressed-size[:item-size])
                    if shuffle:
                        codec_str += "+sh"
                        attr_dict["compression"] = f"{codec_str}:{uncompressed_size}:{data.itemsize}"
                    else:
                        attr_dict["compression"] = f"{codec}:{uncompressed_size}"
                else:
                    # If there's no gain in compressing, just discard the compressed block
                    # See https://pixinsight.com/forum.old/index.php?topic=10942.msg68043#msg68043
                    # (In fact, PixInsight will show garbage image data if the data block is
                    # compressed but the uncompressed size is smaller)
                    data_size = uncompressed_size
                    codec_str = None

            return data_block, data_size, codec_str

        # Overwrites/creates XISF metadata
        def _update_xisf_metadata(creator_app, blk_sz, max_inline_blk_sz, codec, level):
            # Create file metadata
            xisf_metadata["XISF:CreationTime"] = {
                "id": "XISF:CreationTime",
                "type": "String",
                "value": datetime.utcnow().isoformat(),
            }
            xisf_metadata["XISF:CreatorApplication"] = {
                "id": "XISF:CreatorApplication",
                "type": "String",
                "value": creator_app if creator_app else XISF._creator_app,
            }
            xisf_metadata["XISF:CreatorModule"] = {
                "id": "XISF:CreatorModule",
                "type": "String",
                "value": XISF._creator_module,
            }
            _OSes = {
                "linux": "Linux",
                "win32": "Windows",
                "cygwin": "Windows",
                "darwin": "macOS",
            }
            xisf_metadata["XISF:CreatorOS"] = {
                "id": "XISF:CreatorOS",
                "type": "String",
                "value": _OSes[sys.platform],
            }
            xisf_metadata["XISF:BlockAlignmentSize"] = {
                "id": "XISF:BlockAlignmentSize",
                "type": "UInt16",
                "value": blk_sz,
            }
            xisf_metadata["XISF:MaxInlineBlockSize"] = {
                "id": "XISF:MaxInlineBlockSize",
                "type": "UInt16",
                "value": max_inline_blk_sz,
            }
            if codec is not None:
                # Add XISF:CompressionCodecs and XISF:CompressionLevel to file metadata
                xisf_metadata["XISF:CompressionCodecs"] = {
                    "id": "XISF:CompressionCodecs",
                    "type": "String",
                    "value": codec,
                }
                xisf_metadata["XISF:CompressionLevel"] = {
                    "id": "XISF:CompressionLevel",
                    "type": "Int",
                    "value": level if level else XISF._compression_def_level[codec],
                }
            else:
                # Remove compression metadata if exists
                try:
                    del xisf_metadata["XISF:CompressionCodecs"]
                    del xisf_metadata["XISF:CompressionLevel"]
                except:
                    pass

        def _compute_attached_positions(hdr_prov_sz, attached_blocks_locations):
            # Computes aligned position nearest to the given one
            _aligned_position = lambda pos: ((pos + blk_sz - 1) // blk_sz) * blk_sz

            # Iterates data block positions until header size stabilizes
            # (positions are represented as strings in the header so their
            # values may impact header size, therefore changing data block
            # positions in the file)
            hdr_sz = hdr_prov_sz
            prev_sum_len_positions = 0
            while True:
                # account for the size of the (provisional) header
                pos = _aligned_position(hdr_sz)

                # positions for data blocks of properties with attachment location
                sum_len_positions = 0
                for loc in attached_blocks_locations:
                    # Save the (possibly provisional) position
                    loc['position'] = pos
                    # Accumulate the size of the position string
                    sum_len_positions += len(str(pos))
                    # Fast forward position adding the size, honoring alignment
                    pos = _aligned_position(pos + loc['size'])

                if sum_len_positions == prev_sum_len_positions:
                    break

                prev_sum_len_positions = sum_len_positions
                hdr_sz = hdr_prov_sz + sum_len_positions

            # Update data blocks positions in XML Header
            for b in attached_blocks_locations:
                xml_elem, pos, sz = b["xml"], b["position"], b["size"]
                xml_elem.attrib["location"] = XISF._to_location(("attachment", pos, sz))

        # Zero padding (used for reserved fields and data block alignment)
        def _zero_pad(length):
            assert length >= 0
            return (0).to_bytes(length, byteorder="little")

        # __/ Prepare image and its metadata \__________
        im_id = image_metadata.get("id", "image")
        im_attrs, data_format = _create_image_metadata(im_data, im_id)
        im_data = _prepare_image_data_block(im_data, data_format)
        im_data_block, data_size, codec_str = _serialize_data_block(
            im_data, im_attrs, codec, level, shuffle
        )

        # Assemble location attribute, *provisional* until we can compute the data block position
        im_attrs["location"] = XISF._to_location(("attachment", "", data_size))

        # __/ Build (provisional) XML Header \__________
        # (for attached data blocks, the location is provisional)
        #   Convert metadata (dict) to XML Header
        xisf_header_xml = ET.Element("xisf", XISF._xisf_attrs)

        #   Image
        image_xml = ET.SubElement(xisf_header_xml, "Image", im_attrs)

        #     Image FITSKeywords
        for kw_name, kw_values in image_metadata.get("FITSKeywords", {}).items():
            XISF._insert_fitskeyword(image_xml, kw_name, kw_values)

        # attached_blocks_locations will reference every element whose data block is to be attached
        #   = [{"xml": ElementTree, "position": int, "size": int, "data": ndarray or str}]
        #   (position key is actually a placeholder, it will be overwritten by
        #   _compute_attached_positions)
        # The first element is the image (*provisional* location):
        attached_blocks_locations = [
            {
                "xml": image_xml,
                "position": 0,
                "size": data_size,
                "data": im_data_block,
            }
        ]

        #     Image XISFProperties
        for p_dict in image_metadata.get("XISFProperties", {}).values():
            if attached_block := XISF._insert_property(image_xml, p_dict, max_inline_blk_sz):
                attached_blocks_locations.append(attached_block)

        #   File Metadata
        metadata_xml = ET.SubElement(xisf_header_xml, "Metadata")
        _update_xisf_metadata(creator_app, blk_sz, max_inline_blk_sz, codec, level)
        for property_dict in xisf_metadata.values():
            if attached_block := XISF._insert_property(
                metadata_xml, property_dict, max_inline_blk_sz
            ):
                attached_blocks_locations.append(attached_block)

        # Header provisional size (without attachment positions)
        xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
        header_provisional_sz = (
            len(XISF._signature) + XISF._headerlength_len + len(xisf_header) + XISF._reserved_len
        )

        # Update location for every block in attached_blocks_locations
        _compute_attached_positions(header_provisional_sz, attached_blocks_locations)

        with open(fname, "wb") as f:
            # Write XISF signature
            f.write(XISF._signature)

            xisf_header = ET.tostring(xisf_header_xml, encoding="utf8")
            headerlength = len(xisf_header)
            # Write header length
            f.write(headerlength.to_bytes(XISF._headerlength_len, byteorder="little"))

            # Write reserved field
            reserved_field = _zero_pad(XISF._reserved_len)
            f.write(reserved_field)

            # Write header
            f.write(xisf_header)

            # Write data blocks
            for b in attached_blocks_locations:
                pos, data_block = b["position"], b["data"]
                f.write(_zero_pad(pos - f.tell()))
                assert f.tell() == pos
                f.write(data_block)
            bytes_written = f.tell()

        return bytes_written, codec_str

    # __/ Auxiliary functions to handle XISF attributes \________

    # Process property attributes and convert to dict
    def _process_property(self, p_et):
        p_dict = p_et.attrib.copy()

        if p_dict["type"] == "TimePoint":
            # Timepoint 'value' attribute already set (as str)
            # TODO: convert to datetime?
            pass
        elif p_dict["type"] == "String":
            p_dict["value"] = p_et.text
            if "location" in p_dict:
                # Process location and compression attributes to find data block
                self._process_location_compression(p_dict)
                p_dict["value"] = self._read_data_block(p_dict).decode("utf-8")
        elif p_dict["type"] == "Boolean":
            # Boolean valid values are "true" and "false"
            p_dict["value"] = p_dict["value"] == "true"
        elif "value" in p_et.attrib:
            # Scalars (Float64, UInt32, etc.) and Complex*
            p_dict["value"] = ast.literal_eval(p_dict["value"])
        elif "Vector" in p_dict["type"]:
            p_dict["value"] = p_et.text
            p_dict["length"] = int(p_dict["length"])
            p_dict["dtype"] = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            raw_data = self._read_data_block(p_dict)
            p_dict["value"] = np.frombuffer(raw_data, dtype=p_dict["dtype"], count=p_dict["length"])
        elif "Matrix" in p_dict["type"]:
            p_dict["value"] = p_et.text
            p_dict["rows"] = int(p_dict["rows"])
            p_dict["columns"] = int(p_dict["columns"])
            length = p_dict["rows"] * p_dict["columns"]
            p_dict["dtype"] = self._parse_vector_dtype(p_dict["type"])
            self._process_location_compression(p_dict)
            raw_data = self._read_data_block(p_dict)
            p_dict["value"] = np.frombuffer(raw_data, dtype=p_dict["dtype"], count=length)
            p_dict["value"] = p_dict["value"].reshape((p_dict["rows"], p_dict["columns"]))
        else:
            print(f"Unsupported Property type {p_dict['type']}: {p_et}")
            p_dict = False

        return p_dict

    @staticmethod
    def _process_location_compression(p_dict):
        p_dict["location"] = XISF._parse_location(p_dict["location"])
        if "compression" in p_dict:
            p_dict["compression"] = XISF._parse_compression(p_dict["compression"])

    # Insert XISF properties in the XML tree
    @staticmethod
    def _insert_property(parent, p_dict, max_inline_block_size):
        # TODO ignores optional attributes (format, comment)
        scalars = ["Int", "Byte", "Short", "Float", "Boolean", "TimePoint"]

        if any(t in p_dict["type"] for t in scalars):
            # scalars and TimePoint
            # TODO add check for scalar or TimePoint
            # TODO Boolean requires lowercase
            ET.SubElement(
                parent,
                "Property",
                {
                    "id": p_dict["id"],
                    "type": p_dict["type"],
                    "value": str(p_dict["value"]),
                },
            )
        elif p_dict["type"] == "String":
            text = str(p_dict["value"])
            sz = len(text.encode("utf-8"))
            if sz > max_inline_block_size:
                # Attach string as data block (position pending)
                # TODO ignores compression
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": text.encode()}
            else:
                # string directly as child (no 'location' attribute)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                    },
                ).text = text
        elif "Vector" in p_dict["type"]:
            # TODO ignores compression
            data = p_dict["value"]
            sz = data.nbytes
            if sz > max_inline_block_size:
                # Attach vector as data block (position pending)
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "length": str(data.size),
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": data}
            else:
                # Inline data block (assuming base64)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "length": str(data.size),
                        "location": XISF._to_location(("inline", "base64")),
                    },
                ).text = str(base64.b64encode(data.tobytes()), "ascii")
        elif "Matrix" in p_dict["type"]:
            # TODO ignores compression
            data = p_dict["value"]
            sz = data.nbytes
            if sz > max_inline_block_size:
                # Attach vector as data block (position pending)
                xml = ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "rows": str(data.shape[0]),
                        "columns": str(data.shape[1]),
                        "location": XISF._to_location(("attachment", "", sz)),
                    },
                )
                return {"xml": xml, "location": 0, "size": sz, "data": data}
            else:
                # Inline data block (assuming base64)
                ET.SubElement(
                    parent,
                    "Property",
                    {
                        "id": p_dict["id"],
                        "type": p_dict["type"],
                        "rows": str(data.shape[0]),
                        "columns": str(data.shape[1]),
                        "location": XISF._to_location(("inline", "base64")),
                    },
                ).text = str(base64.b64encode(data.tobytes()), "ascii")
        else:
            print(f"Warning: skipping unsupported property {p_dict}")

        return False

    # Insert FITS Keywords in the XML tree
    @staticmethod
    def _insert_fitskeyword(image_xml, keyword_name, keyword_values):
        for entry in keyword_values:
            ET.SubElement(
                image_xml,
                "FITSKeyword",
                {
                    "name": keyword_name,
                    "value": entry["value"],
                    "comment": entry["comment"],
                },
            )

    # Returns image shape, e.g. (x, y, channels)
    @staticmethod
    def _parse_geometry(g):
        return tuple(map(int, g.split(":")))

    # Returns ("attachment", position, size), ("inline", encoding) or ("embedded")
    @staticmethod
    def _parse_location(l):
        ll = l.split(":")
        if ll[0] not in ["inline", "embedded", "attachment"]:
            raise NotImplementedError(f"Data block location type '{ll[0]}' not implemented")
        return (ll[0], int(ll[1]), int(ll[2])) if ll[0] == "attachment" else ll

    # Serialize location tuple to string, as value for location attribute
    @staticmethod
    def _to_location(location_tuple):
        return ":".join([str(e) for e in location_tuple])

    # Returns (codec, uncompressed_size, item_size); item_size is None if not using byte shuffling
    @staticmethod
    def _parse_compression(c):
        cl = c.split(":")
        if len(cl) == 3:
            # (codec+byteshuffling, uncompressed_size, shuffling_item_size)
            return (cl[0], int(cl[1]), int(cl[2]))
        else:
            # (codec, uncompressed_size, None)
            return (cl[0], int(cl[1]), None)

    # Return equivalent numpy dtype
    @staticmethod
    def _parse_sampleFormat(s):
        # Translate alternate names to "canonical" type names
        alternate_names = {
            'Byte': 'UInt8',
            'Short': 'Int16',
            'UShort': 'UInt16',
            'Int': 'Int32',
            'UInt': 'UInt32',
            'Float': 'Float32',
            'Double': 'Float64',
        }
        try:
            s = alternate_names[s]
        except KeyError:
            pass

        _dtypes = {
            "UInt8": np.dtype("uint8"),
            "UInt16": np.dtype("uint16"),
            "UInt32": np.dtype("uint32"),
            "Float32": np.dtype("float32"),
            "Float64": np.dtype("float64"),
        }
        try:
            return _dtypes[s]
        except:
            raise NotImplementedError(f"sampleFormat {s} not implemented")

    # Return XISF data type from numpy dtype
    @staticmethod
    def _get_sampleFormat(dtype):
        _sampleFormats = {
            "uint8": "UInt8",
            "uint16": "UInt16",
            "uint32": "UInt32",
            "float32": "Float32",
            "float64": "Float64",
        }
        try:
            return _sampleFormats[str(dtype)]
        except:
            raise NotImplementedError(f"sampleFormat for {dtype} not implemented")

    @staticmethod
    def _parse_vector_dtype(type_name):
        # Translate alternate names to "canonical" type names
        alternate_names = {
            'ByteArray': 'UI8Vector',
            'IVector': 'I32Vector',
            'UIVector': 'UI32Vector',
            'Vector': 'F64Vector',
        }
        try:
            type_name = alternate_names[type_name]
        except KeyError:
            pass

        type_prefix = type_name[:-6]  # removes "Vector" and "Matrix" suffixes
        _dtypes = {
            "I8": np.dtype("int8"),
            "UI8": np.dtype("uint8"),
            "I16": np.dtype("int16"),
            "UI16": np.dtype("uint16"),
            "I32": np.dtype("int32"),
            "UI32": np.dtype("uint32"),
            "I64": np.dtype("int64"),
            "UI64": np.dtype("uint64"),
            "F32": np.dtype("float32"),
            "F64": np.dtype("float64"),
            "C32": np.dtype("csingle"),
            "C64": np.dtype("cdouble"),
        }
        try:
            return _dtypes[type_prefix]
        except:
            raise NotImplementedError(f"data type {type_name} not implemented")

    # __/ Auxiliary functions for compression/shuffling \________

    # Un-byteshuffling implementation based on numpy
    @staticmethod
    def _unshuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8"))
        a = a.reshape((item_size, -1))
        return np.transpose(a).tobytes()

    # Byteshuffling implementation based on numpy
    @staticmethod
    def _shuffle(d, item_size):
        a = np.frombuffer(d, dtype=np.dtype("uint8"))
        a = a.reshape((-1, item_size))
        return np.transpose(a).tobytes()

    # LZ4/zlib/zstd decompression
    @staticmethod
    def _decompress(data, elem):
        # (codec, uncompressed-size, item-size); item-size is None if not using byte shuffling
        codec, uncompressed_size, item_size = elem["compression"]

        if codec.startswith("lz4"):
            data = lz4.block.decompress(data, uncompressed_size=uncompressed_size)
        elif codec.startswith("zstd"):
            data = zstandard.decompress(data, max_output_size=uncompressed_size)
        elif codec.startswith("zlib"):
            data = zlib.decompress(data)
        else:
            raise NotImplementedError(f"Unimplemented compression codec {codec}")

        if item_size:  # using byte-shuffling
            data = XISF._unshuffle(data, item_size)

        return data

    # LZ4/zlib/zstd compression
    @staticmethod
    def _compress(data, codec, level=None, shuffle=False, itemsize=None):
        compressed = XISF._shuffle(data, itemsize) if shuffle else data

        if codec == "lz4hc":
            level = level if level else XISF._compression_def_level["lz4hc"]
            compressed = lz4.block.compress(
                compressed, mode="high_compression", compression=level, store_size=False
            )
        elif codec == "lz4":
            compressed = lz4.block.compress(compressed, store_size=False)
        elif codec == "zstd":
            level = level if level else XISF._compression_def_level["zstd"]
            compressed = zstandard.compress(compressed, level=level)
        elif codec == "zlib":
            level = level if level else XISF._compression_def_level["zlib"]
            compressed = zlib.compress(compressed, level=level)
        else:
            raise NotImplementedError(f"Unimplemented compression codec {codec}")

        return compressed
