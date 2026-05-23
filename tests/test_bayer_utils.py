from setiastro.saspro.bayer_utils import (
    bayer_apply_xy_offset,
    detect_bayer_offsets_and_roworder,
    detect_bayer_pattern,
    normalize_bayer_token,
    roworder_is_bottom_up,
)


def test_normalize_bayer_token_accepts_common_formats():
    assert normalize_bayer_token("r g g b") == "RGGB"
    assert normalize_bayer_token("bg/gr") == "BGGR"
    assert normalize_bayer_token("gbrg") == "GBRG"


def test_detect_bayer_pattern_from_supported_keys():
    assert detect_bayer_pattern({"BAYERPAT": "RGGB"}) == "RGGB"
    assert detect_bayer_pattern({"CFAPATTERN": "BGGR"}) == "BGGR"
    assert detect_bayer_pattern({"COLORFILTERARRAY": "GRBG"}) == "GRBG"
    assert detect_bayer_pattern({"CFA": "GBRG"}) == "GBRG"


def test_detect_bayer_offsets_and_roworder_from_header_and_metadata():
    header = {"XBAYROFF": "1", "YBAYROFF": 0}
    metadata = {"roworder": "bottom-up"}
    assert detect_bayer_offsets_and_roworder(header, metadata) == (1, 0, "BOTTOM-UP")


def test_bayer_apply_xy_offset_rotates_pattern():
    assert bayer_apply_xy_offset("RGGB", 1, 0) == "GRBG"
    assert bayer_apply_xy_offset("RGGB", 0, 1) == "GBRG"


def test_detect_bayer_pattern_applies_offsets_and_bottom_up_adjustment():
    header = {"BAYERPAT": "RGGB", "XBAYROFF": 1, "YBAYROFF": 0, "ROWORDER": "BOTTOM-UP"}
    assert detect_bayer_pattern(header, image_shape=(10, 10)) == "BGGR"
    assert detect_bayer_pattern(header, image_shape=(11, 10)) == "GRBG"


def test_detect_bayer_pattern_uses_metadata_when_header_is_missing():
    metadata = {"colorfilterarray": "RGGB"}
    assert detect_bayer_pattern(None, metadata=metadata) == "RGGB"


def test_detect_bayer_pattern_returns_none_for_invalid_values():
    assert detect_bayer_pattern({"BAYERPAT": "XTRANS"}) is None
    assert normalize_bayer_token("RGB") is None
    assert roworder_is_bottom_up("TOP-DOWN") is False
