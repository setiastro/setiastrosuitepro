from __future__ import annotations

from typing import Optional


VALID_BAYER_PATTERNS = frozenset({"RGGB", "BGGR", "GRBG", "GBRG"})

_PAT2MAT = {
    "RGGB": (("R", "G"), ("G", "B")),
    "BGGR": (("B", "G"), ("G", "R")),
    "GRBG": (("G", "R"), ("B", "G")),
    "GBRG": (("G", "B"), ("R", "G")),
}
_MAT2PAT = {v: k for k, v in _PAT2MAT.items()}

_BAYER_KEYS = (
    "BAYERPAT",
    "BAYERPATN",
    "BAYER_PATTERN",
    "BAYERPATTERN",
    "CFAPATTERN",
    "CFA_PATTERN",
    "PATTERN",
    "COLORTYPE",
    "COLORFILTERARRAY",
    "CFA",
)


def _parse_int_maybe(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return int(x)
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _build_probe(header, metadata=None) -> dict[str, str]:
    probe: dict[str, str] = {}

    if header is not None:
        try:
            keys = list(header.keys()) if hasattr(header, "keys") else []
            for k in keys:
                try:
                    v = header.get(k) if hasattr(header, "get") else header[k]
                except Exception:
                    v = None
                probe[str(k).upper()] = "" if v is None else str(v)
        except Exception:
            pass

    if isinstance(metadata, dict):
        for k, v in metadata.items():
            try:
                probe[str(k).upper()] = "" if v is None else str(v)
            except Exception:
                continue

    return probe


def normalize_bayer_token(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.upper().replace(",", "").replace(" ", "").replace("/", "").replace("|", "")
    if len(s) == 4 and set(s) <= {"R", "G", "B"}:
        if s.count("R") == 1 and s.count("G") == 2 and s.count("B") == 1:
            return s if s in VALID_BAYER_PATTERNS else None
    return None


def roworder_is_bottom_up(roworder: str) -> bool:
    s = (roworder or "").upper()
    return ("BOTTOM" in s) or (s in ("BU", "BOTTOMUP", "BOTTOM-UP", "UPWARDS", "UPWARD"))


def detect_bayer_offsets_and_roworder(header, metadata=None) -> tuple[int, int, str]:
    probe = _build_probe(header, metadata)
    xoff = _parse_int_maybe(probe.get("XBAYROFF")) or 0
    yoff = _parse_int_maybe(probe.get("YBAYROFF")) or 0
    roworder = (probe.get("ROWORDER") or "").upper().strip()
    return xoff, yoff, roworder


def bayer_apply_xy_offset(pattern: str, xoff: int, yoff: int) -> str:
    pattern = (pattern or "").upper()
    if pattern not in _PAT2MAT:
        return pattern
    mat = _PAT2MAT[pattern]
    xo = int(xoff) & 1
    yo = int(yoff) & 1
    shifted = (
        (mat[yo][xo], mat[yo][xo ^ 1]),
        (mat[yo ^ 1][xo], mat[yo ^ 1][xo ^ 1]),
    )
    return _MAT2PAT.get(shifted, pattern)


def detect_bayer_pattern(header, metadata=None, image_shape=None) -> Optional[str]:
    probe = _build_probe(header, metadata)

    base_pattern = None
    for key in _BAYER_KEYS:
        raw = probe.get(key)
        if not raw:
            continue
        upper = str(raw).upper()
        for pattern in VALID_BAYER_PATTERNS:
            if pattern in upper:
                base_pattern = pattern
                break
        if base_pattern:
            break
        normalized = normalize_bayer_token(upper)
        if normalized:
            base_pattern = normalized
            break

    if not base_pattern:
        return None

    xoff, yoff, roworder = detect_bayer_offsets_and_roworder(header, metadata)
    height = None
    if image_shape:
        try:
            height = int(image_shape[0])
        except Exception:
            height = None

    if roworder_is_bottom_up(roworder) and height is not None and height > 0:
        yoff = yoff + ((height - 1) & 1)

    return bayer_apply_xy_offset(base_pattern, xoff, yoff)