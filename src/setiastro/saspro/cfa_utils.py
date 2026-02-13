# setiastro/saspro/cfa_utils.py
from __future__ import annotations
import numpy as np
from typing import Optional

_VALID = {"RGGB", "BGGR", "GRBG", "GBRG"}

_PAT2MAT = {
    "RGGB": (("R","G"),("G","B")),
    "BGGR": (("B","G"),("G","R")),
    "GRBG": (("G","R"),("B","G")),
    "GBRG": (("G","B"),("R","G")),
}
_MAT2PAT = {v: k for k, v in _PAT2MAT.items()}

def _parse_int_maybe(x) -> Optional[int]:
    try:
        if x is None: return None
        if isinstance(x, (int, np.integer)): return int(x)
        s = str(x).strip()
        if not s: return None
        return int(float(s))
    except Exception:
        return None

def roworder_is_bottom_up(roworder: str) -> bool:
    s = (roworder or "").upper().strip()
    return ("BOTTOM" in s) or (s in ("BU", "BOTTOMUP", "BOTTOM-UP"))

def bayer_apply_xy_offset(pat: str, xoff: int, yoff: int) -> str:
    pat = (pat or "").upper()
    if pat not in _PAT2MAT: return pat
    m = _PAT2MAT[pat]
    xo = int(xoff) & 1
    yo = int(yoff) & 1
    nm = (
        (m[yo][xo],     m[yo][xo ^ 1]),
        (m[yo ^ 1][xo], m[yo ^ 1][xo ^ 1]),
    )
    return _MAT2PAT.get(nm, pat)

def effective_bayer_from_header(header, height: int) -> tuple[Optional[str], int, int, str]:
    """
    Returns (effective_pat, xoff, yoff, roworder).
    effective_pat is one of _VALID or None.
    """
    # base token from common keys
    bp = str(
        header.get("BAYERPAT")
        or header.get("BAYERPATN")
        or header.get("CFA_PATTERN")
        or header.get("BAYER_PATTERN")
        or ""
    ).upper().strip()

    if bp not in _VALID:
        return (None, 0, 0, str(header.get("ROWORDER") or ""))

    xoff = _parse_int_maybe(header.get("XBAYROFF")) or 0
    yoff = _parse_int_maybe(header.get("YBAYROFF")) or 0
    roworder = str(header.get("ROWORDER") or "")

    # If we flip bottom-up to top-down, y parity shifts by (H-1) mod 2
    if roworder_is_bottom_up(roworder) and height and height > 0:
        yoff = yoff + ((int(height) - 1) & 1)

    eff = bayer_apply_xy_offset(bp, xoff, yoff)
    if eff not in _VALID:
        return (bp, xoff, yoff, roworder)
    return (eff, xoff, yoff, roworder)
