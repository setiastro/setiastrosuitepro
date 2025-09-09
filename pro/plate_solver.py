# pro/plate_solver.py
from __future__ import annotations

import os, re, math, tempfile
from typing import Tuple, Dict, Any, Optional

import numpy as np

from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS

from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QStackedWidget, QWidget, QMessageBox,
    QLineEdit, QTextEdit, QApplication
)

# === our I/O & stretch — migrate from SASv2 ===
from legacy.image_manager import load_image, save_image   # <<<< IMPORTANT
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None


# ---------------------------------------------------------------------
# Utilities (headers, parsing, normalization)
# ---------------------------------------------------------------------

def _parse_astap_wcs_file(wcs_path: str) -> Dict[str, Any]:
    """
    Robustly load the .wcs file using astropy (instead of line parsing).
    Returns a dictionary of key → value.
    """
    if not wcs_path or not os.path.exists(wcs_path):
        return {}

    try:
        header = fits.getheader(wcs_path)
        return dict(header)
    except Exception as e:
        print(f"[ASTAP] Failed to parse .wcs with astropy: {e}")
        return {}


def _get_astap_exe(settings) -> str:
    # Support both SASpro and SASv2 keys.
    cand = [
        settings.value("paths/astap", "", type=str) or "",
        settings.value("astap/exe_path", "", type=str) or "",
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return cand[0]  # return first even if missing so we can error nicely


def _as_header(hdr_like: Any) -> Header | None:
    """
    Try to coerce whatever we have in metadata to a proper astropy Header.
    Accepts: fits.Header, dict, flattened string blobs (best effort).
    """
    if hdr_like is None:
        return None
    if isinstance(hdr_like, Header):
        return hdr_like

    # 1) flattened single string? try hard to parse
    if isinstance(hdr_like, str):
        h = _parse_header_blob_to_header(hdr_like)
        return h if len(h.keys()) else None

    # 2) dict-ish
    try:
        d = dict(hdr_like)
        h = Header()
        int_keys = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER", "WCSAXES", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"}
        for k, v in d.items():
            K = str(k).upper()
            try:
                if K in int_keys:
                    h[K] = int(float(str(v).strip().split()[0]))
                elif re.match(r"^(?:A|B|AP|BP)_\d+_\d+$", K) or \
                     re.match(r"^(?:CRPIX|CRVAL|CDELT|CD|PC|CROTA|LATPOLE|LONPOLE|EQUINOX)\d?_?\d*$", K):
                    h[K] = float(str(v).strip().split()[0])
                elif K.startswith("CTYPE") or K.startswith("CUNIT") or K in {"RADECSYS"}:
                    h[K] = str(v).strip().strip("'\"")
                else:
                    h[K] = v
            except Exception:
                pass
        # SIP order parity
        if "A_ORDER" in h and "B_ORDER" not in h:
            h["B_ORDER"] = int(h["A_ORDER"])
        if "B_ORDER" in h and "A_ORDER" not in h:
            h["A_ORDER"] = int(h["B_ORDER"])
        return h
    except Exception:
        return None


def _parse_header_blob_to_header(blob: str) -> Header:
    """
    Turn a flattened header blob into a real fits.Header.
    Handles 80-char concatenated cards or KEY=VAL regex fallback.
    """
    s = (blob or "").strip()
    h = fits.Header()

    # A) 80-char card chunking (if truly concatenated FITS cards)
    if len(s) >= 80 and len(s) % 80 == 0:
        cards = [s[i:i+80] for i in range(0, len(s), 80)]
        for line in cards:
            try:
                card = fits.Card.fromstring(line)
                if card.keyword not in ("COMMENT", "HISTORY", "END", ""):
                    h.append(card)
            except Exception:
                pass
        if len(h.keys()):
            return h

    # B) Fallback regex KEY = value … next KEY
    pattern = r"([A-Z0-9_]+)\s*=\s*([^=]*?)(?=\s{2,}[A-Z0-9_]+\s*=|$)"
    for m in re.finditer(pattern, s):
        key  = m.group(1).strip().upper()
        vraw = m.group(2).strip()
        if vraw.startswith("'") and vraw.endswith("'"):
            val = vraw[1:-1].strip()
        else:
            try:
                if re.fullmatch(r"[+-]?\d+", vraw): val = int(vraw)
                else: val = float(vraw)
            except Exception:
                val = vraw
        try: h[key] = val
        except Exception: pass

    if "A_ORDER" in h and "B_ORDER" not in h:
        h["B_ORDER"] = int(h["A_ORDER"])
    if "B_ORDER" in h and "A_ORDER" not in h:
        h["A_ORDER"] = int(h["B_ORDER"])

    return h


def _strip_wcs_keys(h: Header) -> Header:
    """Return a copy without WCS/SIP keys (so ASTAP can write fresh)."""
    h = h.copy()
    for key in list(h.keys()):
        ku = key.upper()
        for prefix in (
            "CRPIX", "CRVAL", "CDELT", "CROTA",
            "CD1_", "CD2_", "PC", "CTYPE", "CUNIT",
            "WCSAXES", "LATPOLE", "LONPOLE", "EQUINOX",
            "PV1_", "PV2_", "SIP",
            "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
            "A_", "B_", "AP_", "BP_", "PLTSOLVD"
        ):
            if ku.startswith(prefix):
                h.pop(key, None)
                break
    return h

def _minimal_header_for_gray2d(h: int, w: int) -> Header:
    hdu = Header()
    hdu["SIMPLE"] = True
    hdu["BITPIX"] = -32
    hdu["NAXIS"]  = 2
    hdu["NAXIS1"] = int(w)
    hdu["NAXIS2"] = int(h)
    hdu["BZERO"]  = 0.0
    hdu["BSCALE"] = 1.0
    hdu.add_comment("Temp FITS written for ASTAP solve.")
    return hdu

def _minimal_header_for(img: np.ndarray, is_mono: bool) -> Header:
    H = int(img.shape[0]) if img.ndim >= 2 else 1
    W = int(img.shape[1]) if img.ndim >= 2 else 1
    C = int(img.shape[2]) if (img.ndim == 3) else 1
    h = Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    h["NAXIS"]  = 2 if is_mono else 3
    h["NAXIS1"] = W
    h["NAXIS2"] = H
    if not is_mono:
        h["NAXIS3"] = C
    h["BZERO"]  = 0.0
    h["BSCALE"] = 1.0
    h.add_comment("Temp FITS written for ASTAP solve.")
    return h


def _float01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        if info.max == 0: return a.astype(np.float32)
        return (a.astype(np.float32) / float(info.max))
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def _normalize_for_astap(img: np.ndarray) -> np.ndarray:
    """
    Use migrated stretch functions when available.
    Returns float32 in [0,1], 2D for mono or 3D for color.
    """
    f01 = _float01(img)
    if f01.ndim == 2 or (f01.ndim == 3 and f01.shape[2] == 1):
        try:
            print("DEBUG stretching mono")
            out = stretch_mono_image(f01, 0.1, False)
            return out
        except Exception:
            pass

    else:
        try:
            print("DEBUG stretching color")
            out = stretch_color_image(f01, 0.1, False, False)
            return out
        except Exception:
            pass



def _first_float(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v)
    m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", s)
    return float(m.group(0)) if m else None


def _first_int(v):
    if v is None: return None
    if isinstance(v, int): return v
    if isinstance(v, float): return int(round(v))
    s = str(v)
    m = re.search(r"[+-]?\d+", s)
    return int(m.group(0)) if m else None


def _parse_ra_deg(h: Header) -> float | None:
    ra = _first_float(h.get("CRVAL1"))
    if ra is not None: return ra
    ra = _first_float(h.get("RA"))
    if ra is not None and 0.0 <= ra < 360.0: return ra
    for key in ("OBJCTRA", "RA"):
        s = h.get(key); 
        if not s: continue
        s = str(s).strip()
        parts = re.split(r"[:\s]+", s)
        try:
            if len(parts) >= 3:
                hh, mm, ss = float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                hh, mm, ss = float(parts[0]), float(parts[1]), 0.0
            else:
                x = float(parts[0]); 
                return x if x > 24 else x*15.0
            return (abs(hh) + mm/60.0 + ss/3600.0) * 15.0
        except Exception:
            pass
    return None


def _parse_dec_deg(h: Header) -> float | None:
    dec = _first_float(h.get("CRVAL2"))
    if dec is not None: return dec
    dec = _first_float(h.get("DEC"))
    if dec is not None and -90 <= dec <= 90: return dec
    for key in ("OBJCTDEC","DEC"):
        s = h.get(key); 
        if not s: continue
        s = str(s).strip()
        sign = -1.0 if s.startswith("-") else 1.0
        s = s.lstrip("+-")
        parts = re.split(r"[:\s]+", s)
        try:
            if len(parts) >= 3:
                dd, mm, ss = float(parts[0]), float(parts[1]), float(parts[2])
            elif len(parts) == 2:
                dd, mm = float(parts[0]), float(parts[1]); ss = 0.0
            else:
                return sign * float(parts[0])
            return sign * (abs(dd) + mm/60.0 + ss/3600.0)
        except Exception:
            pass
    return None


def _compute_scale_arcsec_per_pix(h: Header) -> float | None:
    cd11 = _first_float(h.get("CD1_1"))
    cd21 = _first_float(h.get("CD2_1"))
    cdelt1 = _first_float(h.get("CDELT1"))
    cdelt2 = _first_float(h.get("CDELT2"))
    if cd11 is not None or cd21 is not None:
        cd11 = cd11 or 0.0
        cd21 = cd21 or 0.0
        return ((cd11**2 + cd21**2)**0.5) * 3600.0
    if cdelt1 is not None or cdelt2 is not None:
        cdelt1 = cdelt1 or 0.0
        cdelt2 = cdelt2 or 0.0
        return ((cdelt1**2 + cdelt2**2)**0.5) * 3600.0
    px_um_x = _first_float(h.get("XPIXSZ"))
    px_um_y = _first_float(h.get("YPIXSZ"))
    focal_mm = _first_float(h.get("FOCALLEN"))
    if focal_mm and (px_um_x or px_um_y):
        px_um = px_um_x if (px_um_x and not px_um_y) else px_um_y if (px_um_y and not px_um_x) else (None)
        if px_um is None: px_um = (px_um_x + px_um_y) / 2.0
        bx = _first_int(h.get("XBINNING")) or _first_int(h.get("XBIN")) or 1
        by = _first_int(h.get("YBINNING")) or _first_int(h.get("YBIN")) or 1
        bin_factor = (bx + by) / 2.0
        px_um_eff = px_um * bin_factor
        return 206.264806 * px_um_eff / float(focal_mm)
    return None


def _build_astap_seed(h: Header) -> Tuple[list[str], str]:
    dbg = []
    ra_deg  = _parse_ra_deg(h)
    dec_deg = _parse_dec_deg(h)
    scale   = _compute_scale_arcsec_per_pix(h)

    if ra_deg is None:  dbg.append("RA unknown")
    if dec_deg is None: dbg.append("Dec unknown")
    if not scale or not np.isfinite(scale) or scale <= 0: dbg.append("scale unknown")

    if dbg:
        return [], " / ".join(dbg)

    ra_h = ra_deg / 15.0
    spd  = dec_deg + 90.0
    args = ["-ra", f"{ra_h:.6f}", "-spd", f"{spd:.6f}", "-scale", f"{scale:.3f}"]
    return args, f"RA={ra_h:.6f} h | SPD={spd:.6f}° | scale={scale:.3f}\"/px"


def _read_header_from_fits(path: str) -> Dict[str, Any]:
    with fits.open(path, memmap=False) as hdul:
        d = dict(hdul[0].header)
    d.pop("COMMENT", None); d.pop("HISTORY", None); d.pop("END", None)
    return d


def _header_from_text_block(s: str) -> Header:
    """Parse ASTAP .wcs or flattened blocks into a proper Header."""
    h = Header()
    if not s: return h
    lines = s.splitlines()
    if len(lines) <= 1:
        # single blob: split on KEY=
        lines = re.split(r"(?=(?:^|\s{2,})([A-Za-z][A-Za-z0-9_]+)\s*=)", s)
        lines = ["".join(lines[i:i+2]).strip() for i in range(1, len(lines), 2)]
    card_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]+)\s*=\s*(.*)$")
    num_re  = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.upper().startswith(("COMMENT","HISTORY","END")):
            continue
        m = card_re.match(raw)
        if not m: continue
        key, rest = m.group(1).upper(), m.group(2).strip()
        if " /" in rest:
            val_str = rest.split(" /", 1)[0].strip()
        else:
            val_str = rest
        if (len(val_str) >= 2) and ((val_str[0] == "'" and val_str[-1] == "'") or (val_str[0] == '"' and val_str[-1] == '"')):
            val = val_str[1:-1].strip()
        else:
            try:
                if num_re.match(val_str):
                    val = float(val_str)
                    if re.match(r"^[+-]?\d+$", val_str): val = int(val)
                else:
                    val = val_str
            except Exception:
                val = val_str
        try: h[key] = val
        except Exception: pass
    if "A_ORDER" in h and "B_ORDER" not in h:
        h["B_ORDER"] = int(h["A_ORDER"])
    if "B_ORDER" in h and "A_ORDER" not in h:
        h["A_ORDER"] = int(h["B_ORDER"])
    return h

def _coerce_wcs_numbers(d: dict[str, Any]) -> dict[str, Any]:
    """
    Convert values for common WCS/SIP keys to int/float where appropriate.
    Mirrors SASv2 logic.
    """
    numeric_keys = {
        "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CROTA1", "CROTA2",
        "EQUINOX", "WCSAXES", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
    }

    out = {}
    for k, v in d.items():
        key = k.upper()
        try:
            if key in numeric_keys or re.match(r"^(A|B|AP|BP)_\d+_\d+$", key):
                if isinstance(v, str):
                    val = float(v.strip())
                    if val.is_integer(): val = int(val)
                else:
                    val = v
                out[key] = val
            else:
                out[key] = v
        except Exception:
            out[key] = v
    return out


def _ensure_ctypes(d: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure CTYPE1/2 exist and are proper strings. Fallback to TAN if missing.
    """
    if "CTYPE1" not in d:
        d["CTYPE1"] = "RA---TAN"
    if "CTYPE2" not in d:
        d["CTYPE2"] = "DEC--TAN"
    d["CTYPE1"] = str(d["CTYPE1"]).strip()
    d["CTYPE2"] = str(d["CTYPE2"]).strip()
    return d


def _build_header_from_astap_outputs(tmp_fits: str, sidecar_wcs: Optional[str]) -> Header:
    # --- 1) read plain header from temp FITS (ASTAP usually doesn't inject WCS here) ---
    base_dict: Dict[str, Any] = {}
    try:
        with fits.open(tmp_fits, memmap=False) as hdul:
            base_dict = dict(hdul[0].header)
        # drop noisy cards
        for k in ("COMMENT", "HISTORY", "END"):
            base_dict.pop(k, None)
    except Exception as e:
        print("Failed reading temp FITS header:", e)

    # Debug: dump what we actually got from the temp FITS
    try:
        print("\n================ ASTAP: reading solved header ==================")
        print("---- RAW header from temp FITS ----")
        for k, v in base_dict.items():
            print(f"  FITS[{k}] = {v!r}")
    except Exception:
        pass

    # --- 2) merge .wcs sidecar if present (this is where ASTAP writes WCS/SIP) ---
    wcs_dict: Dict[str, Any] = {}
    if sidecar_wcs and os.path.exists(sidecar_wcs):
        try:
            wcs_dict = _parse_astap_wcs_file(sidecar_wcs)
            # Debug
            print("\n---- .WCS sidecar (ASTAP) ----")
            for k, v in wcs_dict.items():
                print(f"  WCS[{k}] = {v!r}")
        except Exception as e:
            print("Error parsing .wcs file:", e)

    # Merge (sidecar wins)
    merged: Dict[str, Any] = dict(base_dict)
    merged.update(wcs_dict)

    # --- 3) coerce numeric types for common WCS/SIP keys ---
    merged = _ensure_ctypes(_coerce_wcs_numbers(merged))

    # If SIP is present, ensure TAN-SIP on CTYPEs
    try:
        sip_present = any(re.match(r"^(A|B|AP|BP)_\d+_\d+$", k) for k in merged.keys())
        if sip_present:
            c1 = str(merged.get("CTYPE1", "RA---TAN"))
            c2 = str(merged.get("CTYPE2", "DEC--TAN"))
            if not c1.endswith("-SIP"): merged["CTYPE1"] = "RA---TAN-SIP"
            if not c2.endswith("-SIP"): merged["CTYPE2"] = "DEC--TAN-SIP"
    except Exception:
        pass

    # parity for SIP orders (if only one present)
    if "A_ORDER" in merged and "B_ORDER" not in merged:
        try: merged["B_ORDER"] = int(merged["A_ORDER"])
        except Exception: pass
    if "B_ORDER" in merged and "A_ORDER" not in merged:
        try: merged["A_ORDER"] = int(merged["B_ORDER"])
        except Exception: pass

    # --- 4) build a real astropy Header from the merged dict ---
    final_hdr = Header()
    for k, v in merged.items():
        try:
            final_hdr[k] = v
        except Exception:
            # skip any weird/overlong keys
            pass

    # CROTA if missing but CD present
    try:
        if ("CROTA1" not in final_hdr or "CROTA2" not in final_hdr) and \
           ("CD1_1" in final_hdr and "CD1_2" in final_hdr):
            rot = math.degrees(math.atan2(float(final_hdr["CD1_2"]), float(final_hdr["CD1_1"])))
            final_hdr["CROTA1"] = rot
            final_hdr["CROTA2"] = rot
    except Exception:
        pass


    return final_hdr




def _write_temp_fit_via_save_image(gray2d: np.ndarray, _header: Header | None) -> tuple[str, str]:
    """
    Write a 2-D mono float32 FITS using legacy.save_image(), return (fit_path, sidecar_wcs_path).

    NOTE: We intentionally ignore the incoming header's axis cards and
    build a clean 2-axis header to avoid 'NAXISj out of range' errors.
    """
    # ensure 2-D float32 in [0,1]
    if gray2d.ndim != 2:
        raise ValueError("Expected a 2-D grayscale array for ASTAP temp FITS.")
    g = np.clip(gray2d.astype(np.float32), 0.0, 1.0)

    H, W = int(g.shape[0]), int(g.shape[1])

    # Build a *fresh* 2-axis header (no NAXIS3, no old WCS)
    clean_header = Header()
    clean_header["SIMPLE"] = True
    clean_header["BITPIX"] = -32
    clean_header["NAXIS"]  = 2
    clean_header["NAXIS1"] = W
    clean_header["NAXIS2"] = H
    clean_header["BZERO"]  = 0.0
    clean_header["BSCALE"] = 1.0
    clean_header.add_comment("Temp FITS written for ASTAP solve (mono 2-D).")

    # Write using legacy.save_image (forces a valid 2-axis primary HDU)
    tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
    tmp_path = tmp.name
    tmp.close()

    save_image(
        img_array=g,
        filename=tmp_path,
        original_format="fit",                 # (our stack expects 'fit')
        bit_depth="32-bit floating point",
        original_header=clean_header,
        is_mono=True                           # <-- important: keep it 2-D/mono
    )

    # Resolve the actual path in case save_image normalized the extension
    base, _ = os.path.splitext(tmp_path)
    candidates = [tmp_path, base + ".fit", base + ".fits", base + ".FIT", base + ".FITS"]
    fit_path = next((p for p in candidates if os.path.exists(p)), tmp_path)

    print(f"Saved FITS image to: {fit_path}")
    return fit_path, os.path.splitext(fit_path)[0] + ".wcs"



def _save_temp_fits_via_save_image(norm_img: np.ndarray, clean_header: Header, is_mono: bool) -> str:
    """
    Legacy helper used elsewhere. Make sure header axes match the data we write.
    If is_mono=True we force a 2-D primary HDU; otherwise we allow 3-D (H,W,C).
    """
    hdr = Header()
    # sanitize header/axes
    if is_mono:
        # force 2-axis
        if norm_img.ndim != 2:
            raise ValueError("Expected 2-D array for mono temp FITS.")
        H, W = int(norm_img.shape[0]), int(norm_img.shape[1])
        hdr["SIMPLE"] = True
        hdr["BITPIX"] = -32
        hdr["NAXIS"]  = 2
        hdr["NAXIS1"] = W
        hdr["NAXIS2"] = H
    else:
        # allow color (H, W, C)
        if norm_img.ndim != 3 or norm_img.shape[2] < 3:
            raise ValueError("Expected 3-D array (H,W,C) for color temp FITS.")
        H, W, C = int(norm_img.shape[0]), int(norm_img.shape[1]), int(norm_img.shape[2])
        hdr["SIMPLE"] = True
        hdr["BITPIX"] = -32
        hdr["NAXIS"]  = 3
        hdr["NAXIS1"] = W
        hdr["NAXIS2"] = H
        hdr["NAXIS3"] = C

    hdr["BZERO"]  = 0.0
    hdr["BSCALE"] = 1.0
    hdr.add_comment("Temp FITS written for ASTAP solve.")

    # write
    tmp = tempfile.NamedTemporaryFile(suffix=".fit", delete=False)
    tmp_path = tmp.name
    tmp.close()

    save_image(
        img_array=np.clip(norm_img.astype(np.float32), 0.0, 1.0),
        filename=tmp_path,
        original_format="fit",
        bit_depth="32-bit floating point",
        original_header=hdr,
        is_mono=is_mono
    )

    return tmp_path



def _active_doc_from_parent(parent) -> object | None:
    """Try your helpers to get the active document."""
    if hasattr(parent, "_active_doc"):
        try:
            return parent._active_doc()
        except Exception:
            pass
    sw = getattr(parent, "mdi", None)
    if sw and hasattr(sw, "activeSubWindow"):
        asw = sw.activeSubWindow()
        if asw:
            w = asw.widget()
            return getattr(w, "document", None)
    return None

def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Always produce a 2-D grayscale float32 in [0,1]."""
    a = np.asarray(arr)
    # normalize to 0..1 first
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        a = a.astype(np.float32) / max(float(info.max), 1.0)
    else:
        a = np.clip(a.astype(np.float32), 0.0, 1.0)

    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[2] >= 3:
            return (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]).astype(np.float32)
        return a[...,0].astype(np.float32)
    # anything else, just flatten safely
    return a.reshape(a.shape[0], -1).astype(np.float32)

def _to_gray2d_unit(arr: np.ndarray) -> np.ndarray:
    """
    Return a 2-D float32 array in [0,1].
    """
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        a = a.astype(np.float32) / max(float(info.max), 1.0)
    else:
        a = np.clip(a.astype(np.float32), 0.0, 1.0)

    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # perceptual luminance → 2-D
        if a.shape[2] >= 3:
            return (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]).astype(np.float32)
        return a[...,0].astype(np.float32)
    # last resort: collapse to (H, W)
    return a.reshape(a.shape[0], -1).astype(np.float32)


# ---------------------------------------------------------------------
# Core ASTAP solving for a numpy image + seed header
# ---------------------------------------------------------------------

def _solve_numpy_with_astap(parent, settings, image: np.ndarray, seed_header: Header | None) -> Tuple[bool, Header | str]:
    """
    Normalize → write temp mono FITS → run ASTAP → return the EXACT FITS header ASTAP wrote.
    """
    astap_exe = _get_astap_exe(settings)
    if not astap_exe or not os.path.exists(astap_exe):
        return False, "ASTAP path is not set (see Preferences) or file not found."

    # normalize and force 2-D luminance in [0,1]
    norm = _normalize_for_astap(image)
    #gray = _to_gray2d_unit(image)
    gray = _to_gray2d_unit(norm)

    # build a clean temp header (strip old WCS if we have one)
    clean_for_temp = _strip_wcs_keys(seed_header) if isinstance(seed_header, Header) else _minimal_header_for_gray2d(*gray.shape)

    # write temp FITS via our legacy save_image (returns actual path on disk)
    tmp_fit, sidecar_wcs = _write_temp_fit_via_save_image(gray, clean_for_temp)

    # seed if possible; otherwise blind
    seed_args: list[str] = []
    if seed_header is not None:
        try:
            seed_args, dbg = _build_astap_seed(seed_header)
            if seed_args:
                print("ASTAP seed:", dbg)
        except Exception as e:
            print("Seed build error:", e)
    if not seed_args:
        seed_args = ["-r", "179", "-fov", "0", "-z", "0"]
        print("ASTAP BLIND: using arguments:", " ".join(seed_args))

    args = ["-f", tmp_fit] + seed_args + ["-wcs", "-sip"]
    print("Running ASTAP with:", " ".join([astap_exe] + args))

    proc = QProcess(parent)
    proc.start(astap_exe, args)
    if not proc.waitForStarted(5000):
        try: os.remove(tmp_fit)
        except Exception: pass
        return False, f"Failed to start ASTAP: {proc.errorString()}"
    if not proc.waitForFinished(300000):
        try: os.remove(tmp_fit)
        except Exception: pass
        return False, "ASTAP timed out."

    if proc.exitCode() != 0:
        out = bytes(proc.readAllStandardOutput()).decode(errors="ignore")
        err = bytes(proc.readAllStandardError()).decode(errors="ignore")
        print("ASTAP failed.\nSTDOUT:\n", out, "\nSTDERR:\n", err)
        try: os.remove(tmp_fit)
        except Exception: pass
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception: pass
        return False, "ASTAP returned a non-zero exit code."

    # >>> THIS is the key change: read the header **directly** from the FITS ASTAP wrote
    try:
        hdr = _build_header_from_astap_outputs(tmp_fit, sidecar_wcs)
    finally:
        try: os.remove(tmp_fit)
        except Exception: pass
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception: pass

    # return a REAL fits.Header (no blobs/strings/dicts)
    return True, hdr



# ---------------------------------------------------------------------
# Solve active doc in-place
# ---------------------------------------------------------------------

def plate_solve_doc_inplace(parent, doc, settings) -> Tuple[bool, Header | str]:
    """
    Run ASTAP on the doc's image; merge WCS/SIP back into doc.metadata.
    Adds debug prints for what we store into metadata.
    """
    img = getattr(doc, "image", None)
    if img is None:
        return False, "Active document has no image data."

    meta  = getattr(doc, "metadata", {}) or {}
    seed_h = _as_header(meta.get("original_header") or meta)

    ok, res = _solve_numpy_with_astap(parent, settings, img, seed_h)
    if not ok:
        return False, res
    hdr: Header = res

    # ---- DEBUG: show exactly what we’re about to write to metadata
    try:
        print("\n================ Storing into doc.metadata =================")
        print(f"original_header -> FITS.Header with {len(hdr)} keys")
        for k, v in hdr.items():
            print(f"  META_HDR[{k}] = {v!r}")
        print("===========================================================\n")
    except Exception as e:
        print("Debug print of stored header failed:", e)

    # write back
    if not isinstance(doc.metadata, dict):
        setattr(doc, "metadata", {})
    doc.metadata["original_header"] = hdr

    # Build WCS (best-effort)
    try:
        wcs_obj = WCS(hdr)
        doc.metadata["wcs"] = wcs_obj
        try:
            naxis = getattr(wcs_obj, "naxis", None)
            print(f"WCS constructed successfully (naxis={naxis}).")
        except Exception:
            print("WCS constructed successfully.")
    except Exception as e:
        print("WCS build FAILED:", e)

    # notify UI immediately
    if hasattr(doc, "changed"):
        try:
            doc.changed.emit()
        except Exception:
            pass

    if hasattr(parent, "header_viewer") and hasattr(parent.header_viewer, "set_document"):
        QTimer.singleShot(0, lambda: parent.header_viewer.set_document(doc))
    if hasattr(parent, "_refresh_header_viewer"):
        QTimer.singleShot(0, lambda: parent._refresh_header_viewer(doc))
    if hasattr(parent, "currentDocumentChanged"):
        QTimer.singleShot(0, lambda: parent.currentDocumentChanged.emit(doc))

    return True, hdr



# ---------------------------------------------------------------------
# Dialog UI with Active/File and Batch modes
# ---------------------------------------------------------------------

class PlateSolverDialog(QDialog):
    """
    Plate-solve either:
      - Active View (default)
      - Single File (via load_image/save_image)
      - Batch (directory → directory)
    Uses settings key: 'paths/astap' or 'astap/exe_path' for ASTAP executable.
    """
    def __init__(self, settings, parent=None, icon: QIcon | None = None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Plate Solver")
        self.setMinimumWidth(520)

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Active View", "File", "Batch"])

        self.stack = QStackedWidget(self)

        # Page 0: Active view info
        p0 = QWidget(self); l0 = QVBoxLayout(p0)
        l0.addWidget(QLabel("Solve the currently active image view.", p0))
        l0.addStretch(1)
        self.stack.addWidget(p0)

        # Page 1: File picker
        p1 = QWidget(self); l1 = QVBoxLayout(p1)
        self.le_path = QLineEdit(p1); self.le_path.setPlaceholderText("Choose an image…")
        btn_browse = QPushButton("Browse…", p1)
        r1 = QHBoxLayout(); r1.addWidget(self.le_path, 1); r1.addWidget(btn_browse)
        l1.addLayout(r1); l1.addStretch(1)
        self.stack.addWidget(p1)

        # Page 2: Batch
        p2 = QWidget(self); l2 = QVBoxLayout(p2)
        self.le_in  = QLineEdit(p2);  self.le_in.setPlaceholderText("Input directory")
        self.le_out = QLineEdit(p2);  self.le_out.setPlaceholderText("Output directory")
        b_in  = QPushButton("Browse Input…", p2)
        b_out = QPushButton("Browse Output…", p2)
        self.log  = QTextEdit(p2); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        r2 = QHBoxLayout(); r2.addWidget(self.le_in, 1); r2.addWidget(b_in)
        r3 = QHBoxLayout(); r3.addWidget(self.le_out, 1); r3.addWidget(b_out)
        l2.addLayout(r2); l2.addLayout(r3); l2.addWidget(QLabel("Status:")); l2.addWidget(self.log)
        self.stack.addWidget(p2)

        self.status = QLabel("", self)
        self.btn_go  = QPushButton("Start")
        self.btn_close = QPushButton("Close")

        top = QHBoxLayout(); top.addWidget(QLabel("Mode:")); top.addWidget(self.mode_combo); top.addStretch(1)
        bot = QHBoxLayout(); bot.addStretch(1); bot.addWidget(self.btn_go); bot.addWidget(self.btn_close)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.stack)
        lay.addWidget(self.status)
        lay.addLayout(bot)

        self.mode_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        btn_browse.clicked.connect(self._browse_file)
        b_in.clicked.connect(self._browse_in)
        b_out.clicked.connect(self._browse_out)
        self.btn_go.clicked.connect(self._run)
        self.btn_close.clicked.connect(self.close)

        if icon:
            self.setWindowIcon(icon)

    # ---------- file/batch pickers ----------
    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Choose Image",
            "", "Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)"
        )
        if f:
            self.le_path.setText(f)

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Choose input directory")
        if d: self.le_in.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if d: self.le_out.setText(d)

    # ---------- actions ----------
    def _run(self):
        astap_exe = _get_astap_exe(self.settings)
        if not astap_exe or not os.path.exists(astap_exe):
            self.status.setText("ASTAP path missing. Set Preferences → ASTAP executable.")
            QMessageBox.warning(self, "Plate Solver", "ASTAP path missing.\nSet it in Preferences → ASTAP executable.")
            return

        mode = self.stack.currentIndex()
        if mode == 0:
            # Active view
            doc = _active_doc_from_parent(self.parent())
            if not doc:
                QMessageBox.information(self, "Plate Solver", "No active image view.")
                return
            ok, res = plate_solve_doc_inplace(self.parent(), doc, self.settings)
            if ok:
                self.status.setText("Solved with ASTAP (WCS + SIP applied to active doc).")
                QTimer.singleShot(0, self.accept)  # close when done
            else:
                self.status.setText(str(res))
        elif mode == 1:
            # Single file
            path = self.le_path.text().strip()
            if not path:
                QMessageBox.information(self, "Plate Solver", "Choose a file to solve.")
                return
            if not os.path.exists(path):
                QMessageBox.warning(self, "Plate Solver", "Selected file does not exist.")
                return
            self._solve_file(path)
        else:
            self._run_batch()

    def _solve_file(self, path: str):
        # Load using legacy.load_image()
        try:
            image_data, original_header, bit_depth, is_mono = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Plate Solver", f"Cannot read image:\n{e}")
            return
        if image_data is None:
            QMessageBox.warning(self, "Plate Solver", "Unsupported or unreadable image.")
            return

        seed_h = _as_header(original_header) if isinstance(original_header, (dict, Header)) else None

        ok, res = _solve_numpy_with_astap(self, self.settings, image_data, seed_h)
        if not ok:
            self.status.setText(str(res)); return
        hdr: Header = res

        # Save-as using legacy.save_image() with ORIGINAL pixels (not normalized)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Plate-Solved FITS", "", "FITS files (*.fits *.fit)")
        if save_path:
            try:
                # never persist 'file_path' inside FITS
                h2 = Header()
                for k in hdr.keys(): 
                    if k.upper() != "FILE_PATH":
                        h2[k] = hdr[k]
                save_image(
                    img_array=image_data,
                    filename=save_path,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=h2,
                    is_mono=is_mono
                )
                self.status.setText(f"Solved FITS saved:\n{save_path}")
                QTimer.singleShot(0, self.accept)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")
        else:
            self.status.setText("Solved (not saved).")

    def _run_batch(self):
        in_dir  = self.le_in.text().strip()
        out_dir = self.le_out.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, "Batch", "Please choose a valid input directory.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, "Batch", "Please choose a valid output directory.")
            return

        exts = {".xisf", ".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in exts]
        if not files:
            QMessageBox.information(self, "Batch", "No acceptable image files found.")
            return

        self.log.clear()
        self.log.append(f"Found {len(files)} files. Starting batch…")
        QApplication.processEvents()

        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            out  = os.path.join(out_dir, base + "_plate_solved.fits")
            self.log.append(f"▶ {path}")
            QApplication.processEvents()

            try:
                image_data, original_header, bit_depth, is_mono = load_image(path)
                if image_data is None:
                    self.log.append("  ❌ Failed to load"); continue

                seed_h = _as_header(original_header) if isinstance(original_header, (dict, Header)) else None
                ok, res = _solve_numpy_with_astap(self, self.settings, image_data, seed_h)
                if not ok:
                    self.log.append(f"  ❌ {res}"); continue
                hdr: Header = res

                h2 = Header()
                for k in hdr.keys():
                    if k.upper() != "FILE_PATH":
                        h2[k] = hdr[k]

                save_image(
                    img_array=image_data,
                    filename=out,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=h2,
                    is_mono=is_mono
                )
                self.log.append("  ✔ saved: " + out)
            except Exception as e:
                self.log.append("  ❌ error: " + str(e))

            QApplication.processEvents()

        self.log.append("Batch plate solving completed.")
