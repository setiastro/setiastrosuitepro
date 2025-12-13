# pro/plate_solver.py
from __future__ import annotations

import os
import re
import math
import tempfile
from typing import Tuple, Dict, Any, Optional
from functools import lru_cache

import numpy as np
import json
import time
import requests
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS

from PyQt6.QtCore import QProcess, QTimer, QEventLoop, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QStackedWidget, QWidget, QMessageBox,
    QLineEdit, QTextEdit, QApplication, QProgressBar
)

# === our I/O & stretch — migrate from SASv2 ===
from legacy.image_manager import load_image, save_image   # <<<< IMPORTANT
try:
    from imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None


_NONFITS_META_KEYS = {
    "FILE_PATH",
    "FITS_HEADER",
    "BIT_DEPTH",
    "WCS_HEADER",
    "__HEADER_SNAPSHOT__",
    "ORIGINAL_HEADER",
    "PRE_SOLVE_HEADER",
}

def _strip_nonfits_meta_keys_from_header(h: Header | None) -> Header:
    """
    Return a copy of the header with all of our internal, non-FITS metadata
    keys removed. This prevents HIERARCH warnings and WCS failures on keys
    like FILE_PATH with very long values.
    """
    if not isinstance(h, Header):
        return Header()

    out = h.copy()
    for k in list(out.keys()):
        if k.upper() in _NONFITS_META_KEYS:
            try:
                out.remove(k)
            except Exception:
                pass
    return out

# --- Lightweight, modeless status popup for headless runs ---
_STATUS_POPUP = None  # module-level singleton

class _SolveStatusPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setObjectName("plate_solve_status_popup")
        self.setWindowTitle("Plate Solving")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        self.label = QLabel("Starting…", self)
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)  # indeterminate
        lay.addWidget(self.bar)

        row = QHBoxLayout()
        row.addStretch(1)
        hide_btn = QPushButton("Hide", self)
        hide_btn.clicked.connect(self.hide)
        row.addWidget(hide_btn)
        lay.addLayout(row)

    def update_text(self, text: str):
        self.label.setText(text or "")
        self.label.repaint()  # quick visual feedback
        QApplication.processEvents()

def _status_popup_open(parent, text: str = ""):
    """Show (or create) the singleton status popup for headless runs."""
    global _STATUS_POPUP
    if _STATUS_POPUP is None:
        host = parent if isinstance(parent, QWidget) else QApplication.activeWindow()
        _STATUS_POPUP = _SolveStatusPopup(host)
    if text:
        _STATUS_POPUP.update_text(text)
    _STATUS_POPUP.show()
    _STATUS_POPUP.raise_()
    QApplication.processEvents()
    return _STATUS_POPUP

def _status_popup_update(text: str):
    global _STATUS_POPUP
    if _STATUS_POPUP is not None:
        _STATUS_POPUP.update_text(text)

def _status_popup_close():
    """Hide (but do not destroy) the singleton status popup if it exists."""
    global _STATUS_POPUP
    try:
        if _STATUS_POPUP is not None:
            _STATUS_POPUP.hide()
            # keep instance for reuse (fast re-open)
    except Exception:
        # Completely safe to ignore; worst case the popup was already gone.
        pass

def _sleep_ui(ms: int):
    """Non-blocking sleep that keeps the UI responsive."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

def _with_events():
    """Yield to the UI event loop briefly."""
    QApplication.processEvents()

def _set_status_ui(parent, text: str):
    """
    Update dialog/main-window status or batch log; if neither exists (headless),
    show/update a small modeless popup. Always pumps events for responsiveness.
    """
    try:
        updated_any = False

        def _do():
            nonlocal updated_any
            target = None
            # Dialog status label?
            if hasattr(parent, "status") and isinstance(getattr(parent, "status"), QLabel):
                target = parent.status
            # Named child fallback
            if target is None and hasattr(parent, "findChild"):
                target = parent.findChild(QLabel, "status_label")
            if target is not None:
                target.setText(text)
                updated_any = True

            # Batch log?
            logw = getattr(parent, "log", None)
            if logw and hasattr(logw, "append"):
                if text and (text.startswith("Status:") or text.startswith("▶") or text.startswith("✔") or text.startswith("❌")):
                    logw.append(text)
                    updated_any = True

            # If we couldn't update any inline widget, use the headless popup.
            if not updated_any:
                _status_popup_open(parent, text)
            else:
                # If inline widgets exist and popup is visible, keep it quiet.
                _status_popup_update(text)

            QApplication.processEvents()

        if isinstance(parent, QWidget):
            QTimer.singleShot(0, _do)
        else:
            _do()
    except Exception:
        # Last-resort popup if even the above failed
        try:
            _status_popup_open(parent, text)
        except Exception:
            pass

def _wait_process(proc: QProcess, timeout_ms: int, parent=None) -> bool:
    """
    Incrementally wait for a QProcess while pumping UI events so the dialog stays responsive.
    Returns True if the process finished with NormalExit, else False.
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    step_ms = 100

    while time.monotonic() < deadline:
        if proc.state() == QProcess.ProcessState.NotRunning:
            break
        _sleep_ui(step_ms)

    if proc.state() != QProcess.ProcessState.NotRunning:
        # Timed out: try to stop the process cleanly, then force kill.
        try:
            proc.terminate()
            if not proc.waitForFinished(2000):
                proc.kill()
                proc.waitForFinished(2000)
        except Exception:
            pass
        _set_status_ui(parent, "Status: process timed out.")
        return False

    if proc.exitStatus() != QProcess.ExitStatus.NormalExit:
        _set_status_ui(parent, "Status: process did not exit normally.")
        return False

    return True

# --- astrometry.net config (web API) ---
ASTROMETRY_API_URL_DEFAULT = "https://nova.astrometry.net/api/"

def _get_astrometry_api_url(settings) -> str:
    return (settings.value("astrometry/server_url", "", type=str) or ASTROMETRY_API_URL_DEFAULT).rstrip("/") + "/"

def _get_solvefield_exe(settings) -> str:
    # Support both SASpro-style and legacy keys
    cand = [
        settings.value("paths/solve_field", "", type=str) or "",
        settings.value("astrometry/solvefield_path", "", type=str) or "",
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return cand[0]  # may be empty (used to decide web vs. local)

def _get_astrometry_api_key(settings) -> str:
    return settings.value("astrometry/api_key", "", type=str) or ""

def _set_astrometry_api_key(settings, key: str):
    settings.setValue("astrometry/api_key", key or "")

def _wcs_header_from_astrometry_calib(calib: dict, image_shape: tuple[int, ...]) -> Header:
    """
    calib: dict with keys 'ra','dec','pixscale'(arcsec/px),'orientation'(deg, +CCW).
    image_shape: (H, W) or (H, W, C). CRPIX is image center (1-based vs 0-based—astropy expects pixel coordinates in "fits" sense; mid-frame is fine).
    """
    H = int(image_shape[0]); W = int(image_shape[1])
    h = Header()
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = W / 2.0
    h["CRPIX2"] = H / 2.0
    h["CRVAL1"] = float(calib["ra"])
    h["CRVAL2"] = float(calib["dec"])
    scale_deg = float(calib["pixscale"]) / 3600.0  # deg/px
    theta = math.radians(float(calib.get("orientation", 0.0)))
    # note: same sign convention as your SASv2 builder
    h["CD1_1"] = -scale_deg * math.cos(theta)
    h["CD1_2"] =  scale_deg * math.sin(theta)
    h["CD2_1"] = -scale_deg * math.sin(theta)
    h["CD2_2"] = -scale_deg * math.cos(theta)
    h["RADECSYS"] = "ICRS"
    h["WCSAXES"] = 2
    return h

# If you already ship 'requests', this is simplest:

# ---- Seed controls (persisted in QSettings) ----
def _get_seed_mode(settings) -> str:
    # "auto" (from header), "manual" (use user values), "none" (blind)
    return (settings.value("astap/seed_mode", "auto", type=str) or "auto").lower()

def _set_seed_mode(settings, mode: str):
    settings.setValue("astap/seed_mode", (mode or "auto").lower())

def _get_manual_ra(settings) -> str:
    # store raw string so user can type hh:mm:ss or degrees
    return settings.value("astap/manual_ra", "", type=str) or ""

def _get_manual_dec(settings) -> str:
    return settings.value("astap/manual_dec", "", type=str) or ""

def _get_manual_scale(settings) -> float | None:
    try:
        v = settings.value("astap/manual_scale_arcsec", "", type=str)
        return float(v) if v not in (None, "",) else None
    except Exception:
        return None

@lru_cache(maxsize=256)
def _parse_ra_input_to_deg(s: str) -> float | None:
    """Parse RA input string to degrees. Cached for repeated lookups."""
    s = (s or "").strip()
    if not s: return None
    # allow plain degrees if > 24 or contains "deg"
    try:
        if re.search(r"[a-zA-Z]", s) is None and ":" not in s and " " not in s:
            x = float(s)
            return x if x > 24.0 else x * 15.0
    except Exception:
        pass
    parts = re.split(r"[:\s]+", s)
    try:
        if len(parts) >= 3:
            hh, mm, ss = float(parts[0]), float(parts[1]), float(parts[2])
        elif len(parts) == 2:
            hh, mm, ss = float(parts[0]), float(parts[1]), 0.0
        else:
            hh, mm, ss = float(parts[0]), 0.0, 0.0
        return (abs(hh) + mm/60.0 + ss/3600.0) * 15.0
    except Exception:
        return None

@lru_cache(maxsize=256)
def _parse_dec_input_to_deg(s: str) -> float | None:
    """Parse DEC input string to degrees. Cached for repeated lookups."""
    s = (s or "").strip()
    if not s: return None
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    parts = re.split(r"[:\s]+", s)
    try:
        if len(parts) >= 3:
            dd, mm, ss = float(parts[0]), float(parts[1]), float(parts[2])
        elif len(parts) == 2:
            dd, mm, ss = float(parts[0]), float(parts[1]), 0.0
        else:
            return sign * float(parts[0])
        return sign * (abs(dd) + mm/60.0 + ss/3600.0)
    except Exception:
        return None

def _set_manual_seed(settings, ra: str, dec: str, scale_arcsec: float | None):
    settings.setValue("astap/manual_ra", ra or "")
    settings.setValue("astap/manual_dec", dec or "")
    if scale_arcsec is None:
        settings.setValue("astap/manual_scale_arcsec", "")
    else:
        settings.setValue("astap/manual_scale_arcsec", str(float(scale_arcsec)))

def _astrometry_api_request(method: str, url: str, *, data=None, files=None,
                            timeout=(10, 60),
                            max_retries: int = 5,
                            parent=None,
                            stage: str = "") -> dict | None:
    """
    Robust request with retries, exponential backoff + jitter.
    """
    if requests is None:
        print("Requests not available for astrometry.net API.")
        return None

    import random
    import requests as _rq
    for attempt in range(1, max_retries + 1):
        try:
            if method.upper() == "POST":
                # ✅ IMPORTANT: rewind any file handles before each attempt,
                # because requests consumes them.
                if files:
                    try:
                        for v in files.values():
                            if hasattr(v, "seek"):
                                v.seek(0)
                    except Exception:
                        pass

                r = requests.post(url, data=data, files=files, timeout=timeout)
            else:
                r = requests.get(url, timeout=timeout)

            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None

            if r.status_code in (429, 500, 502, 503, 504):
                raise _rq.RequestException(f"HTTP {r.status_code}")
            else:
                print(f"Astrometry API HTTP {r.status_code} (no retry).")
                return None

        except (_rq.Timeout, _rq.ConnectionError, _rq.RequestException) as e:
            print(f"Astrometry API request error ({stage}): {e}")
            if attempt >= max_retries:
                break
            delay = min(8.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.2
            _set_status_ui(parent, f"Status: {stage or 'request'} retry {attempt}/{max_retries}…")
            _sleep_ui(int(delay * 1000))
    return None


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

            # 🚫 Never promote our internal metadata keys to FITS cards
            if K in _NONFITS_META_KEYS:
                continue

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
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

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

def _write_temp_fit_web_16bit(gray2d_unit: np.ndarray) -> str:
    """
    Write full-res mono FITS as 16-bit unsigned for web upload.
    gray2d_unit must be float32 in [0,1].
    Returns path to temp .fits.
    """
    import os
    import tempfile
    import numpy as np
    from astropy.io import fits
    from astropy.io.fits import Header

    if gray2d_unit.ndim != 2:
        raise ValueError("Expected 2-D grayscale array for web FITS.")

    g = np.clip(gray2d_unit.astype(np.float32), 0.0, 1.0)
    u16 = (g * 65535.0 + 0.5).astype(np.uint16)

    H, W = u16.shape
    hdr = Header()
    hdr["SIMPLE"] = True
    hdr["BITPIX"] = 16
    hdr["NAXIS"]  = 2
    hdr["NAXIS1"] = int(W)
    hdr["NAXIS2"] = int(H)
    hdr.add_comment("Temp FITS (16-bit) written for Astrometry.net upload.")

    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fits.PrimaryHDU(u16, header=hdr).writeto(tmp_path, overwrite=True, output_verify="silentfix")

    try:
        print(f"[tempfits-web] Saved 16-bit FITS to: {tmp_path} (size={os.path.getsize(tmp_path)} bytes)")
    except Exception:
        pass

    return tmp_path


def _astrometry_download_wcs_file(settings, job_id: int, parent=None) -> Header | None:
    """
    Download the solved WCS FITS from astrometry.net.
    This includes SIP terms when present.
    Returns fits.Header or None.
    """
    import os
    import tempfile
    from astropy.io import fits
    from astropy.io.fits import Header

    base_site = _get_astrometry_api_url(settings).split("/api/")[0].rstrip("/") + "/"
    url = base_site + f"wcs_file/{int(job_id)}"

    _set_status_ui(parent, "Status: Downloading WCS file (with SIP) from Astrometry.net…")
    try:
        r = requests.get(url, timeout=(10, 60))
        if r.status_code != 200 or len(r.content) < 2000:
            print(f"[Astrometry] WCS download failed HTTP {r.status_code}, bytes={len(r.content)}")
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wcs.fits", delete=False)
        tmp_path = tmp.name
        tmp.write(r.content)
        tmp.close()

        try:
            hdr = fits.getheader(tmp_path)
            h2 = Header()
            for k, v in dict(hdr).items():
                if k not in ("COMMENT", "HISTORY", "END"):
                    h2[k] = v
            return h2
        finally:
            try: os.remove(tmp_path)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    except Exception as e:
        print("[Astrometry] WCS download exception:", e)
        return None


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
    Guaranteed to return something usable even if stretch funcs fail.
    """
    f01 = _float01(img)

    # Mono
    if f01.ndim == 2 or (f01.ndim == 3 and f01.shape[2] == 1):
        if stretch_mono_image is not None:
            try:
                print("DEBUG stretching mono")
                out = stretch_mono_image(f01, 0.1, False)
                return np.clip(out.astype(np.float32), 0.0, 1.0)
            except Exception as e:
                print("DEBUG mono stretch failed, fallback:", e)
        return np.clip(f01.astype(np.float32), 0.0, 1.0)

    # Color
    if stretch_color_image is not None:
        try:
            print("DEBUG stretching color")
            out = stretch_color_image(f01, 0.1, False, False)
            return np.clip(out.astype(np.float32), 0.0, 1.0)
        except Exception as e:
            print("DEBUG color stretch failed, fallback:", e)

    return np.clip(f01.astype(np.float32), 0.0, 1.0)




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
    """
    Try to compute pixel scale from WCS / instrument metadata.
    If the result is obviously insane, return None so we can fall back
    to RA/Dec-only seeding.
    """
    def _sanity(val: float | None) -> float | None:
        if val is None or not np.isfinite(val) or val <= 0:
            return None
        # Typical imaging: ~0.1"–100"/px. Allow up to ~1000"/px for very wide,
        # but anything beyond that is almost certainly bogus.
        if val > 1000.0:
            return None
        return float(val)

    cd11 = _first_float(h.get("CD1_1"))
    cd21 = _first_float(h.get("CD2_1"))
    cdelt1 = _first_float(h.get("CDELT1"))
    cdelt2 = _first_float(h.get("CDELT2"))

    # 1) CD matrix
    if cd11 is not None or cd21 is not None:
        cd11 = cd11 or 0.0
        cd21 = cd21 or 0.0
        val = ((cd11**2 + cd21**2)**0.5) * 3600.0
        val = _sanity(val)
        if val is not None:
            return val

    # 2) CDELT
    if cdelt1 is not None or cdelt2 is not None:
        cdelt1 = cdelt1 or 0.0
        cdelt2 = cdelt2 or 0.0
        val = ((cdelt1**2 + cdelt2**2)**0.5) * 3600.0
        val = _sanity(val)
        if val is not None:
            return val

    # 3) Pixel size + focal length
    px_um_x = _first_float(h.get("XPIXSZ"))
    px_um_y = _first_float(h.get("YPIXSZ"))
    focal_mm = _first_float(h.get("FOCALLEN"))
    if focal_mm and (px_um_x or px_um_y):
        px_um = px_um_x if (px_um_x and not px_um_y) else px_um_y if (px_um_y and not px_um_x) else None
        if px_um is None:
            px_um = (px_um_x + px_um_y) / 2.0
        bx = _first_int(h.get("XBINNING")) or _first_int(h.get("XBIN")) or 1
        by = _first_int(h.get("YBINNING")) or _first_int(h.get("YBIN")) or 1
        bin_factor = (bx + by) / 2.0
        px_um_eff = px_um * bin_factor
        val = 206.264806 * px_um_eff / float(focal_mm)
        val = _sanity(val)
        if val is not None:
            return val

    return None


def _build_astap_seed_with_overrides(settings, header: Header | None, image: np.ndarray) -> tuple[list[str], str, float | None]:
    """
    Decide seed based on seed_mode:
      - auto: derive from header (existing logic)
      - manual: use user-provided RA/Dec/Scale
      - none: return [], "blind"
    Returns: (args, dbg, scale_arcsec)
    """
    mode = _get_seed_mode(settings)

    if mode == "none":
        return [], "seed disabled (blind)", None

    if mode == "manual":
        ra_s  = _get_manual_ra(settings)
        dec_s = _get_manual_dec(settings)
        scl   = _get_manual_scale(settings)
        ra_deg  = _parse_ra_input_to_deg(ra_s)
        dec_deg = _parse_dec_input_to_deg(dec_s)
        dbg = []
        if ra_deg is None:  dbg.append("RA?")
        if dec_deg is None: dbg.append("Dec?")
        if scl is None or not np.isfinite(scl) or scl <= 0: dbg.append("scale?")
        if dbg:
            return [], "manual seed invalid: " + ", ".join(dbg), None
        ra_h = ra_deg / 15.0
        spd  = dec_deg + 90.0
        args = ["-ra", f"{ra_h:.6f}", "-spd", f"{spd:.6f}", "-scale", f"{scl:.3f}"]
        return args, f"manual RA={ra_h:.6f}h | SPD={spd:.6f}° | scale={scl:.3f}\"/px", float(scl)

    # auto (default): from header
    if isinstance(header, Header):
        args, dbg = _build_astap_seed(header)
        scl = None
        if args:
            try:
                if "-scale" in args:
                    scl = float(args[args.index("-scale")+1])
            except Exception:
                scl = None
        return args, "auto: " + dbg, scl

    return [], "no header available for auto seed", None


def _build_astap_seed(h: Header) -> Tuple[list[str], str]:
    """
    Build ASTAP seed args from a header.
    RA/Dec are REQUIRED. Scale is OPTIONAL and sanity-checked.
    """
    dbg = []
    ra_deg  = _parse_ra_deg(h)
    dec_deg = _parse_dec_deg(h)

    if ra_deg is None:
        dbg.append("RA unknown")
    if dec_deg is None:
        dbg.append("Dec unknown")

    # If we don't have RA/Dec, there's nothing useful to seed.
    if ra_deg is None or dec_deg is None:
        return [], " / ".join(dbg) if dbg else "RA/Dec unknown"

    # Scale is now optional
    scale = _estimate_scale_arcsec_from_header(h)
    if scale is None:
        dbg.append("scale unknown")

    ra_h = ra_deg / 15.0
    spd  = dec_deg + 90.0

    args = ["-ra", f"{ra_h:.6f}", "-spd", f"{spd:.6f}"]
    if scale is not None:
        args += ["-scale", f"{scale:.3f}"]

    dbg_str = f"RA={ra_h:.6f} h | SPD={spd:.6f}°"
    if scale is not None:
        dbg_str += f" | scale={scale:.3f}\"/px"
    else:
        dbg_str += " | scale unknown"

    return args, dbg_str



def _astrometry_login(settings, parent=None) -> str | None:
    _set_status_ui(parent, "Status: Logging in to Astrometry.net…")
    api_key = _get_astrometry_api_key(settings)
    if not api_key:
        from PyQt6.QtWidgets import QInputDialog
        key, ok = QInputDialog.getText(None, "Astrometry.net API Key", "Enter your Astrometry.net API key:")
        if not ok or not key:
            _set_status_ui(parent, "Status: Login canceled (no API key).")
            return None
        _set_astrometry_api_key(settings, key)
        api_key = key

    base = _get_astrometry_api_url(settings)
    resp = _astrometry_api_request(
        "POST", base + "login",
        data={'request-json': json.dumps({"apikey": api_key})},
        parent=parent, stage="login"
    )
    if resp and resp.get("status") == "success":
        _set_status_ui(parent, "Status: Login successful.")
        return resp.get("session")
    _set_status_ui(parent, "Status: Login failed.")
    return None

def _astrometry_upload(settings, session: str, image_path: str, parent=None) -> int | None:
    _set_status_ui(parent, "Status: Uploading image to Astrometry.net…")
    base = _get_astrometry_api_url(settings)

    try:
        sz = os.path.getsize(image_path)
        if sz < 1024:  # fits headers alone are ~2880 bytes
            print(f"[Astrometry] temp FITS too small ({sz} bytes): {image_path}")
            _set_status_ui(parent, "Status: Upload failed (temp FITS empty).")
            return None
    except Exception:
        pass

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {'request-json': json.dumps({
                "publicly_visible": "y",
                "allow_modifications": "d",
                "session": session,
                "allow_commercial_use": "d"
            })}
            resp = _astrometry_api_request(
                "POST", base + "upload",
                data=data, files=files,
                timeout=(15, 180),
                parent=parent, stage="upload"
            )
        if resp and resp.get("status") == "success":
            _set_status_ui(parent, "Status: Upload complete.")
            return int(resp["subid"])
    except Exception as e:
        print("Upload error:", e)

    _set_status_ui(parent, "Status: Upload failed.")
    return None



def _solve_with_local_solvefield(parent, settings, tmp_fit_path: str) -> tuple[bool, Header | str]:
    solvefield = _get_solvefield_exe(settings)
    if not solvefield or not os.path.exists(solvefield):
        return False, "solve-field not configured."

    args = [
        "--overwrite",
        "--no-remove-lines",
        "--cpulimit", "300",
        "--downsample", "2",
        "--write-wcs", "wcs",
        tmp_fit_path
    ]
    _set_status_ui(parent, "Status: Running local solve-field…")
    print("Running solve-field:", solvefield, " ".join(args))
    p = QProcess(parent)
    p.start(solvefield, args)
    if not p.waitForStarted(5000):
        _set_status_ui(parent, "Status: solve-field failed to start.")
        return False, f"Failed to start solve-field: {p.errorString()}"

    if not _wait_process(p, 300000, parent=parent):
        _set_status_ui(parent, "Status: solve-field timed out.")
        return False, "solve-field timed out."

    if p.exitCode() != 0:
        out = bytes(p.readAllStandardOutput()).decode(errors="ignore")
        err = bytes(p.readAllStandardError()).decode(errors="ignore")
        _set_status_ui(parent, "Status: solve-field failed.")
        print("solve-field failed.\nSTDOUT:\n", out, "\nSTDERR:\n", err)
        return False, "solve-field returned non-zero exit."

    wcs_path = os.path.splitext(tmp_fit_path)[0] + ".wcs"
    new_path = os.path.splitext(tmp_fit_path)[0] + ".new"

    if os.path.exists(wcs_path):
        d = _parse_astap_wcs_file(wcs_path)
        if d:
            d = _ensure_ctypes(_coerce_wcs_numbers(d))
            return True, Header({k: v for k, v in d.items()})

    if os.path.exists(new_path):
        try:
            with fits.open(new_path, memmap=False) as hdul:
                h = Header()
                for k, v in dict(hdul[0].header).items():
                    if k not in ("COMMENT","HISTORY","END"):
                        h[k] = v
            return True, h
        except Exception as e:
            print("Failed reading .new FITS:", e)

    return False, "solve-field produced no WCS."


def _astrometry_poll_job(settings, subid: int, *, max_wait_s=900, parent=None) -> int | None:
    _set_status_ui(parent, "Status: Waiting for job assignment…")
    base = _get_astrometry_api_url(settings)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        resp = _astrometry_api_request("GET", base + f"submissions/{subid}",
                                       parent=parent, stage="poll job")
        if resp:
            jobs = resp.get("jobs", [])
            if jobs and jobs[0] is not None:
                _set_status_ui(parent, f"Status: Job assigned (ID {jobs[0]}).")
                try: return int(jobs[0])
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        _sleep_ui(1000)
    return None

def _astrometry_poll_calib(settings, job_id: int, *, max_wait_s=900, parent=None) -> dict | None:
    _set_status_ui(parent, "Status: Waiting for solution…")
    base = _get_astrometry_api_url(settings)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        resp = _astrometry_api_request("GET", base + f"jobs/{job_id}/calibration/",
                                       parent=parent, stage="poll calib")
        if resp and all(k in resp for k in ("ra","dec","pixscale")):
            _set_status_ui(parent, "Status: Solution received.")
            return resp
        _sleep_ui(1500)
    return None

# ---- ASTAP seed controls ----
# modes for radius: "auto" -> -r 0, "value" -> -r <user>, default "auto"
def _get_astap_radius_mode(settings) -> str:
    return (settings.value("astap/seed_radius_mode", "auto", type=str) or "auto").lower()

def _get_astap_radius_value(settings) -> float:
    try:
        return float(settings.value("astap/seed_radius_value", 5.0, type=float))
    except Exception:
        return 5.0

# modes for fov: "auto" -> -fov 0, "compute" -> use computed FOV, "value" -> user number; default "compute"
def _get_astap_fov_mode(settings) -> str:
    return (settings.value("astap/seed_fov_mode", "compute", type=str) or "compute").lower()

def _get_astap_fov_value(settings) -> float:
    try:
        return float(settings.value("astap/seed_fov_value", 0.0, type=float))
    except Exception:
        return 0.0


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
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
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

def _merge_wcs_into_base_header(base_header: Header | None, wcs_header: Header | None) -> Header:
    """
    Merge a WCS/SIP solution into a base acquisition header.

    - base_header: original FITS header with OBJECT, EXPTIME, GAIN, etc.
    - wcs_header:  header containing CRPIX/CRVAL/CD/SIP/etc. from ASTAP or Astrometry.

    Non-WCS cards in base_header are preserved.
    WCS/SIP/PLTSOLVD/etc. from wcs_header override any existing ones.
    """
    if not isinstance(base_header, Header):
        base_header = Header()
    # Always strip our internal meta keys from the acquisition header
    base_header = _strip_nonfits_meta_keys_from_header(base_header)

    if not isinstance(wcs_header, Header):
        # nothing special to merge; just normalize the base and return it.
        d0 = _ensure_ctypes(_coerce_wcs_numbers(dict(base_header)))
        out = Header()
        for k, v in d0.items():
            try:
                out[k] = v
            except Exception:
                pass
        return out

    # Start from a copy of the acquisition header (drop COMMENT/HISTORY from it)
    base = base_header.copy()


    # Start from a copy of the acquisition header (drop COMMENT/HISTORY from it)
    base = base_header.copy()
    for k in ("COMMENT", "HISTORY", "END"):
        if k in base:
            base.remove(k)

    merged = dict(base)

    # Only import *WCS-ish* keys from the solver, not things like BITPIX/NAXIS.
    wcs_prefixes = (
        "CRPIX", "CRVAL", "CDELT", "CD1_", "CD2_", "PC",
        "CTYPE", "CUNIT", "PV1_", "PV2_", "A_", "B_", "AP_", "BP_"
    )
    wcs_extras = {
        "WCSAXES", "LATPOLE", "LONPOLE", "EQUINOX",
        "PLTSOLVD", "WARNING", "RADESYS", "RADECSYS", "RADECSYS"
    }

    for key, val in wcs_header.items():
        ku = key.upper()
        if ku.startswith(wcs_prefixes) or ku in wcs_extras:
            merged[ku] = val

    # Coerce numeric types and ensure CTYPEs.
    merged = _ensure_ctypes(_coerce_wcs_numbers(merged))

    # Ensure TAN-SIP if SIP terms exist.
    try:
        sip_present = any(re.match(r"^(A|B|AP|BP)_\d+_\d+$", k) for k in merged.keys())
        if sip_present:
            c1 = str(merged.get("CTYPE1", "RA---TAN"))
            c2 = str(merged.get("CTYPE2", "DEC--TAN"))
            if not c1.endswith("-SIP"):
                merged["CTYPE1"] = "RA---TAN-SIP"
            if not c2.endswith("-SIP"):
                merged["CTYPE2"] = "DEC--TAN-SIP"
    except Exception:
        pass

    # CROTA from CD if missing.
    try:
        if ("CROTA1" not in merged or "CROTA2" not in merged) and \
           ("CD1_1" in merged and "CD1_2" in merged):
            rot = math.degrees(math.atan2(float(merged["CD1_2"]), float(merged["CD1_1"])))
            merged["CROTA1"] = rot
            merged["CROTA2"] = rot
    except Exception:
        pass

    out = Header()
    for k, v in merged.items():
        try:
            out[k] = v
        except Exception:
            # Skip weird/invalid keys silently
            pass
    return out


def _build_header_from_astap_outputs(
    tmp_fits: str,
    sidecar_wcs: Optional[str],
    base_header: Header | None
) -> Header:
    """
    Build final header as:  base_header (acquisition) + WCS/SIP from .wcs.
    """
    _debug_dump_header("ASTAP: BASE_HEADER ARG INTO _build_header_from_astap_outputs", base_header)
    """
    Build final header as:  base_header (acquisition) + WCS/SIP from .wcs.

    tmp_fits is only used as a last-resort source if base_header is None.
    """
    # 1) Determine base header (acquisition)
    if isinstance(base_header, Header):
        base_hdr = base_header
    else:
        # Fallback: read whatever ASTAP wrote into the temp FITS.
        base_dict: Dict[str, Any] = {}
        try:
            with fits.open(tmp_fits, memmap=False) as hdul:
                base_dict = dict(hdul[0].header)
            for k in ("COMMENT", "HISTORY", "END"):
                base_dict.pop(k, None)
        except Exception as e:
            print("Failed reading temp FITS header:", e)
        base_hdr = Header()
        for k, v in base_dict.items():
            try:
                base_hdr[k] = v
            except Exception:
                pass
    _debug_dump_header("ASTAP: BASE_HDR (acquisition header after fallback)", base_hdr)
    # 2) Load WCS from sidecar
    wcs_hdr = Header()
    if sidecar_wcs and os.path.exists(sidecar_wcs):
        try:
            wcs_dict = _parse_astap_wcs_file(sidecar_wcs)
            for k, v in wcs_dict.items():
                if k not in ("COMMENT", "HISTORY", "END"):
                    try:
                        wcs_hdr[k] = v
                    except Exception:
                        pass
        except Exception as e:
            print("Error parsing .wcs file:", e)
    _debug_dump_header("ASTAP: WCS_HDR FROM SIDECAR .WCS", wcs_hdr)
    # 3) Merge WCS into base acquisition header (base wins for non-WCS keys)
    final_hdr = _merge_wcs_into_base_header(base_hdr, wcs_hdr)

    _debug_dump_header("ASTAP: FINAL MERGED HEADER (base_hdr + wcs_hdr)", final_hdr)


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

def _solve_numpy_with_astrometry(
    parent,
    settings,
    image: np.ndarray,
    base_header: Header | None
) -> tuple[bool, Header | str]:
    """
    Try local solve-field first; if unavailable/failed, try astrometry.net web API.

    WEB MODE:
      - keep ORIGINAL dimensions (no downsample)
      - stretch to non-linear for star detectability
      - quantize to 16-bit unsigned FITS to reduce upload size
      - prefer solved WCS file from astrometry.net (includes SIP)
    """
    import os
    import numpy as np
    from astropy.io.fits import Header

    # Build full-res mono in [0,1], but NON-LINEAR (stretched) for detectability
    norm_full = _normalize_for_astap(image)                 # float32 [0,1], mono/color
    gray_full = _to_gray2d_unit(norm_full)                 # 2D float32 [0,1]
    Hfull, Wfull = int(gray_full.shape[0]), int(gray_full.shape[1])

    # Always write a full-res temp for LOCAL solve-field (float32)
    tmp_fit_full, _unused_sidecar = _write_temp_fit_via_save_image(gray_full, None)

    try:
        # 1) local solve-field path (full-res float FITS)
        ok, res = _solve_with_local_solvefield(parent, settings, tmp_fit_full)
        if ok:
            hdr = res if isinstance(res, Header) else None
            if hdr is not None:
                d = _ensure_ctypes(_coerce_wcs_numbers(dict(hdr)))
                if any(re.match(r"^(A|B|AP|BP)_\d+_\d+$", k) for k in d.keys()):
                    if not str(d.get("CTYPE1","RA---TAN")).endswith("-SIP"):
                        d["CTYPE1"] = "RA---TAN-SIP"
                    if not str(d.get("CTYPE2","DEC--TAN")).endswith("-SIP"):
                        d["CTYPE2"] = "DEC--TAN-SIP"
                hh = Header()
                for k, v in d.items():
                    try: hh[k] = v
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                return True, hh
            return False, "solve-field returned no header."

        # 2) web API fallback (full-res, 16-bit upload)
        if requests is None:
            return False, "requests not available for astrometry.net API."

        _set_status_ui(parent, "Status: Preparing full-res 16-bit FITS for web solve…")

        tmp_fit_web = _write_temp_fit_web_16bit(gray_full)

        # Verify web temp file isn't empty
        try:
            sz = os.path.getsize(tmp_fit_web)
            if sz < 3000:
                return False, f"Temp FITS for web upload is empty/tiny ({sz} bytes)."
        except Exception:
            pass

        session = _astrometry_login(settings, parent=parent)
        if not session:
            return False, "Astrometry.net login failed."

        subid = _astrometry_upload(settings, session, tmp_fit_web, parent=parent)
        if not subid:
            return False, "Astrometry.net upload failed."

        job_id = _astrometry_poll_job(settings, subid, parent=parent)
        if not job_id:
            return False, "Astrometry.net job ID not received in time."

        # Prefer full WCS file (includes SIP)
        hdr_wcs = _astrometry_download_wcs_file(settings, job_id, parent=parent)

        if hdr_wcs is None:
            # fallback to calibration (no SIP)
            calib = _astrometry_poll_calib(settings, job_id, parent=parent)
            if not calib:
                return False, "Astrometry.net calibration not received in time."

            _set_status_ui(parent, "Status: Building WCS header from calibration…")
            hdr_wcs = _wcs_header_from_astrometry_calib(calib, (Hfull, Wfull))

        # Coerce & ensure TAN-SIP if SIP terms exist
        d = _ensure_ctypes(_coerce_wcs_numbers(dict(hdr_wcs)))
        if any(re.match(r"^(A|B|AP|BP)_\d+_\d+$", k) for k in d.keys()):
            if not str(d.get("CTYPE1","RA---TAN")).endswith("-SIP"):
                d["CTYPE1"] = "RA---TAN-SIP"
            if not str(d.get("CTYPE2","DEC--TAN")).endswith("-SIP"):
                d["CTYPE2"] = "DEC--TAN-SIP"

        # Build a WCS-only Header from d
        wcs_hdr = Header()
        for k, v in d.items():
            try:
                wcs_hdr[k] = v
            except Exception:
                pass

        # Merge with acquisition header (base_header)
        merged = _merge_wcs_into_base_header(base_header, wcs_hdr)

        # clean temp web file ...
        try:
            if os.path.exists(tmp_fit_web):
                os.remove(tmp_fit_web)
        except Exception:
            pass

        return True, merged

    finally:
        # clean temp + solve-field byproducts next to tmp_fit_full
        try:
            base = os.path.splitext(tmp_fit_full)[0]
            for ext in (".fit",".fits",".wcs",".axy",".corr",".rdls",".solved",".new",".match",".ngc",".png",".ppm",".xyls"):
                p = base + ext
                if os.path.exists(p):
                    os.remove(p)
        except Exception:
            pass


def _solve_numpy_with_fallback(parent, settings, image: np.ndarray, seed_header: Header | None) -> tuple[bool, Header | str]:
    # Try ASTAP first
    _set_status_ui(parent, "Status: Solving with ASTAP…")
    ok, res = _solve_numpy_with_astap(parent, settings, image, seed_header)
    if ok:
        _set_status_ui(parent, "Status: Solved with ASTAP.")
        return True, res

    # ASTAP failed → tell the user and fall back
    err_msg = str(res) if res is not None else "unknown error"
    print("ASTAP failed:", err_msg)
    _set_status_ui(parent, f"Status: ASTAP failed ({err_msg}). Falling back to Astrometry.net…")
    QApplication.processEvents()

    # Fallback: astrometry.net (local solve-field first, then web API inside)
    ok2, res2 = _solve_numpy_with_astrometry(parent, settings, image, seed_header)
    if ok2:
        _set_status_ui(parent, "Status: Solved via Astrometry.net.")
    else:
        _set_status_ui(parent, f"Status: Astrometry.net failed ({res2}).")

    return ok2, res2


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

    # build a clean temp header (strip old WCS but KEEP acquisition keys)
    if isinstance(seed_header, Header):
        clean_for_temp = _strip_wcs_keys(seed_header)
        base_for_merge = clean_for_temp        # acquisition info lives here
        _debug_dump_header("ASTAP: CLEAN_FOR_TEMP (seed_header with WCS stripped)", clean_for_temp)
        _debug_dump_header("ASTAP: BASE_FOR_MERGE (acquisition header we expect to preserve)", base_for_merge)
    else:
        clean_for_temp = _minimal_header_for_gray2d(*gray.shape)
        base_for_merge = None
        _debug_dump_header("ASTAP: CLEAN_FOR_TEMP (minimal header, no seed)", clean_for_temp)

    tmp_fit, sidecar_wcs = _write_temp_fit_via_save_image(gray, clean_for_temp)
    print(f"[ASTAP] Temp FITS: {tmp_fit}, sidecar WCS: {sidecar_wcs}")

    # seed if possible; otherwise blind
    seed_args: list[str] = []
    scale_arcsec = None
    try:
        seed_args, dbg, scale_arcsec = _build_astap_seed_with_overrides(settings, seed_header, gray)
        if seed_args:
            # radius & fov modes (already implemented)
            radius_mode = _get_astap_radius_mode(settings)   # "auto" or "value"
            fov_mode    = _get_astap_fov_mode(settings)      # "auto", "compute", "value"

            # radius
            if radius_mode == "auto":
                r_arg = ["-r", "0"]     # ASTAP auto
                r_dbg = "r=auto(0)"
            else:
                r_val = max(0.0, float(_get_astap_radius_value(settings)))
                r_arg = ["-r", f"{r_val:.3f}"]
                r_dbg = f"r={r_val:.3f}°"

            # fov
            if fov_mode == "auto":
                fov_arg = ["-fov", "0"]
                f_dbg = "fov=auto(0)"
            elif fov_mode == "value":
                fv = max(0.0, float(_get_astap_fov_value(settings)))
                fov_arg = ["-fov", f"{fv:.4f}"]
                f_dbg = f"fov={fv:.4f}°"
            else:  # "compute"
                fv = _compute_fov_deg(gray, scale_arcsec) or 0.0
                fov_arg = ["-fov", f"{fv:.4f}"]
                f_dbg = f"fov(computed)={fv:.4f}°"

            seed_args = seed_args + r_arg + fov_arg
            print("ASTAP seed:", dbg, "|", r_dbg, "|", f_dbg)
        else:
            print("Seed disabled/invalid → blind:", dbg)
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
        _set_status_ui(parent, "Status: ASTAP failed to start.")
        return False, f"Failed to start ASTAP: {proc.errorString()}"

    _set_status_ui(parent, "Status: ASTAP solving…")
    if not _wait_process(proc, 300000, parent=parent):
        _set_status_ui(parent, "Status: ASTAP timed out.")
        return False, "ASTAP timed out."

    if proc.exitCode() != 0:
        out = bytes(proc.readAllStandardOutput()).decode(errors="ignore")
        err = bytes(proc.readAllStandardError()).decode(errors="ignore")
        print("ASTAP failed.\nSTDOUT:\n", out, "\nSTDERR:\n", err)
        try: os.remove(tmp_fit)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return False, "ASTAP returned a non-zero exit code."

    # >>> THIS is the key change: read the header **directly** from the FITS ASTAP wrote
    try:
        # Use acquisition header as base + WCS from .wcs
        hdr = _build_header_from_astap_outputs(tmp_fit, sidecar_wcs, base_for_merge)
    finally:
        try: os.remove(tmp_fit)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    # return a REAL fits.Header (no blobs/strings/dicts)
    return True, hdr



# ---------------------------------------------------------------------
# Solve active doc in-place
# ---------------------------------------------------------------------

# --- Debug helpers ---------------------------------------------------
DEBUG_PLATESOLVE_HEADERS = False  # set False to silence all header dumps


def _debug_dump_header(label: str, hdr: Header | None):
    """Print a full FITS Header to the console for debugging."""
    if not DEBUG_PLATESOLVE_HEADERS:
        return

    print(f"\n===== {label} =====")
    if hdr is None:
        print("  (None)")
    elif isinstance(hdr, Header):
        print(f"  (#cards = {len(hdr)})")
        for k, v in hdr.items():
            print(f"  {k:8s} = {v!r}")
    else:
        print(f"  (not a Header: {type(hdr)!r})")
    print("========================================\n")

def _debug_dump_meta(label: str, meta: dict):
    if not DEBUG_PLATESOLVE_HEADERS:
        return
    print(f"\n===== {label} (meta keys) =====")
    for k in sorted(meta.keys()):
        v = meta[k]
        print(f"  {k}: {type(v).__name__}")
    print("================================\n")



def plate_solve_doc_inplace(parent, doc, settings) -> Tuple[bool, Header | str]:
    img = getattr(doc, "image", None)
    if img is None:
        return False, "Active document has no image data."

    # Make sure metadata is a dict we can mutate
    meta = getattr(doc, "metadata", {}) or {}
    if not isinstance(meta, dict):
        try:
            meta = dict(meta)
        except Exception:
            meta = {}

    _debug_dump_meta("META BEFORE SOLVE", meta)
    _debug_dump_header("META['original_header'] BEFORE SOLVE", meta.get("original_header"))

    seed_h = _seed_header_from_meta(meta)
    _debug_dump_header("SEED HEADER FROM META (seed_h)", seed_h)

    # Keep a copy of acquisition header (no WCS) for merge
    # Prefer the true acquisition header if we have it, otherwise fall back.
    raw_acq = meta.get("original_header") or meta.get("fits_header")

    acq_base: Header | None = None
    if isinstance(raw_acq, Header):
        # Use the original acquisition header (OBJECT, EXPTIME, GAIN, etc.)
        acq_base = _strip_wcs_keys(raw_acq.copy())
        _debug_dump_header("ACQ_BASE (original/fits header with WCS stripped)", acq_base)
    elif isinstance(seed_h, Header):
        # Fallback: use the seed header as our acquisition base
        acq_base = _strip_wcs_keys(seed_h.copy())
        _debug_dump_header("ACQ_BASE (seed_h with WCS stripped)", acq_base)
    else:
        acq_base = None
        _debug_dump_header("ACQ_BASE (none available)", None)

    # Better debug: use our new scale estimator
    try:
        if isinstance(seed_h, Header):
            ra  = seed_h.get("CRVAL1", None)
            dec = seed_h.get("CRVAL2", None)
            scale = _estimate_scale_arcsec_from_header(seed_h)
            print(f"[PlateSolve seed] CRVAL1={ra}, CRVAL2={dec}, scale≈{scale} \"/px")
        else:
            print("[PlateSolve seed] No valid seed header available.")
    except Exception as e:
        print("Seed: debug print failed:", e)

    # Determine if we have inline status/log widgets; if not, show the popup.
    headless = not (
        (hasattr(parent, "status") and isinstance(getattr(parent, "status"), QLabel)) or
        (hasattr(parent, "log") and hasattr(getattr(parent, "log"), "append")) or
        (hasattr(parent, "findChild") and parent.findChild(QLabel, "status_label") is not None)
    )
    if headless:
        _status_popup_open(parent, "Status: Preparing plate solve…")

    try:
        ok, res = _solve_numpy_with_fallback(parent, settings, img, seed_h)
        if not ok:
            return False, res

        hdr: Header = res
        _debug_dump_header("SOLVER RAW HEADER (from _solve_numpy_with_fallback)", hdr)

        # Final header = acquisition + new WCS (solver)
        if isinstance(acq_base, Header) and isinstance(hdr, Header):
            hdr_final = _merge_wcs_into_base_header(acq_base, hdr)
        else:
            hdr_final = hdr if isinstance(hdr, Header) else Header()

        _debug_dump_header("FINAL MERGED HEADER (hdr_final)", hdr_final)
        # 🔹 NEW: stash pre-solve header ONCE so we never lose it
        try:
            if "original_header" in meta and "pre_solve_header" not in meta:
                old = meta["original_header"]
                if isinstance(old, Header):
                    meta["pre_solve_header"] = old.copy()
        except Exception as e:
            print("plate_solve_doc_inplace: failed to stash pre_solve_header:", e)

        # 🔹 Ensure doc.metadata is our updated dict
        doc.metadata = meta

        # Store merged header as the current "original_header"
        doc.metadata["original_header"] = hdr_final
        _debug_dump_header("DOC.METADATA['original_header'] AFTER SOLVE", doc.metadata.get("original_header"))


        # Build WCS object from the same header we just stored
        try:
            wcs_obj = WCS(hdr_final)
            doc.metadata["wcs"] = wcs_obj
        except Exception as e:
            print("WCS build FAILED:", e)

        # Notify UI
        if hasattr(doc, "changed"):
            try: doc.changed.emit()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        if hasattr(parent, "header_viewer") and hasattr(parent.header_viewer, "set_document"):
            QTimer.singleShot(0, lambda: parent.header_viewer.set_document(doc))
        if hasattr(parent, "_refresh_header_viewer"):
            QTimer.singleShot(0, lambda: parent._refresh_header_viewer(doc))
        if hasattr(parent, "currentDocumentChanged"):
            QTimer.singleShot(0, lambda: parent.currentDocumentChanged.emit(doc))

        _set_status_ui(parent, "Status: Plate solve completed.")
        _status_popup_close()
        return True, hdr
    finally:
        _status_popup_close()



def _estimate_scale_arcsec_from_header(hdr: Header) -> float | None:
    """
    Estimate pixel scale in arcsec/pixel from a FITS Header.
    Tries WCS, then CD matrix, then PC*CDELT, then PIXSCALE-style keys.
    Returns None if we can't get a sane value.
    """
    # Always work on a copy with our internal meta keys stripped
    hdr = _strip_nonfits_meta_keys_from_header(hdr)

    # 1) Try astropy WCS, which handles CD vs PC*CDELT automatically
    try:
        w = WCS(hdr)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales_deg = proj_plane_pixel_scales(w)  # degrees/pixel
        if scales_deg is not None and len(scales_deg) >= 2:
            s_deg = float(np.mean(scales_deg[:2]))
            scale = s_deg * 3600.0  # arcsec/pixel
            if 0 < scale < 10000:
                return scale
    except Exception as e:
        print("Seed: WCS->scale via proj_plane_pixel_scales failed:", e)

    # 2) Try CD matrix directly
    cd11 = hdr.get("CD1_1")
    cd21 = hdr.get("CD2_1")
    try:
        if cd11 is not None or cd21 is not None:
            cd11 = float(cd11 or 0.0)
            cd21 = float(cd21 or 0.0)
            s_deg = (cd11 * cd11 + cd21 * cd21) ** 0.5
            scale = s_deg * 3600.0
            if 0 < scale < 10000:
                return scale
    except Exception as e:
        print("Seed: CD-based scale failed:", e)

    # 3) Try PC * CDELT fallback
    try:
        cdelt1 = hdr.get("CDELT1")
        cdelt2 = hdr.get("CDELT2")
        pc11   = hdr.get("PC1_1")
        pc21   = hdr.get("PC2_1")
        if cdelt1 is not None and pc11 is not None:
            cd11 = float(cdelt1) * float(pc11)
        else:
            cd11 = None
        if cdelt2 is not None and pc21 is not None:
            cd21 = float(cdelt2) * float(pc21)
        else:
            cd21 = None

        if cd11 is not None or cd21 is not None:
            s_deg = ( (cd11 or 0.0)**2 + (cd21 or 0.0)**2 ) ** 0.5
            scale = s_deg * 3600.0
            if 0 < scale < 10000:
                return scale
    except Exception as e:
        print("Seed: PC*CDELT-based scale failed:", e)

    # 4) Fallback on explicit pixscale-like keywords, if present
    for key in ("PIXSCALE", "SECPIX"):
        if key in hdr:
            try:
                scale = float(hdr[key])
                if 0 < scale < 10000:
                    return scale
            except Exception:
                pass

    # If we get here, we couldn't find a sane scale
    return None

def _seed_header_from_meta(meta: dict) -> Header:
    """
    Build the header used for ASTAP seeding from doc.metadata.

    Priority:
      1. original_header (if present)
      2. meta as a dict
    Then merge in any WCS info from:
      - meta['wcs_header'] (Header or string)
      - meta['wcs'] (WCS object)
    """
    # Base: original FITS header if present, otherwise treat meta dict as header
    base_src = meta.get("original_header") or meta.get("fits_header") or meta
    base = _as_header(base_src)

    wcs_hdr: Header | None = None

    # 1) Use explicit wcs_header if present
    raw_wcs = meta.get("wcs_header")
    if isinstance(raw_wcs, Header):
        wcs_hdr = raw_wcs
    elif isinstance(raw_wcs, str):
        # This is your case: stored as Header.tostring()
        try:
            # In real metadata this likely has newlines; sep='\n' handles that.
            wcs_hdr = fits.Header.fromstring(raw_wcs, sep='\n')
        except Exception as e:
            print("Seed: failed to parse wcs_header string:", e)

    # 2) Fallback: derive from WCS object if we still don't have a header
    if wcs_hdr is None:
        wcs_obj = meta.get("wcs")
        if isinstance(wcs_obj, WCS):
            try:
                wcs_hdr = wcs_obj.to_header(relax=True)
            except Exception as e:
                print("Seed: failed to derive WCS header from WCS object:", e)

    # 3) Merge WCS header into base header, with WCS keys winning
    if wcs_hdr is not None:
        if not isinstance(base, Header):
            base = Header()
        else:
            base = base.copy()
        for k, v in wcs_hdr.items():
            try:
                base[k] = v
            except Exception:
                pass

    return _strip_nonfits_meta_keys_from_header(base)


def _compute_fov_deg(image: np.ndarray, arcsec_per_px: float | None) -> float | None:
    if arcsec_per_px is None or not np.isfinite(arcsec_per_px) or arcsec_per_px <= 0:
        return None
    H = int(image.shape[0]) if image.ndim >= 2 else 0
    if H <= 0:
        return None
    return (H * arcsec_per_px) / 3600.0  # vertical FOV in degrees

def plate_solve_active_document(parent, settings) -> tuple[bool, Header | str]:
    """
    Convenience wrapper:
      - Finds the active document from the given parent (main window, ImagePeeker, etc.)
      - Calls plate_solve_doc_inplace(...)
    
    Returns (ok, Header | error_message).
    """
    doc = _active_doc_from_parent(parent)
    if doc is None:
        return False, "No active document to plate-solve."

    return plate_solve_doc_inplace(parent, doc, settings)

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
        self.setMinimumWidth(560)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        # ---------------- Main containers ----------------
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        # ---- Top row: Mode selector ----
        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:", self))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Active View", "File", "Batch"])
        top.addWidget(self.mode_combo, 1)
        top.addStretch(1)
        main.addLayout(top)

        # ---- Seeding group (shared) ----
        from PyQt6.QtWidgets import QGroupBox, QFormLayout
        seed_box = QGroupBox("Seeding & Constraints", self)
        seed_form = QFormLayout(seed_box)
        seed_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        seed_form.setHorizontalSpacing(8)
        seed_form.setVerticalSpacing(6)

        # Seed mode
        self.cb_seed_mode = QComboBox(seed_box)
        self.cb_seed_mode.addItems(["Auto (from header)", "Manual", "None (blind)"])
        seed_form.addRow("Seed mode:", self.cb_seed_mode)

        # Manual RA/Dec/Scale row
        manual_row = QHBoxLayout()
        self.le_ra = QLineEdit(seed_box);   self.le_ra.setPlaceholderText("RA (e.g. 22:32:14 or 338.1385)")
        self.le_dec = QLineEdit(seed_box);  self.le_dec.setPlaceholderText("Dec (e.g. +40:42:43 or 40.7123)")
        self.le_scale = QLineEdit(seed_box); self.le_scale.setPlaceholderText('Scale [" / px] (e.g. 1.46)')
        manual_row.addWidget(self.le_ra, 1)
        manual_row.addWidget(self.le_dec, 1)
        manual_row.addWidget(self.le_scale, 1)
        seed_form.addRow("Manual RA/Dec/Scale:", manual_row)

        # Search radius (-r)
        rad_row = QHBoxLayout()
        self.cb_radius_mode = QComboBox(seed_box)
        self.cb_radius_mode.addItems(["Auto (-r 0)", "Value (deg)"])
        self.le_radius_val = QLineEdit(seed_box); self.le_radius_val.setPlaceholderText("e.g. 5.0")
        self.le_radius_val.setFixedWidth(120)
        rad_row.addWidget(self.cb_radius_mode)
        rad_row.addWidget(self.le_radius_val)
        rad_row.addStretch(1)
        seed_form.addRow("Search radius:", rad_row)

        # FOV (-fov)
        fov_row = QHBoxLayout()
        self.cb_fov_mode = QComboBox(seed_box)
        self.cb_fov_mode.addItems(["Compute from scale", "Auto (-fov 0)", "Value (deg)"])
        self.le_fov_val = QLineEdit(seed_box); self.le_fov_val.setPlaceholderText("e.g. 1.80")
        self.le_fov_val.setFixedWidth(120)
        fov_row.addWidget(self.cb_fov_mode)
        fov_row.addWidget(self.le_fov_val)
        fov_row.addStretch(1)
        seed_form.addRow("FOV:", fov_row)

        # Tooltips
        self.cb_seed_mode.setToolTip("Use FITS header, your manual RA/Dec/scale, or blind solve.")
        self.le_scale.setToolTip('Pixel scale in arcseconds/pixel (e.g., 1.46).')
        self.cb_radius_mode.setToolTip("ASTAP -r. Auto lets ASTAP choose; Value forces a cone radius.")
        self.cb_fov_mode.setToolTip("ASTAP -fov. Compute uses image height × scale; Auto lets ASTAP infer.")

        main.addWidget(seed_box)

        # ---------------- Stacked pages ----------------
        self.stack = QStackedWidget(self)
        main.addWidget(self.stack, 1)

        # Page 0: Active View
        p0 = QWidget(self); l0 = QVBoxLayout(p0)
        l0.addWidget(QLabel("Solve the currently active image view.", p0))
        l0.addStretch(1)
        self.stack.addWidget(p0)

        # Page 1: File picker
        p1 = QWidget(self); l1 = QVBoxLayout(p1)
        file_row = QHBoxLayout()
        self.le_path = QLineEdit(p1); self.le_path.setPlaceholderText("Choose an image…")
        btn_browse = QPushButton("Browse…", p1)
        file_row.addWidget(self.le_path, 1); file_row.addWidget(btn_browse)
        l1.addLayout(file_row); l1.addStretch(1)
        self.stack.addWidget(p1)

        # Page 2: Batch
        p2 = QWidget(self); l2 = QVBoxLayout(p2)
        in_row  = QHBoxLayout(); out_row = QHBoxLayout()
        self.le_in  = QLineEdit(p2);  self.le_in.setPlaceholderText("Input directory")
        self.le_out = QLineEdit(p2);  self.le_out.setPlaceholderText("Output directory")
        b_in  = QPushButton("Browse Input…", p2)
        b_out = QPushButton("Browse Output…", p2)
        in_row.addWidget(self.le_in, 1);   in_row.addWidget(b_in)
        out_row.addWidget(self.le_out, 1); out_row.addWidget(b_out)
        self.log = QTextEdit(p2); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        l2.addLayout(in_row); l2.addLayout(out_row); l2.addWidget(QLabel("Status:", p2)); l2.addWidget(self.log, 1)
        self.stack.addWidget(p2)

        # ---------------- Status + buttons ----------------
        self.status = QLabel("", self)
        self.status.setMinimumHeight(20)
        main.addWidget(self.status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_go = QPushButton("Start", self)
        self.btn_close = QPushButton("Close", self)
        btn_row.addWidget(self.btn_go)
        btn_row.addWidget(self.btn_close)
        main.addLayout(btn_row)

        # ---------------- Connections ----------------
        self.mode_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        btn_browse.clicked.connect(self._browse_file)
        b_in.clicked.connect(self._browse_in)
        b_out.clicked.connect(self._browse_out)
        self.btn_go.clicked.connect(self._run)
        self.btn_close.clicked.connect(self.close)

        # ---------------- Load settings & init UI ----------------
        mode_map = {"auto": 0, "manual": 1, "none": 2}
        self.cb_seed_mode.setCurrentIndex(mode_map.get(_get_seed_mode(self.settings), 0))
        self.le_ra.setText(_get_manual_ra(self.settings))
        self.le_dec.setText(_get_manual_dec(self.settings))
        scl = _get_manual_scale(self.settings)
        self.le_scale.setText("" if scl is None else str(scl))

        self.cb_radius_mode.setCurrentIndex(0 if _get_astap_radius_mode(self.settings) == "auto" else 1)
        self.le_radius_val.setText(str(_get_astap_radius_value(self.settings)))

        fov_mode = _get_astap_fov_mode(self.settings)
        self.cb_fov_mode.setCurrentIndex(1 if fov_mode == "auto" else (2 if fov_mode == "value" else 0))
        self.le_fov_val.setText(str(_get_astap_fov_value(self.settings)))

        def _update_visibility():
            manual = (self.cb_seed_mode.currentIndex() == 1)
            self.le_ra.setEnabled(manual)
            self.le_dec.setEnabled(manual)
            self.le_scale.setEnabled(manual)
            self.le_radius_val.setEnabled(self.cb_radius_mode.currentIndex() == 1)
            self.le_fov_val.setEnabled(self.cb_fov_mode.currentIndex() == 2)

        self.cb_seed_mode.currentIndexChanged.connect(_update_visibility)
        self.cb_radius_mode.currentIndexChanged.connect(_update_visibility)
        self.cb_fov_mode.currentIndexChanged.connect(_update_visibility)
        _update_visibility()

        if icon:
            self.setWindowIcon(icon)

        self.status.setObjectName("status_label")
        # if batch page exists:
        self.log.setObjectName("batch_log")

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

        idx = self.cb_seed_mode.currentIndex()
        _set_seed_mode(self.settings, "auto" if idx == 0 else ("manual" if idx == 1 else "none"))
        # manual values
        try:
            manual_scale = float(self.le_scale.text().strip()) if self.le_scale.text().strip() else None
        except Exception:
            manual_scale = None
        _set_manual_seed(self.settings, self.le_ra.text().strip(), self.le_dec.text().strip(), manual_scale)
        # radius
        self.settings.setValue("astap/seed_radius_mode", "auto" if self.cb_radius_mode.currentIndex()==0 else "value")
        try:
            self.settings.setValue("astap/seed_radius_value", float(self.le_radius_val.text().strip()))
        except Exception:
            pass
        # fov
        self.settings.setValue("astap/seed_fov_mode",
                               "compute" if self.cb_fov_mode.currentIndex()==0 else ("auto" if self.cb_fov_mode.currentIndex()==1 else "value"))
        try:
            self.settings.setValue("astap/seed_fov_value", float(self.le_fov_val.text().strip()))
        except Exception:
            pass

        mode = self.stack.currentIndex()
        if mode == 0:
            # Active view
            doc = _active_doc_from_parent(self.parent())
            if not doc:
                QMessageBox.information(self, "Plate Solver", "No active image view.")
                return
            ok, res = plate_solve_doc_inplace(self, doc, self.settings)
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

        # Seed header from original_header
        seed_h = _as_header(original_header) if isinstance(original_header, (dict, Header)) else None

        # Acquisition base for final merge (strip old WCS)
        acq_base: Header | None = None
        if isinstance(seed_h, Header):
            acq_base = _strip_wcs_keys(seed_h)

        # Solve
        ok, res = _solve_numpy_with_fallback(self, self.settings, image_data, seed_h)
        if not ok:
            self.status.setText(str(res))
            return
        solver_hdr: Header = res

        # Merge solver WCS into acquisition header
        if isinstance(acq_base, Header) and isinstance(solver_hdr, Header):
            hdr_final = _merge_wcs_into_base_header(acq_base, solver_hdr)
        else:
            hdr_final = solver_hdr if isinstance(solver_hdr, Header) else Header()

        # Save-as using legacy.save_image() with ORIGINAL pixels (not normalized)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plate-Solved FITS",
            "",
            "FITS files (*.fits *.fit)"
        )
        if save_path:
            try:
                # never persist 'file_path' inside FITS
                h2 = Header()
                for k in hdr_final.keys():
                    if k.upper() != "FILE_PATH":
                        h2[k] = hdr_final[k]

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
        files = [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
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
                # Load using legacy.load_image()
                image_data, original_header, bit_depth, is_mono = load_image(path)
                if image_data is None:
                    self.log.append("  ❌ Failed to load")
                    continue

                # Seed header from original_header
                seed_h = _as_header(original_header) if isinstance(original_header, (dict, Header)) else None

                # Acquisition base for final merge (strip old WCS)
                acq_base: Header | None = None
                if isinstance(seed_h, Header):
                    acq_base = _strip_wcs_keys(seed_h)

                # Solve
                ok, res = _solve_numpy_with_fallback(self, self.settings, image_data, seed_h)
                if not ok:
                    self.log.append(f"  ❌ {res}")
                    continue
                hdr: Header = res

                # Merge solver WCS into acquisition header
                if isinstance(acq_base, Header) and isinstance(hdr, Header):
                    hdr_final = _merge_wcs_into_base_header(acq_base, hdr)
                else:
                    hdr_final = hdr if isinstance(hdr, Header) else Header()

                # Build header to save (and strip FILE_PATH)
                h2 = Header()
                for k in hdr_final.keys():
                    if k.upper() != "FILE_PATH":
                        h2[k] = hdr_final[k]

                # Save using original pixels
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

