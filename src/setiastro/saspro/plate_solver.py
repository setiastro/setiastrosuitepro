# src/setiastro/saspro/plate_solver.py
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
from astropy.io import fits
from astropy.io.fits import Header
from astropy.wcs import WCS

from PyQt6.QtCore import QProcess, QTimer, QEventLoop, Qt, QCoreApplication, QSettings, pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QComboBox, QStackedWidget, QWidget, QMessageBox, QFormLayout,
    QLineEdit, QTextEdit, QApplication, QProgressBar, QListWidget, QListWidgetItem,
    QCheckBox,
)

# === our I/O & stretch — migrate from SASv2 ===
from setiastro.saspro.legacy.image_manager import load_image, save_image   # <<<< IMPORTANT
try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None
def make_help_button(*args, **kwargs):
    """Deferred shim for help_support.make_help_button.

    help_support -> shortcuts -> the command registry -> the tool/widget
    modules -> matplotlib. This module is imported by the plate-solve CLI,
    which never builds a dialog and must not pay for any of that. The only
    caller is a dialog button row, which is GUI-only by definition.
    """
    from setiastro.saspro.help_support import make_help_button as _real
    return _real(*args, **kwargs)
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
        self.setWindowTitle(self.tr("Plate Solving"))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        self.label = QLabel(self.tr("Starting…"), self)
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)  # indeterminate
        lay.addWidget(self.bar)

        row = QHBoxLayout()
        row.addStretch(1)
        hide_btn = QPushButton(self.tr("Hide"), self)
        hide_btn.clicked.connect(self.hide)
        row.addWidget(hide_btn)
        lay.addLayout(row)

    def update_text(self, text: str):
        self.label.setText(text or "")
        self.label.repaint()  # quick visual feedback
        QApplication.processEvents()

# A scripted or CLI run has nowhere to put a window and nobody watching it.
# Without this the solver opens a real floating dialog on the user's desktop
# part-way through a batch, which is both startling and, on a headless box,
# a crash waiting to happen.
SUPPRESS_STATUS_POPUP = False


def set_status_popup_enabled(on: bool) -> None:
    """Turn the floating solve-status popup on or off process-wide."""
    global SUPPRESS_STATUS_POPUP
    SUPPRESS_STATUS_POPUP = not bool(on)


def _status_popup_open(parent, text: str = ""):
    """Show (or create) the singleton status popup for headless runs."""
    global _STATUS_POPUP
    if SUPPRESS_STATUS_POPUP:
        # Suppressed must not mean discarded. This popup is the ONLY sink for
        # status text when there is no status widget -- which is every CLI and
        # scripted run -- so silently dropping it loses the blind solver's
        # whole narration: stars detected, layers tried, where the quad index
        # placed the field, and on what log-odds. Print it instead.
        if text:
            try:
                print(text, flush=True)
            except Exception:
                pass
        return None
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
    global _STATUS_POPUP
    if _STATUS_POPUP is None:
        return
    try:
        _STATUS_POPUP.hide()
    except Exception:
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
    try:
        updated_any = False

        target = None
        if hasattr(parent, "status") and isinstance(getattr(parent, "status"), QLabel):
            target = parent.status
        if target is None and hasattr(parent, "findChild"):
            target = parent.findChild(QLabel, "status_label")
        if target is not None:
            target.setText(text)
            updated_any = True

        logw = getattr(parent, "log", None)
        if logw and hasattr(logw, "append"):
            tr_status = QCoreApplication.translate("PlateSolver", "Status:")
            if text and (text.startswith("Status:") or text.startswith(tr_status) or text.startswith("▶") or text.startswith("✔") or text.startswith("❌")):
                logw.append(text)
                updated_any = True

        if not updated_any:
            _status_popup_open(parent, text)
        else:
            _status_popup_update(text)

        QApplication.processEvents()
    except Exception:
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
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: process timed out."))
        return False

    if proc.exitStatus() != QProcess.ExitStatus.NormalExit:
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: process did not exit normally."))
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
    """
    Canonical key: 'api/astrometry_key' (matches SettingsDialog).
    Also check older legacy keys for backward compatibility.
    """
    if settings is None:
        return ""

    # ✅ canonical
    key = settings.value("api/astrometry_key", "", type=str) or ""
    key = key.strip()
    if key:
        return key

    # 🔁 legacy fallbacks (if you ever stored them differently)
    for k in (
        "api/astrometry",          # old guess
        "astrometry/api_key",
        "astrometry/key",
        "astrometry_key",
        "plate_solver/astrometry_key",
    ):
        v = settings.value(k, "", type=str) or ""
        v = v.strip()
        if v:
            # migrate forward so it works next time
            settings.setValue("api/astrometry_key", v)
            try:
                settings.remove(k)
            except Exception:
                pass
            try:
                settings.sync()
            except Exception:
                pass
            return v

    return ""


def _set_astrometry_api_key(settings, key: str) -> None:
    if settings is None:
        return
    settings.setValue("api/astrometry_key", (key or "").strip())
    try:
        settings.sync()
    except Exception:
        pass

def _wcs_header_from_astrometry_calib(calib: dict, image_shape: tuple[int, ...]) -> Header:
    """
    calib: dict with keys 'ra','dec','pixscale'(arcsec/px),'orientation'(deg, +CCW from North).
           Also uses 'parity' (1.0 = flipped/mirrored, -1.0 = normal) when present.
    image_shape: (H, W) or (H, W, C).
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

    # parity: Astrometry.net returns 1.0 when the solution required
    # an E/W flip (i.e. the image is mirrored). -1.0 = normal orientation.
    # When parity=1 we must negate the RA axis to un-mirror.
    parity = float(calib.get("parity", -1.0))  # default -1.0 = normal
    parity_sign = -1.0 if parity > 0 else 1.0   # flip RA axis when mirrored

    (h["CD1_1"], h["CD1_2"]), (h["CD2_1"], h["CD2_2"]) = _cd_matrix(
        scale_deg, theta, mirrored=(parity_sign < 0))
    h["CROTA2"] = float(calib.get("orientation", 0.0))

    h["RADECSYS"] = "ICRS"
    h["WCSAXES"]  = 2
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
    import requests  # deferred: ~138 ms, and a local solve never touches the network
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
            msg = QCoreApplication.translate("PlateSolver", "Status: {0} retry {1}/{2}…").format(stage or 'request', attempt, max_retries)
            _set_status_ui(parent, msg)
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
        return h if len(h) else None

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
    import requests  # deferred: ~138 ms, and a local solve never touches the network
    import os
    import tempfile
    from astropy.io import fits
    from astropy.io.fits import Header

    base_site = _get_astrometry_api_url(settings).split("/api/")[0].rstrip("/") + "/"
    url = base_site + f"wcs_file/{int(job_id)}"

    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Downloading WCS file (with SIP) from Astrometry.net…"))
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
    Normalize image to [0,1] float32 and apply stretch for star visibility.
    For integer arrays: divide by dtype max.
    For float arrays: min-max normalize (handles raw ADU values outside [0,1]).
    Returns float32 in [0,1], 2D for mono or 3D for color.
    """
    a = np.asarray(img, dtype=np.float32)

    # Normalize to [0,1] — handle both int-origin and float-ADU inputs
    if img.dtype.kind in "ui":
        info = np.iinfo(img.dtype)
        if info.max > 0:
            a = a / float(info.max)
    else:
        # Float array — may be raw ADU (e.g. 772–12913), not [0,1]
        mn, mx = float(a.min()), float(a.max())
        if mx > mn:
            a = (a - mn) / (mx - mn)
        else:
            a = np.zeros_like(a)

    a = np.clip(a, 0.0, 1.0)

    # Mono
    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
        if stretch_mono_image is not None:
            try:
                out = stretch_mono_image(a, 0.1, False, no_black_clip=True)
                return np.clip(out.astype(np.float32), 0.0, 1.0)
            except Exception as e:
                print("DEBUG mono stretch failed, fallback:", e)
        return a.astype(np.float32)

    # Color
    if stretch_color_image is not None:
        try:
            out = stretch_color_image(a, 0.1, False, False, no_black_clip=True)
            return np.clip(out.astype(np.float32), 0.0, 1.0)
        except Exception as e:
            print("DEBUG color stretch failed, fallback:", e)

    return a.astype(np.float32)

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
    if ra is not None:
        if 0.0 <= ra < 360.0: return ra
        if 0.0 <= ra <= 24.0: return ra * 15.0  # stored in hours
    for key in ("OBJCTRA", "RA", "TELRA", "MNTRA", "CENTERRA", "RA_OBJ", "ALPHA_J2000"):
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
    for key in ("OBJCTDEC", "DEC", "TELDEC", "MNTDEC", "CENTERDEC", "DEC_OBJ", "DELTA_J2000"):
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
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Logging in to Astrometry.net…"))
    api_key = _get_astrometry_api_key(settings)
    if not api_key:
        from PyQt6.QtWidgets import QInputDialog
        key, ok = QInputDialog.getText(None, QCoreApplication.translate("PlateSolver", "Astrometry.net API Key"), QCoreApplication.translate("PlateSolver", "Enter your Astrometry.net API key:"))
        if not ok or not key:
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Login canceled (no API key)."))
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
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Login successful."))
        return resp.get("session")
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Login failed."))
    return None

def _astrometry_upload(settings, session: str, image_path: str, parent=None) -> int | None:
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Uploading image to Astrometry.net…"))
    base = _get_astrometry_api_url(settings)

    try:
        sz = os.path.getsize(image_path)
        if sz < 1024:  # fits headers alone are ~2880 bytes
            print(f"[Astrometry] temp FITS too small ({sz} bytes): {image_path}")
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Upload failed (temp FITS empty)."))
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
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Upload complete."))
            return int(resp["subid"])
    except Exception as e:
        print("Upload error:", e)

    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Upload failed."))
    return None



def _solve_with_local_solvefield(parent, settings, tmp_fit_path: str) -> tuple[bool, Header | str]:
    solvefield = _get_solvefield_exe(settings)
    if not solvefield or not os.path.exists(solvefield):
        return False, QCoreApplication.translate("PlateSolver", "solve-field not configured.")

    args = [
        "--overwrite",
        "--no-remove-lines",
        "--cpulimit", "300",
        "--downsample", "2",
        "--write-wcs", "wcs",
        tmp_fit_path
    ]
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Running local solve-field…"))
    print("Running solve-field:", solvefield, " ".join(args))
    p = QProcess(parent)
    p.start(solvefield, args)
    if not p.waitForStarted(5000):
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: solve-field failed to start."))
        return False, f"Failed to start solve-field: {p.errorString()}"

    if not _wait_process(p, 300000, parent=parent):
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: solve-field timed out."))
        return False, "solve-field timed out."

    if p.exitCode() != 0:
        out = bytes(p.readAllStandardOutput()).decode(errors="ignore")
        err = bytes(p.readAllStandardError()).decode(errors="ignore")
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: solve-field failed."))
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
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Waiting for job assignment…"))
    base = _get_astrometry_api_url(settings)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        resp = _astrometry_api_request("GET", base + f"submissions/{subid}",
                                       parent=parent, stage="poll job")
        if resp:
            jobs = resp.get("jobs", [])
            if jobs and jobs[0] is not None:
                _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Job assigned (ID {0}).").format(jobs[0]))
                try: return int(jobs[0])
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        _sleep_ui(1000)
    return None

def _astrometry_poll_calib(settings, job_id: int, *, max_wait_s=900, parent=None) -> dict | None:
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Waiting for solution…"))
    base = _get_astrometry_api_url(settings)
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        resp = _astrometry_api_request("GET", base + f"jobs/{job_id}/calibration/",
                                       parent=parent, stage="poll calib")
        if resp and all(k in resp for k in ("ra","dec","pixscale")):
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: Solution received."))
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

def _get_solver_preference(settings) -> str:
    val = (settings.value("plate_solver/preference", "both", type=str) or "both").lower()
    # migrate old vizier_only key to GAIA_only
    if val == "vizier_only":
        settings.setValue("plate_solver/preference", "gaia_only")
        return "gaia_only"
    return val

def _set_solver_preference(settings, pref: str):
    settings.setValue("plate_solver/preference", (pref or "both").lower())

_SEED_MODES = {"auto", "manual", "none"}
_SOLVER_PREFS = {"both", "gaia_only", "astap_only", "astrometry_only"}
_RADIUS_MODES = {"auto", "value"}
_FOV_MODES = {"compute", "auto", "value"}


def _choice(val, allowed: set[str], default: str) -> str:
    s = str(val if val is not None else default).strip().lower()
    return s if s in allowed else default


def _opt_bool(val, default=False):
    if val is None or val == "":
        return bool(default)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


def _opt_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def normalize_plate_solve_preset(preset: dict | None) -> dict:
    p = dict(preset or {})
    solver = p.get("solver") or p.get("solver_pref") or p.get("preference") or "both"
    scale = p.get("scale")
    if scale is None:
        scale = p.get("scale_arcsec")
    radius_value = _opt_float(p.get("radius_value"))
    if radius_value is None:
        radius_value = 5.0
    fov_value = _opt_float(p.get("fov_value"))
    if fov_value is None:
        fov_value = 0.0
    return {
        "seed_mode": _choice(p.get("seed_mode"), _SEED_MODES, "auto"),
        "ra": str(p.get("ra") or "").strip(),
        "dec": str(p.get("dec") or "").strip(),
        "scale": _opt_float(scale),
        "radius_mode": _choice(p.get("radius_mode"), _RADIUS_MODES, "auto"),
        "radius_value": float(radius_value),
        "fov_mode": _choice(p.get("fov_mode"), _FOV_MODES, "compute"),
        "fov_value": float(fov_value),
        "solver": _choice(solver, _SOLVER_PREFS, "both"),
        "blind": _opt_bool(p.get("blind"), True),
        "force_blind": _opt_bool(p.get("force_blind"), False),
    }


def apply_plate_solve_preset(settings, preset: dict | None) -> dict:
    spec = normalize_plate_solve_preset(preset)
    if settings is None:
        return spec
    _set_seed_mode(settings, spec["seed_mode"])
    _set_manual_seed(settings, spec["ra"], spec["dec"], spec["scale"])
    settings.setValue("astap/seed_radius_mode", spec["radius_mode"])
    settings.setValue("astap/seed_radius_value", float(spec["radius_value"]))
    settings.setValue("astap/seed_fov_mode", spec["fov_mode"])
    settings.setValue("astap/seed_fov_value", float(spec["fov_value"]))
    _set_solver_preference(settings, spec["solver"])
    _set_blind_enabled(settings, spec["blind"])
    _set_force_blind(settings, spec["force_blind"])
    return spec


def apply_plate_solve_to_doc(parent, doc, settings, preset: dict | None = None):
    """Headless plate-solve of a specific document, using an optional preset."""
    apply_plate_solve_preset(settings, preset)
    return plate_solve_doc_inplace(parent, doc, settings)

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

    ASTAP output and header round-trips sometimes store these as strings,
    which makes astropy raise "'<' not supported between instances of 'str'
    and 'int'" while parsing. Widened from the SASv2 logic to also cover the
    PC/PV matrices, the pole keywords, and the axis-count keywords so no stray
    string value can trip a comparison deeper inside astropy.
    """
    int_keys = {
        "WCSAXES", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
        "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
    }
    float_keys = {
        "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CROTA1", "CROTA2",
        "EQUINOX", "LONPOLE", "LATPOLE",
    }
    float_re = re.compile(r"^(?:PC\d+_\d+|PV\d+_\d+|(?:A|B|AP|BP)_\d+_\d+)$")

    def _tok(v):
        m = re.search(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", str(v))
        return m.group(0) if m else None

    out = {}
    for k, v in d.items():
        key = k.upper()
        try:
            if key in int_keys:
                if isinstance(v, bool):
                    out[key] = v
                elif isinstance(v, str):
                    t = _tok(v)
                    out[key] = int(float(t)) if t is not None else v
                elif isinstance(v, float):
                    out[key] = int(round(v))
                else:
                    out[key] = v
            elif key in float_keys or float_re.match(key):
                if isinstance(v, str):
                    t = _tok(v)
                    out[key] = float(t) if t is not None else v
                else:
                    out[key] = v
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
        "PLTSOLVD", "WARNING", "RADESYS", "RADECSYS",
        "SOLVRMS", "SOLVP95", "SOLVNSTR"
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
    #
    # Written in the convention CROTA2 is defined in and _summarize reports:
    # the position angle of celestial north. The previous form,
    # atan2(CD1_2, CD1_1), was a THIRD convention -- it works out to
    # 180 - theta, which agrees with the position angle only at theta = 0 and
    # drifts by 2*theta everywhere else. Whatever it wrote, a later solve of
    # the same file read back through _make_seed_wcs as if it were theta.
    #
    #     det > 0:   PA = atan2(CD1_2, -CD1_1) + 180
    #     det < 0:   PA = atan2(CD1_2, -CD1_1)
    #
    # which tracks the numerically measured north angle to 0.06 deg, and to
    # better than 0.01 deg once a real solution's skew is present.
    try:
        if ("CROTA1" not in merged or "CROTA2" not in merged) and \
           ("CD1_1" in merged and "CD1_2" in merged):
            _c11 = float(merged["CD1_1"])
            _c12 = float(merged["CD1_2"])
            _c21 = float(merged.get("CD2_1", 0.0))
            _c22 = float(merged.get("CD2_2", 0.0))
            _det = _c11 * _c22 - _c12 * _c21
            rot = math.degrees(math.atan2(_c12, -_c11))
            if _det > 0:
                rot += 180.0
            rot %= 360.0
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

    # ── Recompute the true focal length from the solved plate scale ──
    # scale("/px) = 206.265 · (pixel_size_um · binning) / focal_mm
    #   ⇒ focal_mm = 206.265 · pixel_size_um_binned / scale("/px)
    # Scale is read the SAME way as the success popup (proj_plane_pixel_scales),
    # so it works whether the solver wrote a CD matrix (ASTAP) or PC+CDELT
    # (astropy/Gaia). Pixel size comes from the acquisition header.
    _orig_focal = _first_float(out.get("FOCALLEN"))
    _derived_focal = None
    try:
        scale_arcsec = None
        # 1) preferred: astropy plate scale off the assembled WCS (CD, PC+CDELT, SIP)
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            _w = _wcs_from_header_2d(out, relax=True)
            _sc = proj_plane_pixel_scales(_w)
            scale_arcsec = float(np.mean(_sc[:2])) * 3600.0
        except Exception:
            scale_arcsec = None
        # 2) fallback: CD determinant
        if not (scale_arcsec and scale_arcsec > 0):
            cd11 = _first_float(out.get("CD1_1")); cd12 = _first_float(out.get("CD1_2"))
            cd21 = _first_float(out.get("CD2_1")); cd22 = _first_float(out.get("CD2_2"))
            if None not in (cd11, cd12, cd21, cd22):
                det = abs(cd11 * cd22 - cd12 * cd21)
                if det > 0:
                    scale_arcsec = (det ** 0.5) * 3600.0
        # 3) fallback: CDELT (areal)
        if not (scale_arcsec and scale_arcsec > 0):
            c1 = _first_float(out.get("CDELT1")); c2 = _first_float(out.get("CDELT2"))
            if c1 and c2:
                scale_arcsec = (abs(c1 * c2)) ** 0.5 * 3600.0
            elif c1:
                scale_arcsec = abs(c1) * 3600.0
            elif c2:
                scale_arcsec = abs(c2) * 3600.0

        px_um_x = _first_float(out.get("XPIXSZ"))
        px_um_y = _first_float(out.get("YPIXSZ"))
        px_um = None
        if px_um_x and px_um_y:
            px_um = 0.5 * (px_um_x + px_um_y)
        elif px_um_x:
            px_um = px_um_x
        elif px_um_y:
            px_um = px_um_y

        if scale_arcsec and scale_arcsec > 0 and px_um and px_um > 0:
            bx = _first_int(out.get("XBINNING")) or _first_int(out.get("XBIN")) or 1
            by = _first_int(out.get("YBINNING")) or _first_int(out.get("YBIN")) or 1
            focal_mm = 206.264806 * (px_um * (bx + by) / 2.0) / scale_arcsec
            if np.isfinite(focal_mm) and 1.0 <= focal_mm <= 50000.0:
                _derived_focal = float(focal_mm)
    except Exception:
        _derived_focal = None

    if _derived_focal is not None:
        try:
            out["FOCALLEN"] = (round(_derived_focal, 3), "focal length from plate solve (mm)")
            if _orig_focal and abs(_orig_focal - _derived_focal) >= 0.5:
                out.add_history(
                    f"FOCALLEN recomputed from plate solve: "
                    f"{_orig_focal:.1f} -> {_derived_focal:.1f} mm"
                )
        except Exception:
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

def _solve_numpy_with_fallback(parent, settings, image, seed_header):
    """
    Four-pass solve strategy — thin wrapper that guarantees the status
    popup goes away when the solver finishes.

    The popup had a Hide button but no auto-close, so a solve kicked off
    from a tool other than the main PlateSolverDialog (the Supernova /
    Asteroid Hunter runs it directly) left a modeless "Plate Solving"
    window stuck on top until the user hunted it down and clicked Hide.
    Wrapping the real implementation here closes it on every exit path —
    success, failure, or raised exception — in exactly one place rather
    than at every one of the implementation's many `return` statements.
    """
    try:
        return _solve_numpy_with_fallback_impl(parent, settings, image, seed_header)
    finally:
        try:
            _status_popup_close()
        except Exception:
            pass


def _gaia_pass(parent, settings, image, seed_header, say=None):
    """
    Pass 1, plus one blind retry when a seeded attempt fails.

    The blind path only runs on its own when the header has NO RA/Dec at all,
    and _parse_ra_deg accepts eight keys -- CRVAL1, RA, OBJCTRA, TELRA, MNTRA,
    CENTERRA, RA_OBJ, ALPHA_J2000. Capture software writes the mount's RA/DEC
    into every light, and a stacked master inherits them, so a master almost
    always LOOKS seeded. When those numbers are stale or the scale is wrong,
    the spiral searches around a position or scale the field is not at, finds
    nothing, and the whole chain fails -- even though the quad index could
    have identified the field outright without either number.

    So: if a seeded attempt fails, throw the seed away and ask the index.
    That is the one case the seed cannot be trusted, by definition. It costs
    a couple of seconds on a path that has already failed, and nothing at all
    when the first attempt succeeds.

    Returns (ok, Header | error_str).
    """
    ok, res = _solve_with_GAIA_threaded(parent, settings, image, seed_header)
    if ok:
        return True, res

    # Only worth retrying if the first pass actually USED a seed. If it was
    # already blind -- no pointing in the header, seed mode "none", or the
    # force-blind box ticked -- the retry would repeat it exactly.
    try:
        if _get_force_blind(settings) or _get_seed_mode(settings) == "none":
            return False, res
        if not _get_blind_enabled(settings):
            return False, res
        if not isinstance(seed_header, Header):
            return False, res
        if _parse_ra_deg(seed_header) is None or _parse_dec_deg(seed_header) is None:
            return False, res
        from setiastro.saspro.gaia_quad_index import get_quad_library
        if not get_quad_library().any_installed():
            return False, res
    except Exception:
        return False, res

    # Drop the SCALE as well as the pointing. Force-blind alone keeps the
    # header's pixel scale, which is the right call for the user's checkbox --
    # focal length times pixel size stays true even when the mount lied. But
    # here we only know that SOMETHING in the seed was wrong, and if it was
    # the scale, keeping it poisons the retry twice over: layers_for_field
    # picks the wrong quad layers from it, and the spiral searches the wrong
    # scale band. With no hint, the blind solver tries every built layer
    # coarsest-first and works it out. That is slower, on a path that has
    # already failed.
    retry_header = seed_header.copy()
    for _k in ("FOCALLEN", "PIXSCALE", "SCALE", "PLTSCALE", "SECPIX",
               "SECPIX1", "SECPIX2", "XPIXSZ", "YPIXSZ", "PIXSZ",
               "PIXSIZE1", "PIXSIZE2", "XBINNING", "YBINNING"):
        try:
            del retry_header[_k]
        except Exception:
            pass
    retry_header = _strip_wcs_keys(retry_header)

    msg = ("Status: Seeded solve failed \u2014 the header pointing or scale may be "
           "wrong. Retrying blind from the quad index\u2026")
    print("[PlateSolver] Pass 1 failed with a seed; retrying blind "
          "(pointing and scale discarded)")
    if say is not None:
        say(msg)
    else:
        _set_status_ui(parent, msg)

    ok2, res2 = _solve_with_GAIA_threaded(parent, settings, image, retry_header,
                                          force_blind=True)
    if ok2:
        print("[PlateSolver] blind retry succeeded where the seeded pass failed")
        return True, res2
    # Report the seeded failure: it is the one that describes the user's
    # actual header, and the blind failure is a consequence of it.
    return False, f"{res}  (blind retry also failed: {res2})"


def _solve_numpy_with_fallback_impl(
    parent, settings, image: np.ndarray, seed_header
) -> tuple[bool, "Header | str"]:
    """
    Four-pass solve strategy:

    Pass 1 — In-House Gaia DR3 (Hough/triangle matching against the local
              Gaia catalog). Fast, no external process, handles rotation/scale
              variants. When the header carries no usable RA/Dec this pass now
              recovers one blind from the local quad index first, so a missing
              or wrong pointing no longer forces the chain down to ASTAP.

    Pass 2 — ASTAP seeded (auto from header, compute FOV from scale).
              Fast when header has good RA/Dec.

    Pass 3 — ASTAP blind (-r 179, -fov 0, -z 0, no RA/Dec seed).
              Slower but handles completely wrong or missing coordinates.

    Pass 4 — Astrometry.net (local solve-field if configured, then web API).
              Last resort for fields nothing else can solve.
    """
    from astropy.io.fits import Header
    pref = _get_solver_preference(settings)
    print(f"[PlateSolver] pref='{pref}'")

    # ── Astrometry.net only ──────────────────────────────────────────────────
    if pref == "astrometry_only":
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Solver preference = Astrometry.net only…"))
        return _solve_numpy_with_astrometry(parent, settings, image, seed_header)

    # ── Gaia DR3 only ────────────────────────────────────────────────────────
    if pref in ("gaia_only", "vizier_only"):
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Solver preference = In-House Gaia DR3 only…"))
        ok, res = _gaia_pass(parent, settings, image, seed_header)
        if ok:
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
                "Status: Solved via in-house Gaia DR3."))
        else:
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
                "Status: In-house Gaia DR3 failed: {0}").format(str(res)))
        return ok, res

    # ── Pass 1: In-House Gaia DR3 ────────────────────────────────────────────
    if pref not in ("astap_only",):
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Trying in-house Gaia DR3 solver…"))
        QApplication.processEvents()

        ok1, res1 = _gaia_pass(parent, settings, image, seed_header)
        if ok1:
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
                "Status: Solved via in-house Gaia DR3."))
            return True, res1

        err1 = str(res1)
        print(f"[PlateSolver] Pass 1 (Gaia DR3) failed: {err1}")
    else:
        err1 = "skipped (ASTAP only)"
        print(f"[PlateSolver] Pass 1 skipped — {err1}")

    # ── ASTAP only early exit after Gaia fails ───────────────────────────────
    # (still runs ASTAP passes below, just won't fall through to Astrometry.net)

    # ── Save original settings to restore after each ASTAP pass ─────────────
    _orig_radius_mode = settings.value("astap/seed_radius_mode", "auto",    type=str)
    _orig_radius_val  = settings.value("astap/seed_radius_value", 5.0,      type=float)
    _orig_fov_mode    = settings.value("astap/seed_fov_mode",    "compute", type=str)
    _orig_fov_val     = settings.value("astap/seed_fov_value",    0.0,      type=float)
    _orig_seed_mode   = settings.value("astap/seed_mode",        "auto",    type=str)

    def _restore():
        settings.setValue("astap/seed_radius_mode",  _orig_radius_mode)
        settings.setValue("astap/seed_radius_value",  _orig_radius_val)
        settings.setValue("astap/seed_fov_mode",      _orig_fov_mode)
        settings.setValue("astap/seed_fov_value",     _orig_fov_val)
        settings.setValue("astap/seed_mode",          _orig_seed_mode)

    # ── Determine if we have useful seed data ────────────────────────────────
    _has_seed = False
    _seed_mode = _get_seed_mode(settings)

    if _get_force_blind(settings):
        # Pass 1 ignored the header position, so Pass 2 must not reinstate it:
        # ASTAP seeded on the same wrong coordinates would just fail slowly
        # before the blind pass gets its turn.
        _seed_mode = "none"
        print("[PlateSolver] Force blind solve — header position ignored "
              "for every pass")

    if _seed_mode == "manual":
        # Check if manual fields have valid RA/Dec
        ra_s  = _get_manual_ra(settings)
        dec_s = _get_manual_dec(settings)
        ra_deg  = _parse_ra_input_to_deg(ra_s)
        dec_deg = _parse_dec_input_to_deg(dec_s)
        _has_seed = (ra_deg is not None and dec_deg is not None)
        if _has_seed:
            print(f"[PlateSolver] Manual seed: RA={ra_deg:.4f}° Dec={dec_deg:.4f}°")
        else:
            print("[PlateSolver] Manual mode selected but RA/Dec fields are invalid/empty")
    elif isinstance(seed_header, Header):
        ra  = _parse_ra_deg(seed_header)
        dec = _parse_dec_deg(seed_header)
        _has_seed = (ra is not None and dec is not None)

    # ── Pass 2: ASTAP seeded ─────────────────────────────────────────────────
    if _has_seed:
        _r_display = "auto" if _orig_radius_mode == "auto" else f"{_orig_radius_val}°"
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: ASTAP solving (seeded from header, r={0})…").format(_r_display))

        _pass2_scale = None
        _pass2_fov   = None
        if isinstance(seed_header, Header):
            _pass2_scale = _estimate_scale_arcsec_from_header(seed_header)
            if _pass2_scale is not None:
                H_img = int(image.shape[0]) if image.ndim >= 2 else 0
                if H_img > 0:
                    naxis2 = seed_header.get("NAXIS2", None)
                    try:
                        H_full = int(naxis2)
                        bin_factor = max(1, round(H_full / H_img))
                    except (TypeError, ValueError):
                        bin_factor = 1
                    H_effective = H_img * bin_factor
                    _pass2_fov = (H_effective * _pass2_scale) / 3600.0
                    print(f"[PlateSolver] Pass 2: H_img={H_img}  H_full(hdr)={H_effective}  bin_factor={bin_factor}")

        settings.setValue("astap/seed_mode", "auto")
        if _pass2_fov is not None and _pass2_fov > 0:
            settings.setValue("astap/seed_fov_mode",  "value")
            settings.setValue("astap/seed_fov_value",  _pass2_fov)
            print(f"[PlateSolver] Pass 2: scale={_pass2_scale:.3f}\"/px  fov={_pass2_fov:.4f}°  r={_orig_radius_mode}/{_orig_radius_val}")
        else:
            settings.setValue("astap/seed_fov_mode",  "auto")
            settings.setValue("astap/seed_fov_value",  0.0)
            print(f"[PlateSolver] Pass 2: no scale in header, fov=auto  r={_orig_radius_mode}/{_orig_radius_val}")

        try:
            ok2, res2 = _solve_numpy_with_astap(parent, settings, image, seed_header)
        finally:
            _restore()

        if ok2:
            _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
                "Status: Solved with ASTAP (seeded)."))
            return True, res2

        err2 = str(res2) if res2 is not None else "unknown error"
        print(f"[PlateSolver] Pass 2 (ASTAP seeded) failed: {err2}")
    else:
        err2 = "no seed available"
        print(f"[PlateSolver] Pass 2 skipped — {err2}")

    # ── Pass 3: ASTAP blind ──────────────────────────────────────────────────
    _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
        "Status: {0} Trying ASTAP blind solve…"
    ).format(f"Seeded solve failed ({err2})." if _has_seed else "No seed available."))
    QApplication.processEvents()

    settings.setValue("astap/seed_mode",         "none")
    settings.setValue("astap/seed_radius_mode",  "value")
    settings.setValue("astap/seed_radius_value",  179.0)
    settings.setValue("astap/seed_fov_mode",      "auto")
    settings.setValue("astap/seed_fov_value",      0.0)
    print("[PlateSolver] Pass 3: blind  r=179  fov=auto")

    try:
        ok3, res3 = _solve_numpy_with_astap(parent, settings, image, None)
    finally:
        _restore()

    if ok3:
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Solved with ASTAP (blind)."))
        return True, res3

    err3 = str(res3) if res3 is not None else "unknown error"
    print(f"[PlateSolver] Pass 3 (ASTAP blind) failed: {err3}")

    # ── Pass 4: Astrometry.net ───────────────────────────────────────────────
    if pref == "astap_only":
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: ASTAP failed and solver is set to ASTAP only."))
        return False, QCoreApplication.translate("PlateSolver",
            "ASTAP solve failed (both seeded and blind). Astrometry.net fallback is disabled.")

    if pref == "gaia_only":
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Gaia DR3 solve failed and solver is set to Gaia only."))
        return False, QCoreApplication.translate("PlateSolver",
            "In-house Gaia DR3 solve failed. ASTAP and Astrometry.net fallback are disabled.")

    _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
        "Status: ASTAP failed ({0}). Falling back to Astrometry.net…").format(err3))
    QApplication.processEvents()

    ok4, res4 = _solve_numpy_with_astrometry(parent, settings, image, seed_header)
    if ok4:
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: Solved via Astrometry.net."))
    else:
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver",
            "Status: All solvers failed. Last error: {0}").format(str(res4)))
    return ok4, res4

def _solve_numpy_with_astrometry(
    parent, settings, image: np.ndarray, seed_header
) -> tuple[bool, "Header | str"]:
    """
    Solve via Astrometry.net (local solve-field first, then web API fallback).
    Returns (ok, Header) on success, (False, error_str) on failure.
    """
    from astropy.io.fits import Header

    # ── Try local solve-field first if configured ────────────────────────────
    solvefield = _get_solvefield_exe(settings)
    if solvefield and os.path.exists(solvefield):
        _set_status_ui(
            parent,
            QCoreApplication.translate("PlateSolver", "Status: Trying local solve-field…")
        )
        try:
            gray = _to_gray2d_unit(_normalize_for_astap(image))
            tmp_path = _write_temp_fit_web_16bit(gray)
            try:
                ok, res = _solve_with_local_solvefield(parent, settings, tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            if ok and isinstance(res, Header):
                acq_base = None
                if isinstance(seed_header, Header):
                    acq_base = _strip_wcs_keys(seed_header.copy())
                if acq_base is not None:
                    return True, _merge_wcs_into_base_header(acq_base, res)
                return True, res
        except Exception as e:
            print(f"[Astrometry] local solve-field exception: {e}")

    # ── Web API ──────────────────────────────────────────────────────────────
    api_key = _get_astrometry_api_key(settings)
    if not api_key:
        return False, QCoreApplication.translate(
            "PlateSolver",
            "Astrometry.net API key not configured (Preferences → API Keys)."
        )

    session = _astrometry_login(settings, parent=parent)
    if not session:
        return False, QCoreApplication.translate(
            "PlateSolver", "Astrometry.net login failed."
        )

    # Write a 16-bit grayscale FITS for upload
    try:
        gray = _to_gray2d_unit(_normalize_for_astap(image))
        tmp_path = _write_temp_fit_web_16bit(gray)
    except Exception as e:
        return False, QCoreApplication.translate(
            "PlateSolver", "Failed to write upload FITS: {0}"
        ).format(str(e))

    try:
        subid = _astrometry_upload(settings, session, tmp_path, parent=parent)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not subid:
        return False, QCoreApplication.translate(
            "PlateSolver", "Astrometry.net upload failed."
        )

    job_id = _astrometry_poll_job(settings, subid, parent=parent)
    if not job_id:
        return False, QCoreApplication.translate(
            "PlateSolver", "Astrometry.net job assignment timed out."
        )

    calib = _astrometry_poll_calib(settings, job_id, parent=parent)
    if not calib:
        return False, QCoreApplication.translate(
            "PlateSolver", "Astrometry.net solve timed out or failed."
        )

    # Try to get the full WCS FITS (includes SIP terms if available)
    wcs_hdr = _astrometry_download_wcs_file(settings, job_id, parent=parent)

    if wcs_hdr is None:
        # Fall back to building a plain TAN header from the calibration dict
        wcs_hdr = _wcs_header_from_astrometry_calib(calib, image.shape)

    # Merge with acquisition header
    acq_base: Header | None = None
    if isinstance(seed_header, Header):
        acq_base = _strip_wcs_keys(seed_header.copy())

    if acq_base is not None:
        return True, _merge_wcs_into_base_header(acq_base, wcs_hdr)
    return True, wcs_hdr


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

def _make_seed_wcs(ra_deg: float, dec_deg: float, scale_arcsec: float,
                   img_w: int, img_h: int,
                   seed_header: "Header | None" = None,
                   flip: bool = False) -> WCS:
    """Build a TAN seed WCS, incorporating rotation from header if available.

    flip=False → det(CD) > 0
    flip=True  → det(CD) < 0 (standard astronomical parity)
    The correct parity depends on the optical train, so the search loop
    tries both.
    """
    import math

    rot_deg = 0.0
    if seed_header is not None:
        for key in ("ANGLE", "POSANGLE", "CROTA2", "CROTA1", "OBJCTROT",
                    "ROTANG", "ROT_ANGL", "CAMROTAN"):
            v = seed_header.get(key)
            if v is not None:
                try:
                    rot_deg = float(v)
                    print(f"[GaiaLocal] using rotation {rot_deg:.2f}° from header key {key}")
                    break
                except Exception:
                    pass

    w = WCS(naxis=2)
    w.wcs.crpix = [img_w / 2.0, img_h / 2.0]
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    scale_deg = float(scale_arcsec) / 3600.0

    # CROTA2 is a POSITION ANGLE. _cd_matrix's theta is a different number.
    # For a CD built by _cd_matrix, the position angle of celestial north --
    # which is what CROTA2 means, and what plate_solve_cli._summarize
    # reports -- works out to
    #
    #     det > 0  (mirrored=False):   PA = theta + 180
    #     det < 0  (mirrored=True):    PA = -theta
    #
    # exact to 0.06 deg from the equator to dec +89.3, that residual being
    # the TAN projection's own field rotation rather than an error.
    #
    # Passing CROTA2 straight in as theta therefore built a seed rotated 180
    # deg wrong on the det > 0 branch and sign-flipped on the other. The
    # triangle matcher absorbs it -- side-ratio invariants survive any
    # similarity transform -- which is why it stayed invisible for so long.
    # The evidence was in the matcher's own output all along: a seeded solve
    # whose header CROTA2 was already correct reported
    #
    #     [TriMatch] affine consensus: 48/60  scale=1.0000  parity=+  rot=-180.00
    #
    # a seed perfect in every respect except exactly 180 degrees of roll.
    # Inverting the mapping here lets the matcher start where the header
    # actually says, so it has nothing to undo -- which matters most on a
    # marginal field, where the rotation it must search through is the
    # difference between a match and a miss.
    theta_deg = (-rot_deg) if flip else (rot_deg - 180.0)
    if seed_header is not None and rot_deg:
        print(f"[GaiaLocal] seed rotation: CROTA2 {rot_deg:.2f}\u00b0 "
              f"(position angle) -> CD theta {theta_deg % 360.0:.2f}\u00b0 "
              f"for {'det<0' if flip else 'det>0'} parity")
    rot_rad = math.radians(theta_deg)

    # Stashed for the rotation diagnostic in _solve_fingerprint(): the seed is
    # built here and the solution is judged there, so comparing the two is the
    # only way to see how much roll the matcher had to remove.
    global _LAST_SEED_ROT
    _LAST_SEED_ROT = (float(rot_deg), float(theta_deg % 360.0), bool(flip))

    w.wcs.cd = np.array(_cd_matrix(scale_deg, rot_rad, mirrored=flip))
    w.wcs.set()
    return w

def _triangle_match_catalog_to_image(
    img_stars: np.ndarray,
    cat_xy: np.ndarray,
    img_w: int,
    img_h: int,
    max_stars: int = 60,
    min_matches: int = 6,
    match_tol_px: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Rotation-, scale-, translation- and parity-invariant star matching via
    triangle similarity invariants.

    Algorithm references:
      - Groth (1986), AJ 91, 1244 — triangle invariant matching; the same
        family of algorithm used by Siril's atpmatch.
      - Valdes et al. (1995), PASP 107, 1119 — FOCAS voting variant.
      - Beroiz, Cabral & Sanchez (2020), A&C 32, 100384 — astroalign; this
        implementation follows its invariant + vertex-vote scheme.
      - Fischler & Bolles (1981), Comm. ACM 24(6) — RANSAC affine consensus.

    Unlike the translation-only Hough vote, this makes NO assumption about
    the seed WCS rotation, scale, or parity: side-ratio invariants survive
    any similarity transform including reflections, so a CROTA2 sign error,
    a 90° rotator offset, or a mirrored optical train cannot break it.
    """
    from itertools import combinations
    from scipy.spatial import cKDTree

    def _brightest_xy(pts: np.ndarray, n: int) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64)
        if a.ndim == 2 and a.shape[1] >= 3:
            a = a[np.argsort(-a[:, 2])]
        return np.ascontiguousarray(a[:n, :2])

    # cat_xy must arrive brightness-sorted (it does: cat_infos sorted by gmag)
    src = _brightest_xy(img_stars, max_stars)
    ref = np.ascontiguousarray(np.asarray(cat_xy, dtype=np.float64)[:max_stars, :2])

    n_src, n_ref = len(src), len(ref)
    print(f"[TriMatch] {n_src} img stars vs {n_ref} cat stars (brightest)")
    if n_src < 3 or n_ref < 3:
        return None, None

    def _tri_invariants(pts: np.ndarray):
        idx = np.array(list(combinations(range(len(pts)), 3)), dtype=np.int32)
        p0, p1, p2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        # side i is opposite vertex i
        s0 = np.hypot(p1[:, 0] - p2[:, 0], p1[:, 1] - p2[:, 1])
        s1 = np.hypot(p0[:, 0] - p2[:, 0], p0[:, 1] - p2[:, 1])
        s2 = np.hypot(p0[:, 0] - p1[:, 0], p0[:, 1] - p1[:, 1])
        S = np.column_stack([s0, s1, s2])
        order = np.argsort(S, axis=1)                    # short → long
        Ss = np.take_along_axis(S, order, axis=1)
        V = np.take_along_axis(idx, order, axis=1)       # vertices in side order
        # reject tiny and needle-thin triangles (centroid noise dominates them)
        good = (Ss[:, 0] > 8.0) & (Ss[:, 2] / np.maximum(Ss[:, 0], 1e-9) < 8.0)
        Ss, V = Ss[good], V[good]
        inv = np.column_stack([Ss[:, 2] / Ss[:, 0], Ss[:, 1] / Ss[:, 0]])
        return inv, V

    inv_s, V_s = _tri_invariants(src)
    inv_r, V_r = _tri_invariants(ref)
    if len(inv_s) == 0 or len(inv_r) == 0:
        return None, None

    tree = cKDTree(inv_r)
    dists, idxs = tree.query(inv_s, k=1, distance_upper_bound=0.05, workers=-1)
    hits = np.isfinite(dists)
    print(f"[TriMatch] {int(hits.sum())} triangle pairs within invariant tol")
    if hits.sum() < 3:
        return None, None

    # each matched triangle votes for its 3 vertex correspondences;
    # true correspondences accumulate votes, coincidences don't
    votes: dict = {}
    for si, ri in zip(np.where(hits)[0], idxs[hits]):
        for a in range(3):
            key = (int(V_s[si, a]), int(V_r[ri, a]))
            votes[key] = votes.get(key, 0) + 1

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    used_s, used_r = set(), set()
    cand = []
    for (i, j), c in ranked:
        if c < 2:
            break
        if i in used_s or j in used_r:
            continue
        used_s.add(i); used_r.add(j)
        cand.append((i, j))
    print(f"[TriMatch] {len(cand)} voted correspondences")
    if len(cand) < 3:
        return None, None

    cand_src = src[[i for i, j in cand]]
    cand_ref = ref[[j for i, j in cand]]

    # RANSAC full-affine (6 dof: handles rotation, scale, AND reflection)
    def _fit_affine(ps, pr):
        A = np.hstack([ps, np.ones((len(ps), 1))])
        Mx, *_ = np.linalg.lstsq(A, pr[:, 0], rcond=None)
        My, *_ = np.linalg.lstsq(A, pr[:, 1], rcond=None)
        return Mx, My

    def _apply_affine(pts, Mx, My):
        A = np.hstack([pts, np.ones((len(pts), 1))])
        return np.column_stack([A @ Mx, A @ My])

    rng = np.random.default_rng(0)
    n_cand = len(cand)
    best_inl, best_M = None, None
    for _ in range(300):
        pick = rng.choice(n_cand, 3, replace=False)
        try:
            Mx, My = _fit_affine(cand_src[pick], cand_ref[pick])
        except Exception:
            continue
        pred = _apply_affine(cand_src, Mx, My)
        err = np.hypot(pred[:, 0] - cand_ref[:, 0], pred[:, 1] - cand_ref[:, 1])
        inl = err < match_tol_px
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_M = inl, (Mx, My)
    min_consensus = max(6, int(0.15 * n_cand))
    if best_inl is None or best_inl.sum() < min_consensus:
        print(f"[TriMatch] RANSAC consensus too weak "
              f"({0 if best_inl is None else int(best_inl.sum())}/{n_cand}, "
              f"need {min_consensus})")
        return None, None

    Mx, My = _fit_affine(cand_src[best_inl], cand_ref[best_inl])
    det = Mx[0] * My[1] - Mx[1] * My[0]
    print(f"[TriMatch] affine consensus: {int(best_inl.sum())}/{n_cand}  "
          f"scale={float(np.sqrt(abs(det))):.4f}  parity={'-' if det < 0 else '+'}  "
          f"rot={math.degrees(math.atan2(My[0], Mx[0])):.2f}°")

    # re-match the FULL star sets with the recovered transform
    all_src = np.asarray(img_stars, dtype=np.float64)[:, :2]
    all_ref = np.asarray(cat_xy, dtype=np.float64)[:, :2]
    pred = _apply_affine(all_src, Mx, My)
    rtree = cKDTree(all_ref)
    dists, idxs = rtree.query(pred, k=1, workers=-1)
    print(f"[TriMatch] full-set distances: min={dists.min():.2f}px "
          f"median={np.median(dists):.2f}px p90={np.percentile(dists, 90):.2f}px")
    # Iterative tighten: refit the affine on inliers, re-match, shrink the
    # tolerance toward the observed residual scale. In dense fields (10^4+
    # catalog stars in frame) a fixed 1.5x tolerance harvests wrong nearest
    # neighbors that inflate the WCS-fit RMS past the quality gate.
    tol = match_tol_px * 1.5
    for _ in range(3):
        inlier_mask = dists < tol
        if int(inlier_mask.sum()) < max(min_matches, 10):
            break
        Mx, My = _fit_affine(all_src[inlier_mask], all_ref[idxs[inlier_mask]])
        pred = _apply_affine(all_src, Mx, My)
        dists, idxs = rtree.query(pred, k=1, workers=-1)
        core = dists[dists < tol]
        if len(core) == 0:
            break
        new_tol = max(2.0, 4.0 * float(np.median(core)))
        if new_tol >= tol:
            break
        tol = new_tol

    inlier_mask = dists < tol
    img_m = all_src[inlier_mask].astype(np.float32)
    cat_m = all_ref[idxs[inlier_mask]].astype(np.float32)
    print(f"[TriMatch] final: {int(inlier_mask.sum())} matched pairs at tol={tol:.1f}px")
    if len(img_m) < min_matches:
        return None, None
    return img_m, cat_m

def _hough_match_catalog_to_image(
    img_stars: np.ndarray,
    cat_xy: np.ndarray,
    img_w: int,
    img_h: int,
    max_stars: int = 100,
    min_matches: int = 6,
    match_tol_px: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Match image stars to catalog stars using a Hough-style vote on
    (dx, dy) translation space, followed by a RANSAC similarity-transform
    refinement pass before final affine inlier rejection.

    Algorithm references:
      - Hough (1962), U.S. Patent 3,069,654 — original transform concept.
      - Ballard (1981), Pattern Recognition 13(2):111–122 — Generalized
        Hough Transform extended to arbitrary shapes and translations.
      - Tabur (2007), PASA 24, 189 — voting on translation offsets between
        projected catalog and detected image stars; closest analog to this impl.
      - Valdes et al. (1995), PASP 107, 1119 — FOCAS catalog matching via
        voting/histogram approach (USNO astrometric matching library basis).
      - Groth (1986), AJ 91, 1244 — original triangle invariant matching,
        foundational to all subsequent star-pattern recognition work.
      - Fischler & Bolles (1981), Comm. ACM 24(6):381-395 — RANSAC, used
        here to fit a robust similarity transform from a noisy candidate
        pool without being dragged off by false correspondences.

    The seed WCS may carry meaningful residual rotation/scale error on top
    of the dominant translation (e.g. when CRVAL itself is off by a
    significant fraction of a degree). A pure-translation Hough vote only
    captures stars near the rotation center tightly; everything else drifts
    out of the bin as you move away from that center. To handle this:
      1. Vote on (dx, dy) to find the coarse bulk offset.
      2. Gather a loose 3x3-neighborhood candidate pool around that bin.
      3. Fit a similarity transform (rotation + uniform scale + translation)
         via RANSAC over that pool — minimal 2-point samples scored by
         inlier consensus — rejecting any false cross-bin correspondences
         that would otherwise bias a plain least-squares fit.
      4. Re-match using the corrected transform at a tight tolerance and
         do a final affine fit + inlier rejection for the returned pairs.
    """
    from scipy.spatial import KDTree

    def _grid_sample(pts, n, w, h):
        if len(pts) <= n:
            return pts.copy()
        cols = max(1, int(np.sqrt(n * w / max(h, 1))) + 1)
        rows = max(1, int(np.sqrt(n * h / max(w, 1))) + 1)
        per  = max(1, n // (cols * rows) + 1)
        cw, rh = w / cols, h / rows
        out = []
        for r in range(rows):
            for c in range(cols):
                mask = (
                    (pts[:,0] >= c*cw) & (pts[:,0] < (c+1)*cw) &
                    (pts[:,1] >= r*rh) & (pts[:,1] < (r+1)*rh)
                )
                cell = pts[mask]
                if len(cell):
                    out.append(cell[:per])
        result = np.vstack(out) if out else pts[:n]
        return result[:n]

    src = _grid_sample(img_stars, max_stars, img_w, img_h)
    ref = _grid_sample(cat_xy,    max_stars, img_w, img_h)

    n_src, n_ref = len(src), len(ref)
    print(f"[HoughMatch] {n_src} img stars, {n_ref} cat stars")

    if n_src < 3 or n_ref < 3:
        return None, None

    # ── Vote on (dx, dy) translation ────────────────────────────────────
    bin_px = match_tol_px * 2.0
    dx_range = img_w
    dy_range = img_h

    votes = {}   # (bx, by) → [(src_i, ref_j), ...]
    for i, s in enumerate(src):
        for j, r in enumerate(ref):
            dx = r[0] - s[0]
            dy = r[1] - s[1]
            if abs(dx) > dx_range * 0.5 or abs(dy) > dy_range * 0.5:
                continue
            bx = int(np.round(dx / bin_px))
            by = int(np.round(dy / bin_px))
            key = (bx, by)
            if key not in votes:
                votes[key] = []
            votes[key].append((i, j))

    if not votes:
        print("[HoughMatch] no votes accumulated")
        return None, None

    best_key  = max(votes, key=lambda k: len(votes[k]))
    best_count = len(votes[best_key])
    print(f"[HoughMatch] peak bin {best_key} has {best_count} votes "
          f"(dx≈{best_key[0]*bin_px:.1f}px, dy≈{best_key[1]*bin_px:.1f}px)")

    bx0, by0 = best_key
    merged = list(votes[best_key])
    # Always pull in the 3x3 neighborhood — this gives the similarity-fit
    # stage a wider, noisier candidate set to work with even when the peak
    # bin itself is tight (residual rotation/scale spreads votes out).
    for dbx in range(-1, 2):
        for dby in range(-1, 2):
            if dbx == 0 and dby == 0:
                continue
            k = (bx0 + dbx, by0 + dby)
            merged.extend(votes.get(k, []))

    if len(merged) < 4:
        print(f"[HoughMatch] too few candidates even after 3x3 merge: {len(merged)}")
        return None, None

    print(f"[HoughMatch] candidate pool after 3x3 merge: {len(merged)} pairs")

    # ── Similarity-transform refinement via RANSAC ────────────────────────
    # Fit src -> ref using a 4-parameter similarity transform:
    #   [x']   [ a -b ] [x]   [tx]
    #   [y'] = [ b  a ] [y] + [ty]
    # A plain least-squares fit over the whole noisy candidate pool can be
    # dragged off by a handful of false cross-bin correspondences (which is
    # exactly what was happening here: scale/rotation looked sane but the
    # post-transform distance distribution was garbage for most stars).
    # RANSAC over minimal 2-point samples avoids that.
    cand_src = np.array([src[i] for i, j in merged], dtype=np.float64)
    cand_ref = np.array([ref[j] for i, j in merged], dtype=np.float64)

    def _fit_similarity(pts_src, pts_ref):
        n = len(pts_src)
        A = np.zeros((2 * n, 4), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        x, y = pts_src[:, 0], pts_src[:, 1]
        A[0::2, 0] = x;  A[0::2, 1] = -y;  A[0::2, 2] = 1.0
        A[1::2, 0] = y;  A[1::2, 1] =  x;  A[1::2, 3] = 1.0
        b[0::2] = pts_ref[:, 0]
        b[1::2] = pts_ref[:, 1]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol  # a_, b_, tx, ty

    def _apply_sim(pts, sol):
        a_, b_, tx, ty = sol
        xs, ys = pts[:, 0], pts[:, 1]
        return np.column_stack([
            a_ * xs - b_ * ys + tx,
            b_ * xs + a_ * ys + ty,
        ])

    try:
        n_cand = len(cand_src)
        if n_cand < 4:
            raise ValueError("not enough candidates for similarity fit")

        rng = np.random.default_rng(0)
        ransac_tol = match_tol_px * 2.0

        idx_pairs = [(i, j) for i in range(n_cand) for j in range(i + 1, n_cand)]
        rng.shuffle(idx_pairs)
        n_trials = min(200, len(idx_pairs))

        best_inliers = None
        best_sol = None

        for i, j in idx_pairs[:n_trials]:
            sub_src = cand_src[[i, j]]
            sub_ref = cand_ref[[i, j]]
            try:
                sol = _fit_similarity(sub_src, sub_ref)
            except Exception:
                continue
            pred = _apply_sim(cand_src, sol)
            err = np.hypot(pred[:, 0] - cand_ref[:, 0], pred[:, 1] - cand_ref[:, 1])
            inliers = err < ransac_tol
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_sol = sol

        if best_sol is None or best_inliers.sum() < 4:
            raise ValueError("RANSAC found no consensus")

        print(f"[HoughMatch] RANSAC similarity: {best_inliers.sum()}/{n_cand} inliers")

        # Refit on the inlier consensus set for a cleaner final transform
        sol = _fit_similarity(cand_src[best_inliers], cand_ref[best_inliers])
        a_, b_, tx, ty = sol

        src_transformed = _apply_sim(src, sol)
        scale_recovered = float(np.hypot(a_, b_))
        rot_recovered = float(np.degrees(np.arctan2(b_, a_)))
        print(f"[HoughMatch] similarity fit: scale={scale_recovered:.4f} "
              f"rot={rot_recovered:.2f}°")
    except Exception as e:
        print(f"[HoughMatch] similarity fit failed ({e}), falling back to translation only")
        best_dx = best_key[0] * bin_px
        best_dy = best_key[1] * bin_px
        src_transformed = src + np.array([best_dx, best_dy])

    # ── Re-match with the corrected (similarity-transformed) positions ───
    tree = KDTree(ref)
    dists, idxs = tree.query(src_transformed, k=1, workers=-1)

    print(f"[HoughMatch] post-similarity distance stats: "
          f"min={dists.min():.2f}px median={np.median(dists):.2f}px "
          f"p90={np.percentile(dists, 90):.2f}px max={dists.max():.2f}px")

    inlier_mask = dists < match_tol_px * 1.5
    img_m = src[inlier_mask]
    cat_m = ref[idxs[inlier_mask]]

    print(f"[HoughMatch] {inlier_mask.sum()} pairs within {match_tol_px*1.5:.1f}px after similarity correction")

    if len(img_m) < min_matches:
        return None, None

    # ── Affine fit + inlier rejection (final polish) ─────────────────────
    try:
        from scipy.linalg import lstsq
        ones  = np.ones((len(img_m), 1))
        A_mat = np.hstack([img_m, ones])
        Ax, _, _, _ = lstsq(A_mat, cat_m[:, 0])
        Ay, _, _, _ = lstsq(A_mat, cat_m[:, 1])
        pred_x = A_mat @ Ax
        pred_y = A_mat @ Ay
        res = np.sqrt((cat_m[:,0]-pred_x)**2 + (cat_m[:,1]-pred_y)**2)
        print(f"[HoughMatch] affine residuals: median={np.median(res):.1f}px  "
              f"max={res.max():.1f}px")
        inliers = res < match_tol_px
        if inliers.sum() < min_matches:
            inliers = res < match_tol_px * 3
            print(f"[HoughMatch] relaxed tol: {inliers.sum()} inliers")
        if inliers.sum() < min_matches:
            return None, None
        img_m = img_m[inliers]
        cat_m = cat_m[inliers]
    except Exception as e:
        print(f"[HoughMatch] affine check failed: {e}")

    print(f"[HoughMatch] final: {len(img_m)} matched pairs")
    return img_m, cat_m

# ---------- Threaded Gaia solve ----------

class _GaiaSolveWorker(QThread):
    """Runs the in-house Gaia DR3 solve off the GUI thread."""
    status = pyqtSignal(str)
    done = pyqtSignal(bool, object)   # (ok, Header|error_str)

    def __init__(self, image: np.ndarray, seed_header, settings_scope=None,
                 parent=None, force_blind: "bool | None" = None):
        super().__init__(parent)
        self._image = image
        self._seed_header = seed_header
        self._force_blind = force_blind
        # Capture the org/app name so the worker thread opens the same
        # QSettings file as the caller (avoids the "SASpro" vs
        # "Seti Astro Suite Pro" mismatch when the main app uses QSettings()).
        self._settings_scope = settings_scope  # (org, app) tuple or None

    def run(self):
        try:
            # QSettings is not safe to share across threads — make a
            # thread-local instance rather than borrowing the GUI's.
            from PyQt6.QtCore import QSettings
            if self._settings_scope:
                org, app = self._settings_scope
                settings = QSettings(org, app)
            else:
                settings = QSettings("SetiAstro", "SASpro")
            ok, res = _solve_with_GAIA(
                self._image, self._seed_header,
                parent=None, settings=settings,
                status_cb=self.status.emit,
                force_blind=self._force_blind,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            ok, res = False, f"Gaia DR3 solver: worker exception: {e}"
        self.done.emit(ok, res)

def _solve_with_GAIA_threaded(parent, settings, image, seed_header,
                              force_blind: "bool | None" = None) -> tuple[bool, "Header | str"]:
    """
    Same signature/contract as _solve_with_GAIA but runs it in a QThread,
    spinning a local event loop so the UI stays responsive. Synchronous to
    the caller — the fallback chain doesn't need restructuring.
    """
    result = {}
    loop = QEventLoop()

    # Pass the caller's QSettings scope so the worker reads from the same
    # storage file (e.g. "Seti Astro Suite Pro" when called from the main
    # window, not the hard-coded fallback "SASpro").
    _scope = None
    if settings is not None:
        try:
            _org = settings.organizationName()
            _app = settings.applicationName()
            if _org or _app:
                _scope = (_org, _app)
        except Exception:
            pass

    worker = _GaiaSolveWorker(image, seed_header, settings_scope=_scope,
                              force_blind=force_blind)
    worker.status.connect(lambda t: _set_status_ui(parent, t))   # queued → GUI thread

    def _on_done(ok, res):
        result["ok"], result["res"] = ok, res
        loop.quit()

    worker.done.connect(_on_done)
    worker.start()
    loop.exec()
    worker.wait(10_000)
    worker.deleteLater()
    return result.get("ok", False), result.get("res", "Gaia DR3 solver: worker did not return")

# ---------- Blind solving via the local quad index ----------

def _get_blind_enabled(settings) -> bool:
    """Blind solving through the quad index. On by default."""
    if settings is None:
        return True
    try:
        v = settings.value("plate_solver/blind_quads", True)
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no")
        return bool(v)
    except Exception:
        return True


def _set_blind_enabled(settings, on: bool):
    if settings is not None:
        settings.setValue("plate_solver/blind_quads", bool(on))


def _get_force_blind(settings) -> bool:
    """
    Ignore the header RA/Dec and locate the field from the star pattern.

    For when the pointing is WRONG rather than missing. A wrong seed is worse
    than no seed: the spiral search quarters a patch of sky the field is not
    in, fails slowly, and then hands ASTAP the same bad position to fail on
    again. The pixel scale is kept, because it is derived from focal length
    and pixel size and stays right even when the mount was lying.
    """
    if settings is None:
        return False
    try:
        v = settings.value("plate_solver/force_blind", False)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)
    except Exception:
        return False


def _set_force_blind(settings, on: bool):
    if settings is not None:
        settings.setValue("plate_solver/force_blind", bool(on))


def _blind_seed_header(image: np.ndarray, scale_hint, say) -> tuple[bool, object]:
    """
    Ask the quad index where this image points. Returns (True, Header) with
    CRVAL/CD/CROTA2 only, or (False, reason).
    """
    try:
        from setiastro.saspro.gaia_blind_solver import blind_seed
    except Exception as e:
        return False, (f"blind solver unavailable ({e}). Build a Blind Solve "
                       f"Index on the Gaia DR3 Library dialog.")
    # _set_status_ui only forwards a line to the batch log when it is tagged
    # like the rest of the chain, so tag everything the blind solver says
    # here rather than prefixing a dozen strings over there.
    def blind_say(msg: str):
        m = str(msg)
        say(m if m.startswith(("Status:", "\u25b6", "\u2714", "\u274c")) else f"Status: {m}")

    blind_say("Status: Blind solving \u2014 no usable RA/Dec, identifying the "
              "field from the star pattern\u2026")
    try:
        return blind_seed(image, scale_hint_arcsec=scale_hint,
                          status_cb=blind_say)
    except Exception as e:
        import traceback
        traceback.print_exc()
        blind_say(f"Status: Blind solve failed: {type(e).__name__}: {e}")
        return False, f"blind solve raised {type(e).__name__}: {e}"


def _merge_blind_seed(original: "Header | None", blind: "Header") -> "Header":
    """
    Acquisition header + the blind WCS. The original's own WCS keys are dropped
    first: if they were any good we would not be here, and a stale CROTA2 would
    otherwise override the roll the blind solve just measured.
    """
    base = (_strip_wcs_keys(original.copy())
            if isinstance(original, Header) else Header())
    for k in ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CROTA1", "CROTA2",
              "RADESYS", "WCSAXES", "BLNDODDS", "BLNDLAYR"):
        if k in blind:
            try:
                base[k] = blind[k]
            except Exception:
                pass
    return base


#: Catalogue stars to aim for in one query. Four thousand rather than the
#: ~1000 the 3x3 grid keeps, because the grid wants its quota in EVERY cell
#: and the brightest stars in a field are not spread evenly -- a cluster or
#: the galactic plane will hog one cell while another starves. Headroom here
#: is what lets the grid make a uniform selection out of a clumpy one.
_CATALOG_TARGET_STARS = 4000

#: Per-detection multiplier, and the floor/ceiling around it. Tying the target
#: to the detection count is what keeps the catalogue reliably DEEPER than the
#: image: both scale the same way with field area, so a frame that detects
#: more stars asks for proportionally more catalogue.
_CATALOG_STARS_PER_DETECTION = 4
_CATALOG_TARGET_MIN = 2000
_CATALOG_TARGET_MAX = 20000


def _catalog_mag_limit(ra_lo, ra_hi, dec_lo, dec_hi, n_detected: int = 0) -> float:
    """
    Faintest G worth querying for this sky box.

    Unlimited, a query returns every Gaia source in the field, which on a wide
    frame is enormous and almost entirely useless: a 12.6 x 8.2 degree field
    holds about 1.2 million sources at full depth, and the matcher downstream
    keeps at most 999 of them. The rest are fetched, sorted, built into a
    SkyCoord, projected to pixels and discarded -- once per search centre,
    scale and parity attempted.

    Because the limit falls as the field grows, the row count stays roughly
    constant whatever the field size, so the query cost no longer depends on
    how wide the frame is. A narrow deep field is unaffected: at a third of a
    square degree it still takes G ~ 18 to find a few thousand stars.

    Never raises -- a solve must not fail because this estimate could not be
    made, so anything unexpected falls back to no limit at all.
    """
    try:
        from setiastro.saspro.gaia_database import (
            field_area_deg2, mag_limit_for_field)
        area = field_area_deg2(ra_lo, ra_hi, dec_lo, dec_hi)
        # A non-finite box must fall back to NO limit, not to the bright
        # floor. Both are "wrong", but one of them under-fetches and fails the
        # solve, while the other merely costs what today already costs.
        if not np.isfinite(area) or area <= 0.0:
            return 99.0
        target = _CATALOG_TARGET_STARS
        if n_detected > 0:
            target = int(np.clip(_CATALOG_STARS_PER_DETECTION * n_detected,
                                 _CATALOG_TARGET_MIN, _CATALOG_TARGET_MAX))
        return float(mag_limit_for_field(area, target_stars=target))
    except Exception as e:
        print(f"[GaiaLocal] could not size the magnitude limit ({e}); "
              f"querying unlimited")
        return 99.0


def _solve_fingerprint(wcs_solution, img_matched, sep_arcsec,
                       img_w: int, img_h: int) -> None:
    """
    Print what two solves of the same frame can actually be compared on.

    RMS and pair count cannot be compared between runs: they are computed over
    whatever pairs each run happened to keep, and the matcher chooses its own
    tolerance, so a run that paired more stars reports a worse RMS even when
    its astrometry is identical. Everything here is independent of that choice.

      * The FINGERPRINT evaluates the final WCS at fixed pixel positions. Two
        runs that agree here are the same solution, whatever their RMS says --
        and the corners are the test that matters, because the centre can
        agree while the SIP terms disagree out at the edges.
      * The PERCENTILES show the shape of the residual distribution. A
        population of genuine mismatches is a heavy tail; noisier centroids on
        fainter stars are a broad, tight-tailed spread. These are 2-D angular
        separations, so isotropic Gaussian position error makes them Rayleigh
        distributed, giving p95/RMS = 2.448/sqrt(2) ~ 1.73; below that there
        is no outlier population to prune.
      * The RADIAL PROFILE says whether the error is uniform or lives at the
        field edge, which separates a distortion model that has not converged
        from pairs that are simply wrong.

    Purely diagnostic. Never raises -- a solve must not fail because a print
    could not be produced.
    """
    try:
        print("[GaiaLocal] --- comparable between runs "
              "(RMS and pair count are not) ---")

        # How much roll did the matcher have to remove from the seed?
        #
        # A residual near zero means the seed rotation was right. Near +/-180
        # means the convention is inverted somewhere. Anything else is a real
        # error in whatever produced the hint -- and the blind solver's
        # roll_deg has shown ~16 deg of it on top of the 180, which is too
        # large to be a convention and wants watching across several frames.
        #
        # globals().get rather than a bare name: a solve that never built a
        # seed through _make_seed_wcs simply has nothing to report.
        _seed = globals().get("_LAST_SEED_ROT")
        if _seed is not None:
            _crota, _theta, _flip = _seed
            try:
                _cx, _cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0
                _c = wcs_solution.pixel_to_world(_cx, _cy)
                _xn, _yn = wcs_solution.world_to_pixel_values(
                    float(_c.ra.deg), float(_c.dec.deg) + 10.0 / 3600.0)
                _pa = math.degrees(math.atan2(float(_xn) - _cx,
                                              float(_yn) - _cy)) % 360.0
                _solved_theta = ((-_pa) if _flip else (_pa - 180.0)) % 360.0
                _resid = (_solved_theta - _theta + 180.0) % 360.0 - 180.0
                print(f"[GaiaLocal]   rotation: seed CROTA2 "
                      f"{_crota:.2f}\u00b0 -> CD theta {_theta:.2f}\u00b0; "
                      f"solved theta {_solved_theta:.2f}\u00b0 "
                      f"(north PA {_pa:.2f}\u00b0); matcher removed "
                      f"{_resid:+.2f}\u00b0")
                if abs(abs(_resid) - 180.0) < 5.0:
                    print("[GaiaLocal]   rotation: a residual near "
                          "+/-180\u00b0 means the seed convention is inverted")
                elif abs(_resid) > 5.0:
                    print(f"[GaiaLocal]   rotation: {_resid:+.2f}\u00b0 is a "
                          f"real seed error, not a convention offset")
            except Exception:
                pass

        pts = [("centre", (img_w - 1) / 2.0, (img_h - 1) / 2.0),
               ("TL", 0.0, 0.0), ("TR", img_w - 1.0, 0.0),
               ("BL", 0.0, img_h - 1.0), ("BR", img_w - 1.0, img_h - 1.0)]
        for name, x, y in pts:
            c = wcs_solution.pixel_to_world(x, y)
            print(f"[GaiaLocal]   fingerprint {name:<7} "
                  f"RA={float(c.ra.deg):11.6f}  Dec={float(c.dec.deg):+11.6f}")

        q = np.percentile(sep_arcsec, [50, 75, 90, 95, 99])
        rms = float(np.sqrt(np.mean(sep_arcsec ** 2)))
        print(f"[GaiaLocal]   residuals p50={q[0]:.2f}\" p75={q[1]:.2f}\" "
              f"p90={q[2]:.2f}\" p95={q[3]:.2f}\" p99={q[4]:.2f}\" "
              f"max={float(sep_arcsec.max()):.2f}\"")
        # p95/RMS was the obvious shape test and it is a trap: a tail big
        # enough to matter inflates the RMS it is being compared against, so
        # the ratio SHRINKS as the tail grows. Measured on this very field,
        # the run with a 77" worst pair scored 1.48 ("no outliers") while the
        # clean one scored 2.31. Ratios against the MEDIAN do not move with
        # the tail, so they actually report it.
        _med = q[0] if q[0] > 0 else float("nan")
        print(f"[GaiaLocal]   shape p95/p50={q[3] / _med:.1f}  "
              f"p99/p50={q[4] / _med:.1f}  "
              f"max/p50={float(sep_arcsec.max()) / _med:.1f}  "
              f"(Rayleigh gives 2.1 / 2.6 / 3.3 at ~1000 pairs; "
              f"far above means real outliers)")

        # Residual against distance from the frame centre, in four equal-count
        # rings so every bin carries the same weight.
        cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0
        r = np.hypot(img_matched[:, 0] - cx, img_matched[:, 1] - cy)
        order = np.argsort(r)
        rings = np.array_split(order, 4)
        parts = []
        for i, ring in enumerate(rings):
            if len(ring) == 0:
                continue
            rr = float(np.sqrt(np.mean(sep_arcsec[ring] ** 2)))
            parts.append(f"r{i}({len(ring)}@{float(r[ring].max()) / max(cx, 1):.2f}R)"
                         f"={rr:.2f}\"")
        print("[GaiaLocal]   radial RMS  " + "  ".join(parts))

        # Coverage, stated outright. This is the line that distinguishes a
        # solution fitted across the frame from one fitted in the middle and
        # extrapolated outward -- and the RMS alone cannot tell them apart,
        # because the second number is lower precisely BECAUSE it describes
        # less of the image.
        _corner = math.hypot(cx, cy)
        _reach = float(r.max()) / _corner if _corner > 0 else 0.0
        _inside = float(np.mean(
            np.hypot(np.random.default_rng(0).uniform(0, img_w, 20000) - cx,
                     np.random.default_rng(1).uniform(0, img_h, 20000) - cy)
            <= float(r.max())))
        print(f"[GaiaLocal]   coverage: outermost pair at "
              f"{_reach * 100:.0f}% of the corner distance, "
              f"pairs span ~{_inside * 100:.0f}% of the frame area"
              + ("" if _reach > 0.95 else
                 f"  <-- the outer {(1 - _inside) * 100:.0f}% is EXTRAPOLATED"))
    except Exception as e:
        print(f"[GaiaLocal] fingerprint unavailable: {e}")


def _field_sky_box(corner_sky, margin_deg):
    """
    RA/Dec bounding box for a field, correct at the poles and across RA=0.

    Taking min/max of the four corner RAs is wrong twice over:

      * A field containing a pole has corner RAs scattered anywhere around the
        circle. min/max then spans the gap BETWEEN them and excludes
        everything outside it -- which is most of the field. The catalogue
        comes back missing a wedge of real stars, the matcher pairs the
        survivors with whatever is nearest, and the fit lands hundreds of
        arcseconds out while still reporting a confident-looking consensus.
      * A field straddling RA=0 has corners at (say) 359.9 and 0.2, and the
        box runs the long way round the sky.

    Both are fixed the same way: when the corners span more than half the sky
    in RA, or the field reaches a pole, take all RA and let the in-frame
    projection filter downstream do the trimming. Over-fetching is harmless --
    the caller already drops anything that does not project into the frame --
    whereas under-fetching is silent and fatal.
    """
    ras = [float(c.ra.deg) % 360.0 for c in corner_sky]
    decs = [float(c.dec.deg) for c in corner_sky]
    dec_min = min(decs) - margin_deg
    dec_max = max(decs) + margin_deg

    polar = (dec_max >= 89.0) or (dec_min <= -89.0)
    if polar:
        if dec_max >= 89.0:
            dec_max = 90.0
        if dec_min <= -89.0:
            dec_min = -90.0
    if polar or (max(ras) - min(ras)) > 180.0:
        return 0.0, 360.0, max(-90.0, dec_min), min(90.0, dec_max)
    return (min(ras) - margin_deg, max(ras) + margin_deg,
            max(-90.0, dec_min), min(90.0, dec_max))


def _quad_refine_seed(img_stars, img_w, img_h, scale_hint, say):
    """
    Ask the quad index where this field actually is, reusing the stars the
    caller already detected.

    This exists to collapse the search grid. The spiral is 19 centres x 5
    scales x 2 parities = 190 attempts, and each failing attempt that falls
    through to the Hough matcher costs a pure-Python O(n^2) vote: measured at
    4.3 s for 1000 stars. One hash lookup gets the answer outright, so the
    grid becomes a single attempt that is already correct.

    Returns the solve_blind_core result dict, or None to leave the grid alone.
    Every failure path returns None: this is an accelerator, never a gate.

    NOT CURRENTLY CALLED. It used to run on every solve, which made the
    "blind-solve when there is no RA/Dec" checkbox consult the quad index even
    when the header had a good pointing. Kept because the idea is sound and
    cheap to re-enable -- but a wrong seed is now handled by the blind retry
    in _gaia_pass(), so switching it back on would only buy speed on frames
    that already solve on the first attempt.
    """
    try:
        from setiastro.saspro.gaia_quad_index import get_quad_library
        from setiastro.saspro.gaia_blind_solver import (
            solve_blind_core, make_fetch_box)
    except Exception as e:
        print(f"[GaiaLocal] quad index unavailable: {e}")
        return None
    try:
        lib = get_quad_library()
        if not lib.any_installed():
            return None
        fetch = make_fetch_box()
        if fetch is None:
            return None

        a = np.asarray(img_stars, dtype=np.float64)
        if a.ndim != 2 or len(a) < 8:
            return None
        # The index keeps the brightest stars per sky cell, so the solver has
        # to rank globally by brightness or the two sets never correspond.
        if a.shape[1] >= 3:
            a = a[np.argsort(-a[:, 2])]

        field_arcmin = ((max(img_w, img_h) * scale_hint / 60.0)
                        if scale_hint else None)
        layers = lib.layers_for_field(field_arcmin)
        if not layers:
            return None
        res = solve_blind_core(a[:, :2], int(img_w), int(img_h), layers, fetch,
                               scale_hint_arcsec=scale_hint, status_cb=say)
        return res if res.get("ok") else None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GaiaLocal] quad refine failed, using the search grid: {e}")
        return None


def _solve_with_GAIA(image: np.ndarray,
                     seed_header: "Header | None",
                     parent=None,
                     settings=None,
                     status_cb=None,
                     force_blind: "bool | None" = None) -> tuple[bool, "Header | str"]:
    """
        Match image stars to catalog stars by voting on (dx, dy) translation space
        — a Generalized Hough Transform restricted to pure translation.

        Since the seed WCS already encodes scale and rotation (including camera
        angle from ANGLE/CROTA2/OBJCTROT header keys), the residual transform
        between img_stars and cat_xy is dominated by a small translation only.
        Each (image_star, catalog_star) pair votes for the implied (dx, dy) offset.
        The peak bin in the 2-D accumulator reveals the true translation; candidate
        pairs are then confirmed by nearest-neighbour search and affine inlier
        rejection.

        Algorithm references:
        - Hough (1962), U.S. Patent 3,069,654 — original transform concept
        - Ballard (1981), Pattern Recognition 13(2):111–122 — Generalized
            Hough Transform extended to arbitrary shapes and translations
        - Groth (1986), AJ 91, 1244 — foundational star triangle matching,
            basis for all subsequent star-pattern recognition work
        - Valdes et al. (1995), PASP 107, 1119 — FOCAS voting-based catalog
            matching (USNO astrometric matching library)
        - Tabur (2007), PASA 24, 189 — fast triangle/cosine-metric matching
            for CCD-to-catalog astrometry (arXiv:0710.3618)
        """

    # ── 1) Extract seed ──────────────────────────────────────────────────────
    from PyQt6.QtCore import QSettings
    
    # Check if we're in manual seed mode — if so, prefer settings over header
    _seed_mode = _get_seed_mode(settings) if settings is not None else "auto"

    def _say(text: str):
        if status_cb is not None:
            status_cb(text)
        else:
            _set_status_ui(parent, text)

    # None = honour the user's checkbox; True/False = caller decides, used
    # by the blind retry below so it need not write to persisted settings.
    _force_blind = (_get_force_blind(settings) if force_blind is None
                    else bool(force_blind))
    _from_blind = False

    if _seed_mode == "none" or _force_blind:
        # Explicit blind, or forced. Throw the POSITION away but keep the
        # pixel scale — it comes from focal length and pixel size, so it stays
        # right even when the mount was lying, and it narrows the quad search
        # to one or two layers. A manually entered scale outranks the header,
        # since someone who typed it in knows something we do not.
        ra = dec = None
        # Only trust the manual scale in MANUAL mode. The manual fields are
        # disabled in every other mode, but current_preset() still reads and
        # persists whatever text is sitting in them, so outside manual mode
        # they are stale leftovers. Honouring one here is actively harmful: a
        # stale 1.0"/px against a real 1.63"/px puts the scale filter at
        # 0.7-1.3, which discards every correct candidate before it is ever
        # verified, and the solve grinds through junk instead of failing.
        scale = None
        _scale_src = "manual entry"
        if _seed_mode == "manual" and settings is not None:
            scale = _get_manual_scale(settings)
        if scale is None:
            scale = (_estimate_scale_arcsec_from_header(seed_header)
                     if isinstance(seed_header, Header) else None)
            _scale_src = "the header"
        if _force_blind and _seed_mode != "none":
            _say("Status: Force blind solve — ignoring the given position"
                 + (f", keeping {scale:.3f}\"/px from {_scale_src}"
                    if scale else " and searching every scale"))
    elif _seed_mode == "manual" and settings is not None:
        ra_s  = _get_manual_ra(settings)
        dec_s = _get_manual_dec(settings)
        ra    = _parse_ra_input_to_deg(ra_s)
        dec   = _parse_dec_input_to_deg(dec_s)
        scale = _get_manual_scale(settings)
        # scale from header as fallback if not specified manually
        if scale is None and isinstance(seed_header, Header):
            scale = _estimate_scale_arcsec_from_header(seed_header)
        print(f"[GAIA] manual seed: RA={ra} Dec={dec} scale={scale}")
    else:
        ra    = _parse_ra_deg(seed_header)  if isinstance(seed_header, Header) else None
        dec   = _parse_dec_deg(seed_header) if isinstance(seed_header, Header) else None
        scale = _estimate_scale_arcsec_from_header(seed_header) if isinstance(seed_header, Header) else None

    # ── 1b) No position? Recover one blind from the quad index ───────────────
    # The quad index identifies the field from the star pattern alone. What it
    # returns is a SEED, not a solution: everything below — the spiral search,
    # the matcher, the refinement, the SIP fit and the quality gate — then runs
    # exactly as it does for a header-seeded solve.
    if ra is None or dec is None:
        if not (_get_blind_enabled(settings) or _force_blind):
            return False, ("Gaia DR3 solver: no RA/Dec seed in header, and blind "
                           "solving via the quad index is disabled in settings.")
        ok_b, blind = _blind_seed_header(image, scale, _say)
        if not ok_b:
            return False, f"Gaia DR3 solver: {blind}"
        _from_blind = True
        ra = _parse_ra_deg(blind)
        dec = _parse_dec_deg(blind)
        scale = _estimate_scale_arcsec_from_header(blind) or scale
        seed_header = _merge_blind_seed(seed_header, blind)
        _odds = blind.get("BLNDODDS", None)
        _lyr = blind.get("BLNDLAYR", None)
        _say("Status: Blind solve located the field at "
             f"RA={ra:.4f}° Dec={dec:+.4f}°, {scale:.3f}\"/px"
             + (f", log-odds {float(_odds):,.0f}" if _odds is not None else "")
             + (f", layer {int(_lyr)}" if _lyr is not None else "")
             + " — handing to the Gaia solver to refine…")

    if ra is None or dec is None:
        return False, "Gaia DR3 solver: no RA/Dec seed in header"
    if scale is None:
        return False, "Gaia DR3 solver: no pixel scale in header (need FOCALLEN+XPIXSZ or CD matrix)"

    img_h, img_w = int(image.shape[0]), int(image.shape[1])
    fov_deg = (max(img_h, img_w) * scale) / 3600.0

    # ── 2) Build search grid: center + spiral of offsets + scale variants ────
    def _spiral_centers(ra0, dec0, fov_deg, n_rings=2):
        centers = [(ra0, dec0)]
        step_deg = fov_deg * 0.4
        for ring in range(1, n_rings + 1):
            radius = ring * step_deg
            n_pts = max(6, int(2 * math.pi * ring))
            for i in range(n_pts):
                angle = 2 * math.pi * i / n_pts
                d_ra  = radius * math.cos(angle) / max(math.cos(math.radians(dec0)), 0.01)
                d_dec = radius * math.sin(angle)
                new_ra  = (ra0 + d_ra) % 360.0
                new_dec = max(-90.0, min(90.0, dec0 + d_dec))
                centers.append((new_ra, new_dec))
        return centers

    scale_variants = [scale, scale * 0.95, scale * 1.05, scale * 0.90, scale * 1.10]
    parity_variants = [False, True]
    search_centers = _spiral_centers(ra, dec, fov_deg, n_rings=2)

    def _status(text: str):
        if status_cb is not None:
            status_cb(text)
        else:
            _set_status_ui(parent, text)

    print(f"[GAIA] seed RA={ra:.4f}° Dec={dec:.4f}° scale={scale:.3f}\"/px "
          f"FOV={fov_deg:.3f}° — {len(search_centers)} centers × {len(scale_variants)} scales")
    _status("Status: Querying local Gaia DR3 library…")

    # ── 3) Load Gaia library (astrometric preferred, falls back to spectral) ─
    use_astro = False
    lib = None
    astro_lib = None
    try:
        from setiastro.saspro.gaia_database import get_astro_library, get_library
        astro_lib = get_astro_library()
        if astro_lib.installed_bands():
            use_astro = True
        else:
            lib = get_library()
            if not lib.installed_bands():
                return False, ("Gaia DR3 solver: no Gaia library installed — install the "
                              "Astrometry tiers (recommended, denser) or XP Spectral bands "
                              "via Settings → Gaia DR3 Library")
    except Exception as e:
        return False, f"Gaia DR3 solver: could not load Gaia library: {e}"

    print(f"[GAIA] using {'astrometric' if use_astro else 'spectral'} library")

    # ── 4) Detect image stars once (reused across all attempts) ──────────────
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        norm = _normalize_for_astap(image)
        gray = _to_gray2d_unit(norm)
        mn, mx = float(gray.min()), float(gray.max())
        if mx > mn:
            gray = ((gray - mn) / (mx - mn)).astype(np.float32)
        gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    except Exception as e:
        return False, f"Gaia DR3 solver: image normalization failed: {e}"

    try:
        import sep
        sep.set_extract_pixstack(20000000)
        from setiastro.saspro.star_alignment import _detect_stars_uniform

        _sigma_levels = [50, 25, 15, 10, 5, 3]
        _target_min, _target_max = 1000, 2000
        img_stars = None
        used_sigma = None

        for _sigma in _sigma_levels:
            _candidates = _detect_stars_uniform(
                gray, det_sigma=float(_sigma), minarea=5,
                grid=(5, 5), max_per_cell=60, max_total=1200,
            )
            n = len(_candidates)
            print(f"[GAIA] star detection σ={_sigma}: {n} stars")
            if n >= _target_min:
                if n > _target_max:
                    _grid_rows, _grid_cols = 3, 3
                    _max_per_cell = _target_max // (_grid_rows * _grid_cols)
                    _cell_w = img_w / _grid_cols
                    _cell_h = img_h / _grid_rows
                    _balanced = []
                    for _gr in range(_grid_rows):
                        for _gc in range(_grid_cols):
                            _x0 = _gc * _cell_w; _x1 = (_gc+1) * _cell_w
                            _y0 = _gr * _cell_h; _y1 = (_gr+1) * _cell_h
                            _in_cell = _candidates[
                                (_candidates[:,0] >= _x0) & (_candidates[:,0] < _x1) &
                                (_candidates[:,1] >= _y0) & (_candidates[:,1] < _y1)
                            ]
                            if len(_in_cell) > 0:
                                if _in_cell.shape[1] >= 3:
                                    # sort by brightness (col 2, descending) so
                                    # we keep the most reliable detections per
                                    # cell, not an arbitrary subset
                                    _order = np.argsort(-_in_cell[:, 2])
                                    _in_cell = _in_cell[_order]
                                _balanced.append(_in_cell[:_max_per_cell])
                    _candidates = np.vstack(_balanced) if _balanced else _candidates[:_target_max]
                img_stars = _candidates
                used_sigma = _sigma
                break
            if _sigma == _sigma_levels[-1] and n > 0:
                img_stars = _candidates
                used_sigma = _sigma

        if img_stars is None or len(img_stars) == 0:
            return False, "Gaia DR3 solver: no stars detected in image at any sigma level"

        print(f"[GAIA] using {len(img_stars)} stars at σ={used_sigma}")

    except Exception as e:
        return False, f"Gaia DR3 solver: star detection failed: {e}"

    if len(img_stars) < 10:
        return False, f"Gaia DR3 solver: only {len(img_stars)} stars detected — too few"

    # ── 4b) Collapse the search grid with the quad index ────────────────────
    # PREPENDED, never substituted: the quad answer becomes attempt 0, and the
    # whole original grid stays behind it untouched. If the quad solution is
    # subtly wrong the spiral still runs exactly as it does today, so this can
    # only ever save attempts, never cost a solve.
    # Held for the fallback below: if every refinement attempt fails its
    # quality gate, this verified placement is still the best astrometry we
    # have, and returning it beats returning nothing.
    _quad_seed = None

    if _from_blind:
        # Blind already ran the quad index; its answer is the seed. Collapse
        # the grid to it rather than paying for a second identical lookup.
        search_centers = [(ra, dec)] + list(search_centers)
        scale_variants = [scale] + list(scale_variants)
        if isinstance(seed_header, Header) and seed_header.get("BLNDODDS") is not None:
            _quad_seed = {
                "ra": ra, "dec": dec, "scale_arcsec": scale,
                "roll_deg": seed_header.get("CROTA2", 0.0),
                "parity": seed_header.get("BLNDPAR", "normal"),
                "log_odds": seed_header.get("BLNDODDS"),
                "layer": seed_header.get("BLNDLAYR"),
            }
    # No `elif` consulting the quad index here on purpose.
    #
    # This branch used to run _quad_refine_seed on EVERY solve whenever the
    # blind box was ticked, seed or no seed, as an accelerator: one lookup
    # collapses 19 centres x 5 scales x 2 parities to a single attempt that is
    # already right. But the checkbox says "when there is no RA/Dec", and a
    # header carrying a perfectly good pointing still went through the quad
    # index -- the control did not mean what it said.
    #
    # It is also no longer needed. A seed that turns out to be wrong is caught
    # by _gaia_pass(), which retries blind once Pass 1 has failed, so the only
    # thing the unconditional lookup bought was speed on exactly the frames
    # that did not need it. Blind now runs when the header has no usable
    # pointing (handled above) or when Force blind solve is ticked.

    # ── 5) Search loop: spiral centers × scale variants ──────────────────────
    import sqlite3

    excluded_sids: set = set()
    MAX_RETRY_ROUNDS = 2  # initial attempt + 1 retry with bad stars excluded

    for retry_round in range(MAX_RETRY_ROUNDS):
        best_result = None

        for attempt_idx, (c_ra, c_dec) in enumerate(search_centers):
            if best_result is not None:
                break

            for c_scale in scale_variants:
                if best_result is not None:
                    break

                for c_flip in parity_variants:
                    if best_result is not None:
                        break

                    is_center = (attempt_idx == 0 and c_scale == scale and not c_flip)
                    label = ("center" if is_center
                             else f"spiral[{attempt_idx}] scale={c_scale:.3f}\"/px flip={c_flip}")

                    try:
                        seed_wcs = _make_seed_wcs(c_ra, c_dec, c_scale, img_w, img_h,
                                                  seed_header, flip=c_flip)

                        corners_pix = [(0,0),(img_w-1,0),(0,img_h-1),(img_w-1,img_h-1)]
                        corner_sky  = [seed_wcs.pixel_to_world(x, y) for x, y in corners_pix]
                        (ra_min, ra_max, dec_min,
                         dec_max) = _field_sky_box(corner_sky, 0.1)

                        # Only ever ask for as many stars as the matcher can
                        # use. Unlimited, a wide field returns every Gaia
                        # source inside it -- 1.2 million on a 12.6 x 8.2 deg
                        # frame -- all of which get sorted, turned into a
                        # SkyCoord, projected to pixels, and then decimated to
                        # at most 999 by the 3x3 grid below. And that happens
                        # once per search centre, scale and parity tried.
                        #
                        # The limit is derived per box rather than once, so a
                        # scale variant that widens the field correspondingly
                        # brightens the cut.
                        _gmax = _catalog_mag_limit(ra_min, ra_max, dec_min,
                                                   dec_max, len(img_stars))
                        if use_astro:
                            stars = astro_lib.find_stars_in_box(
                                ra_min, ra_max, dec_min, dec_max, max_mag=_gmax)
                            seen_sids = {
                                s.source_id: {"ra": s.ra, "dec": s.dec, "gmag": s.gmag, "sid": s.source_id}
                                for s in stars if s.source_id not in excluded_sids
                            }
                        else:
                            all_sources = []
                            for fname, conn in lib._connections.items():
                                try:
                                    cur = conn.cursor()
                                    cur.execute("""
                                        SELECT source_id, ra, dec, phot_g_mean_mag FROM sources
                                        WHERE dec BETWEEN ? AND ? AND ra BETWEEN ? AND ?
                                          AND phot_g_mean_mag IS NOT NULL
                                          AND phot_g_mean_mag <= ?
                                    """, (dec_min, dec_max, ra_min, ra_max, float(_gmax)))
                                    all_sources.extend(cur.fetchall())
                                except Exception as e:
                                    print(f"[GaiaLocal] query failed on {fname}: {e}")

                            if not all_sources:
                                continue

                            seen_sids = {}
                            for sid, sra, sdec, gmag in all_sources:
                                if sid in excluded_sids:
                                    continue
                                if sid not in seen_sids:
                                    seen_sids[sid] = {"ra": sra, "dec": sdec, "gmag": gmag, "sid": sid}

                        if len(seen_sids) < 10:
                            continue

                        if is_center:
                            print(f"[GaiaLocal] {len(seen_sids)} unique catalog stars found in field "
                                f"(G <= {_gmax:.1f} for this field size, "
                                f"{len(excluded_sids)} excluded from prior round)")
                        _status(f"Status: {len(seen_sids)} Gaia stars — trying {label}…")

                        cat_infos = list(seen_sids.values())
                        cat_infos.sort(key=lambda i: (i["gmag"] if i["gmag"] is not None else 99.0))
                        cat_sky_all = SkyCoord(
                            ra=[i["ra"] for i in cat_infos] * u.deg,
                            dec=[i["dec"] for i in cat_infos] * u.deg,
                        )
                        cat_sid_all = np.array([i["sid"] for i in cat_infos])
                        px, py = seed_wcs.world_to_pixel(cat_sky_all)
                        cat_xy_all = np.column_stack([px, py]).astype(np.float32)

                        margin = 50
                        in_bounds = (
                            (cat_xy_all[:,0] >= -margin) & (cat_xy_all[:,0] < img_w + margin) &
                            (cat_xy_all[:,1] >= -margin) & (cat_xy_all[:,1] < img_h + margin)
                        )
                        cat_xy_all  = cat_xy_all[in_bounds]
                        cat_sky_all = cat_sky_all[in_bounds]
                        cat_sid_all = cat_sid_all[in_bounds]

                        if len(cat_xy_all) < 10:
                            continue

                        grid_rows, grid_cols = 3, 3
                        max_per_cell = 1000 // (grid_rows * grid_cols)
                        cell_w = img_w / grid_cols
                        cell_h = img_h / grid_rows
                        cat_xy_grid, cat_sky_idx = [], []
                        for gr in range(grid_rows):
                            for gc in range(grid_cols):
                                x0, x1 = gc * cell_w, (gc+1) * cell_w
                                y0, y1 = gr * cell_h, (gr+1) * cell_h
                                in_cell = np.where(
                                    (cat_xy_all[:,0] >= x0) & (cat_xy_all[:,0] < x1) &
                                    (cat_xy_all[:,1] >= y0) & (cat_xy_all[:,1] < y1)
                                )[0]
                                if len(in_cell) == 0:
                                    continue
                                if len(in_cell) > max_per_cell:
                                    in_cell = in_cell[:max_per_cell]
                                cat_xy_grid.append(cat_xy_all[in_cell])
                                cat_sky_idx.extend(in_cell.tolist())

                        if not cat_xy_grid:
                            continue

                        cat_xy  = np.vstack(cat_xy_grid)
                        cat_sky = cat_sky_all[np.array(cat_sky_idx)]
                        cat_sid = cat_sid_all[np.array(cat_sky_idx)]

                        print(f"[GaiaLocal] {label}: {len(cat_xy)} catalog stars, matching…")

                        used_matcher = "triangle"
                        img_matched, cat_matched_xy = _triangle_match_catalog_to_image(
                            img_stars, cat_xy_all,
                            img_w=img_w, img_h=img_h,
                            max_stars=60, min_matches=6, match_tol_px=10.0,
                        )
                        if img_matched is None:
                            used_matcher = "hough"
                            img_matched, cat_matched_xy = _hough_match_catalog_to_image(
                                img_stars, cat_xy,
                                img_w=img_w, img_h=img_h,
                                max_stars=1000, min_matches=6, match_tol_px=10.0,
                            )

                        if img_matched is not None and len(img_matched) >= 6:
                            print(f"[GaiaLocal] matched at {label} with {len(img_matched)} pairs ({used_matcher})")
                            # carry along the full catalog sid array so we can
                            # identify which sids the eventual fit actually used
                            best_result = (img_matched, cat_matched_xy, seed_wcs, cat_sky, cat_xy, cat_sid, used_matcher)
                            break

                    except Exception as e:
                        print(f"[GaiaLocal] attempt {label} failed: {e}")
                        continue

        if best_result is None:
            if retry_round + 1 < MAX_RETRY_ROUNDS:
                print("[GaiaLocal] no match this round, nothing to exclude — stopping retries")
            break

        img_matched, cat_matched_xy, seed_wcs, cat_sky, _cat_xy_full, _cat_sid_full, _used_matcher = best_result

        # Map matched catalog pixel positions back to their source_ids so we
        # can exclude them if this round's solve turns out to be bad.
        try:
            _match_sid_lookup = {}
            for _xy, _sid in zip(_cat_xy_full, _cat_sid_full):
                _match_sid_lookup[(round(float(_xy[0]), 2), round(float(_xy[1]), 2))] = int(_sid)
            matched_sids = []
            for _mxy in cat_matched_xy:
                key = (round(float(_mxy[0]), 2), round(float(_mxy[1]), 2))
                if key in _match_sid_lookup:
                    matched_sids.append(_match_sid_lookup[key])
        except Exception:
            matched_sids = []

        ok_round, result_or_err = _gaia_fit_and_validate(
            img_matched, cat_matched_xy, seed_wcs, cat_sky,
            img_w, img_h, lib, parent,
            use_astro=use_astro, astro_lib=astro_lib,
            status_cb=status_cb, img_stars_all=img_stars,
        )

        # Best of both worlds: if the triangle-derived pairs matched but the
        # WCS fit failed the quality gate, retry THIS SAME FIELD with the
        # Hough matcher before excluding anything — the two matchers have
        # independent failure modes (triangle: dense-field NN contamination;
        # hough: seed rotation/parity error).
        if not ok_round and _used_matcher == "triangle":
            print(f"[GaiaLocal] triangle fit failed gate ({result_or_err}) — retrying field with Hough matcher…")
            h_img, h_cat = _hough_match_catalog_to_image(
                img_stars, _cat_xy_full,
                img_w=img_w, img_h=img_h,
                max_stars=1000, min_matches=6, match_tol_px=10.0,
            )
            if h_img is not None and len(h_img) >= 6:
                print(f"[GaiaLocal] Hough fallback matched {len(h_img)} pairs")
                ok_round, result_or_err = _gaia_fit_and_validate(
                    h_img, h_cat, seed_wcs, cat_sky,
                    img_w, img_h, lib, parent,
                    use_astro=use_astro, astro_lib=astro_lib,
                    status_cb=status_cb, img_stars_all=img_stars,
                )
                if ok_round:
                    img_matched, cat_matched_xy = h_img, h_cat

        if ok_round:
            return True, _gaia_build_final_header(result_or_err, seed_header)

        print(f"[GaiaLocal] retry_round={retry_round}: {result_or_err}")
        if matched_sids and retry_round + 1 < MAX_RETRY_ROUNDS:
            excluded_sids.update(matched_sids)
            print(f"[GaiaLocal] excluding {len(matched_sids)} catalog stars from this bad match, retrying…")
            continue
        else:
            _fb = _quad_seed_solution(_quad_seed, img_w, img_h, seed_header,
                                      result_or_err)
            if _fb is not None:
                print("[GaiaLocal] refinement failed the gate — returning the "
                      "quad index seed as a seed-grade solution")
                _say("Status: Refinement could not reach the accuracy gate. "
                     "Returning the quad index placement instead — the field "
                     "is correctly identified, but the WCS is approximate "
                     "(no distortion fit). SOLVSRC='quad-seed'.")
                return True, _fb
            return False, result_or_err

    _fb = _quad_seed_solution(_quad_seed, img_w, img_h, seed_header,
                              "no match across all search attempts")
    if _fb is not None:
        print("[GaiaLocal] no refined match — returning the quad index seed")
        _say("Status: No refined match. Returning the quad index placement "
             "— approximate WCS, SOLVSRC='quad-seed'.")
        return True, _fb
    return False, "Gaia DR3 solver: no match found across all search attempts"

def _quad_index_status_html() -> str:
    """
    One line on what the blind solver actually has to work with.

    The useful fact is not how many files exist but which FIELD WIDTHS they
    cover: each layer answers a band of about 4x in field size, so a user
    with three layers built still gets nothing on a frame that falls outside
    them. Lead with the covered range, and say plainly when a gap sends the
    solve to ASTAP instead.
    """
    import html as _html
    try:
        from setiastro.saspro.gaia_quad_index import (
            get_quad_library, refresh_quad_library, QUAD_LAYER_DEFS, _fmt_ang)
    except Exception:
        return ("<span style='color:#c87f00'>Quad index module unavailable \u2014 "
                "blind solving will use ASTAP.</span>")
    try:
        refresh_quad_library()
        lib = get_quad_library()
        built = [l for l in lib.installed() if l.defn is not None]
        total = len(QUAD_LAYER_DEFS)
    except Exception as e:
        return ("<span style='color:#c87f00'>Quad index could not be read: "
                f"{_html.escape(str(e))}</span>")

    try:
        from setiastro.saspro.gaia_database import get_astro_library
        n_bands = len(get_astro_library().installed_bands() or [])
    except Exception:
        n_bands = 0
    bands = (f"{n_bands} Gaia catalogue band{'s' if n_bands != 1 else ''} installed"
             if n_bands else "<b>no Gaia catalogue bands installed</b>")

    if not built:
        return ("<span style='color:#c87f00'><b>Quad index: not built.</b> "
                "Blind solving will fall back to ASTAP. Build it from the Gaia "
                "data you already have \u2014 Blind Solve Index tab.</span>"
                f"<br><small>{bands}.</small>")

    lo = min(l.defn.field_lo_arcmin for l in built)
    hi = max(l.defn.field_hi_arcmin for l in built)
    quads = sum(int(l.n_quads) for l in built)
    missing = total - len(built)

    txt = (f"<b>Quad index: {len(built)} of {total} layers built</b> "
           f"({quads:,} quads) \u2014 solves fields "
           f"{_fmt_ang(lo)} to {_fmt_ang(hi)} wide.<br><small>{bands}.")
    if missing:
        txt += (f" {missing} layer{'s' if missing != 1 else ''} not built; "
                "fields outside that range fall back to ASTAP.")
    return txt + "</small>"


def _cd_matrix(scale_deg: float, theta_rad: float, mirrored: bool):
    """
    The CD matrix for a TAN image at pixel scale `scale_deg` and roll
    `theta_rad`, mirrored or not. Returns ((CD1_1, CD1_2), (CD2_1, CD2_2)).

    One definition, because getting this wrong is quiet. The earlier form
    multiplied CD1_1 and CD2_1 by the parity sign and left CD1_2 and CD2_2
    alone. That is not a mirror -- a mirror negates a whole axis, so it must
    take a full row or a full column, never half of each. What it produced
    was a shear: det = -parity * scale^2 * cos(2*theta), which is zero at a
    45 degree roll (singular matrix), changes sign past it (parity silently
    inverts), and leaves the two column vectors non-perpendicular at every
    roll that is not a multiple of 90 degrees. Column norms stay correct, so
    a pixel scale read back from it still looks right while the WCS itself
    is skewed.

    Here det is +scale^2 or -scale^2 exactly, at every roll.
    """
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    if mirrored:
        # det = -scale^2  (standard CROTA2 convention, CDELT1=-s, CDELT2=+s)
        return ((-scale_deg * c, -scale_deg * s),
                (-scale_deg * s,  scale_deg * c))
    # det = +scale^2
    return ((-scale_deg * c,  scale_deg * s),
            (-scale_deg * s, -scale_deg * c))


def _sky_centroid(coords) -> "SkyCoord":
    """
    Mean direction of a set of sky coordinates, as a mean of unit vectors.

    Averaging RA in degrees is wrong wherever the set wraps. A field straddling
    RA 0h averages to 12h. A field containing a celestial pole holds stars at
    every RA at once, so its mean RA is meaningless -- on the Polaris frame the
    stars span 0.1 deg to 360.0 deg and average to 137.7 deg against a true
    centre of 37.0 deg. Either way the tangent point lands degrees away from
    the field, and a TAN fit anchored there cannot represent the true WCS: the
    same 489 pairs fit to 551" RMS about the averaged point and 0.84" about the
    real centre, with the plate scale inflated from 4.03"/px to 4.59"/px.

    Averaging the unit vectors instead has no wrap and no pole.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    v = np.atleast_2d(coords.cartesian.xyz.value.reshape(3, -1).T)
    m = v.mean(axis=0)
    n = float(np.linalg.norm(m))
    if not np.isfinite(n) or n < 1e-12:
        # Antipodally symmetric: no mean direction exists. Any member of the
        # set beats a point outside it.
        f = coords.flatten()[0]
        return SkyCoord(ra=f.ra, dec=f.dec, frame=coords.frame)
    m = m / n
    return SkyCoord(
        ra=(math.degrees(math.atan2(m[1], m[0])) % 360.0) * u.deg,
        dec=math.degrees(math.asin(max(-1.0, min(1.0, float(m[2]))))) * u.deg,
        frame=coords.frame,
    )


def _quad_seed_solution(quad_seed, img_w, img_h, seed_header, reason):
    """
    Last resort: hand back the quad index's own placement of the field.

    The refinement pipeline has failed its quality gate, so there is no
    distortion-corrected fit to return. The quad match itself, though, was
    accepted on log-odds against the background hypothesis -- it is a real
    identification of the field, carrying its own centre, scale, roll and
    parity, accurate to about what a 4-star similarity fit allows. For a user
    who would otherwise get nothing at all, a correctly centred WCS good to a
    few pixels is worth far more than a failure.

    It is returned clearly labelled, and deliberately WITHOUT a SOLVRMS card:
    there is no fit to quote a residual from, and inventing one would let a
    seed-grade WCS pass for a refined one downstream.

    Returns a Header, or None if there is no quad match to fall back on.
    """
    if not quad_seed:
        return None
    try:
        # The quad match brought its own roll. The seed header's rotation is
        # not trustworthy here -- a wrong CROTA2 is one of the reasons the
        # refinement can fail in the first place -- so carry the quad's.
        _rot = Header()
        _rot["CROTA2"] = float(quad_seed.get("roll_deg", 0.0) or 0.0)
        w = _make_seed_wcs(float(quad_seed["ra"]), float(quad_seed["dec"]),
                           float(quad_seed["scale_arcsec"]), img_w, img_h,
                           _rot, flip=(quad_seed.get("parity") == "flipped"))
        hdr = _gaia_build_final_header(w, seed_header)
        for _k in ("SOLVRMS", "SOLVP95", "SOLVNSTR"):
            hdr.pop(_k, None)
        hdr["SOLVSRC"] = ("quad-seed",
                          "WCS from quad index match - NOT refined")
        if quad_seed.get("log_odds") is not None:
            hdr["BLNDODDS"] = (round(float(quad_seed["log_odds"]), 1),
                               "Quad match log-odds vs background")
        if quad_seed.get("layer") is not None:
            hdr["BLNDLAYR"] = (quad_seed["layer"],
                               "Quad index layer that matched")
        hdr.add_history("SASpro: astrometric refinement failed its quality "
                        "gate; this WCS is the quad index seed")
        hdr.add_history("SASpro: seed-grade astrometry - approximate centre "
                        "and scale, no distortion terms")
        hdr.add_history(f"SASpro: gate failure was: {str(reason)[:60]}")
        return hdr
    except Exception as e:
        print(f"[GaiaLocal] quad seed fallback failed: {e}")
        return None


def _gaia_build_final_header(wcs_solution: WCS, seed_header: "Header | None") -> Header:
    wcs_hdr  = _as_header(wcs_solution.to_header(relax=True))
    acq_base = _strip_wcs_keys(seed_header.copy()) if isinstance(seed_header, Header) else None
    out = _merge_wcs_into_base_header(acq_base, wcs_hdr)
    try:
        rms  = getattr(wcs_solution, "_sas_solve_rms", None)
        p95  = getattr(wcs_solution, "_sas_solve_p95", None)
        nst  = getattr(wcs_solution, "_sas_solve_nstars", None)
        if rms is not None:
            out["SOLVRMS"] = (round(float(rms), 3), "Astrometric fit RMS vs Gaia DR3 [arcsec]")
        if p95 is not None:
            out["SOLVP95"] = (round(float(p95), 3), "95th percentile fit residual [arcsec]")
        if nst is not None:
            out["SOLVNSTR"] = (int(nst), "Number of stars in astrometric fit")
    except Exception as e:
        print("solve-stats header cards failed:", e)
    return out


def _gaia_fit_and_validate(
    img_matched: np.ndarray,
    cat_matched_xy: np.ndarray,
    seed_wcs: WCS,
    cat_sky,
    img_w: int,
    img_h: int,
    lib,
    parent,
    use_astro: bool = False,
    astro_lib=None,
    status_cb=None,
    img_stars_all=None,
) -> tuple[bool, "WCS | str"]:
    """
    Run steps 6-9 of the Gaia solve (TAN fit, iterative refinement,
    quality gate, SIP fit) on one matched-pair candidate set.

    img_stars_all is EVERY star detected in the frame, not just the matched
    ones. Two things need it: the per-iteration catalogue query is sized from
    the detection count (the matched count would under-size it, since
    refinement exists to grow the pair set), and the outer-field recovery pass
    below re-matches the stars the affine never managed to pair.
    Returns (True, wcs_solution) on success, (False, error_str) on failure.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    from scipy.spatial import KDTree

    n_matched = len(img_matched)
    _n_detected = (len(img_stars_all) if img_stars_all is not None
                   else n_matched)

    # ── Map matched catalog pixel positions back to sky coords ───────────────
    try:
        matched_cat_px = cat_matched_xy[:, 0]
        matched_cat_py = cat_matched_xy[:, 1]
        matched_sky = seed_wcs.pixel_to_world(matched_cat_px, matched_cat_py)
    except Exception as e:
        return False, f"Gaia DR3 solver: sky coordinate recovery failed: {e}"

    # ── Fit initial TAN WCS ────────────────────────────────────────────────
    try:
        proj_point = _sky_centroid(matched_sky)
        wcs_tan = fit_wcs_from_points(
            (img_matched[:, 0], img_matched[:, 1]),
            matched_sky,
            projection='TAN',
            proj_point=proj_point,
        )
        wcs_tan.array_shape = (img_h, img_w)
        wcs_solution = wcs_tan
    except Exception as e:
        return False, f"Gaia DR3 solver: WCS fit failed: {e}"

    # ── Iterative refinement via fitted-WCS reprojection ──────────────────────
    try:
        MAX_ITER     = 5
        NN_TOL_PX    = 8.0        # ceiling, and the value used on iteration 0
        NN_TOL_FLOOR = 2.5        # never prune tighter than this
        NN_TOL_K     = 3.0        # multiples of the current RMS to accept
        MIN_PAIRS    = 10

        # How much worse the RMS may get in exchange for MORE pairs. Growing
        # the pair set is normally a good sign, but it is not self-evidently
        # one: at 8 px of nearest-neighbour slack -- which on a wide field is
        # a minute of arc -- a coarse seed pairs plenty of image stars with
        # the WRONG catalogue star, and those pairs still count. 10% absorbs
        # the honest case, where a larger sample simply reports a slightly
        # truer RMS than a small one did, while refusing the case where the
        # extra pairs are wrong.
        PAIR_GAIN_RMS_SLACK = 1.10

        # Convergence threshold is pixel-relative, like the matcher and gate:
        # stop when the RMS improvement drops below a small fraction of a pixel.
        # A flat 0.05" is 0.007px on a 7"/px rig (never fires) and 0.1px on a
        # 0.5"/px rig (fires early) — neither is right.
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            _pxsc_conv  = proj_plane_pixel_scales(wcs_solution)
            _scale_conv = float(np.mean(_pxsc_conv[:2])) * 3600.0
        except Exception:
            _scale_conv = None
        RMS_CONVERGE_PX = 0.02
        RMS_CONVERGE = (RMS_CONVERGE_PX * _scale_conv) if (_scale_conv and _scale_conv > 0) else 0.05
        prev_rms = None
        rms = None

        for iteration in range(MAX_ITER):
            sky_fit    = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
            sep_arcsec = matched_sky.separation(sky_fit).arcsec
            rms        = float(np.sqrt(np.mean(sep_arcsec**2)))
            print(f"[GaiaLocal] iter={iteration} RMS={rms:.3f}\"  n={n_matched}")

            if prev_rms is not None and (prev_rms - rms) < RMS_CONVERGE:
                print(f"[GaiaLocal] converged (improvement {prev_rms-rms:.3f}\" < {RMS_CONVERGE:.3f}\")")
                break
            prev_rms = rms

            try:
                corners_pix = [(0,0),(img_w-1,0),(0,img_h-1),(img_w-1,img_h-1)]
                corner_sky  = [wcs_solution.pixel_to_world(x, y) for x, y in corners_pix]
                (t_ra_min, t_ra_max, t_dec_min,
                 t_dec_max) = _field_sky_box(corner_sky, 0.05)

                # Same limit as the search above, and for a sharper reason: the
                # refinement runs this query once per iteration and builds a
                # KD-tree over everything it returns. A faint star the image
                # never detected cannot become a matched pair -- it can only
                # sit in the tree and occasionally capture a detection as its
                # nearest neighbour, which is a wrong pair, not a useful one.
                _t_gmax = _catalog_mag_limit(t_ra_min, t_ra_max, t_dec_min,
                                             t_dec_max, _n_detected)
                t_seen = {}
                if use_astro:
                    for s in astro_lib.find_stars_in_box(
                            t_ra_min, t_ra_max, t_dec_min, t_dec_max,
                            max_mag=_t_gmax):
                        if s.source_id not in t_seen:
                            t_seen[s.source_id] = {"ra": s.ra, "dec": s.dec}
                else:
                    t_sources = []
                    for fname, conn in lib._connections.items():
                        try:
                            cur = conn.cursor()
                            cur.execute("""
                                SELECT source_id, ra, dec FROM sources
                                WHERE dec BETWEEN ? AND ? AND ra BETWEEN ? AND ?
                                  AND phot_g_mean_mag IS NOT NULL
                                  AND phot_g_mean_mag <= ?
                            """, (t_dec_min, t_dec_max, t_ra_min, t_ra_max,
                                  float(_t_gmax)))
                            t_sources.extend(cur.fetchall())
                        except Exception:
                            pass

                    for sid, sra, sdec in t_sources:
                        if sid not in t_seen:
                            t_seen[sid] = {"ra": sra, "dec": sdec}

                if not t_seen:
                    print(f"[GaiaLocal] iter={iteration}: no catalog sources in bbox, stopping")
                    break

                t_cat_infos = list(t_seen.values())
                t_cat_sky_all = SkyCoord(
                    ra=[i["ra"] for i in t_cat_infos] * u.deg,
                    dec=[i["dec"] for i in t_cat_infos] * u.deg,
                )
                t_px, t_py = wcs_solution.world_to_pixel(t_cat_sky_all)
                t_cat_xy = np.column_stack([t_px, t_py]).astype(np.float32)

                margin = 5
                in_b = (
                    (t_cat_xy[:,0] >= -margin) & (t_cat_xy[:,0] < img_w + margin) &
                    (t_cat_xy[:,1] >= -margin) & (t_cat_xy[:,1] < img_h + margin)
                )
                t_cat_xy    = t_cat_xy[in_b]
                t_cat_sky_f = t_cat_sky_all[in_b]

                if len(t_cat_xy) < MIN_PAIRS:
                    print(f"[GaiaLocal] iter={iteration}: only {len(t_cat_xy)} catalog stars in frame, stopping")
                    break

                # Tolerance follows the fit instead of sitting at a constant.
                # A fixed 8 px is downstream of an ADAPTIVE matcher tolerance
                # (_triangle_match_catalog_to_image settles on 4x the median
                # residual, which has been seen anywhere from 5.0 to 10.2 px),
                # so the refinement prunes nothing at all whenever the matcher
                # was stricter than 8 px, and prunes almost nothing when it was
                # looser. Keyed to the current RMS it tightens as the solution
                # improves, which is the only point at which a residual outlier
                # is distinguishable from honest distortion.
                #
                # Iteration 0 is deliberately unchanged: the RMS is still the
                # seed's, so the ceiling governs and behaviour matches today's.
                _nn_tol = NN_TOL_PX
                if _scale_conv and _scale_conv > 0 and rms and rms > 0:
                    _nn_tol = float(np.clip(NN_TOL_K * rms / _scale_conv,
                                            NN_TOL_FLOOR, NN_TOL_PX))

                tree = KDTree(t_cat_xy)
                dists, idxs = tree.query(img_matched, k=1, workers=-1)
                inlier_mask = dists < _nn_tol
                t_img_m     = img_matched[inlier_mask]
                t_cat_sky_m = t_cat_sky_f[idxs[inlier_mask]]

                print(f"[GaiaLocal] iter={iteration}: {inlier_mask.sum()} pairs within "
                      f"{_nn_tol:.1f}px ({_nn_tol * (_scale_conv or 0):.1f}\") "
                      f"from {len(t_cat_xy)} catalog stars"
                      + ("" if _nn_tol >= NN_TOL_PX - 1e-9
                         else f" [tightened from {NN_TOL_PX:.1f}px by the "
                              f"{rms:.2f}\" fit]"))

                if len(t_img_m) < MIN_PAIRS:
                    print(f"[GaiaLocal] iter={iteration}: too few pairs ({len(t_img_m)}), stopping")
                    break

                t_proj = _sky_centroid(t_cat_sky_m)
                t_wcs_tan = fit_wcs_from_points(
                    (t_img_m[:,0], t_img_m[:,1]),
                    t_cat_sky_m,
                    projection='TAN',
                    proj_point=t_proj,
                )
                t_wcs_tan.array_shape = (img_h, img_w)

                t_sky_fit = t_wcs_tan.pixel_to_world(t_img_m[:,0], t_img_m[:,1])
                t_rms     = float(np.sqrt(np.mean(t_cat_sky_m.separation(t_sky_fit).arcsec**2)))
                print(f"[GaiaLocal] iter={iteration}: new WCS RMS={t_rms:.3f}\" ({len(t_img_m)} pairs)")

                # A pair gain has to come with a fit that is not materially
                # worse. Unconditional, this clause rewards COUNT rather than
                # QUALITY, and the two part company exactly when the seed is
                # coarse: a blind seed arrives as a linear TAN with no
                # distortion terms and can sit an arc-minute off, so each
                # iteration brings more image stars within the 8 px tolerance
                # of SOME catalogue star -- not necessarily the right one.
                # Every such increase was adopted, so the refinement ratcheted
                # toward a larger, dirtier pair set. Measured on a 12.6 x 8.2
                # degree frame: blind finished at 1136 pairs and 6.20" where
                # the same field seeded from an existing solution finished at
                # 791 pairs and 3.93". Same WCS to a few tenths of an arcsec;
                # the whole difference was which pairs went into the fit.
                _gained = (len(t_img_m) > n_matched
                           and t_rms <= rms * PAIR_GAIN_RMS_SLACK + 1e-9)
                _kept   = len(t_img_m) >= 0.8 * n_matched   # tolerate small pair loss
                _better = t_rms <= rms + 1e-9
                # The 0.8 retention rule is right when the fit is already good:
                # do not shed a fifth of the pairs for a marginal gain. It is
                # backwards when the fit is broken. A fit whose RMS halves was
                # being dragged by pairs that do not belong in it, and keeping
                # them to protect the count throws the solve away -- measured
                # on a near-pole field that went 519" -> 24" and was refused
                # for dropping 975 pairs to 507.
                _rescue = (t_rms < 0.5 * rms) and len(t_img_m) >= 20
                if _gained or (_better and _kept) or _rescue:
                    print(f"[GaiaLocal] iter={iteration}: adopting ({len(t_img_m)} pairs, RMS={t_rms:.3f}\")")
                    img_matched  = t_img_m
                    matched_sky  = t_cat_sky_m
                    n_matched    = len(t_img_m)
                    rms          = t_rms
                    wcs_solution = t_wcs_tan
                else:
                    # Say WHICH way it failed. "More pairs but a worse fit" and
                    # "fewer pairs" are opposite problems, and lumping them
                    # under "not better" hides the one that used to be
                    # silently accepted.
                    if len(t_img_m) > n_matched:
                        print(f"[GaiaLocal] iter={iteration}: rejecting "
                              f"{len(t_img_m)} pairs (up from {n_matched}) — "
                              f"RMS would go {rms:.3f}\" -> {t_rms:.3f}\", "
                              f"past the {PAIR_GAIN_RMS_SLACK:.2f}x gain "
                              f"allowance; the extra pairs are mismatches")
                    else:
                        print(f"[GaiaLocal] iter={iteration}: not better "
                              f"({len(t_img_m)} pairs, RMS={t_rms:.3f}\"), "
                              f"stopping")
                    break
            except Exception as e:
                print(f"[GaiaLocal] iter={iteration} refinement failed: {e}")
                break

        # Final RMS + quality gate
        sky_fit    = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
        sep_arcsec = matched_sky.separation(sky_fit).arcsec
        rms        = float(np.sqrt(np.mean(sep_arcsec**2)))
        p95        = float(np.percentile(sep_arcsec, 95))
        print(f"[GaiaLocal] TAN RMS={rms:.3f}\"  p95={p95:.3f}\"  n={n_matched}")

        # ── Fit SIP BEFORE the quality gate ──────────────────────────────────
        # A linear TAN projection cannot absorb wide-field lens distortion, so a
        # distorted rig can sit above the gate on TAN yet well under it on SIP.
        # Fit SIP here (own try/except -> fall back to TAN) and let the gate below
        # judge whichever is better; otherwise a fixable distorted field is
        # rejected before SIP ever runs.
        if n_matched >= 20:
            try:
                _sip_proj = _sky_centroid(matched_sky)

                def _sip_terms(_d):
                    # (d+1)(d+2)/2 - 3 distortion coefficients per axis (the
                    # constant and both linear terms are already in the CD
                    # matrix), two axes, plus the 6 linear WCS parameters.
                    return 2 * ((_d + 1) * (_d + 2) // 2 - 3) + 6

                def _sip_feasible(_d):
                    if not (2 <= _d <= SIP_MAX_DEGREE):
                        return False
                    # Degree 5 needs a genuinely well-populated frame: its 42
                    # parameters are only worth solving for when the stars
                    # cover the field densely enough to pin the edges down.
                    if _d >= 5 and n_matched < SIP5_MIN_PAIRS:
                        return False
                    # 4x oversampling: 2 residuals per matched pair.
                    return 2 * n_matched >= 4 * _sip_terms(_d)

                _sip_cache = {}

                def _fit_sip(_d):
                    """Fit one SIP degree -> (wcs, rms, p95, penalised rms)."""
                    if _d in _sip_cache:
                        return _sip_cache[_d]
                    _res = None
                    try:
                        _w = fit_wcs_from_points(
                            (img_matched[:, 0], img_matched[:, 1]),
                            matched_sky,
                            projection='TAN',
                            proj_point=_sip_proj,
                            sip_degree=_d,
                        )
                        _w.array_shape = (img_h, img_w)
                        _sep = matched_sky.separation(
                            _w.pixel_to_world(img_matched[:, 0],
                                              img_matched[:, 1])).arcsec
                        _r = float(np.sqrt(np.mean(_sep ** 2)))
                        _p95 = float(np.percentile(_sep, 95))
                        # In-sample RMS almost always falls as coefficients are
                        # added, so comparing degrees on raw RMS biases toward
                        # the highest one. Penalise by the small-sample
                        # correction sqrt(2N / (2N - k)): negligible when pairs
                        # vastly outnumber coefficients, decisive when they do
                        # not.
                        _k = _sip_terms(_d)
                        _adj = _r * math.sqrt((2.0 * n_matched)
                                              / max(2 * n_matched - _k, 1))
                        print(f"[GaiaLocal] SIP-{_d}: RMS={_r:.3f}\" "
                              f"p95={_p95:.3f}\" ({_k} terms, "
                              f"penalised {_adj:.3f}\")")
                        _res = (_w, _r, _p95, _adj)
                    except Exception as _e:
                        print(f"[GaiaLocal] SIP-{_d} fit failed: {_e}")
                    _sip_cache[_d] = _res
                    return _res

                # Start at the degree the matched-star count comfortably
                # supports, then search OUTWARD in both directions. A higher
                # degree is not merely redundant when it fails -- the extra
                # freedom lets the polynomial diverge where stars are sparsest,
                # usually the frame corners, so it can be far worse than a
                # lower one. Climbing only upward would miss that; we want the
                # best-fitting polynomial, not the highest-order one.
                if n_matched >= 100:
                    _base_degree = 4
                elif n_matched >= 50:
                    _base_degree = 3
                else:
                    _base_degree = 2
                while _base_degree > 2 and not _sip_feasible(_base_degree):
                    _base_degree -= 1

                _best = _fit_sip(_base_degree)       # (wcs, rms, p95, adj)
                _best_deg = _base_degree if _best is not None else None

                # Both walks are anchored at the BASE degree and chained from
                # their own last accepted step, never from the running winner.
                # Anchoring the downward walk on the winner would make it start
                # one step below whatever the upward walk reached, re-test that
                # degree and stop -- leaving a far better low degree unfitted.
                for _step in (+1, -1):
                    _ref = _sip_cache.get(_base_degree)
                    _d = _base_degree + _step
                    while _sip_feasible(_d):
                        _got = _fit_sip(_d)
                        if _got is None:
                            break
                        if _ref is None:
                            pass                     # base fit failed; take it
                        elif _step > 0:
                            # Going up has to EARN its extra terms, and must not
                            # blow out the residual tail -- divergence at the
                            # edges shows up in p95 long before it moves the
                            # mean.
                            if not (_got[3] < _ref[3] * SIP_STEP_UP_GAIN
                                    and _got[2] <= _ref[2] * 1.25):
                                break
                        else:
                            # Going down is already favoured by the penalty, so
                            # any strict improvement is enough.
                            if not _got[3] < _ref[3]:
                                break
                        _ref = _got
                        if _best is None or _got[3] < _best[3]:
                            _best, _best_deg = _got, _d
                        _d += _step

                if _best is not None:
                    # Judge SIP against TAN the same way the degrees were
                    # judged against each other: penalised for parameter count,
                    # and having to clear the same margin. Otherwise an
                    # undistorted field adopts a distortion polynomial for a
                    # fraction of a percent of RMS, which is just fitting the
                    # centroid noise.
                    _tan_adj = rms * math.sqrt((2.0 * n_matched)
                                               / max(2 * n_matched - 6, 1))
                    print(f"[GaiaLocal] TAN RMS={rms:.3f}\" "
                          f"(penalised {_tan_adj:.3f}\")  "
                          f"best SIP-{_best_deg} RMS={_best[1]:.3f}\" "
                          f"(penalised {_best[3]:.3f}\")")
                    if _best[3] < _tan_adj * SIP_STEP_UP_GAIN:
                        wcs_solution = _best[0]
                        print(f"[GaiaLocal] using SIP-{_best_deg} solution")
                    else:
                        print("[GaiaLocal] no SIP degree earned its terms "
                              "- keeping the linear solution")
            except Exception as e:
                print(f"[GaiaLocal] SIP fit failed, using TAN: {e}")

        # ── Recover the outer field against the chosen solution ──────────────
        # The affine the matcher recovers is fitted on a few dozen
        # correspondences, so it describes the middle of the frame and
        # extrapolates badly beyond them -- measured at p90 = 701 px on a
        # 12.6 x 8.2 degree frame. The matcher's tolerance is then set from the
        # residual MEDIAN, which that core dominates, so every star out where
        # the affine has drifted fails the gate and is never paired at all.
        #
        # The consequence is not a worse RMS, it is a DISHONEST one. Two solves
        # of the same image: one paired stars out to 1.18R and reported 6.18",
        # the other stopped at 0.87R and reported 3.09". The second number is
        # better only because it describes the inner 78% of the frame and
        # extrapolates a 30-term polynomial across the rest -- which is exactly
        # where the two disagreed, by 29" at one corner.
        #
        # Once a SIP solution exists that stops being a problem, because the
        # model is now good everywhere to within a few pixels, and a few pixels
        # is far less than the ~49 px that separates neighbouring catalogue
        # stars. So the nearest catalogue star is the RIGHT star even out where
        # the fit is still extrapolating, and the pairs can simply be found and
        # the model refitted with the whole frame constrained.
        #
        # Note the direction: this ADDS the edge pairs that were missed. A
        # previous attempt did the reverse -- clipping the worst residuals,
        # which are at the edge by construction -- and made every measure worse
        # for exactly that reason.
        try:
            RECOVER_K      = 3.0     # tolerance in multiples of the current RMS
            RECOVER_MIN_PX = 5.0     # but never tighter than the extrapolation
            RECOVER_MAX_PX = 15.0    # and never near the catalogue spacing
            RECOVER_SPACING_FRAC = 0.40   # nor past this much of it

            _cx0, _cy0 = (img_w - 1) / 2.0, (img_h - 1) / 2.0
            _cov_before = float(np.hypot(img_matched[:, 0] - _cx0,
                                         img_matched[:, 1] - _cy0).max())

            if img_stars_all is None or len(img_stars_all) <= n_matched:
                print("[GaiaLocal] recover: no unpaired detections to try")
            else:
                _sep_now = matched_sky.separation(
                    wcs_solution.pixel_to_world(img_matched[:, 0],
                                                img_matched[:, 1])).arcsec
                _rms_now = float(np.sqrt(np.mean(_sep_now ** 2)))
                _px = _scale_conv or 1.0
                _rtol = float(np.clip(RECOVER_K * _rms_now / _px,
                                      RECOVER_MIN_PX, RECOVER_MAX_PX))

                # Fresh catalogue box from the CURRENT solution, magnitude
                # limited the same way every other query is.
                _rc = [wcs_solution.pixel_to_world(x, y) for x, y in
                       ((0, 0), (img_w - 1, 0), (0, img_h - 1),
                        (img_w - 1, img_h - 1))]
                _rra0, _rra1, _rd0, _rd1 = _field_sky_box(_rc, 0.05)
                _rg = _catalog_mag_limit(_rra0, _rra1, _rd0, _rd1, _n_detected)
                _seen = {}
                if use_astro:
                    for _st in astro_lib.find_stars_in_box(
                            _rra0, _rra1, _rd0, _rd1, max_mag=_rg):
                        if _st.source_id not in _seen:
                            _seen[_st.source_id] = (_st.ra, _st.dec)
                else:
                    for _fn, _cn in lib._connections.items():
                        try:
                            for _sid, _sra, _sdec in _cn.execute(
                                    "SELECT source_id, ra, dec FROM sources "
                                    "WHERE dec BETWEEN ? AND ? "
                                    "  AND ra BETWEEN ? AND ? "
                                    "  AND phot_g_mean_mag IS NOT NULL "
                                    "  AND phot_g_mean_mag <= ?",
                                    (_rd0, _rd1, _rra0, _rra1, float(_rg))):
                                if _sid not in _seen:
                                    _seen[_sid] = (_sra, _sdec)
                        except Exception:
                            pass

                if len(_seen) < MIN_PAIRS:
                    print(f"[GaiaLocal] recover: catalogue returned only "
                          f"{len(_seen)} stars, skipping")
                else:
                    _rv = list(_seen.values())
                    _rsky = SkyCoord(ra=[v[0] for v in _rv] * u.deg,
                                     dec=[v[1] for v in _rv] * u.deg)
                    _rpx, _rpy = wcs_solution.world_to_pixel(_rsky)
                    _rxy = np.column_stack([_rpx, _rpy])
                    _rin = ((_rxy[:, 0] > -50) & (_rxy[:, 0] < img_w + 50) &
                            (_rxy[:, 1] > -50) & (_rxy[:, 1] < img_h + 50))
                    _rxy, _rsky = _rxy[_rin], _rsky[_rin]

                    # Hard ceiling from the catalogue's own density: a
                    # tolerance approaching half the mean separation can reach
                    # the WRONG star, which is the one way this pass could do
                    # real harm. At ~10k stars over this frame the spacing is
                    # about 49 px, so the 15 px cap already binds -- but on a
                    # sparse field it would not, and the guard has to come from
                    # the data rather than a constant.
                    _spacing = math.sqrt(img_w * img_h / max(len(_rxy), 1))
                    _rtol = min(_rtol, RECOVER_SPACING_FRAC * _spacing)
                    print(f"[GaiaLocal] recover: catalogue spacing "
                          f"~{_spacing:.0f}px, matching within {_rtol:.1f}px")

                    _all_img = np.asarray(img_stars_all, np.float64)[:, :2]
                    _rtree = KDTree(_rxy)
                    _rd, _ri = _rtree.query(_all_img, k=1, workers=-1)
                    _rkeep = _rd < _rtol
                    _n_rec = int(_rkeep.sum())

                    if _n_rec <= n_matched:
                        print(f"[GaiaLocal] recover: {_n_rec} pairs within "
                              f"{_rtol:.1f}px is no more than the {n_matched} "
                              f"already held, keeping them")
                    else:
                        _r_img = _all_img[_rkeep]
                        _r_sky = _rsky[_ri[_rkeep]]
                        _kw = dict(projection='TAN',
                                   proj_point=_sky_centroid(_r_sky))
                        _rdeg = None
                        try:
                            if getattr(wcs_solution, "sip", None) is not None:
                                _rdeg = int(wcs_solution.sip.a_order)
                        except Exception:
                            _rdeg = None
                        if _rdeg:
                            _kw["sip_degree"] = _rdeg
                        _r_wcs = fit_wcs_from_points(
                            (_r_img[:, 0], _r_img[:, 1]), _r_sky, **_kw)
                        _r_wcs.array_shape = (img_h, img_w)
                        _r_sep = _r_sky.separation(
                            _r_wcs.pixel_to_world(_r_img[:, 0],
                                                  _r_img[:, 1])).arcsec
                        _r_rms = float(np.sqrt(np.mean(_r_sep ** 2)))
                        _cov_after = float(np.hypot(_r_img[:, 0] - _cx0,
                                                    _r_img[:, 1] - _cy0).max())
                        _corner = float(np.hypot(_cx0, _cy0))

                        print(f"[GaiaLocal] recover: {n_matched} -> {_n_rec} pairs "
                              f"within {_rtol:.1f}px of the "
                              + (f"SIP-{_rdeg}" if _rdeg else "TAN")
                              + f" solution; reach "
                                f"{_cov_before / _corner * 100:.0f}% -> "
                                f"{_cov_after / _corner * 100:.0f}% of the way "
                                f"to the corner")
                        # The RMS is EXPECTED to rise -- the edge of a
                        # distorted field is genuinely harder than its middle,
                        # and that rise is the honest number replacing a
                        # flattering one. So a "RMS got worse" test would
                        # refuse exactly the case this exists for; on a
                        # synthetic field the true corner error halved while
                        # RMS went 3.06" -> 23.39".
                        #
                        # What would really be wrong is MIS-pairing, and that
                        # has a different signature: pairs land on neighbouring
                        # stars, so residuals approach half the catalogue
                        # spacing rather than staying inside the tolerance they
                        # were matched within. A refit whose RMS exceeds its own
                        # matching tolerance is incoherent; one inside it is
                        # merely describing a harder region.
                        if _r_rms / _px > _rtol:
                            print(f"[GaiaLocal] recover: rejected — refit RMS "
                                  f"{_r_rms / _px:.1f}px exceeds the {_rtol:.1f}px "
                                  f"it matched within, so these are mismatches "
                                  f"rather than missing stars")
                        else:
                            img_matched = _r_img
                            matched_sky = _r_sky
                            n_matched = _n_rec
                            wcs_solution = _r_wcs
                            print(f"[GaiaLocal] recover: adopted — RMS "
                                  f"{_rms_now:.3f}\" -> {_r_rms:.3f}\" now "
                                  f"describes the WHOLE frame"
                                  + (" (higher, and honest)"
                                     if _r_rms > _rms_now else ""))
        except Exception as e:
            print(f"[GaiaLocal] recover skipped: {e}")

        # Final residuals of the CHOSEN (SIP or TAN) solution — the gate uses these.
        sky_fit    = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
        sep_arcsec = matched_sky.separation(sky_fit).arcsec
        rms        = float(np.sqrt(np.mean(sep_arcsec**2)))
        p95        = float(np.percentile(sep_arcsec, 95))
        print(f"[GaiaLocal] final RMS={rms:.3f}\"  p95={p95:.3f}\"  n={n_matched}")
        _solve_fingerprint(wcs_solution, img_matched, sep_arcsec,
                           img_w, img_h)

        # ── Quality gate — PIXEL-RELATIVE, not raw arcsec ─────────────────────
        # The matcher and NN refinement are already in pixels; the gate must be
        # too, or a wide-field rig gets failed for a sub-pixel solve (e.g. a
        # 7"/px image where 3.0" is 0.4px). Derive the solved plate scale and
        # express the RMS limit in pixels, with an arcsec floor so fine-scale
        # rigs keep their old strictness (no regression at/above ~1.5"/px).
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            _pxsc = proj_plane_pixel_scales(wcs_solution)
            scale_arcsec = float(np.mean(_pxsc[:2])) * 3600.0
        except Exception:
            scale_arcsec = None

        RMS_GATE_PX           = 2.0    # pixel-relative tolerance
        RMS_GATE_FLOOR_ARCSEC = 3.0    # don't loosen below this on fine scales
        QUALITY_MIN_PAIRS     = 20

        if scale_arcsec and scale_arcsec > 0:
            QUALITY_RMS_LIMIT = max(RMS_GATE_FLOOR_ARCSEC, RMS_GATE_PX * scale_arcsec)
            rms_px_str = f", {rms / scale_arcsec:.2f}px @ {scale_arcsec:.2f}\"/px"
        else:
            QUALITY_RMS_LIMIT = RMS_GATE_FLOOR_ARCSEC
            rms_px_str = ""

        if rms > QUALITY_RMS_LIMIT or n_matched < QUALITY_MIN_PAIRS:
            return False, (
                f"match quality too low (RMS={rms:.2f}\"{rms_px_str}, n={n_matched}; "
                f"need RMS<={QUALITY_RMS_LIMIT:.2f}\" and n>={QUALITY_MIN_PAIRS})"
            )
    except Exception as e:
        return False, f"Gaia DR3 solver: refinement exception: {e}"

    # ── SIP fit ────────────────────────────────────────────────────────────
    # SIP fit now runs BEFORE the quality gate (see above); nothing to do here.

    # Recompute final residuals against whichever solution (TAN or SIP) won,
    # and stash on the WCS object for _gaia_build_final_header to pick up.
    try:
        sky_final = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
        res_final = matched_sky.separation(sky_final).arcsec
        wcs_solution._sas_solve_rms  = float(np.sqrt(np.mean(res_final**2)))
        wcs_solution._sas_solve_p95  = float(np.percentile(res_final, 95))
        wcs_solution._sas_solve_nstars = int(n_matched)
    except Exception:
        pass

    if status_cb is not None:
        status_cb(f"Status: Gaia DR3 solve — RMS={rms:.2f}\" ({n_matched} stars)")
    else:
        _set_status_ui(parent, f"Status: Gaia DR3 solve — RMS={rms:.2f}\" ({n_matched} stars)")
    print("Plate Solve Completed Successfully")
    return True, wcs_solution

# ---------------------------------------------------------------------
# Core ASTAP solving for a numpy image + seed header
# ---------------------------------------------------------------------

def _solve_numpy_with_astap(parent, settings, image: np.ndarray, seed_header: Header | None) -> Tuple[bool, Header | str]:
    """
    Normalize → write temp mono FITS → run ASTAP → return the EXACT FITS header ASTAP wrote.
    """
    astap_exe = _get_astap_exe(settings)
    if not astap_exe or not os.path.exists(astap_exe):
        return False, QCoreApplication.translate("PlateSolver", "ASTAP path is not set (see Preferences) or file not found.")

    # normalize and force 2-D luminance in [0,1]
    norm = _normalize_for_astap(image)
    #gray = _to_gray2d_unit(image)
    gray = _to_gray2d_unit(norm)
    # Safety clamp — ensure gray is strictly [0,1] before writing temp FITS
    # regardless of what normalization path was taken above
    mn, mx = float(gray.min()), float(gray.max())
    if mx > mn:
        gray = ((gray - mn) / (mx - mn)).astype(np.float32)
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
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
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: ASTAP failed to start."))
        return False, QCoreApplication.translate("PlateSolver", "Failed to start ASTAP: {0}").format(proc.errorString())

    _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: ASTAP solving…"))
    if not _wait_process(proc, 300000, parent=parent):
        _set_status_ui(parent, QCoreApplication.translate("PlateSolver", "Status: ASTAP timed out."))
        return False, QCoreApplication.translate("PlateSolver", "ASTAP timed out.")

    # Always capture ASTAP's full stdout/stderr — print it regardless of exit
    # code. Note: on Windows, astap.exe is a GUI-subsystem binary and often
    # writes NOTHING to redirected stdout/stderr even when launched via
    # QProcess — the real failure information is encoded only in the process
    # exit code. We decode that here so the console shows the same "F" error
    # the GUI would display.
    astap_out = bytes(proc.readAllStandardOutput()).decode(errors="ignore")
    astap_err = bytes(proc.readAllStandardError()).decode(errors="ignore")

    _ASTAP_EXIT_CODES = {
        0:  "Solution found.",
        1:  "No solution found (the field did not match the reference catalog "
            "within the given search area/scale).",
        2:  "Not enough stars detected in the image to attempt a solve.",
        16: "Error reading the image file (unsupported or corrupt FITS).",
        17: "Error reading the star reference database (catalog files missing "
            "or not configured in ASTAP).",
        32: "Not enough memory.",
        33: "Error writing output files (.wcs/.ini) — check permissions on the "
            "temp directory.",
    }
    exit_code = proc.exitCode()
    exit_desc = _ASTAP_EXIT_CODES.get(exit_code, "Unknown/undocumented exit code.")
    print(f"[ASTAP] exit code: {exit_code}  ->  {exit_desc}")
    if astap_out.strip():
        print("[ASTAP] STDOUT:\n" + astap_out)
    if astap_err.strip():
        print("[ASTAP] STDERR:\n" + astap_err)

    if exit_code != 0:
        try: os.remove(tmp_fit)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return False, QCoreApplication.translate(
            "PlateSolver", "ASTAP exit code {0}: {1}"
        ).format(exit_code, exit_desc)

    if proc.exitCode() != 0:
        try: os.remove(tmp_fit)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            if os.path.exists(sidecar_wcs): os.remove(sidecar_wcs)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return False, QCoreApplication.translate("PlateSolver", "ASTAP returned a non-zero exit code.")

    # Even on exit code 0, ASTAP may have failed to find a solution and
    # simply not written the .wcs sidecar (or written one with no CRVAL/CTYPE).
    # Treat that case as a failure too, instead of silently returning a
    # WCS-less header that breaks downstream SFCC/SSSC.
    if not os.path.exists(sidecar_wcs):
        print(f"[ASTAP] exit code 0 but no .wcs sidecar was written: {sidecar_wcs}")
        try: os.remove(tmp_fit)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return False, QCoreApplication.translate(
            "PlateSolver",
            "ASTAP exited normally but did not produce a WCS solution."
        )

    try:
        _wcs_check = _parse_astap_wcs_file(sidecar_wcs)
        if "CRVAL1" not in _wcs_check and "CTYPE1" not in _wcs_check:
            print(f"[ASTAP] .wcs sidecar exists but has no CRVAL1/CTYPE1: {dict(_wcs_check)}")
            try: os.remove(tmp_fit)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            try: os.remove(sidecar_wcs)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            return False, QCoreApplication.translate(
                "PlateSolver",
                "ASTAP exited normally but the .wcs file contains no solution."
            )
    except Exception as e:
        print(f"[ASTAP] Could not validate .wcs sidecar: {e}")

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

# SIP distortion fitting. Degree 5 carries 36 distortion coefficients, so it
# wants a few hundred matched stars before it is better determined than
# degree 4; below that the penalty below would reject it anyway, and fitting
# it is just wasted seconds.
# Highest SIP order offered. Degree 5 adds the odd-order terms a
# radial lens profile actually needs; a sixth order mostly
# re-describes what 4 already captured, at 14 more parameters.
SIP_MAX_DEGREE = 5
SIP5_MIN_PAIRS = 300
# A higher degree must cut the penalised RMS by at least this factor to be
# adopted. 0.97 = a real 3% gain, not the fraction of a percent that extra
# free parameters always buy on the very points they were fitted to.
SIP_STEP_UP_GAIN = 0.97


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

def tr(s: str) -> str:
    return QCoreApplication.translate("PlateSolver", s)

def plate_solve_numpy_headless(
    parent,
    settings,
    image: np.ndarray,
    seed_header: "Header | None" = None,
) -> "tuple[bool, Header | str]":
    """
    Solve a numpy image array headlessly — no active document required.
    Uses the full ASTAP → astrometry.net fallback chain.
    Returns (ok, merged_header) on success, (False, error_str) on failure.
    """
    from astropy.io.fits import Header

    acq_base: Header | None = None
    if isinstance(seed_header, Header):
        acq_base = _strip_wcs_keys(seed_header.copy())

    ok, res = _solve_numpy_with_fallback(parent, settings, image, seed_header)
    if not ok:
        return False, res

    if isinstance(acq_base, Header) and isinstance(res, Header):
        return True, _merge_wcs_into_base_header(acq_base, res)
    return True, res

def plate_solve_doc_inplace(parent, doc, settings) -> Tuple[bool, Header | str]:
    img = getattr(doc, "image", None)
    if img is None:
        return False, QCoreApplication.translate("PlateSolver", "Active document has no image data.")

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
        _status_popup_open(parent, tr("Status: Preparing plate solve…"))

    ok_solve = False
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
            hdr_final = _sanitized_header_for_wcs(hdr) if isinstance(hdr, Header) else Header()

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
        _post_solve_metadata_cleanup(doc.metadata, hdr_final)

        _debug_dump_header("DOC.METADATA['original_header'] AFTER SOLVE", doc.metadata.get("original_header"))
        _debug_dump_header("DOC.METADATA['wcs_header'] AFTER SOLVE", doc.metadata.get("wcs_header"))

        # Notify UI — emit changed first, then force header refresh directly
        if hasattr(doc, "changed"):
            try:
                doc.changed.emit()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        # Walk up to the main window so we can call the mixin methods directly
        def _find_main(w):
            while w is not None:
                if hasattr(w, "_refresh_header_viewer") and hasattr(w, "header_viewer"):
                    return w
                w = getattr(w, "parent", lambda: None)()
            return None

        main_win = _find_main(parent)

        # Force the header dock to reload — bypass any "same doc" guard by
        # calling set_document(None) then set_document(doc) so the dock sees
        # a real change, then fall back to _refresh_header_viewer.
        def _force_header_refresh():
            try:
                hv = getattr(main_win, "header_viewer", None) if main_win else None
                if hv and hasattr(hv, "set_document"):
                    try:
                        hv.set_document(None)   # clear guard state
                    except Exception:
                        pass
                    try:
                        hv.set_document(doc)
                    except Exception:
                        pass
                if main_win and hasattr(main_win, "_refresh_header_viewer"):
                    main_win._refresh_header_viewer(doc)
            except Exception as e:
                import logging
                logging.debug(f"Header refresh suppressed: {e}")

        QTimer.singleShot(0, _force_header_refresh)

        if hasattr(parent, "currentDocumentChanged"):
            QTimer.singleShot(0, lambda: parent.currentDocumentChanged.emit(doc))

        _set_status_ui(parent, tr("Status: Plate solve completed."))


        ok_solve = True
        if headless:
            QTimer.singleShot(1200, _status_popup_close)
        else:
            _status_popup_close()
        return True, hdr
    finally:
        if not ok_solve:
            _status_popup_close()

def _sanitized_header_for_wcs(hdr: Header) -> Header:
    """
    Return a copy of *hdr* that is safe to hand to astropy's WCS().

    ASTAP output and header round-trips sometimes leave numeric WCS/SIP
    keywords stored as *strings* (e.g. A_ORDER='2', CRPIX1='1024.0').
    astropy then compares such a string against an int while parsing and
    raises "'<' not supported between instances of 'str' and 'int'".
    Coercing everything to real int/float first prevents that — this is the
    same coercion _wcs_only_from_header already relies on, extended to a few
    extra keywords so no stray string value can trip a comparison deeper in.
    """
    hdr = _strip_nonfits_meta_keys_from_header(hdr)
    d = _ensure_ctypes(_coerce_wcs_numbers(dict(hdr)))

    extra_int   = re.compile(r"^(?:NAXIS\d*|WCSAXES)$")
    extra_float = re.compile(r"^(?:PC\d+_\d+|PV\d+_\d+|LONPOLE|LATPOLE|"
                             r"CRPIX\d+|CRVAL\d+|CDELT\d+|CD\d+_\d+|CROTA\d+)$")
    for k in list(d.keys()):
        ku = str(k).upper()
        v  = d[k]
        if not isinstance(v, str):
            continue
        try:
            if extra_int.match(ku):
                d[k] = int(float(v.strip().split()[0]))
            elif extra_float.match(ku):
                d[k] = float(v.strip().split()[0])
        except Exception:
            pass

    out = Header()
    for k, v in d.items():
        try:
            out[k] = v
        except Exception:
            pass
    return out


def _wcs_from_header_2d(hdr: Header, *, relax: bool = True) -> WCS:
    """
    Build a 2-D celestial WCS even if the FITS header advertises NAXIS=3 (RGB cube).
    This avoids WCSLIB SIP/distortion errors with 3D core WCS.
    """
    hdr = _sanitized_header_for_wcs(hdr)
    return WCS(hdr, naxis=2, relax=relax)


def _estimate_scale_arcsec_from_header(hdr: Header) -> float | None:
    """
    Estimate pixel scale in arcsec/pixel from a FITS Header.
    Tries WCS, then CD matrix, then PC*CDELT, then PIXSCALE-style keys.
    Returns None if we can't get a sane value.
    """
    hdr = _strip_nonfits_meta_keys_from_header(hdr)

    def _sane_arcsec(val: float | None) -> float | None:
        """Return val only if it looks like a plausible arcsec/pixel scale."""
        if val is None or not np.isfinite(val) or val <= 0:
            return None
        # If the value is suspiciously tiny it's probably still in degrees —
        # try multiplying by 3600 once more as a rescue.
        if val < 0.01:
            val_rescued = val * 3600.0
            if 0.01 <= val_rescued <= 1000.0:
                print(f"[scale] value {val} looks like degrees/px, rescuing → {val_rescued} arcsec/px")
                return val_rescued
            return None   # still bad, give up
        if val > 1000.0:
            return None   # implausibly large
        return float(val)

    # 1) Try astropy WCS
    try:
        w = _wcs_from_header_2d(hdr, relax=True)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales_deg = proj_plane_pixel_scales(w)   # degrees/pixel
        if scales_deg is not None and len(scales_deg) >= 2:
            s_arcsec = float(np.mean(scales_deg[:2])) * 3600.0
            result = _sane_arcsec(s_arcsec)
            if result is not None:
                return result
    except Exception as e:
        print("Seed: WCS->scale via proj_plane_pixel_scales failed:", e)

    # 2) CD matrix directly (values are in degrees/pixel)
    cd11 = hdr.get("CD1_1")
    cd21 = hdr.get("CD2_1")
    try:
        if cd11 is not None or cd21 is not None:
            s_arcsec = ((float(cd11 or 0.0)**2 + float(cd21 or 0.0)**2)**0.5) * 3600.0
            result = _sane_arcsec(s_arcsec)
            if result is not None:
                return result
    except Exception as e:
        print("Seed: CD-based scale failed:", e)

    # 3) PC * CDELT (CDELT is in degrees/pixel)
    try:
        cdelt1 = hdr.get("CDELT1")
        cdelt2 = hdr.get("CDELT2")
        pc11   = hdr.get("PC1_1")
        pc21   = hdr.get("PC2_1")
        cd11_v = float(cdelt1) * float(pc11) if (cdelt1 is not None and pc11 is not None) else None
        cd21_v = float(cdelt2) * float(pc21) if (cdelt2 is not None and pc21 is not None) else None
        if cd11_v is not None or cd21_v is not None:
            s_arcsec = ((cd11_v or 0.0)**2 + (cd21_v or 0.0)**2)**0.5 * 3600.0
            result = _sane_arcsec(s_arcsec)
            if result is not None:
                return result
    except Exception as e:
        print("Seed: PC*CDELT-based scale failed:", e)

    # 4) Explicit pixscale keywords (already in arcsec/pixel)
    for key in ("PIXSCALE", "SECPIX", "SECPIX1", "SECPIX2", "SCALE", "PLATESCL", "PLTSCALE"):
        if key in hdr:
            try:
                result = _sane_arcsec(float(hdr[key]))
                if result is not None:
                    return result
            except Exception:
                pass

    # 5) Pixel size + focal length
    px_um_x  = _first_float(hdr.get("XPIXSZ"))
    px_um_y  = _first_float(hdr.get("YPIXSZ"))
    focal_mm = _first_float(hdr.get("FOCALLEN"))
    if focal_mm and (px_um_x or px_um_y):
        px_um = px_um_x or px_um_y
        if px_um_x and px_um_y:
            px_um = (px_um_x + px_um_y) / 2.0
        bx = _first_int(hdr.get("XBINNING")) or _first_int(hdr.get("XBIN")) or 1
        by = _first_int(hdr.get("YBINNING")) or _first_int(hdr.get("YBIN")) or 1
        px_um_eff = px_um * ((bx + by) / 2.0)
        result = _sane_arcsec(206.264806 * px_um_eff / float(focal_mm))
        if result is not None:
            return result

    # 6) PIXSIZE1/PIXSIZE2 (base pixel size in microns, multiply by binning)
    px_um_x  = _first_float(hdr.get("PIXSIZE1"))
    px_um_y  = _first_float(hdr.get("PIXSIZE2"))
    focal_mm = _first_float(hdr.get("FOCALLEN"))
    if focal_mm and (px_um_x or px_um_y):
        px_um = px_um_x or px_um_y
        if px_um_x and px_um_y:
            px_um = (px_um_x + px_um_y) / 2.0
        bx = _first_int(hdr.get("XBINNING")) or _first_int(hdr.get("XBIN")) or 1
        by = _first_int(hdr.get("YBINNING")) or _first_int(hdr.get("YBIN")) or 1
        px_um_eff = px_um * ((bx + by) / 2.0)
        result = _sane_arcsec(206.264806 * px_um_eff / float(focal_mm))
        if result is not None:
            return result

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

    # Prefer Header objects only (canonical). Ignore strings by default to avoid stale blobs.
    if isinstance(raw_wcs, Header):
        wcs_hdr = raw_wcs
    elif isinstance(raw_wcs, str):
        # Optional: only accept string blobs if we DO NOT already have original_header WCS
        # (this prevents stale string overrides from winning)
        if not isinstance(base, Header) or ("CRVAL1" not in base and "CD1_1" not in base and "PC1_1" not in base):
            try:
                wcs_hdr = fits.Header.fromstring(raw_wcs, sep="\n")
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
        return False, QCoreApplication.translate("PlateSolver", "No active document to plate-solve.")

    return plate_solve_doc_inplace(parent, doc, settings)

def _wcs_only_from_header(hdr: Header) -> Header:
    """
    Extract only WCS/SIP-ish cards from a header and return them as a new Header.
    This is what we store in meta['wcs_header'] so it can never go stale or include junk.
    """
    if not isinstance(hdr, Header):
        return Header()

    hdr = _strip_nonfits_meta_keys_from_header(hdr)

    w = Header()
    wcs_prefixes = (
        "CRPIX", "CRVAL", "CDELT", "CD1_", "CD2_", "PC",
        "CTYPE", "CUNIT", "PV1_", "PV2_", "A_", "B_", "AP_", "BP_"
    )
    wcs_extras = {
        "WCSAXES", "LATPOLE", "LONPOLE", "EQUINOX",
        "RADESYS", "RADECSYS", "PLTSOLVD", "WARNING",
        "CROTA1", "CROTA2", "A_ORDER","B_ORDER","AP_ORDER","BP_ORDER",
    }

    for k, v in hdr.items():
        ku = str(k).upper()
        if ku.startswith(wcs_prefixes) or ku in wcs_extras:
            try:
                w[ku] = v
            except Exception:
                pass

    # numeric coercion + TAN-SIP normalization
    d = _ensure_ctypes(_coerce_wcs_numbers(dict(w)))

    try:
        sip_present = any(re.match(r"^(A|B|AP|BP)_\d+_\d+$", kk) for kk in d.keys())
        if sip_present:
            if not str(d.get("CTYPE1", "RA---TAN")).endswith("-SIP"):
                d["CTYPE1"] = "RA---TAN-SIP"
            if not str(d.get("CTYPE2", "DEC--TAN")).endswith("-SIP"):
                d["CTYPE2"] = "DEC--TAN-SIP"
    except Exception:
        pass

    out = Header()
    for k, v in d.items():
        try:
            out[k] = v
        except Exception:
            pass
    return out


def _post_solve_metadata_cleanup(meta: dict, hdr_final: Header) -> None:
    """
    Make solved WCS canonical in metadata and eliminate stale/legacy WCS sources.
    Mutates meta in place.
    """
    # Numerically sanitize once so original_header, the WCS object, and the
    # WCS-only header below are all built from clean int/float keywords.
    if isinstance(hdr_final, Header):
        hdr_final = _sanitized_header_for_wcs(hdr_final)

    # Canonical current header
    meta["original_header"] = hdr_final

    # Canonical WCS object
    try:
        meta["wcs"] = WCS(hdr_final, naxis=2)
    except Exception as e:
        print("post_solve: WCS build failed:", e)
        meta.pop("wcs", None)

    # Canonical WCS header (WCS-only)
    meta["wcs_header"] = _wcs_only_from_header(hdr_final)

    # Kill common stale/duplicate keys (case variants + old conventions)
    for k in (
        "WCS_HEADER", "FITS_HEADER", "ORIGINAL_HEADER", "PRE_SOLVE_HEADER", "__HEADER_SNAPSHOT__",
        "FILE_PATH", "BIT_DEPTH", "WCSAXES"
    ):
        # only remove the ALLCAPS legacy variants — keep your lowercase canonical ones
        if k in meta:
            meta.pop(k, None)

    # Invalidate caches that depend on WCS (these names are examples; add yours)
    for k in (
        "SFCC_star_list", "SFCC_catalog", "SFCC_wcs_sig",
        "star_list", "catalog_stars", "matched_stars", "photometry_cache"
    ):
        if k in meta:
            meta[k] = None  # set-to-None works even if metadata merges elsewhere

# ---------- Solve summary popup ----------

def _fmt_ra_hms(ra_deg: float) -> str:
    ra_h = (float(ra_deg) % 360.0) / 15.0
    h = int(ra_h)
    m = int((ra_h - h) * 60.0)
    s = (ra_h - h - m / 60.0) * 3600.0
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def _fmt_dec_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "\u2212"
    d = abs(float(dec_deg))
    dd = int(d)
    mm = int((d - dd) * 60.0)
    ss = (d - dd - mm / 60.0) * 3600.0
    return f"{sign}{dd:02d}\u00b0 {mm:02d}' {ss:04.1f}\""


def _fmt_fov(deg: float) -> str:
    if deg >= 1.0:
        return f"{deg:.2f}\u00b0"
    if deg * 60.0 >= 1.0:
        return f"{deg * 60.0:.1f}'"
    return f"{deg * 3600.0:.1f}\""


def show_solve_summary(parent, hdr: Header, img_w: int, img_h: int,
                       seed_ra: float | None = None,
                       seed_dec: float | None = None):
    """
    Non-modal success popup with plate-solve facts.
    Only call this from interactive (dialog) paths — never headless.
    """
    if not isinstance(hdr, Header):
        return

    lines = []
    try:
        w = _wcs_from_header_2d(hdr, relax=True)
        cx, cy = img_w / 2.0, img_h / 2.0
        center = w.pixel_to_world(cx, cy)
        ra_c, dec_c = float(center.ra.deg), float(center.dec.deg)
        lines.append(f"<b>Center:</b>  {_fmt_ra_hms(ra_c)}   {_fmt_dec_dms(dec_c)}")

        from astropy.wcs.utils import proj_plane_pixel_scales
        scales_deg = proj_plane_pixel_scales(w)
        scale = float(np.mean(scales_deg[:2])) * 3600.0
        lines.append(f"<b>Pixel scale:</b>  {scale:.3f}\"/px")

        fov_w = img_w * scale / 3600.0
        fov_h = img_h * scale / 3600.0
        lines.append(f"<b>Field of view:</b>  {_fmt_fov(fov_w)} \u00d7 {_fmt_fov(fov_h)}")

        _fl = _first_float(hdr.get("FOCALLEN"))
        if _fl and _fl > 0:
            lines.append(f"<b>Focal length:</b>  {_fl:.1f} mm  (from solve)")

        # Rotation: angle of North from image +y, measured at center
        try:
            xN, yN = w.world_to_pixel_values(ra_c, dec_c + 10.0 / 3600.0)
            rot = math.degrees(math.atan2(float(xN) - cx, float(yN) - cy))
            if rot < 0:
                rot += 360.0
            lines.append(f"<b>Rotation:</b>  {rot:.2f}\u00b0 (N up = 0\u00b0)")
        except Exception:
            pass

        # Parity from det(CD): det > 0 = mirrored vs standard orientation
        try:
            cd = w.pixel_scale_matrix
            det = float(cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0])
            lines.append(f"<b>Parity:</b>  {'flipped' if det > 0 else 'normal'}")
        except Exception:
            pass

        if seed_ra is not None and seed_dec is not None:
            try:
                d_ra = math.radians(ra_c - seed_ra)
                d0 = math.radians(seed_dec); d1 = math.radians(dec_c)
                cosd = (math.sin(d0) * math.sin(d1)
                        + math.cos(d0) * math.cos(d1) * math.cos(d_ra))
                dist_arcmin = math.degrees(math.acos(max(-1.0, min(1.0, cosd)))) * 60.0
                lines.append(f"<b>Offset from seed:</b>  {dist_arcmin:.2f}'")
            except Exception:
                pass
    except Exception as e:
        print("show_solve_summary: WCS summary failed:", e)

    sip_terms = [k for k in hdr.keys() if re.match(r"^(A|B)_\d+_\d+$", str(k))]
    if sip_terms:
        try:
            order = int(hdr.get("A_ORDER", hdr.get("B_ORDER", 0)))
            lines.append(f"<b>Distortion:</b>  SIP order {order} ({len(sip_terms)} terms)")
        except Exception:
            lines.append("<b>Distortion:</b>  SIP")
    else:
        lines.append("<b>Distortion:</b>  none (linear TAN)")
    # Fit quality (Gaia in-house solver only; ASTAP/asnet don't report RMS)
    try:
        rms = hdr.get("SOLVRMS")
        nst = hdr.get("SOLVNSTR")
        if rms is not None:
            n_txt = f" over {int(nst)} stars" if nst is not None else ""
            px_txt = ""
            try:
                px_txt = f" ({float(rms) / scale:.2f} px)"
            except Exception:
                pass
            lines.append(f"<b>Fit RMS vs Gaia:</b>  {float(rms):.2f}\"{px_txt}{n_txt}")
    except Exception:
        pass
    # A seed-grade solution carries no RMS, so without this it would look
    # exactly like a normal solve in the panel. Say plainly what it is.
    try:
        if str(hdr.get("SOLVSRC", "")).strip().lower() == "quad-seed":
            lines.append(
                "<b>Fit quality:</b>  <span style=\'color:#c87f00\'>seed-grade"
                " — quad index placement only</span><br>"
                "<span style=\'color:#c87f00\'>The field is identified, but "
                "refinement did not reach the accuracy gate: centre and scale "
                "are approximate (expect a few pixels) and there are no "
                "distortion terms. Not suitable for photometry or "
                "mosaicking.</span>")
    except Exception:
        pass
    if not lines:
        return

    host = parent
    if not isinstance(host, QWidget):
        host = QApplication.activeWindow()

    box = QMessageBox(host)
    box.setWindowTitle("Plate Solve Successful")
    box.setIcon(QMessageBox.Icon.Information)
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setText("<h3>\u2714 Plate Solve Successful</h3>" + "<br>".join(lines))
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setWindowModality(Qt.WindowModality.NonModal)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    box.show()
    box.raise_()

# ---------------------------------------------------------------------
# Dialog UI with Active/File and Batch modes
# ---------------------------------------------------------------------

class PlateSolverPresetDialog(QDialog):
    """Compact preset editor used by shortcuts / Function Bundles."""

    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Plate Solver — Preset"))
        spec = normalize_plate_solve_preset(initial)
        form = QFormLayout(self)

        self.cb_seed_mode = QComboBox(self)
        self.cb_seed_mode.addItem(self.tr("Auto (from header)"), "auto")
        self.cb_seed_mode.addItem(self.tr("Manual"), "manual")
        self.cb_seed_mode.addItem(self.tr("None (blind)"), "none")
        self.cb_seed_mode.setCurrentIndex(max(0, self.cb_seed_mode.findData(spec["seed_mode"])))
        form.addRow(self.tr("Seed mode:"), self.cb_seed_mode)

        self.le_ra = QLineEdit(spec["ra"]); self.le_ra.setPlaceholderText(self.tr("RA"))
        self.le_dec = QLineEdit(spec["dec"]); self.le_dec.setPlaceholderText(self.tr("Dec"))
        self.le_scale = QLineEdit("" if spec["scale"] is None else str(spec["scale"]))
        self.le_scale.setPlaceholderText(self.tr('Scale ["/px]'))
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.le_ra, 1)
        seed_row.addWidget(self.le_dec, 1)
        seed_row.addWidget(self.le_scale, 1)
        seed_w = QWidget(self); seed_w.setLayout(seed_row)
        form.addRow(self.tr("Manual RA/Dec/Scale:"), seed_w)

        self.cb_radius_mode = QComboBox(self)
        self.cb_radius_mode.addItem(self.tr("Auto"), "auto")
        self.cb_radius_mode.addItem(self.tr("Value (deg)"), "value")
        self.cb_radius_mode.setCurrentIndex(max(0, self.cb_radius_mode.findData(spec["radius_mode"])))
        self.le_radius_val = QLineEdit(str(spec["radius_value"]))
        rad_row = QHBoxLayout(); rad_row.addWidget(self.cb_radius_mode); rad_row.addWidget(self.le_radius_val)
        rad_w = QWidget(self); rad_w.setLayout(rad_row)
        form.addRow(self.tr("Search radius:"), rad_w)

        self.cb_fov_mode = QComboBox(self)
        self.cb_fov_mode.addItem(self.tr("Compute from scale"), "compute")
        self.cb_fov_mode.addItem(self.tr("Auto"), "auto")
        self.cb_fov_mode.addItem(self.tr("Value (deg)"), "value")
        self.cb_fov_mode.setCurrentIndex(max(0, self.cb_fov_mode.findData(spec["fov_mode"])))
        self.le_fov_val = QLineEdit(str(spec["fov_value"]))
        fov_row = QHBoxLayout(); fov_row.addWidget(self.cb_fov_mode); fov_row.addWidget(self.le_fov_val)
        fov_w = QWidget(self); fov_w.setLayout(fov_row)
        form.addRow(self.tr("FOV:"), fov_w)

        self.cb_solver_pref = QComboBox(self)
        self.cb_solver_pref.addItem(self.tr("Gaia DR3 → ASTAP → Astrometry.net (all)"), "both")
        self.cb_solver_pref.addItem(self.tr("In-House Gaia DR3 only"), "gaia_only")
        self.cb_solver_pref.addItem(self.tr("ASTAP only"), "astap_only")
        self.cb_solver_pref.addItem(self.tr("Astrometry.net only"), "astrometry_only")
        self.cb_solver_pref.setCurrentIndex(max(0, self.cb_solver_pref.findData(spec["solver"])))
        form.addRow(self.tr("Solver:"), self.cb_solver_pref)

        self.chk_blind = QCheckBox(self.tr("Blind-solve when there is no RA/Dec"), self)
        self.chk_blind.setChecked(spec["blind"])
        form.addRow("", self.chk_blind)
        self.chk_force_blind = QCheckBox(
            self.tr("Force blind solve (ignore any RA/Dec seed)"), self)
        self.chk_force_blind.setChecked(spec["force_blind"])
        self.chk_force_blind.toggled.connect(
            lambda on: self.chk_blind.setEnabled(not on))
        self.chk_blind.setEnabled(not spec["force_blind"])
        form.addRow("", self.chk_force_blind)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        try:
            radius_value = float(self.le_radius_val.text().strip())
        except Exception:
            radius_value = 5.0
        try:
            fov_value = float(self.le_fov_val.text().strip())
        except Exception:
            fov_value = 0.0
        try:
            scale = float(self.le_scale.text().strip()) if self.le_scale.text().strip() else None
        except Exception:
            scale = None
        return normalize_plate_solve_preset({
            "seed_mode": self.cb_seed_mode.currentData(),
            "ra": self.le_ra.text().strip(),
            "dec": self.le_dec.text().strip(),
            "scale": scale,
            "radius_mode": self.cb_radius_mode.currentData(),
            "radius_value": radius_value,
            "fov_mode": self.cb_fov_mode.currentData(),
            "fov_value": fov_value,
            "solver": self.cb_solver_pref.currentData(),
            "blind": self.chk_blind.isChecked(),
            "force_blind": self.chk_force_blind.isChecked(),
        })


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
        self.setWindowTitle(self.tr("Plate Solver"))
        self.setMinimumWidth(560)
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        # ---------------- Main containers ----------------
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        # ---- Top row: Mode selector ----
        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("Mode:"), self))
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem(self.tr("Active View"), "Active View")
        self.mode_combo.addItem(self.tr("File"), "File")
        self.mode_combo.addItem(self.tr("Batch"), "Batch")
        top.addWidget(self.mode_combo, 1)
        top.addStretch(1)
        main.addLayout(top)

        # ---- Seeding group (shared) ----
        from PyQt6.QtWidgets import QGroupBox, QFormLayout
        seed_box = QGroupBox(self.tr("Seeding & Constraints"), self)
        seed_form = QFormLayout(seed_box)
        seed_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        seed_form.setHorizontalSpacing(8)
        seed_form.setVerticalSpacing(6)

        # Seed mode
        self.cb_seed_mode = QComboBox(seed_box)
        self.cb_seed_mode.addItem(self.tr("Auto (from header)"), "Auto (from header)")
        self.cb_seed_mode.addItem(self.tr("Manual"), "Manual")
        self.cb_seed_mode.addItem(self.tr("None (blind)"), "None (blind)")
        seed_form.addRow(self.tr("Seed mode:"), self.cb_seed_mode)

        # Manual RA/Dec/Scale row
        manual_row = QHBoxLayout()
        self.le_ra = QLineEdit(seed_box);   self.le_ra.setPlaceholderText(self.tr("RA (e.g. 22:32:14 or 338.1385)"))
        self.le_dec = QLineEdit(seed_box);  self.le_dec.setPlaceholderText(self.tr("Dec (e.g. +40:42:43 or 40.7123)"))
        self.le_scale = QLineEdit(seed_box); self.le_scale.setPlaceholderText(self.tr('Scale [" / px] (e.g. 1.46)'))
        self.btn_lookup = QPushButton(self.tr("Lookup…"), seed_box)
        self.btn_lookup.setToolTip(self.tr("Search the celestial catalog by name (e.g. M42, NGC 1499, Abell 426)"))
        self.btn_lookup.clicked.connect(self._lookup_catalog_object)
        manual_row.addWidget(self.le_ra, 1)
        manual_row.addWidget(self.le_dec, 1)
        manual_row.addWidget(self.le_scale, 1)
        manual_row.addWidget(self.btn_lookup)
        seed_form.addRow(self.tr("Manual RA/Dec/Scale:"), manual_row)
        # Search radius (-r)
        rad_row = QHBoxLayout()
        self.cb_radius_mode = QComboBox(seed_box)
        self.cb_radius_mode.addItem(self.tr("Auto (-r 0)"), "Auto (-r 0)")
        self.cb_radius_mode.addItem(self.tr("Value (deg)"), "Value (deg)")
        self.le_radius_val = QLineEdit(seed_box); self.le_radius_val.setPlaceholderText(self.tr("e.g. 5.0"))
        self.le_radius_val.setFixedWidth(120)
        rad_row.addWidget(self.cb_radius_mode)
        rad_row.addWidget(self.le_radius_val)
        rad_row.addStretch(1)
        seed_form.addRow(self.tr("Search radius:"), rad_row)

        # FOV (-fov)
        fov_row = QHBoxLayout()
        self.cb_fov_mode = QComboBox(seed_box)
        self.cb_fov_mode.addItem(self.tr("Compute from scale"), "Compute from scale")
        self.cb_fov_mode.addItem(self.tr("Auto (-fov 0)"), "Auto (-fov 0)")
        self.cb_fov_mode.addItem(self.tr("Value (deg)"), "Value (deg)")
        self.le_fov_val = QLineEdit(seed_box); self.le_fov_val.setPlaceholderText(self.tr("e.g. 1.80"))
        self.le_fov_val.setFixedWidth(120)
        fov_row.addWidget(self.cb_fov_mode)
        fov_row.addWidget(self.le_fov_val)
        fov_row.addStretch(1)
        seed_form.addRow(self.tr("FOV:"), fov_row)

        # Solver preference
        self.cb_solver_pref = QComboBox(seed_box)
        self.cb_solver_pref.addItem(self.tr("Gaia DR3 → ASTAP → Astrometry.net (all)"), "both")
        self.cb_solver_pref.addItem(self.tr("In-House Gaia DR3 only"),                   "gaia_only")
        self.cb_solver_pref.addItem(self.tr("ASTAP only"),                               "astap_only")
        self.cb_solver_pref.addItem(self.tr("Astrometry.net only"),                      "astrometry_only") 
        seed_form.addRow(self.tr("Solver:"), self.cb_solver_pref)

        self.chk_blind = QCheckBox(
            self.tr("Blind-solve from the local quad index when there is no RA/Dec"),
            seed_box)
        self.chk_blind.setToolTip(self.tr(
            "Identify the field from the star pattern alone — no coordinates and no\n"
            "field of view needed. Build the index on the Gaia DR3 Library dialog,\n"
            "Blind Solve Index tab; it is built from the Gaia data you already have.\n\n"
            "Turn this off to fall back to ASTAP's blind solve instead."))
        seed_form.addRow("", self.chk_blind)

        self.chk_force_blind = QCheckBox(
            self.tr("Force blind solve (ignore any RA/Dec seed)"), seed_box)
        self.chk_force_blind.setToolTip(self.tr(
            "Throw the header pointing away and locate the field from the star\n"
            "pattern, even when the header does contain an RA/Dec.\n\n"
            "Overrides the seed mode above, including manual coordinates.\n\n"
            "Use this when the coordinates are WRONG rather than missing \u2014 a\n"
            "wrong seed is worse than none, because the solver searches around\n"
            "a position the field is not in, and ASTAP then fails on the same\n"
            "bad position afterwards.\n\n"
            "The pixel scale from the header is still used; it comes from focal\n"
            "length and pixel size and stays right even when the mount lied."))
        self.chk_force_blind.toggled.connect(
            lambda on: self.chk_blind.setEnabled(not on))
        seed_form.addRow("", self.chk_force_blind)

        # What the blind solver has to work with, and a way to go change it.
        quad_row = QHBoxLayout()
        self.lbl_quad_status = QLabel("", seed_box)
        self.lbl_quad_status.setWordWrap(True)
        self.lbl_quad_status.setTextFormat(Qt.TextFormat.RichText)
        self.btn_gaia_db = QPushButton(self.tr("Open Gaia Database\u2026"), seed_box)
        self.btn_gaia_db.setToolTip(self.tr(
            "Open the Gaia DR3 Library: download catalogue bands, and build the\n"
            "blind-solve quad index from the data you already have."))
        self.btn_gaia_db.clicked.connect(self._open_gaia_database)
        quad_row.addWidget(self.lbl_quad_status, 1)
        quad_row.addWidget(self.btn_gaia_db, 0, Qt.AlignmentFlag.AlignTop)
        seed_form.addRow("", quad_row)

        # Tooltips
        self.cb_seed_mode.setToolTip(self.tr("Use FITS header, your manual RA/Dec/scale, or blind solve."))
        self.le_scale.setToolTip(self.tr('Pixel scale in arcseconds/pixel (e.g., 1.46).'))
        self.cb_radius_mode.setToolTip(self.tr("ASTAP -r. Auto lets ASTAP choose; Value forces a cone radius."))
        self.cb_fov_mode.setToolTip(self.tr("ASTAP -fov. Compute uses image height × scale; Auto lets ASTAP infer."))

        main.addWidget(seed_box)

        # ---------------- Stacked pages ----------------
        self.stack = QStackedWidget(self)
        main.addWidget(self.stack, 1)

        # Page 0: Active View
        p0 = QWidget(self); l0 = QVBoxLayout(p0)
        l0.addWidget(QLabel(self.tr("Solve the currently active image view."), p0))
        l0.addStretch(1)
        self.stack.addWidget(p0)

        # Page 1: File picker
        p1 = QWidget(self); l1 = QVBoxLayout(p1)
        file_row = QHBoxLayout()
        self.le_path = QLineEdit(p1); self.le_path.setPlaceholderText(self.tr("Choose an image…"))
        btn_browse = QPushButton(self.tr("Browse…"), p1)
        file_row.addWidget(self.le_path, 1); file_row.addWidget(btn_browse)
        l1.addLayout(file_row); l1.addStretch(1)
        self.stack.addWidget(p1)

        # Page 2: Batch
        p2 = QWidget(self); l2 = QVBoxLayout(p2)
        in_row  = QHBoxLayout(); out_row = QHBoxLayout()
        self.le_in  = QLineEdit(p2);  self.le_in.setPlaceholderText(self.tr("Input directory"))
        self.le_out = QLineEdit(p2);  self.le_out.setPlaceholderText(self.tr("Output directory"))
        b_in  = QPushButton(self.tr("Browse Input…"), p2)
        b_out = QPushButton(self.tr("Browse Output…"), p2)
        in_row.addWidget(self.le_in, 1);   in_row.addWidget(b_in)
        out_row.addWidget(self.le_out, 1); out_row.addWidget(b_out)
        self.log = QTextEdit(p2); self.log.setReadOnly(True); self.log.setMinimumHeight(160)
        l2.addLayout(in_row); l2.addLayout(out_row); l2.addWidget(QLabel(self.tr("Status:"), p2)); l2.addWidget(self.log, 1)
        self.stack.addWidget(p2)

        # ---------------- Status + buttons ----------------
        self.status = QLabel("", self)
        self.status.setMinimumHeight(20)
        self.status.setWordWrap(True)
        main.addWidget(self.status)

        btn_row = QHBoxLayout()
        self.btn_apply_vbundle = QPushButton(self.tr("Apply to View Bundle…"), self)
        self.btn_apply_vbundle.setToolTip(
            self.tr("Plate-solve every open view (and on-disk files) in a View Bundle.")
        )
        btn_row.addWidget(self.btn_apply_vbundle)
        btn_row.addStretch(1)
        btn_row.addWidget(make_help_button("plate_solver", self))
        self.btn_go = QPushButton(self.tr("Start"), self)
        self.btn_close = QPushButton(self.tr("Close"), self)
        btn_row.addWidget(self.btn_go)
        btn_row.addWidget(self.btn_close)
        main.addLayout(btn_row)

        from setiastro.saspro.shortcuts import PresetDragHandle
        try:
            from setiastro.saspro.resources import platesolve_path as _ps_icon
        except Exception:
            _ps_icon = None
        drag_row = QHBoxLayout()
        self.preset_drag_handle = PresetDragHandle(
            "plate_solve",
            self.current_preset,
            icon=QIcon(_ps_icon) if _ps_icon else QIcon(),
            tooltip=self.tr(
                "Drag to the canvas to create a Plate Solver shortcut\n"
                "with these settings. Drop on an image or a View Bundle chip\n"
                "to solve headlessly."
            ),
            parent=self,
        )
        drag_row.addWidget(self.preset_drag_handle)
        drag_row.addStretch(1)
        main.addLayout(drag_row)

        # ---------------- Connections ----------------
        self.mode_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        btn_browse.clicked.connect(self._browse_file)
        b_in.clicked.connect(self._browse_in)
        b_out.clicked.connect(self._browse_out)
        self.btn_go.clicked.connect(self._run)
        self.btn_close.clicked.connect(self.close)
        self.btn_apply_vbundle.clicked.connect(self._apply_to_view_bundle)

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
        pref_map = {"both": 0, "gaia_only": 1, "astap_only": 2, "astrometry_only": 3}
        self.cb_solver_pref.setCurrentIndex(
            pref_map.get(_get_solver_preference(self.settings), 0)
        )
        self.chk_blind.setChecked(_get_blind_enabled(self.settings))
        self.chk_force_blind.setChecked(_get_force_blind(self.settings))
        self.chk_blind.setEnabled(not self.chk_force_blind.isChecked())
        self._refresh_quad_status()

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

    def _refresh_quad_status(self):
        """Re-read the quad library and repaint the status line."""
        try:
            self.lbl_quad_status.setText(_quad_index_status_html())
        except Exception as e:
            print(f"[PlateSolver] quad status refresh failed: {e}")

    def _open_gaia_database(self):
        from setiastro.saspro.gaia_database import GaiaDatabaseDialog
        if getattr(self, "_gaia_db_window", None) and self._gaia_db_window.isVisible():
            self._gaia_db_window.raise_()
            self._gaia_db_window.activateWindow()
            return
        self._gaia_db_window = GaiaDatabaseDialog(parent=self)
        try:
            from setiastro.saspro.resources import gaia_path
            self._gaia_db_window.setWindowIcon(QIcon(gaia_path))
        except Exception:
            pass

        def _gone(*_):
            # Layers are usually built from inside that dialog, so the count
            # on this one is stale the moment it closes. Clear the handle and
            # repaint together.
            self._gaia_db_window = None
            self._refresh_quad_status()

        for _sig in ("destroyed", "finished"):
            try:
                getattr(self._gaia_db_window, _sig).connect(_gone)
            except Exception:
                pass
        self._gaia_db_window.show()

    def current_preset(self) -> dict:
        idx = self.cb_seed_mode.currentIndex()
        seed = "auto" if idx == 0 else ("manual" if idx == 1 else "none")
        try:
            scale = float(self.le_scale.text().strip()) if self.le_scale.text().strip() else None
        except Exception:
            scale = None
        try:
            radius_value = float(self.le_radius_val.text().strip()) if self.le_radius_val.text().strip() else 5.0
        except Exception:
            radius_value = 5.0
        try:
            fov_value = float(self.le_fov_val.text().strip()) if self.le_fov_val.text().strip() else 0.0
        except Exception:
            fov_value = 0.0
        pref_idx_map = {0: "both", 1: "gaia_only", 2: "astap_only", 3: "astrometry_only"}
        return normalize_plate_solve_preset({
            "seed_mode": seed,
            "ra": self.le_ra.text().strip(),
            "dec": self.le_dec.text().strip(),
            "scale": scale,
            "radius_mode": "auto" if self.cb_radius_mode.currentIndex() == 0 else "value",
            "radius_value": radius_value,
            "fov_mode": (
                "compute" if self.cb_fov_mode.currentIndex() == 0
                else ("auto" if self.cb_fov_mode.currentIndex() == 1 else "value")
            ),
            "fov_value": fov_value,
            "solver": pref_idx_map.get(self.cb_solver_pref.currentIndex(), "both"),
            "blind": self.chk_blind.isChecked(),
            "force_blind": self.chk_force_blind.isChecked(),
        })

    def seed_from_preset(self, preset: dict | None):
        spec = normalize_plate_solve_preset(preset)
        mode_map = {"auto": 0, "manual": 1, "none": 2}
        self.cb_seed_mode.setCurrentIndex(mode_map.get(spec["seed_mode"], 0))
        self.le_ra.setText(spec["ra"])
        self.le_dec.setText(spec["dec"])
        self.le_scale.setText("" if spec["scale"] is None else str(spec["scale"]))
        self.cb_radius_mode.setCurrentIndex(0 if spec["radius_mode"] == "auto" else 1)
        self.le_radius_val.setText(str(spec["radius_value"]))
        fov_map = {"compute": 0, "auto": 1, "value": 2}
        self.cb_fov_mode.setCurrentIndex(fov_map.get(spec["fov_mode"], 0))
        self.le_fov_val.setText(str(spec["fov_value"]))
        pref_map = {"both": 0, "gaia_only": 1, "astap_only": 2, "astrometry_only": 3}
        self.cb_solver_pref.setCurrentIndex(pref_map.get(spec["solver"], 0))
        self.chk_blind.setChecked(spec["blind"])
        self.chk_force_blind.setChecked(spec["force_blind"])
        self.chk_blind.setEnabled(not spec["force_blind"])

    def _persist_ui_to_settings(self) -> dict:
        spec = self.current_preset()
        apply_plate_solve_preset(self.settings, spec)
        return spec

    def _lookup_catalog_object(self):
        """
        Search the bundled celestial catalog + bright star list by object name
        and populate the manual RA/Dec fields.
        """
        import re as _re
        import os as _os

        # ── Ask user for object name first (fail fast before loading anything) ──
        from PyQt6.QtWidgets import QInputDialog
        query, ok = QInputDialog.getText(
            self,
            self.tr("Object Lookup"),
            self.tr("Enter object name (e.g. M42, NGC 1333, Abell 426, Sh2-155, Sirius):"),
        )
        if not ok or not query.strip():
            return
        query = query.strip()
        q = query.lower().strip()

        # ── Normalisation helpers (shared by both search paths) ───────────────
        def _normalize(s):
            s = str(s).lower().strip()
            s = _re.sub(r'^pk\s*', 'pn-g ', s)
            return _re.sub(
                r'^(ngc|ic|m|ugc|pgc|aco|lbn|ldn|sh2|png|pn-g|abell|sh2-?)\s*(\d)',
                r'\1 \2', s
            )

        q_is_pk  = bool(_re.match(r'^pk[\s\d]', q))
        q_png    = _re.sub(r'^pk\s*', 'pn-g ', q) if q_is_pk else q
        q_spaced = _re.sub(
            r'^(ngc|ic|m|ugc|pgc|aco|lbn|ldn|sh2|png|pn-g|abell|sh2-?|pk|arp)\s*(\d)',
            r'\1 \2', q
        )
        q_compact = _re.sub(r'\s+', '', q)

        def _name_matches(name: str) -> bool:
            n = _normalize(name)
            if q in n or q_spaced in n or q_compact in n:
                return True
            if q_is_pk:
                q_png_s = _re.sub(r'\s+', ' ', q_png).strip()
                q_png_c = _re.sub(r'\s+', '', q_png)
                if q_png in n or q_png_s in n or q_png_c in n:
                    return True
            return False

        # ── Results list: each entry is dict(name, ra, dec, note) ────────────
        results = []

        # ── 1) Bright star list ───────────────────────────────────────────────
        try:
            from setiastro.saspro.bright_stars import BRIGHT_STARS
            for entry in BRIGHT_STARS:
                star_name, ra, dec = str(entry[0]), float(entry[1]), float(entry[2])
                mag = entry[3] if len(entry) > 3 else None
                if _name_matches(star_name):
                    note = f"Star  mag {mag:.2f}" if mag is not None else "Star"
                    results.append({
                        "name": star_name, "ra": ra, "dec": dec,
                        "alt": "", "note": note,
                    })
        except ImportError:
            pass   # bright_stars not available — silently skip
        except Exception as e:
            print(f"[Lookup] bright_stars search error: {e}")

        # ── 2) Celestial catalog ──────────────────────────────────────────────
        try:
            from setiastro.saspro.wims import _load_full_catalog
            import sys as _sys, pandas as pd

            app_root = getattr(_sys, "_MEIPASS",
                               _os.path.dirname(_os.path.abspath(__file__)))
            candidates = [
                _os.path.join(app_root, "data", "catalogs", "celestial_catalog.csv"),
                _os.path.join(app_root, "..", "data", "catalogs", "celestial_catalog.csv"),
                _os.path.join(app_root, "..", "..", "data", "catalogs", "celestial_catalog.csv"),
                _os.path.join(app_root, "celestial_catalog.csv"),
                _os.path.join(app_root, "..", "..", "..", "data", "catalogs", "celestial_catalog.csv"),
            ]
            catalog_path = next(
                (_os.path.normpath(p) for p in candidates if _os.path.exists(p)), None
            )

            if catalog_path:
                df = _load_full_catalog(catalog_path)
                if not df.empty and "Name" in df.columns:
                    name_col    = df["Name"].astype(str).apply(_normalize)
                    altname_col = df.get("Alt Name",
                                         pd.Series(dtype=str)).astype(str).apply(_normalize)

                    mask = (
                        name_col.str.contains(_re.escape(q),          na=False) |
                        name_col.str.contains(_re.escape(q_spaced),   na=False) |
                        name_col.str.contains(_re.escape(q_compact),  na=False) |
                        altname_col.str.contains(_re.escape(q),       na=False) |
                        altname_col.str.contains(_re.escape(q_spaced),na=False) |
                        df["Name"].astype(str).str.lower().str.contains(
                            _re.escape(q_compact), na=False)
                    )
                    if q_is_pk:
                        q_png_s = _re.sub(r'\s+', ' ', q_png).strip()
                        q_png_c = _re.sub(r'\s+', '', q_png)
                        mask = mask | (
                            name_col.str.contains(_re.escape(q_png),  na=False) |
                            name_col.str.contains(_re.escape(q_png_s),na=False) |
                            name_col.str.contains(_re.escape(q_png_c),na=False)
                        )

                    for _, row in df[mask].head(50).iterrows():
                        try:
                            ra  = float(row["RA"])
                            dec = float(row["Dec"])
                        except (ValueError, TypeError):
                            continue
                        alt = str(row.get("Alt Name", ""))
                        if alt in ("nan", "None", "—", ""):
                            alt = ""
                        typ = str(row.get("Type", ""))
                        if typ in ("nan", "None", ""):
                            typ = ""
                        mag_raw = row.get("Magnitude", "")
                        mag_str = f"  mag {mag_raw}" if (
                            mag_raw and str(mag_raw) not in ("nan","None","")) else ""
                        note = f"{typ}{mag_str}".strip() if typ else mag_str.strip()
                        results.append({
                            "name": str(row["Name"]), "ra": ra, "dec": dec,
                            "alt": alt, "note": note,
                        })
        except ImportError:
            pass   # pandas / whatsinmysky not available
        except Exception as e:
            print(f"[Lookup] catalog search error: {e}")

        # ── Nothing found ─────────────────────────────────────────────────────
        if not results:
            QMessageBox.information(
                self, self.tr("Object Lookup"),
                self.tr("No match found for '{0}'.").format(query)
            )
            return

        # ── One result: use it directly ───────────────────────────────────────
        if len(results) == 1:
            chosen = results[0]
        else:
            # De-duplicate by (name, ra, dec) keeping first occurrence
            seen   = set()
            unique = []
            for r in results:
                key = (r["name"].lower(), round(r["ra"], 4), round(r["dec"], 4))
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            results = unique

            if len(results) == 1:
                chosen = results[0]
            else:
                items = []
                for r in results:
                    label = r["name"]
                    if r["alt"]:
                        label += f"  /  {r['alt']}"
                    if r["note"]:
                        label += f"  [{r['note']}]"
                    items.append(label)

                item_str, ok2 = QInputDialog.getItem(
                    self,
                    self.tr("Select Object"),
                    self.tr("Multiple matches — choose one:"),
                    items, 0, False,
                )
                if not ok2:
                    return
                chosen = results[items.index(item_str)]

        # ── Populate fields ───────────────────────────────────────────────────
        self.le_ra.setText(f"{chosen['ra']:.6f}")
        self.le_dec.setText(f"{chosen['dec']:.6f}")

        # Switch seed mode to Manual so the values are actually used
        self.cb_seed_mode.setCurrentIndex(1)

        disp = chosen["name"]
        if chosen.get("alt"):
            disp += f" / {chosen['alt']}"
        self.status.setText(
            self.tr("Loaded: {0}  RA={1:.4f}°  Dec={2:+.4f}°").format(
                disp, chosen["ra"], chosen["dec"])
        )
    # ---------- file/batch pickers ----------
    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, self.tr("Choose Image"),
            "", self.tr("Images (*.fits *.fit *.xisf *.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        )
        if f:
            self.le_path.setText(f)

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Choose input directory"))
        if d: self.le_in.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Choose output directory"))
        if d: self.le_out.setText(d)

    # ---------- actions ----------
    def _run(self):
        if not self.btn_go.isEnabled():
            return
        self.btn_go.setEnabled(False)
        # The Gaia dialog can rebuild the quad index, and refreshing the
        # library closes and reopens the memmaps the running solve is reading
        # from. _run_impl is synchronous but pumps events, so this button is
        # reachable mid-solve unless it is disabled with the Go button.
        try:
            self.btn_gaia_db.setEnabled(False)
        except Exception:
            pass
        try:
            self._run_impl()
        finally:
            self.btn_go.setEnabled(True)
            try:
                self.btn_gaia_db.setEnabled(True)
            except Exception:
                pass

    def _run_impl(self):
        # ── Save all settings first ──────────────────────────────────────────
        spec = self._persist_ui_to_settings()
        pref = spec["solver"]

        # ── ASTAP check only when ASTAP will actually be used ────────────────
        if pref not in ("gaia_only", "astrometry_only"):
            astap_exe = _get_astap_exe(self.settings)
            if not astap_exe or not os.path.exists(astap_exe):
                self.status.setText(self.tr("ASTAP path missing. Set Preferences → ASTAP executable."))
                QMessageBox.warning(self, self.tr("Plate Solver"), self.tr("ASTAP path missing.\nSet it in Preferences → ASTAP executable."))
                return

        # ── Dispatch ─────────────────────────────────────────────────────────
        mode = self.stack.currentIndex()
        if mode == 0:
            doc = _active_doc_from_parent(self.parent())
            if not doc:
                QMessageBox.information(self, self.tr("Plate Solver"), self.tr("No active image view."))
                return
            ok, res = plate_solve_doc_inplace(self, doc, self.settings)
            if ok:
                self.status.setText(self.tr("Solved (WCS + SIP applied to active doc)."))
                try:
                    meta = getattr(doc, "metadata", {}) or {}
                    hdr_final = meta.get("original_header")
                    img = getattr(doc, "image", None)
                    if isinstance(hdr_final, Header) and img is not None:
                        seed_ra = seed_dec = None
                        seed_h = _seed_header_from_meta(meta.get("pre_solve_header") and
                                                        {"original_header": meta["pre_solve_header"]} or meta)
                        if isinstance(seed_h, Header):
                            seed_ra = _parse_ra_deg(seed_h)
                            seed_dec = _parse_dec_deg(seed_h)
                        show_solve_summary(self.parent() or self, hdr_final,
                                           int(img.shape[1]), int(img.shape[0]),
                                           seed_ra=seed_ra, seed_dec=seed_dec)
                except Exception as e:
                    print("solve summary popup failed:", e)
                QTimer.singleShot(0, self.accept)
            else:
                self.status.setText(str(res))
        elif mode == 1:
            path = self.le_path.text().strip()
            if not path:
                QMessageBox.information(self, self.tr("Plate Solver"), self.tr("Choose a file to solve."))
                return
            if not os.path.exists(path):
                QMessageBox.warning(self, self.tr("Plate Solver"), self.tr("Selected file does not exist."))
                return
            self._solve_file(path)
        else:
            self._run_batch()

    def _solve_file(self, path: str):
        # Load using legacy.load_image()
        try:
            image_data, original_header, bit_depth, is_mono = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Plate Solver"), self.tr("Cannot read image:\n{0}").format(e))
            return
        if image_data is None:
            QMessageBox.warning(self, self.tr("Plate Solver"), self.tr("Unsupported or unreadable image."))
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
            hdr_final = _sanitized_header_for_wcs(solver_hdr) if isinstance(solver_hdr, Header) else Header()

        # Save-as using legacy.save_image() with ORIGINAL pixels (not normalized)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Plate-Solved FITS"),
            "",
            self.tr("FITS files (*.fits *.fit)")
        )
        if save_path:
            try:
                # never persist 'file_path' inside FITS
                # Copy the header wholesale, then drop FILE_PATH. Rebuilding
                # it card by card looks equivalent and is not: for a
                # commentary keyword, hdr[k] returns ALL of its cards joined
                # by newlines, and assigning that back raises ValueError on
                # the newline. Any solve that writes two HISTORY cards --
                # a recomputed FOCALLEN, or a seed-grade fallback -- would
                # therefore solve fine and then fail to save.
                h2 = hdr_final.copy()
                for _k in list(h2.keys()):
                    if _k.upper() == "FILE_PATH":
                        try:
                            del h2[_k]
                        except Exception:
                            pass

                save_image(
                    img_array=image_data,
                    filename=save_path,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=h2,
                    is_mono=is_mono
                )
                self.status.setText(self.tr("Solved FITS saved:\n{0}").format(save_path))
                try:
                    show_solve_summary(self.parent() or self, hdr_final,
                                       int(image_data.shape[1]), int(image_data.shape[0]),
                                       seed_ra=_parse_ra_deg(seed_h) if isinstance(seed_h, Header) else None,
                                       seed_dec=_parse_dec_deg(seed_h) if isinstance(seed_h, Header) else None)
                except Exception as e:
                    print("solve summary popup failed:", e)
                QTimer.singleShot(0, self.accept)
            except Exception as e:
                QMessageBox.critical(self, self.tr("Save Error"), self.tr("Failed to save: {0}").format(e))
        else:
            self.status.setText(self.tr("Solved (not saved)."))


    def _run_batch(self):
        in_dir  = self.le_in.text().strip()
        out_dir = self.le_out.text().strip()
        if not in_dir or not os.path.isdir(in_dir):
            QMessageBox.warning(self, self.tr("Batch"), self.tr("Please choose a valid input directory."))
            return
        if not out_dir or not os.path.isdir(out_dir):
            QMessageBox.warning(self, self.tr("Batch"), self.tr("Please choose a valid output directory."))
            return

        exts = {".xisf", ".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        files = [
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not files:
            QMessageBox.information(self, self.tr("Batch"), self.tr("No acceptable image files found."))
            return

        self.log.clear()
        self.log.append(self.tr("Found {0} files. Starting batch…").format(len(files)))
        QApplication.processEvents()

        for path in files:
            base = os.path.splitext(os.path.basename(path))[0]
            out  = os.path.join(out_dir, base + "_plate_solved.fits")
            self.log.append(f"▶ {path}") # Symbol, no need to translate
            QApplication.processEvents()

            try:
                # Load using legacy.load_image()
                image_data, original_header, bit_depth, is_mono = load_image(path)
                if image_data is None:
                    self.log.append(self.tr("  ❌ Failed to load"))
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
                    hdr_final = _sanitized_header_for_wcs(hdr) if isinstance(hdr, Header) else Header()

                # Build header to save (and strip FILE_PATH)
                # Copy the header wholesale, then drop FILE_PATH. Rebuilding
                # it card by card looks equivalent and is not: for a
                # commentary keyword, hdr[k] returns ALL of its cards joined
                # by newlines, and assigning that back raises ValueError on
                # the newline. Any solve that writes two HISTORY cards --
                # a recomputed FOCALLEN, or a seed-grade fallback -- would
                # therefore solve fine and then fail to save.
                h2 = hdr_final.copy()
                for _k in list(h2.keys()):
                    if _k.upper() == "FILE_PATH":
                        try:
                            del h2[_k]
                        except Exception:
                            pass

                # Save using original pixels
                save_image(
                    img_array=image_data,
                    filename=out,
                    original_format="fit",
                    bit_depth="32-bit floating point",
                    original_header=h2,
                    is_mono=is_mono
                )
                self.log.append(self.tr("  ✔ saved: ") + out)

            except Exception as e:
                self.log.append(self.tr("  ❌ error: ") + str(e))

            QApplication.processEvents()

        self.log.append(self.tr("Batch plate solving completed."))

    def _load_view_bundle_choices(self):
        settings = QSettings()
        settings.sync()
        raw = ""
        for key in ("viewbundles/v3", "viewbundles/v2", "viewbundles/v1"):
            raw = settings.value(key, "", type=str) or ""
            if raw:
                break
        try:
            data = json.loads(raw or "[]")
        except Exception:
            data = []
        choices = []
        for b in data:
            if not isinstance(b, dict):
                continue
            name = str(b.get("name") or "Bundle").strip() or "Bundle"
            ptrs = []
            for x in (b.get("doc_ptrs") or []):
                try:
                    ptrs.append(int(x))
                except Exception:
                    pass
            files = [str(p) for p in (b.get("file_paths") or []) if p]
            choices.append((name, ptrs, files))
        return choices

    def _pick_view_bundle(self):
        choices = self._load_view_bundle_choices()
        if not choices:
            QMessageBox.information(self, self.tr("Plate Solver"), self.tr("No View Bundles found."))
            return None
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Apply to View Bundle…"))
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(self.tr("Select a View Bundle:")))
        lb = QListWidget(dlg)
        for name, ptrs, files in choices:
            it = QListWidgetItem(f"{name}  ({len(ptrs)} views, {len(files)} files)")
            it.setData(Qt.ItemDataRole.UserRole, (ptrs, files))
            lb.addItem(it)
        if lb.count():
            lb.setCurrentRow(0)
        v.addWidget(lb, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        cur = lb.currentItem()
        if not cur:
            return None
        return cur.data(Qt.ItemDataRole.UserRole)

    def _solve_bundle_file(self, path: str, preset: dict) -> None:
        from setiastro.saspro.doc_manager import ImageDocument
        img, header, bit_depth, is_mono = load_image(path)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "fits"
        doc = ImageDocument(img, metadata={
            "file_path": path,
            "original_header": header,
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "original_format": ext,
        })
        ok, res = apply_plate_solve_to_doc(self, doc, self.settings, preset)
        if not ok:
            raise RuntimeError(str(res))
        save_image(
            img_array=doc.image,
            filename=path,
            original_format=ext,
            bit_depth=doc.metadata.get("bit_depth", bit_depth or "32-bit floating point"),
            original_header=doc.metadata.get("original_header"),
            is_mono=doc.metadata.get("is_mono", is_mono),
        )

    def _apply_to_view_bundle(self):
        spec = self._persist_ui_to_settings()
        picked = self._pick_view_bundle()
        if not picked:
            return
        ptrs, files = picked
        from setiastro.saspro.view_bundle import _find_main_window, _resolve_doc_and_subwindow
        mw = _find_main_window(self) or self.parent()
        applied = 0
        errors = []
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        except Exception:
            pass
        try:
            for p in ptrs or []:
                doc = None
                sw = None
                if mw is not None:
                    doc, sw = _resolve_doc_and_subwindow(mw, p)
                    if sw is not None and hasattr(mw, "mdi"):
                        try:
                            mw.mdi.setActiveSubWindow(sw)
                            QApplication.processEvents()
                        except Exception:
                            pass
                if doc is None or getattr(doc, "image", None) is None:
                    continue
                ok, res = apply_plate_solve_to_doc(self, doc, self.settings, spec)
                if ok:
                    applied += 1
                    title = getattr(doc, "display_name", lambda: "view")()
                    self.status.setText(self.tr("Solved: {0}").format(title))
                else:
                    errors.append(str(res))
                QApplication.processEvents()
            for path in files or []:
                try:
                    self._solve_bundle_file(path, spec)
                    applied += 1
                    self.status.setText(self.tr("Solved: {0}").format(os.path.basename(path)))
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")
                QApplication.processEvents()
        finally:
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
        if applied == 0 and not errors:
            QMessageBox.information(self, self.tr("Plate Solver"), self.tr("No valid targets in the selected bundle."))
            return
        if errors:
            QMessageBox.warning(
                self,
                self.tr("Plate Solver"),
                self.tr("Solved {0} target(s).\n\nErrors:\n{1}").format(applied, "\n".join(errors[:12])),
            )
        else:
            self.status.setText(self.tr("Plate solve complete for {0} target(s).").format(applied))


def open_plate_solver_with_preset(main_window, preset: dict | None = None):
    settings = getattr(main_window, "settings", None)
    if settings is None:
        settings = QSettings()
    dlg = PlateSolverDialog(settings, parent=main_window)
    try:
        from setiastro.saspro.resources import platesolve_path
        dlg.setWindowIcon(QIcon(platesolve_path))
    except Exception:
        pass
    try:
        if preset:
            dlg.seed_from_preset(preset)
    except Exception:
        pass
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg