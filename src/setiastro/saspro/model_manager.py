# src/setiastro/saspro/model_manager.py
from __future__ import annotations

import os
import re
import json
import time
import shutil
import hashlib
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Optional, Callable
from urllib.parse import urlparse, parse_qs
import requests
from pathlib import Path
from PyQt6.QtCore import QStandardPaths


APP_FOLDER_NAME = "SetiAstroSuitePro"  # keep stable
ProgressCB = Optional[Callable[[str], None]]

def app_data_root() -> str:
    # e.g. Windows: C:\Users\X\AppData\Roaming\SetiAstroSuitePro
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    # Qt often returns .../YourOrg/YourApp; you can normalize if you want:
    # but simplest is to just append your folder.
    root = os.path.join(base, APP_FOLDER_NAME)
    os.makedirs(root, exist_ok=True)
    return root


def models_root() -> str:
    p = os.path.join(app_data_root(), "models")
    os.makedirs(p, exist_ok=True)
    return p


def installed_manifest_path() -> str:
    return os.path.join(models_root(), "manifest.json")


def read_installed_manifest() -> dict:
    try:
        with open(installed_manifest_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_installed_manifest(d: dict) -> None:
    try:
        with open(installed_manifest_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# ---------------- Google Drive helpers ----------------

_DRIVE_FILE_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_ID_RE   = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")



def extract_drive_file_id(url_or_id: str) -> Optional[str]:
    s = (url_or_id or "").strip()
    if not s:
        return None
    # raw id
    if re.fullmatch(r"[0-9A-Za-z_-]{10,}", s):
        return s
    try:
        u = urlparse(s)
        if "drive.google.com" not in (u.netloc or "") and "docs.google.com" not in (u.netloc or ""):
            return None
        m = re.search(r"/file/d/([^/]+)", u.path or "")
        if m:
            return m.group(1)
        qs = parse_qs(u.query or "")
        if "id" in qs and qs["id"]:
            return qs["id"][0]
    except Exception:
        return None
    return None

def _looks_like_html_prefix(b: bytes) -> bool:
    head = (b or b"").lstrip()[:256].lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head

def _parse_gdrive_download_form(html: str) -> tuple[Optional[str], Optional[dict]]:
    m = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not m:
        return None, None
    action = m.group(1)
    params = {}
    for name, val in re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html):
        params[name] = val
    for name in re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"(?![^>]*value=)', html):
        params.setdefault(name, "")
    return action, params

def download_google_drive_file(
    file_id: str,
    dst_path: str | os.PathLike,
    *,
    progress_cb: ProgressCB = None,
    should_cancel=None,          # callable -> bool
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Downloads a Google Drive file by ID, handling virus-scan interstitial HTML.
    Writes atomically (dst.part -> dst).
    """
    import requests  # local import

    fid = extract_drive_file_id(file_id) or file_id
    if not fid:
        raise RuntimeError("No Google Drive file id provided.")

    dst = Path(dst_path)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # The “uc” endpoint is best for download
    url = f"https://drive.google.com/uc?export=download&id={fid}"

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    # Clean any old partial
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    with requests.Session() as s:
        log("Connecting to Google Drive…")
        r = s.get(url, stream=True, timeout=timeout, allow_redirects=True)

        ctype = (r.headers.get("Content-Type") or "").lower()

        # If HTML, parse the interstitial "download anyway" form and re-request.
        if "text/html" in ctype:
            html = r.text
            r.close()
            action, params = _parse_gdrive_download_form(html)
            if not action or not params:
                raise RuntimeError("Google Drive returned an interstitial HTML page, but the download form could not be parsed.")
            log("Google Drive interstitial detected — confirming download…")
            r = s.get(action, params=params, stream=True, timeout=timeout, allow_redirects=True)
            ctype = (r.headers.get("Content-Type") or "").lower()

        r.raise_for_status()

        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t_last = time.time()
        done_last = 0

        first = True
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if should_cancel and should_cancel():
                    try:
                        f.close()
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise RuntimeError("Download canceled.")

                if not chunk:
                    continue

                if first:
                    first = False
                    # extra safety: even if content-type lies
                    if _looks_like_html_prefix(chunk[:256]):
                        raise RuntimeError("Google Drive returned HTML instead of the file (permission/confirm issue).")

                f.write(chunk)
                done += len(chunk)

                now = time.time()
                if now - t_last >= 0.5:
                    if total > 0:
                        pct = (done * 100.0) / total
                        log(f"Downloading… {pct:5.1f}% ({done}/{total} bytes)")
                    else:
                        bps = (done - done_last) / max(now - t_last, 1e-9)
                        log(f"Downloading… {done} bytes ({bps/1024/1024:.1f} MB/s)")
                    t_last = now
                    done_last = done

    os.replace(str(tmp), str(dst))
    log(f"Download complete: {dst}")
    return dst


def _drive_confirm_token_from_cookies(cookies) -> Optional[str]:
    # Google uses download_warning cookies for large files
    for k, v in cookies.items():
        if k.startswith("download_warning"):
            return v
    return None
