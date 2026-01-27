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
from typing import Optional, Callable
from urllib.parse import urlparse, parse_qs
from pathlib import Path

APP_FOLDER_NAME = "SetiAstroSuitePro"  # keep stable
ProgressCB = Optional[Callable[[str], None]]


def model_path(filename: str) -> Path:
    p = Path(models_root()) / filename
    return p

def require_model(filename: str) -> Path:
    """
    Return full path to a runtime-managed model file.
    Raises FileNotFoundError with a helpful message if missing.
    """
    p = model_path(filename)
    if not p.exists():
        raise FileNotFoundError(
            f"Model not found: {p}\n"
            f"Expected models in: {models_root()}\n"
            f"Please install/download the Cosmic Clarity models."
        )
    return p

def app_data_root() -> str:
    """
    Frozen-safe persistent data root.
    MUST match the benchmark cache dir base (runtime_torch._user_runtime_dir()).
    Example on Windows:
      C:\\Users\\YOU\\AppData\\Local\\SASpro
    """
    from setiastro.saspro.runtime_torch import _user_runtime_dir
    root = Path(_user_runtime_dir())  # this is what benchmark_cache_dir() uses
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def models_root() -> str:
    p = Path(app_data_root()) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def installed_manifest_path() -> str:
    return str(Path(models_root()) / "manifest.json")


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
_DRIVE_ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


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
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or (b"<html" in head)


def _parse_gdrive_download_form(html: str) -> tuple[Optional[str], Optional[dict]]:
    m = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not m:
        return None, None
    action = m.group(1)
    params: dict[str, str] = {}

    for name, val in re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html
    ):
        params[name] = val

    for name in re.findall(
        r'<input[^>]+type="hidden"[^>]+name="([^"]+)"(?![^>]*value=)', html
    ):
        params.setdefault(name, "")

    return action, params

def download_http_file(
    url: str,
    dst_path: str | os.PathLike,
    *,
    progress_cb: ProgressCB = None,
    should_cancel=None,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    import requests

    dst = Path(dst_path)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    with requests.Session() as s:
        log(f"Connecting… {url}")
        r = s.get(url, stream=True, timeout=timeout, allow_redirects=True)
        r.raise_for_status()

        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t_last = time.time()
        done_last = 0

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


def download_google_drive_file(
    file_id: str,
    dst_path: str | os.PathLike,
    *,
    progress_cb: ProgressCB = None,
    should_cancel=None,  # callable -> bool
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Downloads a Google Drive file by ID, handling virus-scan interstitial HTML.
    Writes atomically (dst.part -> dst).
    """
    import requests  # local import to keep import cost down

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
                raise RuntimeError(
                    "Google Drive returned an interstitial HTML page, but the download form could not be parsed."
                )
            log("Google Drive interstitial detected — confirming download…")
            r = s.get(action, params=params, stream=True, timeout=timeout, allow_redirects=True)

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
                        raise RuntimeError(
                            "Google Drive returned HTML instead of the file (permission/confirm issue)."
                        )

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


def install_models_zip(
    zip_path: str | os.PathLike,
    *,
    progress_cb: ProgressCB = None,
    manifest: dict | None = None,
) -> None:
    """
    Extracts a models zip and installs it into models_root(), replacing previous contents.
    Writes manifest.json if provided.
    """
    dst = Path(models_root())

    # Use unique temp dirs per install to avoid collisions
    tmp_extract = Path(tempfile.gettempdir()) / f"saspro_models_extract_{os.getpid()}_{int(time.time())}"
    tmp_stage = Path(tempfile.gettempdir()) / f"saspro_models_stage_{os.getpid()}_{int(time.time())}"

    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    # clean temp (best-effort)
    try:
        shutil.rmtree(tmp_extract, ignore_errors=True)
        shutil.rmtree(tmp_stage, ignore_errors=True)
    except Exception:
        pass

    try:
        log("Extracting models zip…")
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(tmp_extract)

        # Some zips contain a top-level folder; normalize:
        root = tmp_extract
        kids = list(root.iterdir())
        if len(kids) == 1 and kids[0].is_dir():
            root = kids[0]

        # sanity: must contain at least one model file
        any_model = any(p.suffix.lower() in (".pth", ".onnx") for p in root.rglob("*"))
        if not any_model:
            raise RuntimeError("Models zip did not contain any .pth/.onnx files.")

        log(f"Installing to: {dst}")

        # Stage copy
        shutil.copytree(root, tmp_stage)

        # Clear destination contents (keep dst folder stable)
        dst.mkdir(parents=True, exist_ok=True)
        for item in dst.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass

        # Copy staged contents into dst
        for item in tmp_stage.iterdir():
            target = dst / item.name
            if item.is_dir():
                # dirs_exist_ok requires Python 3.8+, you're on 3.12 so OK
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

        if manifest:
            log("Writing manifest…")
            write_installed_manifest(manifest)

        log("Models installed.")
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)
        shutil.rmtree(tmp_stage, ignore_errors=True)


def sha256_file(path: str | os.PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
