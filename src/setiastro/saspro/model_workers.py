# src/setiastro/saspro/model_workers.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal

import os
import tempfile
import zipfile

from setiastro.saspro.model_manager import (
    extract_drive_file_id,
    download_google_drive_file,
    download_http_file,
    install_models_zip,
    sha256_file,
)

class ModelsInstallZipWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, zip_path: str, should_cancel=None):
        super().__init__()
        self.zip_path = zip_path
        self.should_cancel = should_cancel  # optional callable

    def run(self):
        try:
            from setiastro.saspro.model_manager import install_models_zip, sha256_file

            if not self.zip_path or not os.path.exists(self.zip_path):
                raise RuntimeError("ZIP file not found.")

            self.progress.emit("Verifying ZIP…")
            # quick hash (optional but helpful for support logs)
            zhash = sha256_file(self.zip_path)

            manifest = {
                "source": "manual_zip",
                "file": os.path.basename(self.zip_path),
                "sha256": zhash,
            }

            install_models_zip(
                self.zip_path,
                progress_cb=lambda s: self.progress.emit(s),
                manifest=manifest,
            )

            self.finished.emit(True, "Models installed successfully from ZIP.")
        except Exception as e:
            self.finished.emit(False, str(e))


class ModelsDownloadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, primary: str, backup: str, tertiary: str | None = None,
                 expected_sha256: str | None = None, should_cancel=None):
        super().__init__()
        self.primary = primary
        self.backup = backup
        self.tertiary = (tertiary or "").strip() or None
        self.expected_sha256 = (expected_sha256 or "").strip() or None
        self.should_cancel = should_cancel  # callable -> bool

    def run(self):
        try:
            tmp = os.path.join(tempfile.gettempdir(), "saspro_models_latest.zip")

            # 1) Try Google Drive primary/backup (by file id)
            fid_primary = extract_drive_file_id(self.primary)
            fid_backup  = extract_drive_file_id(self.backup)

            drive_ok = False
            used_source = None

            if fid_primary or fid_backup:
                try:
                    if fid_primary:
                        self.progress.emit("Downloading from primary (Google Drive)…")
                        download_google_drive_file(
                            fid_primary, tmp,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel
                        )
                        drive_ok = True
                        used_source = ("google_drive", fid_primary)
                    else:
                        raise RuntimeError("Primary is not a valid Drive file link/id.")
                except Exception:
                    # Try backup if different id
                    if fid_backup and fid_backup != fid_primary:
                        self.progress.emit("Primary failed. Trying backup (Google Drive)…")
                        download_google_drive_file(
                            fid_backup, tmp,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel
                        )
                        drive_ok = True
                        used_source = ("google_drive", fid_backup)

            # 2) If Drive failed (or links weren’t Drive), try GitHub / HTTP tertiary
            if not drive_ok:
                if not self.tertiary:
                    raise RuntimeError(
                        "Google Drive download failed and no tertiary mirror URL was provided."
                    )
                self.progress.emit("Google Drive failed. Trying GitHub mirror…")
                download_http_file(
                    self.tertiary, tmp,
                    progress_cb=lambda s: self.progress.emit(s),
                    should_cancel=self.should_cancel
                )
                used_source = ("http", self.tertiary)

            # 3) Optional checksum
            if self.expected_sha256:
                self.progress.emit("Verifying checksum…")
                got = sha256_file(tmp)
                if got.lower() != self.expected_sha256.lower():
                    raise RuntimeError(f"SHA256 mismatch.\nExpected: {self.expected_sha256}\nGot:      {got}")

            # 4) Install + manifest
            src_kind, src_val = used_source if used_source else ("unknown", "")
            manifest = {
                "source": src_kind,
                "source_ref": src_val,
                "sha256": self.expected_sha256 or "",
            }

            install_models_zip(tmp, progress_cb=lambda s: self.progress.emit(s), manifest=manifest)
            self.finished.emit(True, "Models updated successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))

