# src/setiastro/saspro/model_workers.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal

import os
import tempfile
import zipfile

from setiastro.saspro.model_manager import (
    extract_drive_file_id,
    download_google_drive_file,
    install_models_zip,
    sha256_file,
)

class ModelsDownloadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, primary: str, backup: str, expected_sha256: str | None = None, should_cancel=None):
        super().__init__()
        self.primary = primary
        self.backup = backup
        self.expected_sha256 = (expected_sha256 or "").strip() or None
        self.should_cancel = should_cancel  # callable -> bool

    def run(self):
        try:
            # The inputs should be FILE links (or IDs), not folder links.
            fid = extract_drive_file_id(self.primary) or extract_drive_file_id(self.backup)
            if not fid:
                raise RuntimeError(
                    "Models URL is not a Google Drive *file* link or id.\n"
                    "Please provide a shared file link (…/file/d/<ID>/view) to the models zip."
                )

            tmp = os.path.join(tempfile.gettempdir(), "saspro_models_latest.zip")
            try:
                self.progress.emit("Downloading from primary…")
                download_google_drive_file(fid, tmp, progress_cb=lambda s: self.progress.emit(s), should_cancel=self.should_cancel)
            except Exception as e:
                # Try backup if primary fails AND backup has a different file id
                fid2 = extract_drive_file_id(self.backup)
                if fid2 and fid2 != fid:
                    self.progress.emit("Primary failed. Trying backup…")
                    download_google_drive_file(fid2, tmp, progress_cb=lambda s: self.progress.emit(s), should_cancel=self.should_cancel)
                else:
                    raise

            if self.expected_sha256:
                self.progress.emit("Verifying checksum…")
                got = sha256_file(tmp)
                if got.lower() != self.expected_sha256.lower():
                    raise RuntimeError(f"SHA256 mismatch.\nExpected: {self.expected_sha256}\nGot:      {got}")

            manifest = {
                "source": "google_drive",
                "file_id": fid,
                "sha256": self.expected_sha256,
            }
            install_models_zip(tmp, progress_cb=lambda s: self.progress.emit(s), manifest=manifest)

            self.finished.emit(True, "Models updated successfully.")
        except Exception as e:
            self.finished.emit(False, str(e))
