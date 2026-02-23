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

            # Build ordered source list: primary -> backup -> tertiary
            sources = []

            fid_primary = extract_drive_file_id(self.primary)
            fid_backup = extract_drive_file_id(self.backup)

            if fid_primary:
                sources.append(("google_drive", fid_primary, "primary (Google Drive)"))
            elif (self.primary or "").strip():
                # Keep a note if user passed something invalid as "primary"
                sources.append(("invalid_drive", self.primary, "primary (invalid Google Drive link/id)"))

            if fid_backup and fid_backup != fid_primary:
                sources.append(("google_drive", fid_backup, "backup (Google Drive)"))
            elif (self.backup or "").strip() and fid_backup == fid_primary:
                # Duplicate backup is harmless, but tell logs/support why we skipped it
                self.progress.emit("Backup Google Drive link matches primary; skipping duplicate.")
            elif (self.backup or "").strip() and not fid_backup:
                sources.append(("invalid_drive", self.backup, "backup (invalid Google Drive link/id)"))

            if self.tertiary:
                sources.append(("http", self.tertiary, "tertiary (GitHub/HTTP mirror)"))

            if not sources:
                raise RuntimeError("No valid model download sources were provided.")

            used_source = None
            errors: list[str] = []

            for idx, (kind, value, label) in enumerate(sources, start=1):
                try:
                    # Check cancellation before starting each source attempt
                    if self.should_cancel and self.should_cancel():
                        raise RuntimeError("Download canceled.")

                    # Clean previous partial target before next attempt
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:
                        pass

                    if kind == "invalid_drive":
                        raise RuntimeError(f"{label}: not a valid Google Drive file link or file ID")

                    if kind == "google_drive":
                        self.progress.emit(f"Trying {label}…")
                        download_google_drive_file(
                            value, tmp,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel
                        )
                        used_source = ("google_drive", value)
                        break

                    if kind == "http":
                        self.progress.emit(f"Trying {label}…")
                        download_http_file(
                            value, tmp,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel
                        )
                        used_source = ("http", value)
                        break

                    raise RuntimeError(f"Unknown source type: {kind}")

                except Exception as e:
                    errors.append(f"{label}: {e}")
                    # If there are more sources left, continue to next one
                    if idx < len(sources):
                        self.progress.emit(f"{label} failed. Trying next mirror…")
                        continue
                    # otherwise loop ends and we fail below

            if not used_source:
                msg = "All model download sources failed."
                if errors:
                    msg += "\n\n" + "\n".join(f"• {e}" for e in errors)
                raise RuntimeError(msg)

            # Optional checksum verification
            actual_sha256 = sha256_file(tmp)
            if self.expected_sha256:
                self.progress.emit("Verifying checksum…")
                if actual_sha256.lower() != self.expected_sha256.lower():
                    raise RuntimeError(
                        f"SHA256 mismatch.\nExpected: {self.expected_sha256}\nGot:      {actual_sha256}"
                    )

            # Install + manifest
            src_kind, src_val = used_source
            manifest = {
                "source": src_kind,
                "source_ref": src_val,
                # Store actual hash when expected hash isn't provided (more useful for support)
                "sha256": self.expected_sha256 or actual_sha256,
            }

            install_models_zip(
                tmp,
                progress_cb=lambda s: self.progress.emit(s),
                manifest=manifest
            )

            # Nice success message showing which mirror worked
            if src_kind == "google_drive":
                self.finished.emit(True, "Models updated successfully (Google Drive).")
            else:
                self.finished.emit(True, "Models updated successfully (GitHub mirror).")

        except Exception as e:
            self.finished.emit(False, str(e))