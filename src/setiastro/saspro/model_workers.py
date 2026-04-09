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
        self.should_cancel = should_cancel

    def run(self):
        try:
            from setiastro.saspro.model_manager import install_models_zip_supplement, sha256_file

            if not self.zip_path or not os.path.exists(self.zip_path):
                raise RuntimeError("ZIP file not found.")

            self.progress.emit("Verifying ZIP…")
            zhash = sha256_file(self.zip_path)
            self.progress.emit(f"SHA256: {zhash[:16]}…")

            # Use supplement installer — overlays files without deleting existing models
            install_models_zip_supplement(
                self.zip_path,
                progress_cb=lambda s: self.progress.emit(s),
            )

            self.finished.emit(True, "Models installed successfully from ZIP.")
        except Exception as e:
            self.finished.emit(False, str(e))

class ModelsDownloadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, primary: str, backup: str, tertiary: str | None = None,
                 expected_sha256: str | None = None, should_cancel=None,
                 walking_zip_url: str | None = None,
                 walking_zip_backup: str | None = None,
                 walking_zip_tertiary: str | None = None):
        super().__init__()
        self.primary = primary
        self.backup = backup
        self.tertiary = (tertiary or "").strip() or None
        self.expected_sha256 = (expected_sha256 or "").strip() or None
        self.should_cancel = should_cancel
        self.walking_zip_url      = (walking_zip_url      or "").strip() or None
        self.walking_zip_backup   = (walking_zip_backup   or "").strip() or None
        self.walking_zip_tertiary = (walking_zip_tertiary or "").strip() or None

    def run(self):
        try:
            import os, tempfile

            tmp_main    = os.path.join(tempfile.gettempdir(), "saspro_models_latest.zip")
            tmp_walking = os.path.join(tempfile.gettempdir(), "saspro_models_walking.zip")

            # ── Phase 1: main zip (Google Drive primary → backup → GitHub) ──
            sources_main = []

            fid_primary = extract_drive_file_id(self.primary)
            fid_backup  = extract_drive_file_id(self.backup)

            if fid_primary:
                sources_main.append(("google_drive", fid_primary, "primary (Google Drive)"))
            if fid_backup and fid_backup != fid_primary:
                sources_main.append(("google_drive", fid_backup,  "backup (Google Drive)"))
            if self.tertiary:
                sources_main.append(("http", self.tertiary, "tertiary (GitHub/HTTP mirror)"))

            if not sources_main:
                raise RuntimeError("No valid model download sources were provided.")

            used_source = None
            errors: list[str] = []

            for idx, (kind, value, label) in enumerate(sources_main, start=1):
                try:
                    if self.should_cancel and self.should_cancel():
                        raise RuntimeError("Download canceled.")
                    try:
                        if os.path.exists(tmp_main):
                            os.remove(tmp_main)
                    except Exception:
                        pass

                    if kind == "google_drive":
                        self.progress.emit(f"Trying {label}…")
                        download_google_drive_file(
                            value, tmp_main,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel,
                        )
                        used_source = ("google_drive", value)
                        break
                    elif kind == "http":
                        self.progress.emit(f"Trying {label}…")
                        download_http_file(
                            value, tmp_main,
                            progress_cb=lambda s: self.progress.emit(s),
                            should_cancel=self.should_cancel,
                        )
                        used_source = ("http", value)
                        break
                except Exception as e:
                    errors.append(f"{label}: {e}")
                    if idx < len(sources_main):
                        self.progress.emit(f"{label} failed. Trying next mirror…")

            if not used_source:
                msg = "All model download sources failed."
                if errors:
                    msg += "\n\n" + "\n".join(f"• {e}" for e in errors)
                raise RuntimeError(msg)

            # Checksum if provided
            actual_sha256 = sha256_file(tmp_main)
            if self.expected_sha256:
                self.progress.emit("Verifying checksum…")
                if actual_sha256.lower() != self.expected_sha256.lower():
                    raise RuntimeError(
                        f"SHA256 mismatch.\nExpected: {self.expected_sha256}\nGot: {actual_sha256}"
                    )

            # Install main zip
            src_kind, src_val = used_source
            manifest = {
                "source": src_kind,
                "source_ref": src_val,
                "sha256": self.expected_sha256 or actual_sha256,
            }
            install_models_zip(
                tmp_main,
                progress_cb=lambda s: self.progress.emit(s),
                manifest=manifest,
            )

            # ── Phase 2: walking noise supplement ──
            walking_sources = []

            fid_w1 = extract_drive_file_id(self.walking_zip_url or "")
            fid_w2 = extract_drive_file_id(self.walking_zip_backup or "")

            if fid_w1:
                walking_sources.append(("google_drive", fid_w1, "walking primary (Google Drive)"))
            if fid_w2 and fid_w2 != fid_w1:
                walking_sources.append(("google_drive", fid_w2, "walking backup (Google Drive)"))
            if self.walking_zip_tertiary:
                walking_sources.append(("http", self.walking_zip_tertiary, "walking tertiary (GitHub)"))

            if walking_sources:
                self.progress.emit("Downloading Walking Noise models (supplement)…")
                walking_ok = False
                for idx, (kind, value, label) in enumerate(walking_sources, start=1):
                    try:
                        if self.should_cancel and self.should_cancel():
                            raise RuntimeError("Download canceled.")
                        try:
                            if os.path.exists(tmp_walking):
                                os.remove(tmp_walking)
                        except Exception:
                            pass

                        if kind == "google_drive":
                            self.progress.emit(f"Trying {label}…")
                            download_google_drive_file(
                                value, tmp_walking,
                                progress_cb=lambda s: self.progress.emit(s),
                                should_cancel=self.should_cancel,
                            )
                        elif kind == "http":
                            self.progress.emit(f"Trying {label}…")
                            download_http_file(
                                value, tmp_walking,
                                progress_cb=lambda s: self.progress.emit(s),
                                should_cancel=self.should_cancel,
                            )

                        from setiastro.saspro.model_manager import install_models_zip_supplement
                        install_models_zip_supplement(
                            tmp_walking,
                            progress_cb=lambda s: self.progress.emit(s),
                        )
                        self.progress.emit("Walking Noise models installed.")
                        walking_ok = True
                        break

                    except Exception as e:
                        if idx < len(walking_sources):
                            self.progress.emit(f"{label} failed, trying next… ({e})")
                        else:
                            self.progress.emit(
                                f"Warning: Walking Noise supplement failed ({e}). "
                                "You can install it later from Settings → AI Models."
                            )

            # Done
            if src_kind == "google_drive":
                self.finished.emit(True, "Models updated successfully (Google Drive).")
            else:
                self.finished.emit(True, "Models updated successfully (GitHub mirror).")

        except Exception as e:
            self.finished.emit(False, str(e))