# pro/gui/mixins/update_mixin.py
"""
Update check mixin for AstroSuiteProMainWindow.

This mixin contains all functionality for checking for application updates,
downloading updates, and handling the update installation process.
"""
from __future__ import annotations
import json
import sys
import webbrowser
from typing import TYPE_CHECKING

from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply
from PyQt6.QtWidgets import QMessageBox, QApplication

if TYPE_CHECKING:
    pass


class UpdateMixin:
    """
    Mixin for application update functionality.
    
    Provides methods for checking for updates, downloading updates,
    and managing the update installation process.
    """

    # Default URL for update checks
    _updates_url = "https://raw.githubusercontent.com/user/repo/main/updates.json"

    @property
    def _current_version_str(self) -> str:
        """
        Return the current app version as a string.

        Prefer an attribute set on the main window (self._version),
        fall back to any module-level VERSION if present, then "0.0.0".
        """
        v = getattr(self, "_version", None)
        if not v:
            v = globals().get("VERSION", None)
        return str(v or "0.0.0")

    def _ensure_network_manager(self):
        """Ensure the network access manager exists."""
        from PyQt6.QtNetwork import QNetworkAccessManager
        
        if not hasattr(self, "_nam") or self._nam is None:
            self._nam = QNetworkAccessManager(self)
            self._nam.finished.connect(self._on_update_reply)

    def _kick_update_check(self, *, interactive: bool):
        """
        Start an update check request.
        
        Args:
            interactive: If True, show UI feedback for the check
        """
        self._ensure_network_manager()
        url_str = self.settings.value("updates/url", self._updates_url, type=str) or self._updates_url
        req = QNetworkRequest(QUrl(url_str))
        req.setRawHeader(
            b"User-Agent",
            f"SASPro/{self._current_version_str}".encode("utf-8")
        )
        reply = self._nam.get(req)
        reply.setProperty("interactive", interactive)

    def check_for_updates_now(self):
        """Check for updates interactively (show result to user)."""
        if self.statusBar():
            self.statusBar().showMessage("Checking for updates...")
        self._kick_update_check(interactive=True)

    def check_for_updates_startup(self):
        """Check for updates silently at startup."""
        self._kick_update_check(interactive=False)

    def _parse_version_tuple(self, v: str):
        """
        Parse a version string into a tuple for comparison.
        
        Args:
            v: Version string like "1.2.3"
            
        Returns:
            Tuple of integers, or None if parsing fails
        """
        try:
            parts = str(v).strip().split(".")
            return tuple(int(p) for p in parts)
        except Exception:
            return None

    def _on_update_reply(self, reply: QNetworkReply):
        """Handle network reply from update check or download."""
        interactive = bool(reply.property("interactive"))
        
        # Was this the second request (the actual installer download)?
        if bool(reply.property("is_update_download")):
            self._on_windows_update_download_finished(reply)
            return
        
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                err = reply.errorString()
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed.", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        f"Unable to check for updates.\n\n{err}")
                else:
                    print(f"[updates] check failed: {err}")
                return

            raw = bytes(reply.readAll())
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as je:
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed (bad JSON).", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        f"Update JSON is invalid.\n\n{je}")
                else:
                    print(f"[updates] bad JSON: {je}")
                return

            latest_str = str(data.get("version", "")).strip()
            notes = str(data.get("notes", "") or "")
            downloads = data.get("downloads", {}) or {}

            if not latest_str:
                if self.statusBar():
                    self.statusBar().showMessage("Update check failed (no 'version').", 5000)
                if interactive:
                    QMessageBox.warning(self, "Update Check Failed",
                                        "Update JSON missing the 'version' field.")
                else:
                    print("[updates] JSON missing 'version'")
                return

            cur_tuple = self._parse_version_tuple(self._current_version_str)
            latest_tuple = self._parse_version_tuple(latest_str)
            available = bool(latest_tuple and cur_tuple and latest_tuple > cur_tuple)

            if available:
                if self.statusBar():
                    self.statusBar().showMessage(f"Update available: {latest_str}", 5000)
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setWindowTitle("Update Available")
                msg_box.setText(f"A new version ({latest_str}) is available!")
                if notes:
                    msg_box.setInformativeText(f"Release Notes:\n{notes}")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

                if downloads:
                    details = "\n".join([f"{k}: {v}" for k, v in downloads.items()])
                    msg_box.setDetailedText(details)

                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    plat = sys.platform
                    link = downloads.get(
                        "Windows" if plat.startswith("win") else
                        "macOS" if plat.startswith("darwin") else
                        "Linux" if plat.startswith("linux") else "", ""
                    )
                    if not link:
                        QMessageBox.warning(self, "Download", "No download link available for this platform.")
                        return

                    if plat.startswith("win"):
                        # Use in-app updater for Windows
                        self._start_windows_update_download(link)
                    else:
                        # Open browser for other platforms
                        webbrowser.open(link)
            else:
                if self.statusBar():
                    self.statusBar().showMessage("You're up to date.", 3000)
                if interactive:
                    QMessageBox.information(self, "Up to Date",
                                            "You're already running the latest version.")
        finally:
            reply.deleteLater()

    def _is_windows(self) -> bool:
        """Check if running on Windows."""
        return sys.platform.startswith("win")

    def _start_windows_update_download(self, url: str):
        """
        Download the update file for Windows.
        
        Args:
            url: URL to download from
        """
        from PyQt6.QtCore import QStandardPaths
        from pathlib import Path
        import os

        self._ensure_network_manager()

        downloads_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        if not downloads_dir:
            import tempfile
            downloads_dir = tempfile.gettempdir()

        os.makedirs(downloads_dir, exist_ok=True)

        # filename from URL
        fname = url.split("/")[-1] or "setiastrosuitepro_windows.zip"
        target_path = Path(downloads_dir) / fname

        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(
            b"User-Agent",
            f"SASPro/{self._current_version_str}".encode("utf-8")
        )

        reply = self._nam.get(req)
        # mark this reply as "this is the actual installer file, not updates.json"
        reply.setProperty("is_update_download", True)
        reply.setProperty("target_path", str(target_path))

        reply.downloadProgress.connect(
            lambda rec, tot: self.statusBar().showMessage(
                f"Downloading update... {rec / 1024:.1f} KB / {tot / 1024:.1f} KB" if tot > 0 else "Downloading update..."
            )
        )

    def _on_windows_update_download_finished(self, reply: QNetworkReply):
        """Handle completion of Windows update download."""
        from pathlib import Path
        import os
        import zipfile
        import subprocess
        import tempfile

        target_path = Path(reply.property("target_path"))

        if reply.error() != QNetworkReply.NetworkError.NoError:
            QMessageBox.warning(self, "Update Failed",
                                f"Could not download update:\n{reply.errorString()}")
            return

        # Write the .zip
        data = bytes(reply.readAll())
        try:
            with open(target_path, "wb") as f:
                f.write(data)
        except Exception as e:
            QMessageBox.warning(self, "Update Failed",
                                f"Could not save update to disk:\n{e}")
            return

        self.statusBar().showMessage(f"Update downloaded to {target_path}", 5000)

        # Extract zip if needed
        if target_path.suffix.lower() == ".zip":
            extract_dir = Path(tempfile.mkdtemp(prefix="saspro-update-"))
            try:
                with zipfile.ZipFile(target_path, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                QMessageBox.warning(self, "Update Failed",
                                    f"Could not extract update zip:\n{e}")
                return

            # Look recursively for an .exe
            exe_cands = list(extract_dir.rglob("*.exe"))
            if not exe_cands:
                QMessageBox.warning(
                    self,
                    "Update Failed",
                    f"Downloaded ZIP did not contain an .exe installer.\nFolder: {extract_dir}"
                )
                return

            installer_path = exe_cands[0]
        else:
            # In case one day Windows points straight to .exe
            installer_path = target_path

        # Ask to run
        ok = QMessageBox.question(
            self,
            "Run Installer",
            "The update has been downloaded.\n\nRun the installer now? (SAS will close.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        # Launch installer
        try:
            subprocess.Popen([str(installer_path)], shell=False)
        except Exception as e:
            QMessageBox.warning(self, "Update Failed",
                                f"Could not start installer:\n{e}")
            return

        # Close app so the installer can overwrite files
        QApplication.instance().quit()
