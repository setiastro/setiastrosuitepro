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
import re
from PyQt6.QtCore import QUrl
from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply
from PyQt6.QtWidgets import QMessageBox, QApplication

from PyQt6.QtNetwork import QSslSocket

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
        from PyQt6.QtNetwork import QNetworkAccessManager
        if getattr(self, "_nam", None) is None:
            self._nam = QNetworkAccessManager(self)
            self._nam.finished.connect(self._on_update_reply)

    def _kick_update_check(self, *, interactive: bool):
        """
        Start an update check request.
        """
        self._ensure_network_manager()
        url_str = self.settings.value("updates/url", self._updates_url, type=str) or self._updates_url

        if url_str.lower().startswith("https://"):
            try:
                if not QSslSocket.supportsSsl():
                    if self.statusBar():
                        self.statusBar().showMessage(self.tr("Update check unavailable (TLS missing)."), 8000)
                    if interactive:
                        QMessageBox.information(
                            self, self.tr("Update Check"),
                            self.tr("Update check is unavailable because TLS is not available on this system.")
                        )
                    else:
                        print("[updates] TLS unavailable in Qt; skipping update check.")
                    return
            except Exception as e:
                print(f"[updates] TLS probe failed ({e}); skipping update check.")
                return

        req = QNetworkRequest(QUrl(url_str))
        req.setRawHeader(b"User-Agent", f"SASPro/{self._current_version_str}".encode("utf-8"))

        reply = self._nam.get(req)
        reply.setProperty("interactive", interactive)

    def check_for_updates_now(self):
        """Check for updates interactively (show result to user)."""
        if self.statusBar():
            self.statusBar().showMessage(self.tr("Checking for updates..."))
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
                    self.statusBar().showMessage(self.tr("Update check failed."), 5000)
                if interactive:
                    QMessageBox.warning(
                        self, self.tr("Update Check Failed"),
                        self.tr("Unable to check for updates.\n\n{0}").format(err)
                    )
                else:
                    print(f"[updates] check failed: {err}")
                return

            raw = bytes(reply.readAll())
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception as je:
                if self.statusBar():
                    self.statusBar().showMessage(self.tr("Update check failed (bad JSON)."), 5000)
                if interactive:
                    QMessageBox.warning(
                        self, self.tr("Update Check Failed"),
                        self.tr("Update JSON is invalid.\n\n{0}").format(str(je))
                    )
                else:
                    print(f"[updates] bad JSON: {je!r}")
                return

            latest_str = str(data.get("version", "")).strip()
            notes = str(data.get("notes", "") or "")
            downloads = data.get("downloads", {}) or {}

            if not latest_str:
                if self.statusBar():
                    self.statusBar().showMessage(self.tr("Update check failed (no version)."), 5000)
                if interactive:
                    QMessageBox.warning(
                        self, self.tr("Update Check Failed"),
                        self.tr("Update JSON missing the 'version' field.")
                    )
                else:
                    print("[updates] JSON missing 'version'")
                return

            # ---- PEP 440 version compare ----
            try:
                from packaging.version import Version
                cur_v = Version(str(self._current_version_str).strip())
                latest_v = Version(latest_str)
            except Exception as e:
                if self.statusBar():
                    self.statusBar().showMessage(self.tr("Update check failed (version parse)."), 5000)
                if interactive:
                    QMessageBox.warning(
                        self, self.tr("Update Check Failed"),
                        self.tr("Could not compare versions.\n\nCurrent: {0}\nLatest: {1}\n\n{2}")
                            .format(self._current_version_str, latest_str, str(e))
                    )
                else:
                    print(f"[updates] version parse failed: cur={self._current_version_str!r} latest={latest_str!r} err={e!r}")
                return

            available = latest_v > cur_v

            if available:
                if self.statusBar():
                    self.statusBar().showMessage(self.tr("Update available: {0}").format(latest_str), 5000)

                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setWindowTitle(self.tr("Update Available"))
                installed_norm = str(cur_v)
                reported_norm  = str(latest_v)

                msg_box.setText(
                    self.tr(
                        "An update is available!\n\n"
                        "Installed version: {0}\n"
                        "Available version: {1}"
                    ).format(installed_norm, reported_norm)
                )
                if notes:
                    msg_box.setInformativeText(self.tr("Release Notes:\n{0}").format(notes))
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)

                if downloads:
                    details = "\n".join([f"{k}: {v}" for k, v in downloads.items()])
                    msg_box.setDetailedText(details)

                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    plat = sys.platform
                    key = (
                        "Windows" if plat.startswith("win") else
                        "macOS"   if plat.startswith("darwin") else
                        "Linux"   if plat.startswith("linux") else
                        ""
                    )
                    link = downloads.get(key, "")
                    if not link:
                        QMessageBox.warning(self, self.tr("Download"),
                                            self.tr("No download link available for this platform."))
                        return

                    if plat.startswith("win"):
                        self._start_windows_update_download(link)
                    else:
                        webbrowser.open(link)
            else:
                if self.statusBar():
                    self.statusBar().showMessage(self.tr("You're up to date."), 3000)

                if interactive:
                    # Use the same parsed versions you already computed
                    installed_str = str(self._current_version_str).strip()
                    reported_str  = str(latest_str).strip()

                    # If you have cur_v/latest_v (packaging.Version), use their string forms too
                    try:
                        installed_norm = str(cur_v)   # normalized PEP440 (e.g. 1.6.6.post3)
                        reported_norm  = str(latest_v)
                    except Exception:
                        installed_norm = installed_str
                        reported_norm  = reported_str

                    QMessageBox.information(
                        self,
                        self.tr("Up to Date"),
                        self.tr(
                            "You're already running the latest version.\n\n"
                            "Installed version: {0}\n"
                            "Update source reports: {1}"
                        ).format(installed_norm, reported_norm)
                    )
            try:
                self._maybe_warn_missing_ai4_models(interactive=interactive)
            except Exception as e:
                print(f"[models] ai4 check failed: {type(e).__name__}: {e}")                   
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
                self.tr("Downloading update... {0:.1f} KB / {1:.1f} KB").format(rec / 1024, tot / 1024) if tot > 0 else self.tr("Downloading update...")
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
            QMessageBox.warning(self, self.tr("Update Failed"),
                                self.tr("Could not download update:\n{0}").format(reply.errorString()))
            return

        # Write the .zip
        data = bytes(reply.readAll())
        try:
            with open(target_path, "wb") as f:
                f.write(data)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Update Failed"),
                                self.tr("Could not save update to disk:\n{0}").format(e))
            return

        self.statusBar().showMessage(f"Update downloaded to {target_path}", 5000)

        # Extract zip if needed
        if target_path.suffix.lower() == ".zip":
            extract_dir = Path(tempfile.mkdtemp(prefix="saspro-update-"))
            try:
                with zipfile.ZipFile(target_path, "r") as zf:
                    zf.extractall(extract_dir)
            except Exception as e:
                QMessageBox.warning(self, self.tr("Update Failed"),
                                    self.tr("Could not extract update zip:\n{0}").format(e))
                return

            # Look recursively for an .exe
            exe_cands = list(extract_dir.rglob("*.exe"))
            if not exe_cands:
                QMessageBox.warning(
                    self,
                    "Update Failed",
                    self.tr("Downloaded ZIP did not contain an .exe installer.\nFolder: {0}").format(extract_dir)
                )
                return

            installer_path = exe_cands[0]
        else:
            # In case one day Windows points straight to .exe
            installer_path = target_path

        # Ask to run
        ok = QMessageBox.question(
            self,
            self.tr("Run Installer"),
            self.tr("The update has been downloaded.\n\nRun the installer now? (SAS will close.)"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        # Launch installer
        try:
            subprocess.Popen([str(installer_path)], shell=False)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Update Failed"),
                                self.tr("Could not start installer:\n{0}").format(e))
            return

        # Close app so the installer can overwrite files
        QApplication.instance().quit()


    def _normalize_version_str(self, v: str) -> str:
        v = (v or "").strip()
        # common cases: "v1.2.3", "Version 1.2.3", "1.2.3 (build xyz)"
        v = re.sub(r'^[^\d]*', '', v)                # strip leading non-digits
        v = re.split(r'[\s\(\[]', v, 1)[0].strip()   # stop at whitespace/( or [
        return v

    def _parse_version(self, v: str):
        v = self._normalize_version_str(v)
        if not v:
            return None
        # Prefer packaging if present
        try:
            from packaging.version import Version
            return Version(v)
        except Exception:
            # Fallback: compare numeric dot parts only
            parts = re.findall(r'\d+', v)
            if not parts:
                return None
            # normalize length to 3+ (so 1.2 == 1.2.0)
            nums = [int(x) for x in parts[:6]]
            while len(nums) < 3:
                nums.append(0)
            return tuple(nums)

    def _is_update_available(self, latest_str: str) -> bool:
        cur = self._parse_version(self._current_version_str)
        latest = self._parse_version(latest_str)
        if cur is None or latest is None:
            # If we cannot compare, do NOT claim "up to date".
            # Treat as "unknown" and show a failure message in interactive mode.
            return False
        return latest > cur
    
    def _ai4_required_models(self) -> dict[str, list[str]]:
        # Group them so the dialog can be clearer
        return {
            "Cosmic Clarity Sharpen (AI4)": [
                "deep_sharp_stellar_AI4.pth",
                "deep_sharp_stellar_AI4.onnx",
                "deep_nonstellar_sharp_conditional_psf_AI4.pth",
                "deep_nonstellar_sharp_conditional_psf_AI4.onnx",
            ],
            "Cosmic Clarity Denoise (AI4)": [
                "deep_denoise_mono_AI4.pth",
                "deep_denoise_mono_AI4.onnx",
                "deep_denoise_color_AI4.pth",
                "deep_denoise_color_AI4.onnx",
            ],
        }

    def _find_missing_ai4_models(self) -> dict[str, list[str]]:
        from setiastro.saspro.model_manager import model_path
        missing: dict[str, list[str]] = {}
        for group, files in self._ai4_required_models().items():
            m = [fn for fn in files if not model_path(fn).exists()]
            if m:
                missing[group] = m
        return missing

    def _maybe_warn_missing_ai4_models(self, *, interactive: bool):
        """
        Warn user if AI4 models are missing.
        If missing models exist, ALWAYS alert the user (no “warn once” suppression).
        """
        missing = self._find_missing_ai4_models()
        if not missing:
            return

        from setiastro.saspro.model_manager import models_root, read_installed_manifest
        man = read_installed_manifest() or {}
        src = (man.get("source") or "").strip()
        sha = (man.get("sha256") or "").strip()

        lines = []
        for group, files in missing.items():
            lines.append(group + ":")
            for fn in files:
                lines.append(f"  - {fn}")

        msg = (
            "New Cosmic Clarity AI4 models are required, but they were not found.\n\n"
            f"Models folder:\n{models_root()}\n\n"
            "Missing files:\n" + "\n".join(lines)
        )

        info_bits = []
        if src:
            info_bits.append(f"Installed source: {src}")
        if sha:
            info_bits.append(f"Manifest SHA256: {sha}")
        info = "\n".join(info_bits)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(self.tr("Cosmic Clarity Models Missing"))
        box.setText(self.tr(msg))
        if info:
            box.setInformativeText(self.tr(info))

        dl_btn = box.addButton(self.tr("Download/Update Models…"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Ok)

        box.exec()

        if box.clickedButton() == dl_btn:
            try:
                self._download_models_now()
            except Exception:
                pass

    def _open_preferences_models(self):
        """Open Preferences and focus the AI Models section."""
        from setiastro.saspro.ops.settings import SettingsDialog

        # If you already cache settings dialog elsewhere, reuse it.
        if getattr(self, "_settings_dlg", None) is None:
            # 'self.settings' is your QSettings instance on the main window
            self._settings_dlg = SettingsDialog(self, self.settings)

        dlg = self._settings_dlg
        try:
            dlg.refresh_ui()
        except Exception:
            pass

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        # optional: visually cue the models button
        try:
            btn = dlg.btn_models_update
            btn.setStyleSheet("border:2px solid #f5c542;")  # gold
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2500, lambda: btn.setStyleSheet(""))
        except Exception:
            pass


    def _download_models_now(self):
        """Open Preferences and immediately start the models update workflow."""
        self._open_preferences_models()
        try:
            self._settings_dlg.start_models_update()
        except Exception:
            pass
