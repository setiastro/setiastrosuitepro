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
                    
                    if plat.startswith("darwin"):
                        import platform
                        machine = platform.machine().lower()
                        if machine in ("arm64", "aarch64"):
                            key = "macOS_AppleSilicon"
                        else:
                            # Rosetta 2 reports x86_64 even on Apple Silicon —
                            # check sysctl as a fallback before assuming Intel
                            try:
                                import subprocess
                                result = subprocess.run(
                                    ["sysctl", "-n", "hw.optional.arm64"],
                                    capture_output=True, text=True, timeout=2
                                )
                                if result.stdout.strip() == "1":
                                    key = "macOS_AppleSilicon"
                                else:
                                    key = "macOS_Intel"
                            except Exception:
                                key = "macOS_Intel"
                    elif plat.startswith("win"):
                        key = "Windows"
                    elif plat.startswith("linux"):
                        key = "Linux"
                    else:
                        key = ""

                    link = downloads.get(key, "")
                    if not link:
                        QMessageBox.warning(self, self.tr("Download"),
                                            self.tr("No download link available for this platform."))
                        return

                    if plat.startswith("win"):
                        self._start_windows_update_download(link)
                    elif plat.startswith("linux"):
                        self._start_linux_update(latest_str)
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
            try:
                self._maybe_notify_correct_model_available()
            except Exception as e:
                print(f"[models] correct model check failed: {type(e).__name__}: {e}")               
        finally:
            reply.deleteLater()

    def _start_linux_update(self, latest_str: str):
        import os
        import subprocess
        from pathlib import Path
        from PyQt6.QtWidgets import QMessageBox

        update_script = Path.home() / ".local" / "share" / "SASpro" / "update-saspro.sh"

        if not update_script.exists():
            ok = QMessageBox.question(
                self,
                self.tr("Update Available"),
                self.tr(
                    "Version {0} is available.\n\n"
                    "To update, please download the new installer from the releases page.\n\n"
                    "Open the download page now?"
                ).format(latest_str),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if ok == QMessageBox.StandardButton.Yes:
                webbrowser.open("https://github.com/setiastro/setiastrosuitepro/releases/latest")
            return

        ok = QMessageBox.question(
            self,
            self.tr("Update Available"),
            self.tr(
                "Version {0} is available.\n\n"
                "SASpro will close and the updater will run in a terminal.\n\n"
                "Proceed?"
            ).format(latest_str),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ok != QMessageBox.StandardButton.Yes:
            return

        # Launch updater in a terminal and exit
        try:
            for terminal in ["gnome-terminal", "konsole", "xterm", "xfce4-terminal", "mate-terminal"]:
                if subprocess.run(["which", terminal], capture_output=True).returncode == 0:
                    subprocess.Popen([terminal, "--", "bash", str(update_script)])
                    break
            else:
                # No terminal found — run headless
                subprocess.Popen(["bash", str(update_script)])
        except Exception as e:
            QMessageBox.warning(self, self.tr("Update"), self.tr(f"Could not launch updater:\n{e}"))
            return

        from PyQt6.QtWidgets import QApplication
        QApplication.instance().quit()

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

        # Remove Zone.Identifier ADS from downloaded zip (Windows "blocked" flag)
        if sys.platform.startswith("win"):
            try:
                ads_path = str(target_path) + ":Zone.Identifier"
                if os.path.exists(ads_path):
                    os.remove(ads_path)
            except Exception:
                pass

        # Extract zip if needed
        if target_path.suffix.lower() == ".zip":
            # Extract alongside the zip in Downloads (not temp, which has stricter ACLs)
            extract_dir = target_path.parent / f"saspro-update-{target_path.stem}"
            os.makedirs(extract_dir, exist_ok=True)
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

        # Launch installer with elevation (fixes WinError 5 on Windows)
        try:
            if sys.platform.startswith("win"):
                # Remove Zone.Identifier ADS so Windows doesn't block the exe
                try:
                    ads_path = str(installer_path) + ":Zone.Identifier"
                    if os.path.exists(ads_path):
                        os.remove(ads_path)
                except Exception:
                    pass  # Not fatal if ADS removal fails

                # Use ShellExecuteW with "runas" to request UAC elevation
                import ctypes
                ret = ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", str(installer_path), None, None, 1
                )
                if ret <= 32:
                    raise OSError(f"ShellExecuteW failed with code {ret}")
            else:
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

    def _maybe_notify_correct_model_available(self):
        """
        Notify the user once if the Aberration Correction model is now available
        on GitHub but not yet installed on disk.
        """
        from setiastro.saspro.model_manager import correct_model_installed, check_correct_model_available

        # Already installed — nothing to say
        installed = correct_model_installed()

        if installed:
            return

        # Only nag once per session
        if getattr(self, "_correct_model_notified", False):

            return

        # HEAD probe — skip silently if network is unavailable

        available = check_correct_model_available()

        if not available:
            return

        self._correct_model_notified = True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(self.tr("New Cosmic Clarity Model Available"))
        box.setText(self.tr(
            "New Cosmic Clarity Aberration Correction model is now available!\n\n"
            "This model corrects for common stellar aberration artifacts. \n\n"
            "Click 'Download Models' to install it alongside your existing models."
        ))
        dl_btn = box.addButton(self.tr("Download Models…"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(self.tr("Not Now"), QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if box.clickedButton() == dl_btn:
            try:
                self._download_correct_model_only()
            except Exception:
                pass

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

    def _download_correct_model_only(self):
        """Download only the Aberration Correction model supplement, skipping main/walking."""
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt, QThread
        from setiastro.saspro.model_workers import ModelsDownloadWorker

        # Open preferences so status updates are visible, but don't trigger full download
        self._open_preferences_models()

        CORRECT_PRIMARY  = "https://drive.google.com/file/d/11Qb6C46OlJG7rmKM-zOPCkzN4xbRsvIc/view?usp=sharing"
        CORRECT_BACKUP   = "https://drive.google.com/file/d/1XgqKNd8iBgV3LW8CfzGyS4jigxsxIf86/view?usp=sharing"
        CORRECT_TERTIARY = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/SASPro_Models_AI4_Correct.zip"

        # We need a dummy primary/backup for the worker's Phase 1, but we can
        # skip it entirely by passing no valid Drive IDs and no tertiary for main.
        # Easiest: just use the supplement-only path via a minimal worker wrapper.
        # We'll do it directly here using install_models_zip_supplement in a thread.

        pd = QProgressDialog("Downloading Aberration Correction model…", "Cancel", 0, 0, self)
        pd.setWindowTitle("Cosmic Clarity — Aberration Correction Model")
        pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        pd.setAutoClose(True)
        pd.setMinimumDuration(0)
        pd.show()

        from PyQt6.QtCore import QObject, pyqtSignal

        class _CorrectOnlyWorker(QObject):
            progress = pyqtSignal(str)
            finished = pyqtSignal(bool, str)

            def __init__(self, primary, backup, tertiary):
                super().__init__()
                self._primary  = primary
                self._backup   = backup
                self._tertiary = tertiary
                self._cancel   = False

            def cancel(self): self._cancel = True

            def run(self):
                import os, tempfile
                from setiastro.saspro.model_manager import (
                    extract_drive_file_id, download_google_drive_file,
                    download_http_file, install_models_zip_supplement,
                )
                tmp = os.path.join(tempfile.gettempdir(), "saspro_models_correct.zip")
                sources = []
                fid1 = extract_drive_file_id(self._primary or "")
                fid2 = extract_drive_file_id(self._backup or "")
                if fid1:
                    sources.append(("google_drive", fid1, "primary (Google Drive)"))
                if fid2 and fid2 != fid1:
                    sources.append(("google_drive", fid2, "backup (Google Drive)"))
                if self._tertiary:
                    sources.append(("http", self._tertiary, "GitHub mirror"))

                for idx, (kind, value, label) in enumerate(sources, start=1):
                    try:
                        if self._cancel:
                            self.finished.emit(False, "Canceled.")
                            return
                        try:
                            if os.path.exists(tmp): os.remove(tmp)
                        except Exception:
                            pass
                        self.progress.emit(f"Trying {label}…")
                        if kind == "google_drive":
                            download_google_drive_file(
                                value, tmp,
                                progress_cb=lambda s: self.progress.emit(s),
                                should_cancel=lambda: self._cancel,
                            )
                        else:
                            download_http_file(
                                value, tmp,
                                progress_cb=lambda s: self.progress.emit(s),
                                should_cancel=lambda: self._cancel,
                            )
                        install_models_zip_supplement(
                            tmp, progress_cb=lambda s: self.progress.emit(s)
                        )
                        self.finished.emit(True, "Aberration Correction model installed.")
                        return
                    except Exception as e:
                        if idx < len(sources):
                            self.progress.emit(f"{label} failed, trying next… ({e})")
                        else:
                            self.finished.emit(False, str(e))

        self._correct_thread = QThread(self)
        self._correct_worker = _CorrectOnlyWorker(CORRECT_PRIMARY, CORRECT_BACKUP, CORRECT_TERTIARY)
        self._correct_worker.moveToThread(self._correct_thread)
        self._correct_thread.started.connect(self._correct_worker.run, Qt.ConnectionType.QueuedConnection)
        self._correct_worker.progress.connect(pd.setLabelText, Qt.ConnectionType.QueuedConnection)
        pd.canceled.connect(self._correct_worker.cancel, Qt.ConnectionType.QueuedConnection)

        def _done(ok, msg):
            pd.reset(); pd.deleteLater()
            self._correct_thread.quit(); self._correct_thread.wait()
            from PyQt6.QtWidgets import QMessageBox
            if ok:
                QMessageBox.information(self, "Cosmic Clarity", f"✅ {msg}")
                try: self._settings_dlg._refresh_models_status()
                except Exception: pass
            else:
                QMessageBox.warning(self, "Cosmic Clarity", f"❌ {msg}")

        self._correct_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
        self._correct_thread.finished.connect(self._correct_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
        self._correct_thread.finished.connect(self._correct_thread.deleteLater, Qt.ConnectionType.QueuedConnection)
        self._correct_thread.start()

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
