# src/setiastro/saspro/file_associations.py
"""
File-association management for Seti Astro Suite Pro.

Provides:
  * A dialog (About -> File Associations...) letting the user make SASpro
    the default handler for astronomy and image file types.
  * Per-user registration on Windows (HKCU — no admin required) and Linux
    (~/.local/share/applications + xdg-mime).
  * Guidance for macOS, where the real file-association story is the .app
    bundle's Info.plist and there is no portable user-space API to swap
    defaults from a running process.
  * A detector that figures out whether SASpro is running from the frozen
    installer, a pip install, or a source checkout, and tells the user
    which executable path file associations will point at.

The Inno Setup installer registers the same extensions system-wide at
install time if the user opts in. The in-app dialog is the fallback for:
  * pip-only installs (pip wheels can't legally touch the registry)
  * source checkouts
  * users who skipped the installer checkbox
  * users switching between install methods
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QTextBrowser, QVBoxLayout,
)


# Public URL for the long-form documentation. Updated in one place so the
# dialog and _ManualStepsDialog can both reference it.
WIKI_URL = (
    "https://github.com/setiastro/setiastrosuitepro/wiki/File-Associations"
)


# ─── Extension catalog ──────────────────────────────────────────────────────

# "Nothing but an astronomy tool opens these." Safe to default-check these
# in the dialog, since associating them with SASpro won't clobber anything
# a user is likely to care about.
PRIMARY_ASTRONOMY_EXTENSIONS: list[tuple[str, str]] = [
    (".fits", "FITS Image"),
    (".fit",  "FITS Image"),
    (".fts",  "FITS Image"),
    (".xisf", "XISF Image"),
]

# SASpro can open these, but so can Photoshop, Lightroom, your RAW editor,
# a system image viewer, etc. Opt-in only, with a warning.
SECONDARY_EXTENSIONS: list[tuple[str, str]] = [
    (".tif",  "TIFF Image"),
    (".tiff", "TIFF Image"),
    (".psd",  "Photoshop Document"),
    (".psb",  "Photoshop Large Document"),
    (".exr",  "OpenEXR Image"),
    (".hdr",  "Radiance HDR"),
    (".cr2",  "Canon RAW"),
    (".cr3",  "Canon RAW"),
    (".nef",  "Nikon RAW"),
    (".arw",  "Sony RAW"),
    (".dng",  "Adobe DNG"),
]


# ─── Launcher detection ─────────────────────────────────────────────────────

class LauncherInfo(NamedTuple):
    kind: str          # "frozen" | "pip" | "source"
    exe: str           # path to the thing the OS should launch for file assoc
    display: str       # one-line human description for the dialog
    can_register: bool # True if we have a single-exe path to register against


def detect_launcher() -> LauncherInfo:
    """Figure out how SASpro was started and whether we can register
    file associations for it automatically."""
    # Frozen (PyInstaller installer): sys.executable IS the launcher.
    if getattr(sys, "frozen", False):
        return LauncherInfo(
            kind="frozen",
            exe=sys.executable,
            display=f"Installer build — {sys.executable}",
            can_register=True,
        )

    # Pip install: look for the console-script shim on PATH.
    shim_name = "setiastrosuitepro"
    if sys.platform.startswith("win"):
        shim_name += ".exe"
    shim = shutil.which(shim_name)
    if shim:
        return LauncherInfo(
            kind="pip",
            exe=shim,
            display=f"pip install — {shim}",
            can_register=True,
        )

    # Source checkout: there's no single exe to point associations at.
    script = ""
    try:
        if sys.argv and sys.argv[0]:
            script = os.path.abspath(sys.argv[0])
    except Exception:
        pass
    display = f"Source — {sys.executable}"
    if script:
        display += f"  {script}"
    return LauncherInfo(
        kind="source",
        exe=sys.executable,
        display=display,
        can_register=False,
    )


# ─── Windows per-user registration (HKCU, no admin) ────────────────────────

def _progid_for(ext: str) -> str:
    # e.g. ".fits" -> "SetiAstroSuitePro.fits"
    # (versionless so upgrades don't orphan it.)
    return f"SetiAstroSuitePro{ext.lower()}"


def register_windows_per_user(exe_path: str,
                              extensions: list[str]) -> tuple[int, list[str]]:
    """Write HKCU\\Software\\Classes entries so Explorer treats SASpro as the
    default handler for the given extensions. Returns (success_count, errors).
    """
    import winreg
    count = 0
    errors: list[str] = []

    for ext in extensions:
        progid = _progid_for(ext)
        label = _label_for_extension(ext)
        try:
            # .ext -> ProgID (default verb)
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  f"Software\\Classes\\{ext}") as k:
                winreg.SetValue(k, "", winreg.REG_SZ, progid)

            # .ext -> OpenWithProgids (so SASpro shows up in Open With list
            # even when it's not the default)
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER,
                f"Software\\Classes\\{ext}\\OpenWithProgids",
            ) as k:
                winreg.SetValueEx(k, progid, 0, winreg.REG_SZ, "")

            # ProgID -> description
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                                  f"Software\\Classes\\{progid}") as k:
                winreg.SetValue(k, "", winreg.REG_SZ, label)

            # ProgID -> icon
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER,
                f"Software\\Classes\\{progid}\\DefaultIcon",
            ) as k:
                winreg.SetValue(k, "", winreg.REG_SZ, f'"{exe_path}",0')

            # ProgID -> open command
            with winreg.CreateKey(
                winreg.HKEY_CURRENT_USER,
                f"Software\\Classes\\{progid}\\shell\\open\\command",
            ) as k:
                winreg.SetValue(k, "", winreg.REG_SZ, f'"{exe_path}" "%1"')

            count += 1
        except OSError as e:
            errors.append(f"{ext}: {e}")

    _notify_shell_assoc_changed()
    return count, errors


def unregister_windows_per_user(extensions: list[str]) -> tuple[int, list[str]]:
    """Remove the HKCU entries this app wrote. Doesn't touch HKLM entries
    the installer wrote — those are managed by the installer's uninstaller."""
    import winreg
    count = 0
    errors: list[str] = []

    for ext in extensions:
        progid = _progid_for(ext)

        # Remove the .ext default pointer (ONLY if it points at our progid).
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                f"Software\\Classes\\{ext}",
                                0, winreg.KEY_READ) as k:
                try:
                    current, _ = winreg.QueryValueEx(k, "")
                except FileNotFoundError:
                    current = ""
            if current == progid:
                try:
                    with winreg.CreateKey(
                        winreg.HKEY_CURRENT_USER, f"Software\\Classes\\{ext}"
                    ) as k:
                        winreg.SetValue(k, "", winreg.REG_SZ, "")
                except OSError:
                    pass
        except FileNotFoundError:
            pass

        # Remove our entry from OpenWithProgids
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                f"Software\\Classes\\{ext}\\OpenWithProgids",
                0, winreg.KEY_SET_VALUE,
            ) as k:
                try:
                    winreg.DeleteValue(k, progid)
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass

        # Delete the ProgID tree
        for sub in (
            f"Software\\Classes\\{progid}\\shell\\open\\command",
            f"Software\\Classes\\{progid}\\shell\\open",
            f"Software\\Classes\\{progid}\\shell",
            f"Software\\Classes\\{progid}\\DefaultIcon",
            f"Software\\Classes\\{progid}",
        ):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            except OSError:
                pass

        count += 1

    _notify_shell_assoc_changed()
    return count, errors


def _notify_shell_assoc_changed() -> None:
    """Tell Explorer the associations changed so icons/menus refresh now."""
    try:
        import ctypes
        # SHCNE_ASSOCCHANGED = 0x08000000, SHCNF_IDLIST = 0
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
    except Exception:
        pass


# ─── Linux: .desktop file + xdg-mime ────────────────────────────────────────

def _mime_for_extension(ext: str) -> str:
    """Best-effort MIME type for a given extension. Empty string means
    'we don't know, skip this one for xdg-mime registration'."""
    return {
        ".fits": "image/fits",
        ".fit":  "image/fits",
        ".fts":  "image/fits",
        ".xisf": "application/x-xisf",
        ".tif":  "image/tiff",
        ".tiff": "image/tiff",
        ".psd":  "image/vnd.adobe.photoshop",
        ".psb":  "image/vnd.adobe.photoshop",
        ".exr":  "image/x-exr",
        ".hdr":  "image/vnd.radiance",
        ".cr2":  "image/x-canon-cr2",
        ".cr3":  "image/x-canon-cr3",
        ".nef":  "image/x-nikon-nef",
        ".arw":  "image/x-sony-arw",
        ".dng":  "image/x-adobe-dng",
    }.get(ext.lower(), "")


def _label_for_extension(ext: str) -> str:
    """Human-readable description for a given extension. Falls back to a
    generic '{ext} Image' label for anything not in our catalog."""
    ext = ext.lower()
    for e, label in PRIMARY_ASTRONOMY_EXTENSIONS + SECONDARY_EXTENSIONS:
        if e == ext:
            return label
    return f"{ext.lstrip('.').upper()} Image"


def register_linux_user(exe_path: str,
                        extensions: list[str]) -> tuple[int, list[str]]:
    """Install a user-level .desktop file and claim MIME defaults for the
    extensions the user chose. No root required."""
    errors: list[str] = []
    try:
        apps_dir = Path.home() / ".local" / "share" / "applications"
        apps_dir.mkdir(parents=True, exist_ok=True)
        desktop_path = apps_dir / "setiastrosuitepro.desktop"

        mime_types: list[str] = []
        for ext in extensions:
            mime = _mime_for_extension(ext)
            if mime and mime not in mime_types:
                mime_types.append(mime)

        content = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=Seti Astro Suite Pro\n"
            "GenericName=Astrophotography Processing\n"
            "Comment=Open astronomical images with SASpro\n"
            f"Exec={exe_path} %F\n"
            "Terminal=false\n"
            "Categories=Graphics;Science;Astronomy;\n"
            f"MimeType={';'.join(mime_types)};\n"
        )
        desktop_path.write_text(content, encoding="utf-8")

        for mime in mime_types:
            try:
                subprocess.run(
                    ["xdg-mime", "default",
                     "setiastrosuitepro.desktop", mime],
                    check=False, capture_output=True, timeout=5,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                errors.append(f"xdg-mime {mime}: {e}")

        try:
            subprocess.run(
                ["update-desktop-database", str(apps_dir)],
                check=False, capture_output=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return len(mime_types), errors
    except OSError as e:
        return 0, [str(e)]


def unregister_linux_user() -> tuple[int, list[str]]:
    """Delete our .desktop file. Note: user-chosen defaults in
    ~/.config/mimeapps.list may linger; we don't try to parse that file."""
    try:
        desktop = (Path.home() / ".local" / "share"
                   / "applications" / "setiastrosuitepro.desktop")
        if desktop.exists():
            desktop.unlink()
            return 1, []
        return 0, []
    except OSError as e:
        return 0, [str(e)]


# ─── Dialog ─────────────────────────────────────────────────────────────────

class FileAssociationsDialog(QDialog):
    """About → File Associations… — makes SASpro the default handler for
    astronomy and (optionally) general image formats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("File Associations"))
        self.setMinimumWidth(600)

        self._info = detect_launcher()
        self._primary_checkboxes: list[QCheckBox] = []
        self._secondary_checkboxes: list[QCheckBox] = []

        root = QVBoxLayout(self)

        # ── Launcher info ─────────────────────────────────────────────────
        header = QGroupBox(self.tr("Detected launch method"))
        hl = QFormLayout(header)
        hl.addRow(self.tr("Type:"), QLabel(self._info.kind))
        exe_lbl = QLabel(self._info.display)
        exe_lbl.setWordWrap(True)
        exe_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        hl.addRow(self.tr("Launcher:"), exe_lbl)
        if not self._info.can_register:
            warn = QLabel(self.tr(
                "SASpro is running from source. Automatic registration isn't "
                "supported from here — see “Show manual steps” below for how "
                "to wrap it in a launcher first."
            ))
            warn.setStyleSheet("color: #c0a000;")
            warn.setWordWrap(True)
            hl.addRow(warn)
        root.addWidget(header)

        # ── Primary (astronomy) ───────────────────────────────────────────
        primary_box = QGroupBox(self.tr("Astronomy file types"))
        pv = QVBoxLayout(primary_box)
        pv.addWidget(QLabel(self.tr(
            "Only astronomy software opens these — safe to associate with SASpro."
        )))
        for ext, label in PRIMARY_ASTRONOMY_EXTENSIONS:
            cb = QCheckBox(f"{ext:<7}  {label}")
            cb.setChecked(True)
            cb.setProperty("ext", ext)
            self._primary_checkboxes.append(cb)
            pv.addWidget(cb)
        root.addWidget(primary_box)

        # ── Secondary (opt-in with warning) ──────────────────────────────
        sec_box = QGroupBox(self.tr("Other image types (opt-in)"))
        sv = QVBoxLayout(sec_box)
        warn = QLabel(self.tr(
            "These may already be handled by Photoshop, Lightroom, your RAW "
            "editor, or a system viewer. Associating them with SASpro will "
            "override that until you change it back."
        ))
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #c0a000;")
        sv.addWidget(warn)
        for ext, label in SECONDARY_EXTENSIONS:
            cb = QCheckBox(f"{ext:<7}  {label}")
            cb.setChecked(False)
            cb.setProperty("ext", ext)
            self._secondary_checkboxes.append(cb)
            sv.addWidget(cb)
        root.addWidget(sec_box)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_register = QPushButton(self.tr("Register selected"))
        self._btn_unregister = QPushButton(self.tr("Unregister all"))
        self._btn_manual = QPushButton(self.tr("Show manual steps…"))
        self._btn_close = QPushButton(self.tr("Close"))

        self._btn_register.clicked.connect(self._do_register)
        self._btn_unregister.clicked.connect(self._do_unregister)
        self._btn_manual.clicked.connect(self._show_manual_steps)
        self._btn_close.clicked.connect(self.accept)

        # Disable auto-registration where we can't do it cleanly.
        if not self._info.can_register:
            self._btn_register.setEnabled(False)
            self._btn_unregister.setEnabled(False)
            self._btn_register.setToolTip(self.tr(
                "Source checkouts have no single executable to point "
                "associations at. Use the manual steps."
            ))

        if sys.platform == "darwin":
            # macOS: real associations come from the .app bundle's Info.plist.
            # There's no clean user-space API to swap defaults from here.
            self._btn_register.setEnabled(False)
            self._btn_unregister.setEnabled(False)
            self._btn_register.setToolTip(self.tr(
                "On macOS, file associations come from the .app bundle. "
                "Use the manual steps below."
            ))

        btn_row.addWidget(self._btn_register)
        btn_row.addWidget(self._btn_unregister)
        btn_row.addStretch(1)
        btn_row.addWidget(self._btn_manual)
        btn_row.addWidget(self._btn_close)
        root.addLayout(btn_row)

        # ── Wiki link ────────────────────────────────────────────────────
        help_link = QLabel(self.tr(
            'More info and troubleshooting: '
            '<a href="{url}">File Associations wiki</a>'
        ).format(url=WIKI_URL))
        help_link.setOpenExternalLinks(True)
        help_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        help_link.setStyleSheet("color: #888; font-size: 11px; padding-top: 4px;")
        root.addWidget(help_link)

    def _selected_extensions(self) -> list[str]:
        exts = []
        for cb in self._primary_checkboxes + self._secondary_checkboxes:
            if cb.isChecked():
                exts.append(cb.property("ext"))
        return exts

    def _all_extensions(self) -> list[str]:
        return [cb.property("ext")
                for cb in self._primary_checkboxes
                + self._secondary_checkboxes]

    def _do_register(self):
        exts = self._selected_extensions()
        if not exts:
            QMessageBox.information(
                self, self.tr("Register"),
                self.tr("Tick at least one extension first."),
            )
            return

        secondary_chosen = [cb.property("ext")
                            for cb in self._secondary_checkboxes
                            if cb.isChecked()]
        if secondary_chosen:
            ret = QMessageBox.question(
                self, self.tr("Override other apps?"),
                self.tr(
                    "You've selected some file types that are usually opened "
                    "by other apps (Photoshop, RAW editors, etc.):\n\n{types}\n\n"
                    "Setting SASpro as the default will override those. "
                    "Continue?"
                ).format(types=", ".join(secondary_chosen)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        if sys.platform.startswith("win"):
            count, errors = register_windows_per_user(self._info.exe, exts)
        elif sys.platform.startswith("linux"):
            count, errors = register_linux_user(self._info.exe, exts)
        else:
            self._show_manual_steps()
            return

        if errors:
            QMessageBox.warning(
                self, self.tr("Register"),
                self.tr("Registered {count}, with errors:\n\n{errs}").format(
                    count=count, errs="\n".join(errors),
                ),
            )
        else:
            QMessageBox.information(
                self, self.tr("Register"),
                self.tr("Associated {count} file types with SASpro.")
                .format(count=count),
            )

    def _do_unregister(self):
        if sys.platform.startswith("win"):
            exts = self._all_extensions()
            count, errors = unregister_windows_per_user(exts)
            msg = self.tr("Removed registrations for {count} file types.")
        elif sys.platform.startswith("linux"):
            count, errors = unregister_linux_user()
            msg = self.tr(
                "Removed SASpro .desktop file. User-selected defaults in "
                "~/.config/mimeapps.list may still point at SASpro — clear "
                "them from your file manager's 'Open with' menu if needed."
            )
        else:
            QMessageBox.information(
                self, self.tr("Unregister"),
                self.tr(
                    "On macOS, use Finder → Get Info → Open with → Change All… "
                    "on each extension to point it somewhere else."
                ),
            )
            return

        if errors:
            QMessageBox.warning(
                self, self.tr("Unregister"),
                self.tr("Removed {count}, with errors:\n\n{errs}").format(
                    count=count, errs="\n".join(errors),
                ),
            )
        else:
            QMessageBox.information(self, self.tr("Unregister"),
                                    msg.format(count=count))

    def _show_manual_steps(self):
        dlg = _ManualStepsDialog(self._info, self)
        dlg.exec()


class _ManualStepsDialog(QDialog):
    def __init__(self, info: LauncherInfo, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Manual file association steps"))
        self.setMinimumSize(640, 520)

        root = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setOpenExternalLinks(True)
        tb.setHtml(self._html_for_platform(info))
        root.addWidget(tb)

        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(self.reject)
        btn.accepted.connect(self.accept)
        root.addWidget(btn)

    def _html_for_platform(self, info: LauncherInfo) -> str:
        exe = info.exe
        # Shared footer with the deep link to the wiki — appended to each
        # platform's HTML so users looking for more depth land in one place.
        wiki_footer = (
            f'<hr><p style="color:#888;font-size:11px;">'
            f'Full documentation and troubleshooting: '
            f'<a href="{WIKI_URL}">{WIKI_URL}</a></p>'
        )

        if sys.platform.startswith("win"):
            body = f"""
            <h2>Windows — manual file association</h2>
            <p>Executable to point at:</p>
            <pre>{exe}</pre>
            <ol>
              <li>Right-click a <code>.fits</code> file in File Explorer.</li>
              <li>Choose <b>Open with → Choose another app</b>.</li>
              <li>Click <b>More apps</b>, scroll to the bottom, choose
                  <b>Look for another app on this PC</b>.</li>
              <li>Browse to the path above and click <b>Open</b>.</li>
              <li>Tick <b>Always use this app</b> to make it permanent.</li>
            </ol>
            <p>Repeat for each extension you want
              (<code>.fit</code>, <code>.fts</code>, <code>.xisf</code>, …).</p>
            <h3>Source checkouts</h3>
            <p>If you're launching with <code>python setiastrosuitepro.py</code>,
               Windows can't point a file association at a bare script. Create
               a one-line batch wrapper (e.g.
               <code>C:\\Tools\\saspro.bat</code>) containing:</p>
            <pre>@python "C:\\path\\to\\setiastrosuitepro.py" %*</pre>
            <p>Then point Open With at that <code>.bat</code>.</p>
            """
        elif sys.platform == "darwin":
            body = f"""
            <h2>macOS — manual file association</h2>
            <p>File associations on macOS come from the <code>.app</code>
               bundle's <code>Info.plist</code>. If you don't have the
               <code>.app</code> installed, there's no portable way to make
               SASpro the default handler.</p>
            <ol>
              <li>Install <b>SetiAstroSuitePro.app</b> from the official
                  release.</li>
              <li>Right-click a <code>.fits</code> file in Finder, choose
                  <b>Get Info</b>.</li>
              <li>Expand <b>Open with:</b>, pick <b>SetiAstroSuitePro</b>.</li>
              <li>Click <b>Change All…</b> to apply the choice to every file
                  of that type.</li>
            </ol>
            <p>Repeat for each extension you want. Even without the
               <code>.app</code>, SASpro still accepts files you drag onto its
               running window — but Finder won't list it as a handler.</p>
            <p>Current launcher: <code>{exe}</code></p>
            """
        else:  # Linux
            body = f"""
            <h2>Linux — manual file association</h2>
            <p>Executable:</p>
            <pre>{exe}</pre>
            <p>Use <b>Register selected</b> above if it's available — it
               writes a user-level <code>.desktop</code> file and runs
               <code>xdg-mime default</code>. Manual steps if you'd rather:</p>
            <ol>
              <li>Create
                <code>~/.local/share/applications/setiastrosuitepro.desktop</code>:
<pre>[Desktop Entry]
Type=Application
Name=Seti Astro Suite Pro
Exec={exe} %F
Terminal=false
Categories=Graphics;Science;Astronomy;
MimeType=image/fits;application/x-xisf;</pre>
              </li>
              <li>Refresh the desktop DB:
<pre>update-desktop-database ~/.local/share/applications</pre>
              </li>
              <li>Make SASpro the default for each MIME type:
<pre>xdg-mime default setiastrosuitepro.desktop image/fits
xdg-mime default setiastrosuitepro.desktop application/x-xisf</pre>
              </li>
            </ol>
            """

        return body + wiki_footer