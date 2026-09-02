# src/setiastro/saspro/rcastro.py — RC-Astro CLI Integration
# =============================================================================
#
#  Integrates Russ Croman's rc-astro CLI tools:
#    • BlurXTerminator  (bxt) — AI deconvolution / sharpening
#    • StarXTerminator  (sxt) — AI star removal
#    • NoiseXTerminator (nxt) — AI noise reduction
#
#  Each product requires a separate license activated via:
#    rc-astro <product> --activate <email> <key>
#
#  Written by Franklin Marek  |  www.setiastro.com
#
# =============================================================================
from __future__ import annotations

import os
import re
import platform
import tempfile
import shutil
import numpy as np

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QGroupBox, QSlider, QFileDialog, QMessageBox, QProgressBar,
    QWidget, QSizePolicy, QCheckBox, QRadioButton, QTabWidget, QTextEdit,
    QLineEdit, QApplication, QScrollArea,
)
from PyQt6.QtGui import QFont

# ── SetiAstro copyright header ────────────────────────────────────────────────
#
#   _____      __  _ ___         __
#  / ___/___  / /_(_)   |  _____/ /__________
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/
#
# =============================================================================

PRODUCT_LABELS = {
    "bxt": "BlurXTerminator",
    "sxt": "StarXTerminator",
    "nxt": "NoiseXTerminator",
}

def _prefer_high_perf_gpu(exe_path: str) -> None:
    if platform.system() != "Windows" or not exe_path:
        return
    try:
        import winreg
        full = os.path.abspath(exe_path)
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\DirectX\UserGpuPreferences",
            0, winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, full, 0, winreg.REG_SZ, "GpuPreference=2;")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# --host support (rc-astro CLI schemaVersion >= 6)
#
# Russ Croman's rc-astro CLI 2.6.2 (schemaVersion 6) adds an optional --host
# flag so support can see which integration a machine is running. We discover
# the schemaVersion via a bare `rc-astro --no-banner --json` on startup /
# executable-path change, cache it in QSettings under rcastro/schema_version,
# and gate --host on version >= 6. Older CLIs reject --host as unknown.
# ---------------------------------------------------------------------------
def _saspro_host_tag() -> str:
    """
    Returns the short 'SASPro-<version>' tag passed via --host. Falls back to
    plain 'SASPro' if the version can't be resolved. Sanitization on the CLI
    side is documented as: printable ASCII only, ',', '(', ')' dropped,
    whitespace collapsed, length-capped -- so we don't need to be clever here.
    """
    try:
        from setiastro.saspro._generated.build_info import APP_VERSION as _v  # type: ignore
        v = str(_v).strip()
        if v:
            return f"SASPro-{v}"
    except Exception:
        pass
    return "SASPro"


# # === SASpro rcastro v5/lp v1 ===
def _probe_rcastro_json(exe: str) -> dict | None:
    """Runs `rc-astro --no-banner --json` and returns the parsed JSON as a
    dict, or None if it can't be obtained (older CLI, no exe, timeout,
    parse error). Called by _probe_schema_version and _probe_version so
    we only pay the subprocess cost once per probe cycle.
    """
    if not exe or not os.path.exists(exe):
        return None
    import subprocess, json as _json
    try:
        r = subprocess.run(
            [exe, "--no-banner", "--json"],
            capture_output=True, text=True, timeout=8,
        )
        out = (r.stdout or "").strip()
        if not out:
            return None
        i = out.find("{")
        if i < 0:
            return None
        return _json.loads(out[i:])
    except Exception:
        return None


def _ml_versions_for(products_json: list | None, key: str) -> list[str]:
    """Extract mlVersions for a given product key from the --json products
    list, formatted as printable strings that preserve floats (3.1 stays
    "3.1", 5 becomes "5"). Returns [] if not present.
    """
    if not products_json:
        return []
    for p in products_json:
        if not isinstance(p, dict):
            continue
        if p.get("key") == key:
            versions = p.get("mlVersions") or []
            out: list[str] = []
            for v in versions:
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    # Preserve floats; drop trailing .0 on whole numbers.
                    s = f"{v:g}"
                    out.append(s)
                elif isinstance(v, str) and v.strip():
                    out.append(v.strip())
            return out
    return []


def _probe_schema_version(exe: str):
    """
    Runs `rc-astro --no-banner --json` and returns the top-level schemaVersion
    as an int, or None if it can't be determined (older CLI, parse error, no
    exe, timeout, etc.). Cheap: no product, no work.
    """
    data = _probe_rcastro_json(exe)
    if not data:
        return None
    sv = data.get("schemaVersion")
    if isinstance(sv, bool):  # bool is a subclass of int; reject
        return None
    if isinstance(sv, int):
        return sv
    if isinstance(sv, str) and sv.strip().isdigit():
        return int(sv.strip())
    return None


def _host_args() -> list[str]:
    """
    Returns ['--host', '<tag>'] when the cached rc-astro schemaVersion is
    >= 6, else []. Appended to the end of every rc-astro invocation.
    """
    try:
        s = QSettings()
        sv = s.value("rcastro/schema_version", 0, type=int)
        if isinstance(sv, int) and sv >= 6:
            return ["--host", _saspro_host_tag()]
    except Exception:
        pass
    return []


def _detect_cli_uses_device_flag(exe: str) -> bool:
    """
    Returns True if this rc-astro binary uses --device (0.9.7+),
    False if it uses the old --engine flag (0.9.6 and earlier).
    Probes via --no-banner --help and looks for '--device' in the output.
    """
    if not exe or not os.path.exists(exe):
        return True  # assume new if unknown
    import subprocess
    try:
        r = subprocess.run(
            [exe, "--no-banner", "--help"],
            capture_output=True, text=True, timeout=8
        )
        out = (r.stdout or "") + (r.stderr or "")
        return "--device" in out
    except Exception:
        return True  # assume new on error    
# ---------------------------------------------------------------------------
# Worker — runs any rc-astro subprocess, streams stdout+stderr
# ---------------------------------------------------------------------------

class _RCAstroWorker(QThread):
    output_signal   = pyqtSignal(str)   # one line of text
    finished_signal = pyqtSignal(int)   # process return code

    def __init__(self, command: list[str], cwd: str, parent=None):
        super().__init__(parent)
        self._command = command
        self._cwd     = cwd
        self._proc    = None

    def cancel(self):
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self):
        import subprocess
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._cwd,
                text=True,
            )
            for line in self._proc.stdout:
                self.output_signal.emit(line.rstrip("\n"))
            self._proc.wait()
            self.finished_signal.emit(self._proc.returncode)
        except Exception as e:
            self.output_signal.emit(f"[Error launching process] {e}")
            self.finished_signal.emit(-1)


# ---------------------------------------------------------------------------
# Progress / log dialog
# ---------------------------------------------------------------------------

class _ProgressDialog(QDialog):
    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(320)

        outer = QVBoxLayout(self)

        self.lbl_stage = QLabel("Starting…")
        outer.addWidget(self.lbl_stage)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 0)
        outer.addWidget(self.pbar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        self.log.setFont(QFont("Courier New", 9))
        outer.addWidget(self.log, 1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        outer.addWidget(self.btn_cancel)

        self._cancel_fn = None   # set by caller to cancel the worker

    def set_cancel_fn(self, fn):
        """Register the function to call when Cancel is clicked during processing."""
        self._cancel_fn = fn

    def _on_cancel_clicked(self):
        if self.btn_cancel.text() == "Close":
            self.accept()
        else:
            if self._cancel_fn:
                self._cancel_fn()

    def mark_done(self):
        """Switch Cancel -> Close and wire it to close the dialog."""
        self.btn_cancel.setText("Close")

    def append(self, text: str):
        self.log.append(text)
        self.log.ensureCursorVisible()

    def set_stage(self, stage: str):
        self.lbl_stage.setText(stage)

    def set_progress(self, done: int, total: int, stage: str = ""):
        self.pbar.setRange(0, max(total, 1))
        self.pbar.setValue(done)
        if stage:
            self.lbl_stage.setText(stage)


# ---------------------------------------------------------------------------
# Slider helper
# ---------------------------------------------------------------------------

def _form_slider(form: QFormLayout, label: str,
                 lo: float, hi: float, default: float,
                 decimals: int = 2, scale: int = 100) -> QSlider:
    row = QWidget()
    h   = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
    sld = QSlider(Qt.Orientation.Horizontal)
    sld.setRange(int(lo * scale), int(hi * scale))
    sld.setValue(int(default * scale))
    h.addWidget(sld, 1)
    val_lbl = QLabel(f"{default:.{decimals}f}")
    val_lbl.setFixedWidth(48)
    h.addWidget(val_lbl)
    sld.valueChanged.connect(
        lambda v, l=val_lbl, d=decimals, s=scale: l.setText(f"{v/s:.{d}f}"))
    form.addRow(label, row)
    return sld


# ---------------------------------------------------------------------------
# BXT parameter panel
# ---------------------------------------------------------------------------

class _BXTPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        # Correct Only checkbox — disables star sharpening when checked
        self.chk_correct_only = QCheckBox(
            "Correct Only  (PSF aberration correction without any sharpening)"
        )
        self.chk_correct_only.setChecked(False)
        self.chk_correct_only.setToolTip(
            "Passes --correct-only to BXT.\n"
            "Corrects PSF aberrations without applying any star sharpening.\n"
            "Equivalent to leaving Sharpen Stars at 0."
        )
        form.addRow("", self.chk_correct_only)

        self.cmb_model = QComboBox()
        # # === SASpro rcastro v5/lp v1 ===: starts with legacy 2-item list; rebuilt by
        # set_ml_versions() once the --json probe has run.
        self.cmb_model.addItems(["Latest", "AI2 (legacy)"])
        self.cmb_model.setToolTip(
            "BXT ML model version.\n"
            "Latest — current default model (recommended).\n"
            "Other entries pass --ml-version <N>.\n"
            "Requires RC-Astro CLI 0.9.8 or later.")
        form.addRow("Model:", self.cmb_model)
        self._model_form   = form
        self._ml_supported = False
        self._ml_versions: list[str] = []   # populated by set_ml_versions
        self.set_ml_version_supported(False)

        # # === SASpro rcastro v5/lp v1 ===: lunar / planetary mode (--lp flag, BXT only).
        # Hidden by default; the parent dialog un-hides it once it has
        # confirmed the CLI supports --lp (via a --help grep).
        self.chk_lunar_planetary = QCheckBox("Lunar / planetary mode")
        self.chk_lunar_planetary.setChecked(False)
        self.chk_lunar_planetary.setToolTip(
            "Passes --lp to BXT for lunar and planetary imagery.\n"
            "Uses a different processing path tuned for high-contrast\n"
            "surface detail rather than deep-sky stars + faint structure.\n"
            "Requires RC-Astro CLI 2.6.6 or later.")
        form.addRow("", self.chk_lunar_planetary)
        self._lp_supported = False
        self.chk_lunar_planetary.setVisible(False)
        # Grey out star-related controls when LP is on (they're pinned to 0
        # by the CLI; leaving them enabled would confuse users who moved
        # the sliders and wondered why nothing changed).
        self.chk_lunar_planetary.toggled.connect(self._on_lp_toggled)

        self.sld_ss  = _form_slider(form, "Sharpen Stars (0 – 0.7):",
                                    0.0, 0.7, 0.0, decimals=2, scale=100)
        self.sld_ash = _form_slider(form, "Adjust Star Halos (−0.5 – 0.5):",
                                    -0.5, 0.5, 0.0, decimals=2, scale=100)

        self.chk_auto_nsr = QCheckBox("Auto-detect nonstellar PSF (recommended)")
        self.chk_auto_nsr.setChecked(True)
        form.addRow("", self.chk_auto_nsr)

        self.sld_nsr = _form_slider(form, "Manual Nonstellar PSF (0 – 8 px):",
                                    0.0, 8.0, 0.0, decimals=1, scale=10)
        self.sld_nsr.setEnabled(False)
        self.chk_auto_nsr.toggled.connect(
            lambda on: self.sld_nsr.setEnabled(not on))

        self.sld_sn  = _form_slider(form, "Sharpen Nonstellar (0 – 1):",
                                    0.0, 1.0, 0.0, decimals=2, scale=100)

        note = QLabel(
            "BXT handles linear / non-linear detection automatically.\n"
            "No pre-stretch needed — just run it on your linear or stretched image.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        form.addRow("", note)

        # Wire Correct Only → disable/enable Sharpen Stars slider
        self.chk_correct_only.toggled.connect(self._on_correct_only_toggled)

    def set_ml_version_supported(self, supported: bool):
        self._ml_supported = supported
        self.cmb_model.setVisible(supported)
        lbl = self._model_form.labelForField(self.cmb_model)
        if lbl:
            lbl.setVisible(supported)
        if not supported:
            self.cmb_model.setCurrentIndex(0)

    def _on_correct_only_toggled(self, checked: bool):
        self.sld_ss.setEnabled(not checked)
        self.sld_ash.setEnabled(not checked)
        self.sld_sn.setEnabled(not checked)

    def _on_lp_toggled(self, checked: bool):
        # LP mode: no stars, so star sharpening + halo adjust + auto-PSF
        # are all invalid.  Nonstellar sharpening still applies (surface
        # detail is the whole point).
        self.sld_ss.setEnabled(not checked)
        self.sld_ash.setEnabled(not checked)
        self.chk_auto_nsr.setEnabled(not checked)
        # Nonstellar PSF slider becomes always-enabled in LP (auto is off);
        # outside LP it follows the auto-detect checkbox as before.
        if checked:
            self.sld_nsr.setEnabled(True)
        else:
            self.sld_nsr.setEnabled(not self.chk_auto_nsr.isChecked())

    def build_args(self) -> list[str]:
        # CLI ≥ 1.0.0 renamed:
        #   --no-auto-nonstellar-radius → --no-auto-nonstellar-psf
        #   --nonstellar-radius         → --nonstellar-diameter
        s = QSettings()
        uses_psf_flag = bool(s.value("rcastro/uses_nonstellar_psf_flag", False, type=bool))
        no_auto_flag = "--no-auto-nonstellar-psf" if uses_psf_flag else "--no-auto-nonstellar-radius"
        nsr_value_flag = "--nonstellar-diameter" if uses_psf_flag else "--nonstellar-radius"

        args: list[str] = []
        lp_on = self._lp_supported and self.chk_lunar_planetary.isChecked()

        if self.chk_correct_only.isChecked():
            # --correct-only pins ansp=true, ash/nsd/sn/ss=0 on the CLI side.
            # Passing any of those flags ourselves contradicts the mode's pins,
            # so emit the single flag and nothing else.
            args.append("--correct-only")
        else:
            # LP mode pins star-related flags to zero (no stars on the moon)
            # AND auto-PSF is meaningless (it estimates from stars).  Skip
            # all three; the nonstellar sharpening still applies to surface
            # detail and is the whole point of LP mode.
            if not lp_on:
                ss = self.sld_ss.value() / 100.0
                if ss > 0:
                    args += ["--sharpen-stars", f"{ss:.2f}"]

                ash = self.sld_ash.value() / 100.0
                if abs(ash) > 0:
                    args += ["--adjust-star-halos", f"{ash:.2f}"]

                if not self.chk_auto_nsr.isChecked():
                    nsr = self.sld_nsr.value() / 10.0
                    args += [no_auto_flag,
                             nsr_value_flag, f"{nsr:.1f}"]
            else:
                # In LP mode always use manual nonstellar PSF (auto needs stars).
                # Fall back to a sensible default if the user hasn't picked one.
                nsr = self.sld_nsr.value() / 10.0
                if nsr <= 0.0:
                    nsr = 2.0  # reasonable starting point for planetary
                args += [no_auto_flag,
                         nsr_value_flag, f"{nsr:.1f}"]

            sn = self.sld_sn.value() / 100.0
            if sn > 0:
                args += ["--sharpen-nonstellar", f"{sn:.2f}"]

        # # === SASpro rcastro v5/lp v1 ===: dynamic ml-version + --lp
        if self._ml_supported:
            ver = self._selected_ml_version()
            if ver:
                args += ["--ml-version", ver]

        if self._lp_supported and self.chk_lunar_planetary.isChecked():
            args.append("--lp")

        return args

    # # === SASpro rcastro v5/lp v1 ===
    def _selected_ml_version(self) -> str | None:
        """Return the ML-version string to pass to --ml-version, or None
        for the 'Latest' entry (which passes nothing so the CLI picks
        its own default). Handles both the legacy hardcoded combo and
        the dynamically-populated one."""
        idx = self.cmb_model.currentIndex()
        # New-style dynamic list: ["Latest (MLN)", "MLN", "MLM", ...]
        if self._ml_versions:
            if idx <= 0:
                return None  # "Latest"
            j = idx - 1  # skip the "Latest" row
            if 0 <= j < len(self._ml_versions):
                return self._ml_versions[j]
            return None
        # Legacy hardcoded list: ["Latest", "AI2 (legacy)"]
        return "2" if idx == 1 else None

    # # === SASpro rcastro v5/lp v1 ===
    def set_ml_versions(self, versions: list[str]):
        """Rebuild the model combo from a list like ["5", "4", "2"].
        Empty list falls back to the legacy hardcoded combo so nothing
        regresses when the CLI's --json doesn't yield anything usable."""
        self._ml_versions = list(versions or [])
        # Remember the current pick so we can restore it if possible
        prev = self._selected_ml_version()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        if self._ml_versions:
            latest = self._ml_versions[0]
            self.cmb_model.addItem(f"Latest (ML{latest})")
            for v in self._ml_versions:
                self.cmb_model.addItem(f"ML{v}")
            # Restore selection: match the version string if we can
            if prev is None:
                self.cmb_model.setCurrentIndex(0)
            else:
                try:
                    j = self._ml_versions.index(prev)
                    self.cmb_model.setCurrentIndex(j + 1)
                except ValueError:
                    self.cmb_model.setCurrentIndex(0)
        else:
            self.cmb_model.addItems(["Latest", "AI2 (legacy)"])
            self.cmb_model.setCurrentIndex(1 if prev == "2" else 0)
        self.cmb_model.blockSignals(False)

    # # === SASpro rcastro v5/lp v1 ===
    def set_lp_supported(self, supported: bool):
        """Show/hide the lunar/planetary checkbox based on CLI capability."""
        self._lp_supported = bool(supported)
        self.chk_lunar_planetary.setVisible(self._lp_supported)
        if not self._lp_supported:
            self.chk_lunar_planetary.setChecked(False)

    def save_settings(self, s: QSettings):
        s.setValue("rcastro/bxt_correct_only", self.chk_correct_only.isChecked())
        s.setValue("rcastro/bxt_ss",   self.sld_ss.value())
        s.setValue("rcastro/bxt_ash",  self.sld_ash.value())
        s.setValue("rcastro/bxt_auto", self.chk_auto_nsr.isChecked())
        s.setValue("rcastro/bxt_nsr",  self.sld_nsr.value())
        s.setValue("rcastro/bxt_sn",   self.sld_sn.value())
        # # === SASpro rcastro v5/lp v1 ===
        s.setValue("rcastro/bxt_ml_version", self.cmb_model.currentIndex())  # legacy
        _v = self._selected_ml_version()
        s.setValue("rcastro/bxt_ml_version_num", "" if _v is None else _v)
        s.setValue("rcastro/bxt_lunar_planetary", self.chk_lunar_planetary.isChecked())

    def load_settings(self, s: QSettings):
        self.chk_correct_only.setChecked(
            bool(s.value("rcastro/bxt_correct_only", False, type=bool)))
        self.sld_ss.setValue(          int( s.value("rcastro/bxt_ss",   0)))
        self.sld_ash.setValue(         int( s.value("rcastro/bxt_ash",  0)))
        self.chk_auto_nsr.setChecked( bool( s.value("rcastro/bxt_auto", True, type=bool)))
        self.sld_nsr.setValue(         int( s.value("rcastro/bxt_nsr",  0)))
        self.sld_sn.setValue(          int( s.value("rcastro/bxt_sn",   0)))
        # # === SASpro rcastro v5/lp v1 ===
        # Prefer new versioned key; fall back to old index-based key for
        # existing installs.
        _saved_ver = str(s.value("rcastro/bxt_ml_version_num", "") or "")
        if _saved_ver and self._ml_versions:
            try:
                j = self._ml_versions.index(_saved_ver)
                self.cmb_model.setCurrentIndex(j + 1)
            except ValueError:
                self.cmb_model.setCurrentIndex(0)
        elif _saved_ver == "2" and not self._ml_versions:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(int(s.value("rcastro/bxt_ml_version", 0)))
        self.chk_lunar_planetary.setChecked(
            bool(s.value("rcastro/bxt_lunar_planetary", False, type=bool)))
        # Sync enabled state after load
        self._on_correct_only_toggled(self.chk_correct_only.isChecked())

# ---------------------------------------------------------------------------
# SXT parameter panel
# ---------------------------------------------------------------------------
class _SXTPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)

        self.chk_stars = QCheckBox(
            "Also write stars-only image  (original − starless)")
        self.chk_stars.setChecked(True)
        form.addRow("", self.chk_stars)

        self.chk_unscreen = QCheckBox(
            "Unscreen — recover star intensities lost to screening\n"
            "(requires stars-only output above)")
        self.chk_unscreen.setChecked(False)
        self.chk_unscreen.setEnabled(False)
        self.chk_stars.toggled.connect(self.chk_unscreen.setEnabled)
        self.chk_unscreen.setEnabled(self.chk_stars.isChecked())
        form.addRow("", self.chk_unscreen)

        self.sld_overlap = _form_slider(
            form, "Tile Overlap (0 – 0.5):",
            0.0, 0.5, 0.2, decimals=2, scale=100
        )
        self.sld_overlap.setToolTip(
            "Tile overlap fraction passed to --overlap.\n"
            "Default 0.20 (20%). Higher values reduce seam artifacts\n"
            "but increase processing time."
        )

        note = QLabel(
            "SASpro will load the starless result into the current document\n"
            "and push the stars-only image as a new document.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        form.addRow("", note)

    def build_args(self) -> list[str]:
        # --stars / --unscreen are deliberately NOT emitted anymore.
        # RC-Astro CLI 2.x renamed the stars output to a difference image
        # (<input>-sxt-difference[-unscreened].tif), which broke our old
        # <input>-sxt-stars.tif lookup. We now build the stars-only image
        # ourselves from (original, starless) in _compute_stars(); the
        # chk_stars / chk_unscreen checkboxes drive THAT, not the CLI.
        args: list[str] = []
        overlap = self.sld_overlap.value() / 100.0
        if abs(overlap - 0.2) > 0.005:
            args += ["--overlap", f"{overlap:.2f}"]
        return args

    def save_settings(self, s: QSettings):
        s.setValue("rcastro/sxt_stars",    self.chk_stars.isChecked())
        s.setValue("rcastro/sxt_unscreen", self.chk_unscreen.isChecked())
        s.setValue("rcastro/sxt_overlap",  self.sld_overlap.value())

    def load_settings(self, s: QSettings):
        self.chk_stars.setChecked(   bool(s.value("rcastro/sxt_stars",    True,  type=bool)))
        self.chk_unscreen.setChecked(bool(s.value("rcastro/sxt_unscreen", False, type=bool)))
        self.sld_overlap.setValue(    int(s.value("rcastro/sxt_overlap",  20)))

# ---------------------------------------------------------------------------
# NXT parameter panel
# ---------------------------------------------------------------------------
class _NXTPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_box = QGroupBox("Denoise Mode")
        mode_h   = QHBoxLayout(mode_box)
        self.rb_simple = QRadioButton("Simple")
        self.rb_ic     = QRadioButton("Intensity && Color")
        self.rb_freq   = QRadioButton("Frequency")
        self.rb_simple.setChecked(True)
        for rb in (self.rb_simple, self.rb_ic, self.rb_freq):
            mode_h.addWidget(rb)
        mode_h.addStretch(1)
        outer.addWidget(mode_box)
        # ── Model version ─────────────────────────────────────────────────────
        model_row = QWidget()
        model_h   = QHBoxLayout(model_row)
        model_h.setContentsMargins(0, 0, 0, 0)
        self.lbl_model = QLabel("Model:")
        self.cmb_model = QComboBox()
        # # === SASpro rcastro v5/lp v1 ===: starts with legacy 2-item list; rebuilt by
        # set_ml_versions() once the --json probe has run.
        self.cmb_model.addItems(["Latest", "AI2 (legacy)"])
        self.cmb_model.setToolTip(
            "NXT ML model version.\n"
            "Latest — current default model (recommended).\n"
            "Other entries pass --ml-version <N>.\n"
            "Requires RC-Astro CLI 0.9.8 or later.")
        self._ml_versions: list[str] = []
        model_h.addWidget(self.lbl_model)
        model_h.addWidget(self.cmb_model)
        model_h.addStretch(1)
        outer.addWidget(model_row)
        self._model_row     = model_row
        self._ml_supported   = False
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.set_ml_version_supported(False)
        # ── Simple ────────────────────────────────────────────────────────────
        self._simple_box = QGroupBox("Simple")
        simple_form = QFormLayout(self._simple_box)
        self.sld_dn = _form_slider(simple_form, "Denoise (0–1):", 0, 1, 0.0)
        outer.addWidget(self._simple_box)

        # ── Intensity & Color ─────────────────────────────────────────────────
        self._ic_box = QGroupBox("Intensity && Color")
        ic_form = QFormLayout(self._ic_box)
        self.sld_di = _form_slider(ic_form, "Denoise Intensity (0–1):", 0, 1, 0.0)
        self.sld_dc = _form_slider(ic_form, "Denoise Color (0–1):",     0, 1, 0.0)
        outer.addWidget(self._ic_box)

        # ── Frequency ─────────────────────────────────────────────────────────
        self._freq_box = QGroupBox("Frequency")
        freq_form = QFormLayout(self._freq_box)
        self.sld_hf  = _form_slider(freq_form, "High-freq (0–1):",              0, 1, 0.0)
        self.sld_lf  = _form_slider(freq_form, "Low-freq (0–1):",               0, 1, 0.0)
        self.sld_ihf = _form_slider(freq_form, "Intensity High-freq (0–1):",    0, 1, 0.0)
        self.sld_ilf = _form_slider(freq_form, "Intensity Low-freq (0–1):",     0, 1, 0.0)
        self.sld_chf = _form_slider(freq_form, "Color High-freq (0–1):",        0, 1, 0.0)
        self.sld_clf = _form_slider(freq_form, "Color Low-freq (0–1):",         0, 1, 0.0)

        fs_row = QWidget()
        fs_h   = QHBoxLayout(fs_row)
        fs_h.setContentsMargins(0, 0, 0, 0)
        fs_h.setSpacing(6)
        self.sld_fs  = QSlider(Qt.Orientation.Horizontal)
        self.sld_fs.setRange(10, 1000)
        self.sld_fs.setValue(50)
        self.lbl_fs  = QLabel("5.0")
        self.lbl_fs.setFixedWidth(48)
        self.sld_fs.valueChanged.connect(
            lambda v: self.lbl_fs.setText(f"{v/10.0:.1f}"))
        fs_h.addWidget(self.sld_fs, 1)
        fs_h.addWidget(self.lbl_fs)
        freq_form.addRow("Frequency Scale (1–100 px):", fs_row)
        outer.addWidget(self._freq_box)

        # ── Iterations (common to all modes) ──────────────────────────────────
        iter_row = QWidget()
        iter_h   = QHBoxLayout(iter_row)
        iter_h.setContentsMargins(0, 0, 0, 0)
        self.sp_iter = QDoubleSpinBox()
        self.sp_iter.setRange(1.0, 5.0)
        self.sp_iter.setSingleStep(0.5)
        self.sp_iter.setValue(2.0)
        self.sp_iter.setDecimals(1)
        iter_h.addWidget(self.sp_iter)
        iter_h.addStretch(1)
        iter_form = QFormLayout()
        iter_form.addRow("Iterations (1–5):", iter_row)
        outer.addLayout(iter_form)
        outer.addStretch(1)

        # ── Wire radio buttons ────────────────────────────────────────────────
        self.rb_simple.toggled.connect(self._update_mode)
        self.rb_ic.toggled.connect(self._update_mode)
        self.rb_freq.toggled.connect(self._update_mode)
        self._update_mode()

    def _on_model_changed(self, idx: int):
        is_legacy = (idx == 1)
        self.rb_ic.setEnabled(not is_legacy)
        self.rb_freq.setEnabled(not is_legacy)
        if is_legacy:
            self.rb_simple.setChecked(True)

    def set_ml_version_supported(self, supported: bool):
        self._ml_supported = supported
        self._model_row.setVisible(supported)
        if not supported:
            self.cmb_model.setCurrentIndex(0)

    # # === SASpro rcastro v5/lp v1 ===
    def _selected_ml_version(self) -> str | None:
        idx = self.cmb_model.currentIndex()
        if self._ml_versions:
            if idx <= 0:
                return None
            j = idx - 1
            if 0 <= j < len(self._ml_versions):
                return self._ml_versions[j]
            return None
        return "2" if idx == 1 else None

    # # === SASpro rcastro v5/lp v1 ===
    def set_ml_versions(self, versions: list[str]):
        self._ml_versions = list(versions or [])
        prev = self._selected_ml_version()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        if self._ml_versions:
            latest = self._ml_versions[0]
            self.cmb_model.addItem(f"Latest (ML{latest})")
            for v in self._ml_versions:
                self.cmb_model.addItem(f"ML{v}")
            if prev is None:
                self.cmb_model.setCurrentIndex(0)
            else:
                try:
                    j = self._ml_versions.index(prev)
                    self.cmb_model.setCurrentIndex(j + 1)
                except ValueError:
                    self.cmb_model.setCurrentIndex(0)
        else:
            self.cmb_model.addItems(["Latest", "AI2 (legacy)"])
            self.cmb_model.setCurrentIndex(1 if prev == "2" else 0)
        self.cmb_model.blockSignals(False)

    def _update_mode(self):
        simple = self.rb_simple.isChecked()
        ic     = self.rb_ic.isChecked()
        freq   = self.rb_freq.isChecked()
        self._simple_box.setEnabled(simple)
        self._ic_box.setEnabled(ic)
        self._freq_box.setEnabled(freq)

    def _active_mode(self) -> str:
        if self.rb_ic.isChecked():
            return "ic"
        if self.rb_freq.isChecked():
            return "freq"
        return "simple"

    def build_args(self) -> list[str]:
        args: list[str] = []

        def _a(flag, val):
            if val > 0:
                args.append(flag)
                args.append(f"{val:.2f}")

        mode = self._active_mode()

        if mode == "simple":
            _a("--denoise", self.sld_dn.value() / 100.0)

        elif mode == "ic":
            _a("--denoise-intensity", self.sld_di.value() / 100.0)
            _a("--denoise-color",     self.sld_dc.value() / 100.0)

        elif mode == "freq":
            _a("--denoise-high-freq",           self.sld_hf.value()  / 100.0)
            _a("--denoise-low-freq",            self.sld_lf.value()  / 100.0)
            _a("--denoise-intensity-high-freq", self.sld_ihf.value() / 100.0)
            _a("--denoise-intensity-low-freq",  self.sld_ilf.value() / 100.0)
            _a("--denoise-color-high-freq",     self.sld_chf.value() / 100.0)
            _a("--denoise-color-low-freq",      self.sld_clf.value() / 100.0)
            fs = self.sld_fs.value() / 10.0
            if abs(fs - 5.0) > 0.05:
                args += ["--frequency-scale", f"{fs:.1f}"]

        it = float(self.sp_iter.value())
        if abs(it - 2.0) > 0.05:
            args += ["--iterations", f"{it:.1f}"]

        if self._ml_supported:
            # # === SASpro rcastro v5/lp v1 ===: dynamic ml-version selection
            ver = self._selected_ml_version()
            if ver:
                args += ["--ml-version", ver]

        return args

    def save_settings(self, s: QSettings):
        mode = self._active_mode()
        s.setValue("rcastro/nxt_mode", mode)
        for attr, key in [
            ("sld_dn",  "rcastro/nxt_dn"),  ("sld_di",  "rcastro/nxt_di"),
            ("sld_dc",  "rcastro/nxt_dc"),  ("sld_hf",  "rcastro/nxt_hf"),
            ("sld_lf",  "rcastro/nxt_lf"),  ("sld_ihf", "rcastro/nxt_ihf"),
            ("sld_ilf", "rcastro/nxt_ilf"), ("sld_chf", "rcastro/nxt_chf"),
            ("sld_clf", "rcastro/nxt_clf"), ("sld_fs",  "rcastro/nxt_fs"),
        ]:
            s.setValue(key, getattr(self, attr).value())
        s.setValue("rcastro/nxt_iter", self.sp_iter.value())
        # # === SASpro rcastro v5/lp v1 ===
        s.setValue("rcastro/nxt_ml_version", self.cmb_model.currentIndex())  # legacy
        _v = self._selected_ml_version()
        s.setValue("rcastro/nxt_ml_version_num", "" if _v is None else _v)

    def load_settings(self, s: QSettings):
        mode = str(s.value("rcastro/nxt_mode", "simple"))
        if mode == "ic":
            self.rb_ic.setChecked(True)
        elif mode == "freq":
            self.rb_freq.setChecked(True)
        else:
            self.rb_simple.setChecked(True)

        for attr, key, default in [
            ("sld_dn",  "rcastro/nxt_dn",  0), ("sld_di",  "rcastro/nxt_di",  0),
            ("sld_dc",  "rcastro/nxt_dc",  0), ("sld_hf",  "rcastro/nxt_hf",  0),
            ("sld_lf",  "rcastro/nxt_lf",  0), ("sld_ihf", "rcastro/nxt_ihf", 0),
            ("sld_ilf", "rcastro/nxt_ilf", 0), ("sld_chf", "rcastro/nxt_chf", 0),
            ("sld_clf", "rcastro/nxt_clf", 0), ("sld_fs",  "rcastro/nxt_fs",  50),
        ]:
            getattr(self, attr).setValue(int(s.value(key, default)))
        self.sp_iter.setValue(float(s.value("rcastro/nxt_iter", 2.0)))
        # # === SASpro rcastro v5/lp v1 ===
        _saved_ver = str(s.value("rcastro/nxt_ml_version_num", "") or "")
        if _saved_ver and self._ml_versions:
            try:
                j = self._ml_versions.index(_saved_ver)
                self.cmb_model.setCurrentIndex(j + 1)
            except ValueError:
                self.cmb_model.setCurrentIndex(0)
        elif _saved_ver == "2" and not self._ml_versions:
            self.cmb_model.setCurrentIndex(1)
        else:
            self.cmb_model.setCurrentIndex(int(s.value("rcastro/nxt_ml_version", 0)))
        self._update_mode()

# ---------------------------------------------------------------------------
# Per-product license / activation panel
# ---------------------------------------------------------------------------

class _LicensePanel(QWidget):
    def __init__(self, product: str, get_exe_fn, parent=None):
        super().__init__(parent)
        self._product   = product
        self._get_exe   = get_exe_fn
        label           = PRODUCT_LABELS[product]

        form = QFormLayout(self)

        self.edit_email = QLineEdit()
        self.edit_email.setPlaceholderText("license@example.com")
        form.addRow("Email:", self.edit_email)

        key_row = QHBoxLayout()
        self.edit_key = QLineEdit()
        self.edit_key.setPlaceholderText("XXXX-XXXX-XXXX-XXXX")
        self.edit_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.btn_show = QPushButton("Show")
        self.btn_show.setCheckable(True)
        self.btn_show.setFixedWidth(54)
        self.btn_show.toggled.connect(self._toggle_show)
        key_row.addWidget(self.edit_key, 1)
        key_row.addWidget(self.btn_show)
        form.addRow("License Key:", key_row)

        btn_row = QHBoxLayout()
        self.btn_activate = QPushButton(f"Activate {label}")
        self.btn_check    = QPushButton("Check Status")
        self.btn_activate.clicked.connect(self._activate)
        self.btn_check.clicked.connect(self._check_status)
        btn_row.addWidget(self.btn_activate)
        btn_row.addWidget(self.btn_check)
        btn_row.addStretch(1)
        form.addRow("", btn_row)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size:11px; color:#aaa;")
        form.addRow("Status:", self.lbl_status)

        self._load()

    def _toggle_show(self, on: bool):
        self.edit_key.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password)
        self.btn_show.setText("Hide" if on else "Show")

    def _save(self):
        s = QSettings()
        s.setValue(f"rcastro/{self._product}_email", self.edit_email.text().strip())
        s.setValue(f"rcastro/{self._product}_key",   self.edit_key.text().strip())

    def _load(self):
        s = QSettings()
        self.edit_email.setText(str(s.value(f"rcastro/{self._product}_email", "")))
        self.edit_key.setText(  str(s.value(f"rcastro/{self._product}_key",   "")))

    def _run_cli(self, extra_args: list[str], stage: str):
        exe = self._get_exe()
        if not exe or not os.path.exists(exe):
            QMessageBox.warning(self, "RC-Astro",
                "RC-Astro executable not set. Browse for it in the main tab.")
            return
        import subprocess
        self.lbl_status.setText(stage)
        QApplication.processEvents()
        try:
            cmd = [exe, "--no-banner", self._product] + extra_args + _host_args()
            r   = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            # strip ASCII-art banner lines
            lines = [l for l in out.splitlines()
                     if l.strip() and not l.strip()[0] in "/_\\|"]
            self.lbl_status.setText("\n".join(lines) or "Done.")
        except Exception as e:
            self.lbl_status.setText(f"Error: {e}")

    def _activate(self):
        email = self.edit_email.text().strip()
        key   = self.edit_key.text().strip()
        if not email or not key:
            QMessageBox.warning(self, "Activate",
                "Enter your email and license key first.")
            return
        self._save()
        self._run_cli(["--activate", email, key],
                      f"Activating {PRODUCT_LABELS[self._product]}…")

    def _check_status(self):
        self._run_cli(["--license"], "Checking license status…")

def _parse_cli_version(text: str):
    """Extract a (major, minor, patch) tuple from rc-astro --help output."""
    m = re.search(r'version\s+(\d+)\.(\d+)\.(\d+)', text, re.IGNORECASE)
    if not m:
        m = re.search(r'\b(\d+)\.(\d+)\.(\d+)\b', text)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())
# ---------------------------------------------------------------------------
# Main RC-Astro dialog
# ---------------------------------------------------------------------------

class RCAstroDialog(QDialog):
    def __init__(self, parent, doc=None, doc_manager=None,
                 list_open_docs_fn=None, rcastro_icon=None):
        super().__init__(parent)
        self.setWindowTitle("RC-Astro Tools")
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumWidth(640)

        if rcastro_icon:
            self.setWindowIcon(rcastro_icon)

        self._doc  = doc
        self._main = parent
        self._worker: _RCAstroWorker | None = None

        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self._build_ui()
        self._load_settings()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Executable row ────────────────────────────────────────────────────
        exe_box  = QGroupBox("RC-Astro Executable")
        exe_form = QFormLayout(exe_box)

        exe_row = QHBoxLayout()
        self.edit_exe = QLineEdit()
        self.edit_exe.setReadOnly(True)
        self.edit_exe.setPlaceholderText("Path to rc-astro  (.exe on Windows)")
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse_exe)
        exe_row.addWidget(self.edit_exe, 1)
        exe_row.addWidget(self.btn_browse)
        exe_form.addRow("Executable:", exe_row)

        self.lbl_version = QLabel("")
        self.lbl_version.setStyleSheet("color:#888; font-size:11px;")
        exe_form.addRow("", self.lbl_version)

        dl_row = QHBoxLayout()
        self.btn_dl_models = QPushButton("Download All Models")
        self.btn_dl_models.setToolTip(
            "Downloads model weights for every activated product.\n"
            "Requires an internet connection.")
        self.btn_dl_models.clicked.connect(self._download_models)
        dl_row.addWidget(self.btn_dl_models)

        self.btn_update = QPushButton("Upgrade RC-Astro CLI")
        self.btn_update.setToolTip(
            "Downloads and installs the latest RC-Astro CLI version.\n"
            "Requires an internet connection.")
        self.btn_update.clicked.connect(self._upgrade_cli)
        dl_row.addWidget(self.btn_update)

        dl_row.addStretch(1)
        exe_form.addRow("", dl_row)

        root.addWidget(exe_box)

        # ── Tabs: BXT / SXT / NXT / Licenses ─────────────────────────────────
        self.tabs = QTabWidget()

        def _scroll(w: QWidget) -> QScrollArea:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setWidget(w)
            return sa

        self.bxt_panel = _BXTPanel()
        self.sxt_panel = _SXTPanel()
        self.nxt_panel = _NXTPanel()
        self.tabs.addTab(_scroll(self.bxt_panel), "BlurXTerminator")
        self.tabs.addTab(_scroll(self.sxt_panel), "StarXTerminator")
        self.tabs.addTab(_scroll(self.nxt_panel), "NoiseXTerminator")

        # Licenses tab — one sub-tab per product
        lic_outer = QWidget()
        lic_v     = QVBoxLayout(lic_outer)
        lic_tabs  = QTabWidget()
        for prod in ("bxt", "sxt", "nxt"):
            panel = _LicensePanel(prod, self._get_exe)
            setattr(self, f"_lic_{prod}", panel)
            lic_tabs.addTab(panel, PRODUCT_LABELS[prod])
        lic_v.addWidget(lic_tabs)
        self.tabs.addTab(lic_outer, "Licenses / Activation")

        root.addWidget(self.tabs, 1)

        # ── Common options ────────────────────────────────────────────────────
        common_box  = QGroupBox("Common Options")
        common_form = QFormLayout(common_box)

        eng_row = QHBoxLayout()
        self.cmb_engine = QComboBox()
        self.cmb_engine.addItems(["auto", "gpu", "cpu"])
        self.cmb_engine.setEditable(True)
        self.cmb_engine.setToolTip(
            "auto  — let rc-astro pick the best available device\n"
            "gpu   — use GPU (0.9.7+) / was 'dml' on older CLI\n"
            "cpu   — force CPU (slow but always works)\n"
            "gpu0, gpu1 etc. — select a specific GPU (0.9.7+ only)")
        eng_row.addWidget(self.cmb_engine)
        self.btn_list_devices = QPushButton("List Devices")
        self.btn_list_devices.setToolTip("Run rc-astro --device to show available compute devices.")
        self.btn_list_devices.clicked.connect(self._list_devices)
        eng_row.addWidget(self.btn_list_devices)
        eng_row.addStretch(1)
        common_form.addRow("Compute Device:", eng_row)

        self.lbl_devices = QLabel("")
        self.lbl_devices.setWordWrap(True)
        self.lbl_devices.setStyleSheet("color:#aaa; font-size:11px;")
        common_form.addRow("", self.lbl_devices)

        self.chk_overwrite = QCheckBox("Overwrite existing output files")
        self.chk_overwrite.setChecked(True)
        common_form.addRow("", self.chk_overwrite)
        self.chk_high_perf_gpu = QCheckBox("Prefer high-performance GPU (NVIDIA)")
        self.chk_high_perf_gpu.setChecked(True)
        self.chk_high_perf_gpu.setToolTip(
            "On hybrid-GPU Windows laptops, DirectML defaults to the Intel\n"
            "integrated GPU. This tells Windows to run rc-astro on the\n"
            "discrete NVIDIA GPU instead. No effect on macOS/Linux or\n"
            "single-GPU systems.")
        common_form.addRow("", self.chk_high_perf_gpu)

        self.cmb_engine.currentTextChanged.connect(self._update_gpu_pref_visibility)
        self._update_gpu_pref_visibility(self.cmb_engine.currentText())
        root.addWidget(common_box)

        # ── Run / Close ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run")
        self.btn_run.setStyleSheet("font-weight:bold; padding:6px 22px;")
        self.btn_run.clicked.connect(self._run)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_run)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from PyQt6.QtGui import QIcon
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import rcastro_path
                _grip_icon = QIcon(rcastro_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "rcastro", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create an RC-Astro shortcut for the ACTIVE tab's "
                    "product and settings.\nDrop directly on an image to apply them headlessly."),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)   # push grip hard to the LEFT edge
            root.addLayout(drag_row)
        except Exception:
            pass

        foot = QLabel("Franklin Marek  |  www.setiastro.com")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        foot.setStyleSheet("color:#444; font-size:10px;")
        root.addWidget(foot)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _list_devices(self):
        exe = self._get_exe()
        if not exe or not os.path.exists(exe):
            self.lbl_devices.setText("Set the rc-astro executable path first.")
            return
        s = QSettings()
        if not bool(s.value("rcastro/uses_device_flag", True, type=bool)):
            self.lbl_devices.setText(
                "Device listing requires RC-Astro CLI 0.9.7 or later. "
                "Please upgrade using the button above.")
            return
        import subprocess
        self.lbl_devices.setText("Querying devices…")
        QApplication.processEvents()
        QApplication.processEvents()
        try:
            r = subprocess.run(
                [exe, "--no-banner", "--device"] + _host_args(),
                capture_output=True, text=True, timeout=10
            )
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            # Filter out banner art lines and empty lines
            lines = [
                l for l in out.splitlines()
                if l.strip() and l.strip()[0] not in "/_\\|"
            ]
            # Drop the "Select a device with..." trailing instruction line
            lines = [l for l in lines if not l.strip().lower().startswith("select a device")]
            self.lbl_devices.setText("\n".join(lines) or "No device info returned.")
        except Exception as e:
            self.lbl_devices.setText(f"Error: {e}")

    def _device_flag(self) -> str:
        """Returns '--device' for 0.9.7+ or '--engine' for older CLI."""
        s = QSettings()
        uses_device = bool(s.value("rcastro/uses_device_flag", True, type=bool))
        return "--device" if uses_device else "--engine"

    def _update_gpu_pref_visibility(self, engine: str):
        self.chk_high_perf_gpu.setVisible(str(engine).strip().lower() != "cpu")

    def _upgrade_cli(self):
        exe = self._get_exe()
        if not exe or not os.path.exists(exe):
            QMessageBox.warning(self, "RC-Astro",
                "Set the rc-astro executable path first.")
            return
        dlg = _ProgressDialog(self, "Upgrade RC-Astro CLI")
        dlg.set_stage("Connecting…")
        cmd = [exe, "update", "--install"] + _host_args()
        worker = _RCAstroWorker(cmd, os.path.dirname(exe) or os.getcwd())
        dlg.set_cancel_fn(worker.cancel)
        worker.output_signal.connect(dlg.append)
        def _on_finish(rc: int):
            if rc == 0:
                dlg.set_stage("Upgrade complete.")
                self._probe_version(exe)  # refresh version label
            else:
                dlg.set_stage(f"Upgrade failed (code {rc}).")
            dlg.mark_done()
        worker.finished_signal.connect(_on_finish)
        worker.start()
        dlg.exec()

    def _get_exe(self) -> str:
        return self.edit_exe.text().strip()

    def _browse_exe(self):
        if platform.system() == "Windows":
            filt = "Executable Files (*.exe);;All Files (*)"
        else:
            filt = "All Files (*)"
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select rc-astro Executable", "", filt)
        if not fn:
            return
        self.edit_exe.setText(fn)
        s = QSettings()
        s.setValue("rcastro/exe_path", fn)
        s.setValue("rcastro/uses_device_flag", bool(_detect_cli_uses_device_flag(fn)))
        self._probe_version(fn)

    def _probe_version(self, exe: str):
        import subprocess
        # Refresh cached rc-astro schemaVersion first so subsequent invocations
        # (including the --help probe below, if we ever want to attach --host
        # to it) can gate on it. Value is 0 when unknown / older CLI.
        try:
            sv = _probe_schema_version(exe)
            QSettings().setValue("rcastro/schema_version", int(sv) if isinstance(sv, int) else 0)
        except Exception:
            QSettings().setValue("rcastro/schema_version", 0)
        try:
            r = subprocess.run(
                [exe, "--no-banner", "--help"] + _host_args(),
                capture_output=True, text=True, timeout=8)
            out = (r.stdout or "") + (r.stderr or "")
            # Update device-flag detection while we have the help output
            s = QSettings()
            s.setValue("rcastro/uses_device_flag", bool("--device" in out))

            ver = _parse_cli_version(out)
            supports_ml2 = bool(ver and ver >= (0, 9, 8))
            s.setValue("rcastro/supports_ml_version", supports_ml2)
            self.bxt_panel.set_ml_version_supported(supports_ml2)
            self.nxt_panel.set_ml_version_supported(supports_ml2)

            # CLI ≥ 1.0.0 renamed --no-auto-nonstellar-radius to --no-auto-nonstellar-psf
            uses_nonstellar_psf = bool(ver and ver >= (1, 0, 0))
            s.setValue("rcastro/uses_nonstellar_psf_flag", uses_nonstellar_psf)

            # # === SASpro rcastro v5/lp v1 ===: pull mlVersions per product from --json and
            # rebuild the model combos.  Falls back gracefully if --json is
            # absent (older CLI): both panels keep their legacy 2-item combo.
            try:
                data = _probe_rcastro_json(exe)
                products = (data or {}).get("products") if isinstance(data, dict) else None
                bxt_versions = _ml_versions_for(products, "bxt")
                nxt_versions = _ml_versions_for(products, "nxt")
                self.bxt_panel.set_ml_versions(bxt_versions)
                self.nxt_panel.set_ml_versions(nxt_versions)
                # Re-apply saved selections now that combos have their real items
                self.bxt_panel.load_settings(s)
                self.nxt_panel.load_settings(s)
                # Cache probe results for RCAstroPresetDialog to read (avoids
                # a second round of subprocess probes when opening the preset
                # editor — those probes can add several seconds of lag).
                import json as _json
                s.setValue("rcastro/bxt_ml_versions",
                           _json.dumps(list(bxt_versions or [])))
                s.setValue("rcastro/nxt_ml_versions",
                           _json.dumps(list(nxt_versions or [])))
            except Exception:
                pass

            # # === SASpro rcastro v5/lp v1 ===: --lp is a BXT-only flag added in
            # CLI 2.6.6. Per-command help is what `rc-astro bxt` (no args) prints —
            # NOT `rc-astro --help`, which only lists the top-level commands.
            try:
                import subprocess as _sp
                _r = _sp.run([exe, "--no-banner", "bxt"] + _host_args(),
                             capture_output=True, text=True, timeout=8)
                _bxt_help = (_r.stdout or "") + (_r.stderr or "")
                _lp_supported = ("--lp" in _bxt_help
                                 or "--lunar-planetary" in _bxt_help)
                self.bxt_panel.set_lp_supported(_lp_supported)
                s.setValue("rcastro/bxt_lp_supported", bool(_lp_supported))
            except Exception:
                self.bxt_panel.set_lp_supported(False)
                s.setValue("rcastro/bxt_lp_supported", False)

            for line in out.splitlines():
                line = line.strip()
                if line.lower().startswith("version"):
                    self.lbl_version.setText(line)
                    return
                if "version" in line.lower() and ("build" in line.lower() or re.search(r'\d+\.\d+', line)):
                    self.lbl_version.setText(line)
                    return
            self.lbl_version.setText("rc-astro found.")
        except Exception as e:
            self.lbl_version.setText(f"rc-astro found (could not read version: {e})")

    def _download_models(self):
        exe = self._get_exe()
        if not exe or not os.path.exists(exe):
            QMessageBox.warning(self, "RC-Astro",
                "Set the rc-astro executable path first.")
            return
        dlg = _ProgressDialog(self, "Download Models")
        dlg.set_stage("Connecting…")
        cmd = [exe, "--no-banner", "download-models"] + _host_args()
        worker = _RCAstroWorker(cmd, os.path.dirname(exe) or os.getcwd())
        dlg.set_cancel_fn(worker.cancel)
        worker.output_signal.connect(dlg.append)
        def _on_finish(rc: int):
            dlg.set_stage("Download complete." if rc == 0 else f"Failed (code {rc}).")
            dlg.mark_done()
        worker.finished_signal.connect(_on_finish)
        worker.start()
        dlg.exec()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _load_settings(self):
        s = QSettings()
        exe = str(s.value("rcastro/exe_path", ""))
        self.edit_exe.setText(exe)
        if exe and os.path.exists(exe):
            self._probe_version(exe)
        else:
            self.bxt_panel.set_ml_version_supported(False)
            self.nxt_panel.set_ml_version_supported(False)
        # Migrate old engine values from pre-0.9.7 (--engine → --device)
        raw_engine = str(s.value("rcastro/engine", "auto"))
        # Migrate old provider names from pre-0.9.7
        if raw_engine in ("dml", "coreml", "cuda"):
            raw_engine = "gpu"
            s.setValue("rcastro/engine", "gpu")
        idx = self.cmb_engine.findText(raw_engine)
        if idx >= 0:
            self.cmb_engine.setCurrentIndex(idx)
        else:
            self.cmb_engine.setCurrentText(raw_engine)
        self.chk_overwrite.setChecked(
            bool(s.value("rcastro/overwrite", True, type=bool)))
        self.chk_high_perf_gpu.setChecked(
            bool(s.value("rcastro/high_perf_gpu", True, type=bool)))
        self.bxt_panel.load_settings(s)
        self.sxt_panel.load_settings(s)
        self.nxt_panel.load_settings(s)

        tab_idx = int(s.value("rcastro/last_tab", 0))
        if 0 <= tab_idx < self.tabs.count():
            self.tabs.setCurrentIndex(tab_idx)

    def _save_settings(self):
        s = QSettings()
        s.setValue("rcastro/engine",   self.cmb_engine.currentText())
        s.setValue("rcastro/overwrite", self.chk_overwrite.isChecked())
        s.setValue("rcastro/high_perf_gpu", self.chk_high_perf_gpu.isChecked())
        s.setValue("rcastro/last_tab",  self.tabs.currentIndex())
        self.bxt_panel.save_settings(s)
        self.sxt_panel.save_settings(s)
        self.nxt_panel.save_settings(s)

    # ── Run ───────────────────────────────────────────────────────────────────
    # ── preset emit / seed (grip + double-click) ───────────────────────
    def get_preset(self) -> dict:
        """
        Emit the ACTIVE tab's product + human-readable params — mirroring how
        _run() dispatches on the current tab. Deliberately OMITS `args`: frozen
        CLI args bake in version-gated flag names (--nonstellar-diameter vs
        --nonstellar-radius, --ml-version) that the consumer re-derives correctly
        from these readable params against current QSettings at drop time.
        Returns {} for the Licenses tab (no product).
        """
        product_map = {0: "bxt", 1: "sxt", 2: "nxt"}
        product = product_map.get(self.tabs.currentIndex())
        if product is None:
            return {}
        p: dict = {"product": product, "engine": self.cmb_engine.currentText()}
        if product == "bxt":
            b = self.bxt_panel
            p.update({
                "correct_only":       b.chk_correct_only.isChecked(),
                "sharpen_stars":      b.sld_ss.value()  / 100.0,
                "adjust_star_halos":  b.sld_ash.value() / 100.0,
                "auto_nsr":           b.chk_auto_nsr.isChecked(),
                "nonstellar_radius":  b.sld_nsr.value() / 10.0,
                "sharpen_nonstellar": b.sld_sn.value()  / 100.0,
                "ml_version": 2 if b.cmb_model.currentIndex() == 1 else None,
            })
        elif product == "sxt":
            x = self.sxt_panel
            p.update({
                "stars":    x.chk_stars.isChecked(),
                "unscreen": x.chk_unscreen.isChecked(),
                "overlap":  x.sld_overlap.value() / 100.0,
            })
        elif product == "nxt":
            n = self.nxt_panel
            p.update({
                "nxt_mode":      n._active_mode(),
                "denoise":       n.sld_dn.value()  / 100.0,
                "denoise_int":   n.sld_di.value()  / 100.0,
                "denoise_color": n.sld_dc.value()  / 100.0,
                "freq_hf":       n.sld_hf.value()  / 100.0,
                "freq_lf":       n.sld_lf.value()  / 100.0,
                "freq_ihf":      n.sld_ihf.value() / 100.0,
                "freq_ilf":      n.sld_ilf.value() / 100.0,
                "freq_chf":      n.sld_chf.value() / 100.0,
                "freq_clf":      n.sld_clf.value() / 100.0,
                "freq_scale":    n.sld_fs.value()  / 10.0,
                "iterations":    float(n.sp_iter.value()),
                "ml_version": 2 if n.cmb_model.currentIndex() == 1 else None,
            })
        return p

    def seed_from_preset(self, preset: dict | None):
        """
        Inverse of get_preset: select the product tab, set engine, then seed that
        panel via the shared _apply_*_preset helpers (which set NXT mode first).
        Reuses the exact seeders the standalone editor uses, so there's one seed
        surface. Does not restore `args` (not emitted; re-derived on run).
        """
        p = dict(preset or {})
        product = str(p.get("product", "bxt")).lower()
        tab_idx = {"bxt": 0, "sxt": 1, "nxt": 2}.get(product, 0)
        self.tabs.setCurrentIndex(tab_idx)

        if "engine" in p:
            eng = str(p["engine"])
            i = self.cmb_engine.findText(eng)
            if i >= 0:
                self.cmb_engine.setCurrentIndex(i)
            else:
                self.cmb_engine.setCurrentText(eng)

        if product == "bxt":
            _apply_bxt_preset(self.bxt_panel, p)
            self.bxt_panel._on_correct_only_toggled(
                self.bxt_panel.chk_correct_only.isChecked())
        elif product == "sxt":
            _apply_sxt_preset(self.sxt_panel, p)
        elif product == "nxt":
            _apply_nxt_preset(self.nxt_panel, p)

    def _run(self):
        exe = self._get_exe()
        if not exe or not os.path.exists(exe):
            QMessageBox.warning(self, "RC-Astro",
                "Set the rc-astro executable path first.")
            return

        doc = self._doc
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "RC-Astro", "No image available.")
            return

        tab = self.tabs.currentIndex()
        product_map = {0: "bxt", 1: "sxt", 2: "nxt"}
        product = product_map.get(tab)
        if product is None:
            QMessageBox.information(self, "RC-Astro",
                "Select a product tab (BXT / SXT / NXT) to run.")
            return

        self._save_settings()

        panel_args = {
            "bxt": self.bxt_panel.build_args,
            "sxt": self.sxt_panel.build_args,
            "nxt": self.nxt_panel.build_args,
        }[product]()

        make_stars = (product == "sxt" and self.sxt_panel.chk_stars.isChecked())
        unscreen   = (product == "sxt" and self.sxt_panel.chk_unscreen.isChecked())
        self._run_product(exe, doc, product, panel_args, make_stars, unscreen)

    def _run_product(self, exe: str, doc, product: str,
                     panel_args: list[str], make_stars: bool = False,
                     unscreen: bool = False):
        from setiastro.saspro.legacy.image_manager import save_image, load_image

        label = PRODUCT_LABELS[product]

        # ── Prepare input array ───────────────────────────────────────────────
        img = np.asarray(doc.image)
        is_mono = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)

        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = np.repeat(img, 3, axis=2)
        else:
            img_rgb = img[..., :3]

        img_rgb = np.clip(img_rgb.astype(np.float32, copy=False), 0.0, 1.0)

        # ── Write 32-bit TIFF to temp dir ─────────────────────────────────────
        work_dir   = tempfile.mkdtemp(prefix="saspro_rcastro_")
        input_path = os.path.join(work_dir, "input.tif")

        try:
            save_image(
                img_rgb, input_path,
                "tif", "32-bit floating point",
                None, False,
                image_meta=None, file_meta=None,
            )
        except Exception as e:
            shutil.rmtree(work_dir, ignore_errors=True)
            QMessageBox.critical(self, label,
                f"Failed to write input TIFF:\n{e}")
            return

        # ── rc-astro output naming convention: <input>-<product>.tif ─────────
        output_path = os.path.join(work_dir, f"input-{product}.tif")
        stars_path  = os.path.join(work_dir, f"input-{product}-stars.tif")

        # ── Build full command ────────────────────────────────────────────────
        cmd = [exe, "--no-banner", product, input_path]
        cmd += panel_args
        cmd += [self._device_flag(), self.cmb_engine.currentText()]
        cmd += ["--depth", "32F"]
        if self.chk_overwrite.isChecked():
            cmd.append("--overwrite")
        if self.cmb_engine.currentText() != "cpu" and self.chk_high_perf_gpu.isChecked():
            _prefer_high_perf_gpu(exe)
        cmd += _host_args()
        # ── Progress dialog ───────────────────────────────────────────────────
        dlg = _ProgressDialog(self, f"{label} — Processing")
        dlg.set_stage(f"Launching {label}…")
        dlg.append("Command: " + " ".join(cmd) + "\n")

        worker = _RCAstroWorker(cmd, cwd=work_dir)
        dlg.set_cancel_fn(worker.cancel)

        _re_pct   = re.compile(r"(\d{1,3})\s*%")
        _re_tiles = re.compile(r"tiles[:\s]+(\d+)", re.IGNORECASE)
        _tile_total: dict = {"n": 0}

        def _on_out(line: str):
            m = _re_tiles.search(line)
            if m:
                try:
                    _tile_total["n"] = int(m.group(1))
                except Exception:
                    pass
            m = _re_pct.search(line)
            if m:
                try:
                    pct  = max(0, min(100, int(m.group(1))))
                    n    = _tile_total["n"] or 100
                    done = int(n * pct / 100.0)
                    dlg.set_progress(done, n, f"Processing… {pct}%")
                except Exception:
                    pass
            dlg.append(line)

        def _on_finish(rc: int):
            dlg.set_progress(100, 100, "Finished. Loading result…")
            _on_finished(
                self, doc, rc, dlg,
                input_path, output_path, stars_path,
                product, is_mono, work_dir, self._main,
                make_stars, unscreen,
            )

        worker.output_signal.connect(_on_out)
        worker.finished_signal.connect(_on_finish)

        self._worker = worker
        worker.start()
        dlg.exec()

    def closeEvent(self, ev):
        self._save_settings()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _compute_stars(original: np.ndarray, starless: np.ndarray,
                   unscreen: bool) -> np.ndarray:
    """
    Derive a stars-only image from the SXT inputs.

        subtraction : stars = O - S
        unscreen    : stars = (O - S) / (1 - S)

    Unscreen inverts the screen recombination SXT uses
    (O = 1 - (1-S)(1-T)  =>  T = (O - S)/(1 - S)), recovering true star
    intensity where stars overlap signal. Both clip to [0, 1]; the
    unscreen denominator is floored to avoid divide-by-zero on saturated
    starless pixels.
    """
    o = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    s = np.clip(np.asarray(starless, dtype=np.float32), 0.0, 1.0)
    diff = o - s
    if unscreen:
        stars = diff / np.maximum(1.0 - s, 1e-6)
    else:
        stars = diff
    return np.clip(stars, 0.0, 1.0)

def _on_finished(main_dlg, doc, return_code, dlg,
                  input_path, output_path, stars_path,
                  product, is_mono, work_dir, main_window,
                  make_stars=False, unscreen=False):
    from setiastro.saspro.legacy.image_manager import load_image

    label = PRODUCT_LABELS[product]

    def _cleanup():
        shutil.rmtree(work_dir, ignore_errors=True)

    dlg.append(f"\nProcess finished with return code {return_code}.\n")

    if return_code != 0:
        QMessageBox.critical(main_dlg, label,
            f"{label} failed (return code {return_code}).\n"
            "Check the log for details.\n\n"
            "Tip: run --license to verify activation.")
        _cleanup()
        dlg.mark_done()
        return

    if not os.path.exists(output_path):
        QMessageBox.critical(main_dlg, label,
            f"Output file not found:\n{output_path}")
        _cleanup()
        dlg.mark_done()
        return

    # Load primary result
    dlg.append(f"Loading: {os.path.basename(output_path)}\n")
    result, _, _, _ = load_image(output_path)

    if result is None:
        QMessageBox.critical(main_dlg, label, "Failed to load output image.")
        _cleanup()
        dlg.mark_done()
        return

    result = np.clip(result.astype(np.float32, copy=False), 0.0, 1.0)

    # ── Build stars-only ourselves, while result is still RGB ──────────────
    if product == "sxt" and make_stars:
        try:
            from setiastro.saspro.legacy.image_manager import save_image
            original_rgb, _, _, _ = load_image(input_path)
            if original_rgb is None:
                raise RuntimeError("could not reload original input TIFF")
            stars_img = _compute_stars(original_rgb, result, unscreen)
            if is_mono and stars_img.ndim == 3:
                stars_img = stars_img.mean(axis=2).astype(np.float32)
            save_image(
                np.clip(stars_img, 0.0, 1.0), stars_path,
                "tif", "32-bit floating point", None, False,
                image_meta=None, file_meta=None,
            )
            dlg.append("Built stars-only image "
                       f"({'unscreen' if unscreen else 'subtraction'}).\n")
        except Exception as e:
            dlg.append(f"[warn] could not build stars-only image: {e}\n")

    # Collapse starless to mono if source was mono
    if is_mono and result.ndim == 3:
        result = result.mean(axis=2).astype(np.float32)

    # Apply to current document
    try:
        doc.apply_edit(
            result,
            metadata={
                "step_name": label,
                "bit_depth": "32-bit floating point",
                "is_mono": bool(is_mono),
            },
            step_name=label,
        )
        dlg.append(f"{label} result applied to current document.\n")
    except Exception as e:
        QMessageBox.critical(main_dlg, label,
            f"Failed to apply result to document:\n{e}")
        _cleanup()
        dlg.mark_done()
        return

    # Push the derived stars-only image as a new document
    if product == "sxt" and make_stars and os.path.exists(stars_path):
        dlg.append(f"Loading stars-only: {os.path.basename(stars_path)}\n")
        _push_new_doc(main_window, stars_path, source_doc=doc)
        dlg.append("Stars-only image pushed as new document.\n")

    _cleanup()
    dlg.accept()

def _push_new_doc(main, file_path: str, source_doc=None):
    """Open a file via DocManager — registration and subwindow spawn are automatic.
    If source_doc is provided, rename the new doc to source_doc's name + _stars."""
    try:
        dm = getattr(main, "docman", None) or getattr(main, "doc_manager", None)
        if dm is None:
            print("[RC-Astro] _push_new_doc: no doc_manager found on main window")
            return
        new_doc = dm.open_path(file_path)
        if new_doc is not None and source_doc is not None:
            try:
                base = (getattr(source_doc, "display_name", lambda: "")()
                        or getattr(source_doc, "name", "")
                        or "image")
                # Strip any file extension from the base name
                base = os.path.splitext(base)[0]
                new_doc.metadata["display_name"] = f"{base}_stars"
                new_doc.changed.emit()
            except Exception as e:
                print(f"[RC-Astro] _push_new_doc rename failed: {e}")
    except Exception as e:
        print(f"[RC-Astro] _push_new_doc failed: {e}")



# ---------------------------------------------------------------------------
# Preset dialog  (used by shortcuts / function bundles)
# ---------------------------------------------------------------------------

class RCAstroPresetDialog(QDialog):
    """
    Compact preset editor for RC-Astro shortcuts / headless runs.
    Mirrors _CosmicClarityPresetDialog pattern.
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("RC-Astro — Preset")
        p = dict(initial or {})

        from PyQt6.QtWidgets import QDialogButtonBox, QScrollArea
        outer = QVBoxLayout(self)

        # Product selector
        prod_form = QFormLayout()
        self.cmb_product = QComboBox()
        self.cmb_product.addItems(["bxt", "sxt", "nxt"])
        self.cmb_product.setCurrentText(str(p.get("product", "bxt")))
        self.cmb_product.currentTextChanged.connect(self._product_changed)
        prod_form.addRow("Product:", self.cmb_product)

        self.cmb_engine = QComboBox()
        self.cmb_engine.addItems(["auto", "gpu", "cpu"])
        self.cmb_engine.setEditable(True)
        self.cmb_engine.setCurrentText(str(p.get("engine", "auto")))
        prod_form.addRow("Engine:", self.cmb_engine)
        outer.addLayout(prod_form)

        # Stacked param area — one widget per product
        self._bxt = _BXTPanel(); self._bxt.load_settings(QSettings())
        self._sxt = _SXTPanel(); self._sxt.load_settings(QSettings())
        self._nxt = _NXTPanel(); self._nxt.load_settings(QSettings())
        self._bxt.set_ml_version_supported(True)
        self._nxt.set_ml_version_supported(True)
        # Pull CLI-probe results (dynamic ML versions + --lp support) from
        # the QSettings cache the main dialog populates on open. If the
        # cache is empty (user never opened the main dialog), panels stay
        # on their legacy 2-item combo + hidden LP checkbox — same
        # behaviour as before this patch.
        self._setup_cli_features()
        # Apply initial preset values to the panels
        if p.get("product") == "bxt":
            _apply_bxt_preset(self._bxt, p)
        elif p.get("product") == "sxt":
            _apply_sxt_preset(self._sxt, p)
        elif p.get("product") == "nxt":
            _apply_nxt_preset(self._nxt, p)

        self._stack = QWidget()
        stack_v = QVBoxLayout(self._stack)
        stack_v.setContentsMargins(0, 0, 0, 0)
        for w in (self._bxt, self._sxt, self._nxt):
            stack_v.addWidget(w)

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(self._stack)
        sa.setMinimumHeight(200)
        outer.addWidget(sa, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._product_changed(self.cmb_product.currentText())
        self.setMinimumWidth(480)

    def _setup_cli_features(self) -> None:
        """Apply CLI-probed features (dynamic ML versions + LP support)
        from the QSettings cache written by the main dialog's
        _probe_version. Read-only w.r.t. the CLI — no subprocess calls
        here, so opening the preset dialog stays snappy."""
        s = QSettings()
        # Dynamic ML versions per product (JSON-encoded list of strings)
        try:
            import json as _json
            bxt_versions = _json.loads(str(s.value("rcastro/bxt_ml_versions", "[]") or "[]"))
            nxt_versions = _json.loads(str(s.value("rcastro/nxt_ml_versions", "[]") or "[]"))
        except Exception:
            bxt_versions = []
            nxt_versions = []
        if bxt_versions:
            self._bxt.set_ml_versions(bxt_versions)
        if nxt_versions:
            self._nxt.set_ml_versions(nxt_versions)
        # After set_ml_versions the combo is rebuilt with real items;
        # re-load settings so the previously saved version is restored.
        if bxt_versions:
            self._bxt.load_settings(s)
        if nxt_versions:
            self._nxt.load_settings(s)
        # BXT lunar/planetary support (CLI 2.6.6+)
        lp_supported = bool(s.value("rcastro/bxt_lp_supported", False, type=bool))
        self._bxt.set_lp_supported(lp_supported)

    def _product_changed(self, product: str):
        self._bxt.setVisible(product == "bxt")
        self._sxt.setVisible(product == "sxt")
        self._nxt.setVisible(product == "nxt")

    def result_dict(self) -> dict:
        product = self.cmb_product.currentText()
        out: dict = {
            "product": product,
            "engine":  self.cmb_engine.currentText(),
        }
        panel = {"bxt": self._bxt, "sxt": self._sxt, "nxt": self._nxt}[product]
        out["args"] = panel.build_args()
        # Also store human-readable params for re-display / durable re-derivation.
        # NB: we intentionally rely on these (not the frozen `args`) for durability,
        # so they must cover every control build_args() reads.
        if product == "bxt":
            out["correct_only"]        = self._bxt.chk_correct_only.isChecked()
            out["sharpen_stars"]       = self._bxt.sld_ss.value()  / 100.0
            out["adjust_star_halos"]   = self._bxt.sld_ash.value() / 100.0
            out["auto_nsr"]            = self._bxt.chk_auto_nsr.isChecked()
            out["nonstellar_radius"]   = self._bxt.sld_nsr.value() / 10.0
            out["sharpen_nonstellar"]  = self._bxt.sld_sn.value()  / 100.0
            # ml_version: string version ("5", "4", "2", …) from the
            # dynamic combo, or None for "Latest". Legacy panels return
            # "2" or None from the same call, so downstream code that
            # accepts either shape keeps working.
            out["ml_version"]          = self._bxt._selected_ml_version()
            # lunar_planetary is a BXT-only flag from _BXTPanel (--lp,
            # CLI 2.6.6+). Emit unconditionally — build_args gates it
            # on _lp_supported anyway, so a stored True on an old CLI
            # is a harmless no-op.
            out["lunar_planetary"]     = self._bxt.chk_lunar_planetary.isChecked()
        elif product == "sxt":
            out["stars"]    = self._sxt.chk_stars.isChecked()
            out["unscreen"] = self._sxt.chk_unscreen.isChecked()
            out["overlap"]  = self._sxt.sld_overlap.value() / 100.0
        elif product == "nxt":
            out["nxt_mode"]      = self._nxt._active_mode()   # simple | ic | freq
            out["denoise"]       = self._nxt.sld_dn.value()  / 100.0
            out["denoise_int"]   = self._nxt.sld_di.value()  / 100.0
            out["denoise_color"] = self._nxt.sld_dc.value()  / 100.0
            out["freq_hf"]       = self._nxt.sld_hf.value()  / 100.0
            out["freq_lf"]       = self._nxt.sld_lf.value()  / 100.0
            out["freq_ihf"]      = self._nxt.sld_ihf.value() / 100.0
            out["freq_ilf"]      = self._nxt.sld_ilf.value() / 100.0
            out["freq_chf"]      = self._nxt.sld_chf.value() / 100.0
            out["freq_clf"]      = self._nxt.sld_clf.value() / 100.0
            out["freq_scale"]    = self._nxt.sld_fs.value()  / 10.0
            out["iterations"]    = float(self._nxt.sp_iter.value())
            # Same as BXT — dynamic string, backward-compatible with int
            out["ml_version"]          = self._nxt._selected_ml_version()
        return out


def _apply_bxt_preset(panel: _BXTPanel, p: dict):
    if "correct_only" in p:
        panel.chk_correct_only.setChecked(bool(p["correct_only"]))
    if "sharpen_stars" in p:
        panel.sld_ss.setValue(int(float(p["sharpen_stars"]) * 100))
    if "adjust_star_halos" in p:
        panel.sld_ash.setValue(int(float(p["adjust_star_halos"]) * 100))
    if "auto_nsr" in p:
        panel.chk_auto_nsr.setChecked(bool(p["auto_nsr"]))
    if "nonstellar_radius" in p:
        panel.sld_nsr.setValue(int(float(p["nonstellar_radius"]) * 10))
    if "sharpen_nonstellar" in p:
        panel.sld_sn.setValue(int(float(p["sharpen_nonstellar"]) * 100))
    # Lunar/planetary mode (only meaningful if CLI supports --lp)
    if "lunar_planetary" in p:
        panel.chk_lunar_planetary.setChecked(bool(p["lunar_planetary"]))
    if "ml_version" in p:
        _v = p["ml_version"]
        if _v is None or _v == 0:
            panel.cmb_model.setCurrentIndex(0)
        else:
            _vs = str(_v)
            if panel._ml_versions:
                try:
                    j = panel._ml_versions.index(_vs)
                    panel.cmb_model.setCurrentIndex(j + 1)
                except ValueError:
                    panel.cmb_model.setCurrentIndex(0)
            else:
                # Legacy 2-item combo — accepts both int 2 and str "2"
                panel.cmb_model.setCurrentIndex(1 if _vs == "2" else 0)

def _apply_sxt_preset(panel: _SXTPanel, p: dict):
    if "stars" in p:
        panel.chk_stars.setChecked(bool(p["stars"]))
    if "unscreen" in p:
        panel.chk_unscreen.setChecked(bool(p["unscreen"]))
    if "overlap" in p:
        panel.sld_overlap.setValue(int(round(float(p["overlap"]) * 100)))


def _apply_nxt_preset(panel: _NXTPanel, p: dict):
    # Mode FIRST — build_args() only emits the active mode's sliders, so the
    # radio must be set before (and the freq sliders below only matter in freq mode).
    mode = str(p.get("nxt_mode", "")).lower()
    if mode == "ic":
        panel.rb_ic.setChecked(True)
    elif mode == "freq":
        panel.rb_freq.setChecked(True)
    elif mode == "simple":
        panel.rb_simple.setChecked(True)
    if "denoise" in p:
        panel.sld_dn.setValue(int(float(p["denoise"]) * 100))
    if "denoise_int" in p:
        panel.sld_di.setValue(int(float(p["denoise_int"]) * 100))
    if "denoise_color" in p:
        panel.sld_dc.setValue(int(float(p["denoise_color"]) * 100))
    # Frequency-mode sliders
    if "freq_hf" in p:
        panel.sld_hf.setValue(int(float(p["freq_hf"]) * 100))
    if "freq_lf" in p:
        panel.sld_lf.setValue(int(float(p["freq_lf"]) * 100))
    if "freq_ihf" in p:
        panel.sld_ihf.setValue(int(float(p["freq_ihf"]) * 100))
    if "freq_ilf" in p:
        panel.sld_ilf.setValue(int(float(p["freq_ilf"]) * 100))
    if "freq_chf" in p:
        panel.sld_chf.setValue(int(float(p["freq_chf"]) * 100))
    if "freq_clf" in p:
        panel.sld_clf.setValue(int(float(p["freq_clf"]) * 100))
    if "freq_scale" in p:
        panel.sld_fs.setValue(int(float(p["freq_scale"]) * 10))
    if "iterations" in p:
        panel.sp_iter.setValue(float(p["iterations"]))
    # ml_version — accept dynamic list versions (str) or legacy int 2
    if "ml_version" in p:
        _v = p["ml_version"]
        if _v is None or _v == 0:
            panel.cmb_model.setCurrentIndex(0)
        else:
            _vs = str(_v)
            if panel._ml_versions:
                try:
                    j = panel._ml_versions.index(_vs)
                    panel.cmb_model.setCurrentIndex(j + 1)
                except ValueError:
                    panel.cmb_model.setCurrentIndex(0)
            else:
                panel.cmb_model.setCurrentIndex(1 if _vs == "2" else 0)
    panel._update_mode()   # sync enabled-state after mode + sliders set


# ---------------------------------------------------------------------------
# Headless runner  (mirrors run_cosmicclarity_via_preset)
# ---------------------------------------------------------------------------

def run_rcastro_via_preset(main, preset: dict | None = None, *, doc=None):
    """
    Run an RC-Astro product headlessly from a preset dict.
    Called by the shortcuts / function-bundle system.

    preset keys:
        product   str   "bxt" | "sxt" | "nxt"
        engine    str   "auto" | "dml" | "cpu"
        args      list  pre-built CLI args from RCAstroPresetDialog.result_dict()
    """
    from PyQt6.QtWidgets import QMessageBox

    p = dict(preset or {})

    # Record for Replay Last
    try:
        remember = getattr(main, "remember_last_headless_command", None) or \
                   getattr(main, "_remember_last_headless_command", None)
        if callable(remember):
            remember("rcastro", p, description="RC-Astro")
        else:
            main._last_headless_command = {"command_id": "rcastro", "preset": dict(p)}
    except Exception:
        pass

    # Resolve doc
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "RC-Astro", "No active image.")
        return

    # Resolve exe
    s = QSettings()
    exe = str(s.value("rcastro/exe_path", ""))
    if not exe or not os.path.exists(exe):
        QMessageBox.warning(main, "RC-Astro",
            "RC-Astro executable not set.\n"
            "Open RC-Astro Tools and browse for the executable first.")
        return

    product = str(p.get("product", "bxt"))
    engine  = str(p.get("engine",  "auto"))
    args    = list(p.get("args", []))

    # Stars-only is derived by SASpro now; capture intent, then strip the
    # CLI flags so RC-Astro doesn't also emit its (renamed) difference image.
    make_stars = False
    unscreen   = False
    if product == "sxt":
        make_stars = bool(p.get("stars", "--stars" in args))
        unscreen   = bool(p.get("unscreen", "--unscreen" in args))
    args = [a for a in args if a not in ("--stars", "--unscreen")]

    # Re-build args from stored human-readable params if args list is empty
    if not args:
        tmp_s = QSettings()
        if product == "bxt":
            panel = _BXTPanel(); _apply_bxt_preset(panel, p)
            panel.set_ml_version_supported(True)
            args = panel.build_args()
        elif product == "sxt":
            panel = _SXTPanel(); _apply_sxt_preset(panel, p)
            args = panel.build_args()
        elif product == "nxt":
            panel = _NXTPanel(); _apply_nxt_preset(panel, p)
            panel.set_ml_version_supported(True)
            args = panel.build_args()

    # Prepare image
    img = np.asarray(doc.image)
    is_mono = img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.repeat(img, 3, axis=2)
    else:
        img_rgb = img[..., :3]
    img_rgb = np.clip(img_rgb.astype(np.float32, copy=False), 0.0, 1.0)

    work_dir   = tempfile.mkdtemp(prefix="saspro_rcastro_headless_")
    input_path = os.path.join(work_dir, "input.tif")
    output_path = os.path.join(work_dir, f"input-{product}.tif")
    stars_path  = os.path.join(work_dir, f"input-{product}-stars.tif")

    from setiastro.saspro.legacy.image_manager import save_image, load_image

    try:
        save_image(img_rgb, input_path,
                   "tif", "32-bit floating point",
                   None, False,
                   image_meta=None, file_meta=None)
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        QMessageBox.critical(main, "RC-Astro", f"Failed to write temp TIFF:\n{e}")
        return

    # Determine correct flag for installed CLI version
    uses_device = bool(s.value("rcastro/uses_device_flag", True, type=bool))
    device_flag = "--device" if uses_device else "--engine"

    cmd = [exe, "--no-banner", product, input_path]
    cmd += args
    cmd += [device_flag, engine, "--depth", "32F", "--overwrite"]
    cmd += _host_args()
    if engine != "cpu" and bool(s.value("rcastro/high_perf_gpu", True, type=bool)):
        _prefer_high_perf_gpu(exe)
    label = PRODUCT_LABELS.get(product, product.upper())

    # Show a simple non-blocking progress dialog
    dlg = _ProgressDialog(main, f"{label} — Processing")
    dlg.set_stage(f"Running {label} headlessly…")
    dlg.append("Command: " + " ".join(cmd) + "\n")

    worker = _RCAstroWorker(cmd, cwd=work_dir)
    dlg.set_cancel_fn(worker.cancel)

    _re_pct   = re.compile(r"(\d{1,3})\s*%")
    _tile_total: dict = {"n": 0}
    _re_tiles = re.compile(r"tiles[:\s]+(\d+)", re.IGNORECASE)

    def _on_out(line: str):
        m = _re_tiles.search(line)
        if m:
            try: _tile_total["n"] = int(m.group(1))
            except Exception: pass
        m = _re_pct.search(line)
        if m:
            try:
                pct  = max(0, min(100, int(m.group(1))))
                n    = _tile_total["n"] or 100
                dlg.set_progress(int(n * pct / 100), n, f"Processing… {pct}%")
            except Exception: pass
        dlg.append(line)

    def _on_finish(rc: int):
        dlg.set_progress(100, 100, "Finished. Loading result…")
        _on_finished(
            dlg, doc, rc, dlg,
            input_path, output_path, stars_path,
            product, is_mono, work_dir, main,
            make_stars, unscreen,
        )

    worker.output_signal.connect(_on_out)
    worker.finished_signal.connect(_on_finish)
    worker.start()
    dlg.exec()


def open_rcastro_dialog(parent, doc=None, doc_manager=None,
                         list_open_docs_fn=None, rcastro_icon=None):
    """Open the RC-Astro tools dialog. doc_manager and list_open_docs_fn are
    accepted for backwards compatibility but not used — the dialog enumerates
    MDI subwindows directly via parent."""
    dlg = RCAstroDialog(
        parent,
        doc=doc,
        rcastro_icon=rcastro_icon,
    )
    dlg.show()
    dlg.raise_()
    return dlg


def open_rcastro_with_preset(main_window, preset: dict | None = None):
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            try:
                doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                       else getattr(dm, "active_document", None))
            except Exception:
                doc = None
    if doc is None:
        try:
            ad = getattr(main_window, "_active_doc", None)
            if callable(ad):
                doc = ad()
        except Exception:
            doc = None
    if doc is None or getattr(doc, "image", None) is None:
        return None

    _icon = None
    try:
        from PyQt6.QtGui import QIcon
        from setiastro.saspro.resources import rcastro_path
        _icon = QIcon(rcastro_path)
    except Exception:
        _icon = None

    dlg = RCAstroDialog(main_window, doc=doc, rcastro_icon=_icon)
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    try:
        main_window._rcastro_dialog = dlg   # retain against GC (WA_DeleteOnClose set in dialog)
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg