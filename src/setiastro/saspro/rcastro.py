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

        self.sld_ss  = _form_slider(form, "Sharpen Stars (0 – 0.7):",
                                    0.0, 0.7, 0.0, decimals=2, scale=100)
        self.sld_ash = _form_slider(form, "Adjust Star Halos (−0.5 – 0.5):",
                                    -0.5, 0.5, 0.0, decimals=2, scale=100)

        self.chk_auto_nsr = QCheckBox("Auto-detect nonstellar PSF radius  (recommended)")
        self.chk_auto_nsr.setChecked(True)
        form.addRow("", self.chk_auto_nsr)

        self.sld_nsr = _form_slider(form, "Manual Nonstellar Radius (0 – 8 px):",
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

    def _on_correct_only_toggled(self, checked: bool):
        self.sld_ss.setEnabled(not checked)
        self.sld_ash.setEnabled(not checked)
        self.sld_sn.setEnabled(not checked)

    def build_args(self) -> list[str]:
        args: list[str] = []

        if self.chk_correct_only.isChecked():
            args.append("--correct-only")
            # --correct-only forces all sharpening to 0 — only NSR is still valid
            if not self.chk_auto_nsr.isChecked():
                nsr = self.sld_nsr.value() / 10.0
                args += ["--no-auto-nonstellar-radius",
                         "--nonstellar-radius", f"{nsr:.1f}"]
        else:
            ss = self.sld_ss.value() / 100.0
            if ss > 0:
                args += ["--sharpen-stars", f"{ss:.2f}"]

            ash = self.sld_ash.value() / 100.0
            if abs(ash) > 0:
                args += ["--adjust-star-halos", f"{ash:.2f}"]

            if not self.chk_auto_nsr.isChecked():
                nsr = self.sld_nsr.value() / 10.0
                args += ["--no-auto-nonstellar-radius",
                         "--nonstellar-radius", f"{nsr:.1f}"]

            sn = self.sld_sn.value() / 100.0
            if sn > 0:
                args += ["--sharpen-nonstellar", f"{sn:.2f}"]

        return args

    def save_settings(self, s: QSettings):
        s.setValue("rcastro/bxt_correct_only", self.chk_correct_only.isChecked())
        s.setValue("rcastro/bxt_ss",   self.sld_ss.value())
        s.setValue("rcastro/bxt_ash",  self.sld_ash.value())
        s.setValue("rcastro/bxt_auto", self.chk_auto_nsr.isChecked())
        s.setValue("rcastro/bxt_nsr",  self.sld_nsr.value())
        s.setValue("rcastro/bxt_sn",   self.sld_sn.value())

    def load_settings(self, s: QSettings):
        self.chk_correct_only.setChecked(
            bool(s.value("rcastro/bxt_correct_only", False, type=bool)))
        self.sld_ss.setValue(          int( s.value("rcastro/bxt_ss",   0)))
        self.sld_ash.setValue(         int( s.value("rcastro/bxt_ash",  0)))
        self.chk_auto_nsr.setChecked( bool( s.value("rcastro/bxt_auto", True, type=bool)))
        self.sld_nsr.setValue(         int( s.value("rcastro/bxt_nsr",  0)))
        self.sld_sn.setValue(          int( s.value("rcastro/bxt_sn",   0)))
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
        args: list[str] = []
        if self.chk_stars.isChecked():
            args.append("--stars")
            if self.chk_unscreen.isChecked():
                args.append("--unscreen")
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
            cmd = [exe, "--no-banner", self._product] + extra_args
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
                [exe, "--no-banner", "--device"],
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
        cmd = [exe, "update", "--install"]
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
        try:
            r = subprocess.run(
                [exe, "--no-banner", "--help"],
                capture_output=True, text=True, timeout=8)
            out = (r.stdout or "") + (r.stderr or "")
            # Update device-flag detection while we have the help output
            s = QSettings()
            s.setValue("rcastro/uses_device_flag", bool("--device" in out))
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
        cmd = [exe, "--no-banner", "download-models"]
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

        also_stars = (product == "sxt" and self.sxt_panel.chk_stars.isChecked())
        self._run_product(exe, doc, product, panel_args, also_stars)

    def _run_product(self, exe: str, doc, product: str,
                     panel_args: list[str], also_stars: bool):
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

def _on_finished(main_dlg, doc, return_code, dlg,
                  input_path, output_path, stars_path,
                  product, is_mono, work_dir, main_window):
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

    # Collapse to mono if source was mono
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

    # SXT stars-only — open via docman.open_path, subwindow spawns automatically
    if product == "sxt" and os.path.exists(stars_path):
        # If the source was mono, collapse the RGB stars output back to mono so it
        # matches the original (otherwise it can't be combined/subtracted with it).
        # Mirrors the mono-collapse applied to the main starless result above.
        if is_mono:
            try:
                from setiastro.saspro.legacy.image_manager import save_image
                s_img, _, _, _ = load_image(stars_path)
                if s_img is not None and s_img.ndim == 3:
                    s_img = s_img.mean(axis=2).astype(np.float32)
                    save_image(np.clip(s_img, 0.0, 1.0), stars_path,
                               "tif", "32-bit floating point", None, False,
                               image_meta=None, file_meta=None)
            except Exception as e:
                dlg.append(f"[warn] could not collapse stars to mono: {e}\n")        
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
        # Also store human-readable params for re-display
        if product == "bxt":
            out["correct_only"]        = self._bxt.chk_correct_only.isChecked()
            out["sharpen_stars"]       = self._bxt.sld_ss.value()  / 100.0
            out["adjust_star_halos"]   = self._bxt.sld_ash.value() / 100.0
            out["auto_nsr"]            = self._bxt.chk_auto_nsr.isChecked()
            out["nonstellar_radius"]   = self._bxt.sld_nsr.value() / 10.0
            out["sharpen_nonstellar"]  = self._bxt.sld_sn.value()  / 100.0
        elif product == "sxt":
            out["stars"]    = self._sxt.chk_stars.isChecked()
            out["unscreen"] = self._sxt.chk_unscreen.isChecked()
        elif product == "nxt":
            out["denoise"]       = self._nxt.sld_dn.value()  / 100.0
            out["denoise_int"]   = self._nxt.sld_di.value()  / 100.0
            out["denoise_color"] = self._nxt.sld_dc.value()  / 100.0
            out["freq_scale"]    = self._nxt.sld_fs.value()  / 10.0
            out["iterations"]    = float(self._nxt.sp_iter.value())
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


def _apply_sxt_preset(panel: _SXTPanel, p: dict):
    if "stars" in p:
        panel.chk_stars.setChecked(bool(p["stars"]))
    if "unscreen" in p:
        panel.chk_unscreen.setChecked(bool(p["unscreen"]))


def _apply_nxt_preset(panel: _NXTPanel, p: dict):
    if "denoise" in p:
        panel.sld_dn.setValue(int(float(p["denoise"]) * 100))
    if "denoise_int" in p:
        panel.sld_di.setValue(int(float(p["denoise_int"]) * 100))
    if "denoise_color" in p:
        panel.sld_dc.setValue(int(float(p["denoise_color"]) * 100))
    if "freq_scale" in p:
        panel.sld_fs.setValue(int(float(p["freq_scale"]) * 10))
    if "iterations" in p:
        panel.sp_iter.setValue(float(p["iterations"]))


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

    # Re-build args from stored human-readable params if args list is empty
    if not args:
        tmp_s = QSettings()
        if product == "bxt":
            panel = _BXTPanel(); _apply_bxt_preset(panel, p)
            args = panel.build_args()
        elif product == "sxt":
            panel = _SXTPanel(); _apply_sxt_preset(panel, p)
            args = panel.build_args()
        elif product == "nxt":
            panel = _NXTPanel(); _apply_nxt_preset(panel, p)
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