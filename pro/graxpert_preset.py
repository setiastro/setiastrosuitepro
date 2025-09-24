# pro/graxpert_preset.py
from __future__ import annotations
import os, shutil, tempfile
import numpy as np
from PyQt6.QtWidgets import (QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox,
    QRadioButton, QLabel)
from PyQt6.QtCore import QTimer

# Reuse your existing implementation
from .graxpert import (
    _resolve_graxpert_exec,
    _write_tiff_like_v2,
    _run_graxpert_command,
)

def _build_graxpert_cmd(exe: str, operation: str, input_path: str, *, smoothing: float | None = None,
                        strength: float | None = None, gpu: bool = True, batch_size: int | None = None) -> list[str]:
    op = "denoising" if operation == "denoise" else "background-extraction"
    cmd = [exe, "-cmd", op, input_path, "-cli", "-gpu", "true" if gpu else "false"]
    if op == "denoising":
        if strength is not None:
            cmd += ["-strength", f"{strength:.2f}"]
        if batch_size is not None:
            cmd += ["-batch_size", str(int(batch_size))]
    else:
        if smoothing is not None:
            cmd += ["-smoothing", f"{smoothing:.2f}"]
    return cmd


# -------------------------- Headless runner --------------------------

def run_graxpert_via_preset(main_window, preset: dict | None = None):
    """
    Headless GraXpert call:
      - No smoothing prompt (uses preset).
      - Optional GPU toggle from preset (default True).
      - Optional explicit exe override in preset["exe"].
      - Writes input_image.tif and runs same CLI as v2.
    """
    # 1) active doc (match your v2 access pattern)
    doc = getattr(main_window, "_active_doc", None)
    if callable(doc):
        doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main_window, "GraXpert", "Load an image first.")
        return

    p = dict(preset or {})
    op = str(p.get("op", "background")).lower()  # "background" or "denoise"
    strength = p.get("strength", 0.50)    
    smoothing = float(p.get("smoothing", 0.10))
    smoothing = max(0.0, min(1.0, smoothing))
    gpu = bool(p.get("gpu", True))
    batch_size = p.get("batch_size", None)

    # 2) GraXpert executable
    exe = p.get("exe") or _resolve_graxpert_exec(main_window)
    if not exe:
        return

    # 3) temp working dir + write input (same basenames as v2)
    workdir = tempfile.mkdtemp(prefix="saspro_graxpert_")
    input_basename = "input_image"
    input_path = os.path.join(workdir, f"{input_basename}.tif")
    try:
        _write_tiff_like_v2(doc.image, input_path)
    except Exception as e:
        QMessageBox.critical(main_window, "GraXpert", f"Failed to write temporary input:\n{e}")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    # 4) exact command (with preset smoothing & gpu flag)
    command = _build_graxpert_cmd(
        exe,
        op,
        input_path,
        smoothing=float(smoothing) if op == "background" else None,
        strength=float(strength) if op == "denoise" else None,
        gpu=bool(p.get("gpu", True)),
        batch_size=int(batch_size) if batch_size is not None else None,
    )

    # 5) run (reuses your threaded dialog + legacy loader pipeline)
    output_basename = f"{input_basename}_GraXpert"

    # mark this session as headless/silent & block the UI with a cool-down guard
    setattr(main_window, "_graxpert_headless_running", True)
    setattr(main_window, "_graxpert_silent", True)
    setattr(main_window, "_graxpert_guard", True)

    try:
        _run_graxpert_command(main_window, command, output_basename, workdir, target_doc=doc)
    finally:
        # don't drop the guard immediately — give the app a moment to settle
        def _clear_flags():
            for name in ("_graxpert_headless_running", "_graxpert_silent", "_graxpert_guard"):
                try:
                    delattr(main_window, name)
                except Exception:
                    setattr(main_window, name, False)
        QTimer.singleShot(1200, _clear_flags) 

# -------------------------- Preset editor (optional) --------------------------

class GraXpertPresetDialog(QDialog):
    """
    Preset editor:
      - Operation: Remove gradient (background) OR Denoise
      - Parameter: Smoothing (0..1) for background OR Strength (0..1) for denoise
      - GPU on/off
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("GraXpert — Preset")
        p = dict(initial or {})
        op = str(p.get("op", "background")).lower()  # "background" or "denoise"

        form = QFormLayout(self)

        # --- operation radios ---
        self.rb_background = QRadioButton("Remove gradient")
        self.rb_denoise    = QRadioButton("Denoise")

        if op == "denoise":
            self.rb_denoise.setChecked(True)
        else:
            self.rb_background.setChecked(True)

        form.addRow(self.rb_background)
        form.addRow(self.rb_denoise)

        # --- shared numeric control (label changes with op) ---
        self.param_label = QLabel()
        self.param = QDoubleSpinBox()
        self.param.setRange(0.0, 1.0)
        self.param.setDecimals(2)
        self.param.setSingleStep(0.01)

        # Defaults/initials
        smoothing = float(p.get("smoothing", 0.10))
        strength  = float(p.get("strength", 0.50))

        def _set_for_background():
            self.param_label.setText("Smoothing (0–1):")
            self.param.setValue(smoothing)

        def _set_for_denoise():
            self.param_label.setText("Strength (0–1):")
            self.param.setValue(strength)

        self.rb_background.toggled.connect(lambda checked: _set_for_background() if checked else None)
        self.rb_denoise.toggled.connect(lambda checked: _set_for_denoise() if checked else None)

        # Initialize label/value once
        if self.rb_denoise.isChecked():
            _set_for_denoise()
        else:
            _set_for_background()

        form.addRow(self.param_label, self.param)

        # --- GPU ---
        self.gpu = QCheckBox("Use GPU (if available)")
        self.gpu.setChecked(bool(p.get("gpu", True)))
        form.addRow(self.gpu)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        op = "denoise" if self.rb_denoise.isChecked() else "background"
        out = {
            "op": op,
            "gpu": bool(self.gpu.isChecked()),
        }
        # Only include the relevant parameter; harmless if you include both,
        # but this keeps the dict clean.
        if op == "background":
            out["smoothing"] = float(self.param.value())
        else:
            out["strength"] = float(self.param.value())
        return out
