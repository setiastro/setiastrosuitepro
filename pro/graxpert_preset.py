# pro/graxpert_preset.py
from __future__ import annotations
import os, shutil, tempfile
import numpy as np
from PyQt6.QtWidgets import QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox
from PyQt6.QtCore import QTimer

# Reuse your existing implementation
from .graxpert import (
    _resolve_graxpert_exec,
    _write_tiff_like_v2,
    _run_graxpert_command,
)



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
    smoothing = float(p.get("smoothing", 0.10))
    smoothing = max(0.0, min(1.0, smoothing))
    gpu = bool(p.get("gpu", True))

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
    command = [
        exe,
        "-cmd", "background-extraction",
        input_path,
        "-cli",
        "-smoothing", f"{smoothing:.2f}",
        "-gpu", "true" if gpu else "false",
    ]

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
    Minimal preset editor: smoothing [0..1], GPU on/off.
    Path to exe is still handled by your resolver and QSettings.
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("GraXpert — Preset")
        p = dict(initial or {})

        form = QFormLayout(self)
        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.0, 1.0)
        self.smoothing.setDecimals(2)
        self.smoothing.setSingleStep(0.01)
        self.smoothing.setValue(float(p.get("smoothing", 0.10)))

        self.gpu = QCheckBox("Use GPU (if available)")
        self.gpu.setChecked(bool(p.get("gpu", True)))

        form.addRow("Smoothing:", self.smoothing)
        form.addRow(self.gpu)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "smoothing": float(self.smoothing.value()),
            "gpu": bool(self.gpu.isChecked()),
            # optional: allow "exe": "/custom/path/to/GraXpert"
        }
