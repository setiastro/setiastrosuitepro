# pro/graxpert_preset.py
from __future__ import annotations
import os, shutil, tempfile
import numpy as np
from PyQt6.QtWidgets import (QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox,
    QRadioButton, QLabel, QComboBox)
from PyQt6.QtCore import QTimer

# Reuse your existing implementation
from .graxpert import (
    _resolve_graxpert_exec,
    _run_graxpert_command, _write_tiff_float32
)

def _build_graxpert_cmd(
    exe: str,
    operation: str,
    input_path: str,
    *,
    smoothing: float | None = None,
    strength: float | None = None,
    ai_version: str | None = None,
    gpu: bool = True,
    batch_size: int | None = None
) -> list[str]:
    op = "denoising" if operation == "denoise" else "background-extraction"
    cmd = [exe, "-cmd", op, input_path, "-cli", "-gpu", "true" if gpu else "false"]
    if op == "denoising":
        if strength is not None:
            cmd += ["-strength", f"{strength:.2f}"]
        if batch_size is not None:
            cmd += ["-batch_size", str(int(batch_size))]
        if ai_version:
            cmd += ["-ai_version", ai_version]
    else:
        if smoothing is not None:
            cmd += ["-smoothing", f"{smoothing:.2f}"]
    return cmd


# -------------------------- Headless runner --------------------------

def run_graxpert_via_preset(main_window, preset: dict | None = None, target_doc=None):
    """
    Headless GraXpert call:
      - No smoothing prompt (uses preset).
      - Optional GPU toggle from preset (default True).
      - Optional explicit exe override in preset["exe"].
      - Writes input_image.tif and runs same CLI as v2.
    """
    # 1) active doc (match your v2 access pattern)
    doc = target_doc
    if doc is None:
        doc = getattr(main_window, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main_window, "GraXpert", "Load an image first.")
        return

    p = dict(preset or {})
    op = str(p.get("op", "background")).lower()

    strength = p.get("strength", 0.50)
    smoothing = float(p.get("smoothing", 0.10))
    smoothing = max(0.0, min(1.0, smoothing))

    # ---- NEW: default GPU from QSettings if preset omitted it
    use_gpu_default = True
    try:
        if hasattr(main_window, "settings"):
            use_gpu_default = main_window.settings.value("graxpert/use_gpu", True, type=bool)
    except Exception:
        pass
    gpu = bool(p.get("gpu", use_gpu_default))

    # Optional batch_size override; otherwise pick sensible default from gpu
    batch_size = p.get("batch_size", None)
    if batch_size is None:
        batch_size = 4 if gpu else 1

    ai_version = p.get("ai_version")  # None or e.g. "3.0.2"

    # 2) GraXpert executable
    exe = p.get("exe") or _resolve_graxpert_exec(main_window)
    if not exe:
        return

    # 3) temp write (unchanged)
    workdir = tempfile.mkdtemp(prefix="saspro_graxpert_")
    input_basename = "input_image"
    input_path = os.path.join(workdir, f"{input_basename}.tif")
    try:
        _write_tiff_float32(doc.image, input_path)
    except Exception as e:
        QMessageBox.critical(main_window, "GraXpert", f"Failed to write temporary input:\n{e}")
        shutil.rmtree(workdir, ignore_errors=True)
        return

    # 4) exact command (now with resolved gpu + batch_size)
    command = _build_graxpert_cmd(
        exe,
        op,
        input_path,
        smoothing=float(smoothing) if op == "background" else None,
        strength=float(strength) if op == "denoise" else None,
        ai_version=str(ai_version) if (op == "denoise" and ai_version) else None,
        gpu=gpu,
        batch_size=int(batch_size) if batch_size is not None else None,
    )

    # Persist the resolved GPU preference so dialogs/presets stay in sync
    try:
        if hasattr(main_window, "settings"):
            main_window.settings.setValue("graxpert/use_gpu", bool(gpu))
    except Exception:
        pass

    # 5) store normalized preset for Replay
    try:
        preset_for_replay = {"op": op, "gpu": bool(gpu)}
        if op == "background":
            preset_for_replay["smoothing"] = float(smoothing)
        else:
            preset_for_replay["strength"] = float(strength)
            if ai_version:
                preset_for_replay["ai_version"] = ai_version
        if batch_size is not None:
            preset_for_replay["batch_size"] = int(batch_size)

        op_label = "GraXpert Denoise" if op == "denoise" else "GraXpert Gradient Removal"

        remember = getattr(main_window, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main_window, "_remember_last_headless_command", None)
        if callable(remember):
            remember("graxpert", preset_for_replay, description=op_label)
            try:
                if hasattr(main_window, "_log"):
                    main_window._log(
                        f"[Replay] GraXpert preset stored: op={op}, "
                        f"keys={list(preset_for_replay.keys())}"
                    )
            except Exception:
                pass
    except Exception:
        op_label = "GraXpert Denoise" if op == "denoise" else "GraXpert Gradient Removal"

    # 6) run
    output_basename = f"{input_basename}_GraXpert"

    meta_extras = {
        "graxpert_operation": op,                             # "denoise" | "background"
        "graxpert_param": float(strength if op == "denoise" else smoothing),
        "graxpert_ai_version": (ai_version or "latest") if op == "denoise" else None,
        "graxpert_gpu": bool(gpu),
    }

    setattr(main_window, "_graxpert_headless_running", True)
    setattr(main_window, "_graxpert_silent", True)
    setattr(main_window, "_graxpert_guard", True)
    try:
        _run_graxpert_command(
            main_window,
            command,
            output_basename,
            workdir,
            target_doc=doc,
        )
    finally:
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
      - Optional denoise model (ai_version); blank = latest/auto
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

        smoothing = float(p.get("smoothing", 0.10))
        strength  = float(p.get("strength", 0.50))

        def _set_for_background():
            self.param_label.setText("Smoothing (0–1):")
            self.param.setValue(smoothing)
            self.model_label.setEnabled(False)
            self.model_combo.setEnabled(False)

        def _set_for_denoise():
            self.param_label.setText("Strength (0–1):")
            self.param.setValue(strength)
            self.model_label.setEnabled(True)
            self.model_combo.setEnabled(True)

        # --- denoise model picker ---
        self.model_label = QLabel("Denoise model:")
        self.model_combo = QComboBox()
        self.model_combo.addItem("Latest (auto)", "")
        for v in ["3.0.2", "3.0.1", "3.0.0", "2.0.0", "1.1.0", "1.0.0"]:
            self.model_combo.addItem(v, v)
        # preselect from preset if provided
        ai_ver = str(p.get("ai_version") or "")
        idx = max(0, self.model_combo.findData(ai_ver))
        self.model_combo.setCurrentIndex(idx)

        self.rb_background.toggled.connect(lambda checked: _set_for_background() if checked else None)
        self.rb_denoise.toggled.connect(lambda checked: _set_for_denoise() if checked else None)

        # Initialize label/value once
        if self.rb_denoise.isChecked():
            _set_for_denoise()
        else:
            _set_for_background()

        form.addRow(self.param_label, self.param)
        form.addRow(self.model_label, self.model_combo)

        # --- GPU ---
        self.gpu = QCheckBox("Use GPU (if available)")
        use_gpu_default = True
        try:
            settings = getattr(parent, "settings", None)
            if settings is not None:
                use_gpu_default = settings.value("graxpert/use_gpu", True, type=bool)
        except Exception:
            pass
        self.gpu.setChecked(bool(p.get("gpu", use_gpu_default)))

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
        out = {"op": op, "gpu": bool(self.gpu.isChecked())}
        if op == "background":
            out["smoothing"] = float(self.param.value())
        else:
            out["strength"] = float(self.param.value())
            ai_version = self.model_combo.currentData() or ""
            if ai_version:  # only include if explicitly chosen
                out["ai_version"] = ai_version
        return out
