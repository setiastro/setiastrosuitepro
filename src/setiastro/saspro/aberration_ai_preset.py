# pro/aberration_ai_preset.py
from __future__ import annotations
import os
import time
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QPushButton, QMessageBox, QFormLayout, QDialogButtonBox, QSpinBox, QCheckBox, QComboBox, QLabel, QApplication

from PyQt6.QtCore import QSettings
# reuse everything from the UI module
from .aberration_ai import (
    ort, IS_APPLE_ARM,
    _ONNXWorker, pick_providers, _preserve_border
)

# ---------------------- Headless entry ----------------------

def run_aberration_ai_via_preset(main, preset: dict | None = None, doc=None):
    """
    Headless Aberration AI

    preset keys (all optional except model):
      - model: str (path to .onnx). If omitted, uses QSettings "AberrationAI/model_path".
      - patch: int (default 512)
      - overlap: int (default 64)
      - border_px: int (default 10)
      - auto_gpu: bool (default True; forced False on Apple Silicon)
      - provider: str (used when auto_gpu=False), e.g. "CPUExecutionProvider",
                  "CUDAExecutionProvider", "DmlExecutionProvider"
    """
    if ort is None:
        QMessageBox.critical(main, "Aberration AI", "onnxruntime not installed.")
        return

    # active doc
    if doc is None:
        d = getattr(main, "_active_doc", None)
        doc = d() if callable(d) else d

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main, "Aberration AI", "Load an image first.")
        return

    p = dict(preset or {})

    # model path (preset beats QSettings)
    model = p.get("model") or QSettings().value("AberrationAI/model_path", type=str)
    if not model or not os.path.isfile(model):
        QMessageBox.warning(main, "Aberration AI", "Model not set. Open the Aberration AI tool once and choose a model, or put 'model' into the preset.")
        return

    patch    = int(p.get("patch", 512))
    overlap  = int(p.get("overlap", 64))
    border_px= int(p.get("border_px", 10))

    # providers
    if IS_APPLE_ARM:
        providers = ["CPUExecutionProvider"]
        auto_gpu = False
        provider_label = "CPUExecutionProvider"
    else:
        auto_gpu = bool(p.get("auto_gpu", True))
        if auto_gpu:
            providers = pick_providers(auto_gpu=True)
            provider_label = "auto"
        else:
            sel = str(p.get("provider", "CPUExecutionProvider"))
            providers = [sel]
            provider_label = sel or "CPUExecutionProvider"

    # Safety for CoreML if someone forces it
    if "CoreMLExecutionProvider" in providers and patch > 128:
        patch = 128

    # Guard so interactive dialog won't pop during/after apply
    setattr(main, "_aberration_ai_headless_running", True)
    setattr(main, "_aberration_ai_guard", True)

    # ---- minimal progress dialog ----
    dlg = QDialog(main)
    dlg.setWindowTitle("Aberration AI (Headless)")
    lay = QVBoxLayout(dlg)
    bar = QProgressBar(); bar.setRange(0, 100); lay.addWidget(bar)
    btn = QPushButton("Cancel"); lay.addWidget(btn)

    img = np.asarray(doc.image)
    orig_for_border = img.copy()

    t0 = time.perf_counter()

    worker = _ONNXWorker(model, img, patch, overlap, providers)
    worker.progressed.connect(bar.setValue)

    def _cancel_clicked():
        btn.setEnabled(False)
        btn.setText("Canceling…")
        worker.cancel()  # <-- SAFE
        QApplication.processEvents()

    def _fail(msg: str):
        try:
            if hasattr(main, "_log"):
                main._log(f"❌ Aberration AI failed: {msg}")
        except Exception:
            pass
        # If canceled, don't pop an error box
        if "Canceled" not in (msg or ""):
            QMessageBox.critical(main, "Aberration AI", msg)
        dlg.close()

    def _canceled():
        try:
            if hasattr(main, "_log"):
                main._log("⛔ Aberration AI canceled.")
        except Exception:
            pass
        dlg.close()

    def _ok(out: np.ndarray):
        # preserve border and commit
        try:
            out2 = _preserve_border(out, orig_for_border, border_px)
        except Exception:
            out2 = out

        meta = {
            "is_mono": (out2.ndim == 2),
            "processing_parameters": {
                "AberrationAI": {
                    "model_path": model,
                    "patch_size": int(patch),
                    "overlap": int(overlap),
                    "provider": provider_label,
                    "border_px": int(border_px),
                }
            }
        }
        try:
            doc.apply_edit(out2, meta, step_name="Aberration AI")
            used = getattr(worker, "used_provider", provider_label)
            dt = time.perf_counter() - t0
            if hasattr(main, "_log"):
                main._log(
                    f"✅ Aberration AI (headless) model={os.path.basename(model)}, "
                    f"provider={used}, patch={patch}, overlap={overlap}, "
                    f"border={border_px}px, time={dt:.2f}s"
                )

            # ---- Register as last_headless_command for Replay ----
            try:
                auto_flag = bool(auto_gpu)
                replay_preset = {
                    "model": model,
                    "patch": int(patch),
                    "overlap": int(overlap),
                    "border_px": int(border_px),
                    "auto_gpu": auto_flag,
                }
                if not auto_flag:
                    replay_preset["provider"] = provider_label

                payload = {
                    "command_id": "aberrationai",
                    "preset": replay_preset,
                }
                setattr(main, "_last_headless_command", payload)
            except Exception:
                pass
            # -------------------------------------------------------
        except Exception as e:
            QMessageBox.critical(main, "Aberration AI", f"Failed to apply result:\n{e}")
        finally:
            dlg.close()

    worker.failed.connect(_fail)
    worker.canceled.connect(_canceled)          # <-- NEW
    worker.finished_ok.connect(_ok)
    worker.finished.connect(lambda: btn.setEnabled(False))

    btn.clicked.connect(_cancel_clicked)

    # If user closes dialog via window X, also cancel
    dlg.rejected.connect(_cancel_clicked)

    worker.start()
    dlg.exec()

    # Ensure the worker is not left running after the modal closes
    if worker.isRunning():
        worker.cancel()
        worker.wait(2000)  # don't hang forever; just give it a moment

    # clear the guard after a brief tick so downstream signals don’t re-open UI
    def _clear():
        for k in ("_aberration_ai_headless_running", "_aberration_ai_guard"):
            try: delattr(main, k)
            except Exception: setattr(main, k, False)
    QTimer.singleShot(1000, _clear)


# ---------------------- Preset editor (for shortcut) ----------------------

class AberrationAIPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Aberration AI — Preset")
        p = dict(initial or {})

        self.spin_patch   = QSpinBox(); self.spin_patch.setRange(128, 2048); self.spin_patch.setValue(int(p.get("patch", 512)))
        self.spin_overlap = QSpinBox(); self.spin_overlap.setRange(16, 512);  self.spin_overlap.setValue(int(p.get("overlap", 64)))
        self.spin_border  = QSpinBox(); self.spin_border.setRange(0, 64);     self.spin_border.setValue(int(p.get("border_px", 10)))

        self.chk_auto = QCheckBox("Auto GPU (prefer DML/CUDA)"); self.chk_auto.setChecked(bool(p.get("auto_gpu", True)))
        self.cmb_provider = QComboBox(); self.cmb_provider.addItems([
            "CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"
        ])
        self.cmb_provider.setCurrentText(str(p.get("provider", "CPUExecutionProvider")))

        # info: model is taken from QSettings unless preset provides an absolute path from code
        info = QLabel("Model path is taken from the Aberration AI tool (QSettings) unless you pass 'model' in the preset programmatically.")
        info.setWordWrap(True); info.setStyleSheet("color:#888; font-size:11px;")

        form = QFormLayout(self)
        form.addRow("Patch:", self.spin_patch)
        form.addRow("Overlap:", self.spin_overlap)
        form.addRow("Preserve border (px):", self.spin_border)
        form.addRow(self.chk_auto)
        form.addRow("Provider (if Auto off):", self.cmb_provider)
        form.addRow(info)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

        # toggle enable
        def _toggle():
            en = not self.chk_auto.isChecked()
            self.cmb_provider.setEnabled(en)
        self.chk_auto.stateChanged.connect(lambda _: _toggle()); _toggle()

    def result_dict(self) -> dict:
        d = {
            "patch": int(self.spin_patch.value()),
            "overlap": int(self.spin_overlap.value()),
            "border_px": int(self.spin_border.value()),
            "auto_gpu": bool(self.chk_auto.isChecked()),
        }
        if not d["auto_gpu"]:
            d["provider"] = self.cmb_provider.currentText()
        return d
