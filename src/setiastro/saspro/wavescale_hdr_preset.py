from __future__ import annotations
from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QMessageBox
from PyQt6.QtCore import QSettings
from .wavescale_hdr import WaveScaleHDRDialogPro

class WaveScaleHDRPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("WaveScale HDR — Preset"))
        p = dict(initial or {})
        f = QFormLayout(self)

        self.n_scales = QSpinBox()
        self.n_scales.setRange(2, 10)
        self.n_scales.setValue(int(p.get("n_scales", 5)))

        self.comp = QDoubleSpinBox()
        self.comp.setRange(0.10, 5.00)
        self.comp.setSingleStep(0.05)
        self.comp.setDecimals(2)
        self.comp.setValue(float(p.get("compression_factor", 1.5)))

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.10, 10.00)
        self.gamma.setSingleStep(0.10)
        self.gamma.setDecimals(2)
        self.gamma.setValue(float(p.get("mask_gamma", 5.0)))

        f.addRow(self.tr("Number of Scales:"), self.n_scales)
        f.addRow(self.tr("Coarse Compression:"), self.comp)
        f.addRow(self.tr("Mask Gamma:"), self.gamma)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        f.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "n_scales": int(self.n_scales.value()),
            "compression_factor": float(self.comp.value()),
            "mask_gamma": float(self.gamma.value()),
        }



def run_wavescale_hdr_via_preset(main, preset: dict | None = None, target_doc=None, ctx=None):
    """
    Drive WaveScale HDR headlessly: set dialog controls from 'preset',
    run Preview, then auto-Apply on finish (same pipeline as UI).

    Returns True when applied. From a script (`ctx` given, as
    ctx.run_command does) a failure raises RuntimeError instead.
    """
    p = dict(preset or {})

    # Register as last action for replay (same schema as dialog)
    try:
        payload = {"command_id": "wavescale_hdr", "preset": dict(p)}
        setattr(main, "_last_headless_command", payload)
    except Exception:
        pass

    # Guard flags like other headless tools
    setattr(main, "_wavescale_headless_running", True)
    setattr(main, "_wavescale_guard", True)
    s = QSettings()
    try:
        import time
        s.setValue("wavescale/headless_in_progress", True)
        s.setValue("wavescale/headless_started_at", time.time())
        s.sync()
    except Exception:
        pass

    from setiastro.saspro.headless_utils import normalize_headless_main, unwrap_docproxy

    main, doc, _dm = normalize_headless_main(main, target_doc)
    if doc is None or getattr(doc, "image", None) is None:
        _clear_wavescale_flags(main, s)
        if ctx is not None:
            raise RuntimeError("WaveScale HDR: no image to process.")
        from PyQt6.QtCore import QCoreApplication
        QMessageBox.warning(main or None, "...", QCoreApplication.translate("WaveScaleHDRPresetDialog", "Load an image first."))
        return False

    # Build headless dialog and apply preset
    dlg = WaveScaleHDRDialogPro(main, doc, headless=True, bypass_guard=True)
    try:
        dlg.apply_preset(p)
    except Exception:
        pass

    # Start compute (same as pressing Preview), then run the event loop until
    # the dialog finishes: accept() after the apply, reject() on any failure.
    result = 0
    try:
        dlg._start_preview()
        result = dlg.exec()
    finally:
        _clear_wavescale_flags(main, s)
        try:
            dlg.close()          # closeEvent releases the full-size buffers
            dlg.deleteLater()
        except Exception:
            pass
    ok = (result == QDialog.DialogCode.Accepted)
    if not ok and ctx is not None:
        raise RuntimeError("WaveScale HDR did not complete (see the message shown).")
    return ok


def _clear_wavescale_flags(main, settings):
    try:
        settings.setValue("wavescale/headless_in_progress", False); settings.sync()
    except Exception:
        pass
    for name in ("_wavescale_headless_running", "_wavescale_guard"):
        try: delattr(main, name)
        except Exception: setattr(main, name, False)
