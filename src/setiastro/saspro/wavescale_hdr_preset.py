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

        from .wavescale_hdr import clamp_amount
        self.amount = QSpinBox()
        self.amount.setRange(0, 100)
        self.amount.setSuffix(" %")
        self.amount.setValue(int(round(clamp_amount(p.get("amount", 1.0)) * 100)))
        self.amount.setToolTip(self.tr("Blend with the original image: 100% is the full effect, 0% leaves it unchanged."))

        f.addRow(self.tr("Number of Scales:"), self.n_scales)
        f.addRow(self.tr("Coarse Compression:"), self.comp)
        f.addRow(self.tr("Mask Gamma:"), self.gamma)
        f.addRow(self.tr("Amount:"), self.amount)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        f.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "n_scales": int(self.n_scales.value()),
            "compression_factor": float(self.comp.value()),
            "mask_gamma": float(self.gamma.value()),
            "amount": float(self.amount.value()) / 100.0,
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


def apply_wavescale_hdr_to_doc(doc, preset: dict | None = None, mask_out: list | None = None) -> None:
    """
    WaveScale HDR straight onto `doc`, with no dialog, thread or event loop:
    the steps the dialog takes in headless mode (_start_preview -> HDRWorker
    -> _on_finished -> _apply_to_doc), with its slider clamps and 0.01
    quantisation. Raises on failure where the dialog would show a box. Used
    by the command-line tool, which has no one to dismiss a box.

    mask_out: a list to receive the mask the blend used (2-D float32, the
    dialog's "HDR Mask" window at full size), appended once the edit is applied.
    """
    import numpy as np
    from .wavescale_hdr import compute_wavescale_hdr, _B3, clamp_amount, blend_amount
    from .wavescalede_preset import _get_doc_active_mask_2d

    p = dict(preset or {})
    ns = max(2, min(10, int(p.get("n_scales", 5))))
    comp = int(max(10, min(500, round(float(p.get("compression_factor", 1.5)) * 100)))) / 100.0
    mg = int(max(10, min(1000, round(float(p.get("mask_gamma", 5.0)) * 100)))) / 100.0
    amount = clamp_amount(p.get("amount", 1.0))

    base = getattr(doc, "image", None)
    if base is None:
        raise RuntimeError("Document has no image.")
    img = np.asarray(base, dtype=np.float32)
    if img.ndim == 2:
        rgb, mono_shape = np.repeat(img[:, :, None], 3, axis=2), img.shape
    elif img.ndim == 3 and img.shape[2] == 1:
        rgb, mono_shape = np.repeat(img, 3, axis=2), img.shape
    else:
        rgb, mono_shape = img[:, :, :3], None
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

    transformed, mask = compute_wavescale_hdr(rgb, ns, comp, mg, _B3)
    m_doc = _get_doc_active_mask_2d(doc, target_hw=rgb.shape[:2])
    if m_doc is not None:
        mask = np.clip(mask * m_doc, 0.0, 1.0)
    m3 = np.repeat(mask[..., None], 3, axis=2)
    out = rgb * (1.0 - m3) + transformed * m3
    out = blend_amount(rgb, out.astype(np.float32, copy=False), amount)

    if mono_shape is not None:
        out = np.mean(out, axis=2, dtype=np.float32)
        if len(mono_shape) == 3:
            out = out[:, :, None]
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    doc.set_image(out, metadata={"step_name": "WaveScale HDR", "command_id": "wavescale_hdr",
                                 "preset": {"n_scales": ns, "compression_factor": comp,
                                            "mask_gamma": mg, "amount": amount}},
                  step_name="WaveScale HDR")
    if mask_out is not None:
        mask_out.append(np.asarray(mask, dtype=np.float32))
