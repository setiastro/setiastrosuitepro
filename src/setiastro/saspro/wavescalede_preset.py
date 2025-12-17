from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QMessageBox
from .wavescalede import compute_wavescale_dse

# ─────────────────────────────────────────────────────────────────────────────
# Preset editor
# ─────────────────────────────────────────────────────────────────────────────
class WaveScaleDSEPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("WaveScale Dark Enhancer — Preset")
        p = dict(initial or {})
        f = QFormLayout(self)

        self.n_scales = QSpinBox()
        self.n_scales.setRange(2, 10)
        self.n_scales.setValue(int(p.get("n_scales", 6)))

        self.boost = QDoubleSpinBox()
        self.boost.setRange(0.10, 10.00)
        self.boost.setDecimals(2)
        self.boost.setSingleStep(0.05)
        self.boost.setValue(float(p.get("boost_factor", 5.0)))

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.10, 10.00)
        self.gamma.setDecimals(2)
        self.gamma.setSingleStep(0.10)
        self.gamma.setValue(float(p.get("mask_gamma", 1.0)))

        self.iters = QSpinBox()
        self.iters.setRange(1, 10)
        self.iters.setValue(int(p.get("iterations", 2)))

        f.addRow("Number of Scales:", self.n_scales)
        f.addRow("Boost Factor:", self.boost)
        f.addRow("Mask Gamma:", self.gamma)
        f.addRow("Iterations:", self.iters)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        f.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "n_scales": int(self.n_scales.value()),
            "boost_factor": float(self.boost.value()),
            "mask_gamma": float(self.gamma.value()),
            "iterations": int(self.iters.value()),
        }




# ─────────────────────────────────────────────────────────────────────────────
# Headless runner (exactly like UI: compute → blend by active mask → apply)
# ─────────────────────────────────────────────────────────────────────────────
def run_wavescalede_via_preset(main, preset: dict | None = None, target_doc=None):
    import numpy as np
    from PyQt6.QtWidgets import QMessageBox

    p = dict(preset or {})

    # --- sanitize to UI limits so replay is clean ---
    try:
        n_scales = int(np.clip(p.get("n_scales", 6), 2, 10))
        boost    = float(np.clip(p.get("boost_factor", 5.0), 0.10, 10.00))
        mgamma   = float(np.clip(p.get("mask_gamma", 1.0), 0.10, 10.00))
        iters    = int(np.clip(p.get("iterations", 2), 1, 10))
    except Exception:
        n_scales = int(p.get("n_scales", 6))
        boost    = float(p.get("boost_factor", 5.0))
        mgamma   = float(p.get("mask_gamma", 1.0))
        iters    = int(p.get("iterations", 2))

    params = {
        "n_scales": n_scales,
        "boost_factor": boost,
        "mask_gamma": mgamma,
        "iterations": iters,
    }

    # --- store for Replay (prefer unified helper) ---
    try:
        remember = getattr(main, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main, "_remember_last_headless_command", None)
        if callable(remember):
            remember("wavescale_dark_enhance", params, description="WaveScale Dark Enhance")
        else:
            setattr(main, "_last_headless_command", {
                "command_id": "wavescale_dark_enhance",
                "preset": dict(params),
            })
    except Exception:
        pass

    # resolve target doc
    from setiastro.saspro.headless_utils import normalize_headless_main, unwrap_docproxy

    main, doc, _dm = normalize_headless_main(main, target_doc)
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.warning(main or None, "...", "Load an image first.")
        return

    # pull & normalize image like the dialog
    base = np.asarray(doc.image, dtype=np.float32)
    was_mono = False
    mono_shape = None
    if base.ndim == 2:
        was_mono = True
        mono_shape = base.shape
        img = np.repeat(base[:, :, None], 3, axis=2)
    elif base.ndim == 3 and base.shape[2] == 1:
        was_mono = True
        mono_shape = base.shape
        img = np.repeat(base, 3, axis=2)
    else:
        img = base[:, :, :3]

    if base.dtype.kind in "ui":
        mx = float(np.nanmax(img)) or 1.0
        img = img / max(1.0, mx)
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

    # fetch active doc mask → 2D [0..1], resized to image (same logic as dialog)
    doc_mask = _get_doc_active_mask_2d(doc, target_hw=img.shape[:2])

    # compute (limit enhancement with external mask, same as UI preview)
    out, _mask_used = compute_wavescale_dse(
        img,
        n_scales=n_scales,
        boost_factor=boost,
        mask_gamma=mgamma,
        iterations=iters,
        external_mask=doc_mask
    )

    # if a doc mask exists, blend final result with original (exactly like dialog)
    if doc_mask is not None:
        m3 = np.repeat(doc_mask[:, :, None], 3, axis=2).astype(np.float32)
        blended = img * (1.0 - m3) + out * m3
    else:
        blended = out

    # collapse back to mono if needed (like dialog apply)
    result = blended
    if was_mono:
        mono = np.mean(result, axis=2, dtype=np.float32)
        if mono_shape and len(mono_shape) == 3 and mono_shape[2] == 1:
            mono = mono[:, :, None]
        result = mono

    result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

    # apply to document (undoable + metadata)
    meta = {
        "step_name": "WaveScale Dark Enhance",
        "wavescale_dark_enhance": dict(params),
        "masked": bool(doc_mask is not None),
        "mask_blend": "m*out + (1-m)*src" if doc_mask is not None else "none",
        "bit_depth": "32-bit floating point",
        "is_mono": (result.ndim == 2 or (result.ndim == 3 and result.shape[2] == 1)),
    }

    try:
        if hasattr(doc, "apply_edit"):
            doc.apply_edit(result, step_name="WaveScale Dark Enhance", metadata=meta)
        elif hasattr(doc, "set_image"):
            doc.set_image(result, step_name="WaveScale Dark Enhance")
        elif hasattr(doc, "apply_numpy"):
            doc.apply_numpy(result, step_name="WaveScale Dark Enhance")
        else:
            doc.image = result
    except Exception as e:
        QMessageBox.critical(main, "WaveScale Dark Enhancer", f"Failed to write to document:\n{e}")
        return


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirror dialog’s mask resolution)
# ─────────────────────────────────────────────────────────────────────────────
def _get_doc_active_mask_2d(doc, *, target_hw):
    """
    Return the document's active mask as 2-D float32 [0..1], resized to target_hw.
    """
    mid = getattr(doc, "active_mask_id", None)
    if not mid:
        return None
    masks = getattr(doc, "masks", {}) or {}
    layer = masks.get(mid)
    if layer is None:
        return None

    data = None
    for attr in ("data", "mask", "image", "array"):
        if hasattr(layer, attr):
            val = getattr(layer, attr)
            if val is not None:
                data = val
                break
    if data is None and isinstance(layer, dict):
        for key in ("data", "mask", "image", "array"):
            if key in layer and layer[key] is not None:
                data = layer[key]
                break
    if data is None and isinstance(layer, np.ndarray):
        data = layer
    if data is None:
        return None

    m = np.asarray(data)
    if m.ndim == 3:
        m = m.mean(axis=2)

    m = m.astype(np.float32, copy=False)
    mx = float(m.max()) if m.size else 1.0
    if mx > 1.0:
        m /= mx
    m = np.clip(m, 0.0, 1.0)

    H, W = target_hw
    if m.shape != (H, W):
        yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
        m = m[yi][:, xi]
    return m
