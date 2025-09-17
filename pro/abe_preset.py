# pro/abe_preset.py
from __future__ import annotations
import os
import numpy as np

from .abe import abe_run  # core engine

# ---------- mask helpers (match ABEDialog semantics) ----------
def _active_mask_layer(doc):
    mid = getattr(doc, "active_mask_id", None)
    if not mid: return None, None, None
    layer = getattr(doc, "masks", {}).get(mid)
    if layer is None: return None, None, None
    m = np.asarray(getattr(layer, "data", None))
    if m is None or m.size == 0: return None, None, None
    m = m.astype(np.float32, copy=False)
    if m.dtype.kind in "ui":
        m /= float(np.iinfo(m.dtype).max)
    else:
        mx = float(m.max()) if m.size else 1.0
        if mx > 1.0: m /= mx
    return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

def _resample_mask(mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
    mh, mw = mask.shape[:2]
    th, tw = out_hw
    if (mh, mw) == (th, tw): return mask
    yi = np.linspace(0, mh - 1, th).astype(np.int32)
    xi = np.linspace(0, mw - 1, tw).astype(np.int32)
    return mask[yi][:, xi]

def _blend_with_mask_float(processed: np.ndarray, src: np.ndarray, doc) -> np.ndarray:
    mask, _mid, _mname = _active_mask_layer(doc)
    if mask is None:
        return processed
    out = processed.astype(np.float32, copy=False)
    s   = src.astype(np.float32, copy=False)

    m = _resample_mask(mask, out.shape[:2])
    # reconcile channels
    if out.ndim == 2 and s.ndim == 3:
        out = out[..., None]
    if s.ndim == 2 and out.ndim == 3:
        s = s[..., None]
    if out.ndim == 3 and out.shape[2] == 3 and m.ndim == 2:
        m = m[..., None]
    blended = (m * out + (1.0 - m) * s).astype(np.float32, copy=False)
    if blended.ndim == 3 and blended.shape[2] == 1:
        blended = blended[..., 0]
    return np.clip(blended, 0.0, 1.0)

# ---------- I/O helpers ----------
def _doc_image_float01(doc) -> np.ndarray | None:
    img = getattr(doc, "image", None)
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype.kind in "ui":
        return (arr.astype(np.float32) / np.iinfo(arr.dtype).max).clip(0.0, 1.0)
    return np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)

# ---------- headless apply (NO EXCLUSION AREA) ----------
def apply_abe_via_preset(main_window, doc, preset: dict | None = None):
    """
    Run ABE headlessly (no exclusion polygons; exclusion_mask=None) using preset params.
    Blends with the active mask layer (m*out + (1-m)*src) before committing.
    """
    p = dict(preset or {})
    degree      = int(np.clip(p.get("degree",      2), 1, 6))
    num_samples = int(np.clip(p.get("samples",   120), 20, 100000))
    downsample  = int(np.clip(p.get("downsample",  6), 1, 64))
    patch_size  = int(np.clip(p.get("patch",     15), 5, 151))
    use_rbf     = bool(p.get("rbf", True))
    rbf_smooth  = float(p.get("rbf_smooth", 1.0))  # dialog default = 100 × 0.01 = 1.0
    make_bg_doc = bool(p.get("make_background_doc", False))

    src01 = _doc_image_float01(doc)
    if src01 is None:
        return

    # 🚫 No exclusion polygons in headless mode
    corrected, bg = abe_run(
        src01,
        degree=degree, num_samples=num_samples, downsample=downsample, patch_size=patch_size,
        use_rbf=use_rbf, rbf_smooth=rbf_smooth,
        exclusion_mask=None,                   # ←← force NO EXCLUSION AREA
        return_background=True,
        progress_cb=None
    )

    # Preserve mono vs color wrt original doc
    out = corrected
    if out.ndim == 3 and out.shape[2] == 3 and (doc.image.ndim == 2 or (doc.image.ndim == 3 and doc.image.shape[2] == 1)):
        out = out[..., 0]

    # mask-aware blend (like dialog)
    out_masked = _blend_with_mask_float(out, src01, doc)

    meta = {
        "step_name": "ABE",
        "abe": {
            "degree": degree, "samples": num_samples, "downsample": downsample,
            "patch": patch_size, "rbf": use_rbf, "rbf_smooth": rbf_smooth,
            "exclusion": "none"  # explicit marker for audit/history
        },
        "masked": bool(getattr(doc, "active_mask_id", None)),
        "mask_id": getattr(doc, "active_mask_id", None),
        "mask_name": getattr(getattr(doc, "masks", {}), "get", lambda *_: None)(getattr(doc, "active_mask_id", None)),
        "mask_blend": "m*out + (1-m)*src",
    }

    step_name = f"ABE (deg={degree}, samples={num_samples}, ds={downsample}, patch={patch_size}, rbf={'on' if use_rbf else 'off'}, s={rbf_smooth:.3f})"
    doc.apply_edit(out_masked.astype(np.float32, copy=False), step_name=step_name, metadata=meta)

    if make_bg_doc and bg is not None:
        dm = getattr(main_window, "docman", None)
        if dm is not None:
            base = os.path.splitext(doc.display_name())[0]
            meta_bg = {
                "bit_depth": "32-bit floating point",
                "is_mono": (bg.ndim == 2),
                "source": "ABE background (headless)",
                "from_step": step_name,
            }
            doc_bg = dm.open_array(bg.astype(np.float32, copy=False), metadata=meta_bg, title=f"{base}_ABE_BG")
            if hasattr(main_window, "_spawn_subwindow_for"):
                main_window._spawn_subwindow_for(doc_bg)

# ---------- open dialog seeded from preset ----------
def open_abe_with_preset(main_window, preset: dict | None = None):
    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
    doc = dm.get_active_document() if (dm and hasattr(dm, "get_active_document")) else getattr(dm, "active_document", None)
    if doc is None:
        return
    from .abe import ABEDialog
    dlg = ABEDialog(main_window, doc)
    p = dict(preset or {})
    try:
        dlg.sp_degree.setValue(int(np.clip(p.get("degree", 2), 1, 6)))
        dlg.sp_samples.setValue(int(np.clip(p.get("samples", 120), 20, 100000)))
        dlg.sp_down.setValue(int(np.clip(p.get("downsample", 6), 1, 64)))
        dlg.sp_patch.setValue(int(np.clip(p.get("patch", 15), 5, 151)))
        dlg.chk_use_rbf.setChecked(bool(p.get("rbf", True)))
        dlg.sp_rbf.setValue(int(np.clip(float(p.get("rbf_smooth", 1.0)) * 100.0, 0, 100000)))
        # polygons intentionally untouched (== none)
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
