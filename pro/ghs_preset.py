# pro/ghs_preset.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# Reuse LUT + apply engine from Curves editor (with a safe fallback import)
try:
    from pro.curves_editor_pro import build_curve_lut, _apply_mode_any
except Exception:
    from pro.curve_editor_pro import build_curve_lut, _apply_mode_any  # if your file name lacks the 's'

# Optional PCHIP interpolation (nice-to-have; we’ll fall back to np.interp)
try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _HAS_PCHIP = True
except Exception:
    _HAS_PCHIP = False


# ---------- helpers: points -> scene -> interpolator -> LUT ----------

def _points_norm_to_scene(points_norm: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(x:[0..1], y:[0..1 up]) → scene coords (x:[0..360], y:[0..360] down)."""
    pts = []
    for x, y in points_norm:
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))
        xs = 360.0 * x
        ys = 360.0 * (1.0 - y)
        pts.append((xs, ys))
    # Ensure strict endpoints exist
    if not any(abs(px - 0.0) < 1e-6 for px, _ in pts):   pts.append((0.0, 360.0))
    if not any(abs(px - 360.0) < 1e-6 for px, _ in pts): pts.append((360.0, 0.0))
    pts = sorted(pts, key=lambda t: t[0])
    # Make X strictly increasing
    out, lastx = [], -1e9
    for x, y in pts:
        if x <= lastx: x = lastx + 1e-3
        out.append((min(360.0, max(0.0, x)), min(360.0, max(0.0, y))))
        lastx = out[-1][0]
    return out

def _interpolator_from_scene_points(points_scene: List[Tuple[float, float]]):
    xs = np.array([p[0] for p in points_scene], dtype=np.float64)
    ys = np.array([p[1] for p in points_scene], dtype=np.float64)
    if _HAS_PCHIP and xs.size >= 2:
        return _PCHIP(xs, ys)
    def _lin(x):
        return np.interp(x, xs, ys)
    return _lin

def _norm_channel(ch: str | None) -> str:
    m = (ch or "K (Brightness)").strip().lower()
    alias = {"k":"K (Brightness)","brightness":"K (Brightness)","rgb":"K (Brightness)",
             "r":"R","g":"G","b":"B"}
    proper = {"k (brightness)":"K (Brightness)","r":"R","g":"G","b":"B"}
    if m in proper: return proper[m]
    return alias.get(m, "K (Brightness)")


# ---------- GHS parameterization → normalized control curve ----------

def _ghs_points_norm(alpha: float, beta: float, gamma: float,
                     pivot: float, lp: float, hp: float,
                     N: int = 128) -> List[Tuple[float, float]]:
    """
    Returns a monotone set of (x,y) in [0..1] implementing your dialog’s math.
    """
    a = float(np.clip(alpha, 0.02, 10.0))     # sliders 1..500 → /50 gives 0.02..10
    b = float(np.clip(beta,  0.02, 10.0))
    g = float(np.clip(gamma, 0.01,  5.0))     # slider 1..500 → /100 gives 0.01..5
    SP = float(np.clip(pivot, 0.0, 1.0))
    LP = float(np.clip(lp,    0.0, 1.0))
    HP = float(np.clip(hp,    0.0, 1.0))

    us = np.linspace(0.0, 1.0, int(max(16, N)))
    left  = us <= 0.5
    right = ~left

    # Generalized hyperbolic halves around 0.5
    rawL = us**a / (us**a + b*(1.0-us)**a)
    rawR = us**a / (us**a + (1.0/b)*(1.0-us)**a)

    midL = (0.5**a) / (0.5**a + b*(0.5)**a)
    midR = (0.5**a) / (0.5**a + (1.0/b)*(0.5)**a)
    eps = 1e-6

    # Domain remap to pivot SP
    up = np.empty_like(us)
    vp = np.empty_like(us)

    # Left → [0..SP]
    up[left] = 2.0 * SP * us[left]
    vp[left] = rawL[left] * (SP / max(midL, eps))

    # Right → [SP..1]
    up[right] = SP + 2.0*(1.0 - SP)*(us[right] - 0.5)
    vp[right] = SP + (rawR[right] - midR) * ((1.0 - SP) / max(1.0 - midR, eps))

    # LP/HP protection (blend toward identity y=x on each side)
    if LP > 0:
        m = up <= SP
        vp[m] = (1.0 - LP)*vp[m] + LP*up[m]
    if HP > 0:
        m = up >= SP
        vp[m] = (1.0 - HP)*vp[m] + HP*up[m]

    # Gamma lift
    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    # Clamp + enforce monotonicity (guards against tiny numerical dips)
    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)

    pts = list(zip(up.tolist(), vp.tolist()))
    # Make sure endpoints exist exactly
    if pts[0][0] != 0.0:      pts.insert(0, (0.0, 0.0))
    if pts[-1][0] != 1.0:     pts.append((1.0, 1.0))
    return pts


def _lut_from_ghs_preset(preset: Dict) -> tuple[np.ndarray, str, Dict]:
    p = dict(preset or {})
    alpha  = float(p.get("alpha", 1.0))
    beta   = float(p.get("beta",  1.0))
    gamma  = float(p.get("gamma", 1.0))
    pivot  = float(p.get("pivot", 0.5))
    lp     = float(p.get("lp",    0.0))
    hp     = float(p.get("hp",    0.0))
    ch     = _norm_channel(p.get("channel", "K (Brightness)"))

    ptsN   = _ghs_points_norm(alpha, beta, gamma, pivot, lp, hp, N=128)
    ptsS   = _points_norm_to_scene(ptsN)
    fn     = _interpolator_from_scene_points(ptsS)
    lut01  = build_curve_lut(fn, size=65536)

    # sanitized params for metadata
    params = {"alpha":alpha, "beta":beta, "gamma":gamma, "pivot":pivot, "lp":lp, "hp":hp, "channel":ch}
    return lut01, ch, params


# ---------- optional: mask-aware blend (same semantics as your dialogs) ----------

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

def _blend_with_mask(processed: np.ndarray, src: np.ndarray, doc) -> np.ndarray:
    mask, _, _ = _active_mask_layer(doc)
    if mask is None:
        return processed
    out = processed.astype(np.float32, copy=False)
    m = _resample_mask(mask, out.shape[:2])
    if out.ndim == 3 and out.shape[2] == 3:
        m = m[..., None]
    s = src
    if s.ndim == 2 and out.ndim == 3:
        s = np.stack([s]*3, axis=-1)
    elif s.ndim == 3 and out.ndim == 2:
        s = s[..., 0]
    return (m * out + (1.0 - m) * s).astype(np.float32, copy=False)


# ---------- headless apply ----------

def apply_ghs_via_preset(main_window, doc, preset: Dict):
    """
    Headless Universal Hyperbolic Stretch:
      - builds the curve from α/β/γ + pivot + LP/HP
      - applies to K/R/G/B channel
      - blends with active mask if any
      - commits to document with metadata
    """
    img = getattr(doc, "image", None)
    if img is None:
        return
    arr = np.asarray(img)
    # normalize to float01
    if arr.dtype.kind in "ui":
        src01 = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    elif arr.dtype.kind == "f":
        mx = float(arr.max()) if arr.size else 1.0
        src01 = (arr / (mx if mx > 1.0 else 1.0)).astype(np.float32)
    else:
        src01 = arr.astype(np.float32)

    lut01, channel, params = _lut_from_ghs_preset(preset)
    out01 = _apply_mode_any(src01, channel, lut01)
    out01 = _blend_with_mask(out01, src01, doc)

    meta = {"step_name": "Hyperbolic Stretch", "ghs": params}
    doc.apply_edit(out01, metadata=meta, step_name="Hyperbolic Stretch")


# ---------- open dialog seeded from preset ----------

def open_ghs_with_preset(main_window, preset: Dict | None = None):
    # find active document
    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
    doc = dm.get_active_document() if (dm and hasattr(dm, "get_active_document")) else getattr(dm, "active_document", None)
    if doc is None:
        return

    from pro.ghs_dialog_pro import GhsDialogPro
    dlg = GhsDialogPro(main_window, doc)

    p = dict(preset or {})
    # sliders use integer storage: α: *50, β:*50, γ:*100; LP/HP:*360
    try:
        dlg.sA.setValue(int(np.clip(float(p.get("alpha", 1.0)) * 50.0, 1, 500)))
        dlg.sB.setValue(int(np.clip(float(p.get("beta",  1.0)) * 50.0, 1, 500)))
        dlg.sG.setValue(int(np.clip(float(p.get("gamma", 1.0)) * 100.0, 1, 500)))
        dlg.sLP.setValue(int(np.clip(float(p.get("lp",    0.0)) * 360.0, 0, 360)))
        dlg.sHP.setValue(int(np.clip(float(p.get("hp",    0.0)) * 360.0, 0, 360)))
        ch = _norm_channel(p.get("channel", "K (Brightness)"))
        i = dlg.cmb_ch.findText(ch); dlg.cmb_ch.setCurrentIndex(i if i >= 0 else 0)
        pv = float(np.clip(p.get("pivot", 0.5), 0.0, 1.0))
        dlg._sym_u = pv
        dlg.editor.setSymmetryPoint(pv * 360.0, 0)
        dlg._rebuild_from_params()
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
