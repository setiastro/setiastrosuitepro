# pro/curves_preset.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np

# optional PCHIP (same as editor)
try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _HAS_PCHIP = True
except Exception:
    _HAS_PCHIP = False

from pro.curve_editor_pro import CurvesDialogPro, build_curve_lut, _apply_mode_any

# ---------------------- preset schema ----------------------
# {
#   "mode": "K (Brightness)" | "R" | "G" | "B" | "L*" | "a*" | "b*" | "Chroma" | "Saturation" | aliases ("rgb","k","lum"…),
#   "shape": "linear" | "s_mild" | "s_med" | "s_strong" | "lift_shadows" | "crush_shadows"
#            | "fade_blacks" | "rolloff_highlights" | "flatten" | "custom",
#   "amount": 0..1  (intensity, ignored for custom),
#   "points_norm": [[x,y], ...]  # optional when shape="custom" (normalized 0..1 domain/range)
# }
#
# Default if missing: mode="K (Brightness)", shape="linear", amount=0.5

# ---------------------- shape library (normalized) ----------------------
def _shape_points_norm(shape: str, amount: float) -> List[Tuple[float, float]]:
    a = float(max(0.0, min(1.0, amount)))
    s = shape.lower()

    # identity
    if s in ("linear", "none", "id"):
        return [(0.0, 0.0), (1.0, 1.0)]

    # simple parametric S-curves (mid anchored)
    if s in ("s", "s_mild", "s-curve-mild"):
        k = 0.15 * a
        return [(0.0, 0.0), (0.25, max(0.0, 0.25 - k)), (0.5, 0.5), (0.75, min(1.0, 0.75 + k)), (1.0, 1.0)]
    if s in ("s_med", "s-curve-med", "s-curve"):
        k = 0.25 * a
        return [(0.0, 0.0), (0.25, max(0.0, 0.25 - k)), (0.5, 0.5), (0.75, min(1.0, 0.75 + k)), (1.0, 1.0)]
    if s in ("s_strong", "s-curve-strong"):
        k = 0.36 * a
        return [(0.0, 0.0), (0.25, max(0.0, 0.25 - k)), (0.5, 0.5), (0.75, min(1.0, 0.75 + k)), (1.0, 1.0)]

    # lift/crush shadows, fade blacks, highlight roll-off, flatten
    if s == "lift_shadows":
        k = 0.35 * a
        return [(0.0, k), (0.3, k + 0.25*a), (1.0, 1.0)]
    if s == "crush_shadows":
        k = 0.35 * a
        return [(0.0, 0.0), (0.3, max(0.0, 0.3 - k)), (1.0, 1.0)]
    if s == "fade_blacks":
        k = 0.25 * a
        return [(0.0, k), (0.2, k*0.8), (0.6, 0.6 + 0.15*a), (1.0, 1.0)]
    if s == "rolloff_highlights":
        k = 0.35 * a
        return [(0.0, 0.0), (0.6, 0.6 + 0.15*a), (1.0, 1.0 - k)]
    if s == "flatten":
        k = 0.40 * a
        return [(0.0, 0.0 + 0.25*k), (0.25, 0.35), (0.5, 0.5), (0.75, 0.65), (1.0, 1.0 - 0.25*k)]

    # default
    return [(0.0, 0.0), (1.0, 1.0)]

def _norm_mode(m: str | None) -> str:
    m = (m or "K (Brightness)").strip().lower()
    alias = {
        "k": "K (Brightness)",
        "brightness": "K (Brightness)",
        "rgb": "K (Brightness)",
        "lum": "L*",
        "l": "L*",
        "lab_l": "L*",
        "lab_a": "a*",
        "lab_b": "b*",
        "chroma": "Chroma",
        "sat": "Saturation",
        "s": "Saturation",
        "r": "R", "g": "G", "b": "B",
    }
    # already proper label?
    proper = {"k (brightness)":"K (Brightness)","r":"R","g":"G","b":"B",
              "l*":"L*","a*":"a*","b*":"b*","chroma":"Chroma","saturation":"Saturation"}
    if m in proper: return proper[m]
    return alias.get(m, "K (Brightness)")

def _points_norm_to_scene(points_norm: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    normalized (x:[0..1], y:[0..1] up-is-positive) -> scene coords (x:[0..360], y:[0..360] down-is-positive)
    """
    pts = []
    for x, y in points_norm:
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))
        xs = 360.0 * x
        ys = 360.0 * (1.0 - y)   # invert Y for scene
        pts.append((xs, ys))
    # ensure endpoints exist
    if not any(abs(px - 0.0)   < 1e-6 for px, _ in pts): pts.append((0.0, 360.0))
    if not any(abs(px - 360.0) < 1e-6 for px, _ in pts): pts.append((360.0, 0.0))
    # x strictly increasing
    pts = sorted(pts, key=lambda t: t[0])
    out = []
    lastx = -1e9
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
    # fallback
    def _lin(x):
        return np.interp(x, xs, ys)
    return _lin

def _lut_from_preset(preset: Dict) -> tuple[np.ndarray, str]:
    shape  = (preset or {}).get("shape", "linear")
    amount = float((preset or {}).get("amount", 0.5))
    ptsN   = preset.get("points_norm")

    # decide control points (normalized)
    if isinstance(ptsN, (list, tuple)) and len(ptsN) >= 2:
        pts_norm = [(float(x), float(y)) for (x, y) in ptsN]
    else:
        pts_norm = _shape_points_norm(str(shape), amount)

    pts_scene = _points_norm_to_scene(pts_norm)
    fn = _interpolator_from_scene_points(pts_scene)
    lut01 = build_curve_lut(fn, size=65536)
    mode = _norm_mode(preset.get("mode"))
    return lut01, mode

# ---------------------- headless apply ----------------------
def apply_curves_via_preset(main_window, doc, preset: Dict):
    """
    Headless Curves apply (used when a Curves shortcut is dropped onto a view).
    """
    import numpy as _np
    img = getattr(doc, "image", None)
    if img is None:
        return
    arr = _np.asarray(img)
    # normalize → float01
    if arr.dtype.kind in "ui":
        arr01 = arr.astype(_np.float32) / _np.iinfo(arr.dtype).max
    elif arr.dtype.kind == "f":
        mx = float(arr.max()) if arr.size else 1.0
        arr01 = (arr / (mx if mx > 1.0 else 1.0)).astype(_np.float32)
    else:
        arr01 = arr.astype(_np.float32)

    lut01, mode = _lut_from_preset(preset or {})
    out01 = _apply_mode_any(arr01, mode, lut01)

    meta = {"step_name": "Curves", "mode": mode, "preset": dict(preset or {})}
    # commit
    doc.apply_edit(out01, metadata=meta, step_name="Curves")

# ---------------------- open UI with preset ----------------------
def open_curves_with_preset(main_window, preset: Dict | None = None):
    """
    Opens Curves dialog and seeds:
      - mode radio from preset["mode"]
      - curve handles from preset["points_norm"] or from shape/amount
    """
    # find active doc (reuse your pattern elsewhere)
    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
    if dm is None:
        return
    doc = dm.get_active_document() if hasattr(dm, "get_active_document") else getattr(dm, "active_document", None)
    if doc is None:
        return

    dlg = CurvesDialogPro(main_window, doc)

    # set mode radio
    want = _norm_mode((preset or {}).get("mode"))
    for b in dlg.mode_group.buttons():
        if b.text().lower() == want.lower():
            b.setChecked(True)
            break

    # seed control handles from preset
    shape  = (preset or {}).get("shape", "linear")
    amount = float((preset or {}).get("amount", 0.5))
    ptsN   = (preset or {}).get("points_norm")

    if isinstance(ptsN, (list, tuple)) and len(ptsN) >= 2:
        pts_norm = [(float(x), float(y)) for (x, y) in ptsN]
    else:
        pts_norm = _shape_points_norm(str(shape), amount)

    # convert to scene coords & strip endpoints (ShortcutButton.setControlHandles expects control handles only)
    pts_scene = _points_norm_to_scene(pts_norm)
    # remove exact endpoints if present
    filt = [(x, y) for (x, y) in pts_scene if x > 0.0 + 1e-6 and x < 360.0 - 1e-6]
    dlg.editor.setControlHandles(filt)

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
