# pro/curves_preset.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import json
from PyQt6.QtCore import QSettings
# optional PCHIP (same as editor)
try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _HAS_PCHIP = True
except Exception:
    _HAS_PCHIP = False



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

def build_curve_lut(curve_func, size=65536):
    """Map v∈[0..1] → y∈[0..1] using a curve defined on x∈[0..360] (scene coords)."""
    x = np.linspace(0.0, 360.0, size, dtype=np.float32)
    y = 360.0 - curve_func(x)
    y = (y / 360.0).clip(0.0, 1.0).astype(np.float32)
    return y

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
    """
    Build LUT from any compatible preset / last-action dict.
    """
    pts_scene = _scene_points_from_preset(preset or {})
    fn = _interpolator_from_scene_points(pts_scene)
    lut01 = build_curve_lut(fn, size=65536)
    mode = _norm_mode((preset or {}).get("mode"))
    return lut01, mode

def _unwrap_preset_dict(preset: Dict) -> Dict:
    """
    Accept a variety of containers and peel down to the actual curve data.

    Examples we handle:
      {"step_name":"Curves","mode":..., "preset":{...}}
      {"curves": {...}}
      {"state": {...}}   # if state contains curve points
    """
    p = dict(preset or {})

    # Case 1: full metadata from doc.apply_edit: {"step_name":"Curves", "mode":..., "preset": {...}}
    inner = p.get("preset")
    if isinstance(inner, dict) and ("points_norm" in inner or "handles" in inner
                                    or "points_scene" in inner or "scene_points" in inner):
        return inner

    # Case 2: payloads like {"curves": {...}}
    inner = p.get("curves")
    if isinstance(inner, dict) and ("points_norm" in inner or "handles" in inner
                                    or "points_scene" in inner or "scene_points" in inner):
        return inner

    # Case 3: {"state": {...}} (if you stored the curve state under that key)
    inner = p.get("state")
    if isinstance(inner, dict) and ("points_norm" in inner or "handles" in inner
                                    or "points_scene" in inner or "scene_points" in inner):
        return inner

    # Otherwise assume p already *is* the preset
    return p


_SETTINGS_KEY = "curves/custom_presets_v1"

def _settings() -> QSettings | None:
    try:
        return QSettings()
    except Exception:
        return None

def list_custom_presets() -> list[dict]:
    """Return a list of dicts: {"name", "mode", "shape":"custom", "points_norm":[[x,y],...]}"""
    s = _settings()
    if not s:
        return []
    raw = s.value(_SETTINGS_KEY, "", type=str) or ""
    try:
        lst = json.loads(raw)
        if isinstance(lst, list):
            return [p for p in lst if isinstance(p, dict)]
    except Exception:
        pass
    return []

def save_custom_preset(name: str, mode: str, points_norm: list[tuple[float,float]]) -> bool:
    """Create/overwrite by name."""
    s = _settings()
    if not s:
        return False
    name = (name or "").strip()
    if not name:
        return False
    preset = {
        "name": name,
        "mode": _norm_mode(mode),
        "shape": "custom",
        "amount": 1.0,
        "points_norm": [(float(x), float(y)) for (x, y) in points_norm],
    }
    lst = list_custom_presets()
    lst = [p for p in lst if (p.get("name","").lower() != name.lower())]
    lst.append(preset)
    s.setValue(_SETTINGS_KEY, json.dumps(lst))
    s.sync()
    return True

def delete_custom_preset(name: str) -> bool:
    s = _settings()
    if not s:
        return False
    lst = list_custom_presets()
    lst = [p for p in lst if (p.get("name","").lower() != (name or "").strip().lower())]
    s.setValue(_SETTINGS_KEY, json.dumps(lst))
    s.sync()
    return True


# ---------------------- headless apply ----------------------
def apply_curves_via_preset(main_window, doc, preset: Dict):
    import numpy as _np
    from setiastro.saspro.curves_preset import _lut_from_preset, _unwrap_preset_dict  # self
    # lazy import to avoid cycle
    from setiastro.saspro.curve_editor_pro import _apply_mode_any

    img = getattr(doc, "image", None)
    if img is None:
        return

    # Accept full last-action dicts and unwrap down to the actual curve definition
    core_preset = _unwrap_preset_dict(preset or {})

    arr = _np.asarray(img)
    if arr.dtype.kind in "ui":
        arr01 = arr.astype(_np.float32) / _np.iinfo(arr.dtype).max
    elif arr.dtype.kind == "f":
        mx = float(arr.max()) if arr.size else 1.0
        arr01 = (arr / (mx if mx > 1.0 else 1.0)).astype(_np.float32)
    else:
        arr01 = arr.astype(_np.float32)

    lut01, mode = _lut_from_preset(core_preset)
    out01 = _apply_mode_any(arr01, mode, lut01)

    meta = {
        "step_name": "Curves",
        "mode": mode,
        "preset": dict(core_preset),  # store the normalized core preset
    }
    doc.apply_edit(out01, metadata=meta, step_name="Curves")


# ---------------------- open UI with preset ----------------------
# ---------------------- open UI with preset ----------------------
def open_curves_with_preset(main_window, preset: Dict | None = None):
    # lazy import UI to avoid cycle
    from setiastro.saspro.curve_editor_pro import CurvesDialogPro

    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
    if dm is None:
        return
    doc = dm.get_active_document() if hasattr(dm, "get_active_document") else getattr(dm, "active_document", None)
    if doc is None:
        return

    dlg = CurvesDialogPro(main_window, doc)

    # Peel down any wrapper (metadata / last-action container) to the actual curve definition
    core_preset = _unwrap_preset_dict(preset or {})

    # set mode radio from the *core* preset
    want = _norm_mode(core_preset.get("mode"))
    for b in dlg.mode_group.buttons():
        if b.text().lower() == want.lower():
            b.setChecked(True)
            break

    # Seed control handles from the same logic used by LUT building
    pts_scene = _scene_points_from_preset(core_preset)

    # remove exact endpoints if present; editor expects control handles only
    filt = [(x, y) for (x, y) in pts_scene if x > 0.0 + 1e-6 and x < 360.0 - 1e-6]
    dlg.editor.setControlHandles(filt)

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()


def _sanitize_scene_points(points_scene: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Take scene-space points (x,y in [0..360]) and:
      - clamp to [0..360]
      - ensure endpoints at x=0 and x=360 exist
      - enforce strictly increasing x
    """
    pts = []
    for x, y in points_scene:
        xs = float(x)
        ys = float(y)
        xs = min(360.0, max(0.0, xs))
        ys = min(360.0, max(0.0, ys))
        pts.append((xs, ys))

    # ensure endpoints exist
    if not any(abs(px - 0.0) < 1e-6 for px, _ in pts):
        pts.append((0.0, 360.0))
    if not any(abs(px - 360.0) < 1e-6 for px, _ in pts):
        pts.append((360.0, 0.0))

    # x strictly increasing
    pts = sorted(pts, key=lambda t: t[0])
    out: List[Tuple[float, float]] = []
    lastx = -1e9
    for x, y in pts:
        if x <= lastx:
            x = lastx + 1e-3
        out.append((min(360.0, max(0.0, x)), min(360.0, max(0.0, y))))
        lastx = out[-1][0]
    return out


def _scene_points_from_preset(preset: Dict) -> List[Tuple[float, float]]:
    """
    Accepts any of:
      - preset["points_scene"] / preset["scene_points"]
         → list of [x,y] or {"x":..,"y":..} in scene coords
      - preset["handles"] / preset["control_points"]
         → same as above, in scene coords (what the editor stores)
      - preset["points_norm"]
         → normalized [0..1] points
      - otherwise falls back to shape/amount library
    Returns a sanitized list of scene-space points.
    """
    p = preset or {}

    # 1) explicit scene coords from several possible keys
    for key in ("points_scene", "scene_points", "handles", "control_points"):
        raw = p.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            pts: List[Tuple[float, float]] = []
            for entry in raw:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    x, y = entry[0], entry[1]
                elif isinstance(entry, dict):
                    x, y = entry.get("x"), entry.get("y")
                else:
                    continue
                if x is None or y is None:
                    continue
                pts.append((float(x), float(y)))
            if pts:
                return _sanitize_scene_points(pts)

    # 2) normalized points (what we already support)
    shape  = str(p.get("shape", "linear"))
    amount = float(p.get("amount", 0.5))
    ptsN   = p.get("points_norm")

    if isinstance(ptsN, (list, tuple)) and len(ptsN) >= 2:
        pts_norm = [(float(x), float(y)) for (x, y) in ptsN]
    else:
        pts_norm = _shape_points_norm(shape, amount)

    return _points_norm_to_scene(pts_norm)

