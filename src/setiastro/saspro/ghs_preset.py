# pro/ghs_preset.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# Apply engine from the Curves editor
from setiastro.saspro.curve_editor_pro import _apply_mode_any


def _dlg():
    """
    Deferred import of the dialog module. Every stretch LUT lives there — one
    analytic implementation per function, shared by the dialog and this
    headless path, so replay and interactive apply cannot diverge.

    Kept deferred to preserve the existing ghs_dialog_pro <-> shortcuts <->
    openers cycle avoidance.
    """
    from setiastro.saspro import ghs_dialog_pro as _m
    return _m


def _resolve_mode(fn: str, D) -> str:
    """
    Stored 'function' string -> a current MODE_* constant.

    Exact match first, then the dialog's legacy aliases, then a substring
    rescue for anything hand-edited. The substring pass checks GHS explicitly:
    the old code had no GHS branch at all, so a Cranfield/Payne preset silently
    fell through to the Universal curve.
    """
    canon = D.canonical_function(str(fn or ""))
    known = (D.MODE_UHS, D.MODE_GHS, D.MODE_ARCSINH,
             D.MODE_LOG, D.MODE_EXP, D.MODE_PIP)
    if canon in known:
        return canon

    low = str(fn or "").lower()
    if "arcsinh" in low:
        return D.MODE_ARCSINH
    if "logarith" in low:
        return D.MODE_LOG
    if "expon" in low:
        return D.MODE_EXP
    if "pip" in low or "inverted" in low:
        return D.MODE_PIP
    if any(k in low for k in ("cranfield", "payne", "generalis", "generaliz")):
        return D.MODE_GHS
    return D.MODE_UHS


def _short_label(mode: str, D) -> str:
    """
    Compact name for history steps. Without this the new mode produces
    "Hyperbolic Stretch (Generalised Hyperbolic Stretch (Cranfield/Payne))"
    in the undo stack.
    """
    return {
        D.MODE_UHS:     "Universal",
        D.MODE_GHS:     "GHS \u2014 Cranfield/Payne",
        D.MODE_ARCSINH: "ArcSinh",
        D.MODE_LOG:     "Logarithmic",
        D.MODE_EXP:     "Exponential",
        D.MODE_PIP:     "Power of Inverted Pixels",
    }.get(mode, mode)


# ---------- helpers: points -> scene -> interpolator -> LUT ----------

def _norm_channel(ch: str | None) -> str:
    m = (ch or "K (Brightness)").strip().lower()
    alias = {"k":"K (Brightness)","brightness":"K (Brightness)","rgb":"K (Brightness)",
             "r":"R","g":"G","b":"B"}
    proper = {"k (brightness)":"K (Brightness)","r":"R","g":"G","b":"B"}
    if m in proper: return proper[m]
    return alias.get(m, "K (Brightness)")


# ---------- GHS parameterization → normalized control curve ----------


def _lut_from_ghs_preset(preset: Dict) -> tuple[np.ndarray, str, Dict]:
    m = _dlg()
    D = m.GhsDialogPro
    p = dict(preset or {})

    mode  = _resolve_mode(p.get("function", ""), D)
    ch    = _norm_channel(p.get("channel", "K (Brightness)"))
    gamma = float(p.get("gamma", 1.0))
    lp    = float(p.get("lp",    0.0))
    hp    = float(p.get("hp",    0.0))
    pivot = float(p.get("pivot", 0.5))

    # 'function' is written back canonicalised, so a preset saved before the
    # rename gets upgraded in place the first time it replays.
    base = {"function": mode, "channel": ch,
            "gamma": gamma, "lp": lp, "hp": hp, "pivot": pivot}

    if mode == D.MODE_GHS:
        ghs_D  = float(p.get("ghs_D",  0.0))
        ghs_b  = float(p.get("ghs_b",  0.0))
        ghs_LP = float(p.get("ghs_LP", 0.0))
        ghs_HP = float(p.get("ghs_HP", 1.0))
        ghs_BP = float(p.get("ghs_BP", 0.0))
        lut01 = m._build_cranfield_lut(ghs_D, ghs_b, pivot,
                                       LP=ghs_LP, HP=ghs_HP, BP=ghs_BP, g=gamma)
        params = {**base, "ghs_D": ghs_D, "ghs_b": ghs_b,
                  "ghs_LP": ghs_LP, "ghs_HP": ghs_HP, "ghs_BP": ghs_BP}

    elif mode == D.MODE_ARCSINH:
        strength = float(p.get("strength", 5.0))
        lut01 = m._build_arcsinh_lut(strength, gamma, lp, hp, pivot)
        params = {**base, "strength": strength}

    elif mode == D.MODE_LOG:
        strength = float(p.get("strength", 5.0))
        lut01 = m._build_log_lut(strength, gamma, lp, hp, pivot)
        params = {**base, "strength": strength}

    elif mode == D.MODE_EXP:
        strength = float(p.get("strength", 5.0))
        lut01 = m._build_exp_lut(strength, gamma, lp, hp, pivot)
        params = {**base, "strength": strength}

    elif mode == D.MODE_PIP:
        strength = float(p.get("strength", 1.0))
        lut01 = m._build_pip_lut(strength, gamma, lp, hp, pivot)
        params = {**base, "strength": strength}

    else:  # D.MODE_UHS
        alpha = float(p.get("alpha", 1.0))
        beta  = float(p.get("beta",  1.0))
        # Same analytic builder the dialog uses — no spline round-trip, so
        # headless replay is bit-identical to an interactive apply.
        lut01 = m._build_uhs_lut(alpha, beta, gamma, lp, hp, pivot)
        params = {**base, "alpha": alpha, "beta": beta}

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
    img = getattr(doc, "image", None)
    if img is None:
        return

    arr = np.asarray(img)
    if arr.dtype.kind in "ui":
        src01 = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    elif arr.dtype.kind == "f":
        mx = float(arr.max()) if arr.size else 1.0
        src01 = (arr / (mx if mx > 1.0 else 1.0)).astype(np.float32)
    else:
        src01 = arr.astype(np.float32)

    lut01, channel, params = _lut_from_ghs_preset(preset or {})
    out01 = _apply_mode_any(src01, channel, lut01)

    try:
        remember = getattr(main_window, "remember_last_headless_command", None)
        if remember is None:
            remember = getattr(main_window, "_remember_last_headless_command", None)
        label = _short_label(params.get("function", ""), _dlg().GhsDialogPro)
        if callable(remember):
            remember("ghs", params, description=f"Hyperbolic Stretch ({label})")
            try:
                if hasattr(main_window, "_log"):
                    main_window._log(f"[Replay] GHS headless stored: fn={params.get('function')}, "
                                     f"keys={list(params.keys())}")
            except Exception:
                pass
    except Exception:
        pass

    mask, mid, mname = _active_mask_layer(doc)
    if mask is not None:
        out01 = _blend_with_mask(out01, src01, doc)

    step = f"Hyperbolic Stretch ({_short_label(params.get('function', ''), _dlg().GhsDialogPro)})"
    meta = {
        "step_name": step,
        "ghs": params,
        "masked": bool(mid),
        "mask_id": mid,
        "mask_name": mname,
        "mask_blend": "m*out + (1-m)*src",
    }

    doc.apply_edit(
        out01.astype(np.float32, copy=False),
        metadata=meta,
        step_name=step,
    )


# ---------- open dialog seeded from preset ----------

def open_ghs_with_preset(main_window, preset: Dict | None = None):
    dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
    doc = dm.get_active_document() if (dm and hasattr(dm, "get_active_document")) else getattr(dm, "active_document", None)
    if doc is None:
        return

    m = _dlg()
    D = m.GhsDialogPro
    dlg = D(main_window, doc)

    p = dict(preset or {})
    mode = _resolve_mode(p.get("function", ""), D)

    try:
        idx = dlg.cmb_fn.findText(mode)
        dlg.cmb_fn.setCurrentIndex(idx if idx >= 0 else 0)

        if mode == D.MODE_UHS:
            dlg.sA.setValue(int(np.clip(float(p.get("alpha", 1.0)) * 50.0, 1, 500)))
            dlg.sB.setValue(int(np.clip(float(p.get("beta",  1.0)) * 50.0, 1, 500)))

        elif mode == D.MODE_GHS:
            dlg.sD.setValue(  int(np.clip(float(p.get("ghs_D",  0.0)) * 100.0,      0,  1000)))
            dlg.sBb.setValue( int(np.clip(float(p.get("ghs_b",  0.0)) * 100.0,   -500,  1500)))
            dlg.sgLP.setValue(int(np.clip(float(p.get("ghs_LP", 0.0)) * 10000.0,    0, 10000)))
            dlg.sgHP.setValue(int(np.clip(float(p.get("ghs_HP", 1.0)) * 10000.0,    0, 10000)))
            dlg.sBP.setValue( int(np.clip(float(p.get("ghs_BP", 0.0)) * 10000.0,    0, 10000)))

        elif mode == D.MODE_PIP:
            dlg.sP.setValue(int(np.clip(float(p.get("strength", 1.0)) * 100.0, 0, 200)))

        else:  # ArcSinh / Log / Exp
            dlg.sS.setValue(int(np.clip(float(p.get("strength", 5.0)) * 10.0, 1, 1000)))

        # Shared
        dlg.sG.setValue(int(np.clip(float(p.get("gamma", 1.0)) * 100.0, 1, 500)))
        dlg.sLP.setValue(int(np.clip(float(p.get("lp", 0.0)) * 360.0, 0, 360)))
        dlg.sHP.setValue(int(np.clip(float(p.get("hp", 0.0)) * 360.0, 0, 360)))

        ch = _norm_channel(p.get("channel", "K (Brightness)"))
        i = dlg.cmb_ch.findText(ch)
        dlg.cmb_ch.setCurrentIndex(i if i >= 0 else 0)

        pv = float(np.clip(p.get("pivot", 0.5), 0.0, 1.0))
        dlg._set_sym_u(pv)
        dlg.editor.setSymmetryPoint(pv * 360.0, 0)
        dlg._rebuild_from_params()
    except Exception as e:
        # A bare pass here means a renamed slider attribute produces a dialog
        # that opens with default values and no indication anything failed.
        try:
            if hasattr(main_window, "_log"):
                main_window._log(f"[GHS] preset seeding failed ({mode}): {e}")
            else:
                print("[GHS] preset seeding failed:", e)
        except Exception:
            pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
