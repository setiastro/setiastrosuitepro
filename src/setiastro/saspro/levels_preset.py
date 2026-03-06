# src/setiastro/saspro/levels_preset.py
from __future__ import annotations

import numpy as np


def _ensure_float01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    # If your docs can store HDR >1, normalize here if desired; otherwise keep [0..1]
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def apply_levels_via_preset(ctx, doc, preset: dict):
    """
    Headless Levels runner used by:
      - command drops (shortcuts/function bundles)
      - replay last action on base

    preset keys:
      black: float [0..1]
      mid: float [0..1]
      white: float [0..1]
      channel: "L"|"R"|"G"|"B"
      step_name: optional override
    """
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("No target document image to apply Levels to.")

    p = dict(preset or {})

    black = float(p.get("black", 0.0))
    mid   = float(p.get("mid", 0.5))
    white = float(p.get("white", 1.0))
    channel = str(p.get("channel", "L") or "L").upper().strip()
    step_name = str(p.get("step_name", "Levels") or "Levels")

    # sanitize
    if white <= black + 1e-8:
        white = min(1.0, black + 1e-8)
    mid = float(np.clip(mid, 0.0, 1.0))

    from setiastro.saspro.histogram_transform_pro import apply_histogram_transform_channel

    img = _ensure_float01(doc.image)
    out = apply_histogram_transform_channel(img, black, mid, white, channel)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # Apply to doc (follow your doc API conventions)
    if hasattr(doc, "apply_edit"):
        doc.apply_edit(out, metadata={"step_name": step_name}, step_name=step_name)
    elif hasattr(doc, "set_image"):
        doc.set_image(out, step_name=step_name)
    else:
        doc.image = out

    # ✅ Track replay payload (matches your system)
    try:
        if ctx is not None and hasattr(ctx, "_last_headless_command"):
            ctx._last_headless_command = {
                "command_id": "levels",
                "preset": {
                    "black": black,
                    "mid": mid,
                    "white": white,
                    "channel": channel,
                    "step_name": step_name,
                }
            }
    except Exception:
        pass

    return out