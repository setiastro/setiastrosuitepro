# pro/crop_preset.py
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import cv2

from setiastro.saspro.legacy.image_manager import load_image, save_image  # not used here, but matches other preset files
from setiastro.saspro.wcs_update import update_wcs_after_crop

def _as_float01(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype.kind in "ui":
        arr = arr.astype(np.float32) / np.iinfo(img.dtype).max
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return np.clip(arr, 0.0, 1.0)

def _quad_from_rect_norm(w: int, h: int, rect: dict) -> np.ndarray:
    """
    rect = {"x":float,"y":float,"w":float,"h":float,"angle_deg":float}
    All values in [0..1] except angle (deg). (x,y) is top-left of the rect before rotation.
    Rotation is around the rect center, positive CCW.
    """
    x = float(rect.get("x", 0.0))
    y = float(rect.get("y", 0.0))
    rw = float(rect.get("w", 1.0))
    rh = float(rect.get("h", 1.0))
    ang = float(rect.get("angle_deg", 0.0))
    # pixel-space axis-aligned box
    px = x * w
    py = y * h
    pw = max(1.0, rw * w)
    ph = max(1.0, rh * h)

    # corners (TL, TR, BR, BL) before rotation
    cx = px + pw * 0.5
    cy = py + ph * 0.5
    pts = np.array([
        [px,       py      ],  # TL
        [px + pw,  py      ],  # TR
        [px + pw,  py + ph ],  # BR
        [px,       py + ph ],  # BL
    ], dtype=np.float32)

    if abs(ang) < 1e-6:
        return pts

    rad = math.radians(ang)
    s, c = math.sin(rad), math.cos(rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts_c = pts - np.array([cx, cy], dtype=np.float32)
    pts_r = (R @ pts_c.T).T + np.array([cx, cy], dtype=np.float32)
    return pts_r.astype(np.float32)

def _dst_size_from_quad(q: np.ndarray) -> tuple[int, int]:
    """
    q: 4x2 TL, TR, BR, BL in pixels.
    """
    q = np.asarray(q, np.float32)
    w_top  = np.linalg.norm(q[1] - q[0])
    w_bot  = np.linalg.norm(q[2] - q[3])
    h_left = np.linalg.norm(q[3] - q[0])
    h_right= np.linalg.norm(q[2] - q[1])
    w_out = int(round(max(w_top, w_bot)))
    h_out = int(round(max(h_left, h_right)))
    w_out = max(1, w_out); h_out = max(1, h_out)
    return w_out, h_out

def _quad_from_margins(w: int, h: int, margins: dict) -> np.ndarray:
    t = max(0, int(margins.get("top", 0)))
    r = max(0, int(margins.get("right", 0)))
    b = max(0, int(margins.get("bottom", 0)))
    l = max(0, int(margins.get("left", 0)))
    # clamp
    t = min(t, h); b = min(b, h); l = min(l, w); r = min(r, w)
    x0, y0 = float(l), float(t)
    x1, y1 = float(w - r), float(h - b)
    # TL, TR, BR, BL
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

def apply_crop_via_preset(mw, doc, preset: dict) -> np.ndarray:
    """
    Headless crop runner.

    Preset schema (choose ONE of the three inputs):
      1) Margins (pixels):
         {"mode":"margins",
          "margins":{"top":0,"right":0,"bottom":0,"left":0},
          "create_new_view": false}

      2) Axis-aligned rect in normalized coords + optional rotation (deg):
         {"mode":"rect_norm",
          "rect":{"x":0.05,"y":0.08,"w":0.85,"h":0.80,"angle_deg": 0.0},
          "create_new_view": false}

      3) Explicit quad in normalized coords (TL,TR,BR,BL):
         {"mode":"quad_norm",
          "quad":[[xTL,yTL],[xTR,yTR],[xBR,yBR],[xBL,yBL]],
          "create_new_view": false}

    Common options:
      - "create_new_view": bool (default False → overwrite target doc)
      - "title": optional title if creating a new view

    Returns the cropped float32 image in [0..1].
    """
    pr = dict(preset or {})
    mode = str(pr.get("mode", "margins")).lower()
    create_new = bool(pr.get("create_new_view", False))
    title = pr.get("title") or "Crop"

    img01 = _as_float01(getattr(doc, "image"))
    h, w = img01.shape[:2]

    if mode == "margins":
        quad = _quad_from_margins(w, h, pr.get("margins", {}) or {})
    elif mode == "rect_norm":
        quad = _quad_from_rect_norm(w, h, pr.get("rect", {}) or {})
    elif mode == "quad_norm":
        qn = np.array(pr.get("quad", []), dtype=np.float32)
        if qn.shape != (4, 2):
            raise ValueError("quad_norm expects 4×2 list of [x,y] in [0..1].")
        quad = qn * np.array([w, h], dtype=np.float32)
    else:
        raise ValueError(f"Unknown crop mode: {mode}")

    # Destination geometry (axis-aligned)
    w_out, h_out = _dst_size_from_quad(quad)
    dst = np.array([[0, 0], [w_out, 0], [w_out, h_out], [0, h_out]], dtype=np.float32)

    # Homography and crop
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    out = cv2.warpPerspective(img01, M, (w_out, h_out), flags=cv2.INTER_LINEAR)

    # Build metadata / WCS
    meta = dict(getattr(doc, "metadata", {}) or {})
    try:
        if update_wcs_after_crop is not None:
            meta = update_wcs_after_crop(meta, M_src_to_dst=M, out_w=w_out, out_h=h_out)
    except Exception:
        pass

    # Apply to document or open new
    if create_new:
        dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        if dm is None:
            raise RuntimeError("Document manager unavailable for create_new_view=True.")
        newdoc = dm.open_array(out.copy(), metadata={**meta, "step_name": "Crop"}, title=title)
        if hasattr(mw, "_spawn_subwindow_for"):
            mw._spawn_subwindow_for(newdoc)
    else:
        # overwrite
        if hasattr(doc, "apply_edit"):
            doc.apply_edit(out.copy(), metadata={**meta, "step_name": "Crop"}, step_name="Crop")
        else:
            # fallback if apply_edit not present
            setattr(doc, "image", out.copy())
            setattr(doc, "metadata", {**meta, "step_name": "Crop"})
            if hasattr(doc, "changed"):
                try: doc.changed.emit()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    return out

def run_crop_via_preset(mw, preset: dict, target_doc=None) -> bool:
    """
    Convenience for non-drag invocations (similar to star_alignment_via_preset).
    """
    # resolve active document if not provided
    doc = target_doc
    if doc is None:
        try:
            if hasattr(mw, "_active_doc"):
                doc = mw._active_doc()
        except Exception:
            doc = None
        if doc is None and hasattr(mw, "mdi"):
            asw = mw.mdi.activeSubWindow()
            if asw:
                doc = getattr(asw.widget(), "document", None)
    if doc is None:
        raise RuntimeError("No active document for crop.")
    apply_crop_via_preset(mw, doc, preset or {})
    return True
