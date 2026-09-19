"""
gradient_abe.py -- GUI-free ABE (poly2 automatic background extraction).

Lifted verbatim from stacking_suite.py so it can run inside ProcessPoolExecutor
children WITHOUT importing PyQt / stacking_suite (which would make Windows spawn
slow and fragile). Import-light: numpy / cv2 / astropy / numba_utils only.

Includes the NaN-safety + speed fixes:
  - sample points and the poly2 fit ignore satellite-trail NaN (the NaN is the
    exclusion mask); the background model stays finite so ch-bg preserves NaN;
  - strength stats on the small model and re-center medians on a subsample
    instead of full-res sorts;
  - no QApplication.processEvents() (thread/process-unsafe).

If you change the ABE math here, there is no in-stacking_suite copy to mirror --
stacking_suite imports these names from this module.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from setiastro.saspro.legacy.numba_utils import gradient_descent_to_dim_spot_numba

try:
    import inspect as _inspect
    _ASARRAY_HAS_COPY = "copy" in _inspect.signature(np.asarray).parameters
except Exception:
    _ASARRAY_HAS_COPY = False


def _asarray(x, dtype=None, copy=False):
    if _ASARRAY_HAS_COPY:
        return np.asarray(x, dtype=dtype, copy=copy)
    a = np.asarray(x, dtype=dtype)
    return a.copy() if copy else a


def _force_shape_hw(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Return img with exactly (target_h, target_w) spatial shape.
    If bigger → center-crop; if smaller → reflect-pad. Preserves channels/layout.
    """
    import numpy as np
    import cv2
    a = np.asarray(img)
    if a.ndim < 2:
        return a

    if a.ndim == 2:
        H, W = a.shape
        Caxis = None
    elif a.ndim == 3:
        # supports HWC and CHW; detect which one by comparing small channel count
        if a.shape[-1] in (1, 3):      # HWC
            H, W = a.shape[:2]; Caxis = -1
        elif a.shape[0] in (1, 3):     # CHW
            H, W = a.shape[1:]; Caxis = 0
        else:
            # assume HWC
            H, W = a.shape[:2]; Caxis = -1
    else:
        return a

    th, tw = int(target_h), int(target_w)
    if (H, W) == (th, tw): 
        return a

    # --- center-crop if larger ---
    y0 = max(0, (H - th) // 2)
    x0 = max(0, (W - tw) // 2)
    y1 = min(H, y0 + th)
    x1 = min(W, x0 + tw)

    if a.ndim == 2:
        cropped = a[y0:y1, x0:x1]
    elif Caxis == -1:  # HWC
        cropped = a[y0:y1, x0:x1, ...]
    else:               # CHW
        cropped = a[:, y0:y1, x0:x1]

    ch, cw = (cropped.shape[:2] if (cropped.ndim == 2 or Caxis == -1) else cropped.shape[1:3])

    # --- reflect-pad if smaller ---
    pad_t = max(0, (th - ch) // 2)
    pad_b = max(0, th - ch - pad_t)
    pad_l = max(0, (tw - cw) // 2)
    pad_r = max(0, tw - cw - pad_l)

    if pad_t or pad_b or pad_l or pad_r:
        if cropped.ndim == 2:
            cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                         borderType=cv2.BORDER_REFLECT_101)
        elif Caxis == -1:
            # HWC: pad spatial, keep channels intact
            cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                         borderType=cv2.BORDER_REFLECT_101)
        else:
            # CHW: pad each channel slice
            chans = []
            for c in range(cropped.shape[0]):
                chans.append(cv2.copyMakeBorder(cropped[c], pad_t, pad_b, pad_l, pad_r,
                                                borderType=cv2.BORDER_REFLECT_101))
            cropped = np.stack(chans, axis=0)

    return cropped


def _downsample_area(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Robust area downsample by an integer scale (>=1).
    - Prefers exact block-mean pooling (no OpenCV) when full blocks fit.
    - Falls back to cv2.resize with a clamped, non-zero dsize.
    - Final fallback uses stride slicing (nearest-like) so this never throws.
    """
    if img is None:
        return None

    # Normalize inputs
    scale = int(max(1, scale))
    a = np.asarray(img, dtype=np.float32)
    if a.ndim < 2 or a.size == 0:
        return a

    H, W = int(a.shape[0]), int(a.shape[1])
    if H <= 0 or W <= 0 or scale == 1:
        return a

    # ---- Prefer exact block mean when we have whole blocks ----
    Hs = (H // scale) * scale
    Ws = (W // scale) * scale
    if Hs >= scale and Ws >= scale:
        a_c = np.ascontiguousarray(a[:Hs, :Ws, ...])  # ensure contiguous for reshape
        if a.ndim == 2:
            out = a_c.reshape(Hs // scale, scale, Ws // scale, scale).mean(axis=(1, 3))
            return out.astype(np.float32, copy=False)
        else:
            C = a.shape[2]
            out = a_c.reshape(Hs // scale, scale, Ws // scale, scale, C).mean(axis=(1, 3))
            return out.astype(np.float32, copy=False)

    # ---- Fallback to OpenCV with explicit, clamped dsize ----
    tw = max(1, W // scale)
    th = max(1, H // scale)
    if tw == W and th == H:
        return a  # nothing to do

    try:
        import cv2
        a_c = np.ascontiguousarray(a)  # OpenCV likes contiguous
        out = cv2.resize(a_c, (int(tw), int(th)), interpolation=cv2.INTER_AREA)
        return out.astype(np.float32, copy=False)
    except Exception:
        # Last resort: stride slicing (nearest-ish), always returns something
        if a.ndim == 2:
            return a[::scale, ::scale].astype(np.float32, copy=False)
        else:
            return a[::scale, ::scale, :].astype(np.float32, copy=False)



def _upscale_bg(bg_small: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """
    Robust upscale of the background model to (oh, ow). Never passes 0 sizes to cv2.
    """
    oh = int(max(1, oh)); ow = int(max(1, ow))
    if bg_small is None:
        return np.zeros((oh, ow), dtype=np.float32)

    b = np.asarray(bg_small, dtype=np.float32)
    if b.ndim < 2 or b.shape[0] == 0 or b.shape[1] == 0:
        return np.zeros((oh, ow), dtype=np.float32)

    try:
        import cv2
        return cv2.resize(b, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32, copy=False)
    except Exception:
        # Safe pure-numpy nearest-neighbor fallback
        y_idx = (np.linspace(0, b.shape[0]-1, oh)).astype(np.int32)
        x_idx = (np.linspace(0, b.shape[1]-1, ow)).astype(np.int32)
        return b[y_idx][:, x_idx].astype(np.float32, copy=False)


def _to_Luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32, copy=False)
    # HWC RGB
    if img.shape[2] == 3:
        try:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32, copy=False)
        except Exception:
            pass # fallback
    r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def _build_poly_terms_deg2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # [1, x, y, x^2, x*y, y^2]
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=1).astype(np.float32, copy=False)

# ----- ABE-like sampling (corners, borders, quartiles with bright-avoid & descent) -----

def _exclude_bright_regions(gray_small: np.ndarray, exclusion_fraction: float = 0.5) -> np.ndarray:
    """
    Returns a boolean mask selecting the dimmer ~exclusion_fraction of pixels.
    Robust to empty arrays and NaNs — never throws.
    """
    a = np.asarray(gray_small, dtype=np.float32)
    if a.size == 0:
        return np.zeros_like(a, dtype=bool)

    # Use nanpercentile; if everything is NaN, allow all (avoid empty elig later)
    q = max(0.0, min(100.0, 100.0 * (1.0 - float(exclusion_fraction))))
    try:
        thresh = np.nanpercentile(a, q)
    except Exception:
        # Fallback: if something went wrong, just allow all
        return np.ones_like(a, dtype=bool)

    mask = a < thresh
    # If mask ends up all False (flat image, or numeric quirks), allow all
    if not np.any(mask):
        mask = np.ones_like(a, dtype=bool)
    return mask


def _gradient_descent_to_dim_spot(gray_small: np.ndarray, x: int, y: int, patch: int) -> tuple[int, int]:
    # Delegate to Numba optimized version
    return gradient_descent_to_dim_spot_numba(gray_small, int(x), int(y), int(patch))

def _generate_sample_points_small(
    img_small: np.ndarray,
    num_samples: int,
    patch_size: int,
    exclusion_mask_small: np.ndarray | None,
    force_border_px: int = 20,          # ← NEW: inset in downsampled-image pixels
    force_border_count: int = 5,        # ← NEW: points per edge (left/right/top/bottom)
    use_forced_anchors: bool = True,    # ← NEW: enable/disable the forced points
) -> np.ndarray:
    """
    Generate background sample points for poly2 gradient fitting.
 
    Changes vs original:
    - Added forced corner + border anchor points that bypass brightness
      exclusion and gradient-descent relocation entirely.
    - This ensures the polynomial fit is constrained at the image edges
      even when those edges are artificially bright (e.g. overcorrected
      flat vignetting), which previously caused sample points to flee
      toward the centre and produce a badly extrapolated background model.
 
    force_border_px  : inset from each edge in small-image pixels.
                       Caller should pass (desired_full_px / downsample).
    force_border_count: number of evenly-spaced points along each of the
                        4 edges (corners are added separately, so total
                        forced = 4 corners + 4 × force_border_count).
    use_forced_anchors: set False to restore legacy behaviour exactly.
    """
    H, W = img_small.shape[:2]
    gray = _to_Luma(img_small) if img_small.ndim == 3 else img_small
    pts: list[tuple[int, int]] = []
 
    # ── If the downsampled image is too small, fall back to minimal grid ──
    if H < max(3, patch_size * 2 + 1) or W < max(3, patch_size * 2 + 1):
        g = max(2, min(4, int(np.sqrt(max(1, num_samples)))))
        xs = np.linspace(0, max(W - 1, 0), g, dtype=int)
        ys = np.linspace(0, max(H - 1, 0), g, dtype=int)
        for y in ys:
            for x in xs:
                pts.append((int(x), int(y)))
        return np.asarray(pts, dtype=np.int32)
 
    border   = max(6, patch_size)
    fb       = max(1, int(force_border_px))   # forced inset, clamped sane
 
    def allowed(x, y):
        if exclusion_mask_small is None:
            return True
        yy = int(np.clip(y, 0, H - 1))
        xx = int(np.clip(x, 0, W - 1))
        return bool(exclusion_mask_small[yy, xx])
 
    # ══════════════════════════════════════════════════════════════════
    # FORCED ANCHOR POINTS  (new)
    # These are placed at fixed positions regardless of pixel brightness.
    # They anchor the polynomial at the image boundary so the model
    # cannot drift away from overcorrected/undercorrected edges.
    # ══════════════════════════════════════════════════════════════════
    if use_forced_anchors:
        forced: list[tuple[int, int]] = []
 
        # --- 4 corners ---
        cx = min(fb, W - 1)
        cy = min(fb, H - 1)
        forced += [
            (cx,         cy),          # top-left
            (W - 1 - cx, cy),          # top-right
            (cx,         H - 1 - cy),  # bottom-left
            (W - 1 - cx, H - 1 - cy), # bottom-right
        ]
 
        # --- points along each of the 4 borders ---
        n = max(1, int(force_border_count))
 
        # top edge
        for x in np.linspace(cx, W - 1 - cx, n + 2, dtype=int)[1:-1]:
            forced.append((int(x), cy))
        # bottom edge
        for x in np.linspace(cx, W - 1 - cx, n + 2, dtype=int)[1:-1]:
            forced.append((int(x), H - 1 - cy))
        # left edge
        for y in np.linspace(cy, H - 1 - cy, n + 2, dtype=int)[1:-1]:
            forced.append((cx, int(y)))
        # right edge
        for y in np.linspace(cy, H - 1 - cy, n + 2, dtype=int)[1:-1]:
            forced.append((W - 1 - cx, int(y)))
 
        # Clamp and deduplicate, then append
        seen_forced = set()
        for (x, y) in forced:
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))
            if (x, y) not in seen_forced:
                seen_forced.add((x, y))
                pts.append((x, y))
 
    # ══════════════════════════════════════════════════════════════════
    # ORIGINAL adaptive sample points (unchanged logic below)
    # ══════════════════════════════════════════════════════════════════
 
    # corners (brightness-avoiding, gradient-descending)
    for (x, y) in [
        (border,         border),
        (W - border - 1, border),
        (border,         H - border - 1),
        (W - border - 1, H - border - 1),
    ]:
        if 0 <= x < W and 0 <= y < H and allowed(x, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, y, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
 
    # border sweeps
    xs = np.linspace(border, max(W - border - 1, border), 5, dtype=int)
    ys = np.linspace(border, max(H - border - 1, border), 5, dtype=int)
    xs = np.unique(xs)
    ys = np.unique(ys)
 
    for x in xs:
        if 0 <= x < W and 0 <= border < H and allowed(x, border):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, border, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
        if 0 <= x < W and 0 <= (H - border - 1) < H and allowed(x, H - border - 1):
            nx, ny = _gradient_descent_to_dim_spot(gray, x, H - border - 1, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
    for y in ys:
        if 0 <= border < W and 0 <= y < H and allowed(border, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, border, y, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
        if 0 <= (W - border - 1) < W and 0 <= y < H and allowed(W - border - 1, y):
            nx, ny = _gradient_descent_to_dim_spot(gray, W - border - 1, y, patch_size)
            if allowed(nx, ny):
                pts.append((nx, ny))
 
    # quartile interior samples
    hh, ww = H // 2, W // 2
    quads = [
        (slice(0, hh),   slice(0, ww),   (0,  0)),
        (slice(0, hh),   slice(ww, W),   (ww, 0)),
        (slice(hh, H),   slice(0, ww),   (0,  hh)),
        (slice(hh, H),   slice(ww, W),   (ww, hh)),
    ]
    per_quad = max(1, num_samples // 4)
 
    for ysl, xsl, (x0, y0) in quads:
        sub = gray[ysl, xsl]
        if sub.size == 0:
            continue
        mask_sub = _exclude_bright_regions(sub, exclusion_fraction=0.5)
        if exclusion_mask_small is not None:
            em = exclusion_mask_small[ysl, xsl]
            if em.size == mask_sub.size:
                mask_sub = mask_sub & em
        elig = np.argwhere(mask_sub)
        if elig.size == 0:
            continue
        k = min(len(elig), per_quad)
        sel = elig[np.random.choice(len(elig), k, replace=False)]
        for (yy, xx) in sel:
            gx, gy = x0 + int(xx), y0 + int(yy)
            if 0 <= gx < W and 0 <= gy < H and allowed(gx, gy):
                nx, ny = _gradient_descent_to_dim_spot(gray, gx, gy, patch_size)
                if allowed(nx, ny):
                    pts.append((nx, ny))
 
    # Absolute fallback: small grid if pts is still somehow empty
    if not pts:
        g = max(3, int(np.sqrt(max(16, num_samples))))
        xs = np.linspace(border, W - border - 1, g, dtype=int)
        ys = np.linspace(border, H - border - 1, g, dtype=int)
        for y in ys:
            for x in xs:
                if 0 <= x < W and 0 <= y < H and allowed(x, y):
                    pts.append((int(x), int(y)))
 
    return np.asarray(pts, dtype=np.int32)
 

# ----- fit/eval on small image -----

def _fit_poly2_on_small(img_small: np.ndarray, pts_small: np.ndarray, patch_size: int) -> np.ndarray:
    """Fit degree-2 polynomial on Luma of small image using patch medians at pts."""
    gray = _to_Luma(img_small) if img_small.ndim == 3 else img_small
    Hs, Ws = gray.shape[:2]
    half = patch_size // 2

    xs = np.clip(pts_small[:, 0], 0, Ws-1).astype(np.int32)
    ys = np.clip(pts_small[:, 1], 0, Hs-1).astype(np.int32)

    z = np.empty(xs.shape[0], dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x0, x1 = max(0, x - half), min(Ws, x + half + 1)
        y0, y1 = max(0, y - half), min(Hs, y + half + 1)
        _patch = gray[y0:y1, x0:x1]
        _fin = _patch[np.isfinite(_patch)]
        z[i] = float(np.median(_fin)) if _fin.size else np.nan

    # Drop all-NaN samples (forced anchors bypass the sampler's finite check),
    # else one NaN makes the whole polynomial NaN and poisons the frame.
    _good = np.isfinite(z)
    xs, ys, z = xs[_good], ys[_good], z[_good]
    if z.size < 6:
        _fa = gray[np.isfinite(gray)]
        return np.full((Hs, Ws), float(np.median(_fa)) if _fa.size else 0.0, dtype=np.float32)

    A = _build_poly_terms_deg2(xs.astype(np.float32), ys.astype(np.float32))
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)

    # evaluate on full small grid
    yy, xx = np.meshgrid(np.arange(Hs, dtype=np.float32),
                         np.arange(Ws, dtype=np.float32), indexing='ij')
    bg_small = (coef[0] + coef[1]*xx + coef[2]*yy + coef[3]*xx*xx + coef[4]*xx*yy + coef[5]*yy*yy).astype(np.float32)
    return bg_small

# ----- public API -----

def remove_poly2_gradient_abe(
    image: np.ndarray,
    *,
    mode: str = "subtract",
    num_samples: int = 120,
    downsample: int = 6,
    patch_size: int = 15,
    min_strength: float = 0.01,
    gain_clip: tuple[float,float] = (0.2, 5.0),
    exclusion_mask: np.ndarray | None = None,
    log_fn=None
) -> np.ndarray:
    if image is None:
        return image

    img = _asarray(image, dtype=np.float32)

    # ---- Detect original layout
    is_2d  = (img.ndim == 2)
    is_hwc = (img.ndim == 3 and img.shape[-1] in (1, 3))
    is_chw = (img.ndim == 3 and img.shape[0]  in (1, 3) and not is_hwc)

    # ---- Convert to HWC "work" view (internal processing)
    if is_2d:
        work = img
    elif is_hwc:
        work = img
    elif is_chw:
        work = np.moveaxis(img, 0, -1)  # CHW -> HWC
    else:
        # Unexpected layout; treat as 2D luma via mean over last axis
        work = img.mean(axis=-1).astype(np.float32, copy=False)

    H, W = work.shape[:2]

    # --- Downsample image & optional mask
    img_small = _downsample_area(work, max(1, int(downsample)))
    # Satellite-trail no-data arrives as NaN (the NaN *is* the exclusion mask
    # now). Build a finite 'keep' mask so samples never land on a trail.
    _lum_small = _to_Luma(img_small) if img_small.ndim == 3 else img_small
    mask_small = np.isfinite(_lum_small)
    if exclusion_mask is not None:
        em  = _asarray(exclusion_mask, dtype=np.float32)
        mask_small &= (_downsample_area(em, max(1, int(downsample))) >= 0.5)

    # --- Sample & fit
    ds = max(1, int(downsample))
    pts_small = _generate_sample_points_small(
        img_small,
        num_samples=int(num_samples),
        patch_size=int(patch_size),
        exclusion_mask_small=mask_small,
        force_border_px=max(1, 20 // ds),   # 20 full-res px → small-image coords
        force_border_count=5,               # 5 pts per edge
        use_forced_anchors=True,
    )
    bg_small = _fit_poly2_on_small(img_small, pts_small, patch_size=int(patch_size))
    bg = _upscale_bg(bg_small, H, W)

    # --- Strength check on the SMALL model (bg is a smooth poly2 upscale of
    #     bg_small, so stats match but cost ~60x less than a full-res sort).
    _bs = np.asarray(bg_small, dtype=np.float32).ravel()
    _bs = _bs[np.isfinite(_bs)]
    if _bs.size:
        bg_med = float(np.median(_bs)) or 1e-6
        p5, p95 = float(np.percentile(_bs, 5)), float(np.percentile(_bs, 95))
    else:
        bg_med, p5, p95 = 1e-6, 0.0, 0.0
    rel_amp = float((p95 - p5) / max(bg_med, 1e-6))
    if log_fn:
        log_fn(f"ABE poly2: samples={num_samples}, ds={downsample}, patch={patch_size} | "
               f"bg_med={bg_med:.6f}, rel_amp={rel_amp*100:.2f}%")
    if rel_amp < float(min_strength):
        # Return original image in original layout
        return img

    # --- Apply (luma-only fit, channel-consistent apply)
    def _bg_med_sub(a):  # robust center on a strided subsample (16x cheaper)
        s = a[::4, ::4]; s = s[np.isfinite(s)]
        return (float(np.median(s)) if s.size else 0.0) or 1e-6

    def _apply_sub(ch):  # re-center to preserve median
        med0 = _bg_med_sub(ch)
        out = ch - bg
        med1 = _bg_med_sub(out)
        out += (med0 - med1)
        return out

    def _apply_div(ch):
        med0 = _bg_med_sub(ch)
        norm_bg = np.clip(bg / bg_med, gain_clip[0], gain_clip[1])
        out = ch / norm_bg
        med1 = _bg_med_sub(out)
        out *= (med0 / med1)
        return out

    if work.ndim == 2:
        ch = work
        out_work = _apply_sub(ch) if mode.lower() == "subtract" else _apply_div(ch)
    else:
        # HWC
        if mode.lower() == "subtract":
            r = _apply_sub(work[..., 0])
            g = _apply_sub(work[..., 1]) if work.shape[-1] > 1 else r
            b = _apply_sub(work[..., 2]) if work.shape[-1] > 2 else r
        else:
            r = _apply_div(work[..., 0])
            g = _apply_div(work[..., 1]) if work.shape[-1] > 1 else r
            b = _apply_div(work[..., 2]) if work.shape[-1] > 2 else r
        out_work = np.stack([r, g, b], axis=-1) if work.shape[-1] == 3 else r[..., None]

    out_work = out_work.astype(np.float32, copy=False)

    # ---- Convert back to original layout
    if is_2d:
        return out_work
    if is_hwc:
        return out_work
    if is_chw:
        return np.moveaxis(out_work, -1, 0).astype(np.float32, copy=False)
    # Fallback: shape-preserving best effort
    return out_work


def _ensure_like(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Return x with the same layout/shape as like (2D, HWC, or CHW)."""
    if x.shape == like.shape:
        return x

    # 2D cases
    if like.ndim == 2:
        if x.ndim == 3 and x.shape[-1] in (1,3):  # HWC -> 2D (luma)
            return (0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]).astype(np.float32, copy=False) if x.shape[-1]==3 else x[...,0]
        if x.ndim == 3 and x.shape[0] in (1,3):   # CHW -> 2D (first/mean)
            return x[0]
        return x

    # HWC target
    if like.ndim == 3 and like.shape[-1] in (1,3):
        if x.ndim == 3 and x.shape[0] in (1,3):   # CHW -> HWC
            return np.moveaxis(x, 0, -1).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[-1] == 3:   # 2D -> HWC repeat
            return np.repeat(x[..., None], 3, axis=-1).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[-1] == 1:
            return x[..., None].astype(np.float32, copy=False)
        return x

    # CHW target
    if like.ndim == 3 and like.shape[0] in (1,3):
        if x.ndim == 3 and x.shape[-1] in (1,3):  # HWC -> CHW
            return np.moveaxis(x, -1, 0).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[0] == 3:    # 2D -> CHW repeat
            return np.repeat(x[None, ...], 3, axis=0).astype(np.float32, copy=False)
        if x.ndim == 2 and like.shape[0] == 1:
            return x[None, ...].astype(np.float32, copy=False)
        return x

    return x

def remove_gradient_stack_abe(stack, target_hw: tuple[int,int] | None = None, **kw):
    """
    stack: (N,H,W) or (N,H,W,C) or (N,C,H,W)
    Returns the same layout as 'stack'. If target_hw=(H,W) is provided,
    every output is forced to exactly (H,W) via center-crop / reflect-pad.
    """

    # ---- local helper: force exact (H,W) via center-crop or reflect-pad ----
    def _force_shape_hw(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        import numpy as np
        import cv2
        a = np.asarray(img)
        if a.ndim < 2:
            return a

        # detect layout
        if a.ndim == 2:
            H, W = a.shape; Caxis = None
        elif a.ndim == 3:
            if a.shape[-1] in (1, 3):      # HWC
                H, W = a.shape[:2]; Caxis = -1
            elif a.shape[0] in (1, 3):     # CHW
                H, W = a.shape[1:]; Caxis = 0
            else:
                H, W = a.shape[:2]; Caxis = -1
        else:
            return a

        th, tw = int(target_h), int(target_w)
        if (H, W) == (th, tw):
            return a

        # center-crop if larger
        y0 = max(0, (H - th) // 2); x0 = max(0, (W - tw) // 2)
        y1 = min(H, y0 + th);       x1 = min(W, x0 + tw)

        if a.ndim == 2:
            cropped = a[y0:y1, x0:x1]
        elif Caxis == -1:
            cropped = a[y0:y1, x0:x1, ...]
        else:
            cropped = a[:, y0:y1, x0:x1]

        ch, cw = (cropped.shape[:2] if (cropped.ndim == 2 or Caxis == -1) else cropped.shape[1:3])

        # reflect-pad if smaller
        pad_t = max(0, (th - ch) // 2)
        pad_b = max(0, th - ch - pad_t)
        pad_l = max(0, (tw - cw) // 2)
        pad_r = max(0, tw - cw - pad_l)

        if pad_t or pad_b or pad_l or pad_r:
            if cropped.ndim == 2:
                cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                             borderType=cv2.BORDER_REFLECT_101)
            elif Caxis == -1:
                cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                             borderType=cv2.BORDER_REFLECT_101)
            else:
                chans = []
                for c in range(cropped.shape[0]):
                    chans.append(cv2.copyMakeBorder(cropped[c], pad_t, pad_b, pad_l, pad_r,
                                                    borderType=cv2.BORDER_REFLECT_101))
                cropped = np.stack(chans, axis=0)

        return cropped

    arr = _asarray(stack, dtype=np.float32)
    N = arr.shape[0]
    out = np.empty_like(arr)

    # Use first frame as layout reference (channels/order), not necessarily size
    ref = arr[0]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {}
        for i in range(N):
            img = arr[i]
            futs[ex.submit(remove_poly2_gradient_abe, img, **kw)] = i

        for fut in as_completed(futs):
            i_ = futs[fut]
            res = fut.result()
            res = _ensure_like(res, ref)   # normalize layout (2D/HWC/CHW)

            # 🔒 size lock: prefer caller-provided canonical size
            if target_hw is not None:
                th, tw = int(target_hw[0]), int(target_hw[1])
                res = _force_shape_hw(res, th, tw)
            else:
                # legacy behavior: conform to ref's current size
                if res.shape != ref.shape:
                    if ref.ndim == 2:
                        res = _force_shape_hw(res, ref.shape[0], ref.shape[1])
                    elif ref.ndim == 3 and ref.shape[-1] in (1, 3):   # HWC
                        res_h, res_w = res.shape[:2]
                        res = _force_shape_hw(res, ref.shape[0], ref.shape[1])
                    elif ref.ndim == 3 and ref.shape[0] in (1, 3):     # CHW
                        res_h, res_w = res.shape[1:3]
                        res = _force_shape_hw(res, ref.shape[1], ref.shape[2])
                    # else: leave as-is (best effort)

            out[i_] = res.astype(out.dtype, copy=False)

    return out

# ============================================================================
# File-based process-pool ABE. Each worker reads a normalized frame from disk,
# runs poly2 ABE, and writes it back in place -- no GIL, and no big-array IPC
# (only file paths cross the process boundary), so it scales flat across cores
# and overlaps disk I/O. Mirrors the stacking_measure_worker / finalize design.
# ============================================================================

def _abe_pick_hdu(hlist):
    for x in hlist:
        if getattr(x, "data", None) is not None and x.data.ndim >= 2:
            return x
    return None


def abe_one_file(job):
    """Process-pool worker (top-level & picklable). job = (path, kw, target_hw)."""
    from astropy.io import fits
    path, kw, target_hw = job
    try:
        with fits.open(path, do_not_scale_image_data=True, memmap=False) as h:
            hdu = _abe_pick_hdu(h)
            if hdu is None:
                return ("skip", path, "no image HDU")
            data = np.asarray(hdu.data, dtype=np.float32)
            hdr = hdu.header.copy()
        out = remove_poly2_gradient_abe(data, **(kw or {}))
        out = np.asarray(out, dtype=np.float32)
        if target_hw is not None:
            out = _force_shape_hw(out, int(target_hw[0]), int(target_hw[1]))
        # float32, header preserved, NaN (satellite no-data) written through.
        fits.PrimaryHDU(data=out, header=hdr).writeto(path, overwrite=True)
        return ("ok", path, None)
    except Exception as e:
        return ("err", path, f"{type(e).__name__}: {e}")


def run_abe_files_parallel(paths, *, kw=None, target_hw=None,
                           max_workers=None, log_fn=None):
    """Run poly2 ABE over `paths` (in place) across a process pool.

    Returns (n_ok, n_skip, n_err). GUI-free: pass a plain callable as log_fn if
    you want progress; it is called from the *main* process only.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as _mp

    paths = [p for p in (paths or []) if p]
    if not paths:
        return (0, 0, 0)

    if max_workers is None:
        try:
            import psutil
            _phys = int(psutil.cpu_count(logical=False) or (os.cpu_count() or 4))
        except Exception:
            _phys = int(os.cpu_count() or 4)
        max_workers = max(1, min(len(paths), _phys - 2))

    jobs = [(p, dict(kw or {}), target_hw) for p in paths]

    try:
        _ctx = _mp.get_context("spawn")   # Qt-safe + matches frozen builds
        ex = ProcessPoolExecutor(max_workers=int(max_workers), mp_context=_ctx)
    except Exception:
        ex = ProcessPoolExecutor(max_workers=int(max_workers))

    n_ok = n_skip = n_err = 0
    try:
        futs = {ex.submit(abe_one_file, j): j[0] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            status, p, err = fut.result()
            done += 1
            base = os.path.basename(p)
            if status == "ok":
                n_ok += 1
                if log_fn:
                    log_fn(f"\U0001F308 ABE {done}/{len(jobs)}: {base}")
            elif status == "skip":
                n_skip += 1
                if log_fn:
                    log_fn(f"\u26A0\uFE0F ABE skip {base}: {err}")
            else:
                n_err += 1
                if log_fn:
                    log_fn(f"\u26A0\uFE0F ABE error {base}: {err}")
    finally:
        ex.shutdown(wait=True)
    return (n_ok, n_skip, n_err)