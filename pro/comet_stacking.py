# pro/comet_stacking.py
from __future__ import annotations
import os, sys, tempfile, subprocess, shutil, math
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import sep
from pro.remove_stars import (
    _get_setting_any,
    _mtf_params_linked, _apply_mtf_linked_rgb, _invert_mtf_linked_rgb,
    _resolve_darkstar_exe, _ensure_exec_bit, _purge_darkstar_io,
    load_image as _rs_load_image, save_image as _rs_save_image  # reuse I/O
)
from legacy.image_manager import load_image, save_image

def _blackpoint_nonzero(img_norm: np.ndarray, p: float = 0.1) -> float:
    """Scalar blackpoint from non-zero pixels across all channels (linked).
       p in [0..100]: small percentile to resist outliers; use 0 for strict min."""
    x = img_norm
    if x.ndim == 3 and x.shape[2] == 3:
        nz = np.any(x > 0.0, axis=2)     # keep pixels where any channel has signal
        vals = x[nz]                      # shape (N,3) → flatten to scalar pool
    else:
        vals = x[x > 0.0]
    if vals.size == 0:
        return float(np.min(x))           # fallback (all zeros?)
    if p <= 0.0:
        return float(np.min(vals))
    return float(np.percentile(vals, p))

def starnet_starless_from_array(src_rgb01: np.ndarray, settings, *, is_linear: bool, **_ignored) -> np.ndarray:
    exe = _get_setting_any(settings, ("starnet/exe_path", "paths/starnet"), "")
    if not exe or not os.path.exists(exe):
        raise RuntimeError("StarNet executable path is not configured.")
    _ensure_exec_bit(exe)

    img = src_rgb01.astype(np.float32, copy=False)
    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1: img = np.repeat(img, 3, axis=2)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    did_stretch = False
    mtf = None
    scale_factor = float(np.max(img))

    if is_linear:
        did_stretch = True
        img_norm = img / scale_factor if scale_factor > 1.0 else img

        # --- normalize pedestal from data (ignore borders/zeros) ---
        bp_norm = _blackpoint_nonzero(img_norm, p=0.1)   # use 0.0 for strict min
        img_zerod = img_norm - bp_norm
        np.maximum(img_zerod, 0.0, out=img_zerod)

        # linked MTF exactly as before, but on zeroed data
        mtf = _mtf_params_linked(img_zerod, shadowclip_sigma=-2.8, targetbg=0.25)
        proc_in = _apply_mtf_linked_rgb(img_zerod, mtf)
    else:
        proc_in = img
        bp_norm = 0.0

    starnet_dir = os.path.dirname(exe) or os.getcwd()
    in_path  = os.path.join(starnet_dir, "imagetoremovestars.tif")
    out_path = os.path.join(starnet_dir, "starless.tif")
    _rs_save_image(proc_in, in_path, "tif", "16-bit", None, False, None, None)

    exe_name = os.path.basename(exe).lower()
    if os.name == "nt" or sys.platform.startswith(("linux","linux2")):
        cmd = [exe, in_path, out_path, "256"]
    else:
        cmd = [exe, "--input", in_path, "--output", out_path] if "starnet2" in exe_name else [exe, in_path, out_path]

    rc = subprocess.call(cmd, cwd=starnet_dir)
    if rc != 0 or not os.path.exists(out_path):
        try: os.remove(in_path)
        except Exception: pass
        raise RuntimeError(f"StarNet failed (rc={rc}).")

    starless, _, _, _ = _rs_load_image(out_path)
    try: os.remove(in_path)
    except Exception: pass
    try: os.remove(out_path)
    except Exception: pass

    if starless is None:
        raise RuntimeError("StarNet produced no output.")
    if starless.ndim == 2: starless = np.stack([starless]*3, axis=-1)
    if starless.ndim == 3 and starless.shape[2] == 1: starless = np.repeat(starless, 3, axis=2)
    starless = starless.astype(np.float32, copy=False)

    if did_stretch:
        # inverse MTF, then restore pedestal + original scale
        starless = _invert_mtf_linked_rgb(starless, mtf)
        starless = starless + bp_norm
        if scale_factor > 1.0:
            starless = starless * scale_factor

    return np.clip(starless, 0.0, 1.0)




def darkstar_starless_from_array(src_rgb01: np.ndarray, settings, **_ignored) -> np.ndarray:
    """
    Headless CosmicClarity DarkStar run for a single RGB frame.
    Returns starless RGB in [0..1]. Uses CC’s input/output folders.
    """
    # normalize channels
    img = src_rgb01.astype(np.float32, copy=False)
    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1: img = np.repeat(img, 3, axis=2)

    # resolve exe and base folder
    exe, base = _resolve_darkstar_exe(type("Dummy", (), {"settings": settings})())
    if not exe or not base:
        raise RuntimeError("Cosmic Clarity DarkStar executable path is not set.")

    _ensure_exec_bit(exe)

    input_dir  = os.path.join(base, "input")
    output_dir = os.path.join(base, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # purge any prior files (safe; scoped to imagetoremovestars*)
    _purge_darkstar_io(base, prefix="imagetoremovestars", clear_input=True, clear_output=True)

    in_path  = os.path.join(input_dir, "imagetoremovestars.tif")
    out_path = os.path.join(output_dir, "imagetoremovestars_starless.tif")

    # save input as float32 TIFF
    _rs_save_image(img, in_path, original_format="tif", bit_depth="32-bit floating point",
                   original_header=None, is_mono=False, image_meta=None, file_meta=None)

    # build command (SASv2 parity): default unscreen, show extracted stars off, stride 512
    cmd = [exe, "--star_removal_mode", "unscreen", "--chunk_size", "512"]

    rc = subprocess.call(cmd, cwd=output_dir)
    if rc != 0 or not os.path.exists(out_path):
        try: os.remove(in_path)
        except Exception: pass
        raise RuntimeError(f"DarkStar failed (rc={rc}).")

    starless, _, _, _ = _rs_load_image(out_path)
    # cleanup
    try:
        os.remove(in_path)
        os.remove(out_path)
        _purge_darkstar_io(base, prefix="imagetoremovestars", clear_input=True, clear_output=True)
    except Exception:
        pass

    if starless is None:
        raise RuntimeError("DarkStar produced no output.")

    if starless.ndim == 2: starless = np.stack([starless]*3, axis=-1)
    if starless.shape[2] == 1: starless = np.repeat(starless, 3, axis=2)
    return np.clip(starless.astype(np.float32, copy=False), 0.0, 1.0)

# ---------- small helpers ----------
def _inv_affine_2x3(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix [[a,b,tx],[c,d,ty]] → [[a',b',tx'],[c',d',ty']]."""
    A = np.asarray(M, dtype=np.float64).reshape(2,3)
    a,b,tx = A[0]; c,d,ty = A[1]
    det = a*d - b*c
    if abs(det) < 1e-12:
        raise ValueError("Affine matrix not invertible")
    inv = np.array([[ d, -b, 0.0],
                    [-c,  a, 0.0]], dtype=np.float64) / det
    # new translation = - inv * t
    inv[:,2] = -inv[:,:2] @ np.array([tx, ty], dtype=np.float64)
    return inv.astype(np.float32)

def _to_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img.astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 3:
        r,g,b = img[...,0], img[...,1], img[...,2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[...,0].astype(np.float32, copy=False)
    return img.astype(np.float32, copy=False)

def _robust_centroid(img: np.ndarray, seed_xy: Optional[Tuple[float,float]]=None, r=40) -> Optional[Tuple[float,float]]:
    """Find a compact bright blob near seed using SEP; fallback to image max."""
    L = _to_luma(img)
    H,W = L.shape
    if seed_xy:
        x0,y0 = int(round(seed_xy[0])), int(round(seed_xy[1]))
        x1,x2 = max(0,x0-r), min(W, x0+r+1)
        y1,y2 = max(0,y0-r), min(H, y0+r+1)
        roi = L[y1:y2, x1:x2]
        if roi.size >= 16:
            bkg = np.median(roi)
            try:
                sep.set_extract_pixstack(int(1e6))
                objs, seg = sep.extract(roi - bkg, thresh=2.0*np.std(roi), minarea=8, filter_type='matched')
                if len(objs):
                    # pick highest peak
                    k = int(np.argmax([o['peak'] for o in objs]))
                    cx = float(objs[k]['x']) + x1
                    cy = float(objs[k]['y']) + y1
                    return (cx, cy)
            except Exception:
                pass
    # fallback: global maximum
    j = int(np.argmax(L))
    cy, cx = divmod(j, W)
    return (float(cx), float(cy))

def _star_suppress(L: np.ndarray) -> np.ndarray:
    """Down-weight stellar pinpoints so big fuzzy cores win."""
    small = cv2.GaussianBlur(L, (0, 0), 1.6).astype(np.float32)
    thr = np.percentile(small, 99.7)
    mask = small > thr              # very bright, compact stuff
    out = L.astype(np.float32).copy()
    out[mask] *= 0.35               # damp stars; keep coma
    return out

def _log_big_blob(L: np.ndarray, sigmas: list[float]) -> tuple[float, float, float]:
    """
    Pick the strongest bright blob across multiple scales using LoG-like response.
    Returns (cx, cy, sigma_used).
    """
    H, W = L.shape
    best_val, best_xy, best_s = -1e9, (W*0.5, H*0.5), sigmas[0]
    for s in sigmas:
        g  = cv2.GaussianBlur(L, (0, 0), s)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        resp = (-lap) * (s * s)     # scale-normalized: favor larger bright blobs
        hi = np.percentile(resp, 99.95)
        resp = np.clip(resp, -1e9, hi)
        j = int(np.argmax(resp))
        cy, cx = divmod(j, W)
        v = resp[cy, cx]
        if v > best_val:
            best_val, best_xy, best_s = float(v), (float(cx), float(cy)), float(s)
    return best_xy[0], best_xy[1], best_s


# --- NEW helpers ---
def _luma_gauss(img: np.ndarray, sigma: float=3.0) -> np.ndarray:
    L = _to_luma(img)
    return cv2.GaussianBlur(L, (0,0), sigmaX=sigma, sigmaY=sigma).astype(np.float32, copy=False)

def _crop_bounds(cx, cy, half, W, H):
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(W, int(round(cx + half)))
    y2 = min(H, int(round(cy + half)))
    return x1, y1, x2, y2

def _norm_patch(p: np.ndarray) -> np.ndarray:
    m = np.median(p)
    s = np.std(p)
    if s < 1e-6: s = 1e-6
    return ((p - m) / s).astype(np.float32, copy=False)

def _minmax_time_key(fp: str) -> float:
    # Try FITS DATE-OBS; fallback to file mtime. Lower is earlier.
    try:
        hdr = fits.getheader(fp, 0)
        t = hdr.get("DATE-OBS") or hdr.get("DATE")
        if t:
            # robust parse: YYYY-MM-DDThh:mm:ss[.sss][Z]
            from datetime import datetime
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    return datetime.strptime(t.replace("Z",""), fmt).timestamp()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        return os.path.getmtime(fp)
    except Exception:
        return 0.0

def _predict(prev_xy: Tuple[float,float], prev2_xy: Optional[Tuple[float,float]]) -> Tuple[float,float]:
    if prev2_xy is None: 
        return prev_xy
    vx = prev_xy[0] - prev2_xy[0]
    vy = prev_xy[1] - prev2_xy[1]
    return (prev_xy[0] + vx, prev_xy[1] + vy)

# --- NEW per-frame star masks (optional, safer than warping) ---
def build_star_masks_per_frame(file_list: List[str], sigma: float=3.5, dilate_px: int=2, status_cb=None) -> Dict[str, np.ndarray]:
    log = status_cb or (lambda *_: None)
    masks = {}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1)) if dilate_px>0 else None
    for fp in file_list:
        img, _, _, _ = load_image(fp)
        if img is None: 
            log(f"  ⚠️ mask: failed to load {os.path.basename(fp)}"); 
            continue
        L = _to_luma(img)
        bkg, _, std = sigma_clipped_stats(L, sigma=3.0, maxiters=5)
        m = (L > (bkg + sigma*std)).astype(np.uint8)
        if k is not None:
            m = cv2.dilate(m, k)
        masks[fp] = (m > 0)
        log(f"  ◦ star mask made for {os.path.basename(fp)}")
    return masks

def _directional_gaussian_kernel(long_px: int, sig_long: float,
                                 sig_cross: float, angle_deg: float) -> np.ndarray:
    """
    Anisotropic Gaussian (elongated) rotated to `angle_deg`.
    long_px controls kernel size along the tail axis.
    """
    long_px = max(21, int(long_px) | 1)
    half = long_px // 2
    yy, xx = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
    # rotate coords
    th = np.deg2rad(angle_deg)
    xr =  np.cos(th)*xx + np.sin(th)*yy     # along-tail
    yr = -np.sin(th)*xx + np.cos(th)*yy     # cross-tail
    g = np.exp(-0.5*( (xr/sig_long)**2 + (yr/sig_cross)**2 ))
    g /= g.sum()
    return g.astype(np.float32)

def _anisotropic_feather(mask_bin: np.ndarray,
                         angle_deg: float,
                         feather_long: float,
                         feather_cross: float) -> np.ndarray:
    """
    Feather with different falloff along vs. across tail by convolving
    the binary mask with an elongated Gaussian oriented at angle_deg.
    """
    k = _directional_gaussian_kernel(
        long_px=int(max(31, 6*max(feather_long, feather_cross))),
        sig_long=float(max(1.0, feather_long/2.5)),
        sig_cross=float(max(1.0, feather_cross/2.5)),
        angle_deg=angle_deg
    )
    soft = cv2.filter2D(mask_bin.astype(np.float32), -1, k, borderType=cv2.BORDER_REPLICATE)
    return np.clip(soft, 0.0, 1.0).astype(np.float32)

def _tail_response(L: np.ndarray, angle_deg: float,
                   bg_sigma: float = 30.0,
                   hp_sigma: float = 2.0,
                   long_px: int = 181,
                   sig_long: float = 40.0,
                   sig_cross: float = 3.0) -> np.ndarray:
    """
    Build a smooth tail-likelihood map: high-pass -> directional blur
    (elongated Gaussian) -> normalize to [0,1].
    """
    # remove large-scale gradient, keep positive high-pass
    low = cv2.GaussianBlur(L, (0,0), bg_sigma)
    hp  = L - low
    hp  = cv2.GaussianBlur(hp, (0,0), hp_sigma)
    hp[hp < 0] = 0.0
    k  = _directional_gaussian_kernel(long_px, sig_long, sig_cross, angle_deg)
    resp = cv2.filter2D(hp, -1, k, borderType=cv2.BORDER_REFLECT)
    # robust scale
    p1, p99 = np.percentile(resp, (1.0, 99.7))
    if p99 <= p1: 
        return np.zeros_like(resp, np.float32)
    return np.clip((resp - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)

def blend_additive_comet(
    comet_only: np.ndarray,
    stars_only: np.ndarray,
    *,
    blackpoint_percentile: float = 50.0,  # 50=median; 35–50 usually safe
    per_channel: bool = True,             # per-channel blackpoint for RGB
    mix: float = 1.0,                     # <1.0 to dial the comet back
    preserve_dtype: bool = False          # keep input dtype (else float32 [0..1])
) -> np.ndarray:
    """
    Additive comet blend:
      1) compute a robust black point from comet_only (percentile),
      2) subtract it (floor at 0) → comet_bgzero,
      3) out = stars_only + mix * comet_bgzero,
      4) clip to [0,1] (or input dtype range).

    Works best when comet_only is starless and roughly linear.
    """
    A = np.asarray(comet_only)
    B = np.asarray(stars_only)

    # harmonize channels
    if A.ndim == 2 and B.ndim == 3 and B.shape[2] == 3:
        A = np.repeat(A[..., None], 3, axis=2)
    if B.ndim == 2 and A.ndim == 3 and A.shape[2] == 3:
        B = np.repeat(B[..., None], 3, axis=2)

    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) robust black point from comet_only
    p = float(np.clip(blackpoint_percentile, 0.0, 100.0))
    if A.ndim == 3 and A.shape[2] == 3 and per_channel:
        bp = np.percentile(A.reshape(-1, 3), p, axis=0).astype(np.float32)  # shape (3,)
    else:
        bp = np.percentile(A, p).astype(np.float32)                          # scalar

    # 2) zero the background
    A0 = A - bp
    A0[A0 < 0] = 0.0

    # 3) additive blend
    out = B + mix * A0

    # 4) clip to [0,1] (assumes normalized floats in the app)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    if preserve_dtype and (stars_only.dtype != np.float32):
        # simple rescale back if you keep non-float data elsewhere
        info = np.iinfo(stars_only.dtype) if np.issubdtype(stars_only.dtype, np.integer) else None
        if info:
            out = (out * info.max + 0.5).astype(stars_only.dtype, copy=False)
    return out


# --- REPLACE measure_comet_positions with this version ---
def measure_comet_positions(
    file_list: List[str],
    seeds: Optional[Dict[str, Tuple[float,float]]] = None,
    status_cb=None,
    *,
    tpl_half: int = 28,
    blur_sigma: float = 3.5,
    max_step_px: float = 45.0,
    min_search_px: float = 16.0,
    max_search_px: float = 80.0,
    score_floor: float = 0.35,
    gamma_pow: float = 0.6,
    refine_r: int = 12,
    adapt_tpl_alpha: float = 0.12
) -> Dict[str, Tuple[float,float]]:
    """
    Track the comet by template matching on blurred luma.
    Frames are processed in temporal order (DATE-OBS; fallback mtime).

    Now with a SECOND PASS local refinement that mirrors the Comet preview “Auto” button.
    """
    log = status_cb or (lambda *_: None)

    # -------- PASS 1: existing template-matching pipeline (unchanged) --------
    ordered = sorted(list(file_list), key=_minmax_time_key)
    out: Dict[str, Tuple[float,float]] = {}
    prev_xy: Optional[Tuple[float,float]] = None
    prev2_xy: Optional[Tuple[float,float]] = None
    tpl: Optional[np.ndarray] = None
    tpl_hw = int(tpl_half)

    # Seed selection logic (unchanged)
    seed_idx = 0
    if seeds:
        for i, f in enumerate(ordered):
            if f in seeds:
                seed_idx = i
                break

    for i, fp in enumerate(ordered):
        img, hdr, _, _ = load_image(fp)
        if img is None:
            log(f"⚠️ measure: failed to load {fp}")
            continue

        # blurred luma + gamma for detection
        L = _luma_gauss(img, sigma=blur_sigma)       # float32
        G = _gamma_stretch(L, gamma=gamma_pow)       # [0..1]
        H, W = G.shape

        if tpl is None:
            # choose seed
            if seeds and fp in seeds:
                cx, cy = seeds[fp]
            elif seeds:
                for f in ordered:
                    if f in seeds: cx, cy = seeds[f]; break
            else:
                j = int(np.argmax(G)); cy, cx = divmod(j, W)

            # keep user/global seed as the first output; refine subpixel on original luma (gamma’d)
            L0g = _gamma_stretch(_to_luma(img), gamma=gamma_pow)
            cx, cy = _refine_centroid(L0g, float(cx), float(cy), r=refine_r)

            x1,y1,x2,y2 = _crop_bounds(cx, cy, tpl_half, W, H)
            tpl = _norm_patch(G[y1:y2, x1:x2])
            prev_xy = (float(cx), float(cy))
            out[fp] = prev_xy
            log(f"  ◦ seed @ {os.path.basename(fp)} → ({prev_xy[0]:.2f},{prev_xy[1]:.2f}) [template {tpl.shape[1]}×{tpl.shape[0]}]")
            continue

        # prediction & adaptive search window
        guess = _predict(prev_xy, prev2_xy)
        if prev2_xy is None:
            sr = max(min_search_px, 0.5*max_step_px)
        else:
            mv = math.hypot(prev_xy[0]-prev2_xy[0], prev_xy[1]-prev2_xy[1])
            sr = np.clip(1.5*mv, min_search_px, max_search_px)

        # ensure search ≥ template
        min_half_needed = 0.5 * max(tpl.shape[1], tpl.shape[0]) + 1.0
        sr = max(sr, min_half_needed)

        # crop and match
        x1, y1, x2, y2 = _bounds_with_min_size(guess[0], guess[1], sr, W, H,
                                               min_w=tpl.shape[1], min_h=tpl.shape[0])
        search = _norm_patch(G[y1:y2, x1:x2])
        res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        px = x1 + loc[0] + tpl.shape[1]*0.5
        py = y1 + loc[1] + tpl.shape[0]*0.5

        step = math.hypot(px - prev_xy[0], py - prev_xy[1])
        ok = (score >= score_floor) and (step <= max_step_px)

        if not ok:
            # one wider search
            x1b, y1b, x2b, y2b = _bounds_with_min_size(guess[0], guess[1], max_search_px, W, H,
                                                       min_w=tpl.shape[1], min_h=tpl.shape[0])
            search2 = _norm_patch(G[y1b:y2b, x1b:x2b])
            res2 = cv2.matchTemplate(search2, tpl, cv2.TM_CCOEFF_NORMED)
            _, score2, _, loc2 = cv2.minMaxLoc(res2)
            px2 = x1b + loc2[0] + tpl.shape[1]*0.5
            py2 = y1b + loc2[1] + tpl.shape[0]*0.5
            step2 = math.hypot(px2 - prev_xy[0], py2 - prev_xy[1])
            if (score2 > score) and (step2 <= max_step_px*1.2):
                px, py, score, step = px2, py2, score2, step2
                ok = (score >= 0.30)

        if not ok:
            px, py = _predict(prev_xy, prev2_xy)
            px = float(np.clip(px, 0, W-1)); py = float(np.clip(py, 0, H-1))
            log(f"  ◦ {os.path.basename(fp)} fallback → ({px:.2f},{py:.2f})")
        else:
            # subpixel refine on original luma (gamma’d)
            L0 = _to_luma(img)
            L0g = _gamma_stretch(L0, gamma=gamma_pow)
            px, py = _refine_centroid(L0g, px, py, r=refine_r)
            log(f"  ◦ {os.path.basename(fp)} match={score:.3f} step={step:.1f}px → ({px:.2f},{py:.2f})")

            # gentle template adaptation
            x1t, y1t, x2t, y2t = _crop_bounds(px, py, tpl_half, W, H)
            new_tpl = _norm_patch(G[y1t:y2t, x1t:x2t])
            if new_tpl.shape == tpl.shape:
                tpl = (1.0 - adapt_tpl_alpha) * tpl + adapt_tpl_alpha * new_tpl

        out[fp] = (px, py)
        prev2_xy, prev_xy = prev_xy, (px, py)

    # light smoothing (unchanged)
    if len(out) >= 5:
        ordered_xy = [out[f] for f in ordered]
        xs = np.array([p[0] for p in ordered_xy], dtype=np.float64)
        ys = np.array([p[1] for p in ordered_xy], dtype=np.float64)
        def _smooth(v):
            s = v.copy()
            for k in range(2, len(v)-2):
                s[k] = (-3*v[k-2] + 12*v[k-1] + 17*v[k] + 12*v[k+1] - 3*v[k+2]) / 35.0
            return s
        xs, ys = _smooth(xs), _smooth(ys)
        for f, x, y in zip(ordered, xs, ys):
            out[f] = (float(x), float(y))

    # -------- PASS 2: local “Auto” refinement around first-pass XY --------
    # Mirrors the dialog’s Auto: star-suppress → multi-scale LoG peak → gamma → subpixel refine
    hint = max(4.0, blur_sigma)                    # reuse blur as the size hint
    sigmas = [0.6*hint, 0.9*hint, 1.3*hint, 1.8*hint, 2.4*hint]
    local_half = int(max(24, 3.0*hint))            # tight local window

    for fp in ordered:
        if fp not in out:
            continue
        img, _, _, _ = load_image(fp)
        if img is None:
            continue
        Lfull = _to_luma(img).astype(np.float32)
        cx0, cy0 = out[fp]
        x1, y1, x2, y2 = _crop_bounds(cx0, cy0, local_half, Lfull.shape[1], Lfull.shape[0])

        # star-suppressed local area + LoG peak
        Ls = _star_suppress(Lfull[y1:y2, x1:x2])
        cx, cy, used = _log_big_blob(Ls, sigmas)
        cx += x1; cy += y1

        # gamma + subpixel refine on the full-luma gamma space
        gL = _gamma_stretch(Lfull, gamma=gamma_pow)
        cx, cy = _refine_centroid(gL, float(cx), float(cy), r=max(refine_r, int(used)))

        out[fp] = (float(cx), float(cy))

    # light re-smoothing (keeps trajectories silky)
    if len(out) >= 5:
        ordered_xy = [out[f] for f in ordered]
        xs = np.array([p[0] for p in ordered_xy], dtype=np.float64)
        ys = np.array([p[1] for p in ordered_xy], dtype=np.float64)
        def _smooth(v):
            s = v.copy()
            for k in range(2, len(v)-2):
                s[k] = (-3*v[k-2] + 12*v[k-1] + 17*v[k] + 12*v[k+1] - 3*v[k+2]) / 35.0
            return s
        xs, ys = _smooth(xs), _smooth(ys)
        for f, x, y in zip(ordered, xs, ys):
            out[f] = (float(x), float(y))

    return out


def _bounds_with_min_size(cx, cy, half, W, H, min_w, min_h):
    # Start from requested half-size
    half = max(half, 1.0)
    # First pass crop
    x1 = int(round(cx - half)); y1 = int(round(cy - half))
    x2 = int(round(cx + half)); y2 = int(round(cy + half))
    # Clamp to image
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)

    # Ensure minimum width/height by expanding/shift-in if needed
    cur_w = x2 - x1; cur_h = y2 - y1
    need_w = max(0, int(min_w - cur_w))
    need_h = max(0, int(min_h - cur_h))

    # Expand symmetrically where possible; otherwise shift inward from edges
    if need_w > 0:
        x1 = max(0, x1 - need_w // 2)
        x2 = min(W, x2 + (need_w - (x1 > 0 and (need_w // 2))))
        # If still short, push entirely to one side
        if (x2 - x1) < min_w:
            if x1 == 0: x2 = min(W, min_w)
            if x2 == W: x1 = max(0, W - min_w)

    if need_h > 0:
        y1 = max(0, y1 - need_h // 2)
        y2 = min(H, y2 + (need_h - (y1 > 0 and (need_h // 2))))
        if (y2 - y1) < min_h:
            if y1 == 0: y2 = min(H, min_h)
            if y2 == H: y1 = max(0, H - min_h)

    # Final clamp/sanity
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))
    return x1, y1, x2, y2


def build_star_masks_from_ref(ref_path: str,
                              ref_star_thresh_sigma: float,
                              inv_transforms: Dict[str, np.ndarray],
                              dilate_px: int = 2,
                              status_cb=None) -> Dict[str, np.ndarray]:
    """Detect stars in ref, then warp mask back to each frame using inverse affine."""
    log = status_cb or (lambda *_: None)
    ref_img, hdr, _, _ = load_image(ref_path)
    L = _to_luma(ref_img)
    bkg, _, std = sigma_clipped_stats(L, sigma=3.0, maxiters=5)
    thresh = bkg + ref_star_thresh_sigma * std
    mask_ref = (L > thresh).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        mask_ref = cv2.dilate(mask_ref, k)

    H, W = L.shape
    masks = {}
    for f, Minv in inv_transforms.items():
        m = cv2.warpAffine(mask_ref, Minv, (W, H),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        masks[f] = m.astype(bool, copy=False)
        log(f"  ◦ star mask warped for {os.path.basename(f)}")
    return masks

def _shift_to_comet(img: np.ndarray, xy: Tuple[float,float], ref_xy: Tuple[float,float]) -> np.ndarray:
    """Translate image so comet xy → ref_xy (subpixel)."""
    dx = ref_xy[0] - xy[0]
    dy = ref_xy[1] - xy[1]
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    H, W = img.shape[:2]
    interp = cv2.INTER_LANCZOS4
    if img.ndim == 2:
        return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)
    # 3-channel
    ch = [cv2.warpAffine(img[...,c], M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT) for c in range(img.shape[-1])]
    return np.stack(ch, axis=-1)

def stack_comet_aligned(file_list: List[str],
                        comet_xy: Dict[str, Tuple[float,float]],
                        star_masks: Optional[Dict[str, np.ndarray]] = None,
                        reducer: str = "biweight",
                        status_cb=None,
                        *,
                        settings=None,
                        enable_star_removal: bool = False,
                        star_removal_tool: str = "StarNet",
                        core_r_px: float = 22.0,
                        core_soft_px: float = 6.0,
                        frames_are_linear: bool = True) -> np.ndarray:
    """
    If enable_star_removal=True, each comet-aligned frame has stars removed
    with the chosen tool and nucleus protected by a soft circular mask.
    """
    log = status_cb or (lambda *_: None)
    ref_xy = comet_xy[file_list[0]]

    accum = []
    core_mask_cache = None

    for fp in file_list:
        img, hdr, _, _ = load_image(fp)
        if img is None: continue

        shifted = _shift_to_comet(img, comet_xy[fp], ref_xy).astype(np.float32)

        if enable_star_removal:
            h, w = shifted.shape[:2]
            if core_mask_cache is None:
                # mask centered at ref_xy after shifting (all frames share this center now)
                core_mask_cache = _protect_core_mask(h, w, ref_xy[0], ref_xy[1], core_r_px, core_soft_px)
            shifted = _starless_frame_for_comet(
                shifted, star_removal_tool, settings,
                is_linear=frames_are_linear, core_mask=core_mask_cache
            )
            # after removal, star_masks are usually unnecessary; ignore them
        else:
            # keep your existing optional masks if not removing stars
            if star_masks and fp in star_masks:
                m = star_masks[fp]
                if shifted.ndim == 2:
                    shifted[m] = np.nan
                else:
                    for c in range(shifted.shape[-1]): shifted[...,c][m] = np.nan

        accum.append(shifted)

    if not accum:
        raise RuntimeError("No valid frames for comet stacking")

    stack = np.stack(accum, axis=0)

    # same reducer as before
    if reducer == "median":
        out = np.nanmedian(stack, axis=0)
    else:
        med = np.nanmedian(stack, axis=0)
        mad = np.nanmedian(np.abs(stack - med), axis=0) + 1e-8
        k = 3.0
        lo, hi = med - k*1.4826*mad, med + k*1.4826*mad
        clipped = np.clip(stack, lo, hi)
        out = np.nanmean(clipped, axis=0)
    return out.astype(np.float32, copy=False)

def make_comet_mask(comet_only: np.ndarray, feather_px: int=24) -> np.ndarray:
    L = _to_luma(comet_only)
    bkg, _, std = sigma_clipped_stats(L, sigma=3.0, maxiters=5)
    m = (L > (bkg + 1.2*std)).astype(np.uint8)
    # binary close + distance feather
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    # feather via distance transform
    inv = 1 - m
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    mask = np.clip(1.0 - dist / max(1, feather_px), 0.0, 1.0)
    return mask.astype(np.float32)



# --- estimate global streak angle from comet motion (deg) ---
def _estimate_streak_angle(comet_xy: dict[str, tuple[float,float]]) -> float:
    if not comet_xy or len(comet_xy) < 2:
        return 0.0
    # order by time-ish from filename sort (good enough here)
    ks = sorted(comet_xy.keys())
    x0, y0 = comet_xy[ks[0]]
    x1, y1 = comet_xy[ks[-1]]
    # stars streak opposite comet motion; angle in image coords
    ang = math.degrees(math.atan2(y0 - y1, x0 - x1))  # y down
    return ang

def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Thin line (1px) dilated to ~3px width; rotated to angle."""
    length = max(3, int(length))
    w = 3
    k = np.zeros((length, length), np.uint8)
    cv2.line(k, (0, length//2), (length-1, length//2), 1, 1)
    M = cv2.getRotationMatrix2D((length/2-0.5, length/2-0.5), angle_deg, 1.0)
    rsz = cv2.warpAffine(k*255, M, (length, length), flags=cv2.INTER_NEAREST)
    if w > 1:
        rsz = cv2.dilate(rsz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w,w)))
    return (rsz > 0).astype(np.uint8)

def _streak_mask_directional(comet_only: np.ndarray,
                             angle_deg: float,
                             hp_sigma: float = 2.0,
                             bg_sigma: float = 15.0,
                             th_sigma: float = 3.0,
                             line_len: int = 19,
                             grow_px: int = 2) -> np.ndarray:
    """
    Detect elongated bright streaks roughly along 'angle_deg'.
    Returns boolean mask (H,W) where True = streak.
    """
    L = _to_luma(comet_only).astype(np.float32)
    # high-pass: remove large-scale coma/tail
    low = cv2.GaussianBlur(L, (0,0), bg_sigma)
    hp  = cv2.GaussianBlur(L - low, (0,0), hp_sigma)

    # robust threshold via MAD
    med = np.median(hp)
    mad = np.median(np.abs(hp - med)) + 1e-6
    z = (hp - med) / (1.4826 * mad)
    m0 = (z > th_sigma).astype(np.uint8)

    # directional opening to keep long, aligned features; suppress compact bits
    kline = _line_kernel(line_len, angle_deg)
    opened = cv2.morphologyEx(m0, cv2.MORPH_OPEN, kline)

    # small cleanups
    if grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*grow_px+1, 2*grow_px+1))
        opened = cv2.dilate(opened, k)
    opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
    return opened.astype(bool)

def _comet_mask_smart(comet_only: np.ndarray,
                      feather_px: int,
                      exclude_mask: np.ndarray | None = None,
                      sigma_k: float = 1.2) -> np.ndarray:
    """
    Stronger comet mask: threshold broad coma/tail, remove star streaks,
    then feather edges by distance.
    """
    L = _to_luma(comet_only).astype(np.float32)
    bkg, _, std = sigma_clipped_stats(L, sigma=3.0, maxiters=5)
    base = (L > (bkg + sigma_k * std)).astype(np.uint8)

    # clean & expand a bit so tail isn’t holey
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)), iterations=1)
    base = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)

    if exclude_mask is not None:
        base[exclude_mask] = 0

    # feather
    inv = 1 - base
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    mask = np.clip(1.0 - dist / max(1, float(feather_px)), 0.0, 1.0)
    return mask.astype(np.float32)

def make_comet_mask_anisotropic(comet_only: np.ndarray,
                                angle_deg: float,
                                *,
                                core_k: float = 1.2,
                                tail_boost: float = 0.7,
                                exclude_streaks: np.ndarray | None = None,
                                feather_long: float = 90.0,
                                feather_cross: float = 18.0) -> np.ndarray:
    """
    Tail-aware comet matte:
      1) core/inner coma via sigma threshold,
      2) add a directional tail likelihood,
      3) remove star streaks,
      4) anisotropic feather along tail.
    """
    L = _to_luma(comet_only).astype(np.float32)
    bkg, _, std = sigma_clipped_stats(L, sigma=3.0, maxiters=5)
    core = (L > (bkg + core_k*std)).astype(np.uint8)

    # grow core a touch so it’s not holey around nucleus
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

    # directional tail map ∈ [0,1]; boost then clamp
    tail = _tail_response(L, angle_deg=angle_deg)
    tail = np.clip(tail * float(tail_boost), 0.0, 1.0)

    # combine: binarize core strongly, add soft tail
    m0 = np.clip(core.astype(np.float32) * 1.0 + tail * (1.0 - core.astype(np.float32)), 0.0, 1.0)

    # remove linear star streaks if provided
    if exclude_streaks is not None:
        m0[exclude_streaks] = 0.0

    # hard floor to keep nucleus fully in
    m_bin = (m0 > 0.15).astype(np.uint8)

    # anisotropic feather (stretches along tail, tight across)
    matte = _anisotropic_feather(m_bin, angle_deg=angle_deg,
                                 feather_long=feather_long,
                                 feather_cross=feather_cross)
    return np.clip(matte, 0.0, 1.0).astype(np.float32)

def blend_comet_stars(
    comet_only: np.ndarray,
    stars_only: np.ndarray,
    feather_px: int = 24,   # kept for compatibility; now used as cross-feather
    mix: float = 1.0,
    *,
    comet_xy: dict[str, tuple[float,float]] | None = None
) -> np.ndarray:
    """
    Tail-aware blend. Uses directional matte instead of radial blob.
    `feather_px` controls *cross-tail* softness; along-tail uses a longer value automatically.
    """
    A = np.asarray(comet_only, dtype=np.float32)
    B = np.asarray(stars_only, dtype=np.float32)

    # channel harmonization
    ch = 3 if ((A.ndim==3 and A.shape[-1]==3) or (B.ndim==3 and B.shape[-1]==3)) else 1
    if ch == 3:
        if A.ndim == 2: A = np.repeat(A[...,None], 3, axis=2)
        if B.ndim == 2: B = np.repeat(B[...,None], 3, axis=2)
    else:
        if A.ndim == 3 and A.shape[-1] == 1: A = A[...,0]
        if B.ndim == 3 and B.shape[-1] == 1: B = B[...,0]

    angle = _estimate_streak_angle(comet_xy) if comet_xy else 0.0
    # streak mask (same as before)
    S = _streak_mask_directional(A, angle_deg=angle)

    # anisotropic comet matte
    M2D = make_comet_mask_anisotropic(
        A, angle_deg=angle,
        core_k=1.2, tail_boost=0.9,
        exclude_streaks=S,
        feather_long=max(70.0, 4.5*feather_px),   # long feather down the tail
        feather_cross=float(feather_px)           # tight across the tail
    )
    M2D *= float(mix)

    if ch == 3:
        M = np.repeat(M2D[...,None], 3, axis=2)
    else:
        M = M2D

    out = A * M + B * (1.0 - M)
    return out.astype(np.float32, copy=False)



time_key = _minmax_time_key 

def debug_save_marks(file_list, comet_xy, out_dir, radius=12):
    os.makedirs(out_dir, exist_ok=True)
    for fp in file_list:
        img, _, _, _ = load_image(fp)
        if img is None or fp not in comet_xy: 
            continue
        x,y = comet_xy[fp]
        L = _to_luma(img)
        disp = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        cv2.circle(rgb, (int(round(x)), int(round(y))), radius, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(fp) + ".png"), rgb)

def _protect_core_mask(h: int, w: int, cx: float, cy: float, r: float, soft: float) -> np.ndarray:
    """
    Radial soft mask centered at (cx,cy): 1 near core (protected), 0 far.
    r = hard radius, soft = feather (pixels).
    Returns 2D float32 [0..1].
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - float(cx), yy - float(cy))
    m = np.clip((r + soft - d) / max(1e-6, soft), 0.0, 1.0)
    return m.astype(np.float32)

def _starless_frame_for_comet(img: np.ndarray,
                              tool: str,
                              settings,
                              *,
                              is_linear: bool,
                              core_mask: np.ndarray) -> np.ndarray:
    """
    Run selected remover on a single frame and protect the nucleus with core_mask (H,W).
    Returns RGB float32 [0..1] starless, with nucleus restored from original.
    """
    # ensure RGB float32 [0..1]
    if img.ndim == 2: src = np.stack([img]*3, axis=-1).astype(np.float32)
    elif img.ndim == 3 and img.shape[2] == 1: src = np.repeat(img, 3, axis=2).astype(np.float32)
    else: src = img.astype(np.float32, copy=False)

    # run
    if tool == "CosmicClarityDarkStar":
        starless = darkstar_starless_from_array(src, settings)
    else:
        starless = starnet_starless_from_array(src, settings, is_linear=is_linear)

    # protect nucleus (blend original back where mask=1)
    m = core_mask.astype(np.float32)
    m3 = np.repeat(m[...,None], 3, axis=2)
    protected = starless * (1.0 - m3) + src * m3
    return np.clip(protected, 0.0, 1.0)


def _gamma_stretch(x: np.ndarray, gamma: float = 0.6,
                   lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    """
    Percentile-clip → normalize to [0,1] → power-law gamma → back to float32.
    gamma < 1 brightens midtones (good for faint coma).
    """
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi <= lo:
        return x  # degenerate; skip
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    y = np.power(y, gamma, dtype=np.float32)
    return y

def _refine_centroid(L: np.ndarray, px: float, py: float, r: int = 12) -> Tuple[float, float]:
    """
    Subpixel refinement around (px,py) using an intensity-weighted centroid
    on a small ROI after subtracting a robust local background.
    """
    H, W = L.shape
    x1 = max(0, int(round(px - r)));  x2 = min(W, int(round(px + r + 1)))
    y1 = max(0, int(round(py - r)));  y2 = min(H, int(round(py + r + 1)))
    roi = L[y1:y2, x1:x2].astype(np.float32, copy=False)
    if roi.size < 16: 
        return px, py

    m = np.median(roi)
    s = np.std(roi)
    thr = m + 1.0 * s
    w = roi - thr
    w[w < 0] = 0.0  # keep only positive contrast (coma/core)
    if not np.any(w):
        return px, py

    ys, xs = np.mgrid[y1:y2, x1:x2]
    Wsum = float(w.sum())
    cx = float((w * xs).sum() / Wsum)
    cy = float((w * ys).sum() / Wsum)
    return cx, cy



# ---------------- Qt6-only centroid review dialog ----------------
try:
    # Prefer PyQt6
    from PyQt6.QtCore import Qt, QPointF, QEvent
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QCursor
    from PyQt6.QtWidgets import (
        QDialog, QListWidget, QListWidgetItem, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QSlider, QWidget, QSpinBox, QCheckBox, QGraphicsView,
        QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
    )
    _QT_BINDING = "PyQt6"
except Exception:
    # Fallback to PySide6 (still Qt6)
    from PySide6.QtCore import Qt, QPointF, QEvent
    from PySide6.QtGui import QImage, QPixmap, QPainter, QPen
    from PySide6.QtWidgets import (
        QDialog, QListWidget, QListWidgetItem, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QSlider, QWidget, QSpinBox, QCheckBox, QGraphicsView,
        QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem
    )
    _QT_BINDING = "PySide6"

CursorShape = Qt.CursorShape 

class CometCentroidPreview(QDialog):
    """
    Qt6 dialog to review/adjust comet centroids for a list of frames.
    Returns { path: (x, y) } via get_seeds() after accept().
    """
    def __init__(self, file_list, initial_xy=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comet: Review & Adjust Centroids")
        self.files = list(file_list)
        self.xy = dict(initial_xy or {})
        self.gamma = 0.6
        self.blur  = 3.5
        self.dot_r = 12
        self.zoom  = 1.0

        # --- left: list ---
        self.listw = QListWidget()
        for p in self.files:
            it = QListWidgetItem(os.path.basename(p))
            it.setToolTip(p)
            self.listw.addItem(it)
        self.listw.currentRowChanged.connect(self._on_select)

        # --- center: graphics view ---
        self.scene = QGraphicsScene(self)
        self.view  = QGraphicsView(self.scene)
        self.view.setRenderHints(
            self.view.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setCursor(QCursor(CursorShape.ArrowCursor))
        self.view.viewport().setCursor(QCursor(CursorShape.ArrowCursor))
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        self.cross = QGraphicsEllipseItem(-self.dot_r, -self.dot_r, 2*self.dot_r, 2*self.dot_r)
        pen = QPen(Qt.GlobalColor.green); pen.setWidthF(1.5)
        self.cross.setPen(pen)
        self.scene.addItem(self.cross)
        self.view.viewport().installEventFilter(self)

        # --- right: controls ---
        self.s_gamma = QSlider(Qt.Orientation.Horizontal); self._prep_slider(self.s_gamma, 10, 200, int(self.gamma*100))
        self.s_blur  = QSlider(Qt.Orientation.Horizontal); self._prep_slider(self.s_blur, 0, 80, int(self.blur*10))
        self.s_zoom  = QSlider(Qt.Orientation.Horizontal); self._prep_slider(self.s_zoom, 10, 300, int(self.zoom*100))
        self.s_gamma.valueChanged.connect(self._refresh_current)
        self.s_blur.valueChanged.connect(self._refresh_current)
        self.s_zoom.valueChanged.connect(self._apply_zoom)

        self.n_prop = QSpinBox(); self.n_prop.setRange(1, 50); self.n_prop.setValue(3)
        self.cb_show_gamma = QCheckBox("Show gamma preview"); self.cb_show_gamma.setChecked(True)

        self.btn_auto   = QPushButton("Auto")
        self.btn_prev   = QPushButton("⟲ Prev")
        self.btn_next   = QPushButton("Next ⟳")
        self.btn_copyf  = QPushButton("Propagate →")
        self.btn_ok     = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")

        self.btn_auto.clicked.connect(self._auto_pick)
        self.btn_prev.clicked.connect(lambda: self._change_row(-1))
        self.btn_next.clicked.connect(lambda: self._change_row(+1))
        self.btn_copyf.clicked.connect(self._propagate_forward)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        ctrls = QVBoxLayout()
        ctrls.addWidget(QLabel("Gamma")); ctrls.addWidget(self.s_gamma)
        ctrls.addWidget(QLabel("Blur σ")); ctrls.addWidget(self.s_blur)
        ctrls.addWidget(QLabel("Zoom")); ctrls.addWidget(self.s_zoom)
        ctrls.addWidget(self.cb_show_gamma)
        r1 = QHBoxLayout(); r1.addWidget(self.btn_auto); r1.addWidget(self.btn_copyf); r1.addWidget(self.n_prop); ctrls.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(self.btn_prev); r2.addWidget(self.btn_next); ctrls.addLayout(r2)
        ctrls.addStretch(1)
        r3 = QHBoxLayout(); r3.addWidget(self.btn_ok); r3.addWidget(self.btn_cancel); ctrls.addLayout(r3)

        main = QHBoxLayout(self)
        main.addWidget(self.listw, 1)
        main.addWidget(self.view, 4)
        w = QWidget(); w.setLayout(ctrls)
        main.addWidget(w, 2)

        self.cb_show_gamma.toggled.connect(self._refresh_current)

        if self.files:
            self.listw.setCurrentRow(0)

        if self.files and self.files[0] not in self.xy:
            self._auto_pick(one_file=self.files[0], silent=True)
            self._place_cross()

        self.view.viewport().installEventFilter(self)     

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            if ev.type() == QEvent.Type.CursorChange:
                obj.setCursor(QCursor(CursorShape.ArrowCursor))
                return True
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton:
                    pos = self.view.mapToScene(ev.position().toPoint())
                    self._set_xy_current(pos.x(), pos.y())
                    return True
        return super().eventFilter(obj, ev)


    # --- Qt6 helpers ---
    def _prep_slider(self, s, lo, hi, val):
        s.setRange(lo, hi); s.setValue(val); s.setSingleStep(1); s.setPageStep(5)

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport() and ev.type() == QEvent.Type.MouseButtonPress:
            if ev.button() == Qt.MouseButton.LeftButton:
                pos = self.view.mapToScene(ev.position().toPoint())
                self._set_xy_current(pos.x(), pos.y())
                return True
        return super().eventFilter(obj, ev)

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            dx = -0.5 if k == Qt.Key.Key_Left else (0.5 if k == Qt.Key.Key_Right else 0.0)
            dy = -0.5 if k == Qt.Key.Key_Up   else (0.5 if k == Qt.Key.Key_Down  else 0.0)
            f = self._cur_file()
            if f in self.xy:
                x,y = self.xy[f]; self.xy[f] = (x+dx, y+dy); self._place_cross()
            ev.accept(); return
        super().keyPressEvent(ev)

    # --- logic ---
    def _cur_file(self):
        r = self.listw.currentRow()
        return self.files[r] if 0 <= r < len(self.files) else None

    def _change_row(self, delta):
        r = self.listw.currentRow()
        self.listw.setCurrentRow(max(0, min(len(self.files)-1, r+delta)))

    def _apply_zoom(self):
        self.zoom = max(0.1, self.s_zoom.value()/100.0)
        self.view.resetTransform()
        self.view.scale(self.zoom, self.zoom)

    def _render_preview(self, img):
        if self.cb_show_gamma.isChecked():
            sigma = max(0.0, self.s_blur.value()/10.0)
            g = max(0.1, self.s_gamma.value()/100.0)
            L = _luma_gauss(img, sigma if sigma>0 else 0.0)
            G = _gamma_stretch(L, gamma=g)
            disp = cv2.normalize(G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            L = _to_luma(img)
            disp = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return disp

    def _on_select(self, row):
        fp = self._cur_file()
        if not fp: return
        img, _, _, _ = load_image(fp)
        if img is None: return
        disp = self._render_preview(img)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0], QImage.Format.Format_Grayscale8)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.scene.setSceneRect(0, 0, disp.shape[1], disp.shape[0])
        if fp not in self.xy:
            self._auto_pick(one_file=fp, silent=True)
        self._place_cross()
        self._apply_zoom()

    def _place_cross(self):
        fp = self._cur_file()
        if not fp or fp not in self.xy: return
        x,y = self.xy[fp]
        self.cross.setPos(QPointF(x, y))

    def _set_xy_current(self, x, y):
        fp = self._cur_file()
        if not fp: return
        self.xy[fp] = (float(x), float(y))
        self._place_cross()

    def _auto_pick(self, one_file=None, silent=False):
        targets = [one_file] if one_file else [self._cur_file()]
        hint = max(4.0, self.s_blur.value()/10.0)
        sigmas = [0.6*hint, 0.9*hint, 1.3*hint, 1.8*hint, 2.4*hint]

        for fp in targets:
            if not fp: continue
            img, _, _, _ = load_image(fp)
            if img is None: continue
            L = _to_luma(img).astype(np.float32)

            # 1) try local search around existing xy (seed or previous)
            cx0, cy0 = self.xy.get(fp, (None, None))
            found = False
            if cx0 is not None:
                half = max(24, int(3*hint))
                x1,y1,x2,y2 = _crop_bounds(cx0, cy0, half, L.shape[1], L.shape[0])
                Ls = _star_suppress(L[y1:y2, x1:x2])
                cx, cy, used = _log_big_blob(Ls, sigmas)
                cx += x1; cy += y1
                g = max(0.1, self.s_gamma.value()/100.0)
                cx, cy = _refine_centroid(_gamma_stretch(L, g), float(cx), float(cy), r=max(10, int(used)))
                self.xy[fp] = (float(cx), float(cy))
                found = True

            # 2) global fallback
            if not found:
                Ls = _star_suppress(L)
                cx, cy, used = _log_big_blob(Ls, sigmas)
                g = max(0.1, self.s_gamma.value()/100.0)
                cx, cy = _refine_centroid(_gamma_stretch(L, g), float(cx), float(cy), r=max(10, int(used)))
                self.xy[fp] = (float(cx), float(cy))

        self._place_cross()
        if not silent:
            self._refresh_current()

    def _propagate_forward(self):
        n = int(self.n_prop.value())
        r = self.listw.currentRow()
        if r < 0: return
        fp = self.files[r]
        if fp not in self.xy: return
        for k in range(1, n+1):
            i = r + k
            if i >= len(self.files): break
            self.xy[self.files[i]] = self.xy[fp]
        self._change_row(+1)

    def get_seeds(self):
        return dict(self.xy)

    def _refresh_current(self):
        """Re-render current frame with the latest gamma/blur and keep the cross in place."""
        r = self.listw.currentRow()
        if r < 0 or r >= len(self.files):
            return
        fp = self.files[r]
        img, _, _, _ = load_image(fp)
        if img is None:
            return
        disp = self._render_preview(img)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0],
                    QImage.Format.Format_Grayscale8)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.scene.setSceneRect(0, 0, disp.shape[1], disp.shape[0])
        self._place_cross()   # keep marker where it was
