from __future__ import annotations
import numpy as np, cv2
from typing import Optional

# destination-mask helper (optional blend)
from pro.add_stars import _active_mask_array_from_doc

# Linear luma weights
_LUMA_REC709  = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_LUMA_REC601  = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
_LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

# ---------- helpers ----------

def _to_float01_strict(a: np.ndarray) -> np.ndarray:
    """
    Convert to float32 in [0,1] without 'normalize by image max'.
    Integers are scaled by their full-range max; float input is passed through.
    """
    a = np.asarray(a)
    if a.dtype == np.float32:
        return a
    if a.dtype == np.float64:
        return a.astype(np.float32)
    if a.dtype == np.uint8:
        return (a.astype(np.float32) / 255.0)
    if a.dtype == np.uint16:
        return (a.astype(np.float32) / 65535.0)
    if np.issubdtype(a.dtype, np.integer):
        maxv = np.float32(np.iinfo(a.dtype).max)
        return (a.astype(np.float32) / maxv)
    # other floats: assume already 0..1
    return a.astype(np.float32)

def _estimate_noise_sigma_per_channel(img01: np.ndarray) -> np.ndarray:
    # unchanged (but call with strict input)
    a = img01
    if a.ndim == 2:
        a = a[..., None]
    a = a[::4, ::4, :].astype(np.float32, copy=False)
    med = np.median(a, axis=(0,1))
    mad = np.median(np.abs(a - med), axis=(0,1))
    sigma = 1.4826 * mad
    sigma[sigma <= 1e-12] = 1e-12
    return sigma.astype(np.float32)

# ---------- luminance compute (linear) ----------

def compute_luminance(
    img: np.ndarray,
    method: str | None = "rec709",
    weights: Optional[np.ndarray] = None,
    noise_sigma: Optional[np.ndarray] = None,
    normalize_weights: bool = True
) -> np.ndarray:
    """
    Returns 2-D linear luminance Y in [0,1] (float32).
    No per-image normalization. If custom `weights` are supplied and
    `normalize_weights=False`, their absolute sum is respected.
    """
    f = _to_float01_strict(img)

    if f.ndim == 2:
        return np.ascontiguousarray(f.astype(np.float32, copy=False))
    if f.ndim != 3:
        raise ValueError("compute_luminance: expected 2-D or 3-D array.")

    H, W, C = f.shape
    if C == 1:
        return np.ascontiguousarray(f[..., 0].astype(np.float32, copy=False))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 1 or w.size not in (C, 3):
            raise ValueError("weights must be 1-D with length equal to channel count or 3.")
        if normalize_weights:
            s = float(w.sum())
            if s != 0.0:
                w = w / s
        useC = w.size
        lum = np.tensordot(f[..., :useC], w, axes=([2], [0]))
    elif method == "equal":
        lum = f[..., :3].mean(axis=2)
    elif method == "snr":
        if noise_sigma is None:
            raise ValueError("snr method requires noise_sigma per channel.")
        ns = np.asarray(noise_sigma, dtype=np.float32)
        if ns.ndim != 1 or ns.size not in (C, 3):
            raise ValueError("noise_sigma must be 1-D with length equal to channel count or 3.")
        useC = ns.size
        w = 1.0 / (ns[:useC]**2 + 1e-12)
        w = w / w.sum()
        lum = np.tensordot(f[..., :useC], w, axes=([2],[0]))
    elif method == "max":
        lum = f.max(axis=2)
    elif method == "median":
        lum = np.median(f, axis=2)
    else:  # default rec709
        lum = np.tensordot(f[..., :3], _LUMA_REC709, axes=([2],[0]))

    return np.clip(lum.astype(np.float32, copy=False), 0.0, 1.0)

# ---------- luminance recombine (linear scaling) ----------

def recombine_luminance_linear_scale(
    target_rgb: np.ndarray,
    new_L: np.ndarray,
    weights: np.ndarray = _LUMA_REC709,
    eps: float = 1e-6,
    blend: float = 1.0,           # 0..1, 1=full replace
    highlight_soft_knee: float = 0.0  # 0..1, optional protection
) -> np.ndarray:
    """
    Replace linear luminance Y (w·RGB) with `new_L` by per-pixel scaling:
      s = new_L / (Y + eps);  RGB' = RGB * s
    This preserves hue/chroma in linear space and round-trips when new_L==Y.
    Optional: blend (mix with original) and highlight soft-knee protection.
    """
    rgb = _to_float01_strict(target_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Recombine Luminance requires an RGB target image.")

    H, W, _ = rgb.shape
    L = new_L.astype(np.float32)
    if L.shape[:2] != (H, W):
        L = cv2.resize(L, (W, H), interpolation=cv2.INTER_LINEAR)

    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (3,):
        raise ValueError("weights must be length-3 for RGB recombine.")

    # current Y
    Y = rgb[..., 0]*w[0] + rgb[..., 1]*w[1] + rgb[..., 2]*w[2]
    s = L / (Y + eps)

    if highlight_soft_knee > 0.0:
        # compress extreme upsizing to avoid blowing out tiny Y
        # knee in [0..1], higher = more protection
        k = np.clip(highlight_soft_knee, 0.0, 1.0)
        s = s / (1.0 + k*(s - 1.0))

    out = rgb * s[..., None]
    out = np.clip(out, 0.0, 1.0)

    if 0.0 <= blend < 1.0:
        out = rgb*(1.0 - blend) + out*blend

    return out.astype(np.float32, copy=False)

def apply_recombine_to_doc(
    target_doc,
    luminance_source_img: np.ndarray,
    method: str = "rec709",
    weights: Optional[np.ndarray] = None,
    noise_sigma: Optional[np.ndarray] = None,
    blend: float = 1.0,
    soft_knee: float = 0.0
):
    """
    Overwrite target_doc.image by recombining with luminance from source (RGB or mono).
    Uses linear scaling recombine; honors destination mask if present.
    """
    base = _to_float01_strict(np.asarray(target_doc.image))

    # Decide weights for both compute+recombine
    if method == "rec601":
        w = _LUMA_REC601
    elif method == "rec2020":
        w = _LUMA_REC2020
    elif weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.size != 3:
            raise ValueError("Custom weights must be length-3.")
    else:
        w = _LUMA_REC709

    # Build L (mono source passes through; RGB is weighted)
    src = _to_float01_strict(luminance_source_img)
    if src.ndim == 2 or (src.ndim == 3 and src.shape[2] == 1):
        L = src if src.ndim == 2 else src[..., 0]
    else:
        ns = None
        if method == "snr":
            ns = _estimate_noise_sigma_per_channel(src)
        L = compute_luminance(src, method=method, weights=w if weights is not None else None, noise_sigma=ns)

    replaced = recombine_luminance_linear_scale(base, L, weights=w, blend=blend, highlight_soft_knee=soft_knee)

    # destination-mask blend if active
    m = _active_mask_array_from_doc(target_doc)
    if m is not None:
        m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32)
        replaced = base * (1.0 - m3) + replaced * m3

    target_doc.apply_edit(
        replaced,
        metadata={"step_name": "Recombine Luminance", "luma_method": method, "luma_weights": w.tolist()},
        step_name="Recombine Luminance",
    )
