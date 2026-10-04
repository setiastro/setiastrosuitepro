# setiastro/saspro/abe_multiscale_gradient.py
"""
Multiscale multiplicative-gradient correction stage for ADBE.

Removes SHARP-EDGED, MULTIPLICATIVE defects that polynomial/RBF
background fits cannot represent -- dust motes, internal reflections,
filter-edge shadows -- AFTER the normal (smooth, additive) background
removal has already run.

Pipeline (all AFTER the normal ABE subtract/divide stage):
  1. (caller) DarkStar star removal -> starless estimation image
  2. protect remaining signal with a robust median + k*MAD mask
     (k is in MAD-sigma units: 1 = aggressive protection, 5 = only the
      very brightest cores)
  3. inpaint the protected regions -- on a DOWNSAMPLED copy, because
     gradients are coarse and a full-res fill is both pointless and the
     cause of the original "stuck on inpainting" hang
  4. decompose in LOG space (multiplicative -> additive) using the SAME
     fast torch/cv2 blur as multiscale_decomp.py (GPU when available)
  5. sum the selected scale band -> log(gradient)
  6. correct the real image:  corrected = img * median(grad) / grad

Dependencies: numpy, scipy (both guaranteed in SASpro). Blur and
decomposition reuse multiscale_decomp._blur_gaussian when importable
(torch/cv2 backed); otherwise fall back to scipy/cv2. The slow
numpy apply_along_axis path is never used.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

_MAD_NORM = 1.4826


# ----------------------------------------------------------------------
# Fast blur -- reuse multiscale_decomp's torch/cv2 path when available
# ----------------------------------------------------------------------
def _get_blur():
    """Return the project's fast gaussian blur (_blur_gaussian: torch/GPU
    or cv2). Falls back to scipy gaussian_filter. Never the slow numpy
    apply_along_axis path."""
    try:
        from setiastro.saspro.multiscale_decomp import _blur_gaussian as _b
        return _b
    except Exception:
        pass
    try:
        import cv2
        def _b(img, sigma):
            if sigma <= 0:
                return img.astype(np.float32, copy=False)
            k = int(max(3, 2 * round(3 * sigma) + 1)) | 1
            return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_REFLECT).astype(np.float32)
        return _b
    except Exception:
        pass
    def _b(img, sigma):
        if sigma <= 0:
            return img.astype(np.float32, copy=False)
        return gaussian_filter(img, sigma=sigma, mode="reflect").astype(np.float32)
    return _b


def _decompose_channel(img2d, layers, base_sigma, blur, stop_layer=None):
    """a-trous pyramid using the fast shared blur. detail = c - blur(c).

    stop_layer: if given, only compute detail layers 0..stop_layer
    (inclusive); the running coarse image is returned as residual. Saves
    the expensive large-sigma blurs for coarse layers we never use when
    include_residual is False and band_hi < layers-1. Unused slots are
    padded with None so band indexing stays valid."""
    c = img2d.astype(np.float32, copy=False)
    details = []
    last = (layers - 1) if stop_layer is None else min(int(stop_layer), layers - 1)
    for k in range(last + 1):
        c_next = blur(c, base_sigma * (2 ** k))
        details.append(c - c_next)
        c = c_next
    while len(details) < layers:
        details.append(None)
    return details, c


def _robust_sigma(arr):
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return _MAD_NORM * mad


# ----------------------------------------------------------------------
# Downsample / upsample
# ----------------------------------------------------------------------
def _downscale(img, factor):
    if factor <= 1:
        return img.astype(np.float32, copy=False)
    try:
        import cv2
        h, w = img.shape[:2]
        return cv2.resize(img, (max(1, w // factor), max(1, h // factor)),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        return img[::factor, ::factor].astype(np.float32)


def _upscale_to(img, out_hw):
    oh, ow = out_hw
    try:
        import cv2
        return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except Exception:
        yi = np.linspace(0, img.shape[0] - 1, oh).astype(np.int32)
        xi = np.linspace(0, img.shape[1] - 1, ow).astype(np.int32)
        return img[yi][:, xi].astype(np.float32)


# ----------------------------------------------------------------------
# Signal protection: median + k*MAD  (k in MAD-sigma units, 1..5)
# ----------------------------------------------------------------------
def build_signal_mask(luma, k=3.0, grow=3):
    """True where luma > median + k*(MAD-sigma). k is in MAD-sigma units:
    k=1 masks a lot (protects faint signal, safest), k=5 masks only the
    brightest cores."""
    med = float(np.median(luma))
    sig = _robust_sigma(luma)
    mask = luma > (med + float(k) * sig)
    if grow > 0 and mask.any():
        mask = binary_dilation(mask, iterations=int(grow))
    return mask


def hole_span_iters(mask, sigma):
    """Estimate how many blur-fill passes are needed to close the largest
    hole: each pass propagates signal inward by ~sigma px, so iters scales
    with (max hole radius / sigma). Uses a distance transform when scipy is
    present, else a conservative constant."""
    try:
        from scipy.ndimage import distance_transform_edt
        # distance from each hole pixel to nearest non-hole pixel
        dist = distance_transform_edt(mask)
        max_reach = float(dist.max()) if dist.size else 0.0
        return int(np.ceil(max_reach / max(1.0, sigma))) + 4
    except Exception:
        return 16


def _feather_mask(mask01, feather_px, blur):
    """
    Soften a 0..1 protect mask's boundary by blurring it with a large Gaussian
    (sigma = feather_px), spreading the 0->1 transition over a wide soft band
    so the protected region dissolves into the corrected background with no
    visible seam. feather_px <= 0 returns the mask unchanged.
    """
    m = np.clip(np.asarray(mask01, dtype=np.float32), 0.0, 1.0)
    f = float(feather_px)
    if f <= 0.0 or not (m.any() and (m < 1.0).any()):
        return m
    # Just blur the mask. A large Gaussian spreads the 0->1 transition over a
    # wide, soft band so the protected region dissolves into the corrected
    # background with no visible seam. feather_px is the blur sigma; at
    # ~10% of the image's short side it's a few hundred px on a real frame,
    # which is what actually looks feathered.
    return np.clip(blur(m, max(1.0, f)), 0.0, 1.0).astype(np.float32)


def _inpaint_fill(img2d, mask, blur, iters=12, sigma=6.0):
    """Blur-fill on a (downscaled) image. Seed holes with background median,
    then relax by repeated fast blur + hole-replace. Few iterations because
    holes are small at the working scale."""
    out = img2d.astype(np.float32, copy=True)
    hole = np.asarray(mask, dtype=bool)
    if not hole.any():
        return out
    seed = float(np.median(img2d[~hole])) if (~hole).any() else float(np.median(img2d))
    out[hole] = seed
    for _ in range(int(iters)):
        b = blur(out, sigma)
        out[hole] = b[hole]
    return out


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
def multiscale_gradient_correct(
    target_image,
    *,
    estimate_from=None,
    layers=9,
    base_sigma=1.0,
    band_lo=6,
    band_hi=8,
    include_residual=False,
    eps_frac=0.01,
    strength=1.0,
    protect_k=3.0,          # MAD-sigma units (1..5 typical)
    protect_grow=3,
    protect_blend_mask=None,  # bool/float HxW, True/1 = keep ORIGINAL pixels
                              # at APPLY time (e.g. a hand-drawn exclusion
                              # polygon around a galaxy). The multiscale
                              # process runs on the whole image; this is just
                              # a final composite so the protected region is
                              # taken straight from the untouched source.
    protect_feather_px=0.0,   # Gaussian blur sigma (px) applied to the exclusion
                              # mask so its edge dissolves into the corrected
                              # background. 0 (default) AUTO-scales to 10% of the
                              # image's short side (~300 px on a 4500x3000 frame),
                              # which is what actually looks feathered — a fixed
                              # small value is invisible at full resolution. A
                              # positive number forces that exact blur sigma.
    estimate_downsample=1,  # 1 = full-res (REQUIRED for sharp motes/edges);
                            # >1 only for a fast coarse preview, never for apply
    gradient_smooth_px=2.0, # final blur (px) of the gradient MAP before
                            # dividing. A gradient is low-frequency by
                            # definition, so smoothing the map removes any
                            # per-pixel noise the band picked up (worst at low
                            # band_lo) without touching the image's own detail.
                            # 0 disables; 2-3 px is a good default.
    clamp_sigma=0.0,
    return_extras=False,
    progress_cb=None,
):
    """
    Multiplicative gradient correction via log-space multiscale band
    extraction. Estimate on `estimate_from` (starless), apply to
    `target_image`. gradient_map median ~1.0; output re-anchored to the
    target's original median so overall brightness is unchanged.

    protect_blend_mask: optional array (same HxW as the image), True/1 where
    the ORIGINAL pixels should be kept. The gradient is still estimated and
    applied across the whole frame; this mask only governs the final
    composite — out = (1-m)*corrected + m*original — so a hand-drawn
    exclusion region (a galaxy) comes back exactly as it was, regardless of
    what the gradient did there.
    """
    def _say(m):
        if callable(progress_cb):
            try: progress_cb(m)
            except Exception: pass

    blur = _get_blur()

    tgt = np.asarray(target_image, dtype=np.float32)
    est = np.asarray(estimate_from if estimate_from is not None else target_image,
                     dtype=np.float32)

    if est.ndim == 2:
        luma_full = est
    elif est.ndim == 3 and est.shape[2] == 1:
        luma_full = est[..., 0]
    else:
        luma_full = 0.2126 * est[..., 0] + 0.7152 * est[..., 1] + 0.0722 * est[..., 2]
    luma_full = np.nan_to_num(luma_full, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    H, W = luma_full.shape[:2]
    med = float(np.median(luma_full))
    eps = max(1e-6, med * float(eps_frac))

    ds = max(1, int(estimate_downsample))
    _say("Downscaling for gradient estimate...")
    luma_s = _downscale(luma_full, ds)

    _say("Building signal-protection mask (median + k*MAD)...")
    sig_mask_s = build_signal_mask(luma_s, k=protect_k, grow=protect_grow)

    _say("Log transform...")
    L = np.log(np.clip(luma_s, 0.0, None) + eps)

    # Fill reach must match the scale we extract the gradient at, so holes
    # are filled with smooth background at the band's scale (not just a tiny
    # local blur that leaves a bump the mid-band then reads as "gradient").
    # The coarsest band layer has scale ~ base_sigma * 2**band_hi; size the
    # fill blur to roughly that, and give enough iterations to propagate
    # across the largest holes at full resolution.
    _say("Inpainting protected regions...")
    _fill_sigma = max(2.0, float(base_sigma) * (2.0 ** max(0, int(band_hi) - 2)))
    _fill_iters = int(np.clip(hole_span_iters(sig_mask_s, _fill_sigma), 8, 40)) \
        if sig_mask_s.any() else 0
    L_fill = _inpaint_fill(L, sig_mask_s, blur,
                           iters=_fill_iters, sigma=_fill_sigma)

    layers = int(layers)
    band_lo = max(0, min(int(band_lo), layers - 1))
    band_hi = max(band_lo, min(int(band_hi), layers - 1))

    # Only compute layers we actually use. If the coarse residual is folded
    # in, we must run the full pyramid; otherwise stop at band_hi and skip
    # the most expensive large-sigma blurs.
    stop_layer = None if include_residual else band_hi

    _say("Multiscale decomposition (log, torch/cv2 blur)...")
    details, residual = _decompose_channel(
        L_fill, layers, float(base_sigma), blur, stop_layer=stop_layer)

    _say(f"Summing gradient band (layers {band_lo}-{band_hi})...")
    G_s = np.zeros_like(L)
    for i in range(band_lo, band_hi + 1):
        layer = details[i]
        if clamp_sigma and clamp_sigma > 0:
            rs = _robust_sigma(layer)
            if rs > 0:
                layer = np.clip(layer, -clamp_sigma * rs, clamp_sigma * rs)
        G_s += layer
    if include_residual:
        G_s += residual

    G_s = G_s - float(np.median(G_s))

    _say("Upscaling gradient map...")
    G = _upscale_to(G_s, (H, W)) if ds > 1 else G_s
    grad_lin = np.exp(G).astype(np.float32)   # median ~1.0

    # A gradient is low-frequency by definition: smooth the MAP (not the
    # image) before dividing so any per-pixel noise the band picked up -- which
    # gets worse as band_lo drops toward the fine, noise-dominated layers --
    # is not divided into the result. This strips noise from the CORRECTION
    # while leaving the image's own detail completely untouched, and re-centres
    # to keep the median at 1.0.
    gsm = float(gradient_smooth_px)
    if gsm > 0.0:
        _say("Smoothing gradient map...")
        grad_lin = blur(grad_lin, gsm)
        gmed = float(np.median(grad_lin))
        if gmed > 1e-8:
            grad_lin = grad_lin / gmed      # re-anchor map median to 1.0
        grad_lin = grad_lin.astype(np.float32, copy=False)

    _say("Applying multiplicative correction...")
    g = grad_lin[..., None] if (tgt.ndim == 3 and grad_lin.ndim == 2) else grad_lin
    corrected = tgt / np.clip(g, 1e-6, None)

    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1.0:
        corrected = tgt * (1.0 - s) + corrected * s

    tm = float(np.median(tgt))
    cm = float(np.median(corrected))
    if cm > 1e-8:
        corrected = corrected * (tm / cm)

    # Apply-time protect mask: keep the ORIGINAL (target) pixels wherever the
    # mask says to. The gradient ran across the whole frame; this is purely a
    # final composite, so a hand-drawn exclusion region returns untouched no
    # matter what the correction did there. The boundary is feathered over a
    # real, controllable width so the protected region blends smoothly into the
    # corrected background instead of showing a hard seam.
    if protect_blend_mask is not None:
        try:
            m = np.asarray(protect_blend_mask, dtype=np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            if m.shape[:2] != (H, W):
                m = _upscale_to(m, (H, W))
            mx = float(m.max()) if m.size else 0.0
            if mx > 1.0:
                m = m / mx
            m = np.clip(m, 0.0, 1.0)

            # Resolve feather width. 0 => auto: 10% of the image's short side,
            # e.g. ~300 px on a 4500x3000 frame. A fixed small value is
            # invisible at full resolution; the blur sigma must scale with the
            # image to actually look feathered. Floor at 16 px for tiny images.
            fpx = float(protect_feather_px)
            if fpx <= 0.0:
                fpx = max(16.0, 0.10 * float(min(H, W)))
            m = _feather_mask(m, fpx, blur)
            _say(f"Feathering exclusion edge ({int(fpx)} px)...")

            mm = m[..., None] if (corrected.ndim == 3 and m.ndim == 2) else m
            corrected = corrected * (1.0 - mm) + tgt * mm
            _say("Protected exclusion region (kept original pixels).")
        except Exception:
            pass

    corrected = corrected.astype(np.float32, copy=False)
    _say("Ready")

    if return_extras:
        try:
            mask_full = _upscale_to(sig_mask_s.astype(np.float32), (H, W)) > 0.5
        except Exception:
            mask_full = sig_mask_s
        return corrected, grad_lin, mask_full
    return corrected