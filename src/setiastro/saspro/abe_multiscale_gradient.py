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
     (k is in MAD-sigma units: 1 = aggressive protection, 5+ = only
      the brightest cores; pass math.inf or any very large value to
      disable protection entirely -- useful when the frame is
      dominated by bright halos/reflections that out-sigma any
      reasonable threshold and should be treated as gradient)
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
from concurrent.futures import ThreadPoolExecutor
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
# Signal protection: median + k*MAD  (k in MAD-sigma units, 1..5+)
# ----------------------------------------------------------------------
def build_signal_mask(luma, k=3.0, grow=3, blur_px=0.0, blur=None):
    """
    Soft signal-protection mask. Threshold at median + k*(MAD-sigma), then
    DILATE and BLUR so the protection covers each detection's full footprint
    with soft edges — not just hard-edged bright cores.

    Why both:
      - The raw threshold catches only a star's bright centre; its faint wings
        fall below and would leak into the gradient estimate. Dilating grows
        each hit outward to cover the whole stellar/structure footprint.
      - Hard mask edges make the inpaint + Gaussian pyramid RING, turning the
        mask's own boundaries into fake mid-scale "gradient". Blurring softens
        the edges so the transition is smooth and nothing rings.

    k is in MAD-sigma units: k=1 masks a lot (protects faint signal), k=5
    only the brightest cores. For data dominated by bright halos or
    reflections (anything that stays well above the sky even at k=5), pass
    math.inf — or any very large k — to disable protection entirely and
    let the gradient model see every pixel. grow is dilation iterations
    (px). blur_px is the Gaussian sigma applied after dilation; 0 keeps the
    (still boolean) mask.

    Returns a float32 array in [0, 1] (soft when blurred, else 0/1).
    """
    # Short-circuit when the caller has explicitly disabled protection
    # (k >= very large / infinite). No pixel will ever exceed med + inf*MAD,
    # so the mask would be all-zeros anyway — skipping saves the threshold,
    # dilation and blur passes, which on a 4500x3000 frame are not cheap.
    k_val = float(k)
    if not np.isfinite(k_val) or k_val >= 1e6:
        return np.zeros_like(luma, dtype=np.float32)

    med = float(np.median(luma))
    sig = _robust_sigma(luma)
    mask = luma > (med + k_val * sig)
    if grow > 0 and mask.any():
        mask = binary_dilation(mask, iterations=int(grow))
    m = mask.astype(np.float32)
    if blur_px and blur_px > 0.0 and m.any():
        if blur is not None:
            m = blur(m, float(blur_px))
        else:
            m = gaussian_filter(m, sigma=float(blur_px), mode="reflect")
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
    return m


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
    Soften a protect mask's boundary into a smooth ramp of width ~feather_px
    that ramps OUTWARD from the region edge — the interior stays at full
    strength (1.0) and only the boundary fades to 0 over the feather width.

    This matters: a plain Gaussian blur of a filled mask ERODES the interior
    once the blur radius approaches the region size (a 40 px blur on a 40 px
    disk pulls the centre down to ~0.4), so a large feather on a modest
    exclusion would leave the protected object only partly protected. A
    signed-distance ramp avoids that — the centre is always 1.0 no matter how
    wide the feather, and the ramp only extends the transition at the edge.

    feather_px <= 0 returns the mask unchanged.
    """
    m = np.clip(np.asarray(mask01, dtype=np.float32), 0.0, 1.0)
    f = float(feather_px)
    if f <= 0.0 or not (m.any() and (m < 1.0).any()):
        return m

    try:
        from scipy.ndimage import distance_transform_edt
        inside = m >= 0.5
        if inside.all() or (~inside).all():
            return m
        # Distance OUTWARD from the region edge (0 inside, growing outside).
        d_out = distance_transform_edt(~inside).astype(np.float32)
        # Ramp: 1 inside the region, falling linearly to 0 at feather_px out.
        ramp = np.clip(1.0 - d_out / f, 0.0, 1.0)
        # Keep the interior hard at 1.0, only the outside gets the ramp.
        ramp = np.where(inside, 1.0, ramp).astype(np.float32)
        # Light smoothing (relative to the feather) rounds the discrete-EDT
        # stair-steps and the smoothstep gives a C1-continuous edge.
        t = ramp
        ramp = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        ramp = blur(ramp, max(1.0, f * 0.15)).astype(np.float32)
        # The blur can nibble the very-centre a hair; re-assert interior=1.
        ramp = np.where(inside, 1.0, ramp).astype(np.float32)
        return np.clip(ramp, 0.0, 1.0)
    except Exception:
        # Fallback: blur, but then restore the interior to 1.0 so we never
        # erode the protected core.
        b = np.clip(blur(m, max(1.0, f)), 0.0, 1.0).astype(np.float32)
        return np.where(m >= 0.5, 1.0, b).astype(np.float32)


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
# Per-channel gradient estimator (shared by mono + RGB paths)
# ----------------------------------------------------------------------
def _estimate_channel_gradient(
    chan_s, sig_hole_s, blur, *,
    eps, layers, band_lo, band_hi, include_residual,
    base_sigma, clamp_sigma, out_hw, ds,
):
    """
    Estimate the multiplicative gradient map for ONE channel.

    chan_s        : 2D float32, the channel at ESTIMATE (downsampled) resolution
    sig_hole_s    : 2D bool, the signal-protection hole mask (same HxW as chan_s)
    blur          : the fast gaussian from _get_blur()
    eps           : floor added before log(); set per channel for correct scale
    out_hw        : (H, W) full output resolution to upscale the result to
    ds            : estimate_downsample (for the >1 upscale branch)

    Returns a float32 2D linear gradient map at full output resolution with
    median anchored to ~1.0, ready to divide into the channel.

    Factored out so RGB inputs can call it per channel and avoid the old
    luma-only limitation, where a colour-selective gradient (a blue halo,
    say) weighs almost nothing in the 0.21R + 0.72G + 0.07B luminance and
    therefore barely got touched by the division.
    """
    L = np.log(np.clip(chan_s, 0.0, None) + eps)

    # Fill reach must match the scale we extract the gradient at, so holes
    # are filled with smooth background at the band's scale (not just a
    # tiny local blur that leaves a bump the mid-band then reads as
    # "gradient").
    _fill_sigma = max(2.0, float(base_sigma) * (2.0 ** max(0, int(band_hi) - 2)))
    _fill_iters = int(np.clip(hole_span_iters(sig_hole_s, _fill_sigma), 8, 40)) \
        if sig_hole_s.any() else 0
    L_fill = _inpaint_fill(L, sig_hole_s, blur,
                           iters=_fill_iters, sigma=_fill_sigma)

    # Only compute layers we actually use. If the coarse residual is folded
    # in, we must run the full pyramid; otherwise stop at band_hi and skip
    # the expensive large-sigma blurs.
    stop_layer = None if include_residual else band_hi

    details, residual = _decompose_channel(
        L_fill, layers, float(base_sigma), blur, stop_layer=stop_layer)

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

    G = _upscale_to(G_s, out_hw) if ds > 1 else G_s
    return np.exp(G).astype(np.float32)   # median ~1.0


# ----------------------------------------------------------------------
# The stage
# ----------------------------------------------------------------------
def multiscale_gradient_correct(
    target_image,
    *,
    estimate_from=None,   # image the GRADIENT BAND is estimated on (starless):
                          # keeps point sources out of the mid-scale structure
    mask_from=None,       # image the PROTECTION MASK is built on (star-INCLUSIVE
                          # preferred): catches stars, star-removal residuals,
                          # and real extended signal, so none of it leaks into
                          # the estimate. Defaults to estimate_from, then target.
    layers=9,
    base_sigma=1.0,
    band_lo=6,
    band_hi=8,
    include_residual=False,
    eps_frac=0.01,
    strength=1.0,
    protect_k=3.0,          # MAD-sigma units (1..5 typical; pass math.inf or
                            # any very large value to disable protection
                            # entirely, e.g. for halo-dominated frames where
                            # no finite k catches the halos)
    protect_grow=3,         # dilation iterations: grow each detection out to
                            # cover its full footprint (star wings, not just
                            # the bright core). Set to 0 to use the raw
                            # threshold without dilation.
    protect_blur_px=4.0,    # Gaussian sigma to soften the protect mask after
                            # dilation, so hard mask edges don't make the
                            # inpaint + pyramid ring into fake gradient. 0 = off
    protect_blend_mask=None,  # bool/float HxW, True/1 = keep ORIGINAL pixels
                              # at APPLY time (e.g. a hand-drawn exclusion
                              # polygon around a galaxy). The multiscale
                              # process runs on the whole image; this is just
                              # a final composite so the protected region is
                              # taken straight from the untouched source.
    protect_feather_frac=0.10,  # exclusion-edge feather as a FRACTION of the
                                # image's short side. Resolution-independent: a
                                # given percentage feathers the same visual
                                # amount at any image size (0.10 => ~300 px on a
                                # 4500x3000 frame, ~88 px on an 880 px frame).
                                # This is the value the UI exposes as a percent.
                                # 0 disables the exclusion feather (hard edge).
    estimate_downsample=1,  # 1 = full-res (REQUIRED for sharp motes/edges);
                            # >1 only for a fast coarse preview, never for apply
    gradient_smooth_px=2.0, # final blur (px) of the gradient MAP before
                            # dividing. A gradient is low-frequency by
                            # definition, so smoothing the map removes any
                            # per-pixel noise the band picked up (worst at low
                            # band_lo) without touching the image's own detail.
                            # 0 disables; 2-3 px is a good default.
    clamp_sigma=0.0,
    channels="rgb",       # "rgb" (default): estimate per-channel so a colour-
                          # selective gradient (bright-blue halo, red LP cast,
                          # one-channel filter leak) is actually visible to
                          # and removable by the band model. "luma": estimate
                          # once from Rec.709 luminance and divide the same
                          # 2D map into all three channels — the pre-fix
                          # behaviour. Keep this for narrowband (where one
                          # channel is near-empty and per-channel noise is
                          # amplified), known-achromatic defects (dust motes,
                          # physical baffle shadows — identical in R/G/B) or
                          # when the 3x cost of per-channel matters.
    return_extras=False,
    return_debug=False,   # when True, also return the raw median+k*MAD signal
                          # mask (full-res, 0..1) as an extra element, for
                          # debugging what the protection threshold catches.
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

    # Luminance for the PROTECTION MASK, from mask_from (star-inclusive image)
    # when supplied — so stars, star-removal residuals, and real signal are all
    # caught and kept out of the estimate, independent of how well star removal
    # worked. Falls back to the starless estimate luma if no separate mask image
    # is given.
    if mask_from is not None:
        mk = np.asarray(mask_from, dtype=np.float32)
        if mk.ndim == 2:
            mask_luma_full = mk
        elif mk.ndim == 3 and mk.shape[2] == 1:
            mask_luma_full = mk[..., 0]
        else:
            mask_luma_full = (0.2126 * mk[..., 0] + 0.7152 * mk[..., 1]
                              + 0.0722 * mk[..., 2]).astype(np.float32)
        mask_luma_full = np.nan_to_num(mask_luma_full, nan=0.0, posinf=0.0,
                                       neginf=0.0).astype(np.float32)
        mask_luma_s = _downscale(mask_luma_full, ds)
    else:
        mask_luma_s = luma_s

    _say("Building signal-protection mask (median + k*MAD, dilated + blurred)...")
    # Soft float mask in [0,1]: threshold -> dilate (protect_grow) -> blur
    # (protect_blur_px). Blur scales down with the estimate downsample so the
    # softening is the same effective radius at full res.
    sig_blur = float(protect_blur_px) / float(ds) if ds > 1 else float(protect_blur_px)
    sig_mask_s = build_signal_mask(mask_luma_s, k=protect_k, grow=protect_grow,
                                   blur_px=sig_blur, blur=blur)
    # Boolean hole set (for the inpaint + iteration estimate): anything the
    # soft mask flags at all. A low threshold keeps the generous dilated
    # footprint rather than shrinking it back to the hard core.
    sig_hole_s = sig_mask_s > 0.05

    # Fold the user's hand-drawn exclusion polygons into the inpaint hole
    # set too. Previously the exclusion was only honoured as a final blend-
    # back: the gradient was still estimated with whatever structure lived
    # inside the drawn region (a galaxy, bright nebulosity, a reflection you
    # want preserved), which means the band saw that structure and divided
    # it out EVERYWHERE ELSE in the frame — contaminating the correction
    # outside the exclusion. Including it in the hole set inpaints the
    # excluded region at estimation time, so the gradient is modelled from
    # the smooth background alone and the blend-back is the only place the
    # user's region matters. Done at the ESTIMATE (downsampled) scale to
    # match sig_hole_s.
    if protect_blend_mask is not None:
        try:
            em = np.asarray(protect_blend_mask, dtype=np.float32)
            if em.ndim == 3:
                em = em[..., 0]
            emax = float(em.max()) if em.size else 0.0
            if emax > 1.0:
                em = em / emax
            em_full = np.clip(em, 0.0, 1.0)
            em_s = _downscale(em_full, ds) if ds > 1 else em_full
            if em_s.shape[:2] == sig_hole_s.shape[:2]:
                sig_hole_s = sig_hole_s | (em_s >= 0.5)
                _say("Folded exclusion polygons into gradient-estimate hole set.")
        except Exception:
            pass

    # Keep a full-res copy of the (soft) MAD mask for optional debug output.
    mad_mask_full = None
    if return_debug:
        try:
            if ds > 1:
                mad_mask_full = _upscale_to(sig_mask_s.astype(np.float32), (H, W))
            else:
                mad_mask_full = sig_mask_s.astype(np.float32)
            mad_mask_full = np.clip(mad_mask_full, 0.0, 1.0).astype(np.float32)
        except Exception:
            mad_mask_full = None

    # Clamp band range once; shared across all channels in RGB mode.
    layers = int(layers)
    band_lo = max(0, min(int(band_lo), layers - 1))
    band_hi = max(band_lo, min(int(band_hi), layers - 1))

    # Per-channel gradient estimation: previously the gradient was estimated
    # from luminance alone, which cannot see colour-selective gradients. A
    # bright-blue halo contributes only ~7% to Rec.709 luminance via the
    # 0.0722 blue weight, so the luma band stayed small and the division
    # left the halo almost untouched. Running the pipeline per channel on
    # RGB inputs gives each channel its own gradient map, so a blue halo
    # produces a strong gradient in the blue plane and gets divided out
    # there; the R and G planes see very little and are barely affected.
    # The protection mask stays single-channel (built from luma / mask_from
    # above) — "signal regions" are brightness-defined and the same mask is
    # the right one to protect for all three channels.
    est_is_rgb = (est.ndim == 3 and est.shape[2] == 3)
    tgt_is_rgb = (tgt.ndim == 3 and tgt.shape[2] == 3)

    # Normalise the channels knob: anything that isn't an explicit "luma"
    # falls back to the per-channel RGB path, which is also the only safe
    # option for a mono input (nothing to combine). "luma" on a mono input
    # is a no-op — the mono branch below already uses luma_s.
    mode = str(channels or "rgb").strip().lower()
    use_per_channel = (mode != "luma") and est_is_rgb and tgt_is_rgb

    if use_per_channel:
        _say("Multiscale decomposition (R, G, B in parallel)...")
        chans_s = []
        for c in range(3):
            cf = np.nan_to_num(est[..., c].astype(np.float32, copy=False),
                               nan=0.0, posinf=0.0, neginf=0.0)
            chans_s.append(_downscale(cf, ds))

        # Pre-compute per-channel eps on the main thread so workers only do
        # pure CPU/GPU numeric work and don't race on anything shared.
        # Per-channel floor matters on narrowband where one channel is
        # mostly empty and would misbehave under the luma eps.
        eps_per_c = [
            max(1e-6, float(np.median(chans_s[c])) * float(eps_frac))
            for c in range(3)
        ]

        def _one_channel(c):
            # Each worker owns its own numpy allocations; sig_hole_s is only
            # READ (never written) across threads, so no lock is needed.
            # cv2 is thread-safe for independent inputs. For GPU torch there
            # is some contention on the default stream — not catastrophic,
            # and CPU backends scale cleanly 2-3x.
            return _estimate_channel_gradient(
                chans_s[c], sig_hole_s, blur,
                eps=eps_per_c[c], layers=layers, band_lo=band_lo,
                band_hi=band_hi, include_residual=include_residual,
                base_sigma=base_sigma, clamp_sigma=clamp_sigma,
                out_hw=(H, W), ds=ds,
            )

        # Three workers, one per channel. map() returns results in the
        # same order as the inputs, so grads[0]=R, [1]=G, [2]=B naturally.
        # Progress callbacks from inside workers are intentionally NOT
        # emitted — a GUI progress_cb may not be thread-safe. The single
        # aggregate message above is enough.
        with ThreadPoolExecutor(max_workers=3) as _pool:
            grads = list(_pool.map(_one_channel, range(3)))
        grad_lin = np.stack(grads, axis=-1).astype(np.float32, copy=False)  # (H, W, 3)
        _say("Per-channel gradients ready.")
    else:
        # Mono path: single channel estimated from the (luma) estimate image.
        # Reached either for genuinely mono input or when the caller explicitly
        # requested channels="luma" on RGB (e.g. narrowband data, achromatic
        # defects, or a speed-sensitive preview).
        if est_is_rgb:
            _say("Multiscale decomposition (luma only — opt-out of per-channel)...")
        else:
            _say("Multiscale decomposition (log, torch/cv2 blur)...")
        grad_lin = _estimate_channel_gradient(
            luma_s, sig_hole_s, blur,
            eps=eps, layers=layers, band_lo=band_lo, band_hi=band_hi,
            include_residual=include_residual, base_sigma=base_sigma,
            clamp_sigma=clamp_sigma, out_hw=(H, W), ds=ds,
        )

    # A gradient is low-frequency by definition: smooth the MAP (not the
    # image) before dividing so any per-pixel noise the band picked up -- which
    # gets worse as band_lo drops toward the fine, noise-dominated layers --
    # is not divided into the result. This strips noise from the CORRECTION
    # while leaving the image's own detail completely untouched, and re-centres
    # to keep the median at 1.0.
    gsm = float(gradient_smooth_px)
    if gsm > 0.0:
        _say("Smoothing gradient map...")
        # Handle both the mono (2D) and per-channel RGB (3D) cases. Smoothing
        # and re-anchoring happen PER PLANE so each channel's gradient median
        # stays at 1.0 — if we re-anchored by the combined median instead, a
        # bright-blue halo (large blue gradient, modest red/green) would shift
        # the global median and silently alter the colour balance.
        if grad_lin.ndim == 3:
            for c in range(grad_lin.shape[2]):
                gc = blur(grad_lin[..., c], gsm)
                gmed = float(np.median(gc))
                if gmed > 1e-8:
                    gc = gc / gmed
                grad_lin[..., c] = gc.astype(np.float32)
        else:
            grad_lin = blur(grad_lin, gsm)
            gmed = float(np.median(grad_lin))
            if gmed > 1e-8:
                grad_lin = grad_lin / gmed  # re-anchor map median to 1.0
            grad_lin = grad_lin.astype(np.float32, copy=False)

    _say("Applying multiplicative correction...")
    g = grad_lin[..., None] if (tgt.ndim == 3 and grad_lin.ndim == 2) else grad_lin
    corrected = tgt / np.clip(g, 1e-6, None)

    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1.0:
        corrected = tgt * (1.0 - s) + corrected * s

    # Anchor each channel's output median back to its own input median.
    # Per-channel (not combined) because the division is per-channel: a
    # single scalar re-anchor would reintroduce any colour-balance drift
    # the division corrected, undoing exactly what the per-channel fix is
    # supposed to achieve. For mono this reduces to the previous behaviour.
    if (corrected.ndim == 3 and tgt.ndim == 3
            and corrected.shape[-1] == tgt.shape[-1]):
        for c in range(corrected.shape[-1]):
            tmc = float(np.median(tgt[..., c]))
            cmc = float(np.median(corrected[..., c]))
            if cmc > 1e-8:
                corrected[..., c] = corrected[..., c] * (tmc / cmc)
    else:
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

            # Resolve feather width from the fraction of the image's short side.
            # Resolution-independent: a given percentage feathers the same
            # visual amount at any image size (a fixed pixel value would be
            # invisible at full resolution). 0 => hard edge; otherwise floor at
            # a few px so a tiny percentage on a small image still softens.
            frac = max(0.0, float(protect_feather_frac))
            if frac <= 0.0:
                fpx = 0.0
            else:
                fpx = max(4.0, frac * float(min(H, W)))
            if fpx > 0.0:
                m = _feather_mask(m, fpx, blur)
                _say(f"Feathering exclusion edge ({frac*100:.0f}% = {int(fpx)} px)...")

            mm = m[..., None] if (corrected.ndim == 3 and m.ndim == 2) else m
            corrected = corrected * (1.0 - mm) + tgt * mm

            # Expose the resolved, feathered protect mask (full-res, 2D, 0..1)
            # so the caller can fold it into the structure map it displays —
            # inside the exclusion the effective gradient was the median (no
            # correction), and the structure document should show that rather
            # than the raw estimate over a protected object.
            self_feathered = m
            _say("Protected exclusion region (kept original pixels).")
        except Exception:
            self_feathered = None
    else:
        self_feathered = None

    corrected = corrected.astype(np.float32, copy=False)
    grad_lin = grad_lin.astype(np.float32, copy=False)
    _say("Ready")

    if return_extras:
        # mask_full: the feathered protect mask actually applied (2D, 0..1) if
        # there was one, else the raw signal-protection mask as a fallback.
        if self_feathered is not None:
            mask_full = np.clip(np.asarray(self_feathered, dtype=np.float32), 0.0, 1.0)
        else:
            try:
                mask_full = (_upscale_to(sig_mask_s.astype(np.float32), (H, W)) > 0.5
                             ).astype(np.float32)
            except Exception:
                mask_full = sig_mask_s.astype(np.float32)
        if return_debug:
            return corrected, grad_lin, mask_full, mad_mask_full
        return corrected, grad_lin, mask_full
    if return_debug:
        return corrected, mad_mask_full
    return corrected