#src/setiastro/saspro/starless_engines/syqon_nafnet_engine.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Callable, Optional, Tuple, Any
import os

ProgressCB = Callable[[int, int, str], None]  # (done, total, stage)

# module-global cached session for comet stacking (or any repeated use)
_SYQON_SESSION = None
_SYQON_CKPT = None

def _infer_device(torch, *, prefer_cuda: bool = True, prefer_dml: bool = True):
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_dml and (os.name == "nt"):
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def _looks_like_decode_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        "codec can't decode" in msg
        or "can't decode byte" in msg
        or "invalid continuation byte" in msg
        or "unicode" in msg
        or "decode" in msg
    )

def _torch_load_once(torch, ckpt_path: str, *, encoding=None):
    kwargs = {"map_location": "cpu"}
    if encoding is not None:
        kwargs["encoding"] = encoding

    with open(ckpt_path, "rb") as f:
        try:
            return torch.load(f, weights_only=False, **kwargs)
        except TypeError as e:
            if "weights_only" not in str(e):
                raise
            f.seek(0)
            return torch.load(f, **kwargs)

def _torch_load_fallback(torch, ckpt_path: str):
    last_exc = None

    try:
        return _torch_load_once(torch, ckpt_path), {"torch_load_encoding": "default"}
    except Exception as e:
        last_exc = e
        if not _looks_like_decode_error(e):
            raise

    for enc in ("latin1", "cp1252", "iso-8859-1"):
        try:
            return _torch_load_once(torch, ckpt_path, encoding=enc), {"torch_load_encoding": enc}
        except Exception as e:
            last_exc = e
            if "unexpected keyword argument" in str(e).lower() and "encoding" in str(e).lower():
                break
            if not _looks_like_decode_error(e):
                raise

    raise last_exc


def _detect_arch(sd: dict) -> str:
    """
    Returns 'axiomv22', 'lite', or 'standard'.

    'axiomv22' — AxiomV2.2 UNet generator: g_conv / g_deconv keys
    'lite'     — NAFNetLite (AxiomV2.1):   fusions / conv3 keys
    'standard' — NAFNet (Nadir / AxiomV2): ffn1 / sca.1 keys
    """
    if any(k.startswith("g_conv") or k.startswith("g_deconv") for k in sd):
        return "axiomv22"
    for k in sd:
        if k.startswith("fusions.") or k.endswith(".conv3.weight"):
            return "lite"
    for k in sd:
        if k.endswith(".ffn1.weight") or k.endswith(".sca.1.weight"):
            return "standard"
    for k, v in sd.items():
        if ".norm1.weight" in k and hasattr(v, "ndim"):
            return "lite" if v.ndim == 1 else "standard"
    return "standard"


def _load_state_dict(torch, ckpt_path: str):
    ckpt, load_meta = _torch_load_fallback(torch, ckpt_path)

    def _small_meta(src):
        out = dict(load_meta)
        if isinstance(src, dict):
            out["checkpoint_keys"] = list(src.keys())[:20]
            for k in ("epoch", "best_val", "residual_output", "psnr", "step", "best_ema"):
                if k in src:
                    try:
                        out[k] = float(src[k]) if not isinstance(src[k], (str, dict, list)) else str(src[k])
                    except Exception:
                        pass
            if "args" in src and isinstance(src["args"], dict):
                for ak in ("base_ch", "depth", "amp"):
                    if ak in src["args"]:
                        out[f"args_{ak}"] = src["args"][ak]
        return out

    if isinstance(ckpt, dict):
        if "generator" in ckpt and isinstance(ckpt["generator"], dict):
            sd = ckpt["generator"]
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            meta = _small_meta(ckpt)
            meta["source_key"] = "generator"
            return sd, meta

        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"], _small_meta(ckpt)

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], _small_meta(ckpt)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"], _small_meta(ckpt)

        if any(
            k.startswith(("intro.", "ending.", "encoders.", "downs.", "middle.",
                          "decoders.", "ups.", "g_conv", "g_deconv"))
            for k in ckpt.keys()
        ):
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
            return sd, dict(load_meta)

    if isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format. Top-level keys: {list(ckpt.keys())[:20]}"
        )
    raise RuntimeError(
        f"Unsupported checkpoint format: top-level object is {type(ckpt).__name__}"
    )


def _detect_ups_variant(sd: dict) -> str:
    for k in sd.keys():
        if k.startswith("ups.") and k.endswith(".1.weight"):
            return "bilinear"
        if k.startswith("ups.") and k.endswith(".0.weight"):
            return "pixelshuffle"
    return "bilinear"


def _infer_starnet_channels(sd: dict) -> int:
    if "g_conv0.weight" in sd:
        return int(sd["g_conv0.weight"].shape[1])
    return 3


def _infer_nafnet_cfg_from_sd(sd: dict) -> tuple[int, tuple, tuple, int]:
    base_ch = int(sd["intro.weight"].shape[0]) if "intro.weight" in sd else 32

    downs_idx = set()
    for k in sd.keys():
        if k.startswith("downs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                downs_idx.add(int(parts[1]))

    if downs_idx:
        enc_levels = max(downs_idx) + 1
    else:
        enc_idx = set()
        for k in sd.keys():
            if k.startswith("encoders."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    enc_idx.add(int(parts[1]))
        enc_levels = (max(enc_idx) + 1) if enc_idx else 4

    enc_levels = max(1, min(8, int(enc_levels)))

    enc_blks = []
    for lvl in range(enc_levels):
        blk_idx = set()
        for k in sd.keys():
            if k.startswith(f"encoders.{lvl}.") and ".norm1.weight" in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    blk_idx.add(int(parts[2]))
        enc_blks.append((max(blk_idx) + 1) if blk_idx else 2)

    dec_blks = []
    for lvl in range(enc_levels):
        blk_idx = set()
        for k in sd.keys():
            if k.startswith(f"decoders.{lvl}.") and ".norm1.weight" in k:
                parts = k.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    blk_idx.add(int(parts[2]))
        dec_blks.append((max(blk_idx) + 1) if blk_idx else 2)

    mid_idx = set()
    for k in sd.keys():
        if k.startswith("middle.") and ".norm1.weight" in k:
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                mid_idx.add(int(parts[1]))
    middle_blk_num = (max(mid_idx) + 1) if mid_idx else 2

    return base_ch, tuple(enc_blks), tuple(dec_blks), middle_blk_num


def load_nafnet_model(
    ckpt_path: str,
    *,
    use_gpu: bool,
    prefer_dml: bool,
    model_kind: str = "nadir",
):
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_model.model import NAFNet, NAFNetLite, StarNetGenerator

    torch = import_torch(
        prefer_cuda=use_gpu,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=lambda *_: None,
    )

    sd, meta = _load_state_dict(torch, ckpt_path)
    arch = _detect_arch(sd)

    if arch == "axiomv22":
        in_channels = _infer_starnet_channels(sd)
        model = StarNetGenerator(in_channels=in_channels)
        model.load_state_dict(sd, strict=True)
        model.eval()
        info = {
            "model_kind":  model_kind,
            "arch":        "axiomv22",
            "in_channels": in_channels,
            "meta":        meta,
        }

    else:
        ups_variant = _detect_ups_variant(sd)
        base_ch, enc_blks, dec_blks, middle_blk_num = _infer_nafnet_cfg_from_sd(sd)

        if arch == "lite":
            model = NAFNetLite(
                width=base_ch,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
                middle_blk_num=middle_blk_num,
            )
        else:
            model = NAFNet(
                width=base_ch,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
                middle_blk_num=middle_blk_num,
                use_sigmoid=False,
                ups_mode=ups_variant,
            )

        model.load_state_dict(sd, strict=True)
        model.eval()

        info = {
            "model_kind":      model_kind,
            "arch":            arch,
            "block_variant":   arch,
            "ups_variant":     ups_variant,
            "base_ch":         base_ch,
            "enc_blks":        enc_blks,
            "dec_blks":        dec_blks,
            "middle_blk_num":  middle_blk_num,
            "meta":            meta,
        }

    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    model.to(device)

    info["device"]        = str(device)
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_file"]    = getattr(torch, "__file__", None)

    return model, device, info, torch


def _to_torch_chw(img_chw, device, torch):
    x = np.asarray(img_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected CHW float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(x[None, ...])
    return t.to(device=device, dtype=torch.float32)


def _pad_to_multiple(t, multiple=8, torch=None):
    """Pad tensor (1, C, H, W) to next multiple on H and W dims."""
    _, _, H, W = t.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return t, H, W
    import torch.nn.functional as F
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, H, W


def _build_blend_weights(tile: int, stride: int, arch: str) -> np.ndarray:
    """
    Return a (tile, tile, 1) float32 blend weight map.

    AxiomV2.2 ('axiomv22'): cosine ramp — matches SyQon training tiling exactly.
    NAFNetLite / NAFNet   : Hanning window — unchanged from original behaviour.
    """
    if arch == "axiomv22":
        overlap     = tile - stride
        blend       = max(1, overlap)
        coords      = np.arange(tile, dtype=np.float32)
        border_dist = np.minimum(coords, coords[::-1])
        t_ramp      = np.clip(border_dist / float(blend), 0.0, 1.0)
        ramp        = (0.5 - 0.5 * np.cos(np.pi * t_ramp)).astype(np.float32)
        ramp        = np.clip(ramp, 1e-3, 1.0)
        w2          = (ramp[:, None] * ramp[None, :]).astype(np.float32)[..., None]
    else:
        wy = np.hanning(tile).astype(np.float32)
        wx = np.hanning(tile).astype(np.float32)
        w2 = (wy[:, None] * wx[None, :]).astype(np.float32)[..., None]
    return w2


def _predict_tile(model, t, *, device, use_amp, amp_dtype, info, torch):
    """
    Run one tile through the model.

    AxiomV2.2 ('axiomv22'):
      - Input normalised [0,1] → [-1,1] before the model (trained range).
      - Output denormalised [-1,1] → [0,1] after the model.

    NAFNetLite / NAFNet:
      - Input and output remain in [0,1].
    """
    arch        = info.get("arch", "standard")
    is_axiomv22 = (arch == "axiomv22")

    t_input = (t * 2.0 - 1.0) if is_axiomv22 else t
    t_padded, orig_H, orig_W = _pad_to_multiple(t_input, multiple=8, torch=torch)

    def _run(inp):
        return model(inp)

    def _to_numpy(pred_t):
        pred_np = pred_t[0].detach().to("cpu").numpy().transpose(1, 2, 0)
        if is_axiomv22:
            pred_np = (pred_np + 1.0) * 0.5
        pred_np = pred_np[:orig_H, :orig_W, :]
        return pred_np.astype(np.float32, copy=False)

    if use_amp and device.type == "cuda":
        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            pred_t = _run(t_padded)
        pred = _to_numpy(pred_t)
        if not np.isfinite(pred).all():
            info["amp_fallback"] = True
            info["amp_reason"] = "non-finite output under CUDA AMP; reran tile in fp32"
            pred_t = _run(t_padded)
            pred = _to_numpy(pred_t)
        return pred

    if use_amp and device.type == "mps":
        try:
            with torch.autocast(device_type="mps"):
                pred_t = _run(t_padded)
            pred = _to_numpy(pred_t)
            if not np.isfinite(pred).all():
                info["amp_fallback"] = True
                info["amp_reason"] = "non-finite output under MPS autocast; reran tile in fp32"
                pred_t = _run(t_padded)
                pred = _to_numpy(pred_t)
            return pred
        except Exception:
            pred_t = _run(t_padded)
            return _to_numpy(pred_t)

    pred_t = _run(t_padded)
    pred = _to_numpy(pred_t)
    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite output detected in fp32 inference.")
    return pred


def _resolve_residual_mode(info: dict) -> bool:
    """
    Determine whether the model output is a star residual (True) or starless directly (False).

    AxiomV2.2 ('axiomv22'): output IS the final blended starless image, so False.
    NAFNetLite ('lite'):    output IS starless (global residual clamped), so False.
    NAFNet bilinear:        output IS starless, so False.
    NAFNet pixelshuffle:    output is the star residual, so True.
    """
    arch = info.get("arch", "standard")
    if arch == "axiomv22":
        return False
    if arch == "lite":
        return False
    ups = info.get("ups_variant", "bilinear")
    return (ups == "pixelshuffle")


# =============================================================================
# Tiled inference helpers
# =============================================================================

def _run_tiled_rgb(
    model, x: np.ndarray, *,
    tile: int, stride: int, arch: str,
    device, torch,
    use_amp: bool, amp_dtype: str,
    info: dict,
    residual_mode: bool,
    progress_cb: Optional[Callable] = None,
    progress_offset: int = 0,
    progress_total: int = 1,
    tile_cb=None,
    label: str = "",
) -> np.ndarray:
    """Full RGB tiled inference pass. Returns (H, W, 3) float32."""
    H, W  = x.shape[:2]
    w2    = _build_blend_weights(tile, stride, arch)
    ys    = list(range(0, H, stride))
    xs    = list(range(0, W, stride))

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)
    done    = 0

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)

                patch = x[y0:y1, x0:x1, :]
                ph, pw = patch.shape[:2]

                if ph != tile or pw != tile:
                    pad = np.zeros((tile, tile, 3), dtype=np.float32)
                    pad[:ph, :pw, :] = patch
                    patch = pad

                chw = patch.transpose(2, 0, 1)
                t   = _to_torch_chw(chw, device, torch)

                pred = _predict_tile(
                    model, t,
                    device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                    info=info, torch=torch,
                )

                starless_patch = (patch - pred) if residual_mode else pred
                starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                starless_patch = starless_patch[:ph, :pw, :]
                wlocal         = w2[:ph, :pw, :]

                out_acc[y0:y1, x0:x1, :] += starless_patch * wlocal
                w_acc  [y0:y1, x0:x1, :] += wlocal

                if callable(tile_cb):
                    tile_cb(y0, x0, ph, pw, starless_patch)

                done += 1
                if callable(progress_cb):
                    stage = f"{label} SyQon tiles…" if label else "SyQon tiles…"
                    progress_cb(progress_offset + done, progress_total, stage)

    return np.clip(out_acc / np.maximum(w_acc, 1e-8), 0.0, 1.0).astype(np.float32)


def _run_tiled_per_channel(
    model, x: np.ndarray, *,
    tile: int, stride: int, arch: str,
    device, torch,
    use_amp: bool, amp_dtype: str,
    info: dict,
    residual_mode: bool,
    progress_cb: Optional[Callable] = None,
    progress_offset: int = 0,
    progress_total: int = 1,
) -> np.ndarray:
    """
    Per-channel tiled inference. Each channel replicated to 3-ch input,
    channel 0 of output taken, recombined. Returns (H, W, 3) float32.
    """
    H, W    = x.shape[:2]
    w2_mono = _build_blend_weights(tile, stride, arch)[..., 0]
    ys      = list(range(0, H, stride))
    xs      = list(range(0, W, stride))
    total   = len(ys) * len(xs)

    starless_channels = []
    patch_buf = np.zeros((tile, tile, 3), dtype=np.float32)

    for c, ch_name in enumerate(("R", "G", "B")):
        out_acc = np.zeros((H, W), dtype=np.float32)
        w_acc   = np.zeros((H, W), dtype=np.float32)
        done    = 0

        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + tile, H)
                    x1 = min(x0 + tile, W)

                    patch_ch = x[y0:y1, x0:x1, c]
                    ph, pw   = patch_ch.shape

                    patch_buf.fill(0.0)
                    patch_buf[:ph, :pw, 0] = patch_ch
                    patch_buf[:ph, :pw, 1] = patch_ch
                    patch_buf[:ph, :pw, 2] = patch_ch

                    chw = patch_buf.transpose(2, 0, 1)
                    t   = _to_torch_chw(chw, device, torch)

                    pred = _predict_tile(
                        model, t,
                        device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                        info=info, torch=torch,
                    )

                    # Take channel 0 — all 3 are identical for replicated mono input
                    pred_ch        = pred[:, :, 0]
                    patch_ch_full  = patch_buf[:ph, :pw, 0]
                    starless_patch = (patch_ch_full - pred_ch[:ph, :pw]) if residual_mode else pred_ch[:ph, :pw]
                    starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                    wlocal         = w2_mono[:ph, :pw]

                    out_acc[y0:y1, x0:x1] += starless_patch * wlocal
                    w_acc  [y0:y1, x0:x1] += wlocal

                    done += 1
                    if callable(progress_cb):
                        progress_cb(
                            progress_offset + c * total + done,
                            progress_total,
                            f"[{ch_name}] SyQon tiles…",
                        )

        starless_channels.append(
            np.clip(out_acc / np.maximum(w_acc, 1e-8), 0.0, 1.0).astype(np.float32)
        )

    return np.stack(starless_channels, axis=-1)


def _lab_chroma_correction(
    original_rgb01: np.ndarray,
    starless_rgb01: np.ndarray,
    *,
    strength: float = 0.65,
    sat_threshold: float = 0.98,
    dark_threshold: float = 0.05,
    star_diff_threshold: float = 0.15,
    blur_sigma: float = 12.0,
) -> np.ndarray:
    """
    Masked chroma transfer from original to starless in Lab space.
    Keeps L from starless, corrects a,b distribution to match original.
    Returns corrected starless float32 RGB [0,1].
    """
    try:
        from skimage import color as skcolor
        import scipy.ndimage as ndi
    except ImportError:
        print("[chroma_correction] skimage/scipy not available — skipping correction.")
        return starless_rgb01

    orig = np.clip(np.asarray(original_rgb01, dtype=np.float32), 0.0, 1.0)
    star = np.clip(np.asarray(starless_rgb01, dtype=np.float32), 0.0, 1.0)

    lum_orig = orig.mean(axis=2)
    lum_star = star.mean(axis=2)

    sat_mask  = (orig.max(axis=2) < sat_threshold)
    dark_mask = (lum_star > dark_threshold)
    star_mask = (np.abs(lum_orig - lum_star) < star_diff_threshold)
    valid     = sat_mask & dark_mask & star_mask

    if valid.sum() < 100:
        print("[chroma_correction] Not enough valid pixels — skipping.")
        return star

    orig_lab = skcolor.rgb2lab(orig)
    star_lab = skcolor.rgb2lab(star)

    def _masked_stats(ch):
        vals = ch[valid]
        return float(vals.mean()), float(vals.std()) + 1e-8

    mean_a_orig, std_a_orig = _masked_stats(orig_lab[..., 1])
    mean_b_orig, std_b_orig = _masked_stats(orig_lab[..., 2])
    mean_a_star, std_a_star = _masked_stats(star_lab[..., 1])
    mean_b_star, std_b_star = _masked_stats(star_lab[..., 2])

    print(f"[chroma_correction] a: orig={mean_a_orig:.3f}±{std_a_orig:.3f}  "
          f"star={mean_a_star:.3f}±{std_a_star:.3f}")
    print(f"[chroma_correction] b: orig={mean_b_orig:.3f}±{std_b_orig:.3f}  "
          f"star={mean_b_star:.3f}±{std_b_star:.3f}")
    print(f"[chroma_correction] valid pixels: {valid.sum()} / {valid.size}")

    corrected_lab = star_lab.copy()
    corrected_lab[..., 1] = (star_lab[..., 1] - mean_a_star) / std_a_star * std_a_orig + mean_a_orig
    corrected_lab[..., 2] = (star_lab[..., 2] - mean_b_star) / std_b_star * std_b_orig + mean_b_orig
    corrected_lab[..., 0] = star_lab[..., 0]   # keep L from starless

    corrected_rgb = np.clip(skcolor.lab2rgb(corrected_lab).astype(np.float32), 0.0, 1.0)

    soft_mask = ndi.gaussian_filter(valid.astype(np.float32), sigma=blur_sigma)[..., None]
    soft_mask = np.clip(soft_mask * strength, 0.0, 1.0)

    return np.clip(star * (1.0 - soft_mask) + corrected_rgb * soft_mask, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Public inference entry points
# =============================================================================

def nafnet_starless_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    prefer_cuda: bool = True,
    residual_mode: bool = True,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    model_kind: str = "nadir",
    use_gpu: bool = True,
    prefer_dml: bool = True,
    channel_mode: str = "rgb",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    tile_cb=None,
):
    """
    channel_mode:
      'rgb'         — standard single RGB pass (default)
      'rgb+perchan' — full RGB pass + per-channel R/G/B pass averaged,
                      then Lab chroma correction applied (works for all archs)
    """
    channel_mode = (channel_mode or "rgb").strip().lower()
    if channel_mode not in ("rgb", "rgb+perchan"):
        channel_mode = "rgb"

    x = np.asarray(img_rgb01, dtype=np.float32)
    was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3]

    H, W    = x.shape[:2]
    tile    = int(tile)
    overlap = int(overlap)
    stride  = max(tile - overlap, 1)

    model, device, info, torch = load_nafnet_model(
        ckpt_path, use_gpu=use_gpu, prefer_dml=prefer_dml, model_kind=model_kind,
    )

    residual_mode = _resolve_residual_mode(info)
    arch          = info.get("arch", "standard")

    info = dict(info)
    info["residual_mode"]  = residual_mode
    info["device"]         = str(device)
    info["torch_version"]  = getattr(torch, "__version__", None)
    info["torch_file"]     = getattr(torch, "__file__", None)
    info["channel_mode"]   = channel_mode

    use_amp_requested = bool(use_amp)
    amp_dtype = (amp_dtype or "fp16").lower()
    if amp_dtype not in ("fp16", "bf16"):
        amp_dtype = "fp16"
    use_amp_effective = bool(use_amp_requested) and (device.type in ("cuda", "mps"))

    info["use_amp_requested"] = use_amp_requested
    info["use_amp_effective"] = use_amp_effective
    info["amp_dtype"]         = amp_dtype

    ys          = list(range(0, H, stride))
    xs_         = list(range(0, W, stride))
    tiles_per   = len(ys) * len(xs_)

    shared = dict(
        tile=tile, stride=stride, arch=arch,
        device=device, torch=torch,
        use_amp=use_amp_effective, amp_dtype=amp_dtype,
        info=info, residual_mode=residual_mode,
    )

    if channel_mode == "rgb+perchan":
        grand_total = 4 * tiles_per

        starless_rgb = _run_tiled_rgb(
            model, x, **shared,
            progress_cb=progress_cb, progress_offset=0,
            progress_total=grand_total, tile_cb=tile_cb, label="[RGB]",
        )

        starless_per_ch = _run_tiled_per_channel(
            model, x, **shared,
            progress_cb=progress_cb, progress_offset=tiles_per,
            progress_total=grand_total,
        )

        if callable(progress_cb):
            progress_cb(grand_total, grand_total, "Averaging RGB and per-channel results…")

        starless_avg = np.clip(0.5 * starless_rgb + 0.5 * starless_per_ch, 0.0, 1.0).astype(np.float32)

        if callable(progress_cb):
            progress_cb(grand_total, grand_total, "Applying Lab chroma correction…")

        starless = _lab_chroma_correction(x, starless_avg)
        info["chroma_correction"] = True

    else:
        grand_total = tiles_per
        starless = _run_tiled_rgb(
            model, x, **shared,
            progress_cb=progress_cb, progress_offset=0,
            progress_total=grand_total, tile_cb=tile_cb, label="",
        )
        info["chroma_correction"] = False

    stars_only = np.clip(x - starless, 0.0, 1.0).astype(np.float32)

    if was_mono:
        return (
            starless.mean(axis=2).astype(np.float32, copy=False),
            stars_only.mean(axis=2).astype(np.float32, copy=False),
            info,
        )
    return starless, stars_only, info


def _get_setting_any(settings, keys, default):
    for k in keys:
        try:
            v = settings.value(k, None)
            if v is not None and v != "":
                return v
        except Exception:
            pass
        try:
            if k in settings and settings[k] not in (None, ""):
                return settings[k]
        except Exception:
            pass
    return default


def syqon_starless_from_array(
    img: np.ndarray,
    settings,
    *,
    ckpt_key: str = "stacking/comet_starrem/syqon_ckpt",
    prefer_dml_key: str = "stacking/comet_starrem/syqon_prefer_dml",
    use_gpu_key: str = "stacking/comet_starrem/syqon_use_gpu",
    tile_key: str = "stacking/comet_starrem/syqon_tile",
    overlap_key: str = "stacking/comet_starrem/syqon_overlap",
    amp_key: str = "stacking/comet_starrem/syqon_amp",
    amp_dtype_key: str = "stacking/comet_starrem/syqon_amp_dtype",
    progress_cb: ProgressCB | None = None,
    residual_mode: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    global _SYQON_SESSION, _SYQON_CKPT

    ckpt = str(_get_setting_any(settings, (ckpt_key, "syqon/ckpt_path"), ""))
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError(f"SyQon checkpoint path is not configured or missing: '{ckpt}'")

    prefer_dml = bool(_get_setting_any(settings, (prefer_dml_key,), True))
    use_gpu    = bool(_get_setting_any(settings, (use_gpu_key,),    True))

    if _SYQON_SESSION is None or _SYQON_CKPT != ckpt:
        _SYQON_SESSION = syqonnafnetSession(ckpt, use_gpu=use_gpu, prefer_dml=prefer_dml)
        _SYQON_CKPT = ckpt

    tile      = int(_get_setting_any(settings, (tile_key,),     512))
    ov        = int(_get_setting_any(settings, (overlap_key,),  64))
    amp       = bool(_get_setting_any(settings, (amp_key,),     False))
    amp_dtype = str(_get_setting_any(settings, (amp_dtype_key,), "fp16"))

    if img.ndim == 2:
        src = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        src = np.repeat(img, 3, axis=2)
    else:
        src = img
    src = src.astype(np.float32, copy=False)

    starless, stars, info = _SYQON_SESSION.run_starless_rgb01(
        src, tile=tile, overlap=ov, use_amp=amp, amp_dtype=amp_dtype,
        progress_cb=progress_cb,
    )
    return starless, stars, info


class syqonnafnetSession:
    def __init__(self, ckpt_path: str, *, use_gpu: bool, prefer_dml: bool, model_kind: str = "nadir"):
        self.ckpt_path  = ckpt_path
        self.model_kind = (model_kind or "nadir").lower().strip()
        self.model, self.device, self.info, self.torch = load_nafnet_model(
            ckpt_path, use_gpu=use_gpu, prefer_dml=prefer_dml, model_kind=self.model_kind
        )
        self._residual_mode = _resolve_residual_mode(self.info)
        self._arch          = self.info.get("arch", "standard")

    def run_starless_rgb01(
        self,
        img_rgb01: np.ndarray,
        *,
        tile: int = 512,
        overlap: int = 64,
        residual_mode: bool = True,
        use_amp: bool = False,
        amp_dtype: str = "fp16",
        channel_mode: str = "rgb",
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ):
        channel_mode  = (channel_mode or "rgb").strip().lower()
        residual_mode = self._residual_mode

        x = np.asarray(img_rgb01, dtype=np.float32)
        was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        elif x.ndim == 3 and x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        else:
            x = x[..., :3]

        tile    = int(tile)
        overlap = int(overlap)
        stride  = max(tile - overlap, 1)

        info = dict(self.info or {})
        info["device"]        = str(self.device)
        info["residual_mode"] = residual_mode
        info["channel_mode"]  = channel_mode

        use_amp_requested = bool(use_amp)
        amp_dtype = (amp_dtype or "fp16").lower()
        if amp_dtype not in ("fp16", "bf16"):
            amp_dtype = "fp16"
        use_amp_effective = bool(use_amp_requested) and (self.device.type in ("cuda", "mps"))

        info["use_amp_requested"] = use_amp_requested
        info["use_amp_effective"] = use_amp_effective
        info["amp_dtype"]         = amp_dtype

        H, W      = x.shape[:2]
        ys        = list(range(0, H, stride))
        xs_       = list(range(0, W, stride))
        tiles_per = len(ys) * len(xs_)

        shared = dict(
            tile=tile, stride=stride, arch=self._arch,
            device=self.device, torch=self.torch,
            use_amp=use_amp_effective, amp_dtype=amp_dtype,
            info=info, residual_mode=residual_mode,
        )

        if channel_mode == "rgb+perchan":
            grand_total = 4 * tiles_per

            starless_rgb = _run_tiled_rgb(
                self.model, x, **shared,
                progress_cb=progress_cb, progress_offset=0,
                progress_total=grand_total, label="[RGB]",
            )
            starless_per_ch = _run_tiled_per_channel(
                self.model, x, **shared,
                progress_cb=progress_cb, progress_offset=tiles_per,
                progress_total=grand_total,
            )

            if callable(progress_cb):
                progress_cb(grand_total, grand_total, "Averaging RGB and per-channel results…")

            starless_avg = np.clip(0.5 * starless_rgb + 0.5 * starless_per_ch, 0.0, 1.0).astype(np.float32)

            if callable(progress_cb):
                progress_cb(grand_total, grand_total, "Applying Lab chroma correction…")

            starless = _lab_chroma_correction(x, starless_avg)
            info["chroma_correction"] = True

        else:
            grand_total = tiles_per
            starless = _run_tiled_rgb(
                self.model, x, **shared,
                progress_cb=progress_cb, progress_offset=0,
                progress_total=grand_total, label="",
            )
            info["chroma_correction"] = False

        stars_only = np.clip(x - starless, 0.0, 1.0).astype(np.float32)

        if was_mono:
            return (
                starless.mean(axis=2).astype(np.float32, copy=False),
                stars_only.mean(axis=2).astype(np.float32, copy=False),
                info,
            )
        return starless, stars_only, info


def clear_axiom_models_cache(*, aggressive: bool = False, status_cb=print) -> None:
    global _SYQON_SESSION, _SYQON_CKPT

    try:
        had_session = _SYQON_SESSION is not None
        had_ckpt    = _SYQON_CKPT    is not None
        _SYQON_SESSION = None
        _SYQON_CKPT    = None
        status_cb(
            f"[SyQon] Cleared cache "
            f"(session={'yes' if had_session else 'no'}, ckpt={'yes' if had_ckpt else 'no'})"
        )
    except Exception as e:
        try:
            status_cb(f"[SyQon] Cache clear failed: {type(e).__name__}: {e}")
        except Exception:
            pass

    if not aggressive:
        return

    try:
        import gc
        gc.collect()
        status_cb("[SyQon] gc.collect() called")
    except Exception:
        pass

    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(
            prefer_cuda=True, prefer_xpu=False, prefer_dml=True,
            status_cb=lambda *_: None,
        )
        try:
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                status_cb("[SyQon] torch.cuda.empty_cache() called")
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    status_cb("[SyQon] torch.mps.empty_cache() called")
        except Exception:
            pass
    except Exception:
        pass