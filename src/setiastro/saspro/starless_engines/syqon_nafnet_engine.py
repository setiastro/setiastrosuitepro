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
    # CUDA
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")

    # DirectML (Windows)
    if prefer_dml and (os.name == "nt"):
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass

    # MPS (macOS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def _torch_load_fallback(torch, ckpt_path: str):
    """
    More robust torch.load() with encoding fallbacks for python2-pickled checkpoints.
    """
    # 1) First try: normal load (fast path)
    try:
        with open(ckpt_path, "rb") as f:
            return torch.load(f, map_location="cpu"), {"torch_load_encoding": "default"}
    except UnicodeDecodeError as e0:
        last = e0
    except Exception:
        # if it fails for non-unicode reasons, let caller handle it
        raise

    # 2) Encoding fallbacks (common for old pickles / non-utf8 metadata)
    for enc in ("latin1", "cp1252", "iso-8859-1"):
        try:
            with open(ckpt_path, "rb") as f:
                return torch.load(f, map_location="cpu", encoding=enc), {"torch_load_encoding": enc}
        except UnicodeDecodeError as e:
            last = e
            continue
        except TypeError:
            # Some torch builds may not accept encoding=...; if so, break out and re-raise last
            break

    # 3) If we got here, we still couldn’t decode the pickle stream
    raise last


def _load_state_dict(torch, ckpt_path: str):
    ckpt, load_meta = _torch_load_fallback(torch, ckpt_path)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            meta = dict(ckpt)
            meta.update(load_meta)
            return ckpt["state_dict"], meta

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            meta = dict(ckpt)
            meta.update(load_meta)
            return ckpt["model"], meta

        if any(
            k.startswith(("intro.", "ending.", "encoders.", "downs.", "middle.", "decoders.", "ups."))
            for k in ckpt.keys()
        ):
            return ckpt, load_meta  # was raw state_dict, so meta is just load_meta

    raise RuntimeError("Unsupported checkpoint format (expected state_dict-like dict).")


def _infer_nafnet_cfg_from_sd(sd: dict) -> Tuple[int, int]:
    base_ch = int(sd["intro.weight"].shape[0]) if "intro.weight" in sd else 32

    downs_idx = set()
    for k in sd.keys():
        if k.startswith("downs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                downs_idx.add(int(parts[1]))

    if downs_idx:
        levels = max(downs_idx) + 1
    else:
        enc_idx = set()
        for k in sd.keys():
            if k.startswith("encoders."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    enc_idx.add(int(parts[1]))
        levels = (max(enc_idx) + 1) if enc_idx else 4

    levels = max(1, min(8, int(levels)))
    return base_ch, levels


def load_nafnet_model(
    ckpt_path: str,
    *,
    use_gpu: bool,
    prefer_dml: bool,
    model_kind: str = "nadir",
):
    from setiastro.saspro.runtime_torch import import_torch

    torch = import_torch(
        prefer_cuda=use_gpu,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=lambda *_: None,
    )

    # unified factory (supports variant="nadir" | "axiomv2")
    from setiastro.saspro.syqon_model.model import create_model

    sd, meta = _load_state_dict(torch, ckpt_path)
    base_ch, depth = _infer_nafnet_cfg_from_sd(sd)

    variant = (model_kind or "nadir").lower().strip()
    if variant not in ("nadir", "axiomv2"):
        variant = "nadir"

    model = create_model(base_ch=base_ch, depth=depth, variant=variant)

    # OPTIONAL: fail fast if it's clearly the wrong architecture/width
    # (still keep strict=False for robustness if SyQon adds extra keys)
    try:
        iw = sd.get("intro.weight", None)
        if iw is not None and hasattr(iw, "shape"):
            sd_width = int(iw.shape[0])  # out_channels of intro conv
            if int(base_ch) != sd_width:
                raise RuntimeError(
                    f"SyQon checkpoint width mismatch: state_dict intro.weight[0]={sd_width} "
                    f"but inferred base_ch={base_ch} (variant={variant})."
                )
    except Exception:
        # if anything about the check fails, just proceed with load_state_dict
        pass

    model.load_state_dict(sd, strict=False)
    model.eval()

    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    model.to(device)

    info = {
        "model_kind": variant,
        "base_ch": base_ch,
        "depth": depth,
        #"meta": meta,
        "device": str(device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_file": getattr(torch, "__file__", None),
    }
    return model, device, info, torch



def _to_torch_chw(img_chw, device, torch):
    x = np.asarray(img_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected CHW float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(x[None, ...])
    return t.to(device=device, dtype=torch.float32)


def _predict_tile(
    model,
    t,
    *,
    device,
    use_amp: bool,
    amp_dtype: str,
    info: dict,
    torch,
):
    """
    Run model(t) and return pred as HWC float32 numpy.

    If AMP is enabled and non-finite output occurs, fallback to fp32 for this tile
    and record info["amp_fallback"] / info["amp_reason"].
    """

    def _to_numpy(pred_t):
        # pred_t: 1CHW torch
        pred_np = pred_t[0].detach().to("cpu").numpy().transpose(1, 2, 0)
        return pred_np.astype(np.float32, copy=False)

    # --- AMP path (if requested & supported) ---
    if use_amp and device.type == "cuda":
        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            pred_t = model(t)
        pred = _to_numpy(pred_t)

        if not np.isfinite(pred).all():
            # fallback to fp32 for this tile
            info["amp_fallback"] = True
            info["amp_reason"] = "non-finite output detected under CUDA AMP; reran tile in fp32"
            pred_t = model(t)
            pred = _to_numpy(pred_t)

        return pred

    if use_amp and device.type == "mps":
        # Conservative: autocast enabled, no explicit dtype forcing
        try:
            with torch.autocast(device_type="mps"):
                pred_t = model(t)
            pred = _to_numpy(pred_t)

            if not np.isfinite(pred).all():
                info["amp_fallback"] = True
                info["amp_reason"] = "non-finite output detected under MPS autocast; reran tile in fp32"
                pred_t = model(t)
                pred = _to_numpy(pred_t)

            return pred
        except Exception:
            # If autocast isn't supported in their torch build, just do fp32
            pred_t = model(t)
            pred = _to_numpy(pred_t)
            return pred

    # --- fp32 path ---
    pred_t = model(t)
    pred = _to_numpy(pred_t)

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite output detected in fp32 inference.")

    return pred


def nafnet_starless_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    prefer_cuda: bool = True,
    residual_mode: bool = True,  # True = model predicts stars; starless = x - pred
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    model_kind: str = "nadir",
    use_gpu: bool = True, prefer_dml: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
):
    """
    Input: HWC float32 [0,1], RGB (or mono HxW)
    Output: starless HWC float32 [0,1] and stars_only HWC float32 [0,1]
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3]

    H, W = x.shape[:2]
    tile = int(tile)
    overlap = int(overlap)
    stride = max(tile - overlap, 1)

    model, device, info, torch = load_nafnet_model(
        ckpt_path,
        use_gpu=use_gpu,
        prefer_dml=prefer_dml,
        model_kind=model_kind,
    )

    info = dict(info or {})
    info["device"] = str(device)
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_file"] = getattr(torch, "__file__", None)
    # AMP config
    use_amp_requested = bool(use_amp)
    amp_dtype = (amp_dtype or "fp16").lower()
    if amp_dtype not in ("fp16", "bf16"):
        amp_dtype = "fp16"

    # only allow AMP where it makes sense
    use_amp_effective = bool(use_amp_requested) and (device.type in ("cuda", "mps"))
    info["use_amp_requested"] = use_amp_requested
    info["use_amp_effective"] = use_amp_effective
    info["amp_dtype"] = amp_dtype

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W, 1), dtype=np.float32)

    # feather weight
    wy = np.hanning(tile).astype(np.float32)
    wx = np.hanning(tile).astype(np.float32)
    w2 = (wy[:, None] * wx[None, :]).astype(np.float32)
    w2 = w2[..., None]  # tile x tile x 1

    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    total = len(ys) * len(xs)
    done = 0

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

                chw = patch.transpose(2, 0, 1)  # CHW
                t = _to_torch_chw(chw, device, torch)

                pred = _predict_tile(
                    model,
                    t,
                    device=device,
                    use_amp=use_amp_effective,
                    amp_dtype=amp_dtype,
                    info=info,
                    torch=torch,
                )  # HWC float32

                if residual_mode:
                    starless_patch = patch - pred
                    stars_patch = pred
                else:
                    starless_patch = pred
                    stars_patch = patch - pred

                starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                stars_patch = np.clip(stars_patch, 0.0, 1.0).astype(np.float32, copy=False)

                starless_patch = starless_patch[:ph, :pw, :]
                wlocal = w2[:ph, :pw, :]

                out_acc[y0:y1, x0:x1, :] += starless_patch * wlocal
                w_acc[y0:y1, x0:x1, :] += wlocal

                done += 1
                if callable(progress_cb):
                    progress_cb(done, total, "SyQon NAFNet tiles…")

    starless = out_acc / np.maximum(w_acc, 1e-8)
    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)
    stars_only = np.clip(x - starless, 0.0, 1.0).astype(np.float32, copy=False)

    if was_mono:
        return (
            starless.mean(axis=2).astype(np.float32, copy=False),
            stars_only.mean(axis=2).astype(np.float32, copy=False),
            info,
        )
    return starless, stars_only, info


def _get_setting_any(settings, keys, default):
    """
    settings may be QSettings or dict-like.
    keys can be a tuple/list of possible setting names.
    """
    for k in keys:
        try:
            # QSettings: value(key, default, type=?)
            v = settings.value(k, None)
            if v is not None and v != "":
                return v
        except Exception:
            pass
        try:
            # dict-like
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
    """
    Convenience wrapper:
      - ensures RGB float32 [0..1]
      - caches the SyQon session (so you don't reload ckpt every frame)
      - runs tiling with settings-driven tile/overlap/AMP
    Returns: (starless_rgb01, stars_rgb01, info)
    """
    global _SYQON_SESSION, _SYQON_CKPT

    ckpt = str(_get_setting_any(settings, (ckpt_key, "syqon/ckpt_path"), ""))
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError(f"SyQon checkpoint path is not configured or missing: '{ckpt}'")

    prefer_dml = bool(_get_setting_any(settings, (prefer_dml_key,), True))
    use_gpu    = bool(_get_setting_any(settings, (use_gpu_key,), True))

    # cache session if ckpt changes or session missing
    if _SYQON_SESSION is None or _SYQON_CKPT != ckpt:
        _SYQON_SESSION = syqonnafnetSession(ckpt, use_gpu=use_gpu, prefer_dml=prefer_dml)
        _SYQON_CKPT = ckpt

    tile = int(_get_setting_any(settings, (tile_key,), 512))
    ov   = int(_get_setting_any(settings, (overlap_key,), 64))
    amp  = bool(_get_setting_any(settings, (amp_key,), False))
    amp_dtype = str(_get_setting_any(settings, (amp_dtype_key,), "fp16"))

    # ensure RGB
    if img.ndim == 2:
        src = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        src = np.repeat(img, 3, axis=2)
    else:
        src = img
    src = src.astype(np.float32, copy=False)

    starless, stars, info = _SYQON_SESSION.run_starless_rgb01(
        src,
        tile=tile,
        overlap=ov,
        residual_mode=residual_mode,
        use_amp=amp,
        amp_dtype=amp_dtype,
        progress_cb=progress_cb,
    )
    return starless, stars, info

class syqonnafnetSession:
    def __init__(self, ckpt_path: str, *, use_gpu: bool, prefer_dml: bool, model_kind: str = "nadir"):
        self.ckpt_path = ckpt_path
        self.model_kind = (model_kind or "nadir").lower().strip()
        self.model, self.device, self.info, self.torch = load_nafnet_model(
            ckpt_path, use_gpu=use_gpu, prefer_dml=prefer_dml, model_kind=self.model_kind
        )

    def run_starless_rgb01(
        self,
        img_rgb01: np.ndarray,
        *,
        tile: int = 512,
        overlap: int = 64,
        residual_mode: bool = True,
        use_amp: bool = False,
        amp_dtype: str = "fp16",
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ):
        # ---- identical logic to nafnet_starless_rgb01, but uses self.model/device/torch ----
        x = np.asarray(img_rgb01, dtype=np.float32)
        was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

        if x.ndim == 2:
            x = np.stack([x] * 3, axis=-1)
        elif x.ndim == 3 and x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        else:
            x = x[..., :3]

        H, W = x.shape[:2]
        tile = int(tile)
        overlap = int(overlap)
        stride = max(tile - overlap, 1)

        info = dict(self.info or {})
        info["device"] = str(self.device)

        # AMP config (same rules)
        use_amp_requested = bool(use_amp)
        amp_dtype = (amp_dtype or "fp16").lower()
        if amp_dtype not in ("fp16", "bf16"):
            amp_dtype = "fp16"
        use_amp_effective = bool(use_amp_requested) and (self.device.type in ("cuda", "mps"))

        info["use_amp_requested"] = use_amp_requested
        info["use_amp_effective"] = use_amp_effective
        info["amp_dtype"] = amp_dtype

        out_acc = np.zeros((H, W, 3), dtype=np.float32)
        w_acc = np.zeros((H, W, 1), dtype=np.float32)

        wy = np.hanning(tile).astype(np.float32)
        wx = np.hanning(tile).astype(np.float32)
        w2 = (wy[:, None] * wx[None, :]).astype(np.float32)[..., None]

        ys = list(range(0, H, stride))
        xs = list(range(0, W, stride))
        total = len(ys) * len(xs)
        done = 0

        torch = self.torch
        model = self.model
        device = self.device

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
                    t = _to_torch_chw(chw, device, torch)

                    pred = _predict_tile(
                        model, t,
                        device=device,
                        use_amp=use_amp_effective,
                        amp_dtype=amp_dtype,
                        info=info,
                        torch=torch,
                    )

                    if residual_mode:
                        starless_patch = patch - pred
                    else:
                        starless_patch = pred

                    starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                    starless_patch = starless_patch[:ph, :pw, :]
                    wlocal = w2[:ph, :pw, :]

                    out_acc[y0:y1, x0:x1, :] += starless_patch * wlocal
                    w_acc[y0:y1, x0:x1, :] += wlocal

                    done += 1
                    if callable(progress_cb):
                        progress_cb(done, total, "SyQon NAFNet tiles…")

        starless = out_acc / np.maximum(w_acc, 1e-8)
        starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)
        stars_only = np.clip(x - starless, 0.0, 1.0).astype(np.float32, copy=False)

        if was_mono:
            return (
                starless.mean(axis=2).astype(np.float32, copy=False),
                stars_only.mean(axis=2).astype(np.float32, copy=False),
                info,
            )
        return starless, stars_only, info
