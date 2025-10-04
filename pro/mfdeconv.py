# pro/mfdeconv.py
from __future__ import annotations
import os, math
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QObject, pyqtSignal
from pro.psf_utils import compute_psf_kernel_for_image
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread
import contextlib
try:
    import sep
except Exception:
    sep = None

torch = None        # filled by runtime loader if available
TORCH_OK = False
NO_GRAD = contextlib.nullcontext  # fallback



def _process_gui_events_safely():
    app = QApplication.instance()
    if app and QThread.currentThread() is app.thread():
        app.processEvents()

EPS = 1e-6

# -----------------------------
# Helpers: image prep / shapes
# -----------------------------

# new: lightweight loader that yields one frame at a time
def _iter_fits(paths):
    for p in paths:
        with fits.open(p, memmap=True) as hdul:
            arr = np.asarray(hdul[0].data, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            hdr = hdul[0].header
        yield arr, hdr

def _to_luma_local(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return a
    # (H,W,3) or (3,H,W)
    if a.ndim == 3 and a.shape[-1] == 3:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[0] == 3:
        r, g, b = a[0], a[1], a[2]
        return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32, copy=False)
    return a.mean(axis=-1).astype(np.float32, copy=False)

def _stack_loader(paths):
    ys, hdrs = [], []
    for p in paths:
        with fits.open(p, memmap=True) as hdul:
            arr = hdul[0].data
            hdr = hdul[0].header
        arr = np.asarray(arr, dtype=np.float32)
        # squeeze trailing singleton channel
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        ys.append(arr)
        hdrs.append(hdr)
    return ys, hdrs

def _normalize_layout_single(a, color_mode):
    """
    Coerce to:
      - 'luma'       -> (H, W)
      - 'perchannel' -> (C, H, W); mono stays (1,H,W), RGB → (3,H,W)
    Accepts (H,W), (H,W,3), or (3,H,W).
    """
    a = np.asarray(a, dtype=np.float32)

    if color_mode == "luma":
        return _to_luma_local(a)  # returns (H,W)

    # perchannel
    if a.ndim == 2:
        return a[None, ...]                     # (1,H,W)  ← keep mono as 1 channel
    if a.ndim == 3 and a.shape[-1] == 3:
        return np.moveaxis(a, -1, 0)            # (3,H,W)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        return a                                 # already (1,H,W) or (3,H,W)
    # fallback: average any weird shape into luma 1×H×W
    l = _to_luma_local(a)
    return l[None, ...]


def _normalize_layout_batch(arrs, color_mode):
    return [_normalize_layout_single(a, color_mode) for a in arrs]

def _common_hw(data_list):
    """Return minimal (H,W) across items; items are (H,W) or (C,H,W)."""
    Hs, Ws = [], []
    for a in data_list:
        if a.ndim == 2:
            H, W = a.shape
        else:
            _, H, W = a.shape
        Hs.append(H); Ws.append(W)
    return int(min(Hs)), int(min(Ws))

def _center_crop(arr, Ht, Wt):
    """Center-crop arr (H,W) or (C,H,W) to (Ht,Wt)."""
    if arr.ndim == 2:
        H, W = arr.shape
        if H == Ht and W == Wt:
            return arr
        y0 = max(0, (H - Ht) // 2)
        x0 = max(0, (W - Wt) // 2)
        return arr[y0:y0+Ht, x0:x0+Wt]
    else:
        C, H, W = arr.shape
        if H == Ht and W == Wt:
            return arr
        y0 = max(0, (H - Ht) // 2)
        x0 = max(0, (W - Wt) // 2)
        return arr[:, y0:y0+Ht, x0:x0+Wt]

def _sanitize_numeric(a):
    """Replace NaN/Inf, clip negatives, make contiguous float32."""
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, None).astype(np.float32, copy=False)
    return np.ascontiguousarray(a)

# -----------------------------
# PSF utilities
# -----------------------------

def _gaussian_psf(fwhm_px: float, ksize: int) -> np.ndarray:
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = (ksize - 1) / 2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    g /= (np.sum(g) + EPS)
    return g.astype(np.float32, copy=False)

def _estimate_fwhm_from_header(hdr) -> float:
    for key in ("FWHM", "FWHM_PIX", "PSF_FWHM"):
        if key in hdr:
            try:
                val = float(hdr[key])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return float("nan")

def _estimate_fwhm_from_image(arr) -> float:
    """Fast FWHM estimate from SEP 'a','b' parameters (≈ sigma in px)."""
    if sep is None:
        return float("nan")
    try:
        img = _to_luma_local(arr)
        bkg = sep.Background(img)
        data = img - bkg.back()
        try:
            err = bkg.globalrms
        except Exception:
            err = float(np.median(bkg.rms()))
        sources = sep.extract(data, 6.0, err=err)
        if sources is None or len(sources) == 0:
            return float("nan")
        a = np.asarray(sources["a"], dtype=np.float32)
        b = np.asarray(sources["b"], dtype=np.float32)
        ab = (a + b) * 0.5
        sigma = float(np.median(ab[np.isfinite(ab) & (ab > 0)]))
        if not np.isfinite(sigma) or sigma <= 0:
            return float("nan")
        return 2.3548 * sigma
    except Exception:
        return float("nan")

def _auto_ksize_from_fwhm(fwhm_px: float, kmin: int = 11, kmax: int = 51) -> int:
    """
    Choose odd kernel size to cover about ±4σ.
    """
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = int(math.ceil(4.0 * sigma))
    k = 2 * r + 1
    k = max(kmin, min(k, kmax))
    if (k % 2) == 0:
        k += 1
    return k

def _flip_kernel(psf):
    # PyTorch dislikes negative strides; make it contiguous.
    return np.flip(np.flip(psf, -1), -2).copy()

def _conv_same_np(img, psf):
    # img: (H,W) or (C,H,W) numpy
    import numpy.fft as fft
    def fftconv2(a, k):
        H, W = a.shape[-2:]
        kh, kw = k.shape
        pad_h, pad_w = H + kh - 1, W + kw - 1
        A = fft.rfftn(a, s=(pad_h, pad_w), axes=(-2, -1))
        K = fft.rfftn(k, s=(pad_h, pad_w), axes=(-2, -1))
        Y = A * K
        y = fft.irfftn(Y, s=(pad_h, pad_w), axes=(-2, -1))
        sh, sw = (kh - 1)//2, (kw - 1)//2
        return y[..., sh:sh+H, sw:sw+W]
    if img.ndim == 2:
        return fftconv2(img[None], psf)[0]
    else:
        return np.stack([fftconv2(img[c:c+1], psf)[0] for c in range(img.shape[0])], axis=0)

def _normalize_psf(psf):
    psf = np.maximum(psf, 0.0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if not np.isfinite(s) or s <= EPS:
        return psf
    return (psf / s).astype(np.float32, copy=False)

def _soften_psf(psf, sigma_px=0.25):
    # optional tiny Gaussian soften to reduce ringing; sigma<=0 disables
    if sigma_px <= 0:
        return psf
    r = int(max(1, round(3 * sigma_px)))
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y) / (2 * sigma_px * sigma_px)).astype(np.float32)
    g /= g.sum() + EPS
    return _conv_same_np(psf[None], g)[0]

def _psf_fwhm_px(psf: np.ndarray) -> float:
    """Approximate FWHM (pixels) from second moments of a normalized kernel."""
    psf = np.maximum(psf, 0).astype(np.float32, copy=False)
    s = float(psf.sum())
    if s <= EPS:
        return float("nan")
    k = psf.shape[0]
    y, x = np.mgrid[:k, :k].astype(np.float32)
    cy = float((psf * y).sum() / s)
    cx = float((psf * x).sum() / s)
    var_y = float((psf * (y - cy) ** 2).sum() / s)
    var_x = float((psf * (x - cx) ** 2).sum() / s)
    sigma = math.sqrt(max(0.0, 0.5 * (var_x + var_y)))
    return 2.3548 * sigma  # FWHM≈2.355σ

def _build_psf_bank_from_data_auto(
    ys_raw, hdrs, status_cb=lambda s: None, save_dir: str | None = None
):
    """
    Build per-frame PSFs with auto kernel size. Returns list of psfs (float32).
    - ksize is chosen from header/image FWHM.
    - If star PSF fails, retries with smaller ksizes then falls back to Gaussian.
    - Logs diagnostics and optionally writes per-frame PSFs.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    psfs = []
    for i, (arr, hdr) in enumerate(zip(ys_raw, hdrs), start=1):
        status_cb(f"MFDeconv: measuring PSF {i}/{len(ys_raw)} …")
        _process_gui_events_safely()

        # FWHM estimate selection
        f_hdr = _estimate_fwhm_from_header(hdr)
        f_img = _estimate_fwhm_from_image(arr)
        f_whm = f_hdr if (np.isfinite(f_hdr)) else f_img
        if not np.isfinite(f_whm) or f_whm <= 0:
            f_whm = 2.5  # safe dull default

        # initial auto ksize (±4σ)
        k_auto = _auto_ksize_from_fwhm(f_whm)

        # Try star-derived PSF with a shortlist of ksizes
        tried = []
        psf = None
        k_used = None
        for k_try in [k_auto, max(k_auto - 4, 11), 21, 17, 15, 13, 11]:
            if k_try in tried:
                continue
            tried.append(k_try)
            try:
                out = compute_psf_kernel_for_image(arr, ksize=k_try, det_sigma=6.0, max_stars=80)
                psf = out[0] if (isinstance(out, tuple) and len(out) >= 1) else out
                if psf is not None:
                    k_used = k_try
                    break
            except Exception:
                psf = None

        info_extra = {}
        if psf is None:
            psf = _gaussian_psf(f_whm, ksize=k_auto)
            info_extra = {"fallback": True, "header_fwhm_px": float(f_hdr) if np.isfinite(f_hdr) else None}
        else:
            info_extra = {"fallback": False, "ksize_used": k_used}

        psf = _normalize_psf(psf.astype(np.float32, copy=False))
        psf = _soften_psf(psf, sigma_px=0.25)
        fwhm_est = _psf_fwhm_px(psf)

        # If compute_psf_kernel_for_image returned a tuple with info, try to log star counts
        used = det = rej = None
        try:
            # Some builds return (psf, info)
            tmp = compute_psf_kernel_for_image  # just to avoid flake warnings
            # no reliable info struct here (since we didn't keep it), leave None
            pass
        except Exception:
            pass

        msg = f"  PSF{i}: ksize={psf.shape[0]} | FWHM≈{fwhm_est:.2f}px"
        if used is not None:
            msg += f" | stars: used={used}"
            if det is not None and rej is not None:
                msg += f"/det={det}, rej={rej}"
        if info_extra.get("fallback"):
            msg += " | (fallback PSF)"
        status_cb(msg)
        _process_gui_events_safely()

        if save_dir:
            fits.PrimaryHDU(psf).writeto(os.path.join(save_dir, f"psf_{i:03d}.fit"), overwrite=True)

        psfs.append(psf)
    return psfs

# -----------------------------
# Robust weighting (Huber)
# -----------------------------

def _estimate_scalar_variance_t(r):
    # r: tensor on device
    med = torch.median(r)
    mad = torch.median(torch.abs(r - med)) + 1e-6
    return (1.4826 * mad) ** 2

def _estimate_scalar_variance(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return float((1.4826 * mad) ** 2)

def _weight_map(y, pred, huber_delta, var_map=None, mask=None):
    """
    Robust per-pixel weights for the MM update.
    If huber_delta < 0, interpret it as a factor × RMS (auto mode):
        delta = (-huber_delta) * background_RMS(residual)
    """
    r = y - pred
    eps = EPS

    # --- robust RMS / delta on the same backend as r ---
    if huber_delta < 0:
        if TORCH_OK and isinstance(r, torch.Tensor):
            med = torch.median(r)
            mad = torch.median(torch.abs(r - med)) + 1e-6
            rms = 1.4826 * mad
            delta = (-huber_delta) * torch.clamp(rms, min=1e-6)
        else:
            med = np.median(r)
            mad = np.median(np.abs(r - med)) + 1e-6
            rms = 1.4826 * mad
            delta = (-huber_delta) * max(rms, 1e-6)
    else:
        delta = huber_delta

    # --- psi(r)/r ---
    if TORCH_OK and isinstance(r, torch.Tensor):
        if float(delta) > 0:
            absr = torch.abs(r)
            psi_over_r = torch.where(absr <= delta,
                                     torch.ones_like(r),
                                     delta / (absr + eps))
        else:
            psi_over_r = torch.ones_like(r)
        if var_map is None:
            v = _estimate_scalar_variance_t(r)
            w = psi_over_r / (v + eps)
        else:
            w = psi_over_r / (var_map + eps)
        if mask is not None:
            w = w * mask
        return w
    else:
        if float(delta) > 0:
            absr = np.abs(r)
            psi_over_r = np.where(absr <= delta,
                                  np.ones_like(r, dtype=np.float32),
                                  delta / (absr + eps))
        else:
            psi_over_r = np.ones_like(r, dtype=np.float32)
        if var_map is None:
            v = _estimate_scalar_variance(r)
            w = psi_over_r / (v + eps)
        else:
            w = psi_over_r / (var_map + eps)
        if mask is not None:
            w = w * mask
        return w


# -----------------------------
# Torch / conv
# -----------------------------

def _fftshape_same(H, W, kh, kw):
    return H + kh - 1, W + kw - 1

# ---------- Torch FFT helpers (FIXED: carry padH/padW) ----------
def _precompute_torch_psf_ffts(psfs, flip_psf, H, W, device, dtype):
    """
    Returns two lists of tuples:
      psf_fft:  [(Kf, padH, padW), ...]
      psfT_fft: [(KTf, padH, padW), ...]
    where Kf/KTf are rfft2 of the padded kernels to (padH, padW).
    """
    import torch.fft as tfft
    psf_fft  = []
    psfT_fft = []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        padH, padW = _fftshape_same(H, W, kh, kw)

        # center-place kernel into a (padH, padW) canvas
        k_pad  = torch.zeros((padH, padW), device=device, dtype=dtype)
        kT_pad = torch.zeros((padH, padW), device=device, dtype=dtype)
        sy, sx = (padH - kh)//2, (padW - kw)//2
        k_pad [sy:sy+kh,  sx:sx+kw] = torch.as_tensor(k,  device=device, dtype=dtype)
        kT_pad[sy:sy+kh, sx:sx+kw]  = torch.as_tensor(kT, device=device, dtype=dtype)

        # rfft over the *real* spatial size (padH, padW)
        Kf  = tfft.rfftn(k_pad,  s=(padH, padW))
        KTf = tfft.rfftn(kT_pad, s=(padH, padW))

        psf_fft.append((Kf,  padH, padW))
        psfT_fft.append((KTf, padH, padW))
    return psf_fft, psfT_fft


def _fft_conv_same_torch(x, Kf_pack, out_spatial):
    """
    x:         (H,W) or (C,H,W) tensor on device
    Kf_pack:   tuple (Kf, padH, padW) from _precompute_torch_psf_ffts
    out_spatial: preallocated tensor like x to receive the 'same' result
    """
    import torch.fft as tfft
    Kf, padH, padW = Kf_pack
    H, W = x.shape[-2], x.shape[-1]

    if x.ndim == 2:
        X = tfft.rfftn(x, s=(padH, padW))
        y = tfft.irfftn(X * Kf, s=(padH, padW))
        sh, sw = (padH - H) // 2, (padW - W) // 2
        out_spatial.copy_(y[sh:sh+H, sw:sw+W])
        return out_spatial
    else:
        X = tfft.rfftn(x, s=(padH, padW), dim=(-2, -1))
        y = tfft.irfftn(X * Kf, s=(padH, padW), dim=(-2, -1))
        sh, sw = (padH - H) // 2, (padW - W) // 2
        out_spatial.copy_(y[..., sh:sh+H, sw:sw+W])
        return out_spatial

# ---------- NumPy FFT helpers ----------
def _precompute_np_psf_ffts(psfs, flip_psf, H, W):
    import numpy.fft as fft
    meta = []
    Kfs  = []
    KTfs = []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        fftH, fftW = _fftshape_same(H, W, kh, kw)
        Kfs.append( fft.rfftn(k,  s=(fftH, fftW)) )
        KTfs.append(fft.rfftn(kT, s=(fftH, fftW)) )
        meta.append((kh, kw, fftH, fftW))
    return Kfs, KTfs, meta

def _fft_conv_same_np(a, Kf, kh, kw, fftH, fftW, out):
    import numpy.fft as fft
    if a.ndim == 2:
        A = fft.rfftn(a, s=(fftH, fftW))
        y = fft.irfftn(A * Kf, s=(fftH, fftW))
        sh, sw = (fftH - a.shape[0])//2, (fftW - a.shape[1])//2
        out[...] = y[sh:sh+a.shape[0], sw:sw+a.shape[1]]
        return out
    else:
        # per-channel
        C, H, W = a.shape
        acc = []
        for c in range(C):
            A = fft.rfftn(a[c], s=(fftH, fftW))
            y = fft.irfftn(A * Kf, s=(fftH, fftW))
            sh, sw = (fftH - H)//2, (fftW - W)//2
            acc.append(y[sh:sh+H, sw:sw+W])
        out[...] = np.stack(acc, 0)
        return out


def _torch_device():
    if TORCH_OK and (torch is not None):
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        # DirectML: we passed dml_device from outer scope; keep a module-global
        if globals().get("dml_ok", False) and globals().get("dml_device", None) is not None:
            return globals()["dml_device"]
    return torch.device("cpu")

def _to_t(x: np.ndarray):
    if not (TORCH_OK and (torch is not None)):
        raise RuntimeError("Torch path requested but torch is unavailable")
    device = _torch_device()
    t = torch.from_numpy(x)
    # DirectML wants explicit .to(device)
    return t.to(device, non_blocking=True) if str(device) != "cpu" else t

def _contig(x):
    return np.ascontiguousarray(x, dtype=np.float32)

def _conv_same_torch(img_t, psf_t):
    """
    img_t: torch tensor on DEVICE, (H,W) or (C,H,W)
    psf_t: torch tensor on DEVICE, (1,1,kh,kw)  (single kernel)
    Pads with 'reflect' to avoid zero-padding ringing.
    """
    kh, kw = psf_t.shape[-2:]
    pad = (kw // 2, kw - kw // 2 - 1,  # left, right
           kh // 2, kh - kh // 2 - 1)  # top, bottom

    if img_t.ndim == 2:
        x = img_t[None, None]
        x = torch.nn.functional.pad(x, pad, mode="reflect")
        y = torch.nn.functional.conv2d(x, psf_t, padding=0)
        return y[0, 0]
    else:
        C = img_t.shape[0]
        x = img_t[None]
        x = torch.nn.functional.pad(x, pad, mode="reflect")
        w = psf_t.repeat(C, 1, 1, 1)
        y = torch.nn.functional.conv2d(x, w, padding=0, groups=C)
        return y[0]

def _safe_inference_context():
    """
    Return a valid, working no-grad context:
      - prefer torch.inference_mode() if it exists *and* can be entered,
      - otherwise fall back to torch.no_grad(),
      - if torch is unavailable, return NO_GRAD.
    """
    if not (TORCH_OK and (torch is not None)):
        return NO_GRAD

    cm = getattr(torch, "inference_mode", None)
    if cm is None:
        return torch.no_grad

    # Probe inference_mode once; if it explodes on this build, fall back.
    try:
        with cm():
            pass
        return cm
    except Exception:
        return torch.no_grad


# -----------------------------
# Core
# -----------------------------

def multiframe_deconv(
    paths,
    out_path,
    iters=20,
    kappa=2.0,
    color_mode="luma",
    huber_delta=0.0,
    status_cb=lambda s: None
):
    

    def _emit_pct(pct: float, msg: str | None = None):
        pct = float(max(0.0, min(1.0, pct)))
        status_cb(f"__PROGRESS__ {pct:.4f}" + (f" {msg}" if msg else ""))

    status_cb(f"MFDeconv: loading {len(paths)} aligned frames…")
    _emit_pct(0.02, "loading")    
    ys_raw, hdrs = _stack_loader(paths)
    _emit_pct(0.05, "preparing")
    relax = 0.7  # 0<alpha<=1; smaller = more damping.
    use_torch = False
    global torch, TORCH_OK

    # -------- try to import torch from per-user runtime venv --------
    global TORCH_OK
    torch = None
    cuda_ok = mps_ok = dml_ok = False
    dml_device = None
    try:
        from pro.runtime_torch import import_torch
        torch = import_torch(prefer_cuda=True, status_cb=status_cb)
        TORCH_OK = True

        try: cuda_ok = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception: cuda_ok = False

        try: mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception: mps_ok = False

        # DirectML (Windows: AMD/Intel iGPU/dGPU)
        try:
            import torch_directml
            dml_device = torch_directml.device()
            # Quick no-op to validate the backend
            _ = (torch.ones(1, device=dml_device) + 1).item()
            dml_ok = True
        except Exception:
            dml_ok = False

        if cuda_ok:
            status_cb(f"PyTorch CUDA available: True | device={torch.cuda.get_device_name(0)}")
        elif mps_ok:
            status_cb("PyTorch MPS (Apple) available: True")
        elif dml_ok:
            status_cb("PyTorch DirectML (Windows) available: True")
        else:
            status_cb("PyTorch present, using CPU backend.")

        status_cb(
            f"PyTorch {getattr(torch, '__version__', '?')} backend: "
            + ("CUDA" if cuda_ok else "MPS" if mps_ok else "DirectML" if dml_ok else "CPU")
        )
    except Exception as e:
        TORCH_OK = False
        status_cb(f"PyTorch not available → CPU path. ({e})")

    use_torch = bool(TORCH_OK)  # GPU only if CUDA/MPS true
    if use_torch:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True  # let cuDNN pick fastest algo for this shape
        except Exception:
            pass
    _process_gui_events_safely()

    # PSFs (auto-size per frame) + flipped copies
    psf_out_dir = None  # set to e.g. os.path.join(os.path.dirname(out_path), "PSFs") to save PSFs
    psfs = _build_psf_bank_from_data_auto(ys_raw, hdrs, status_cb=status_cb, save_dir=psf_out_dir)
    flip_psf = [_flip_kernel(k) for k in psfs]
    _emit_pct(0.20, "psf ready")



    # Normalize layout BEFORE size harmonization
    data = _normalize_layout_batch(ys_raw, color_mode)  # list of (H,W) or (3,H,W)
    _emit_pct(0.25, "seed ready")

    # Center-crop all to common intersection
    Ht, Wt = _common_hw(data)
    if any(((a.shape[-2] != Ht) or (a.shape[-1] != Wt)) for a in data):
        status_cb(f"MFDeconv: standardizing shapes → crop to {Ht}×{Wt}")
        data = [_center_crop(a, Ht, Wt) for a in data]

    # Numeric hygiene
    data = [_sanitize_numeric(a) for a in data]

    # Initial estimate x0 = median across frames
    if data[0].ndim == 2:
        x = np.median(np.stack(data, axis=0), axis=0).astype(np.float32)
    else:
        x = np.median(np.stack(data, axis=0), axis=0).astype(np.float32)  # (C,H,W)

    status_cb("MFDeconv: starting multiplicative updates…")
    _process_gui_events_safely()
    bg_est = np.median([np.median(np.abs(y - np.median(y))) for y in (data if isinstance(data, list) else [data])]) * 1.4826
    status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g})")
    _process_gui_events_safely()

    # Prepare tensors/arrays used by the main loop
    if use_torch:
        # move initial estimate and all frames to the selected device
        x_t = _to_t(_contig(x))                              # (H,W) or (C,H,W) tensor
        y_tensors = [_to_t(_contig(y)) for y in data]        # list of tensors
    else:
        x_t = x                                              # numpy array
        y_np = data                                          # list of numpy arrays

    total_steps = max(1, int(iters))
    status_cb("__PROGRESS__ 0.0000 Starting iterations…")

    # -------- precompute FFTs and allocate scratch --------
    if use_torch:
        # Place kernels on device once; use spatial conv2d each iter (fast for small PSFs)
        psf_t  = [_to_t(_contig(k))[None, None]  for k  in psfs]      # (1,1,kh,kw)
        psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]  # (1,1,kh,kw)

        num = torch.zeros_like(x_t)
        den = torch.zeros_like(x_t)
    else:
        # (keep NumPy/FFT branch as-is)
        x_t = x  # ensure name consistency
        y_np = data
        if x_t.ndim == 2:
            H, W = x_t.shape
        else:
            _, H, W = x_t.shape
        Kfs, KTfs, meta = _precompute_np_psf_ffts(psfs, flip_psf, H, W)
        num      = np.zeros_like(x_t)
        den      = np.zeros_like(x_t)
        pred_buf = np.empty_like(x_t)
        tmp_out  = np.empty_like(x_t)

    # -------- inference/no-grad for the whole loop --------
    cm = _safe_inference_context() if use_torch else NO_GRAD

    with cm():
        for it in range(1, iters + 1):
            if use_torch:
                num.zero_(); den.zero_()
                for yt, wk, wkT in zip(y_tensors, psf_t, psfT_t):
                    pred = _conv_same_torch(x_t, wk)                 # spatial conv2d
                    wmap = _weight_map(yt, pred, huber_delta, var_map=None, mask=None)
                    num  += _conv_same_torch(wmap * yt,   wkT)       # back-projection
                    den  += _conv_same_torch(wmap * pred, wkT)
                upd    = torch.clamp(num / (den + EPS), 1.0 / kappa, kappa)
                x_next = torch.clamp(x_t * upd, min=0.0)
                if torch.median(torch.abs(upd - 1)) < 1e-3:
                    x_t = x_next
                    status_cb(f"MFDeconv: iter {it}/{iters} (early stop)")
                    _process_gui_events_safely()
                    break
                x_t = (1.0 - relax) * x_t + relax * x_next
            else:
                num.fill(0.0); den.fill(0.0)
                for (yt, Kf, KTf, (kh, kw, fftH, fftW)) in zip(y_np, Kfs, KTfs, meta):
                    _fft_conv_same_np(x_t, Kf, kh, kw, fftH, fftW, pred_buf)
                    wmap = _weight_map(yt, pred_buf, huber_delta, var_map=None, mask=None)
                    _fft_conv_same_np(wmap * yt,      KTf, kh, kw, fftH, fftW, tmp_out); num += tmp_out
                    _fft_conv_same_np(wmap * pred_buf, KTf, kh, kw, fftH, fftW, tmp_out); den += tmp_out
                upd    = np.clip(num / (den + EPS), 1.0 / kappa, kappa)
                x_next = np.clip(x_t * upd, 0.0, None)
                if np.median(np.abs(upd - 1.0)) < 1e-3:
                    x_t = x_next
                    status_cb(f"MFDeconv: iter {it}/{iters} (early stop)")
                    _process_gui_events_safely()
                    break
                x_t = (1.0 - relax) * x_t + relax * x_next

            # UI throttled (don’t spam every iter)
            if (it == 1) or (it % 3 == 0) or (it == iters):
                frac = 0.25 + 0.70 * (it / float(iters))
                _emit_pct(frac, f"iter {it}/{iters}")
                status_cb(f"__PROGRESS__ {it/total_steps:.4f} Iter {it}/{iters}")
                _process_gui_events_safely()

    # ----------------------------
    # Save result (keep FITS-friendly order: (C,H,W))
    # ----------------------------
    _emit_pct(0.97, "saving")
    x_final = x_t.detach().cpu().numpy().astype(np.float32) if use_torch \
              else x_t.astype(np.float32)

    # Ensure channels-first for FITS
    if x_final.ndim == 3:
        # If it's already (C,H,W) we're good; if it's (H,W,C) move C → first
        if x_final.shape[0] not in (1, 3) and x_final.shape[-1] in (1, 3):
            x_final = np.moveaxis(x_final, -1, 0)  # (C,H,W)

        # Optional: collapse singleton C=1 to 2D
        if x_final.shape[0] == 1:
            x_final = x_final[0]  # (H,W)

    # (No channels-last write for FITS; viewers expect cubes as (nz,ny,nx))

    try:
        hdr0 = fits.getheader(paths[0], ext=0)
    except Exception:
        hdr0 = fits.Header()

    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution (beta)')
    hdr0['MF_COLOR'] = (str(color_mode), 'Color mode used')
    if isinstance(x_final, np.ndarray):
        if x_final.ndim == 2:
            hdr0['MF_SHAPE'] = (f"{x_final.shape[0]}x{x_final.shape[1]}", 'Saved as 2D image (HxW)')
        elif x_final.ndim == 3:
            C, H, W = x_final.shape
            hdr0['MF_SHAPE'] = (f"{C}x{H}x{W}", 'Saved as 3D cube (CxHxW)')
    status_cb(f"MFDeconv: saving array with shape {x_final.shape} "
            + ("(2D)" if x_final.ndim==2 else "(C×H×W)"))
    fits.PrimaryHDU(data=x_final, header=hdr0).writeto(out_path, overwrite=True)
    status_cb(f"✅ MFDeconv saved: {out_path}")
    _emit_pct(1.00, "done")
    _process_gui_events_safely()
    return out_path

# -----------------------------
# Worker
# -----------------------------

class MultiFrameDeconvWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, out_path

    def __init__(self, parent, aligned_paths, output_path, iters, kappa, color_mode, huber_delta):
        super().__init__(parent)
        self.aligned_paths = aligned_paths
        self.output_path = output_path
        self.iters = iters
        self.kappa = kappa
        self.color_mode = color_mode
        self.huber_delta = huber_delta

    def _log(self, s): self.progress.emit(s)

    def run(self):
        try:
            out = multiframe_deconv(
                self.aligned_paths,
                self.output_path,
                iters=self.iters,
                kappa=self.kappa,
                color_mode=self.color_mode,
                huber_delta=self.huber_delta,
                status_cb=self._log
            )
            self.finished.emit(True, "MF deconvolution complete.", out)
            _process_gui_events_safely()
        except Exception as e:
            self.finished.emit(False, f"MF deconvolution failed: {e}", "")
