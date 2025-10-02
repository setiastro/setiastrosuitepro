# pro/mfdeconv.py
from __future__ import annotations
import os, math
import numpy as np
from astropy.io import fits
from PyQt6.QtCore import QObject, pyqtSignal
from pro.psf_utils import compute_psf_kernel_for_image

try:
    import sep
except Exception:
    sep = None

torch = None        # filled by runtime loader if available
TORCH_OK = False

EPS = 1e-6

# -----------------------------
# Helpers: image prep / shapes
# -----------------------------

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
      - 'perchannel' -> (3, H, W) (mono is repeated)
    Accepts (H,W), (H,W,3), or (3,H,W).
    """
    a = np.asarray(a, dtype=np.float32)
    if color_mode == "luma":
        return _to_luma_local(a)
    # perchannel
    if a.ndim == 2:
        return np.stack([a, a, a], axis=0)
    if a.ndim == 3 and a.shape[-1] == 3:
        return np.moveaxis(a, -1, 0)
    return a  # already (3,H,W) or best-effort

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

        if save_dir:
            fits.PrimaryHDU(psf).writeto(os.path.join(save_dir, f"psf_{i:03d}.fit"), overwrite=True)

        psfs.append(psf)
    return psfs

# -----------------------------
# Robust weighting (Huber)
# -----------------------------

def _estimate_scalar_variance(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-6
    return float((1.4826 * mad) ** 2)

def _weight_map(y, pred, huber_delta, var_map=None, mask=None):
    """
    Robust per-pixel weights for the MM update.

    If huber_delta < 0, interpret it as a factor × RMS (auto mode):
      delta = (-huber_delta) * background_RMS(residual)

    W = m * [psi(r)/r] / (var + eps)
      - Huber psi/r = 1                if |r| <= delta
                      delta / (|r|+eps) otherwise
    """
    r = y - pred

    # Resolve delta (absolute) first
    if huber_delta < 0:
        # estimate scalar RMS from residuals
        if TORCH_OK and isinstance(r, torch.Tensor):
            med = torch.median(r)
            mad = torch.median(torch.abs(r - med))
            rms = float((1.4826 * mad).detach().cpu().numpy())
        else:
            med = np.median(r)
            mad = np.median(np.abs(r - med))
            rms = 1.4826 * mad
        huber_delta = (-huber_delta) * max(rms, 1e-6)

    # psi(r)/r  (array-shaped)
    if huber_delta > 0:
        if TORCH_OK and isinstance(r, torch.Tensor):
            absr = torch.abs(r)
            psi_over_r = torch.where(absr <= huber_delta,
                                     torch.ones_like(r),
                                     huber_delta / (absr + EPS))
        else:
            absr = np.abs(r)
            psi_over_r = np.where(absr <= huber_delta,
                                  np.ones_like(r, dtype=np.float32),
                                  huber_delta / (absr + EPS))
    else:
        psi_over_r = torch.ones_like(r) if (TORCH_OK and isinstance(r, torch.Tensor)) \
                     else np.ones_like(r, dtype=np.float32)

    # variance term (scalar or map)
    if var_map is None:
        if TORCH_OK and isinstance(r, torch.Tensor):
            v = _estimate_scalar_variance(r.detach().cpu().numpy())
            v = torch.tensor(v, dtype=r.dtype, device=r.device)
        else:
            v = _estimate_scalar_variance(r)
        w = psi_over_r / (v + EPS)
    else:
        w = psi_over_r / (var_map + EPS)

    if mask is not None:
        w = w * mask
    return w

# -----------------------------
# Torch / conv
# -----------------------------

def _torch_device():
    if TORCH_OK and (torch is not None) and hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _to_t(x: np.ndarray):
    if not (TORCH_OK and (torch is not None)):
        raise RuntimeError("Torch path requested but torch is unavailable")
    t = torch.from_numpy(x)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return t.to(_torch_device(), non_blocking=True)
    return t

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
    status_cb(f"MFDeconv: loading {len(paths)} aligned frames…")
    ys_raw, hdrs = _stack_loader(paths)
    relax = 0.7  # 0<alpha<=1; smaller = more damping.
    use_torch = False
    global torch, TORCH_OK

    # -------- try to import torch from per-user runtime venv --------
    global TORCH_OK
    torch = None
    cuda_ok = mps_ok = False
    try:
        from pro.runtime_torch import import_torch
        # Ask for CUDA if plausible; if not available the loader falls back inside.
        prefer_cuda = True
        torch = import_torch(prefer_cuda=prefer_cuda, status_cb=status_cb)
        TORCH_OK = True
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            cuda_ok = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        try:
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            mps_ok = False

        if cuda_ok:
            status_cb(f"PyTorch CUDA available: True | device={torch.cuda.get_device_name(0)}")
        elif mps_ok:
            status_cb("PyTorch MPS (Apple) available: True")
        else:
            status_cb("PyTorch present, using CPU backend.")

        status_cb(
            f"PyTorch {getattr(torch, '__version__', '?')} backend: "
            + ("CUDA" if cuda_ok else "MPS" if mps_ok else "CPU")
        )            
    except Exception as e:
        TORCH_OK = False
        status_cb(f"PyTorch not available → CPU path. ({e})")

    use_torch = bool(TORCH_OK)  # GPU only if CUDA/MPS true

    # PSFs (auto-size per frame) + flipped copies
    psf_out_dir = None  # set to e.g. os.path.join(os.path.dirname(out_path), "PSFs") to save PSFs
    psfs = _build_psf_bank_from_data_auto(ys_raw, hdrs, status_cb=status_cb, save_dir=psf_out_dir)
    flip_psf = [_flip_kernel(k) for k in psfs]

    # Normalize layout BEFORE size harmonization
    data = _normalize_layout_batch(ys_raw, color_mode)  # list of (H,W) or (3,H,W)

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
    bg_est = np.median([np.median(np.abs(y - np.median(y))) for y in (data if isinstance(data, list) else [data])]) * 1.4826
    status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g})")

    if use_torch:
        # Move state & inputs to device
        x_t = _to_t(_contig(x))
        y_tensors = [_to_t(_contig(y)) for y in data]
        # PSFs on device once
        psf_t  = [_to_t(_contig(k))[None, None]  for k in psfs]     # (1,1,kh,kw)
        psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]
    else:
        x_t = x
        y_np = data  # use NP conv

    # MM iterations (Eq. 16): u_k = (Σ Fᵀ(W y)) / (Σ Fᵀ(W F x))
    for it in range(1, iters + 1):
        if use_torch:
            num = torch.zeros_like(x_t)
            den = torch.zeros_like(x_t)
            for yt, wk, wkT in zip(y_tensors, psf_t, psfT_t):
                pred = _conv_same_torch(x_t, wk)
                wmap = _weight_map(yt, pred, huber_delta, var_map=None, mask=None)
                num += _conv_same_torch(wmap * yt,  wkT)
                den += _conv_same_torch(wmap * pred, wkT)
            upd = torch.clamp(num / (den + EPS), 1.0 / kappa, kappa)
            x_next = torch.clamp(x_t * upd, min=0.0)
            # early stop on clipped update stagnation
            if torch.median(torch.abs(upd - 1)) < 1e-3:
                x_t = x_next
                status_cb(f"MFDeconv: iter {it}/{iters} (early stop)")
                break
            x_t = (1.0 - relax) * x_t + relax * x_next
        else:
            num = np.zeros_like(x_t)
            den = np.zeros_like(x_t)
            for yt, hk, hkT in zip(y_np, psfs, flip_psf):
                pred = _conv_same_np(x_t, hk)
                wmap = _weight_map(yt, pred, huber_delta, var_map=None, mask=None)
                num += _conv_same_np(wmap * yt,  hkT)
                den += _conv_same_np(wmap * pred, hkT)
            upd = np.clip(num / (den + EPS), 1.0 / kappa, kappa)
            x_next = np.clip(x_t * upd, 0.0, None)
            if np.median(np.abs(upd - 1.0)) < 1e-3:
                x_t = x_next
                status_cb(f"MFDeconv: iter {it}/{iters} (early stop)")
                break
            x_t = (1.0 - relax) * x_t + relax * x_next

        if (it % 5) == 0:
            status_cb(f"MFDeconv: iter {it}/{iters}")

    # Save result
    x_final = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)
    # If perchannel (3,H,W), save as (H,W,3) for your viewer
    if x_final.ndim == 3 and x_final.shape[0] == 3:
        x_final = np.moveaxis(x_final, 0, -1)  # (H,W,3)
    try:
        hdr0 = fits.getheader(paths[0], ext=0)
    except Exception:
        hdr0 = fits.Header()
    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution (beta)')
    fits.PrimaryHDU(data=x_final, header=hdr0).writeto(out_path, overwrite=True)
    status_cb(f"✅ MFDeconv saved: {out_path}")
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
        except Exception as e:
            self.finished.emit(False, f"MF deconvolution failed: {e}", "")
