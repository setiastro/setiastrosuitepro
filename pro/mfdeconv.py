# pro/mfdeconv.py
from __future__ import annotations
import os, math
import numpy as np
from astropy.io import fits
from pro.psf_utils import compute_psf_kernel_for_image  # NEW

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from PyQt6.QtCore import QObject, pyqtSignal

EPS = 1e-6


def _build_psf_bank_from_data(ys_raw, hdrs, ksize=21, status_cb=lambda s: None):
    psfs = []
    for i, (arr, hdr) in enumerate(zip(ys_raw, hdrs), start=1):
        status_cb(f"MFDeconv: measuring PSF {i}/{len(ys_raw)} …")
        try:
            # luma/mono view inside the helper
            psf = compute_psf_kernel_for_image(arr, ksize=ksize, det_sigma=6.0, max_stars=80)
            if psf is None:
                # fallback to Gaussian using header FWHM
                fwhm = _estimate_fwhm_from_header(hdr)
                psf = _gaussian_psf(fwhm, ksize=ksize)
        except Exception:
            fwhm = _estimate_fwhm_from_header(hdr)
            psf = _gaussian_psf(fwhm, ksize=ksize)
        psfs.append(psf.astype(np.float32))
    return psfs

def _to_device(x):
    if TORCH_OK and torch.cuda.is_available():
        return x.cuda(non_blocking=True)
    return x

def _from_device(x):
    if TORCH_OK:
        return x.detach().cpu().numpy()
    return x

def _as_tensor(x):
    if TORCH_OK:
        return torch.from_numpy(x)
    return x  # numpy

def _conv_same(img, psf):
    """
    Depthwise same-sized convolution (H, W) or (C, H, W) * (1, kh, kw).
    Uses torch conv2d if available; else numpy FFT conv.
    """
    if TORCH_OK:
        if img.ndim == 2:
            img_t = torch.from_numpy(img[None, None])  # (N=1,C=1,H,W)
            k_t   = torch.from_numpy(psf[None, None])
            out = torch.nn.functional.conv2d(img_t, k_t, padding='same')
            return out[0,0].numpy()
        else:
            # (C,H,W)
            C = img.shape[0]
            img_t = torch.from_numpy(img[None])        # (1,C,H,W)
            k_t   = torch.from_numpy(psf[None,None])   # shared kernel
            # depthwise: groups=C with same kernel per channel
            w = k_t.repeat(C, 1, 1, 1)                 # (C,1,kh,kw)
            out = torch.nn.functional.conv2d(img_t, w, padding='same', groups=C)
            return out[0].numpy()
    else:
        # numpy FFT fallback
        import numpy.fft as fft
        def fftconv2(a,k):
            H,W = a.shape[-2:]
            kh,kw = k.shape
            pad_h = H+kh-1
            pad_w = W+kw-1
            A = fft.rfftn(a, s=(pad_h,pad_w), axes=(-2,-1))
            K = fft.rfftn(k, s=(pad_h,pad_w), axes=(-2,-1))
            Y = A*K
            y = fft.irfftn(Y, s=(pad_h,pad_w), axes=(-2,-1))
            # center crop to (H,W)
            sh, sw = (kh-1)//2, (kw-1)//2
            return y[..., sh:sh+H, sw:sw+W]
        if img.ndim == 2:
            return fftconv2(img[None], psf)[0]
        else:
            # (C,H,W) conv same kernel per channel
            return np.stack([fftconv2(img[c:c+1], psf)[0] for c in range(img.shape[0])], axis=0)

def _flip_kernel(psf):
    return np.flip(np.flip(psf, -1), -2)

def _gaussian_psf(fwhm_px: float, ksize: int) -> np.ndarray:
    # Create normalized 2D Gaussian with given FWHM in pixels
    sigma = max(fwhm_px, 1.0) / 2.3548
    r = (ksize-1)/2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    g /= np.sum(g) + EPS
    return g.astype(np.float32)

def _estimate_fwhm_from_header(hdr) -> float:
    # Best-effort: use your header fields if present, else fallback
    for key in ("FWHM", "FWHM_PIX", "PSF_FWHM"):
        if key in hdr:
            try:
                val = float(hdr[key])
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    # Dull default
    return 2.5

def _to_luma(img):  # float32 in [0,1] preferred
    if img.ndim == 2: return img
    r,g,b = img[...,0], img[...,1], img[...,2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)

def _stack_loader(paths):
    ys, hdrs = [], []
    for p in paths:
        with fits.open(p, memmap=True) as hdul:
            arr = hdul[0].data.astype(np.float32, copy=False)
            hdr = hdul[0].header
        # Squeeze trailing 1 channel if any
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        ys.append(arr)
        hdrs.append(hdr)
    return ys, hdrs

def _prep_channel_data(ys, mode):
    """
    Returns list of np arrays, each either (H,W) if mode=='luma',
    or (C,H,W) with C=3 for perchannel.
    """
    out = []
    for y in ys:
        if mode == "perchannel" and y.ndim == 3 and y.shape[-1] == 3:
            out.append(np.moveaxis(y, -1, 0))  # (3,H,W)
        else:
            out.append(_to_luma(y))            # (H,W)
    return out

def _build_psf_bank(hdrs, ksize=21):
    psfs = []
    for h in hdrs:
        fwhm = _estimate_fwhm_from_header(h)
        psf  = _gaussian_psf(fwhm, ksize=ksize)
        psfs.append(psf)
    return psfs

def _huber_weights(r, delta):
    # r residual, same shape as prediction; returns weight w = rho'(r)/r
    # Huber: psi(r)=clip(r, -delta, delta); w = psi(r)/r
    if TORCH_OK and isinstance(r, torch.Tensor):
        absr = torch.abs(r)
        w = torch.where(absr <= delta, torch.ones_like(r), delta/(absr+EPS))
        return w
    else:
        absr = np.abs(r)
        w = np.where(absr <= delta, 1.0, delta/(absr+EPS))
        return w

def multiframe_deconv(paths, out_path, iters=20, kappa=2.0, color_mode="luma", huber_delta=0.0, status_cb=lambda s: None):
    status_cb(f"MFDeconv: loading {len(paths)} aligned frames…")
    ys_raw, hdrs = _stack_loader(paths)
    psfs = _build_psf_bank_from_data(ys_raw, hdrs, ksize=21, status_cb=status_cb)

    # Prepare channel layout
    data = _prep_channel_data(ys_raw, color_mode)  # list of (H,W) or (C,H,W)

    # init x0 as median (robust) over frames
    if data[0].ndim == 2:
        stack = np.stack(data, axis=0)  # (T,H,W)
        x = np.median(stack, axis=0).astype(np.float32)
    else:
        stack = np.stack(data, axis=0)  # (T,C,H,W)
        x = np.median(stack, axis=0).astype(np.float32)

    status_cb("MFDeconv: starting multiplicative updates…")
    flip_psf = [_flip_kernel(k) for k in psfs]

    # Torch move (optional)
    use_torch = TORCH_OK
    if use_torch:
        x_t = _to_device(torch.from_numpy(x))
        ones_t = torch.ones_like(x_t)
        y_tensors = []
        for y in data:
            y_tensors.append(_to_device(torch.from_numpy(y)))
    else:
        x_t = x
        ones_t = np.ones_like(x_t)
        y_tensors = data

    for it in range(1, iters+1):
        # numerator = sum_t h_t^T * ( y_t / (h_t * x) )
        # denom     = sum_t h_t^T * 1
        if use_torch:
            num = torch.zeros_like(x_t)
            den = torch.zeros_like(x_t)
            for yt, ht, htT in zip(y_tensors, psfs, flip_psf):
                pred = _as_tensor(_conv_same(_from_device(x_t), ht)) if use_torch else _conv_same(x_t, ht)
                if use_torch:
                    pred = _to_device(pred)
                # robust weight (optional)
                if huber_delta > 0:
                    r = yt - pred
                    w = _huber_weights(r, huber_delta)
                    ratio = (w*yt) / (pred + EPS)
                else:
                    ratio = yt / (pred + EPS)
                back = _as_tensor(_conv_same(_from_device(ratio), htT)) if use_torch else _conv_same(ratio, htT)
                if use_torch:
                    back = _to_device(back)
                num = num + back
                den = den + _to_device(_as_tensor(_conv_same(_from_device(ones_t), htT)))
            upd = num / (den + EPS)
            # clip upd for stability
            upd = torch.clamp(upd, 1.0/kappa, kappa)
            x_t = torch.clamp(x_t * upd, min=0.0)
            if (it % 5) == 0:
                status_cb(f"MFDeconv: iter {it}/{iters}")
        else:
            num = np.zeros_like(x_t)
            den = np.zeros_like(x_t)
            ones = ones_t
            for yt, ht, htT in zip(y_tensors, psfs, flip_psf):
                pred = _conv_same(x_t, ht)
                if huber_delta > 0:
                    r = yt - pred
                    w = _huber_weights(r, huber_delta)
                    ratio = (w*yt) / (pred + EPS)
                else:
                    ratio = yt / (pred + EPS)
                num += _conv_same(ratio, htT)
                den += _conv_same(ones, htT)
            upd = num / (den + EPS)
            upd = np.clip(upd, 1.0/kappa, kappa)
            x_t = np.clip(x_t * upd, 0.0, None)
            if (it % 5) == 0:
                status_cb(f"MFDeconv: iter {it}/{iters}")

    x_final = _from_device(x_t)
    # Save FITS, preserving a reasonable header from the first frame
    try:
        hdr0 = fits.getheader(paths[0], ext=0)
    except Exception:
        hdr0 = fits.Header()
    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution (beta)')
    hdu = fits.PrimaryHDU(data=x_final.astype(np.float32), header=hdr0)
    hdu.writeto(out_path, overwrite=True)
    status_cb(f"✅ MFDeconv saved: {out_path}")
    return out_path


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