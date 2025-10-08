# pro/mfdeconv.py
from __future__ import annotations
import os, math, re
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
from pro.free_torch_memory import _free_torch_memory
torch = None        # filled by runtime loader if available
TORCH_OK = False
NO_GRAD = contextlib.nullcontext  # fallback

from pathlib import Path



_ALLOWED = re.compile(r"[^A-Za-z0-9_-]+")

# known FITS-style multi-extensions (rightmost-first match)
_KNOWN_EXTS = [
    ".fits.fz", ".fit.fz", ".fits.gz", ".fit.gz",
    ".fz", ".gz",
    ".fits", ".fit"
]

def _sanitize_token(s: str) -> str:
    s = _ALLOWED.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _split_known_exts(p: Path) -> tuple[str, str]:
    """
    Return (name_body, full_ext) where full_ext is a REAL extension block
    (e.g. '.fits.fz'). Any junk like '.0s (1310x880)_MFDeconv' stays in body.
    """
    name = p.name
    for ext in _KNOWN_EXTS:
        if name.lower().endswith(ext):
            body = name[:-len(ext)]
            return body, ext
    # fallback: single suffix
    return p.stem, "".join(p.suffixes)

_SIZE_RE = re.compile(r"\(?\s*(\d{2,5})x(\d{2,5})\s*\)?", re.IGNORECASE)
_EXP_RE  = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
_RX_RE   = re.compile(r"(?<![A-Za-z0-9])(\d+)x\b", re.IGNORECASE)

def _extract_size(body: str) -> str | None:
    m = _SIZE_RE.search(body)
    return f"{m.group(1)}x{m.group(2)}" if m else None

def _extract_exposure_secs(body: str) -> str | None:
    m = _EXP_RE.search(body)
    if not m:
        return None
    secs = int(round(float(m.group(1))))
    return f"{secs}s"

def _strip_metadata_from_base(body: str) -> str:
    s = body

    # normalize common separators first
    s = s.replace(" - ", "_")

    # remove known trailing marker '_MFDeconv'
    s = re.sub(r"(?i)[\s_]+MFDeconv$", "", s)

    # remove parenthetical copy counters e.g. '(1)'
    s = re.sub(r"\(\s*\d+\s*\)$", "", s)

    # remove size (with or without parens) anywhere
    s = _SIZE_RE.sub("", s)

    # remove exposures like '0s', '0.5s', ' 45 s' (even if preceded by a dot)
    s = _EXP_RE.sub("", s)

    # remove any _#x tokens
    s = _RX_RE.sub("", s)

    # collapse whitespace/underscores and sanitize
    s = re.sub(r"[\s]+", "_", s)
    s = _sanitize_token(s)
    return s or "output"

def _canonical_out_name_prefix(base: str, r: int, size: str | None,
                               exposure_secs: str | None, tag: str = "MFDeconv") -> str:
    parts = [_sanitize_token(tag), _sanitize_token(base)]
    if size:
        parts.append(_sanitize_token(size))
    if exposure_secs:
        parts.append(_sanitize_token(exposure_secs))
    if int(max(1, r)) > 1:
        parts.append(f"{int(r)}x")
    return "_".join(parts)

def _sr_out_path(out_path: str, r: int) -> Path:
    """
    Build: MFDeconv_<base>[_<HxW>][_<secs>s][_2x], preserving REAL extensions.
    """
    p = Path(out_path)
    body, real_ext = _split_known_exts(p)

    # harvest metadata from the whole body (not Path.stem)
    size   = _extract_size(body)
    ex_sec = _extract_exposure_secs(body)

    # clean base
    base = _strip_metadata_from_base(body)

    new_stem = _canonical_out_name_prefix(base, r=int(max(1, r)), size=size, exposure_secs=ex_sec, tag="MFDeconv")
    return p.with_name(f"{new_stem}{real_ext}")

def _nonclobber_path(path: str) -> str:
    """
    Version collisions as '_v2', '_v3', ... (no spaces/parentheses).
    """
    p = Path(path)
    if not p.exists():
        return str(p)

    # keep the true extension(s)
    body, real_ext = _split_known_exts(p)

    # if already has _vN, bump it
    m = re.search(r"(.*)_v(\d+)$", body)
    if m:
        base = m.group(1); n = int(m.group(2)) + 1
    else:
        base = body; n = 2

    while True:
        candidate = p.with_name(f"{base}_v{n}{real_ext}")
        if not candidate.exists():
            return str(candidate)
        n += 1

def _iter_folder(basefile: str) -> str:
    d, fname = os.path.split(basefile)
    root, ext = os.path.splitext(fname)
    tgt = os.path.join(d, f"{root}.iters")
    if not os.path.exists(tgt):
        try:
            os.makedirs(tgt, exist_ok=True)
        except Exception:
            # last resort: suffix (n)
            n = 1
            while True:
                cand = os.path.join(d, f"{root}.iters ({n})")
                try:
                    os.makedirs(cand, exist_ok=True)
                    return cand
                except Exception:
                    n += 1
    return tgt

def _save_iter_image(arr, hdr_base, folder, tag, color_mode):
    """
    arr: numpy array (H,W) or (C,H,W) float32
    tag: 'seed' or 'iter_###'
    """
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] in (1, 3):
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    hdr = fits.Header(hdr_base) if isinstance(hdr_base, fits.Header) else fits.Header()
    hdr['MF_PART'] = (str(tag), 'MFDeconv intermediate (seed/iter)')
    hdr['MF_COLOR'] = (str(color_mode), 'Color mode used')
    path = os.path.join(folder, f"{tag}.fit")
    # overwrite allowed inside the dedicated folder
    fits.PrimaryHDU(data=arr.astype(np.float32, copy=False), header=hdr).writeto(path, overwrite=True)
    return path


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
        img = _contig(_to_luma_local(arr))             # ← ensure C-contig float32
        bkg = sep.Background(img)
        data = _contig(img - bkg.back())               # ← ensure data is C-contig
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

STAR_MASK_MAXSIDE   = 2048
STAR_MASK_MAXOBJS   = 2000       # cap number of objects
VARMAP_SAMPLE_STRIDE = 8         # (kept for compat; currently unused internally)
THRESHOLD_SIGMA     = 2.0
KEEP_FLOOR          = 0.20
GROW_PX             = 8
MAX_STAR_RADIUS     = 16
SOFT_SIGMA          = 2.0
ELLIPSE_SCALE       = 1.2

def _auto_star_mask_sep(
    img_2d: np.ndarray,
    thresh_sigma: float = THRESHOLD_SIGMA,
    grow_px: int = GROW_PX,
    max_objs: int = STAR_MASK_MAXOBJS,
    max_side: int = STAR_MASK_MAXSIDE,
    ellipse_scale: float = ELLIPSE_SCALE,
    soft_sigma: float = SOFT_SIGMA,
    max_semiaxis_px: float | None = None,
    max_area_px2: float | None = None,
    max_radius_px: int = MAX_STAR_RADIUS,
    keep_floor: float = KEEP_FLOOR,           # <<< NEW
    status_cb=lambda s: None
) -> np.ndarray:
    """
    Build a KEEP mask (1=keep, 0=masked) using simple filled disks + soft edges:
      - SEP detect on the full-resolution image with threshold backoff.
      - For each source: r = ellipse_scale * max(a,b), then clamp to max_radius_px.
      - Dilate by adding grow_px to r (full-res pixels).
      - Draw a filled circle for each detection (cv2 fast path; NumPy fallback).
      - Apply a Gaussian blur of 'soft_sigma' pixels to feather edges (soft weights).
    Emits a status line with counts: detected, kept, drawn, masked area (px).
    """
    if sep is None:
        return np.ones_like(img_2d, dtype=np.float32)

    # Optional OpenCV (fast draw + fast blur)
    try:
        import cv2 as _cv2
        _HAS_CV2 = True
    except Exception:
        _HAS_CV2 = False

    h, w = img_2d.shape
    data = np.ascontiguousarray(img_2d.astype(np.float32, copy=False))

    # Background and residual
    bkg = sep.Background(data)
    data_sub = np.ascontiguousarray(data - bkg.back(), dtype=np.float32)
    try:
        err = bkg.globalrms
    except Exception:
        err = float(np.median(bkg.rms()))

    # Progressive thresholds to avoid explosion on super-dense fields
    thresholds = [thresh_sigma, thresh_sigma * 2, thresh_sigma * 4, thresh_sigma * 8, thresh_sigma * 16]
    objs = None
    used_thresh = float('nan')
    raw_detected = 0
    for t in thresholds:
        try:
            cand = sep.extract(data_sub, thresh=t, err=err)
            count = 0 if (cand is None) else len(cand)
            if count == 0:
                continue
            # If SEP returned an overwhelming number, try a higher threshold
            if count > max_objs * 12:
                continue
            objs = cand
            raw_detected = count
            used_thresh = float(t)
            break
        except Exception:
            continue
    if objs is None or len(objs) == 0:
        # Last-ditch: very high threshold + prune tiny speckles
        try:
            cand = sep.extract(data_sub, thresh=thresholds[-1], err=err, minarea=9)
            if cand is not None and len(cand) > 0:
                objs = cand
                raw_detected = len(cand)
                used_thresh = float(thresholds[-1])
        except Exception:
            objs = None
    if objs is None or len(objs) == 0:
        status_cb("Star mask: no sources found (mask disabled for this frame).")
        return np.ones_like(img_2d, dtype=np.float32)

    # Keep the brightest max_objs
    if "flux" in objs.dtype.names:
        idx = np.argsort(objs["flux"])[-int(max_objs):]
        objs = objs[idx]
    else:
        objs = objs[:int(max_objs)]
    kept_after_cap = int(len(objs))

    # Binary star MASK (1 = masked star region)
    mask = np.zeros((h, w), dtype=np.uint8, order="C")
    MR = int(max(1, max_radius_px))
    G  = int(max(0, grow_px))
    ES = float(max(0.1, ellipse_scale))

    # Draw each source as a filled circle; count how many we actually draw
    drawn = 0
    for o in objs:
        x = int(round(float(o["x"])))
        y = int(round(float(o["y"])))
        if not (0 <= x < w and 0 <= y < h):
            continue

        a = float(o["a"])
        b = float(o["b"])
        r = int(math.ceil(ES * max(a, b)))  # base radius
        if r <= 0:
            continue

        r = min(r + G, MR)                  # clamp & dilate
        if r <= 0:
            continue

        drawn += 1
        if _HAS_CV2:
            _cv2.circle(mask, (x, y), r, 1, thickness=-1, lineType=_cv2.LINE_8)
        else:
            # NumPy fallback: draw in a local box
            y0 = max(0, y - r); y1 = min(h, y + r + 1)
            x0 = max(0, x - r); x1 = min(w, x + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            if yy.size == 0 or xx.size == 0:
                drawn -= 1
                continue
            disk = (yy - y) * (yy - y) + (xx - x) * (xx - x) <= (r * r)
            mask[y0:y1, x0:x1][disk] = 1

    masked_px_hard = int(mask.sum())

    # Convert to float and optionally soften edges (Gaussian feather)
    m = mask.astype(np.float32, copy=False)
    if soft_sigma and soft_sigma > 0.0:
        try:
            if _HAS_CV2:
                # Choose ksize from sigma (≈ 3σ on each side)
                k = int(max(1, math.ceil(soft_sigma * 3)) * 2 + 1)
                m = _cv2.GaussianBlur(m, (k, k), soft_sigma, borderType=_cv2.BORDER_REFLECT)
            else:
                try:
                    from scipy.ndimage import gaussian_filter
                    m = gaussian_filter(m, sigma=float(soft_sigma), mode="reflect")
                except Exception:
                    # Fallback: small Gaussian via our FFT-based conv
                    r = int(max(1, round(3 * float(soft_sigma))))
                    yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
                    g = np.exp(-(xx*xx + yy*yy) / (2.0 * float(soft_sigma) * float(soft_sigma))).astype(np.float32)
                    g /= (g.sum() + EPS)
                    m = _conv_same_np(m, g)
        except Exception:
            # If anything fails, keep the hard mask
            pass

        m = np.clip(m, 0.0, 1.0, out=m)

    # --- NEW: convert to KEEP weights with a floor (down-weight, don't zero) ---
    # KEEP map (1 keep, 0 mask), with configurable floor in [0, 1)
    keep = (1.0 - m).astype(np.float32, copy=False)
    if keep_floor is not None:
        kf = float(max(0.0, min(0.99, keep_floor)))
        keep = kf + (1.0 - kf) * keep          # linear remap to [kf, 1]
        keep = np.clip(keep, 0.0, 1.0, out=keep)

    # Telemetry (add keep_floor too)
    status_cb(
        f"Star mask: thresh={used_thresh:.3g} | detected={raw_detected} | kept={kept_after_cap} | "
        f"drawn={drawn} | masked_px={masked_px_hard} | grow_px={G} | soft_sigma={soft_sigma} | keep_floor={keep_floor}"
    )
    return keep

def _auto_variance_map(
    img_2d: np.ndarray,
    hdr,
    status_cb=lambda s: None,
    sample_stride: int = VARMAP_SAMPLE_STRIDE,  # kept for signature compat; not used
    bw: int = 64,   # SEP background box width  (pixels)
    bh: int = 64,   # SEP background box height (pixels)
    smooth_sigma: float = 1.0,  # Gaussian sigma (px) to smooth the variance map
    floor: float = 1e-8,        # hard floor to prevent blow-up in 1/var
) -> np.ndarray:
    """
    Build a per-pixel variance map in DN^2:

      var_DN ≈ (object_only_DN)/gain  +  var_bg_DN^2

    where:
      - object_only_DN = max(img - sky_DN, 0)
      - var_bg_DN^2 comes from SEP's local background rms (Poisson(sky)+readnoise)
      - if GAIN is missing, estimate 1/gain ≈ median(var_bg)/median(sky)

    Returns float32 array, clipped below by `floor`, optionally smoothed with a
    small Gaussian to stabilize weights. Emits a summary status line.
    """
    img = np.clip(np.asarray(img_2d, dtype=np.float32), 0.0, None)

    # --- Parse header for camera params (optional) ---
    gain = None
    for k in ("EGAIN", "GAIN", "GAIN1", "GAIN2"):
        if k in hdr:
            try:
                g = float(hdr[k])
                if np.isfinite(g) and g > 0:
                    gain = g
                    break
            except Exception:
                pass

    readnoise = None
    for k in ("RDNOISE", "READNOISE", "RN"):
        if k in hdr:
            try:
                rn = float(hdr[k])
                if np.isfinite(rn) and rn >= 0:
                    readnoise = rn
                    break
            except Exception:
                pass

    # --- Local background (full-res) ---
    if sep is not None:
        try:
            b = sep.Background(img, bw=int(bw), bh=int(bh), fw=3, fh=3)
            sky_dn_map = np.asarray(b.back(), dtype=np.float32)
            try:
                rms_dn_map = np.asarray(b.rms(), dtype=np.float32)
            except Exception:
                rms_dn_map = np.full_like(img, float(np.median(b.rms())), dtype=np.float32)
        except Exception:
            sky_dn_map = np.full_like(img, float(np.median(img)), dtype=np.float32)
            med = float(np.median(img))
            mad = float(np.median(np.abs(img - med))) + 1e-6
            rms_dn_map = np.full_like(img, float(1.4826 * mad), dtype=np.float32)
    else:
        sky_dn_map = np.full_like(img, float(np.median(img)), dtype=np.float32)
        med = float(np.median(img))
        mad = float(np.median(np.abs(img - med))) + 1e-6
        rms_dn_map = np.full_like(img, float(1.4826 * mad), dtype=np.float32)

    # Background variance in DN^2
    var_bg_dn2 = np.maximum(rms_dn_map, 1e-6) ** 2

    # Object-only DN
    obj_dn = np.clip(img - sky_dn_map, 0.0, None)

    # Shot-noise coefficient
    if gain is not None and np.isfinite(gain) and gain > 0:
        a_shot = 1.0 / gain
    else:
        sky_med = float(np.median(sky_dn_map))
        varbg_med = float(np.median(var_bg_dn2))
        if sky_med > 1e-6:
            a_shot = np.clip(varbg_med / sky_med, 0.0, 10.0)  # ~ 1/gain estimate
        else:
            a_shot = 0.0

    # Total variance: background + shot noise from object-only flux
    v = var_bg_dn2 + a_shot * obj_dn
    v_raw = v.copy()

    # Optional mild smoothing
    if smooth_sigma and smooth_sigma > 0:
        try:
            import cv2 as _cv2
            k = int(max(1, int(round(3 * float(smooth_sigma)))) * 2 + 1)
            v = _cv2.GaussianBlur(v, (k, k), float(smooth_sigma), borderType=_cv2.BORDER_REFLECT)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter
                v = gaussian_filter(v, sigma=float(smooth_sigma), mode="reflect")
            except Exception:
                r = int(max(1, round(3 * float(smooth_sigma))))
                yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
                gk = np.exp(-(xx*xx + yy*yy) / (2.0 * float(smooth_sigma) * float(smooth_sigma))).astype(np.float32)
                gk /= (gk.sum() + EPS)
                v = _conv_same_np(v, gk)

    # Clip to avoid zero/negative variances
    v = np.clip(v, float(floor), None).astype(np.float32, copy=False)

    # Emit telemetry
    try:
        sky_med  = float(np.median(sky_dn_map))
        rms_med  = float(np.median(np.sqrt(var_bg_dn2)))
        floor_pct = float((v <= floor).mean() * 100.0)
        status_cb(
            "Variance map: "
            f"sky_med={sky_med:.3g} DN | rms_med={rms_med:.3g} DN | "
            f"gain={(gain if gain is not None else 'NA')} | rn={(readnoise if readnoise is not None else 'NA')} | "
            f"smooth_sigma={smooth_sigma} | floor={floor} ({floor_pct:.2f}% at floor)"
        )
    except Exception:
        pass

    return v




def _build_psf_and_assets(
    ys_raw,
    hdrs,
    make_masks=False,
    make_varmaps=False,
    status_cb=lambda s: None,
    save_dir: str | None = None,
    star_mask_cfg: dict | None = None,      # <<< NEW
    varmap_cfg: dict | None = None          # <<< NEW
):
    """
    Like _build_psf_bank_from_data_auto but also (optionally) returns
    per-frame star masks and variance maps (both 2D).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    psfs, masks, vars_ = [], ([] if make_masks else None), ([] if make_varmaps else None)

    for i, (arr, hdr) in enumerate(zip(ys_raw, hdrs), start=1):
        status_cb(f"MFDeconv: measuring PSF {i}/{len(ys_raw)} …")
        _process_gui_events_safely()

        # --- PSF sizing by FWHM ---
        f_hdr = _estimate_fwhm_from_header(hdr)
        f_img = _estimate_fwhm_from_image(arr)
        f_whm = f_hdr if (np.isfinite(f_hdr)) else f_img
        if not np.isfinite(f_whm) or f_whm <= 0:
            f_whm = 2.5
        k_auto = _auto_ksize_from_fwhm(f_whm)

        # --- Star-derived PSF with retries ---
        tried, psf, k_used = [], None, None
        for k_try in [k_auto, max(k_auto - 4, 11), 21, 17, 15, 13, 11]:
            if k_try in tried:
                continue
            tried.append(k_try)
            try:
                out = compute_psf_kernel_for_image(arr, ksize=k_try, det_sigma=6.0, max_stars=80)
                psf_try = out[0] if (isinstance(out, tuple) and len(out) >= 1) else out
                if psf_try is not None:
                    psf = psf_try; k_used = k_try
                    break
            except Exception:
                psf = None

        if psf is None:
            psf = _gaussian_psf(f_whm, ksize=k_auto)

        psf = _soften_psf(_normalize_psf(psf.astype(np.float32, copy=False)), sigma_px=0.25)
        psfs.append(psf)

        # --- Optional assets (2D, same size as arr) ---
        if make_masks or make_varmaps:
            status_cb(f"  PSF{i}: k={psf.shape[0]}  → building assets…")
            _process_gui_events_safely()

        if make_masks:
            smc = star_mask_cfg or {}
            m = _auto_star_mask_sep(
                _to_luma_local(arr),
                status_cb=status_cb,
                thresh_sigma = smc.get("thresh_sigma", THRESHOLD_SIGMA),
                grow_px      = smc.get("grow_px", GROW_PX),
                max_objs     = smc.get("max_objs", STAR_MASK_MAXOBJS),
                ellipse_scale= smc.get("ellipse_scale", ELLIPSE_SCALE),
                soft_sigma   = smc.get("soft_sigma", SOFT_SIGMA),
                max_radius_px= smc.get("max_radius_px", MAX_STAR_RADIUS),
                keep_floor   = smc.get("keep_floor", KEEP_FLOOR),
            )
            masks.append(m)
            status_cb(f"    mask done ({i}/{len(ys_raw)})"); _process_gui_events_safely()

        if make_varmaps:
            vmc = varmap_cfg or {}
            v = _auto_variance_map(
                _to_luma_local(arr), hdr, status_cb=status_cb,
                sample_stride = vmc.get("sample_stride", VARMAP_SAMPLE_STRIDE),
                bw            = vmc.get("bw", 64),
                bh            = vmc.get("bh", 64),
                smooth_sigma  = vmc.get("smooth_sigma", 1.0),
                floor         = vmc.get("floor", 1e-8),
            )
            vars_.append(v)
            status_cb(f"    varmap done ({i}/{len(ys_raw)})"); _process_gui_events_safely()

        # Save PSF if requested
        if save_dir:
            fits.PrimaryHDU(psf).writeto(os.path.join(save_dir, f"psf_{i:03d}.fit"), overwrite=True)

        fwhm_est = _psf_fwhm_px(psf)
        status_cb(f"  PSF{i}: ksize={psf.shape[0]} | FWHM≈{fwhm_est:.2f}px")

    return psfs, masks, vars_



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
    W = [psi(r)/r] * 1/(var + eps) * mask
    If huber_delta < 0, delta = (-huber_delta) * RMS(residual) (auto).
    var_map: per-pixel variance (2D); if None, fall back to robust scalar via MAD.
    mask: 2D {0,1} validity; if None, treat as ones.
    """
    r = y - pred
    eps = EPS

    # resolve Huber delta
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

    # psi(r)/r
    if TORCH_OK and isinstance(r, torch.Tensor):
        absr = torch.abs(r)
        if float(delta) > 0:
            psi_over_r = torch.where(absr <= delta, torch.ones_like(r), delta / (absr + eps))
        else:
            psi_over_r = torch.ones_like(r)
        if var_map is None:
            v = _estimate_scalar_variance_t(r)
        else:
            v = var_map
            if v.ndim == 2 and r.ndim == 3:
                v = v[None, ...]  # broadcast over channels
        w = psi_over_r / (v + eps)
        if mask is not None:
            m = mask if mask.ndim == w.ndim else (mask[None, ...] if w.ndim == 3 else mask)
            w = w * m
        return w
    else:
        absr = np.abs(r)
        if float(delta) > 0:
            psi_over_r = np.where(absr <= delta, 1.0, delta / (absr + eps)).astype(np.float32)
        else:
            psi_over_r = np.ones_like(r, dtype=np.float32)
        if var_map is None:
            v = _estimate_scalar_variance(r)
        else:
            v = var_map
            if v.ndim == 2 and r.ndim == 3:
                v = v[None, ...]
        w = psi_over_r / (v + eps)
        if mask is not None:
            m = mask if mask.ndim == w.ndim else (mask[None, ...] if w.ndim == 3 else mask)
            w = w * m
        return w


# -----------------------------
# Torch / conv
# -----------------------------

def _fftshape_same(H, W, kh, kw):
    return H + kh - 1, W + kw - 1

# ---------- Torch FFT helpers (FIXED: carry padH/padW) ----------
def _precompute_torch_psf_ffts(psfs, flip_psf, H, W, device, dtype):
    tfft = torch.fft
    psf_fft, psfT_fft = [], []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        padH, padW = _fftshape_same(H, W, kh, kw)

        # shift the small kernels to the origin, then FFT into padded size
        k_small  = torch.as_tensor(np.fft.ifftshift(k),  device=device, dtype=dtype)
        kT_small = torch.as_tensor(np.fft.ifftshift(kT), device=device, dtype=dtype)

        Kf  = tfft.rfftn(k_small,  s=(padH, padW))
        KTf = tfft.rfftn(kT_small, s=(padH, padW))

        psf_fft.append((Kf,  padH, padW, kh, kw))
        psfT_fft.append((KTf, padH, padW, kh, kw))
    return psf_fft, psfT_fft


def _fft_conv_same_torch(x, Kf_pack, out_spatial):
    tfft = torch.fft
    Kf, padH, padW, kh, kw = Kf_pack
    H, W = x.shape[-2], x.shape[-1]

    if x.ndim == 2:
        X = tfft.rfftn(x, s=(padH, padW))
        y = tfft.irfftn(X * Kf, s=(padH, padW))
        sh, sw = kh // 2, kw // 2
        out_spatial.copy_(y[sh:sh+H, sw:sw+W])
        return out_spatial
    else:
        X = tfft.rfftn(x, s=(padH, padW), dim=(-2, -1))
        y = tfft.irfftn(X * Kf, s=(padH, padW), dim=(-2, -1))
        sh, sw = kh // 2, kw // 2
        out_spatial.copy_(y[..., sh:sh+H, sw:sw+W])
        return out_spatial

# ---------- NumPy FFT helpers ----------
def _precompute_np_psf_ffts(psfs, flip_psf, H, W):
    import numpy.fft as fft
    meta, Kfs, KTfs = [], [], []
    for k, kT in zip(psfs, flip_psf):
        kh, kw = k.shape
        fftH, fftW = _fftshape_same(H, W, kh, kw)
        Kfs.append( fft.rfftn(np.fft.ifftshift(k),  s=(fftH, fftW)) )
        KTfs.append(fft.rfftn(np.fft.ifftshift(kT), s=(fftH, fftW)) )
        meta.append((kh, kw, fftH, fftW))
    return Kfs, KTfs, meta

def _fft_conv_same_np(a, Kf, kh, kw, fftH, fftW, out):
    import numpy.fft as fft
    if a.ndim == 2:
        A = fft.rfftn(a, s=(fftH, fftW))
        y = fft.irfftn(A * Kf, s=(fftH, fftW))
        sh, sw = kh // 2, kw // 2
        out[...] = y[sh:sh+a.shape[0], sw:sw+a.shape[1]]
        return out
    else:
        C, H, W = a.shape
        acc = []
        for c in range(C):
            A = fft.rfftn(a[c], s=(fftH, fftW))
            y = fft.irfftn(A * Kf, s=(fftH, fftW))
            sh, sw = kh // 2, kw // 2
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

def _ensure_mask_list(masks, data):
    # 1s where valid, 0s where invalid (soft edges allowed)
    if masks is None:
        return [np.ones_like(a if a.ndim==2 else a[0], dtype=np.float32) for a in data]
    out = []
    for a, m in zip(data, masks):
        base = a if a.ndim==2 else a[0]      # mask is 2D; shared across channels
        if m is None:
            out.append(np.ones_like(base, dtype=np.float32))
        else:
            mm = np.asarray(m, dtype=np.float32)
            if mm.ndim == 3:   # tolerate (1,H,W) or (C,H,W)
                mm = mm[0]
            if mm.shape != base.shape:
                # center crop to match (common intersection already applied)
                Ht, Wt = base.shape
                mm = _center_crop(mm, Ht, Wt)
            # keep as float weights in [0,1] (do not threshold!)
            out.append(np.clip(mm.astype(np.float32, copy=False), 0.0, 1.0))
    return out

def _ensure_var_list(variances, data):
    # If None, we’ll estimate a robust scalar per frame on-the-fly.
    if variances is None:
        return [None]*len(data)
    out = []
    for a, v in zip(data, variances):
        if v is None:
            out.append(None)
        else:
            vv = np.asarray(v, dtype=np.float32)
            if vv.ndim == 3:
                vv = vv[0]
            base = a if a.ndim==2 else a[0]
            if vv.shape != base.shape:
                Ht, Wt = base.shape
                vv = _center_crop(vv, Ht, Wt)
            # clip tiny/negatives
            vv = np.clip(vv, 1e-8, None).astype(np.float32, copy=False)
            out.append(vv)
    return out

# ---- SR operators (downsample / upsample-sum) ----
def _downsample_avg(img, r: int):
    """Average-pool over non-overlapping r×r blocks. Works for (H,W) or (C,H,W)."""
    if r <= 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        H, W = a.shape
        Hs, Ws = (H // r) * r, (W // r) * r
        a = a[:Hs, :Ws].reshape(Hs//r, r, Ws//r, r).mean(axis=(1,3))
        return a
    else:
        C, H, W = a.shape
        Hs, Ws = (H // r) * r, (W // r) * r
        a = a[:, :Hs, :Ws].reshape(C, Hs//r, r, Ws//r, r).mean(axis=(2,4))
        return a

def _upsample_sum(img, r: int, target_hw: tuple[int,int] | None = None):
    """Adjoint of average-pooling: replicate-sum each pixel into an r×r block.
       For (H,W) or (C,H,W). If target_hw given, center-crop/pad to that size.
    """
    if r <= 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        H, W = a.shape
        out = np.kron(a, np.ones((r, r), dtype=np.float32))
    else:
        C, H, W = a.shape
        out = np.stack([np.kron(a[c], np.ones((r, r), dtype=np.float32)) for c in range(C)], axis=0)
    if target_hw is not None:
        Ht, Wt = target_hw
        out = _center_crop(out, Ht, Wt)
    return out

def _gaussian2d(ksize: int, sigma: float) -> np.ndarray:
    r = (ksize - 1) // 2
    y, x = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    g = np.exp(-(x*x + y*y)/(2.0*sigma*sigma)).astype(np.float32)
    g /= g.sum() + EPS
    return g

def _conv2_same_np(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    # lightweight wrap for 2D conv on (H,W) or (C,H,W) with same-size output
    return _conv_same_np(a if a.ndim==3 else a[None], k)[0] if a.ndim==2 else _conv_same_np(a, k)

def _solve_super_psf_from_native(f_native: np.ndarray, r: int, sigma: float = 1.1,
                                 iters: int = 500, lr: float = 0.1) -> np.ndarray:
    """
    Solve: h*  = argmin_h || f_native - (D(h) * g_sigma) ||_2^2,
    where h is (r*k)×(r*k) if f_native is k×k. Returns normalized h (sum=1).
    """
    f = np.asarray(f_native, dtype=np.float32)
    k = int(f.shape[0]); assert f.shape[0] == f.shape[1]
    kr = int(k * r)

    # build Gaussian pre-blur at native scale (match paper §4.2)
    g = _gaussian2d(k, max(sigma, 1e-3)).astype(np.float32)

    # init h by zero-insertion (nearest upsample of f) then deconvolving g very mildly
    h0 = np.zeros((kr, kr), dtype=np.float32)
    h0[::r, ::r] = f
    h0 = _normalize_psf(h0)

    if TORCH_OK:
        dev = _torch_device()
        t = torch.tensor(h0, device=dev, dtype=torch.float32, requires_grad=True)
        f_t = torch.tensor(f,  device=dev, dtype=torch.float32)
        g_t = torch.tensor(g,  device=dev, dtype=torch.float32)
        opt = torch.optim.Adam([t], lr=lr)
        for _ in range(max(10, iters)):
            opt.zero_grad(set_to_none=True)
            H, W = t.shape
            Hr, Wr = H//r, W//r
            th = t[:Hr*r, :Wr*r].reshape(Hr, r, Wr, r).mean(dim=(1,3))
            # conv native: (Dh) * g
            conv = torch.nn.functional.conv2d(th[None,None], g_t[None,None], padding=g_t.shape[-1]//2)[0,0]
            loss = torch.mean((conv - f_t)**2)
            loss.backward()
            opt.step()
            with torch.no_grad():
                t.clamp_(min=0.0)
                t /= (t.sum() + 1e-8)
        h = t.detach().cpu().numpy().astype(np.float32)
    else:
        # Tiny gradient-descent fallback on numpy
        h = h0.copy()
        eta = float(lr)
        for _ in range(max(50, iters)):
            Dh = _downsample_avg(h, r)
            conv = _conv2_same_np(Dh, g)
            resid = (conv - f)
            # backprop through conv and D: grad wrt Dh is resid * g^T conv; adjoint of D is upsample-sum
            grad_Dh = _conv2_same_np(resid, np.flip(np.flip(g, 0), 1))
            grad_h  = _upsample_sum(grad_Dh, r, target_hw=h.shape)
            h = np.clip(h - eta * grad_h, 0.0, None)
            s = float(h.sum());  h /= (s + 1e-8)
            eta *= 0.995
    return _normalize_psf(h)

def _downsample_avg_t(x, r: int):
    """
    Average-pool over non-overlapping r×r blocks.
    Works for (H,W) or (C,H,W). Crops to multiples of r.
    """
    if r <= 1:
        return x
    if x.ndim == 2:
        H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x  # nothing to pool
        x2 = x[:Hr, :Wr]
        return x2.view(Hr // r, r, Wr // r, r).mean(dim=(1, 3))
    else:
        C, H, W = x.shape
        Hr, Wr = (H // r) * r, (W // r) * r
        if Hr == 0 or Wr == 0:
            return x
        x2 = x[:, :Hr, :Wr]
        return x2.view(C, Hr // r, r, Wr // r, r).mean(dim=(2, 4))

def _upsample_sum_t(x, r: int):
    if r <= 1:
        return x
    if x.ndim == 2:
        return x.repeat_interleave(r, dim=0).repeat_interleave(r, dim=1)
    else:
        return x.repeat_interleave(r, dim=-2).repeat_interleave(r, dim=-1)

def _sep_bg_rms(frames):
    """Return a robust background RMS using SEP's background model on the first frame."""
    if sep is None or not frames:
        return None
    try:
        y0 = frames[0] if frames[0].ndim == 2 else frames[0][0]  # use luma/first channel
        a = np.ascontiguousarray(y0, dtype=np.float32)
        b = sep.Background(a, bw=64, bh=64, fw=3, fh=3)
        try:
            rms_val = float(b.globalrms)
        except Exception:
            # some SEP builds don’t expose globalrms; fall back to the map’s median
            rms_val = float(np.median(np.asarray(b.rms(), dtype=np.float32)))
        return rms_val
    except Exception:
        return None

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
    masks=None,
    variances=None,
    rho="huber",
    status_cb=lambda s: None,
    min_iters: int = 3,
    use_star_masks: bool = False,
    use_variance_maps: bool = False,
    star_mask_cfg: dict | None = None,
    varmap_cfg: dict | None = None,
    save_intermediate: bool = False,
    save_every: int = 1,
    # >>> SR options
    super_res_factor: int = 1,
    sr_sigma: float = 1.1,
    sr_psf_opt_iters: int = 250,
    sr_psf_opt_lr: float = 0.1,
):
    # sanitize and clamp
    max_iters = max(1, int(iters))
    min_iters = max(1, int(min_iters))
    if min_iters > max_iters:
        min_iters = max_iters

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
    torch = None
    TORCH_OK = False
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

    use_torch = bool(TORCH_OK)
    if use_torch:
        try:
            cudnn = getattr(torch.backends, "cudnn", None)
            if cudnn is not None:
                cudnn.benchmark = True
        except Exception:
            pass
    _process_gui_events_safely()

    # PSFs (auto-size per frame) + flipped copies
    psf_out_dir = None
    psfs, masks_auto, vars_auto = _build_psf_and_assets(
        ys_raw, hdrs,
        make_masks=bool(use_star_masks),
        make_varmaps=bool(use_variance_maps),
        status_cb=status_cb,
        save_dir=psf_out_dir,
        star_mask_cfg=star_mask_cfg,
        varmap_cfg=varmap_cfg
    )

    # >>> SR: lift PSFs to super-res if requested
    r = int(max(1, super_res_factor))
    if r > 1:
        status_cb(f"MFDeconv: Super-resolution r={r} with σ={sr_sigma} — solving SR PSFs…")
        _process_gui_events_safely()
        sr_psfs = []
        for i, k_native in enumerate(psfs, start=1):
            h = _solve_super_psf_from_native(k_native, r=r, sigma=float(sr_sigma),
                                             iters=int(sr_psf_opt_iters), lr=float(sr_psf_opt_lr))
            sr_psfs.append(h)
            status_cb(f"  SR-PSF{i}: native {k_native.shape[0]} → {h.shape[0]} (sum={h.sum():.6f})")
        psfs = sr_psfs

    flip_psf = [_flip_kernel(k) for k in psfs]
    _emit_pct(0.20, "PSF Ready")

    # Normalize layout BEFORE size harmonization
    data = _normalize_layout_batch(ys_raw, color_mode)  # list of (H,W) or (3,H,W)
    _emit_pct(0.25, "Calculating Seed Image...")

    # Center-crop all to common intersection
    Ht, Wt = _common_hw(data)
    if any(((a.shape[-2] != Ht) or (a.shape[-1] != Wt)) for a in data):
        status_cb(f"MFDeconv: Standardizing shapes → crop to {Ht}×{Wt}")
        data = [_center_crop(a, Ht, Wt) for a in data]

    # Numeric hygiene
    data = [_sanitize_numeric(a) for a in data]

    # --- SR/native seed ---
    seed_native = np.median(np.stack(data, axis=0), axis=0).astype(np.float32)
    if r > 1:
        if seed_native.ndim == 2:
            x = _upsample_sum(seed_native / (r*r), r, target_hw=(Ht*r, Wt*r))
        else:
            C, Hn, Wn = seed_native.shape
            x = np.stack(
                [_upsample_sum(seed_native[c] / (r*r), r, target_hw=(Hn*r, Wn*r)) for c in range(C)],
                axis=0
            )
    else:
        x = seed_native
    Hs, Ws = x.shape if x.ndim == 2 else x.shape[-2:]

    # Build mask/var lists (2D) aligned to native 'data'
    auto_masks = masks_auto if use_star_masks else None
    auto_vars  = vars_auto  if use_variance_maps else None
    mask_list = _ensure_mask_list(masks if masks is not None else auto_masks, data)
    var_list  = _ensure_var_list(variances if variances is not None else auto_vars, data)

    iter_dir = None
    hdr0_seed = None
    if save_intermediate:
        iter_dir = _iter_folder(out_path)
        status_cb(f"MFDeconv: Intermediate outputs → {iter_dir}")
        try:
            hdr0_seed = fits.getheader(paths[0], ext=0) if os.path.exists(paths[0]) else fits.Header()
        except Exception:
            hdr0_seed = fits.Header()
        _save_iter_image(x, hdr0_seed, iter_dir, "seed", color_mode)  # save seed

    status_cb("MFDeconv: Calculating Backgrounds and MADs…")
    _process_gui_events_safely()
    bg_est = _sep_bg_rms(data) or (np.median([np.median(np.abs(y - np.median(y))) for y in (data if isinstance(data, list) else [data])]) * 1.4826)
    status_cb(f"MFDeconv: color_mode={color_mode}, huber_delta={huber_delta} (bg RMS~{bg_est:.3g})")
    _process_gui_events_safely()

    status_cb("Computing FFTs and Allocating Scratch…")
    _process_gui_events_safely()

    # -------- precompute and allocate scratch --------
    per_frame_logging = (r > 1)  # only chatty inside SR mode
    if use_torch:
        # tensors
        x_t = _to_t(_contig(x))
        y_tensors = [_to_t(_contig(y)) for y in data]
        mask_tensors = [torch.as_tensor(m, dtype=x_t.dtype, device=x_t.device) for m in mask_list]
        var_tensors  = [None if v is None else torch.as_tensor(v, dtype=x_t.dtype, device=x_t.device)
                        for v in var_list]

        F = torch.nn.functional  # access via runtime torch
        num = torch.zeros_like(x_t)
        den = torch.zeros_like(x_t)

        if r > 1:
            # SR: use FFT packs on the SR grid
            Hs, Ws = (x_t.shape if x_t.ndim == 2 else x_t.shape[-2:])
            psf_fft, psfT_fft = _precompute_torch_psf_ffts(
                psfs, flip_psf, int(Hs), int(Ws), device=x_t.device, dtype=x_t.dtype
            )
            pred_super_t = torch.empty_like(x_t)
            tmp_t        = torch.empty_like(x_t)

            def _upsample_sum_torch(low, r_: int):
                if r_ <= 1:
                    return low
                if low.ndim == 2:
                    return F.interpolate(low[None, None], scale_factor=r_, mode="nearest")[0, 0]
                else:
                    return F.interpolate(low[None], scale_factor=r_, mode="nearest")[0]
        else:
            # Native: stay in spatial conv path (no SR churn)
            psf_t  = [_to_t(_contig(k))[None, None]  for k  in psfs]
            psfT_t = [_to_t(_contig(kT))[None, None] for kT in flip_psf]
            tmp_t  = torch.empty_like(x_t)

    else:
        # NumPy path
        x_t = x
        y_np = data
        if r > 1:
            if x_t.ndim == 2:
                Hs, Ws = x_t.shape
            else:
                _, Hs, Ws = x_t.shape
            Kfs, KTfs, meta = _precompute_np_psf_ffts(psfs, flip_psf, Hs, Ws)
            pred_super = np.empty_like(x_t)
            tmp_out    = np.empty_like(x_t)
        else:
            # native: we can still reuse FFT helper (it uses FFT internally), but no SR down/up
            Kfs, KTfs, meta = _precompute_np_psf_ffts(psfs, flip_psf, Hs, Ws)
            pred_super = np.empty_like(x_t)
            tmp_out    = np.empty_like(x_t)
        num = np.zeros_like(x_t)
        den = np.zeros_like(x_t)

    status_cb("Starting First Multiplicative Iteration…")
    _process_gui_events_safely()

    # -------- inference/no-grad for the whole loop --------
    cm = _safe_inference_context() if use_torch else NO_GRAD
    rho_is_l2 = (str(rho).lower() == "l2")
    local_delta = 0.0 if rho_is_l2 else huber_delta
    used_iters = 0
    early_stopped = False

    # cache a positive Huber δ per frame if auto (update every 5 iters)
    auto_delta_cache = None
    if use_torch and (huber_delta < 0) and (not rho_is_l2):
        auto_delta_cache = [None] * len(ys_raw)

    # early-stop controls
    tol_upd = 1e-3       # threshold on median|upd-1|
    tol_rel = 5e-4       # threshold on relative median change of x
    patience = 2         # require this many consecutive "small" iters
    early_cnt = 0

    with cm():
        for it in range(1, max_iters + 1):
            if use_torch:
                num.zero_(); den.zero_()
                if r > 1:
                    # -------- SR path (FFT + down/up) --------
                    for fidx, (yt, Kf_pack, KTf_pack, mt, vt) in enumerate(
                        zip(y_tensors, psf_fft, psfT_fft, mask_tensors, var_tensors)
                    ):
                        _fft_conv_same_torch(x_t, Kf_pack, pred_super_t)

                        # Correct SR → native downsample (average over r×r)
                        pred_low = _downsample_avg_t(pred_super_t, r)

                        # Robust weights
                        if auto_delta_cache is not None:
                            if (auto_delta_cache[fidx] is None) or (it % 5 == 1):
                                rnat = yt - pred_low
                                med = torch.median(rnat)
                                mad = torch.median(torch.abs(rnat - med)) + 1e-6
                                rms = 1.4826 * mad
                                auto_delta_cache[fidx] = float((-huber_delta) * torch.clamp(rms, min=1e-6).item())
                            wmap_low = _weight_map(yt, pred_low, auto_delta_cache[fidx], var_map=vt, mask=mt)
                        else:
                            wmap_low = _weight_map(yt, pred_low, local_delta, var_map=vt, mask=mt)

                        # Adjoint lift (sum-replicate, not average)
                        up_y    = _upsample_sum_t(wmap_low * yt,       r)
                        up_pred = _upsample_sum_t(wmap_low * pred_low, r)

                        # Accumulate
                        _fft_conv_same_torch(up_y,    KTf_pack, tmp_t); num += tmp_t
                        _fft_conv_same_torch(up_pred, KTf_pack, tmp_t); den += tmp_t

                        if per_frame_logging and ((fidx & 7) == 0):
                            status_cb(f"Iter {it}/{max_iters} — frame {fidx+1}/{len(y_tensors)} (SR grid)")

                else:
                    # -------- Native path (spatial conv, no per-frame chatter) --------
                    for yt, wk, wkT, mt, vt in zip(y_tensors, psf_t, psfT_t, mask_tensors, var_tensors):
                        pred = _conv_same_torch(x_t, wk)
                        wmap = _weight_map(yt, pred, local_delta, var_map=vt, mask=mt)
                        up_y    = wmap * yt
                        up_pred = wmap * pred
                        num += _conv_same_torch(up_y,    wkT)
                        den += _conv_same_torch(up_pred, wkT)

                ratio = num / (den + EPS)
                neutral = (den.abs() < 1e-12) & (num.abs() < 1e-12)
                ratio = torch.where(neutral, torch.ones_like(ratio), ratio)
                upd = torch.clamp(ratio, 1.0 / kappa, kappa)
                x_next = torch.clamp(x_t * upd, min=0.0)

                # --- improved early-stop ---
                upd_med = torch.median(torch.abs(upd - 1))
                rel_change = (torch.median(torch.abs(x_next - x_t)) /
                            (torch.median(torch.abs(x_t)) + 1e-8))
                small = (upd_med < tol_upd) | (rel_change < tol_rel)

                if bool(small) and it >= min_iters:
                    early_cnt += 1
                else:
                    early_cnt = 0

                if early_cnt >= patience:
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    status_cb(f"MFDeconv: Iteration {it}/{max_iters} (early stop)")
                    _process_gui_events_safely()
                    break

                x_t = (1.0 - relax) * x_t + relax * x_next

            else:
                # -------- NumPy path --------
                num.fill(0.0); den.fill(0.0)
                if r > 1:
                    for (Kf, KTf, (kh, kw, fftH, fftW)), m2d, v2d, y_nat in zip(
                        zip(Kfs, KTfs, meta), mask_list, var_list, y_np):
                        _fft_conv_same_np(x_t, Kf, kh, kw, fftH, fftW, pred_super)
                        pred_low = _downsample_avg(pred_super, r)
                        wmap_low = _weight_map(y_nat, pred_low, local_delta, var_map=v2d, mask=m2d)
                        up_y    = _upsample_sum(wmap_low * y_nat,    r, target_hw=pred_super.shape[-2:])
                        up_pred = _upsample_sum(wmap_low * pred_low, r, target_hw=pred_super.shape[-2:])
                        _fft_conv_same_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out); num += tmp_out
                        _fft_conv_same_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out); den += tmp_out
                else:
                    for (Kf, KTf, (kh, kw, fftH, fftW)), m2d, v2d, y_nat in zip(
                        zip(Kfs, KTfs, meta), mask_list, var_list, y_np):
                        _fft_conv_same_np(x_t, Kf, kh, kw, fftH, fftW, pred_super)
                        pred = pred_super
                        wmap = _weight_map(y_nat, pred, local_delta, var_map=v2d, mask=m2d)
                        up_y, up_pred = (wmap * y_nat), (wmap * pred)
                        _fft_conv_same_np(up_y,    KTf, kh, kw, fftH, fftW, tmp_out); num += tmp_out
                        _fft_conv_same_np(up_pred, KTf, kh, kw, fftH, fftW, tmp_out); den += tmp_out

                ratio = num / (den + EPS)
                neutral = (np.abs(den) < 1e-12) & (np.abs(num) < 1e-12)
                ratio[neutral] = 1.0
                upd = np.clip(ratio, 1.0 / kappa, kappa)
                x_next = np.clip(x_t * upd, 0.0, None)

                # --- improved early-stop ---
                upd_med = np.median(np.abs(upd - 1.0))
                rel_change = (np.median(np.abs(x_next - x_t)) /
                            (np.median(np.abs(x_t)) + 1e-8))
                small = (upd_med < tol_upd) or (rel_change < tol_rel)

                if small and it >= min_iters:
                    early_cnt += 1
                else:
                    early_cnt = 0

                if early_cnt >= patience:
                    x_t = x_next
                    used_iters = it
                    early_stopped = True
                    status_cb(f"MFDeconv: Iteration {it}/{max_iters} (early stop)")
                    _process_gui_events_safely()
                    break

                x_t = (1.0 - relax) * x_t + relax * x_next

            # save intermediate
            if save_intermediate and (it % int(max(1, save_every)) == 0):
                try:
                    x_np = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)
                    _save_iter_image(x_np, hdr0_seed, iter_dir, f"iter_{it:03d}", color_mode)
                except Exception as _e:
                    status_cb(f"Intermediate save failed at iter {it}: {_e}")

            # UI throttled — single line per iter (no per-frame spam in native mode)
            frac = 0.25 + 0.70 * (it / float(max_iters))
            _emit_pct(frac, f"Iteration {it}/{max_iters}")
            status_cb(f"Iter {it}/{max_iters}")
            _process_gui_events_safely()

    if not early_stopped:
        used_iters = max_iters


    # ----------------------------
    # Save result (keep FITS-friendly order: (C,H,W))
    # ----------------------------
    _emit_pct(0.97, "saving")
    x_final = x_t.detach().cpu().numpy().astype(np.float32) if use_torch else x_t.astype(np.float32)

    if x_final.ndim == 3:
        if x_final.shape[0] not in (1, 3) and x_final.shape[-1] in (1, 3):
            x_final = np.moveaxis(x_final, -1, 0)  # (C,H,W)
        if x_final.shape[0] == 1:
            x_final = x_final[0]  # (H,W)

    try:
        hdr0 = fits.getheader(paths[0], ext=0)
    except Exception:
        hdr0 = fits.Header()

    hdr0['MFDECONV'] = (True, 'Seti Astro multi-frame deconvolution')
    hdr0['MF_COLOR'] = (str(color_mode), 'Color mode used')
    hdr0['MF_RHO']   = (str(rho), 'Loss: huber|l2')
    hdr0['MF_HDEL']  = (float(huber_delta), 'Huber delta (>0 abs, <0 autoxRMS)')
    hdr0['MF_MASK']  = (bool(use_star_masks),    'Used auto star masks')
    hdr0['MF_VAR']   = (bool(use_variance_maps), 'Used auto variance maps')

    hdr0['MF_SR']     = (int(r), 'Super-resolution factor (1 := native)')
    if r > 1:
        hdr0['MF_SRSIG']  = (float(sr_sigma), 'Gaussian sigma for SR PSF fit (pixels at native)')
        hdr0['MF_SRIT']   = (int(sr_psf_opt_iters), 'SR-PSF solver iters')

    hdr0['MF_ITMAX'] = (int(max_iters),  'Requested max iterations')
    hdr0['MF_ITERS'] = (int(used_iters), 'Actual iterations run')
    hdr0['MF_ESTOP'] = (bool(early_stopped), 'Early stop triggered')

    if isinstance(x_final, np.ndarray):
        if x_final.ndim == 2:
            hdr0['MF_SHAPE'] = (f"{x_final.shape[0]}x{x_final.shape[1]}", 'Saved as 2D image (HxW)')
        elif x_final.ndim == 3:
            C, H, W = x_final.shape
            hdr0['MF_SHAPE'] = (f"{C}x{H}x{W}", 'Saved as 3D cube (CxHxW)')

    save_path     = _sr_out_path(out_path, super_res_factor)
    safe_out_path = _nonclobber_path(str(save_path))
    if safe_out_path != str(save_path):
        status_cb(f"Output exists — saving as: {safe_out_path}")
    fits.PrimaryHDU(data=x_final, header=hdr0).writeto(safe_out_path, overwrite=False)

    status_cb(f"✅ MFDeconv saved: {safe_out_path}  (iters used: {used_iters}{', early stop' if early_stopped else ''})")
    _emit_pct(1.00, "done")
    _process_gui_events_safely()

    try:
        if use_torch:
            # best-effort cleanup (guard names by branch)
            try:
                del num, den
            except Exception:
                pass
            try:
                if r > 1:
                    del pred_super_t, tmp_t, psf_fft, psfT_fft
                else:
                    del tmp_t, psf_t, psfT_t
            except Exception:
                pass
            try:
                del y_tensors, mask_tensors, var_tensors
            except Exception:
                pass
            _free_torch_memory()
    except Exception:
        pass

    return safe_out_path

# -----------------------------
# Worker
# -----------------------------

class MultiFrameDeconvWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str, str)  # success, message, out_path

    def __init__(self, parent, aligned_paths, output_path, iters, kappa, color_mode,
                 huber_delta, min_iters, use_star_masks=False, use_variance_maps=False, rho="huber",
                 star_mask_cfg: dict | None = None, varmap_cfg: dict | None = None,
                 save_intermediate: bool = False,
                 # NEW SR params
                 super_res_factor: int = 1,
                 sr_sigma: float = 1.1,
                 sr_psf_opt_iters: int = 250,
                 sr_psf_opt_lr: float = 0.1):
        super().__init__(parent)
        self.aligned_paths = aligned_paths
        self.output_path = output_path
        self.iters = iters
        self.kappa = kappa
        self.color_mode = color_mode
        self.huber_delta = huber_delta
        self.min_iters = min_iters  # NEW
        self.star_mask_cfg = star_mask_cfg or {}
        self.varmap_cfg    = varmap_cfg or {}
        self.use_star_masks = use_star_masks
        self.use_variance_maps = use_variance_maps
        self.rho = rho
        self.save_intermediate = save_intermediate   
        self.super_res_factor = int(super_res_factor)
        self.sr_sigma = float(sr_sigma)
        self.sr_psf_opt_iters = int(sr_psf_opt_iters)
        self.sr_psf_opt_lr = float(sr_psf_opt_lr)


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
                use_star_masks=self.use_star_masks,
                use_variance_maps=self.use_variance_maps,
                rho=self.rho,
                min_iters=self.min_iters,
                status_cb=self._log,
                star_mask_cfg=self.star_mask_cfg,
                varmap_cfg=self.varmap_cfg,
                save_intermediate=self.save_intermediate,
                # NEW SR forwards
                super_res_factor=self.super_res_factor,
                sr_sigma=self.sr_sigma,
                sr_psf_opt_iters=self.sr_psf_opt_iters,
                sr_psf_opt_lr=self.sr_psf_opt_lr,
            )
            self.finished.emit(True, "MF deconvolution complete.", out)
            _process_gui_events_safely()
        except Exception as e:
            self.finished.emit(False, f"MF deconvolution failed: {e}", "")
