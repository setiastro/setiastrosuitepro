#saspro/wavescale_hdr.py
from __future__ import annotations
import functools
import numpy as np

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLabel, QPushButton,
    QSlider, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
    QMessageBox, QProgressBar
)

# Import centralized widget
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
# Import shared wavelet utilities
from setiastro.saspro.widgets.wavelet_utils import (
    conv_sep_reflect as _conv_sep_reflect,
    build_spaced_kernel as _build_spaced_kernel,
    atrous_decompose as _atrous_decompose,
    atrous_reconstruct as _atrous_reconstruct,
    rgb_to_lab as _rgb_to_lab,
    lab_to_rgb as _lab_to_rgb,
    run_row_bands as _run_row_bands,
    B3_KERNEL as _B3,
    _XYZ_TO_RGB_MATRIX,
    _D65_WHITE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Core math (shared by dialog + headless apply)
# ──────────────────────────────────────────────────────────────────────────────

def _mask_from_L(L: np.ndarray, gamma: float) -> np.ndarray:
    m = np.clip(L / 100.0, 0.0, 1.0).astype(np.float32)
    if gamma != 1.0:
        m = np.power(m, gamma, dtype=np.float32)
    return m

def _apply_dim_curve(rgb: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), gamma, dtype=np.float32)


# ── Headroom limiter ────────────────────────────────────────────────────────
# The detail boost can push a bright patch's L past what its colour can show
# (L > 100, or a linear RGB channel > 1 at the pixel's own a, b). lab_to_rgb
# then clips it flat: a blown, often hue-shifted blob among surroundings the
# tool dimmed and opened up. Where that would happen, a smooth gain map g scales
# the ADDED detail back toward the original L, L' = L0 + g * (L - L0), just far
# enough that nothing leaves the gamut, so the structure inside a limited patch
# is scaled together instead of squashed onto the ceiling. (A per-pixel gain, or
# any per-pixel soft clip, lands every overshooting pixel on or near the ceiling:
# the same flat blob, one notch darker. The base is L0, not the median-aligned
# L: that alignment alone lifts near-white star rims to white.)
# g lives on a 4 px grid: block minimum of each pixel's requirement, a 2-cell
# min-filter, a 1-cell blur, a bilinear upsample (1 cell). Blur + upsample reach
# no further than the min-filter, so g never exceeds any pixel's requirement.
# Pixels with no overshoot nearby keep g = 1 and are bit-identical.
_HEADROOM_BLOCK = 4       # px per gain-map cell
_HEADROOM_ERODE = 2       # cells: min-filter radius (the limiter reaches ~8-12 px)
_HEADROOM_SAT_EPS = 0.05  # L units: originals this close to their ceiling are
                          # already saturated (star cores) and don't steer g
_CEILING_ITERS = 12
_CEILING_CHUNK = 1 << 20


def _gamut_ceiling_L(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     iters: int = _CEILING_ITERS) -> np.ndarray:
    """Largest L in [lo, hi] at which (L, a, b) is still displayable: every
    linear RGB channel <= 1 (both lab_to_rgb paths agree on that test; sRGB
    encoding maps 1 to 1). lo should be displayable (the original L). Returns
    hi where hi itself fits. float64 1-D. Bisection: (hi - lo) / 2**iters."""
    M = _XYZ_TO_RGB_MATRIX.astype(np.float64)
    wx, wy, wz = (float(v) for v in _D65_WHITE)
    d = 6.0 / 29.0

    def finv(t):
        return np.where(t > d, t * t * t, 3.0 * d * d * (t - 4.0 / 29.0))

    def fits(L, ac, bc):
        fy = (L + 16.0) / 116.0
        X = wx * finv(fy + ac / 500.0)
        Y = wy * finv(fy)
        Z = wz * finv(fy - bc / 200.0)
        ok = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z <= 1.0
        ok &= M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z <= 1.0
        ok &= M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z <= 1.0
        return ok

    out = np.empty(a.shape, dtype=np.float64)

    def _band(sl):
        # numpy releases the GIL: bands of candidates run in parallel
        for s in range(sl.start, sl.stop, _CEILING_CHUNK):
            e = min(sl.stop, s + _CEILING_CHUNK)
            ac, bc = a[s:e], b[s:e]
            lo_c, hi_c = lo[s:e].copy(), hi[s:e].copy()
            top_ok = fits(hi_c, ac, bc)
            for _ in range(iters):
                mid = 0.5 * (lo_c + hi_c)
                ok = fits(mid, ac, bc)
                lo_c = np.where(ok, mid, lo_c)
                hi_c = np.where(ok, hi_c, mid)
            out[s:e] = np.where(top_ok, hi[s:e], lo_c)
    _run_row_bands(_band, a.size)
    return out


def _headroom_gain(H: int, W: int, ys: np.ndarray, xs: np.ndarray, gi: np.ndarray) -> np.ndarray:
    """Gain map from sparse requirements gi at (ys, xs), everything else 1.
    Returned padded to whole cells (Hc*4 x Wc*4, float32); index it with the
    image's own (y, x)."""
    import cv2
    s = _HEADROOM_BLOCK
    Hc, Wc = -(-H // s), -(-W // s)
    gc = np.ones((Hc, Wc), dtype=np.float32)
    np.minimum.at(gc, (ys // s, xs // s), gi.astype(np.float32))
    k = 2 * _HEADROOM_ERODE + 1
    gc = cv2.erode(gc, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))  # border: neutral
    gc = cv2.blur(gc, (3, 3), borderType=cv2.BORDER_REPLICATE)
    # exact x4: pixel 4B+j samples cells B-1..B+1 only
    return cv2.resize(gc, (Wc * s, Hc * s), interpolation=cv2.INTER_LINEAR)


def _limit_headroom(rgb: np.ndarray, lab: np.ndarray, L0: np.ndarray, Lb: np.ndarray) -> int:
    """In place on rgb (= lab_to_rgb(lab) with L = clip(Lb, 0, 100)): re-convert
    only the pixels the limiter changes. L0 is the original L, Lb the boosted,
    unclipped L (used as scratch). Returns how many pixels were re-converted."""
    H, W = L0.shape
    cand = np.empty((H, W), dtype=bool)

    def _cand_rows(sl):
        r, c = rgb[sl], cand[sl]
        np.greater(Lb[sl], 100.0, out=c)
        c |= r[..., 0] >= 1.0
        c |= r[..., 1] >= 1.0
        c |= r[..., 2] >= 1.0
    _run_row_bands(_cand_rows, H)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return 0

    labf = lab.reshape(-1, 3)
    L0f, Lbf = L0.ravel(), Lb.ravel()
    L0i = L0f[idx].astype(np.float64)
    Lbi = Lbf[idx].astype(np.float64)
    ab = np.take(labf, idx, axis=0)
    # the original was displayable at this a, b: never cap below it
    cap = np.maximum(_gamut_ceiling_L(ab[:, 1].astype(np.float64), ab[:, 2].astype(np.float64),
                                      L0i, np.clip(Lbi, L0i, 100.0)), L0i)
    del ab
    need = (Lbi > cap) & (L0i < cap - _HEADROOM_SAT_EPS)
    # already-saturated originals (star cores) are simply held at their ceiling;
    # everything that needs limiting stays under it through g <= its requirement
    sat = (Lbi > cap) & ~need
    if sat.any():
        Lbf[idx[sat]] = cap[sat].astype(np.float32)   # Lb is scratch from here on

    gp = None
    sel = idx
    if need.any():
        iy, ix = np.divmod(idx[need], W)
        gi = (cap[need] - L0i[need]) / (Lbi[need] - L0i[need])
        gp = _headroom_gain(H, W, iy, ix, np.clip(gi, 0.0, 1.0))
        upd = gp[:H, :W] < 1.0
        upd.ravel()[idx] = True
        sel = np.flatnonzero(upd)
        del upd

    n = sel.size
    Ls = Lbf[sel]                        # a copy (fancy index)
    if gp is not None:
        sy, sx = np.divmod(sel, W)
        L0s = L0f[sel]
        Ls -= L0s
        Ls *= gp.ravel()[sy * gp.shape[1] + sx]
        Ls += L0s
        del L0s, gp
    np.clip(Ls, 0.0, 100.0, out=Ls)

    # convert as a (rows, 1024, 3) block so the numba kernels run rows in parallel
    cols = 1024
    rows = -(-n // cols)
    blk = np.zeros((rows * cols, 3), dtype=np.float32)
    np.take(labf, sel, axis=0, out=blk[:n])
    blk[:n, 0] = Ls
    conv = _lab_to_rgb(blk.reshape(rows, cols, 3)).reshape(-1, 3)[:n]
    if rgb.flags.c_contiguous:
        rgb.reshape(-1, 3)[sel] = conv
    else:
        ys, xs = np.divmod(sel, W)
        rgb[ys, xs] = conv
    return int(n)


def compute_wavescale_hdr(rgb_image: np.ndarray,
                          n_scales: int = 5,
                          compression_factor: float = 1.5,
                          mask_gamma: float = 1.0,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (transformed_rgb, luminance_mask). transformed_rgb is already
    reconstructed from modified L and gamma-dimmed.
    """
    lab = _rgb_to_lab(rgb_image)
    L0 = lab[..., 0].astype(np.float32, copy=True)
    scales = _atrous_decompose(L0, n_scales, base_kernel)

    mask = _mask_from_L(L0, mask_gamma)
    planes, residual = scales[:-1], scales[-1]

    for i, wp in enumerate(planes):
        decay = decay_rate ** i
        scale = (1.0 + (compression_factor - 1.0) * mask * decay) * 2.0
        planes[i] = wp * scale

    Lr = _atrous_reconstruct(planes + [residual])

    # midtones alignment
    med0 = float(np.median(L0))
    med1 = float(np.median(Lr)) or 1.0
    Lr = np.clip(Lr * (med0 / med1), 0.0, 100.0)

    lab[..., 0] = Lr
    rgb = _lab_to_rgb(lab)

    # gentle dimming curve to tame highlights
    rgb = _apply_dim_curve(rgb, gamma=1.0 + n_scales * 0.2)
    return rgb, mask

def compute_wavescale_hdr(rgb_image: np.ndarray,
                          n_scales: int = 5,
                          compression_factor: float = 1.5,
                          mask_gamma: float = 1.0,
                          base_kernel: np.ndarray = _B3,
                          decay_rate: float = 0.5,
                          dim_gamma: float | None = None,
                          *, lab_cache: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (transformed_rgb, luminance_mask).
    If dim_gamma is None, uses auto gamma = 1.0 + 0.2 * n_scales.
    lab_cache: _rgb_to_lab(rgb_image) computed earlier (copied, not modified).
    Overshoot past white is limited (_limit_headroom), never clipped; pixels
    with no overshoot nearby are unchanged by that step.
    """
    lab = _rgb_to_lab(rgb_image) if lab_cache is None else lab_cache.copy()
    L0 = lab[..., 0].astype(np.float32, copy=True)
    scales = _atrous_decompose(L0, n_scales, base_kernel)
    H = L0.shape[0]

    # The elementwise steps below run on row bands in parallel (numpy releases
    # the GIL); every pixel sees exactly the same float32 operations as one
    # whole-image call.
    mask = np.empty(L0.shape, dtype=np.float32)

    def _mask_rows(sl):
        mask[sl] = _mask_from_L(L0[sl], mask_gamma)
    _run_row_bands(_mask_rows, H)

    planes, residual = scales[:-1], scales[-1]
    Lr = np.empty(residual.shape, dtype=np.float32)

    def _scale_and_rebuild_rows(sl):
        # planes[i] *= (1 + (c-1)·mask·decay^i)·2, then residual + Σ planes
        m = mask[sl]
        out = residual[sl].astype(np.float32, copy=True)
        for i, wp in enumerate(planes):
            decay = decay_rate ** i
            scale = (1.0 + (compression_factor - 1.0) * m * decay) * 2.0
            out += wp[sl] * scale
        Lr[sl] = out
    _run_row_bands(_scale_and_rebuild_rows, H)

    # midtones alignment
    med0 = float(np.median(L0))
    med1 = float(np.median(Lr)) or 1.0
    ratio = med0 / med1

    def _store_L_rows(sl):
        Lr[sl] *= ratio                  # Lr is the boosted L, unclipped, from here on
        lab[sl, :, 0] = np.clip(Lr[sl], 0.0, 100.0)
    _run_row_bands(_store_L_rows, H)
    scales.clear()                       # free the wavelet planes before the RGB passes
    planes.clear()
    residual = None
    rgb = _lab_to_rgb(lab)

    # nothing may clip: where the boost would push a patch past white (or past
    # what its colour can show), scale the added detail back (_limit_headroom)
    _limit_headroom(rgb, lab, L0, Lr)

    # dimming curve
    g = (1.0 + n_scales * 0.2) if dim_gamma is None else float(dim_gamma)
    out_rgb = np.empty(rgb.shape, dtype=np.float32)

    def _dim_rows(sl):
        out_rgb[sl] = _apply_dim_curve(rgb[sl], gamma=g)
    _run_row_bands(_dim_rows, H)
    return out_rgb, mask


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers: plain numpy + QImage, so they run off the GUI thread.
# The GUI then only does QPixmap.fromImage (free for an RGB32 image).
# ──────────────────────────────────────────────────────────────────────────────

def _doc_mask_finish(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Tail of the dialogs' _get_doc_active_mask_2d: clip to [0,1] and resize
    (nearest) to H x W. The heavy part, so the jobs run it off the GUI thread."""
    m = np.clip(m, 0.0, 1.0)
    if m.shape != (H, W):
        yi = (np.linspace(0, m.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, m.shape[1] - 1, W)).astype(np.int32)
        m = m[yi][:, xi]
    return m


def _apply_output(preview: np.ndarray, was_mono: bool, mono_shape) -> np.ndarray:
    """The array Apply hands the document: mono -> channel mean (2-D or H x W x 1
    as the source was), clipped to [0,1], float32. Row bands in parallel; same
    values as np.clip(np.mean(preview, axis=2, dtype=float32), 0, 1)."""
    H = preview.shape[0]
    if was_mono:
        keep3 = bool(mono_shape and len(mono_shape) == 3 and mono_shape[2] == 1)
        out = np.empty(preview.shape[:2] + ((1,) if keep3 else ()), dtype=np.float32)
        o2 = out[:, :, 0] if keep3 else out

        def _rows(sl):
            o2[sl] = np.clip(np.mean(preview[sl], axis=2, dtype=np.float32), 0.0, 1.0)
    else:
        out = np.empty(preview.shape, dtype=np.float32)

        def _rows(sl):
            out[sl] = np.clip(preview[sl], 0.0, 1.0)
    _run_row_bands(_rows, H)
    return out


def clamp_amount(value, default: float = 1.0) -> float:
    """The Amount preset key as the slider holds it: [0, 1] in 0.01 steps.
    Missing or unreadable -> default (1.0 = full effect, the pre-Amount result)."""
    try:
        a = float(value)
    except (TypeError, ValueError):
        a = float(default)
    if not np.isfinite(a):
        a = float(default)
    return int(round(min(1.0, max(0.0, a)) * 100.0)) / 100.0


def blend_amount(original: np.ndarray, result: np.ndarray, amount: float) -> np.ndarray:
    """The Amount slider: (1 - a)*original + a*result. a = 1 returns `result`
    itself, so a full-strength run is bit-identical to the tool before Amount
    existed. Row bands in parallel; float32."""
    a = float(amount)
    if a >= 1.0:
        return result
    out = np.empty(result.shape, dtype=np.float32)
    inv = np.float32(1.0 - a)
    af = np.float32(a)

    def _rows(sl):
        np.add(original[sl] * inv, result[sl] * af, out=out[sl])
    _run_row_bands(_rows, out.shape[0])
    return out


def _amount_job(original: np.ndarray, full: np.ndarray, amount: float, want_qimage: bool) -> dict:
    """Re-blend a finished preview at a new Amount, off the GUI thread."""
    preview = blend_amount(original, full, amount)
    return {"preview": preview, "amount": float(amount),
            "qimg": _rgb_to_qimage(preview) if want_qimage else None}


def _rgb01(img: np.ndarray) -> np.ndarray:
    """np.clip(img as H x W x 3, 0, 1).astype(float32) - what the dialogs built
    with np.repeat (mono: H x W or H x W x 1) + np.clip - on row bands in
    parallel. Same values; the clip runs before the mono replication."""
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        src, mono = (img if img.ndim == 2 else img[:, :, 0]), True
    elif img.ndim == 3 and img.shape[2] == 3:
        src, mono = img, False
    else:
        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    out = np.empty(src.shape[:2] + (3,), dtype=np.float32)

    def _rows(sl):
        c = np.clip(src[sl], 0.0, 1.0)
        if mono:
            for k in range(3):
                out[sl, :, k] = c
        else:
            out[sl] = c
    _run_row_bands(_rows, out.shape[0])
    return out


def _rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """The preview view's image: same pixels _set_pix shows, as an RGB32 QImage
    that owns its data."""
    arr = np.empty(rgb.shape, dtype=np.uint8)

    def _rows(sl):
        arr[sl] = (np.clip(rgb[sl], 0, 1) * 255).astype(np.uint8)
    _run_row_bands(_rows, rgb.shape[0])
    h, w, _ = arr.shape
    q = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q.convertToFormat(QImage.Format.Format_RGB32)


def _mask_thumb_qimage(mask: np.ndarray, box_w: int, box_h: int) -> QImage:
    """The mask window's thumbnail: same pixels MaskDisplayWindow.update_mask shows."""
    m8 = np.empty(mask.shape, dtype=np.uint8)

    def _rows(sl):
        m8[sl] = (np.clip(mask[sl], 0, 1).astype(np.float32) * 255.0).astype(np.uint8)
    _run_row_bands(_rows, mask.shape[0])
    if m8.ndim == 2:
        h, w = m8.shape
        q = QImage(m8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        h, w, _ = m8.shape
        q = QImage(m8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return q.scaled(int(box_w), int(box_h), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)


def _hdr_preview_payload(original: np.ndarray, transformed: np.ndarray, mask: np.ndarray,
                         doc_mask_pre: np.ndarray | None, thumb_wh, want_qimage: bool,
                         amount: float = 1.0) -> dict:
    """What _on_finished used to build on the GUI thread: the doc-mask-combined
    mask, the blended preview, and their display images. "full" is the result
    at Amount 1; "preview" is it blended with the original at `amount`."""
    doc_mask = None if doc_mask_pre is None else _doc_mask_finish(doc_mask_pre, *original.shape[:2])
    # combine HDR's luminance mask with the doc's active mask (if present)
    mask_comb = mask if doc_mask is None else np.clip(mask * doc_mask, 0.0, 1.0)
    # blend preview: original*(1-mask) + transformed*mask, per channel: the same
    # float32 ops as the np.repeat'ed mask, without repeat's GIL-held copy
    inv = 1.0 - mask_comb
    dt = np.result_type(original.dtype, inv.dtype, transformed.dtype, mask_comb.dtype)
    preview = np.empty(original.shape, dtype=dt)
    for c in range(original.shape[2]):
        np.add(original[..., c] * inv, transformed[..., c] * mask_comb, out=preview[..., c])
    shown = blend_amount(original, preview, amount)
    return {
        "full": preview,
        "preview": shown,
        "amount": float(amount),
        "qimg": _rgb_to_qimage(shown) if want_qimage else None,
        "thumb": _mask_thumb_qimage(mask_comb, *thumb_wh) if thumb_wh else None,
        "has_doc_mask": doc_mask is not None,
    }


def _hdr_mask_job(original: np.ndarray, cache: dict, gamma: float,
                  doc_mask_pre: np.ndarray | None, thumb_wh, want_qimage: bool) -> dict:
    """Mask-window refresh (and, first time, the Lab/L cache and the original's
    display image). Only the slider-dependent part is redone per gamma step."""
    out = {}
    doc_mask = None if doc_mask_pre is None else _doc_mask_finish(doc_mask_pre, *original.shape[:2])
    lab, L, mbase = cache.get("lab"), cache.get("L"), cache.get("mbase")
    if L is None:
        lab = _rgb_to_lab(original)
        L = lab[..., 0].astype(np.float32, copy=True)
        out["lab"], out["L"] = lab, L
    if mbase is None:
        # _mask_from_L's gamma-independent part
        mbase = np.clip(L / 100.0, 0.0, 1.0).astype(np.float32)
        out["mbase"] = mbase
    if gamma == 1.0:
        hdr_mask = mbase
    else:
        hdr_mask = np.empty(mbase.shape, dtype=np.float32)

        def _pow_rows(sl):
            hdr_mask[sl] = np.power(mbase[sl], gamma, dtype=np.float32)
        _run_row_bands(_pow_rows, mbase.shape[0])
    mask_comb = hdr_mask if doc_mask is None else np.clip(hdr_mask * doc_mask, 0.0, 1.0)
    out["thumb"] = _mask_thumb_qimage(mask_comb, *thumb_wh)
    out["has_doc_mask"] = doc_mask is not None
    if want_qimage:
        out["qimg"] = _rgb_to_qimage(original)
    return out


class _JobThread(QThread):
    """Runs fn() off the GUI thread; the result arrives via `done` (QThread's own
    `finished` is left alone - thread_keeper uses it to release the thread)."""
    done = pyqtSignal(object, object)                     # (result, error text)

    def __init__(self, fn):
        super().__init__()                                # no parent: thread_keeper owns it
        self._fn = fn

    def run(self):
        try:
            res, err = self._fn(), None
        except Exception as e:
            res, err = None, f"{type(e).__name__}: {e}"
        self._fn = None
        self.done.emit(res, err)


# ──────────────────────────────────────────────────────────────────────────────
# Worker (QObject in its own QThread) for the dialog
# ──────────────────────────────────────────────────────────────────────────────

class HDRWorker(QObject):
    progress_update = pyqtSignal(str, int)                # (step, percent)
    finished = pyqtSignal(object, object)                 # (transformed_rgb, mask) | (payload, None)

    def __init__(self, rgb_image: np.ndarray, n_scales: int, compression_factor: float,
                 mask_gamma: float, base_kernel: np.ndarray, *, post=None,
                 lab_cache: np.ndarray | None = None):
        super().__init__()
        self.rgb_image = rgb_image
        self.n_scales = n_scales
        self.compression_factor = compression_factor
        self.mask_gamma = mask_gamma
        self.base_kernel = base_kernel
        self.post = post              # post(transformed, mask) -> payload dict, run here
        self.lab_cache = lab_cache

    def run(self):
        try:
            self.progress_update.emit(self.tr("Converting to Lab color space…"), 10)
            #sasprogress checkpoints inline here are cosmetic
            self.progress_update.emit(self.tr("Decomposing luminance with starlet…"), 20)
            # full compute
            transformed, mask = compute_wavescale_hdr(
                self.rgb_image, self.n_scales, self.compression_factor, self.mask_gamma, self.base_kernel,
                lab_cache=self.lab_cache
            )
            self.progress_update.emit(self.tr("Finalizing…"), 95)
            if self.post is not None:
                self.finished.emit(self.post(transformed, mask), None)
            else:
                self.finished.emit(transformed, mask)
        except Exception as e:
            print("WaveScale HDR error:", e)
            self.finished.emit(None, None)

# ──────────────────────────────────────────────────────────────────────────────
# Simple mask window
# ──────────────────────────────────────────────────────────────────────────────

class MaskDisplayWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("HDR Mask (L-based)"))
        self.lbl = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl.setFixedSize(400, 400)  # keep it small
        lay = QVBoxLayout(self)
        lay.addWidget(self.lbl)

    def update_mask(self, mask: np.ndarray):
        if mask is None:
            return
        m = np.clip(mask, 0, 1).astype(np.float32)
        m8 = (m * 255.0).astype(np.uint8)
        if m8.ndim == 2:
            h, w = m8.shape
            rgb = np.repeat(m8[..., None], 3, axis=2)
        else:
            h, w, _ = m8.shape
            rgb = m8
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.lbl.setPixmap(pix)

    def set_thumb(self, qimg: QImage):
        """Show a thumbnail already built by _mask_thumb_qimage."""
        self.lbl.setPixmap(QPixmap.fromImage(qimg))

# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class WaveScaleHDRDialogPro(QDialog):
    applied_preset = pyqtSignal(object, dict)

    def __init__(self, parent, doc, icon_path: str | None = None, *, headless: bool=False, bypass_guard: bool=False):
        super().__init__(parent)
        self.setWindowTitle(self.tr("WaveScale HDR"))
        self._headless = bool(headless)
        self._bypass_guard = bool(bypass_guard)
        if self._headless:
            # Don’t show any windows; we’ll still exec() to run the event loop.
            try: self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        if icon_path:
            try: self.setWindowIcon(QIcon(icon_path))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        self.resize(980, 700)
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
        self._doc = doc
        base = getattr(doc, "image", None)
        if base is None:
            raise RuntimeError("Active document has no image.")

        # normalize to float32 [0..1] RGB for processing/preview
        img = np.asarray(base, dtype=np.float32)
        if img.ndim == 2:
            img_rgb = img               # replicated to 3 channels by _rgb01 below
            self._was_mono = True
            self._mono_shape = img.shape
        elif img.ndim == 3 and img.shape[2] == 1:
            img_rgb = img
            self._was_mono = True
            self._mono_shape = img.shape
        else:
            img_rgb = img[:, :, :3]
            self._was_mono = False
            self._mono_shape = None

        if img.dtype.kind in "ui":
            maxv = float(np.nanmax(img_rgb)) or 1.0
            img_rgb = img_rgb / max(1.0, maxv)
        img_rgb = _rgb01(img_rgb)       # == np.clip(3-channel, 0, 1).astype(float32)

        self.original_rgb = img_rgb
        self.preview_rgb  = img_rgb          # never modified in place; replaced by each preview
        self._full_rgb = None                # last preview at Amount 100% (None: no preview yet)
        self._blend_gen = 0                  # bumped by each new preview / Reset; stale re-blends drop
        self._blend_busy = False
        self._blend_pending = False

        # off-GUI-thread state: Lab/L/mask cache, the two view pixmaps, mask jobs
        self._mask_cache = {}
        self._pm_original = None
        self._pm_preview = None
        self._mask_busy = False
        self._mask_pending = False
        self._mask_thr = None

        # scene/view (⚠️ use ZoomableGraphicsView)
        self.scene = QGraphicsScene(self)
        self.view  = ZoomableGraphicsView(self.scene, self)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pix   = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        # optional: keep your scroll area wrapper
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.view)

        # controls (add zoom row)
        self.grp = QGroupBox(self.tr("HDR Controls"))
        form = QFormLayout(self.grp)

        self.s_scales = QSlider(Qt.Orientation.Horizontal); self.s_scales.setRange(2, 10); self.s_scales.setValue(5)
        self.s_comp   = QSlider(Qt.Orientation.Horizontal); self.s_comp.setRange(10, 500); self.s_comp.setValue(150)
        self.s_gamma  = QSlider(Qt.Orientation.Horizontal); self.s_gamma.setRange(10, 1000); self.s_gamma.setValue(500)

        # hover tooltips: what each parameter is and how it affects the image
        _tip_scales = self.tr(
            "Number of Scales (2-10)\n\n"
            "How many wavelet detail layers the luminance is split into. "
            "More scales let the effect reach larger structures (broad spiral "
            "arms, big nebula gradients); fewer keep it focused on fine detail.\n\n"
            "Higher -> larger-scale structure enhanced. Default: 5."
        )
        _tip_comp = self.tr(
            "Coarse Compression (0.10-5.00)\n\n"
            "Strength of the local-contrast boost. It amplifies detail in the "
            "bright regions relative to the broad background, compressing the "
            "overall dynamic range so a bright core reveals structure instead "
            "of clipping to white.\n\n"
            "1.0 = gentle uniform lift; below 1.0 suppresses detail in bright "
            "areas; above ~2.0 can produce halos and a crunchy look. Default: 1.50."
        )
        _tip_gamma = self.tr(
            "Mask Gamma (0.10-10.00)\n\n"
            "Shapes the luminance mask that steers WHERE the effect is applied "
            "(mask = brightness ^ gamma).\n\n"
            "Higher -> effect concentrated on only the brightest regions, e.g. a "
            "galaxy core. Lower -> effect spreads into fainter areas too, which "
            "also lifts background noise. Default: 5.00."
        )
        self.s_scales.setToolTip(_tip_scales)
        self.s_comp.setToolTip(_tip_comp)
        self.s_gamma.setToolTip(_tip_gamma)

        # live value readouts next to each slider
        self.lbl_scales_val = QLabel()
        self.lbl_comp_val   = QLabel()
        self.lbl_gamma_val  = QLabel()
        self.lbl_scales_val.setToolTip(_tip_scales)
        self.lbl_comp_val.setToolTip(_tip_comp)
        self.lbl_gamma_val.setToolTip(_tip_gamma)
        for _l in (self.lbl_scales_val, self.lbl_comp_val, self.lbl_gamma_val):
            _l.setMinimumWidth(42)
            _l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def _slider_row(slider, value_label):
            row = QHBoxLayout()
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            return row

        form.addRow(self.tr("Number of Scales:"), _slider_row(self.s_scales, self.lbl_scales_val))
        form.addRow(self.tr("Coarse Compression:"), _slider_row(self.s_comp, self.lbl_comp_val))
        form.addRow(self.tr("Mask Gamma:"), _slider_row(self.s_gamma, self.lbl_gamma_val))

        # Amount: blend of the finished result with the original (no recompute)
        self.s_amount = QSlider(Qt.Orientation.Horizontal); self.s_amount.setRange(0, 100); self.s_amount.setValue(100)
        self.lbl_amount_val = QLabel()
        _tip_amount = self.tr(
            "Amount (0-100%)\n\n"
            "Blends the finished result with the original image. 100% is the "
            "full effect, 0% leaves the image unchanged.\n\n"
            "Changing it after a Preview re-blends the preview without "
            "recomputing. Default: 100%."
        )
        self.s_amount.setToolTip(_tip_amount)
        self.lbl_amount_val.setToolTip(_tip_amount)
        self.lbl_amount_val.setMinimumWidth(42)
        self.lbl_amount_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.addRow(self.tr("Amount:"), _slider_row(self.s_amount, self.lbl_amount_val))

        # keep readouts in sync with the sliders (comp/gamma are slider value / 100)
        self.s_scales.valueChanged.connect(lambda v: self.lbl_scales_val.setText(str(int(v))))
        self.s_comp.valueChanged.connect(lambda v: self.lbl_comp_val.setText(f"{v / 100.0:.2f}"))
        self.s_gamma.valueChanged.connect(lambda v: self.lbl_gamma_val.setText(f"{v / 100.0:.2f}"))
        self.s_amount.valueChanged.connect(lambda v: self.lbl_amount_val.setText(f"{int(v)}%"))
        # initialize the displayed values
        self.lbl_scales_val.setText(str(int(self.s_scales.value())))
        self.lbl_comp_val.setText(f"{self.s_comp.value() / 100.0:.2f}")
        self.lbl_gamma_val.setText(f"{self.s_gamma.value() / 100.0:.2f}")
        self.lbl_amount_val.setText(f"{int(self.s_amount.value())}%")

        row = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_toggle  = QPushButton(self.tr("Show Original")); self.btn_toggle.setCheckable(True)
        row.addWidget(self.btn_preview); row.addWidget(self.btn_toggle)
        form.addRow(row)

        # ↓ NEW: zoom controls
        zoom_row = QHBoxLayout()
        self.btn_zoom_in  = QPushButton(self.tr("Zoom In"))
        self.btn_zoom_out = QPushButton(self.tr("Zoom Out"))
        self.btn_fit      = QPushButton(self.tr("Fit to Preview"))
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_fit)
        form.addRow(zoom_row)

        #sasprogress group (unchanged)
        self.prog_grp = QGroupBox(self.tr("Processing Progress"))
        vprog = QVBoxLayout(self.prog_grp)
        self.lbl_step = QLabel(self.tr("Idle"))
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setValue(0)
        vprog.addWidget(self.lbl_step); vprog.addWidget(self.bar)

        # bottom buttons (unchanged)
        bot = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply to Document")); self.btn_apply.setEnabled(False)
        self.btn_reset = QPushButton(self.tr("Reset"))
        self.btn_close = QPushButton(self.tr("Close"))
        bot.addStretch(1); bot.addWidget(self.btn_apply); bot.addWidget(self.btn_reset); bot.addWidget(self.btn_close)

        # layout (unchanged)
        main = QVBoxLayout(self)
        main.addWidget(self.scroll)
        h = QHBoxLayout()
        h.addWidget(self.grp, 3)
        h.addWidget(self.prog_grp, 1)
        main.addLayout(h)
        main.addLayout(bot)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import hdr_path
                _grip_icon = QIcon(hdr_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "wavescale_hdr", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a WaveScale HDR shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            self.btn_help = make_help_button("wavescale_hdr", self)
            drag_row.addWidget(self.btn_help)
            
            main.addLayout(drag_row)
        except Exception:
            pass

        # mask window
        self.mask_win = MaskDisplayWindow(self)
        if not self._headless:
            self.mask_win.show()


        # kernel
        self.base_kernel = _B3

        # connections
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_close.clicked.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_toggle.clicked.connect(self._toggle)

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_fit.clicked.connect(lambda: self.view.fit_item(self.pix))

        # ── Mask shown immediately ───────────────────────────────────────────
        # L from the original (Lab), the initial mask and the original's display
        # image are built by a background job; the dialog opens without waiting.
        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.timeout.connect(self._update_mask_from_gamma)
        self.s_gamma.valueChanged.connect(self._schedule_mask_refresh)

        # Amount re-blends the last preview (debounced, off the GUI thread)
        self._blend_timer = QTimer(self)
        self._blend_timer.setSingleShot(True)
        self._blend_timer.timeout.connect(self._update_amount_blend)
        self.s_amount.valueChanged.connect(lambda _v: self._blend_timer.start(120))

        # show initial mask (and the image) as soon as the job is done
        self._update_mask_from_gamma()

    def apply_preset(self, p: dict):
        # sliders are integer; map floats to their scales
        ns = int(p.get("n_scales", 5))
        comp = float(p.get("compression_factor", 1.5))
        mg  = float(p.get("mask_gamma", 5.0))  # dialog default is 5.0 (slider 500)
        # clamp safely
        ns = max(2, min(10, ns))
        comp_i = int(max(10, min(500, round(comp*100))))      # 1.0..5.0 -> 100..500
        mg_i   = int(max(10, min(1000, round(mg*100))))       # 0.1..10.0 -> 10..1000
        self.s_scales.setValue(ns)
        self.s_comp.setValue(comp_i)
        self.s_gamma.setValue(mg_i)
        # presets made before Amount existed have no key: full effect, as they ran then
        self.s_amount.setValue(int(round(clamp_amount(p.get("amount", 1.0)) * 100)))
        # refresh mask preview (even if window is hidden)
        self._update_mask_from_gamma()

    def get_preset(self) -> dict:
        """Emit current slider state as the WaveScale HDR preset schema.
        Exact inverse of apply_preset; mirrors what _apply_to_doc records."""
        return {
            "n_scales": int(self.s_scales.value()),
            "compression_factor": float(self.s_comp.value()) / 100.0,
            "mask_gamma": float(self.s_gamma.value()) / 100.0,
            "amount": float(self.s_amount.value()) / 100.0,
        }

    def seed_from_preset(self, p: dict | None):
        """Alias to apply_preset for the standard opener naming."""
        self.apply_preset(dict(p or {}))

    def _headless_guard_active(self) -> bool:
        """Only guard true concurrent *headless* runs; ignore stale locks."""
        # If we are not launching headless, never block the interactive UI.
        if not self._headless:
            return False

        # Parent flags
        p = self.parent()
        if p and (getattr(p, "_wavescale_guard", False) or getattr(p, "_wavescale_headless_running", False)):
            return True

        # Settings lock with TTL
        try:
            s = QSettings()
            in_prog = bool(s.value("wavescale/headless_in_progress", False))
            started  = float(s.value("wavescale/headless_started_at", 0.0))
        except Exception:
            in_prog, started = False, 0.0

        if not in_prog:
            return False

        # consider anything older than 5 minutes stale
        import time
        if (time.time() - started) > 5 * 60:
            try:
                s.remove("wavescale/headless_in_progress")
                s.remove("wavescale/headless_started_at")
            except Exception:
                pass
            return False

        return True

    def showEvent(self, e):
        super().showEvent(e)
        if not self._bypass_guard and self._headless_guard_active():
            # Soft warning instead of rejecting the dialog
            try:
                QMessageBox.information(
                    self, self.tr("WaveScale HDR"),
                    self.tr("A headless HDR run appears to be in progress. "
                            "This window will remain open; you can still preview safely.")
                )
            except Exception:
                pass

    def exec(self) -> int:
        if not self._bypass_guard and self._headless_guard_active():
            return 0
        return super().exec()

    def _get_doc_active_mask_2d(self) -> np.ndarray | None:
        """
        Return the document's active mask as a 2-D float32 in [0..1],
        resized to the current image size. If none, return None.
        """
        m = self._doc_mask_prenorm()
        if m is None:
            return None
        H, W = self.original_rgb.shape[:2]
        return _doc_mask_finish(m, H, W)

    def _doc_mask_prenorm(self) -> np.ndarray | None:
        """GUI-thread part of _get_doc_active_mask_2d: find the active mask,
        collapse to 2-D float32 and normalise (in place, as before). The clip and
        resize (_doc_mask_finish) are left to the caller / background jobs."""
        doc = getattr(self, "_doc", None)
        if doc is None:
            return None

        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None

        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        if layer is None:
            return None

        # Safely pick the first non-None payload without using boolean 'or'
        data = None
        # object with attributes
        for attr in ("data", "mask", "image", "array"):
            if hasattr(layer, attr):
                val = getattr(layer, attr)
                if val is not None:
                    data = val
                    break
        # plain ndarray?
        if data is None and isinstance(layer, np.ndarray):
            data = layer
        # dict-like layer?
        if data is None and isinstance(layer, dict):
            for key in ("data", "mask", "image", "array"):
                if key in layer and layer[key] is not None:
                    data = layer[key]
                    break

        if data is None:
            return None

        m = np.asarray(data)

        # collapse RGB/alpha to gray if needed
        if m.ndim == 3:
            m = m.mean(axis=2)

        m = m.astype(np.float32, copy=False)
        # normalize to [0,1] if it looks like 0..255 or 0..65535
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0:
                # a new array: astype(copy=False) can hand back the document's
                # own float32 mask, which `m /= mx` divided in place
                m = m / mx
        return m


    def _combine_with_doc_mask(self, hdr_mask: np.ndarray) -> np.ndarray:
        """
        Multiply the HDR luminance mask by the document active mask (if any).
        Shapes are matched to image size.
        """
        m_doc = self._get_doc_active_mask_2d()
        if m_doc is None:
            return hdr_mask
        # both are already (H, W) float32 in [0..1]
        return np.clip(hdr_mask * m_doc, 0.0, 1.0)


    def _set_pix(self, rgb: np.ndarray):
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.pix.setPixmap(QPixmap.fromImage(q))
        self.view.setSceneRect(self.pix.boundingRect())

    def _show_pixmap(self, pm: QPixmap):
        self.pix.setPixmap(pm)
        self.view.setSceneRect(self.pix.boundingRect())

    def _show_original(self):
        if self._pm_original is not None:
            self._show_pixmap(self._pm_original)
        elif self.original_rgb is not None:
            self._set_pix(self.original_rgb)

    def _show_preview(self):
        if self.preview_rgb is self.original_rgb:
            self._show_original()
        elif self._pm_preview is not None:
            self._show_pixmap(self._pm_preview)
        elif self.preview_rgb is not None:
            self._set_pix(self.preview_rgb)

    def _toggle(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setText(self.tr("Show Preview"))
            self._show_original()
        else:
            self.btn_toggle.setText(self.tr("Show Original"))
            self._show_preview()

    def _reset(self):
        self._full_rgb = None
        self._blend_gen += 1                # a re-blend still running is dropped
        self.s_scales.setValue(5)
        self.s_comp.setValue(150)
        self.s_gamma.setValue(500)
        self.s_amount.setValue(100)
        self.preview_rgb = self.original_rgb
        self._pm_preview = None
        self._show_original()
        self.lbl_step.setText(self.tr("Idle")); self.bar.setValue(0)
        self.btn_apply.setEnabled(False)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))

    def _start_preview(self):
        self.btn_preview.setEnabled(False); self.btn_apply.setEnabled(False)
        self.btn_reset.setEnabled(False)    # the running preview owns these settings
        n_scales = int(self.s_scales.value())
        comp     = float(self.s_comp.value()) / 100.0
        mgamma   = float(self.s_gamma.value()) / 100.0

        # blend + display images are built in the worker too; the doc mask is
        # read here, on the GUI thread
        post = functools.partial(
            _hdr_preview_payload, self.original_rgb,
            doc_mask_pre=self._doc_mask_prenorm(),
            thumb_wh=None if self._headless else self._mask_box(),
            want_qimage=not self._headless,
            amount=self._amount(),
        )
        # no parent: thread_keeper holds it, so closing the dialog mid-run is safe
        self.thread = QThread()
        self.worker = HDRWorker(self.original_rgb, n_scales, comp, mgamma, self.base_kernel,
                                post=post, lab_cache=self._mask_cache.get("lab"))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        start_owned_thread(self.thread)

    def _on_progress(self, step: str, pct: int):
        self.lbl_step.setText(step); self.bar.setValue(pct)

    def _on_finished(self, payload, _unused=None):
        if getattr(self, "_memory_released", False):
            return              # dialog closed while the worker ran
        self.btn_preview.setEnabled(True)
        self.btn_reset.setEnabled(True)
        if payload is None:
            QMessageBox.critical(self, self.tr("WaveScale HDR"), self.tr("Processing failed."))
            if self._headless:
                self.reject()   # end the headless exec() loop
            return

        # mask combine + blend + display images were done in the worker
        self._blend_gen += 1                # an older re-blend must not land on this preview
        self._full_rgb = payload["full"]
        self.preview_rgb = payload["preview"]
        qimg = payload.get("qimg")
        self._pm_preview = QPixmap.fromImage(qimg) if qimg is not None else None
        if not self._headless:
            self._show_preview()
            if self._amount() != payload.get("amount"):
                self._update_amount_blend()     # Amount moved while it computed

        # show the *combined* mask in the little window
        self.mask_win.setWindowTitle(
            self.tr("HDR Mask (L × Active Mask)") if payload.get("has_doc_mask") else self.tr("HDR Mask (L-based)")
        )
        if payload.get("thumb") is not None:
            self.mask_win.set_thumb(payload["thumb"])

        self.btn_apply.setEnabled(True)
        self.btn_toggle.setChecked(False); self.btn_toggle.setText(self.tr("Show Original"))
        self.lbl_step.setText(self.tr("Preview ready")); self.bar.setValue(100)
        # Headless: apply immediately (exactly like clicking "Apply to Document")
        if self._headless:
            QTimer.singleShot(0, self._apply_to_doc)

    def _amount(self) -> float:
        return float(self.s_amount.value()) / 100.0

    def _update_amount_blend(self):
        """Re-blend the last preview at the current Amount. Runs in a background
        job; a request made while one runs is queued with the then-current value."""
        if self._headless or getattr(self, "_memory_released", False) or self._full_rgb is None:
            return
        if self._blend_busy:
            self._blend_pending = True
            return
        self._blend_pending = False
        self._blend_busy = True
        self._blend_job_gen = self._blend_gen    # one job at a time, so one slot
        job = functools.partial(_amount_job, self.original_rgb, self._full_rgb, self._amount(), True)
        self._blend_thr = _JobThread(job)
        self._blend_thr.done.connect(self._on_amount_blend_done)
        start_owned_thread(self._blend_thr)

    def _on_amount_blend_done(self, res, err):
        self._blend_busy = False
        if getattr(self, "_memory_released", False):
            return
        if err is not None:
            print("WaveScale HDR amount blend error:", err)
        if res is not None and self._blend_job_gen == self._blend_gen and self._full_rgb is not None:
            self.preview_rgb = res["preview"]
            self._pm_preview = QPixmap.fromImage(res["qimg"]) if res.get("qimg") is not None else None
            if not self.btn_toggle.isChecked():
                self._show_preview()
        if self._blend_pending:
            self._update_amount_blend()

    def _apply_to_doc(self):
        # Amount is blended here from the full result, not read from the shown
        # preview, so a re-blend still in flight cannot make Apply use a stale value.
        rgb = self.preview_rgb if self._full_rgb is None else blend_amount(
            self.original_rgb, self._full_rgb, self._amount())
        # collapse back to mono (keep original shape: 2D or H×W×1), clip to [0,1]
        out = _apply_output(rgb, self._was_mono, self._mono_shape)
        _meta = {"step_name": "WaveScale HDR",
                 "command_id": "wavescale_hdr",
                 "preset": dict(self.get_preset())}
        try:
            self._doc.set_image(out, metadata=_meta, step_name="WaveScale HDR")
        except Exception as e:
            QMessageBox.critical(self, self.tr("WaveScale HDR"), self.tr("Failed to write to document:\n{0}").format(e))
            if self._headless:
                self.reject()   # end the headless exec() loop
            return

        # ── Build preset from current sliders ─────────────────────────
        try:
            preset = dict(self.get_preset())
        except Exception:
            preset = {}

        # ── Register as last_headless_command on the main window ─────
        try:
            main = self.parent()
            if main is not None:
                payload = {
                    "command_id": "wavescale_hdr",
                    "preset": dict(preset),
                }
                setattr(main, "_last_headless_command", payload)

                # Optional debug log (mirrors other tools)
                try:
                    if hasattr(main, "_log"):
                        ns   = int(preset.get("n_scales", 5))
                        comp = float(preset.get("compression_factor", 1.5))
                        mg   = float(preset.get("mask_gamma", 5.0))
                        amt  = float(preset.get("amount", 1.0))
                        main._log(
                            f"[Replay] Registered WaveScale HDR as last action "
                            f"(n_scales={ns}, compression={comp:.2f}, mask_gamma={mg:.2f}, "
                            f"amount={amt:.2f})"
                        )
                except Exception:
                    pass
        except Exception:
            # never let replay wiring break the apply
            pass

        # ── (optional) keep emitting signal if you want it elsewhere ──
        try:
            self.applied_preset.emit(self._doc, preset)
        except Exception:
            pass

        # Applied: the tool's job is done, so it closes (headless too: its
        # runner is blocked in exec() and nothing else would end it).
        self.accept()


    def _schedule_mask_refresh(self, _value):
        # debounce to ~0.25s
        self._mask_timer.start(250)

    def _mask_box(self) -> tuple[int, int]:
        sz = self.mask_win.lbl.size()
        return (sz.width(), sz.height())

    def _update_mask_from_gamma(self):
        """Refresh the mask window. The work runs in a background job; a request
        made while one runs is queued and runs next with the then-current slider."""
        if self._headless or getattr(self, "_memory_released", False):
            return              # headless: the mask window is never shown
        if self._mask_busy:
            self._mask_pending = True
            return
        self._mask_pending = False
        job = functools.partial(
            _hdr_mask_job, self.original_rgb, dict(self._mask_cache),
            float(self.s_gamma.value()) / 100.0,
            self._doc_mask_prenorm(), self._mask_box(),
            self._pm_original is None,
        )
        self._mask_busy = True
        self._mask_thr = _JobThread(job)
        self._mask_thr.done.connect(self._on_mask_job_done)
        start_owned_thread(self._mask_thr)

    def _on_mask_job_done(self, res, err):
        self._mask_busy = False
        if getattr(self, "_memory_released", False):
            return
        if err is not None:
            print("WaveScale HDR mask error:", err)
        if res is not None:
            for k in ("lab", "L", "mbase"):
                if k in res:
                    self._mask_cache[k] = res[k]
            if res.get("qimg") is not None and self._pm_original is None:
                self._pm_original = QPixmap.fromImage(res["qimg"])
                if self.btn_toggle.isChecked() or self.preview_rgb is self.original_rgb:
                    self._show_original()
            self.mask_win.setWindowTitle(
                self.tr("HDR Mask (L × Active Mask)") if res.get("has_doc_mask") else self.tr("HDR Mask (L-based)")
            )
            self.mask_win.set_thumb(res["thumb"])
        if self._mask_pending:
            self._update_mask_from_gamma()

    def closeEvent(self, ev):
        self._close_teardown()
        super().closeEvent(ev)

    def done(self, r):
        # Close / Esc and the headless accept()/reject() end a QDialog through
        # done(), which never delivers closeEvent: without this they skipped
        # the teardown and the dialog kept its full-size buffers. Runs once.
        self._close_teardown()
        super().done(r)

    def _close_teardown(self):
        if getattr(self, "_close_torn_down", False):
            return
        self._close_torn_down = True
        # Running jobs are held by thread_keeper and finish on their own; their
        # results are dropped once the dialog has released its buffers.
        try:
            thr = getattr(self, "thread", None)
            if thread_running(thr):
                thr.quit()
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass

    def _release_memory(self):
        """Drop full-frame RGB/LAB buffers and preview pixmap on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self.original_rgb = None
        self.preview_rgb = None
        self._full_rgb = None
        self._lab_original = None
        self._L_original = None
        self._mask_cache = {}
        self._pm_original = None
        self._pm_preview = None
        self._last_preview = None
        self._doc = None
        self.worker = None
        try:
            if getattr(self, "mask_win", None) is not None:
                self.mask_win.close()
                self.mask_win = None
        except Exception:
            pass
        try:
            self.pix.setPixmap(QPixmap())
        except Exception:
            pass
        try:
            self.scene.clear()
        except Exception:
            pass

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

def open_wavescale_hdr_with_preset(main_window, preset: dict | None = None):
    from PyQt6.QtGui import QIcon  # noqa: F401  (icon_path is a str here)

    # Resolve doc: active MDI subwindow first, then docman fallback.
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    if doc is None or getattr(doc, "image", None) is None:
        return None

    _icon_path = None
    try:
        from setiastro.saspro.resources import hdr_path
        _icon_path = hdr_path
    except Exception:
        _icon_path = None

    # Interactive dialog; bypass the headless-lock guard so opening never blocks.
    dlg = WaveScaleHDRDialogPro(
        main_window, doc, icon_path=_icon_path,
        headless=False, bypass_guard=True,
    )

    # No on-show reset of the sliders -> seed directly before show.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    try:
        main_window._wavescale_hdr_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_wavescale_hdr_dialog", None))
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg        