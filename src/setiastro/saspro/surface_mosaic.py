# src/setiastro/saspro/surface_mosaic.py
#
# SetiAstro Suite Pro — Surface Mosaic
# Copyright (C) SetiAstro / Franklin Marek — www.setiastro.com
# Licensed under GPLv3.
#
# Texture-based mosaic builder for lunar / solar / planetary SURFACE panels.
# Unlike Mosaic Master (WCS + stars), there is nothing to plate-solve here, so
# tiles are registered by their surface texture. The registration primitives are
# reused wholesale from the planetary stacker/tracker — this module only adds the
# things a mosaic needs that a stack doesn't: overlap discovery, a global
# (drift-free) placement solve, photometric equalisation, and compositing.
#
# Input tiles are OPEN VIEWS in SASpro (finished per-panel stacks), never files —
# the view I/O mirrors Perfect Palette Picker.
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:  # pragma: no cover - cv2 is a hard dep in practice
    cv2 = None

# ---- soft dependency: scipy only used for the optional similarity solve ----
try:
    from scipy.optimize import least_squares as _scipy_least_squares  # noqa: N813
except Exception:
    _scipy_least_squares = None

# ---------------------------------------------------------------------------
# Reuse everything we can from the planetary stacker/tracker. These are the
# exact same texture-registration kernels the SER stacker trusts; re-deriving
# them here would just be a second thing to keep in sync.
# ---------------------------------------------------------------------------
from setiastro.saspro.ser_tracking import _to_mono01
from setiastro.saspro.ser_stacker import (
    _bandpass,                    # illumination-robust, Hann-windowed match image
    _phase_corr_shift,            # subpixel translation between two same-size patches
    _estimate_rotation_fm,        # Fourier-Mellin rotation estimate
    _refine_shift_ssd,            # gradient-SSD subpixel polish
    _autoplace_aps,               # grid alignment points over signal
    _ap_phase_shifts_per_ap,      # per-AP local shifts (no search)
    _reject_ap_outliers,          # z-score rejection of AP shifts
    _dense_field_from_ap_shifts,  # sparse AP shifts -> dense displacement field
    _warp_by_dense_field,         # apply the dense field (cv2.remap)
    _quality_score,               # per-tile sharpness/feature richness
    _downsample_mono01,           # mono + downsample helper
)


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class SurfaceMosaicConfig:
    # --- overlap discovery (ORB feature matching) ---
    orb_features: int = 4000
    feature_max_dim: int = 1600      # detect features on a downscale this big, scale kps back
    lowe_ratio: float = 0.75
    ransac_thresh: float = 3.0       # px, in feature-detection scale
    min_inliers: int = 12            # below this, the pair is considered non-overlapping
    min_overlap_px: int = 48         # minimum overlap side length for phase-corr refine
    # a candidate pair is only accepted if the actual overlap correlates this well
    # (rejects "ghost" matches — a confident-but-wrong placement on textureless data)
    min_overlap_ncc: float = 0.15
    sift_fallback: bool = True       # if ORB finds nothing, retry with SIFT (blobs, soft texture)
    phasecorr_fallback: bool = True  # exhaustive correlation search (phase-corr + patch NCC)
    limb_fallback: bool = True       # disk geometry (translation only): align solar/lunar limbs
    # matching-only enhancement so soft/unsharpened stacks still register (never
    # touches output pixels — the mosaic is always composited from originals).
    # Reuses SASpro multiscale decomposition: boost a mid-scale detail band that
    # sits above the per-pixel noise (layer 1 is often pure noise on soft data).
    enhance_matching: bool = True
    match_msd_layers: int = 4        # decomposition depth for the match proxy
    # per-layer boost gains (layer 1 = finest .. layer N). Layers 2 & 3 (~2px/4px
    # structure) default to 6; layer 1 (noise) and 4 (coarse) left at 1.
    match_msd_gains: Tuple[float, ...] = (1.0, 6.0, 6.0, 1.0)

    # --- pairwise refine ---
    refine_rotation: bool = True     # estimate per-pair rotation via Fourier-Mellin
    ssd_radius: int = 8

    # --- global solve ---
    solve_rotation: bool = False     # translation-only (v1) unless scipy is present + enabled
    anchor_index: int = 0            # tile pinned as the gauge origin

    # --- local (non-rigid) warp across seams ---
    local_warp: bool = True
    ap_size: int = 64
    ap_spacing: int = 48
    ap_min_mean: float = 0.02
    ap_field_grid: int = 32
    local_warp_min_shift: float = 0.5   # skip the field if median AP shift is below this (px)
    ap_min_struct: float = 0.5          # drop APs with < this fraction of median local structure
    ap_min_response: float = 0.30       # drop APs whose phase-corr response is below this
    seam_confine: bool = True           # confine each seam's warp to a band near it (edges
                                        # don't fight; unmeasurable edges stay soft, never doubled)
    seam_band_mult: float = 2.0         # band width = this * ap_size (medium)

    # --- photometric equalisation (Brown-Lowe gain compensation) ---
    photometric: bool = True
    photo_sigma_n: float = 10.0      # intensity error std (in 0..255-ish units)
    photo_sigma_g: float = 0.10      # gain prior std around 1.0
    radial_model: bool = True        # solar/lunar: equalise gains against a global
                                     # radial limb-darkening profile when a disk was
                                     # found (couples rows with zero overlap).
                                     # GAIN-ONLY: the sky can never be lifted.
    gradient_fit: bool = True        # per-tile quadratic gain SURFACE so overlaps
                                     # match across smooth per-panel gradients
                                     # (etalon/vignette). Multiplicative -> sky stays
                                     # black; limb-masked so no sky pixel is sampled.

    # --- compositing ---
    blend: str = "feather"           # "feather" | "multiband" | "none"
    quality_weighted_seam: bool = True   # let the sharper tile win in overlaps
    multiband_bands: int = 5
    border_value: float = 0.0

    # --- sphere reprojection (v2, hook only for now) ---
    sphere_reproject: bool = False   # TODO: reuse _build_lonlat_grids / derotate_stack_lonshift


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class Tile:
    idx: int
    name: str
    image: np.ndarray                       # (H,W) or (H,W,3) float32 [0..1]
    quality: float = 0.0
    # global pose: translation of the tile's (0,0) into mosaic space, plus rotation (deg)
    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0
    gain: float = 1.0
    bias: float = 0.0
    # log-domain quadratic gain surface coeffs [1,u,v,u^2,uv,v^2] in normalised
    # tile coords u,v in [-1,1]; None = flat (no gradient correction)
    grad: Optional[tuple] = None
    # cached per-detector features {kind: (kps, des)} (filled lazily)
    _feat: Optional[dict] = field(default=None, repr=False)
    # cached small bandpassed proxy for the correlation search: (img, scale)
    _corr: Optional[tuple] = field(default=None, repr=False)
    # cached limb-arc extraction + free circle fit: ("ok", xs, ys, fit) or ("none",)
    _limb: Optional[tuple] = field(default=None, repr=False)
    # cached sharpened mono used for ALL alignment measurement (matching, subpixel
    # refine, local seam warp) — output is always composited from `image`
    _match: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def h(self) -> int:
        return int(self.image.shape[0])

    @property
    def w(self) -> int:
        return int(self.image.shape[1])

    @property
    def is_color(self) -> bool:
        return self.image.ndim == 3 and self.image.shape[2] >= 3


@dataclass
class PairMatch:
    i: int
    j: int
    dx: float          # translation of tile j relative to tile i (mosaic px)
    dy: float
    dtheta: float      # rotation of j relative to i (deg)
    conf: float        # 0..1
    weight: float      # effective weight for the global solve (inliers / overlap area)


@dataclass
class MosaicResult:
    image: np.ndarray                 # (CH,CW) or (CH,CW,3) float32 [0..1]
    coverage: np.ndarray              # (CH,CW) float32 — how many tiles cover each pixel
    tiles: List[Tile]                 # tiles with solved poses/gains
    pairs: List[PairMatch]
    origin: Tuple[float, float]       # canvas (x0, y0) in mosaic coords
    anchor: int = 0                   # tile index pinned as the placement gauge
    unplaced: List[str] = field(default_factory=list)  # tiles that couldn't be matched


# ===========================================================================
# View I/O — tiles come from OPEN VIEWS, never files (Perfect Palette Picker pattern)
# ===========================================================================
def _find_main_window():
    from PyQt6.QtWidgets import QMainWindow, QApplication
    app = QApplication.instance()
    if app is None:
        return None
    aw = app.activeWindow()
    if isinstance(aw, QMainWindow):
        return aw
    for tlw in app.topLevelWidgets():
        if isinstance(tlw, QMainWindow) and hasattr(tlw, "mdi"):
            return tlw
    for tlw in app.topLevelWidgets():
        if isinstance(tlw, QMainWindow):
            return tlw
    return None


def get_doc_manager(explicit=None):
    if explicit is not None:
        return explicit
    mw = _find_main_window()
    if mw is None:
        return None
    return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)


def list_open_views() -> List[Tuple[str, Any]]:
    """
    Return [(title, subwindow), ...] for every open image view.
    Same discovery strategy as Perfect Palette Picker: MDI subWindowList first,
    ImageSubWindow._registry as the reliable fallback.
    """
    mw = _find_main_window()
    if mw is None:
        return []

    try:
        from setiastro.saspro.subwindow import ImageSubWindow
    except Exception:
        ImageSubWindow = None

    out: List[Tuple[str, Any]] = []

    def _title(sub, view, doc) -> str:
        for getter in (
            lambda: (sub.windowTitle() or "").strip(),
            lambda: (view._effective_title() or "").strip(),
            lambda: (doc.display_name() or "").strip(),
        ):
            try:
                t = getter()
                if t:
                    return t
            except Exception:
                continue
        return "Untitled"

    mdi = getattr(mw, "mdi", None)
    if mdi is not None:
        try:
            for sub in mdi.subWindowList():
                try:
                    view = sub.widget()
                    if ImageSubWindow is not None and not isinstance(view, ImageSubWindow):
                        continue
                    doc = getattr(view, "document", None)
                    if doc is None or getattr(doc, "image", None) is None:
                        continue
                    out.append((_title(sub, view, doc), sub))
                except Exception:
                    continue
        except Exception:
            pass

    if not out and ImageSubWindow is not None:
        try:
            for view in list(ImageSubWindow._registry.values()):
                try:
                    doc = getattr(view, "document", None)
                    if doc is None or getattr(doc, "image", None) is None:
                        continue
                    sub = view._mdi_subwindow()
                    if sub is None:
                        continue
                    out.append((_title(sub, view, doc), sub))
                except Exception:
                    continue
        except Exception:
            pass

    # de-dupe titles, keep order
    seen, uniq = set(), []
    for t, sub in out:
        tt = str(t)
        if tt in seen:
            k = 2
            while f"{tt} ({k})" in seen:
                k += 1
            tt = f"{tt} ({k})"
        seen.add(tt)
        uniq.append((tt, sub))
    return uniq


def _as_float01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return (a.astype(np.float32) / 255.0)
    if a.dtype == np.uint16:
        return (a.astype(np.float32) / 65535.0)
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def tile_from_subwindow(idx: int, title: str, sub) -> Optional[Tile]:
    """Pull the document image out of an MDI subwindow and wrap it as a Tile."""
    try:
        view = sub.widget()
        doc = getattr(view, "document", None)
        img = getattr(doc, "image", None) if doc is not None else None
        if img is None:
            return None
    except Exception:
        return None

    img = _as_float01(img)
    # collapse a singleton channel to mono; keep genuine RGB
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]

    t = Tile(idx=idx, name=title, image=img)
    try:
        t.quality = float(_quality_score(_downsample_mono01(img, max_dim=512)))
    except Exception:
        t.quality = 0.0
    return t


def tiles_from_selection(selection: List[Tuple[str, Any]]) -> List[Tile]:
    tiles: List[Tile] = []
    for i, (title, sub) in enumerate(selection):
        t = tile_from_subwindow(i, title, sub)
        if t is not None:
            t.idx = len(tiles)
            tiles.append(t)
    return tiles


def push_mosaic_to_view(mosaic: np.ndarray, title: str = "Surface Mosaic", doc_manager=None):
    """Open the finished mosaic as a brand-new SASpro view (PPP push pattern)."""
    dm = get_doc_manager(doc_manager)
    if dm is None:
        raise RuntimeError("DocManager not found; cannot open mosaic view.")
    is_mono = (mosaic.ndim == 2)
    meta = {"is_mono": bool(is_mono)}
    if hasattr(dm, "open_array"):
        return dm.open_array(mosaic, metadata=meta, title=title)
    if hasattr(dm, "create_document"):
        return dm.create_document(image=mosaic, metadata=meta, name=title)
    raise RuntimeError("DocManager lacks open_array/create_document")


# ===========================================================================
# Stage 2 — Overlap graph (ORB + RANSAC).  NEW work.
# ===========================================================================
def _feature_mono_u8(mono01: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """
    Contrast-normalised uint8 for ORB from an already-mono (already-sharpened)
    image, downscaled to <= max_dim. Returns (u8, scale); full_res = kp / scale.
    """
    m = np.asarray(mono01, dtype=np.float32)
    H, W = m.shape[:2]
    scale = 1.0
    if max(H, W) > max_dim and cv2 is not None:
        scale = max_dim / float(max(H, W))
        m = cv2.resize(m, (max(2, int(W * scale)), max(2, int(H * scale))),
                       interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(m, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1e-3
    u8 = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    return (u8 * 255.0).astype(np.uint8), scale


_MSD_FUNCS = None


def _get_msd_funcs():
    """Lazy import of SASpro's multiscale decomposition (pulls Qt/resources, so
    only load it when matching enhancement is actually used)."""
    global _MSD_FUNCS
    if _MSD_FUNCS is None:
        try:
            from setiastro.saspro.multiscale_decomp import (
                multiscale_decompose, multiscale_reconstruct)
            _MSD_FUNCS = (multiscale_decompose, multiscale_reconstruct)
        except Exception:
            _MSD_FUNCS = (None, None)
    return _MSD_FUNCS


def _enhance_mono_for_features(m01: np.ndarray, cfg: SurfaceMosaicConfig) -> np.ndarray:
    """
    Boost a mid-scale detail band of the MATCHING proxy so soft / unsharpened
    stacks yield ORB features. Reuses SASpro's multiscale decomposition to isolate
    structure at a chosen scale (layer 3 ~ 4px radius, above the per-pixel noise
    that dominates layer 1 on soft data) and amplify it, then reconstruct. Unlike
    a local-contrast stretch, this pulls out structure that isn't already visible
    as contrast. Matching-only — the mosaic is always composited from the
    original pixels — so it can be aggressive without affecting output fidelity.
    """
    if not getattr(cfg, "enhance_matching", True):
        return m01
    decompose, reconstruct = _get_msd_funcs()
    if decompose is None:
        return m01
    layers = max(2, int(cfg.match_msd_layers))
    gains = list(cfg.match_msd_gains)
    try:
        details, residual = decompose(m01.astype(np.float32, copy=False), layers, 1.0)
        for i in range(len(details)):
            g = float(gains[i]) if i < len(gains) else 1.0
            if abs(g - 1.0) > 1e-6:
                details[i] = details[i] * g
        out = reconstruct(details, residual)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    except Exception:
        return m01


def _match_mono(tile: Tile, cfg: SurfaceMosaicConfig) -> np.ndarray:
    """
    Full-res sharpened mono used for EVERY alignment measurement — coarse ORB,
    subpixel refine, and the local seam-warp dense field. Computed once and
    cached. The mosaic is never built from this; output always comes from the
    original tile.image. With enhance_matching off this is just the plain mono,
    so behaviour reduces to measuring on the originals.
    """
    if tile._match is None:
        m = _to_mono01(tile.image).astype(np.float32, copy=False)
        tile._match = _enhance_mono_for_features(m, cfg)
    return tile._match


def _ensure_features(tile: Tile, cfg: SurfaceMosaicConfig, kind: str = "orb"):
    """
    Lazily detect+cache features of the given kind on the sharpened proxy.
    "orb": fast corner features — great on crisp texture (craters, granulation).
    "sift": DoG blob features — far better on soft, low-contrast mottling where
    ORB's corner detector finds nothing (this is what makes Photoshop-style
    auto-align work on mushy data). Returns (kps, des); des is None on failure.
    """
    if cv2 is None:
        return [], None
    if tile._feat is None:
        tile._feat = {}
    if kind in tile._feat:
        return tile._feat[kind]
    u8, scale = _feature_mono_u8(_match_mono(tile, cfg), cfg.feature_max_dim)
    det = None
    if kind == "orb":
        det = cv2.ORB_create(nfeatures=int(cfg.orb_features))
    elif kind == "sift" and hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=int(cfg.orb_features))
    if det is None:
        tile._feat[kind] = ([], None)
        return tile._feat[kind]
    try:
        kps, des = det.detectAndCompute(u8, None)
    except Exception:
        kps, des = None, None
    # scale keypoints back to full-res tile coordinates
    if kps and scale != 1.0:
        inv = 1.0 / scale
        for kp in kps:
            kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)
    tile._feat[kind] = (list(kps) if kps else [], des)
    return tile._feat[kind]


def _match_pair_features(ti: Tile, tj: Tile, cfg: SurfaceMosaicConfig,
                         kind: str = "orb") -> Optional[PairMatch]:
    """
    Coarse pairwise similarity (rotation+scale+translation) from feature matches.
    Returns j-relative-to-i translation/rotation, or None if the pair doesn't overlap.
    """
    if cv2 is None:
        return None
    kpi, di = _ensure_features(ti, cfg, kind)
    kpj, dj = _ensure_features(tj, cfg, kind)
    if di is None or dj is None or len(kpi) < 4 or len(kpj) < 4:
        return None

    norm = cv2.NORM_HAMMING if kind == "orb" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    try:
        knn = bf.knnMatch(dj, di, k=2)   # query=j, train=i  -> maps j into i
    except Exception:
        return None

    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < cfg.lowe_ratio * n.distance:
            good.append(m)
    if len(good) < cfg.min_inliers:
        return None

    src = np.float32([kpj[m.queryIdx].pt for m in good])  # in j
    dst = np.float32([kpi[m.trainIdx].pt for m in good])  # in i

    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=float(cfg.ransac_thresh),
    )
    if M is None or inliers is None:
        return None
    n_in = int(inliers.sum())
    if n_in < cfg.min_inliers:
        return None

    # M maps a point in j -> its location in i. Translation part places j's origin in i.
    dx = float(M[0, 2])
    dy = float(M[1, 2])
    dtheta = math.degrees(math.atan2(float(M[1, 0]), float(M[0, 0])))
    conf = float(np.clip(n_in / max(1, len(good)), 0.0, 1.0))
    return PairMatch(i=ti.idx, j=tj.idx, dx=dx, dy=dy, dtheta=dtheta,
                     conf=conf, weight=float(n_in))


# ===========================================================================
# Stage 3 — Pairwise refine on the overlap.  Reuses stacker primitives.
# ===========================================================================
def _integer_overlap(wi: int, hi: int, wj: int, hj: int, rdx: int, rdy: int):
    """
    Tile j sits at (rdx,rdy) relative to i (i.e. i-coord = j-coord + (rdx,rdy)).
    Return crop boxes (xi0,yi0, xj0,yj0, ow,oh) for the mutual overlap, or None.
    """
    xi0 = max(0, rdx)
    yi0 = max(0, rdy)
    xi1 = min(wi, wj + rdx)
    yi1 = min(hi, hj + rdy)
    ow, oh = xi1 - xi0, yi1 - yi0
    if ow <= 0 or oh <= 0:
        return None
    xj0 = xi0 - rdx
    yj0 = yi0 - rdy
    return xi0, yi0, xj0, yj0, ow, oh


def _refine_pair(ti: Tile, tj: Tile, coarse: PairMatch, cfg: SurfaceMosaicConfig) -> PairMatch:
    """
    Sub-pixel refine the coarse pair using phase correlation + FM rotation + SSD
    on the mutual overlap. Falls back to the coarse estimate if the overlap is too
    small to be reliable.

    The overlap here is extracted by translation only, so it is only valid when
    the pair has little relative rotation. For a genuinely rotated pair the crop
    would compare rotated-vs-unrotated content and phase correlation would return
    noise — so above a few degrees we trust the ORB similarity fit (which used the
    whole overlap, rotation-invariantly) and let the local seam warp clean up any
    residual at compose time.
    """
    if abs(coarse.dtheta) > 3.0:
        return coarse

    mi = _match_mono(ti, cfg)
    mj = _match_mono(tj, cfg)

    rdx, rdy = int(round(coarse.dx)), int(round(coarse.dy))
    box = _integer_overlap(ti.w, ti.h, tj.w, tj.h, rdx, rdy)
    if box is None:
        return coarse
    xi0, yi0, xj0, yj0, ow, oh = box
    if min(ow, oh) < cfg.min_overlap_px:
        return coarse

    ci = mi[yi0:yi0 + oh, xi0:xi0 + ow]
    cj = mj[yj0:yj0 + oh, xj0:xj0 + ow]

    # illumination-robust, windowed match images (handles the terminator gradient)
    bi = _bandpass(ci)
    bj = _bandpass(cj)

    # subpixel translation residual: shift cj by (sdx,sdy) to match ci
    sdx, sdy, resp = _phase_corr_shift(bi, bj)

    # polish with gradient-SSD
    try:
        rdx2, rdy2, sconf = _refine_shift_ssd(bi, bj, sdx, sdy, radius=int(cfg.ssd_radius))
        sdx += float(rdx2)
        sdy += float(rdy2)
    except Exception:
        sconf = resp

    dtheta = coarse.dtheta
    if cfg.refine_rotation:
        try:
            ang, aconf = _estimate_rotation_fm(bi, bj)
            if aconf > 0.15 and abs(ang) < 15.0:
                dtheta = float(ang)
        except Exception:
            pass

    # residual adds to the coarse relative translation (j relative to i)
    dx = float(coarse.dx + sdx)
    dy = float(coarse.dy + sdy)
    conf = float(np.clip(0.5 * resp + 0.5 * sconf, 0.0, 1.0))
    # weight blends inlier count with overlap area and correlation confidence
    weight = float(coarse.weight * (0.5 + 0.5 * conf) * (1.0 + math.log1p(ow * oh) / 20.0))
    return PairMatch(i=coarse.i, j=coarse.j, dx=dx, dy=dy, dtheta=dtheta,
                     conf=conf, weight=weight)


def _pair_overlap_ncc(ti: Tile, tj: Tile, pair: PairMatch, cfg: SurfaceMosaicConfig) -> float:
    """
    Normalised cross-correlation of the actual overlap after applying `pair`,
    measured on the sharpened match images. A genuine overlap correlates high; a
    ghost / wrong placement sits near zero. Returns NCC in [-1, 1], or -1 if the
    implied overlap is too small to judge.
    """
    if cv2 is None:
        return -1.0
    mi = _match_mono(ti, cfg).astype(np.float32, copy=False)
    mj = _match_mono(tj, cfg).astype(np.float32, copy=False)
    # High-pass both first. Without this, a smooth bright disk (solar/lunar limb)
    # correlates at almost any offset and would validate a ghost — the true
    # overlap only stands out in the high-frequency structure. Bandpass the source
    # content (no Hann window) so the warp's zero border doesn't taint the blur.
    hi_i = cv2.GaussianBlur(mi, (0, 0), 1.2) - cv2.GaussianBlur(mi, (0, 0), 6.0)
    hi_j = cv2.GaussianBlur(mj, (0, 0), 1.2) - cv2.GaussianBlur(mj, (0, 0), 6.0)
    th = math.radians(pair.dtheta)
    c, s = math.cos(th), math.sin(th)
    M = np.array([[c, -s, pair.dx], [s, c, pair.dy]], dtype=np.float32)  # j -> i frame
    jw = cv2.warpAffine(hi_j, M, (ti.w, ti.h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    vj = cv2.warpAffine(np.ones((tj.h, tj.w), np.float32), M, (ti.w, ti.h),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mask = vj > 0.5
    if int(mask.sum()) < cfg.min_overlap_px * cfg.min_overlap_px:
        return -1.0
    a = hi_i[mask].astype(np.float64)
    b = jw[mask].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-9
    return float((a * b).sum() / denom)


def _corr_small(tile: Tile, cfg: SurfaceMosaicConfig):
    """Cached small (<=512px) proxy for the correlation search: light denoise +
    WIDE high-pass. A narrow bandpass can sit below soft data's structure scale
    and keep only noise; subtracting a 16-sigma blur removes just the smooth
    illumination (disk falloff) and keeps structure at every other scale.
    Returns (img, scale) where full_res = small_coord / scale."""
    if tile._corr is None:
        m = _match_mono(tile, cfg)
        H, W = m.shape[:2]
        sc = min(1.0, 512.0 / float(max(H, W)))
        if sc < 1.0:
            sm = cv2.resize(m, (max(8, int(W * sc)), max(8, int(H * sc))),
                            interpolation=cv2.INTER_AREA)
        else:
            sm = m.astype(np.float32, copy=True)
        sm = cv2.GaussianBlur(sm, (0, 0), 1.0)              # denoise
        hp = sm - cv2.GaussianBlur(sm, (0, 0), 16.0)        # remove illumination only
        hp -= float(hp.mean())
        hp /= (float(hp.std()) + 1e-6)
        tile._corr = (hp.astype(np.float32), sc)
    return tile._corr


def _coarse_ncc(bi: np.ndarray, bj: np.ndarray, dx: float, dy: float) -> float:
    """Quick overlap NCC at the small scale for a candidate translation."""
    h, w = bi.shape
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    jw = cv2.warpAffine(bj, M, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    vj = cv2.warpAffine(np.ones_like(bj), M, (w, h), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    m = vj > 0.5
    if int(m.sum()) < 24 * 24:
        return -1.0
    a = bi[m].astype(np.float64); b = jw[m].astype(np.float64)
    a -= a.mean(); b -= b.mean()
    den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-9
    return float((a * b).sum() / den)


def _match_pair_corrsearch(ti: Tile, tj: Tile, cfg: SurfaceMosaicConfig) -> Optional[PairMatch]:
    """
    Exhaustive translation search for when no feature detector bites. Full-frame
    phase correlation alone is weak on mosaic panels (it assumes the two frames
    are mostly the same image, but panels share only a partial overlap), so this
    slides a 3x3 grid of patches from tile j over tile i with normalised
    cross-correlation (cv2.matchTemplate) — partial overlap is handled by
    construction, whichever edge or corner it lives on. Candidates from the
    patch search plus a phase-corr attempt are scored by coarse overlap NCC and
    the best is returned (full validation happens upstream). Translation only.
    """
    if cv2 is None:
        return None
    bi, si = _corr_small(ti, cfg)
    bj, sj = _corr_small(tj, cfg)
    if bi.shape != bj.shape or abs(si - sj) > 1e-9:
        return None   # mosaic panels from one camera share a size; keep it simple

    cands: List[Tuple[float, float]] = []
    try:
        dx, dy, _resp = _phase_corr_shift(bi, bj)
        cands.append((float(dx), float(dy)))
    except Exception:
        pass

    H, W = bj.shape
    gstd = float(bj.std()) + 1e-6
    # two patch sizes: coarse for wide overlaps, fine so a patch still fits
    # entirely inside a thin (~25%) overlap strip; grid reaches every edge
    for frac in (0.4, 0.22):
        ph, pw = int(H * frac), int(W * frac)
        if ph < 12 or pw < 12 or bi.shape[0] <= ph or bi.shape[1] <= pw:
            continue
        for fy in (0.0, 0.39, 0.78):
            for fx in (0.0, 0.39, 0.78):
                y0 = min(int(fy * H), H - ph)
                x0 = min(int(fx * W), W - pw)
                patch = bj[y0:y0 + ph, x0:x0 + pw]
                if float(patch.std()) < 0.25 * gstd:
                    continue   # flat patch (sky / featureless) can't localise
                try:
                    res = cv2.matchTemplate(bi, patch, cv2.TM_CCOEFF_NORMED)
                except Exception:
                    continue
                _mn, mx, _ml, loc = cv2.minMaxLoc(res)
                if mx < 0.2:
                    continue
                cands.append((float(loc[0] - x0), float(loc[1] - y0)))

    if not cands:
        return None

    # dedup (3px bins at small scale), then score every candidate by coarse NCC
    seen, uniq = set(), []
    for dx, dy in cands:
        key = (round(dx / 3.0), round(dy / 3.0))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((dx, dy))
    best = None
    for dx, dy in uniq:
        ncc = _coarse_ncc(bi, bj, dx, dy)
        if best is None or ncc > best[2]:
            best = (dx, dy, ncc)
    if best is None or best[2] < max(0.08, 0.5 * float(cfg.min_overlap_ncc)):
        return None
    dx, dy, ncc = best
    return PairMatch(i=ti.idx, j=tj.idx, dx=float(dx / si), dy=float(dy / si),
                     dtheta=0.0, conf=float(np.clip(ncc, 0.0, 1.0)),
                     weight=float(8.0 + 20.0 * max(0.0, ncc)))


def _fit_circle(xs: np.ndarray, ys: np.ndarray):
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, r, rms) or None."""
    x = xs.astype(np.float64); y = ys.astype(np.float64)
    if len(x) < 12:
        return None
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    val = c + cx * cx + cy * cy
    if val <= 0:
        return None
    r = float(np.sqrt(val))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms = float(np.sqrt(np.mean((d - r) ** 2)))
    return float(cx), float(cy), r, rms


def _limb_points(mono: np.ndarray, cfg: SurfaceMosaicConfig):
    """
    Extract the limb-arc pixels of a partial disk: the disk/sky boundary that is
    adjacent to sky, with the frame-cut border excluded. Returns
    (xs, ys, (disk_cx, disk_cy)) — the disk-mask centroid tells the caller which
    side of the arc the disk centre physically lies on — or None.

    Thresholding is percentile-based, not Otsu: on a tile that is ~98% disk with
    a sliver of sky in one corner the histogram is unimodal, and Otsu splits
    WITHIN the disk instead of disk-vs-sky. The 0.2th percentile still samples
    the sky whenever it covers >=0.2% of the frame, so even a corner sliver
    yields a clean split.
    """
    if cv2 is None:
        return None
    u8 = (np.clip(mono, 0.0, 1.0) * 255.0).astype(np.uint8)
    lo = float(np.percentile(u8, 0.2))
    hi = float(np.percentile(u8, 99.5))
    if hi - lo < 20.0:
        return None                       # no dark/bright split -> all disk (or all sky)
    thr = lo + 0.35 * (hi - lo)
    disk = (u8 > thr).astype(np.uint8)
    frac = float(disk.mean())
    if frac < 0.005 or frac > 0.998:      # keep even a ~0.2% sky sliver
        return None
    k = np.ones((5, 5), np.uint8)
    disk = cv2.morphologyEx(disk, cv2.MORPH_OPEN, k)
    disk = cv2.morphologyEx(disk, cv2.MORPH_CLOSE, k)
    grad = cv2.morphologyEx(disk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    sky_dil = cv2.dilate((disk == 0).astype(np.uint8), k) > 0
    limb = grad & sky_dil
    b = 3
    limb[:b, :] = False; limb[-b:, :] = False; limb[:, :b] = False; limb[:, -b:] = False
    ys, xs = np.where(limb)
    if len(xs) < 40:
        return None
    dys, dxs = np.where(disk > 0)
    dc = (float(dxs.mean()), float(dys.mean()))
    return xs.astype(np.float64), ys.astype(np.float64), dc


def _fit_center_fixed_r(xs, ys, R, cx0, cy0):
    """
    Gauss-Newton fit of a circle CENTRE with the radius held fixed at R (seeded
    from cx0,cy0). Fixing the radius pins the centre even from a short, shallow
    arc that an unconstrained fit would leave unstable. Returns (cx, cy, rms) or
    None.
    """
    cx, cy = float(cx0), float(cy0)
    for _ in range(30):
        dx = xs - cx; dy = ys - cy
        ri = np.sqrt(dx * dx + dy * dy) + 1e-9
        res = ri - R
        Jx = -dx / ri; Jy = -dy / ri
        a11 = float(np.dot(Jx, Jx)); a12 = float(np.dot(Jx, Jy)); a22 = float(np.dot(Jy, Jy))
        b0 = -float(np.dot(Jx, res)); b1 = -float(np.dot(Jy, res))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-9:
            return None
        ddx = (b0 * a22 - b1 * a12) / det
        ddy = (a11 * b1 - a12 * b0) / det
        cx += ddx; cy += ddy
        if abs(ddx) + abs(ddy) < 1e-3:
            break
    ri = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rms = float(np.sqrt(np.mean((ri - R) ** 2)))
    return cx, cy, rms


def _tile_limb(tile: Tile, cfg: SurfaceMosaicConfig):
    """Cached limb-arc extraction for a tile:
    ("ok", xs, ys, freefit_or_None, (disk_cx, disk_cy)) or ("none",).
    freefit may be None for a sliver arc where the unconstrained Kasa fit is
    degenerate — the tile is still usable via the fixed-radius disk-side seed."""
    if tile._limb is None:
        pts = _limb_points(_to_mono01(tile.image), cfg)
        if pts is None:
            tile._limb = ("none",)
        else:
            xs, ys, dc = pts
            fit = _fit_circle(xs, ys)
            tile._limb = ("ok", xs, ys, fit, dc)
    return tile._limb


def _limb_solution(tiles: List[Tile], cfg: SurfaceMosaicConfig):
    """
    Fit the shared disk across all limb-bearing tiles: hub (best-conditioned free
    arc fit) sets the radius R; every limb tile's centre is refit at that fixed
    radius from a GEOMETRIC seed — a sliver arc is nearly straight, so an
    unconstrained fit can put the centre on the WRONG side, but the disk-mask
    centroid tells us which side the disk physically lies on, so we seed at
    arc_midpoint + R * (towards disk) and Gauss-Newton lands in the right basin.
    Returns (R, hub_pos, {pos: (cx, cy, rms)}) in TILE coordinates, or None.
    """
    info = {}   # pos -> [xs, ys, freefit, dc]
    for pos, t in enumerate(tiles):
        lb = _tile_limb(t, cfg)
        if lb[0] != "ok":
            continue
        info[pos] = [lb[1], lb[2], lb[3], lb[4]]
    if not info:
        return None

    def _quality(pos):
        ff = info[pos][2]
        if ff is None:
            return -1.0
        return len(info[pos][0]) / (1.0 + ff[3])
    hub = max(info, key=_quality)
    if info[hub][2] is None:
        return None
    R = float(info[hub][2][2])
    span = max(max(t.h, t.w) for t in tiles)
    if R < 0.2 * min(tiles[hub].h, tiles[hub].w) or R > 200.0 * span:
        return None

    centres = {}
    for pos, (xs, ys, freefit, dc) in info.items():
        pmx, pmy = float(xs.mean()), float(ys.mean())
        vx, vy = dc[0] - pmx, dc[1] - pmy
        nrm = math.hypot(vx, vy)
        if nrm < 1e-6:
            if freefit is None:
                continue
            seed = (freefit[0], freefit[1])
        else:
            seed = (pmx + R * vx / nrm, pmy + R * vy / nrm)
        ref = _fit_center_fixed_r(xs, ys, R, seed[0], seed[1])
        if ref is None:
            continue
        cxf, cyf, rmsf = ref
        if rmsf > max(3.0, 0.05 * R):
            continue    # points don't actually lie on a circle of radius R
        centres[pos] = (cxf, cyf, rmsf)
    if hub not in centres or not centres:
        return None
    return R, hub, centres


def _limb_edges(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                have: set) -> List[PairMatch]:
    """
    GLOBAL limb stage. Every tile that shows a limb arc lies on the SAME disk, so
    the disk centre is a shared reference that links limb tiles even when they
    don't overlap each other at all. Emits hub->tile edges from the shared-disk
    solution. Translation only — callers gate on solve_rotation.
    """
    sol = _limb_solution(tiles, cfg)
    if sol is None:
        return []
    R, hub, centres = sol
    if len(centres) < 2:
        return []

    tight = max(2.5, 0.03 * R)
    out: List[PairMatch] = []
    for pos, (cx, cy, rms) in centres.items():
        if pos == hub:
            continue
        key = (hub, pos) if hub < pos else (pos, hub)
        if key in have:
            continue    # already connected by a validated texture match
        hi, hj = tiles[hub], tiles[pos]
        cand = PairMatch(i=hi.idx, j=hj.idx,
                         dx=float(centres[hub][0] - cx),
                         dy=float(centres[hub][1] - cy),
                         dtheta=0.0, conf=0.45, weight=12.0)
        refined = _refine_pair(hi, hj, cand, cfg)
        ncc = _pair_overlap_ncc(hi, hj, refined, cfg)
        if ncc >= max(0.05, 0.5 * float(cfg.min_overlap_ncc)):
            # tiles overlap and the overlap agrees — full-strength edge
            refined.conf = float(np.clip(0.5 * refined.conf + 0.5 * max(0.0, ncc), 0.0, 1.0))
            refined.weight = float(refined.weight * (0.5 + max(0.0, ncc)))
            out.append(refined)
        elif ncc < -0.5 and rms <= tight and centres[hub][2] <= tight:
            # no mutual overlap to check (NCC sentinel) — accept on geometry
            # alone, but only when both fixed-R fits are tight
            out.append(refined)
    return out


def _validated(ti: Tile, tj: Tile, cand: PairMatch,
               cfg: SurfaceMosaicConfig) -> Optional[PairMatch]:
    """Refine a candidate and gate it on the ghost-proof overlap NCC."""
    refined = _refine_pair(ti, tj, cand, cfg)
    ncc = _pair_overlap_ncc(ti, tj, refined, cfg)
    if ncc < cfg.min_overlap_ncc:
        return None
    refined.conf = float(np.clip(0.5 * refined.conf + 0.5 * max(0.0, ncc), 0.0, 1.0))
    refined.weight = float(refined.weight * (0.5 + max(0.0, ncc)))
    return refined


def _precompute_tile_caches(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                            workers: int) -> None:
    """
    Populate every tile's lazy caches (sharpened match mono, ORB/SIFT features,
    correlation proxy) up front, ONE job per tile so each cache has a single
    writer. After this the pairwise matching is read-only on tiles and safe to
    run across threads. Also faster: features are computed once per tile instead
    of on first-touch during matching.
    """
    _get_msd_funcs()   # pre-warm the global lazy import so threads don't race it

    def job(t: Tile):
        _match_mono(t, cfg)
        if cv2 is not None:
            _ensure_features(t, cfg, "orb")
            if cfg.sift_fallback:
                _ensure_features(t, cfg, "sift")
            if cfg.phasecorr_fallback:
                _corr_small(t, cfg)

    if workers > 1 and len(tiles) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, tiles))
    else:
        for t in tiles:
            job(t)


def build_overlap_graph(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                        progress_cb: Optional[Callable] = None) -> List[PairMatch]:
    """
    Alignment ladder, strongest evidence first. Per pair:
      1a. ORB corners (fast; crisp texture)
      1b. SIFT blobs (soft, low-contrast texture where ORB finds nothing)
      2.  exhaustive correlation search (no features needed; translation only)
    then one GLOBAL stage:
      3.  limb geometry — links every limb-bearing tile through the shared disk
          centre, including tiles that don't overlap each other (rotation off).
    Every candidate from every tier must clear the overlap-NCC ghost gate (limb
    edges between non-overlapping tiles clear a geometric-tightness gate instead).

    Pairwise matching is O(n^2) but every pair is independent, so it runs across
    threads (cv2 calls release the GIL). Caches are pre-computed per tile first to
    avoid a write race, and results are assembled in combo order so the output is
    identical to the serial path.
    """
    combos = list(combinations(range(len(tiles)), 2))
    ncpu = os.cpu_count() or 1

    def match_job(item):
        idx, (i, j) = item
        got = None
        cand = _match_pair_features(tiles[i], tiles[j], cfg, kind="orb")
        if cand is not None:
            got = _validated(tiles[i], tiles[j], cand, cfg)
        if got is None and cfg.sift_fallback:
            cand = _match_pair_features(tiles[i], tiles[j], cfg, kind="sift")
            if cand is not None:
                got = _validated(tiles[i], tiles[j], cand, cfg)
        if got is None and cfg.phasecorr_fallback:
            cand = _match_pair_corrsearch(tiles[i], tiles[j], cfg)
            if cand is not None:
                got = _validated(tiles[i], tiles[j], cand, cfg)
        return idx, got

    results: List[Optional[PairMatch]] = [None] * len(combos)
    old_threads = None
    try:
        old_threads = cv2.getNumThreads(); cv2.setNumThreads(1)
    except Exception:
        old_threads = None
    try:
        _precompute_tile_caches(tiles, cfg, max(1, min(len(tiles), ncpu)))
        workers = max(1, min(len(combos), ncpu))
        if workers > 1 and len(combos) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                done = 0
                for idx, got in ex.map(match_job, list(enumerate(combos))):
                    results[idx] = got
                    done += 1
                    if progress_cb:
                        progress_cb(done, len(combos), "Matching tiles")
        else:
            for item in enumerate(combos):
                idx, got = match_job(item)
                results[idx] = got
                if progress_cb:
                    progress_cb(idx + 1, len(combos), "Matching tiles")
    finally:
        try:
            if old_threads is not None:
                cv2.setNumThreads(old_threads)
        except Exception:
            pass

    # assemble in deterministic combo order (identical to the serial path)
    pairs: List[PairMatch] = []
    have: set = set()
    for idx, (i, j) in enumerate(combos):
        got = results[idx]
        if got is not None:
            pairs.append(got)
            have.add((i, j))

    if cfg.limb_fallback and not cfg.solve_rotation:
        pairs.extend(_limb_edges(tiles, cfg, have))

    if progress_cb:
        progress_cb(len(combos), len(combos), "Matching tiles")
    return pairs


# ===========================================================================
# Stage 4 — Global bundle adjustment.  NEW work.
# ===========================================================================
def _connected_component(n: int, pairs: List[PairMatch], anchor: int) -> set:
    adj: Dict[int, set] = {k: set() for k in range(n)}
    for p in pairs:
        adj[p.i].add(p.j)
        adj[p.j].add(p.i)
    seen, stack = set(), [anchor]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    return seen


def _solve_translation(tiles: List[Tile], pairs: List[PairMatch], cfg: SurfaceMosaicConfig) -> None:
    """
    Weighted least-squares over per-tile translations. Each pair contributes
    (tx_j - tx_i) = dx_ij and (ty_j - ty_i) = dy_ij with weight w_ij.
    Anchor tile pinned at (0,0). x and y solve independently (linear).
    """
    n = len(tiles)
    anchor = int(np.clip(cfg.anchor_index, 0, n - 1))

    def _solve_axis(get: Callable[[PairMatch], float]) -> np.ndarray:
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        for p in pairs:
            w = float(max(1e-6, p.weight))
            i, j, d = p.i, p.j, get(p)
            A[i, i] += w; A[j, j] += w
            A[i, j] -= w; A[j, i] -= w
            b[i] -= w * d          # residual (tx_j - tx_i) = d  -> derivative wrt tx_i
            b[j] += w * d
        # gauge: pin the anchor at 0
        A[anchor, :] = 0.0
        A[anchor, anchor] = 1.0
        b[anchor] = 0.0
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    tx = _solve_axis(lambda p: p.dx)
    ty = _solve_axis(lambda p: p.dy)
    for k, t in enumerate(tiles):
        t.tx = float(tx[k])
        t.ty = float(ty[k])
        t.theta = 0.0


def _seed_similarity(tiles: List[Tile], pairs: List[PairMatch], cfg: SurfaceMosaicConfig) -> None:
    """
    Seed each tile's (theta, tx, ty) by propagating the pairwise (dtheta, d) out
    from the anchor along a spanning tree of the overlap graph. Composing the
    transforms as we go — theta_j = theta_i + dtheta, t_j = t_i + R(theta_i)d —
    gives the nonlinear solver a starting point already near the answer, which is
    what lets it place a rotated tile instead of getting stuck at all-zeros.
    """
    from collections import deque
    n = len(tiles)
    anchor = int(np.clip(cfg.anchor_index, 0, n - 1))

    adj: Dict[int, list] = {k: [] for k in range(n)}
    for p in pairs:
        adj[p.i].append((p.j, +1, p))   # forward: j relative to i
        adj[p.j].append((p.i, -1, p))   # reverse: i relative to j

    th = [None] * n; tx = [None] * n; ty = [None] * n
    th[anchor], tx[anchor], ty[anchor] = 0.0, 0.0, 0.0
    dq = deque([anchor])
    while dq:
        u = dq.popleft()
        for (v, sign, p) in adj[u]:
            if th[v] is not None:
                continue
            dth = math.radians(p.dtheta)
            dx, dy = float(p.dx), float(p.dy)
            if sign > 0:                         # u=i, v=j
                thv = th[u] + dth
                cu, su = math.cos(th[u]), math.sin(th[u])
                tx[v] = tx[u] + (cu * dx - su * dy)
                ty[v] = ty[u] + (su * dx + cu * dy)
            else:                                # u=j, v=i  (invert the relation)
                thv = th[u] - dth
                cv_, sv_ = math.cos(thv), math.sin(thv)
                tx[v] = tx[u] - (cv_ * dx - sv_ * dy)
                ty[v] = ty[u] - (sv_ * dx + cv_ * dy)
            th[v] = thv
            dq.append(v)

    for k in range(n):
        if th[k] is None:                        # tile not connected to anchor
            th[k], tx[k], ty[k] = 0.0, 0.0, 0.0
        tiles[k].theta = math.degrees(th[k])
        tiles[k].tx = float(tx[k])
        tiles[k].ty = float(ty[k])


def _solve_similarity_scipy(tiles: List[Tile], pairs: List[PairMatch], cfg: SurfaceMosaicConfig) -> None:
    """
    Joint (tx, ty, theta) solve. Composes the poses as real 2-D similarity
    transforms — the pairwise translation is rotated into each tile's own frame
    via R(theta_i) — seeds from a spanning-tree propagation, and uses a robust
    loss so a single bad pair can't drag the whole solution. Falls back to the
    translation-only solve if scipy is unavailable.
    """
    if _scipy_least_squares is None:
        # No scipy: the spanning-tree seed already places rotated tiles (it just
        # can't do the robust loop-closure refine). Still far better than a
        # translation-only solve, which ignores rotation entirely.
        _seed_similarity(tiles, pairs, cfg)
        return

    _seed_similarity(tiles, pairs, cfg)

    n = len(tiles)
    anchor = int(np.clip(cfg.anchor_index, 0, n - 1))
    free = [k for k in range(n) if k != anchor]
    index = {k: p for p, k in enumerate(free)}
    # scale rotation residuals into pixel-equivalent units (a 1-rad error is ~R px
    # of edge misalignment) so angles and translations are commensurate in the solve
    rad_to_px = 0.5 * float(np.mean([math.hypot(t.w, t.h) for t in tiles])) or 1.0

    def pack() -> np.ndarray:
        v = np.zeros(3 * len(free))
        for k in free:
            b = 3 * index[k]
            v[b:b + 3] = (tiles[k].tx, tiles[k].ty, math.radians(tiles[k].theta))
        return v

    def pose(k: int, v: np.ndarray) -> Tuple[float, float, float]:
        if k == anchor:
            return 0.0, 0.0, 0.0
        b = 3 * index[k]
        return float(v[b]), float(v[b + 1]), float(v[b + 2])

    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def residuals(v: np.ndarray) -> np.ndarray:
        res = []
        for p in pairs:
            txi, tyi, thi = pose(p.i, v)
            txj, tyj, thj = pose(p.j, v)
            w = math.sqrt(max(1e-6, p.weight))
            ci, si = math.cos(thi), math.sin(thi)
            # pairwise offset expressed in tile i's rotated frame: t_j - t_i = R(thi) d
            drx = ci * p.dx - si * p.dy
            dry = si * p.dx + ci * p.dy
            res.append(w * ((txj - txi) - drx))
            res.append(w * ((tyj - tyi) - dry))
            res.append(w * rad_to_px * _wrap(thj - thi - math.radians(p.dtheta)))
        return np.asarray(res, dtype=np.float64)

    try:
        sol = _scipy_least_squares(residuals, pack(), method="trf",
                                   loss="soft_l1", max_nfev=400)
        xopt = sol.x
    except Exception:
        xopt = pack()   # keep the seed if the optimiser fails

    for k in free:
        txk, tyk, thk = pose(k, xopt)
        tiles[k].tx, tiles[k].ty, tiles[k].theta = txk, tyk, math.degrees(thk)
    tiles[anchor].tx = tiles[anchor].ty = tiles[anchor].theta = 0.0


def global_bundle_adjust(tiles: List[Tile], pairs: List[PairMatch],
                         cfg: SurfaceMosaicConfig) -> List[int]:
    """
    Solve tile poses and return the indices of any tiles that could NOT be placed
    (not connected to the anchor through a validated overlap). These are excluded
    from the mosaic entirely — a tile that didn't match is not composited at all
    (no side-by-side filler), and the caller fails outright if fewer than two
    tiles actually merged.
    """
    n = len(tiles)
    anchor = int(np.clip(cfg.anchor_index, 0, n - 1))

    if pairs:
        if cfg.solve_rotation:
            _solve_similarity_scipy(tiles, pairs, cfg)
        else:
            _solve_translation(tiles, pairs, cfg)
        comp = _connected_component(n, pairs, anchor)
    else:
        comp = {anchor}

    return [k for k in range(n) if k not in comp]


# ===========================================================================
# Stage 7 — Photometric equalisation.  NEW work.
# gain+bias overlap solve, plus a radial limb-darkening model for disk targets
# ===========================================================================
def _equalize_overlap(tiles: List[Tile], pairs: List[PairMatch],
                      cfg: SurfaceMosaicConfig) -> None:
    """
    Robust GAIN-ONLY equalisation from the BRIGHT overlap content. For every
    overlapping pair, take the aligned overlap pixels where BOTH tiles are bright
    (sky excluded entirely, so glint/background can't drag the statistic) and
    constrain the gain ratio by the median bright level: g_i*med_a = g_j*med_b.
    Solved jointly in log-gains with a prior toward 1. No bias term anywhere:
    gains cannot lift the sky (g * 0 = 0), so the background stays black by
    construction.
    """
    if not pairs:
        return
    n = len(tiles)
    slog2 = float(cfg.photo_sigma_g) ** 2
    mono = [(_to_mono01(t.image).astype(np.float32, copy=False)) for t in tiles]
    thr = [0.3 * float(np.percentile(m, 95.0)) for m in mono]   # bright = surface

    A = np.zeros((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    for p in pairs:
        rdx, rdy = int(round(p.dx)), int(round(p.dy))
        box = _integer_overlap(tiles[p.i].w, tiles[p.i].h,
                               tiles[p.j].w, tiles[p.j].h, rdx, rdy)
        if box is None:
            continue
        xi0, yi0, xj0, yj0, ow, oh = box
        a = mono[p.i][yi0:yi0 + oh, xi0:xi0 + ow]
        b = mono[p.j][yj0:yj0 + oh, xj0:xj0 + ow]
        bright = (a > thr[p.i]) & (b > thr[p.j])
        nb = int(bright.sum())
        if nb < 256:
            continue
        med_a = float(np.median(a[bright]))
        med_b = float(np.median(b[bright]))
        if med_a <= 1e-6 or med_b <= 1e-6:
            continue
        d = math.log(med_b) - math.log(med_a)    # x_i - x_j = d  (x = log gain)
        w = float(min(nb, 20000))
        A[p.i, p.i] += w; A[p.j, p.j] += w
        A[p.i, p.j] -= w; A[p.j, p.i] -= w
        c[p.i] += w * d;  c[p.j] -= w * d

    lam = 1.0 / max(1e-6, slog2)
    for i in range(n):
        A[i, i] += lam                            # prior: log-gain -> 0
    try:
        x = np.linalg.solve(A, c)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, c, rcond=None)[0]
    x = x - float(np.median(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else x
    for k, t in enumerate(tiles):
        xv = x[k] if np.isfinite(x[k]) else 0.0
        t.gain = float(np.clip(math.exp(xv), 0.2, 5.0))
        t.bias = 0.0


def _disk_in_mosaic(tiles: List[Tile], cfg: SurfaceMosaicConfig, exclude: set):
    """Shared disk in MOSAIC coordinates: (cx, cy, R) from the limb solution with
    each tile's centre composed through its pose, or None if no disk was found."""
    sol = _limb_solution(tiles, cfg)
    if sol is None:
        return None
    R, _hub, centres = sol
    wsum = 0.0; cxm = 0.0; cym = 0.0
    for pos, (cx, cy, rms) in centres.items():
        if pos in exclude:
            continue
        t = tiles[pos]
        th = math.radians(t.theta)
        co, si = math.cos(th), math.sin(th)
        mx = co * cx - si * cy + t.tx
        my = si * cx + co * cy + t.ty
        w = 1.0 / (1.0 + rms)
        cxm += w * mx; cym += w * my; wsum += w
    if wsum <= 0:
        return None
    return cxm / wsum, cym / wsum, R


def _equalize_radial(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                     exclude: set) -> bool:
    """
    Disk-aware equalisation: solar/lunar brightness is dominated by LIMB
    DARKENING, a radial profile shared by every tile. The limb solution gives the
    disk centre + radius, so build a global median brightness-vs-r/R profile from
    all placed tiles and solve each tile's gain against that shared MODEL instead
    of against its neighbours. This decouples equalisation from overlap topology
    entirely — rows with zero overlap are still constrained because they answer
    to the same profile. Returns True if applied.
    """
    disk = _disk_in_mosaic(tiles, cfg, exclude)
    if disk is None:
        return False
    cxm, cym, R = disk

    use = [pos for pos in range(len(tiles)) if pos not in exclude]
    if len(use) < 2:
        return False

    NBINS = 96
    SMALL = 384

    # per-tile binned medians of brightness vs r/R (computed once, on a downscale)
    tile_bins = {}   # pos -> (bin_idx array, median array, count array)
    for pos in use:
        t = tiles[pos]
        m = _to_mono01(t.image).astype(np.float32, copy=False)
        sc = min(1.0, SMALL / float(max(t.h, t.w)))
        sm = cv2.resize(m, (max(4, int(t.w * sc)), max(4, int(t.h * sc))),
                        interpolation=cv2.INTER_AREA) if sc < 1.0 else m
        hh, ww = sm.shape[:2]
        ys, xs = np.mgrid[0:hh, 0:ww]
        fx = xs.astype(np.float64) / sc
        fy = ys.astype(np.float64) / sc
        th = math.radians(t.theta)
        co, si = math.cos(th), math.sin(th)
        mx = co * fx - si * fy + t.tx
        my = si * fx + co * fy + t.ty
        rn = np.sqrt((mx - cxm) ** 2 + (my - cym) ** 2) / R
        vals = sm.astype(np.float64)
        good = (rn <= 1.0) & (vals > 0.02)     # inside the disk, not sky
        if int(good.sum()) < 500:
            continue
        bidx = np.clip((rn[good] * NBINS).astype(np.int32), 0, NBINS - 1)
        v = vals[good]
        bins_i, meds, cnts = [], [], []
        for bi in np.unique(bidx):
            sel = v[bidx == bi]
            if sel.size < 50:
                continue
            bins_i.append(int(bi)); meds.append(float(np.median(sel))); cnts.append(int(sel.size))
        if len(bins_i) < 1:
            continue
        tile_bins[pos] = (np.asarray(bins_i), np.asarray(meds), np.asarray(cnts, dtype=np.float64))
    if len(tile_bins) < 2:
        return False

    def _wmedian(vals: np.ndarray, wts: np.ndarray) -> float:
        o = np.argsort(vals)
        v = vals[o]; w = np.cumsum(wts[o])
        idx = int(np.searchsorted(w, 0.5 * w[-1]))
        return float(v[min(idx, len(v) - 1)])

    gains = {pos: 1.0 for pos in tile_bins}
    for _it in range(2):
        # global profile: count-weighted mean of gain-corrected tile medians per bin
        acc = np.zeros(NBINS); wac = np.zeros(NBINS)
        for pos, (bi, md, ct) in tile_bins.items():
            np.add.at(acc, bi, ct * gains[pos] * md)
            np.add.at(wac, bi, ct)
        prof = np.where(wac > 0, acc / np.maximum(wac, 1e-9), 0.0)
        # per-tile GAIN-ONLY robust fit: weighted median of per-bin ratios
        # profile/tile. A median of ratios shrugs off glint-corrupted bins where
        # a least-squares would chase them; no bias term, so the sky (which never
        # enters the bins anyway) cannot be lifted.
        for pos, (bi, md, ct) in tile_bins.items():
            P = prof[bi]
            ok = (wac[bi] > 0) & (md > 0.05) & (P > 1e-6)
            if int(ok.sum()) < 1:
                continue
            rho = P[ok] / md[ok]
            gains[pos] = float(np.clip(_wmedian(rho, ct[ok]), 0.2, 5.0))

    garr = np.array(list(gains.values()))
    gmed = float(np.median(garr)) if garr.size else 1.0
    for pos, t in enumerate(tiles):
        if pos in gains:
            t.gain = gains[pos] / gmed
        else:
            t.gain = 1.0
        t.bias = 0.0
    return True


def equalize_photometry(tiles: List[Tile], pairs: List[PairMatch],
                        cfg: SurfaceMosaicConfig, exclude: Optional[set] = None) -> None:
    """
    Solve one robust GAIN per tile (no bias — the sky can never be lifted).
    When a disk was found and cfg.radial_model is on, equalise against the global
    radial limb-darkening profile (couples tiles regardless of overlap topology);
    otherwise match the bright overlap content. Writes tile.gain in place.
    """
    if not cfg.photometric:
        return
    exclude = exclude or set()
    if cfg.radial_model and not cfg.solve_rotation:
        try:
            if _equalize_radial(tiles, cfg, exclude):
                return
        except Exception:
            pass
    _equalize_overlap(tiles, pairs, cfg)


def _grad_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quadratic basis columns [1, u, v, u^2, u*v, v^2] for coords in [-1,1]."""
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)


def equalize_gradients(tiles: List[Tile], pairs: List[PairMatch],
                       cfg: SurfaceMosaicConfig, exclude: Optional[set] = None) -> None:
    """
    Per-tile smooth gain SURFACE so overlaps match across the gradients each panel
    carries (etalon sweet-spot, vignetting). Each tile gets a log-domain quadratic
    poly(u,v) over normalised tile coords; a quadratic (not a plane) because an
    interior tile must agree with neighbours on all four sides, which a single
    linear tilt can't do — and it reduces to a plane on linear data (the u^2/v^2
    terms just solve to ~0).

    At every shared overlap pixel we want s_i*a == s_j*b with s = exp(poly), which
    in logs is LINEAR:  poly_i(u_i,v_i) - poly_j(u_j,v_j) = log(b/a). Solved jointly
    for all tiles (6 coeffs each) via 6x6 block normal equations, with a ridge
    prior toward flat. Multiplicative, so the sky can't be lifted; and samples are
    limb-masked (r < 0.98R) + bright-in-both so no sky/limb pixel ever enters the
    fit. Writes tile.grad. Translation overlaps only (gated on rotation off).
    """
    if not cfg.gradient_fit or cfg.solve_rotation or not pairs:
        return
    exclude = set(exclude or set())
    n = len(tiles)
    disk = _disk_in_mosaic(tiles, cfg, exclude)   # (cx, cy, R) or None
    mono = [(_to_mono01(t.image).astype(np.float32, copy=False)) for t in tiles]
    thr = [0.3 * float(np.percentile(mono[k], 95.0)) for k in range(n)]

    def norm_uv(x, y, t):
        return (x - 0.5 * t.w) / (0.5 * t.w), (y - 0.5 * t.h) / (0.5 * t.h)

    NB = 6
    M = np.zeros((NB * n, NB * n), dtype=np.float64)
    rhs = np.zeros(NB * n, dtype=np.float64)
    used = set()

    for p in pairs:
        if p.i in exclude or p.j in exclude:
            continue
        rdx, rdy = int(round(p.dx)), int(round(p.dy))
        box = _integer_overlap(tiles[p.i].w, tiles[p.i].h,
                               tiles[p.j].w, tiles[p.j].h, rdx, rdy)
        if box is None:
            continue
        xi0, yi0, xj0, yj0, ow, oh = box
        # subsample the overlap on a grid (~step so <= ~4k samples)
        step = max(1, int(math.sqrt(ow * oh / 4000.0)))
        ys = np.arange(0, oh, step)
        xs = np.arange(0, ow, step)
        gx, gy = np.meshgrid(xs, ys)
        gx = gx.ravel(); gy = gy.ravel()
        ai = mono[p.i][yi0 + gy, xi0 + gx]
        bj = mono[p.j][yj0 + gy, xj0 + gx]
        gi = float(tiles[p.i].gain); gj = float(tiles[p.j].gain)
        a = ai * gi; b = bj * gj
        keep = (ai > thr[p.i]) & (bj > thr[p.j]) & (a > 1e-4) & (b > 1e-4)
        if disk is not None:
            cxm, cym, R = disk
            # sample position in mosaic coords (tile i pixel -> canvas)
            ti = tiles[p.i]
            th = math.radians(ti.theta); co, si = math.cos(th), math.sin(th)
            px = xi0 + gx; py = yi0 + gy
            mx = co * px - si * py + ti.tx
            my = si * px + co * py + ti.ty
            rr = np.sqrt((mx - cxm) ** 2 + (my - cym) ** 2)
            keep &= (rr < 0.98 * R)
        if int(keep.sum()) < 24:
            continue

        ui, vi = norm_uv(xi0 + gx[keep], yi0 + gy[keep], tiles[p.i])
        uj, vj = norm_uv(xj0 + gx[keep], yj0 + gy[keep], tiles[p.j])
        Bi = _grad_basis(ui, vi)                      # (Ns, 6)
        Bj = _grad_basis(uj, vj)
        t = np.log(b[keep]) - np.log(a[keep])         # target = log(b/a)

        bi0, bj0 = NB * p.i, NB * p.j
        BiTBi = Bi.T @ Bi; BjTBj = Bj.T @ Bj; BiTBj = Bi.T @ Bj
        M[bi0:bi0 + NB, bi0:bi0 + NB] += BiTBi
        M[bj0:bj0 + NB, bj0:bj0 + NB] += BjTBj
        M[bi0:bi0 + NB, bj0:bj0 + NB] -= BiTBj
        M[bj0:bj0 + NB, bi0:bi0 + NB] -= BiTBj.T
        rhs[bi0:bi0 + NB] += Bi.T @ t
        rhs[bj0:bj0 + NB] -= Bj.T @ t
        used.add(p.i); used.add(p.j)

    if not used:
        return

    # ridge prior toward flat (0 = correction 1); gentle on the constant, firmer on
    # curvature so surfaces stay smooth and the gauge (global constant) is pinned
    ridge = np.array([0.5, 1.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float64)
    for k in range(n):
        for c in range(NB):
            M[NB * k + c, NB * k + c] += ridge[c]

    try:
        x = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(M, rhs, rcond=None)[0]

    for k, t in enumerate(tiles):
        if k in used and np.all(np.isfinite(x[NB * k:NB * k + NB])):
            t.grad = tuple(float(v) for v in x[NB * k:NB * k + NB])
        else:
            t.grad = None


# ===========================================================================
# Stages 5/6/8 — Compose (warp into canvas, local warp, seam + blend)
# ===========================================================================
def _grad_surface(t: Tile) -> Optional[np.ndarray]:
    """Evaluate a tile's log-domain quadratic gain surface -> (H,W) multiplier,
    clamped to a sane range so a runaway fit can't blow out a tile."""
    if t.grad is None:
        return None
    c0, c1, c2, c3, c4, c5 = t.grad
    ys, xs = np.mgrid[0:t.h, 0:t.w]
    u = (xs.astype(np.float32) - 0.5 * t.w) / (0.5 * t.w)
    v = (ys.astype(np.float32) - 0.5 * t.h) / (0.5 * t.h)
    poly = c0 + c1 * u + c2 * v + c3 * u * u + c4 * u * v + c5 * v * v
    return np.exp(np.clip(poly, -0.8, 0.8)).astype(np.float32)


def _tile_affine(t: Tile, ox: float, oy: float) -> np.ndarray:
    """
    2x3 affine mapping tile-local coords -> canvas coords (origin ox,oy).
    Pose is R(theta) about the tile's own (0,0), then translate by (tx,ty) — the
    same origin-rotation composition the global solve uses, so solved poses place
    correctly. For theta==0 this reduces to a plain translation.
    """
    th = math.radians(t.theta)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, t.tx - ox],
                     [s,  c, t.ty - oy]], dtype=np.float32)


def _canvas_bounds(tiles: List[Tile]) -> Tuple[float, float, int, int]:
    xs, ys = [], []
    for t in tiles:
        th = math.radians(t.theta)
        c, s = math.cos(th), math.sin(th)
        for (px, py) in ((0, 0), (t.w, 0), (0, t.h), (t.w, t.h)):
            xs.append(c * px - s * py + t.tx)   # R(theta)*p + (tx,ty)
            ys.append(s * px + c * py + t.ty)
    x0, y0 = math.floor(min(xs)), math.floor(min(ys))
    x1, y1 = math.ceil(max(xs)), math.ceil(max(ys))
    return float(x0), float(y0), int(x1 - x0), int(y1 - y0)


def _feather_weights(mask: np.ndarray) -> np.ndarray:
    """Distance-transform feather: 0 at the tile edge, growing inward."""
    if cv2 is None:
        return mask.astype(np.float32)
    m = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    mx = float(dt.max())
    if mx > 0:
        dt = dt / mx
    return dt.astype(np.float32)


def _ap_local_structure(cur_m: np.ndarray, centers: np.ndarray, ap_size: int) -> np.ndarray:
    """Per-AP high-frequency content on the sharpened proxy — the local std of a
    high-passed copy inside each AP window. An AP on flat mottling scores near
    zero, so its (noisy) phase shift can be dropped before it poisons the field."""
    hp = cur_m - cv2.GaussianBlur(cur_m, (0, 0), max(1.0, ap_size / 8.0))
    H, W = cur_m.shape[:2]
    half = int(ap_size // 2)
    out = np.zeros(len(centers), dtype=np.float64)
    for i, (cx, cy) in enumerate(centers):
        x0 = max(0, int(cx) - half); x1 = min(W, int(cx) + half)
        y0 = max(0, int(cy) - half); y1 = min(H, int(cy) + half)
        if x1 > x0 and y1 > y0:
            out[i] = float(hp[y0:y1, x0:x1].std())
    return out


def _confined_field_from_aps(H: int, W: int, centers: np.ndarray,
                             adx: np.ndarray, ady: np.ndarray, resp: np.ndarray,
                             band: float, grid: int = 24):
    """
    Build a displacement field that is LOCAL: each point's shift is a
    response-weighted average of only the APs within ~band of it, and a point
    with no nearby AP support gets exactly ZERO (never an extrapolation from a
    distant edge's APs). This confines every seam's correction to a band around
    that seam, so a big pull on one edge cannot reach — and disturb — the other
    edges of an interior tile, and an edge with no trustworthy APs simply keeps
    the global placement instead of being warped by its neighbours' shifts.
    """
    gy = np.arange(0, H, grid, dtype=np.float64)
    gx = np.arange(0, W, grid, dtype=np.float64)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    P = np.stack([GX.ravel(), GY.ravel()], axis=1)          # (nn, 2)
    C = centers.astype(np.float64)                          # (na, 2)
    d2 = ((P[:, None, 0] - C[None, :, 0]) ** 2 +
          (P[:, None, 1] - C[None, :, 1]) ** 2)             # (nn, na)
    w = resp[None, :] * np.exp(-d2 / (2.0 * band * band))   # (nn, na)
    wsum = w.sum(axis=1)
    fx = (w * adx[None, :]).sum(axis=1)
    fy = (w * ady[None, :]).sum(axis=1)
    supp = wsum > 0.20 * float(resp.max() if resp.size else 1.0)   # need real nearby support
    fxn = np.where(supp, fx / np.maximum(wsum, 1e-9), 0.0)
    fyn = np.where(supp, fy / np.maximum(wsum, 1e-9), 0.0)
    ny, nx = GY.shape
    dxf = cv2.resize(fxn.reshape(ny, nx).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    dyf = cv2.resize(fyn.reshape(ny, nx).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    s = max(1.0, band * 0.5)
    dxf = cv2.GaussianBlur(dxf, (0, 0), s)
    dyf = cv2.GaussianBlur(dyf, (0, 0), s)
    return dxf, dyf


def _measure_local_field(ref_canvas_mono: np.ndarray, cur_mono: np.ndarray,
                         valid: np.ndarray, cfg: SurfaceMosaicConfig):
    """
    Measure the non-rigid seam-correction dense field for a tile against the
    already-composited neighbours. Runs on the SHARPENED match images (ref and
    cur are both sharpened monos) so the AP phase correlation can lock onto
    structure even on soft data; the caller applies the returned field to the
    ORIGINAL pixels. Returns (dxf, dyf), or (None, None) if no correction is
    warranted.

    APs are gated the same way the aligner gates its correlation search: an AP is
    only trusted if its window holds real high-frequency structure AND its phase
    correlation returns a strong response. A low-signal seam (thin strip, flat
    mottling) therefore keeps the global placement instead of getting a confident-
    but-wrong local warp — the doubling/tripling failure mode.
    """
    if not cfg.local_warp or cv2 is None:
        return None, None
    cur_m = np.asarray(cur_mono, dtype=np.float32)
    # only correct where THIS tile and the existing mosaic both have data
    both = (valid > 0) & (ref_canvas_mono > 1e-6)
    if both.mean() < 0.02:
        return None, None

    ap_centers = _autoplace_aps(np.where(both, cur_m, 0.0),
                                cfg.ap_size, cfg.ap_spacing, cfg.ap_min_mean)
    if ap_centers is None or len(ap_centers) < 4:
        return None, None
    ap_centers = np.asarray(ap_centers)

    # Gate 1 — structure: drop APs on flat patches (no signal to localise on)
    struct = _ap_local_structure(cur_m, ap_centers, cfg.ap_size)
    gmed = float(np.median(struct[struct > 0])) if np.any(struct > 0) else 0.0
    if gmed > 0:
        ap_centers = ap_centers[struct > float(cfg.ap_min_struct) * gmed]
    if len(ap_centers) < 4:
        return None, None

    ap_dx, ap_dy, ap_resp = _ap_phase_shifts_per_ap(
        ref_canvas_mono, cur_m, ap_centers=ap_centers,
        ap_size=cfg.ap_size, max_dim=cfg.ap_size,
    )
    resp = np.clip(ap_resp, 0.0, 1.0)
    # Gate 2 — response: keep only APs whose correlation actually locked on
    keep = _reject_ap_outliers(ap_dx, ap_dy, resp, z=3.5) & (resp >= float(cfg.ap_min_response))
    if int(np.count_nonzero(keep)) < 4:
        return None, None

    # Guard — already well placed: on flat overlap the per-AP shifts are just a
    # broad correlation peak wobbling sub-pixel. If the tile is essentially where
    # it belongs, skip the field entirely instead of baking in that noise.
    mag = np.hypot(ap_dx[keep], ap_dy[keep])
    if float(np.median(mag)) < float(cfg.local_warp_min_shift):
        return None, None

    if cfg.seam_confine:
        dxf, dyf = _confined_field_from_aps(
            cur_m.shape[0], cur_m.shape[1],
            ap_centers[keep], ap_dx[keep], ap_dy[keep], resp[keep],
            band=float(cfg.ap_size) * float(cfg.seam_band_mult),
            grid=max(8, int(cfg.ap_field_grid)),
        )
    else:
        dxf, dyf = _dense_field_from_ap_shifts(
            cur_m.shape[0], cur_m.shape[1],
            ap_centers[keep], ap_dx[keep], ap_dy[keep], resp[keep],
            grid=cfg.ap_field_grid, power=2.0, conf_floor=0.15,
            radius=float(cfg.ap_size) * 3.0,
        )

    # Guard — confine to the overlap: feather the field to zero at the overlap
    # boundary so a tile's non-overlap pixels are never displaced by the dense
    # field extrapolating outward.
    falloff = max(1.0, float(cfg.ap_size))
    dt = cv2.distanceTransform(both.astype(np.uint8), cv2.DIST_L2, 3)
    feather = np.clip(dt / falloff, 0.0, 1.0).astype(np.float32)
    return dxf * feather, dyf * feather


def _fill_nodata(img: np.ndarray, valid: np.ndarray, iters: int = 5, sigma: float = 8.0) -> np.ndarray:
    """
    Replace a tile's no-data pixels with a smooth extension of its own content,
    so the multiband Laplacian pyramid never sees a hard 0->content cliff at the
    rectangular tile boundary (which otherwise bleeds a dark low-frequency ring
    along the seam). Grows valid content a little way into the border and leaves
    far no-data alone (it's masked out of the blend anyway).
    """
    if cv2 is None:
        return img
    m = (valid > 0)
    m3 = m[:, :, None] if img.ndim == 3 else m
    filled = img.copy()
    for _ in range(int(max(1, iters))):
        blur = cv2.GaussianBlur(filled, (0, 0), float(sigma))
        filled = np.where(m3, img, blur)
    return filled


def _tile_bbox(t: Tile, ox: float, oy: float, cw: int, ch: int, pad: int = 2):
    """Canvas-space bounding box (bx, by, bw, bh) of a tile's warped footprint,
    clipped to the canvas. Warping into the bbox instead of the full canvas keeps
    per-tile buffers small (memory-safe for big mosaics) and the warp cheap.

    A few pixels of PAD are added on every side: INTER_LINEAR samples against the
    zero border and darkens the outermost row/column, so without the pad that
    darkened edge lands exactly on the bbox boundary and prints a thin seam line.
    The pad pushes the interpolation seam outside the valid mask, where the blend
    masks it away — same as the old full-canvas warp, which had neighbours
    covering those edge pixels."""
    thr = math.radians(t.theta)
    c, s = math.cos(thr), math.sin(thr)
    xs = []; ys = []
    for px, py in ((0, 0), (t.w, 0), (0, t.h), (t.w, t.h)):
        xs.append(c * px - s * py + t.tx - ox)
        ys.append(s * px + c * py + t.ty - oy)
    x0 = max(0, int(math.floor(min(xs))) - pad); y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(cw, int(math.ceil(max(xs))) + pad); y1 = min(ch, int(math.ceil(max(ys))) + pad)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _prewarp_tile(t: Tile, cfg: SurfaceMosaicConfig, ox: float, oy: float,
                  cw: int, ch: int, want_color: bool):
    """
    Independent per-tile work (no shared state): apply gain + gradient surface,
    then warp the original, the valid mask, and the sharpened match copy into the
    tile's canvas bounding box. This is the embarrassingly-parallel bulk of
    compose; the sequential seam/blend phase consumes these buffers. Returns
    (bx, by, warped, valid, match_warped) or None for an empty footprint.
    """
    bx, by, bw, bh = _tile_bbox(t, ox, oy, cw, ch)
    if bw < 2 or bh < 2:
        return None

    img = t.image
    if img.ndim == 2 and want_color:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = np.maximum(img.astype(np.float32, copy=False) * float(t.gain)
                     + float(t.bias), 0.0)
    surf = _grad_surface(t) if t.grad is not None else None
    if surf is not None:
        img = img * (surf[:, :, None] if img.ndim == 3 else surf)

    M = _tile_affine(t, ox, oy)
    M = M.copy(); M[0, 2] -= bx; M[1, 2] -= by      # warp into bbox-local coords
    warped = cv2.warpAffine(img, M, (bw, bh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=cfg.border_value)
    ones = np.ones((t.h, t.w), dtype=np.float32)
    valid = cv2.warpAffine(ones, M, (bw, bh), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    mm = _match_mono(t, cfg) * float(t.gain)
    if surf is not None:
        mm = mm * surf
    match_warped = cv2.warpAffine(mm, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    if warped.ndim == 2:
        warped = warped[:, :, None]
    return bx, by, warped, valid, match_warped


def compose_mosaic(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                   progress_cb: Optional[Callable] = None) -> MosaicResult:
    ox, oy, cw, ch = _canvas_bounds(tiles)
    is_color = any(t.is_color for t in tiles)
    chans = 3 if is_color else 1

    # MultiBandBlender lives in the (optional) opencv stitching-detail module.
    use_mb = (cfg.blend == "multiband" and hasattr(cv2, "detail_MultiBandBlender"))
    if cfg.blend == "multiband" and not use_mb:
        print("[SurfaceMosaic] cv2.detail_MultiBandBlender not available in this "
              "OpenCV build — falling back to feather blend.")
    MB_SCALE = 8000.0   # float[0..1] -> int16 range for the blender (keeps headroom)

    coverage = np.zeros((ch, cw), dtype=np.float32)
    # local-warp reference is the SHARPENED match canvas — so seam correction
    # measures against structure the AP phase-corr can lock onto, then the same
    # field is applied to the original pixels below.
    running_match = np.zeros((ch, cw), dtype=np.float32)

    if use_mb:
        # multiband needs global ownership, so collect tiles and blend after the loop
        mb_store: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        accum = wsum = None
    else:
        mb_store = None
        accum = np.zeros((ch, cw, chans), dtype=np.float32)
        wsum = np.zeros((ch, cw, 1), dtype=np.float32)

    # composite brightest/sharpest-first so the local warp always has a good ref
    order = sorted(range(len(tiles)), key=lambda k: -tiles[k].quality)

    # ---- Phase A: pre-warp every tile in parallel (independent per tile) ----
    # cv2 ops release the GIL, so threads give real speedup. Keep OpenCV's own
    # per-op threads modest here so the tile-level pool owns the cores (avoids
    # oversubscription); the sequential phase below gets the full cv2 thread pool.
    want_color = bool(is_color or use_mb)
    ncpu = os.cpu_count() or 1
    workers = max(1, min(len(order), ncpu))
    old_threads = None
    try:
        old_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
    except Exception:
        old_threads = None

    prewarped: Dict[int, Any] = {}
    try:
        if workers > 1 and len(order) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_prewarp_tile, tiles[k], cfg, ox, oy, cw, ch, want_color): k
                        for k in order}
                done = 0
                for fut, k in futs.items():
                    prewarped[k] = fut.result()
                    done += 1
                    if progress_cb:
                        progress_cb(done, len(order), "Warping tiles")
        else:
            for i, k in enumerate(order):
                prewarped[k] = _prewarp_tile(tiles[k], cfg, ox, oy, cw, ch, want_color)
                if progress_cb:
                    progress_cb(i + 1, len(order), "Warping tiles")
    finally:
        try:
            if old_threads is not None:
                cv2.setNumThreads(max(1, ncpu))   # sequential phase: full cv2 threading
        except Exception:
            pass

    # ---- Phase B: sequential seam measure + blend (reads running_match) ----
    placed_any = False
    try:
        for step, k in enumerate(order):
            if progress_cb:
                progress_cb(step, len(order), "Compositing")
            pw = prewarped.get(k)
            prewarped[k] = None
            if pw is None:
                continue
            bx, by, warped, valid, match_warped = pw
            bh, bw = valid.shape[:2]
            ref_sub = running_match[by:by + bh, bx:bx + bw]

            # local (non-rigid) seam correction: MEASURE on the sharpened match,
            # APPLY the identical field to the original pixels.
            if placed_any:
                dxf, dyf = _measure_local_field(ref_sub, match_warped, valid, cfg)
                if dxf is not None:
                    warped = _warp_by_dense_field(warped, dxf, dyf)
                    if warped.ndim == 2:
                        warped = warped[:, :, None]
                    match_warped = _warp_by_dense_field(match_warped, dxf, dyf)

            if cfg.blend == "none":
                wt = (valid > 0).astype(np.float32)
            else:
                wt = _feather_weights(valid)
                if cfg.quality_weighted_seam:
                    wt = wt * (0.25 + 0.75 * float(np.clip(tiles[k].quality, 0.0, 1.0)))

            if use_mb:
                wimg = warped if warped.shape[2] == 3 else np.repeat(warped, 3, axis=2)
                mb_store.append((bx, by, wimg, (valid > 0), wt.astype(np.float32, copy=False)))
            else:
                wtc = wt[:, :, None]
                accum[by:by + bh, bx:bx + bw] += warped * wtc
                wsum[by:by + bh, bx:bx + bw] += wtc

            coverage[by:by + bh, bx:bx + bw] += (valid > 0).astype(np.float32)
            running_match[by:by + bh, bx:bx + bw] = np.where(
                valid > 0,
                np.maximum(ref_sub, match_warped),
                ref_sub)
            placed_any = True
    finally:
        try:
            if old_threads is not None:
                cv2.setNumThreads(old_threads)
        except Exception:
            pass

    if use_mb:
        # Hard ownership: each pixel goes to the single valid tile with the
        # highest feather x quality weight (a winner-take-all seam). The blender
        # then does the actual cross-band feathering, so no overlapping soft
        # masks confuse its weight normalisation.
        best = np.full((ch, cw), -1.0, dtype=np.float32)
        owner = np.full((ch, cw), -1, dtype=np.int32)
        for idx, (bx, by, _wi, vmask, wt) in enumerate(mb_store):
            bh, bw = vmask.shape
            sub_best = best[by:by + bh, bx:bx + bw]
            better = (wt > sub_best) & vmask
            owner[by:by + bh, bx:bx + bw] = np.where(
                better, idx, owner[by:by + bh, bx:bx + bw])
            best[by:by + bh, bx:bx + bw] = np.where(better, wt, sub_best)

        blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=int(cfg.multiband_bands))
        blender.prepare((0, 0, cw, ch))
        for idx, (bx, by, wi, vmask, _wt) in enumerate(mb_store):
            bh, bw = vmask.shape
            # fill no-data so the Laplacian pyramid doesn't bleed the black cliff
            filled = _fill_nodata(wi, vmask)
            img16 = np.clip(filled * MB_SCALE, -32768.0, 32767.0).astype(np.int16)
            mask8 = np.where(owner[by:by + bh, bx:bx + bw] == idx,
                             np.uint8(255), np.uint8(0))
            blender.feed(np.ascontiguousarray(img16),
                         np.ascontiguousarray(mask8), (bx, by))
        res16, _res_mask = blender.blend(None, None)
        mosaic = np.clip(res16.astype(np.float32) / MB_SCALE, 0.0, 1.0)
        if chans == 1:
            mosaic = mosaic[:, :, 0]   # channels were replicated equal -> collapse
    else:
        mosaic = np.clip(accum / np.maximum(wsum, 1e-6), 0.0, 1.0)
        if chans == 1:
            mosaic = mosaic[:, :, 0]

    if progress_cb:
        progress_cb(len(order), len(order), "Compositing")

    return MosaicResult(image=mosaic, coverage=coverage, tiles=tiles,
                        pairs=[], origin=(ox, oy))


# ===========================================================================
# Orchestrator
# ===========================================================================
def run_surface_mosaic(tiles: List[Tile], cfg: Optional[SurfaceMosaicConfig] = None,
                       progress_cb: Optional[Callable] = None) -> MosaicResult:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for surface mosaics.")
    if len(tiles) < 2:
        raise ValueError("Need at least two tiles to build a mosaic.")
    cfg = cfg or SurfaceMosaicConfig()

    def _phase(a, b, label):
        if progress_cb:
            progress_cb(a, b, label)

    _phase(0, 1, "Detecting overlaps")
    pairs = build_overlap_graph(tiles, cfg, progress_cb=progress_cb)

    _phase(0, 1, "Solving global placement")
    unplaced = global_bundle_adjust(tiles, pairs, cfg)
    unplaced_set = set(unplaced)
    placed = [t for i, t in enumerate(tiles) if i not in unplaced_set]

    if len(placed) < 2:
        # nothing merged — a non-solution is not an output
        raise RuntimeError(
            "Could not find a reliable overlap between the selected tiles, so no "
            "mosaic was produced. This usually means too little overlap or too "
            "little structure (poor seeing). Try more overlap or sharper panels.")

    _phase(0, 1, "Equalising brightness")
    # pairs index into the full tile list; unplaced tiles appear in no pair, so
    # running on `tiles` is correct — but they're excluded from the radial model
    # since they have no valid pose
    equalize_photometry(tiles, pairs, cfg, exclude=unplaced_set)
    equalize_gradients(tiles, pairs, cfg, exclude=unplaced_set)

    # TODO(sphere_reproject): if cfg.sphere_reproject, reproject each tile onto a
    # common lunar/solar sphere here using _build_lonlat_grids / derotate_stack_lonshift
    # (from setiastro.saspro.derotate) before compositing — needed for full-disk
    # and limb-spanning mosaics. v1 stitches in the image plane.

    result = compose_mosaic(placed, cfg, progress_cb=progress_cb)
    result.pairs = pairs
    anchor_tile = tiles[int(np.clip(cfg.anchor_index, 0, len(tiles) - 1))]
    result.anchor = anchor_tile.idx if anchor_tile in placed else placed[0].idx
    result.unplaced = [tiles[k].name for k in unplaced]
    return result


# ===========================================================================
# Qt worker + minimal dialog
# ===========================================================================
try:
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPointF, QRectF
    from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QFont
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
        QPushButton, QLabel, QProgressBar, QCheckBox, QComboBox, QMessageBox,
        QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QGroupBox,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover
    _HAVE_QT = False


if _HAVE_QT:

    class TileLayoutView(QWidget):
        """
        Schematic overview of where each tile lands in the mosaic, drawn from the
        solved poses — not a full-res preview. Translucent rectangles (so overlaps
        darken where they stack), match-graph edges between overlapping tiles with
        thickness scaled by confidence, per-tile centre dots + names, the anchor
        tile marked, and a dashed outline for the whole mosaic canvas.
        """
        _PALETTE = [
            (55, 138, 221), (239, 159, 39), (29, 158, 117), (216, 90, 48),
            (127, 119, 221), (212, 83, 126), (99, 153, 34), (136, 135, 128),
        ]

        def __init__(self, parent=None):
            super().__init__(parent)
            self._result = None
            self.setMinimumHeight(240)

        def clear(self):
            self._result = None
            self.update()

        def set_result(self, result):
            self._result = result
            self.update()

        @staticmethod
        def _corners_mosaic(t: "Tile"):
            th = math.radians(t.theta)
            c, s = math.cos(th), math.sin(th)
            pts = []
            for px, py in ((0, 0), (t.w, 0), (t.w, t.h), (0, t.h)):
                pts.append((c * px - s * py + t.tx, s * px + c * py + t.ty))
            return pts

        def paintEvent(self, _ev):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.fillRect(self.rect(), self.palette().base())

            txt = self.palette().text().color()
            faint = QColor(txt); faint.setAlpha(150)

            res = self._result
            tiles = list(getattr(res, "tiles", []) or [])
            if res is None or not tiles:
                p.setPen(faint)
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           "Run a mosaic to see the tile layout")
                p.end()
                return

            W, H = self.width(), self.height()
            m = 26.0

            corners = [self._corners_mosaic(t) for t in tiles]
            allc = [pt for cc in corners for pt in cc]
            xs = [c[0] for c in allc]; ys = [c[1] for c in allc]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            bw = max(1.0, maxx - minx); bh = max(1.0, maxy - miny)
            scale = min((W - 2 * m) / bw, (H - 2 * m) / bh)
            offx = m + (W - 2 * m - bw * scale) * 0.5
            offy = m + (H - 2 * m - bh * scale) * 0.5

            def tw(pt):
                return QPointF(offx + (pt[0] - minx) * scale,
                               offy + (pt[1] - miny) * scale)

            # mosaic canvas bounds (dashed)
            p.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(faint, 1.0); pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(QRectF(tw((minx, miny)), tw((maxx, maxy))))

            anchor = int(getattr(res, "anchor", 0))
            pos_by_idx = {t.idx: pos for pos, t in enumerate(tiles)}

            # translucent tile rectangles (overlaps darken via alpha stacking)
            centres = []
            for i, cc in enumerate(corners):
                r, g, b = self._PALETTE[i % len(self._PALETTE)]
                poly = QPolygonF([tw(pt) for pt in cc])
                fill = QColor(r, g, b, 70)
                p.setBrush(QBrush(fill))
                is_anchor = (tiles[i].idx == anchor)
                p.setPen(QPen(QColor(r, g, b), 2.5 if is_anchor else 1.4))
                p.drawPolygon(poly)
                cx = sum(pt.x() for pt in poly) / 4.0
                cy = sum(pt.y() for pt in poly) / 4.0
                centres.append((cx, cy))

            # match-graph edges between overlapping tiles (thickness by confidence)
            for pr in (getattr(res, "pairs", None) or []):
                pi = pos_by_idx.get(pr.i); pj = pos_by_idx.get(pr.j)
                if pi is not None and pj is not None:
                    a, b = centres[pi], centres[pj]
                    lw = 1.0 + 3.0 * float(np.clip(pr.conf, 0.0, 1.0))
                    ec = QColor(txt); ec.setAlpha(190)
                    p.setPen(QPen(ec, lw))
                    p.drawLine(QPointF(*a), QPointF(*b))

            # centre dots + names on top
            p.setFont(QFont(self.font().family(), 9))
            for i, ctr in enumerate(centres):
                r, g, b = self._PALETTE[i % len(self._PALETTE)]
                p.setBrush(QBrush(QColor(r, g, b)))
                p.setPen(QPen(QColor(255, 255, 255, 210), 1))
                p.drawEllipse(QPointF(*ctr), 3.5, 3.5)
                name = (tiles[i].name or f"tile {i}")
                if len(name) > 18:
                    name = name[:17] + "…"
                if tiles[i].idx == anchor:
                    name += "  ★"
                p.setPen(txt)
                p.drawText(QPointF(ctr[0] + 7, ctr[1] - 6), name)

            # header
            p.setPen(faint)
            p.drawText(8, 16, f"Tile layout — {len(tiles)} tiles, "
                              f"{len(getattr(res, 'pairs', None) or [])} overlaps")
            p.end()


    class SurfaceMosaicWorker(QThread):
        progress = pyqtSignal(int, int, str)
        finished_ok = pyqtSignal(object)   # MosaicResult
        failed = pyqtSignal(str)

        def __init__(self, tiles: List[Tile], cfg: SurfaceMosaicConfig):
            # QThread workers use parent=None (SASpro convention)
            super().__init__(None)
            self._tiles = tiles
            self._cfg = cfg

        def run(self):
            try:
                res = run_surface_mosaic(
                    self._tiles, self._cfg,
                    progress_cb=lambda a, b, s: self.progress.emit(int(a), int(b), str(s)),
                )
                self.finished_ok.emit(res)
            except Exception as e:  # surface the error to the GUI thread
                self.failed.emit(str(e))


    class SurfaceMosaicDialog(QWidget):
        """
        Minimal first-draft UI: pick open views as tiles, choose a couple of
        options, run off-thread, push the result to a new view.
        Caller should set Qt.WidgetAttribute.WA_DeleteOnClose at the call site.
        """
        def __init__(self, doc_manager=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Surface Mosaic (Lunar / Solar / Planetary)")
            self.doc_manager = doc_manager
            self._worker: Optional[SurfaceMosaicWorker] = None
            self._views: List[Tuple[str, Any]] = []

            # advanced params (edited via the Advanced dialog), seeded from defaults
            _d = SurfaceMosaicConfig()
            self._adv = {
                "enhance_matching": _d.enhance_matching,
                "match_msd_layers": _d.match_msd_layers,
                "match_msd_gains": tuple(_d.match_msd_gains),
                "orb_features": _d.orb_features,
                "feature_max_dim": _d.feature_max_dim,
                "lowe_ratio": _d.lowe_ratio,
                "min_inliers": _d.min_inliers,
                "ransac_thresh": _d.ransac_thresh,
                "min_overlap_px": _d.min_overlap_px,
                "min_overlap_ncc": _d.min_overlap_ncc,
                "sift_fallback": _d.sift_fallback,
                "phasecorr_fallback": _d.phasecorr_fallback,
                "limb_fallback": _d.limb_fallback,
                "local_warp_min_shift": _d.local_warp_min_shift,
                "multiband_bands": _d.multiband_bands,
                "radial_model": _d.radial_model,
                "gradient_fit": _d.gradient_fit,
                "ap_size": _d.ap_size,
                "quality_weighted_seam": _d.quality_weighted_seam,
            }

            outer = QVBoxLayout(self)
            cols = QHBoxLayout()

            # ---- left column: view selection + options + run ----
            left = QVBoxLayout()
            left.addWidget(QLabel("Select the open views to mosaic (finished per-panel stacks):"))

            self.view_list = QListWidget()
            self.view_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
            left.addWidget(self.view_list, 1)

            btn_row = QHBoxLayout()
            self.btn_refresh = QPushButton("Refresh views")
            self.btn_refresh.clicked.connect(self.refresh_views)
            btn_row.addWidget(self.btn_refresh)
            self.btn_selectall = QPushButton("Select all")
            self.btn_selectall.clicked.connect(self._toggle_select_all)
            btn_row.addWidget(self.btn_selectall)
            btn_row.addStretch(1)
            left.addLayout(btn_row)

            self.view_list.itemChanged.connect(self._update_selectall_label)

            opt_row = QHBoxLayout()
            self.chk_rotation = QCheckBox("Solve rotation")
            self.chk_rotation.setChecked(False)
            self.chk_photo = QCheckBox("Equalise brightness")
            self.chk_photo.setChecked(True)
            self.chk_localwarp = QCheckBox("Local seam warp")
            self.chk_localwarp.setChecked(True)
            opt_row.addWidget(self.chk_rotation)
            opt_row.addWidget(self.chk_photo)
            opt_row.addWidget(self.chk_localwarp)
            opt_row.addStretch(1)
            opt_row.addWidget(QLabel("Blend Type:"))
            self.cmb_blend = QComboBox()
            # display label -> internal cfg.blend key
            self.cmb_blend.addItem("Feather", "feather")
            self.cmb_blend.addItem("MultiBandBlender", "multiband")
            opt_row.addWidget(self.cmb_blend)
            left.addLayout(opt_row)

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            left.addWidget(self.progress)

            self.status = QLabel("")
            left.addWidget(self.status)

            run_row = QHBoxLayout()
            self.btn_advanced = QPushButton("Advanced…")
            self.btn_advanced.clicked.connect(self._open_advanced)
            run_row.addWidget(self.btn_advanced)
            run_row.addStretch(1)
            self.btn_run = QPushButton("Build Mosaic")
            self.btn_run.clicked.connect(self._on_run)
            run_row.addWidget(self.btn_run)
            left.addLayout(run_row)

            # ---- right column: tile layout overview ----
            right = QVBoxLayout()
            right.addWidget(QLabel("Tile layout:"))
            self.layout_view = TileLayoutView()
            self.layout_view.setMinimumWidth(320)
            right.addWidget(self.layout_view, 1)

            cols.addLayout(left, 3)
            cols.addLayout(right, 4)
            outer.addLayout(cols, 1)

            # SetiAstro footer (spans both columns)
            footer = QLabel("Franklin Marek — www.setiastro.com")
            footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            footer.setStyleSheet("color: gray; font-size: 10px;")
            outer.addWidget(footer)

            self.resize(940, 620)
            self.refresh_views()

        # ---- views ----
        def refresh_views(self):
            self._views = list_open_views()
            self.view_list.clear()
            for title, _sub in self._views:
                it = QListWidgetItem(title)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Unchecked)
                self.view_list.addItem(it)
            self.status.setText(f"{len(self._views)} open view(s).")
            self._update_selectall_label()

        def _checked_count(self) -> int:
            return sum(1 for r in range(self.view_list.count())
                       if self.view_list.item(r).checkState() == Qt.CheckState.Checked)

        def _toggle_select_all(self):
            n = self.view_list.count()
            if n == 0:
                return
            new_state = (Qt.CheckState.Unchecked if self._checked_count() == n
                         else Qt.CheckState.Checked)
            self.view_list.blockSignals(True)
            for r in range(n):
                self.view_list.item(r).setCheckState(new_state)
            self.view_list.blockSignals(False)
            self._update_selectall_label()

        def _update_selectall_label(self, *_):
            n = self.view_list.count()
            all_checked = (n > 0 and self._checked_count() == n)
            self.btn_selectall.setText("Select none" if all_checked else "Select all")
            self.btn_selectall.setEnabled(n > 0)

        def _selected_views(self) -> List[Tuple[str, Any]]:
            sel = []
            for row in range(self.view_list.count()):
                it = self.view_list.item(row)
                if it.checkState() == Qt.CheckState.Checked:
                    sel.append(self._views[row])
            return sel

        def _cfg_from_ui(self) -> SurfaceMosaicConfig:
            cfg = SurfaceMosaicConfig()
            cfg.solve_rotation = self.chk_rotation.isChecked()
            cfg.photometric = self.chk_photo.isChecked()
            cfg.local_warp = self.chk_localwarp.isChecked()
            cfg.blend = self.cmb_blend.currentData() or "feather"
            for k, v in self._adv.items():        # advanced overrides
                setattr(cfg, k, v)
            return cfg

        # ---- advanced parameters ----
        def _open_advanced(self):
            dlg = QDialog(self)
            dlg.setWindowTitle("Advanced — Surface Mosaic")
            v = QVBoxLayout(dlg)

            # matching enhancement (the soft-data fix)
            _MAXL = 8
            gb_match = QGroupBox("Matching enhancement (soft / unsharpened stacks)")
            fm = QFormLayout(gb_match)
            chk_enh = QCheckBox("Enable structure boost for matching")
            chk_enh.setChecked(bool(self._adv["enhance_matching"]))
            fm.addRow(chk_enh)
            sp_layers = QSpinBox(); sp_layers.setRange(2, _MAXL)
            sp_layers.setValue(int(self._adv["match_msd_layers"]))
            fm.addRow("Decomposition layers:", sp_layers)

            gains0 = list(self._adv["match_msd_gains"])
            gain_spins = []
            for i in range(_MAXL):
                sp = QDoubleSpinBox(); sp.setRange(0.0, 30.0); sp.setSingleStep(0.5)
                sp.setValue(float(gains0[i]) if i < len(gains0) else 1.0)
                fm.addRow(f"Layer {i + 1} gain ({2 ** i}px):", sp)
                gain_spins.append(sp)

            def _update_layer_rows():
                n = sp_layers.value()
                for i, sp in enumerate(gain_spins):
                    vis = (i < n)
                    sp.setVisible(vis)
                    lab = fm.labelForField(sp)
                    if lab is not None:
                        lab.setVisible(vis)

            sp_layers.valueChanged.connect(lambda _=0: _update_layer_rows())
            _update_layer_rows()
            v.addWidget(gb_match)

            # matching robustness
            gb_reg = QGroupBox("Registration")
            fr = QFormLayout(gb_reg)
            sp_orb = QSpinBox(); sp_orb.setRange(500, 20000); sp_orb.setSingleStep(500)
            sp_orb.setValue(int(self._adv["orb_features"]))
            fr.addRow("ORB features:", sp_orb)
            sp_fmax = QSpinBox(); sp_fmax.setRange(400, 8000); sp_fmax.setSingleStep(100)
            sp_fmax.setValue(int(self._adv["feature_max_dim"]))
            fr.addRow("Feature detect max dim (px):", sp_fmax)
            sp_lowe = QDoubleSpinBox(); sp_lowe.setRange(0.5, 0.95); sp_lowe.setSingleStep(0.05)
            sp_lowe.setValue(float(self._adv["lowe_ratio"]))
            fr.addRow("Lowe ratio (higher = looser):", sp_lowe)
            sp_inl = QSpinBox(); sp_inl.setRange(4, 200)
            sp_inl.setValue(int(self._adv["min_inliers"]))
            fr.addRow("Min inliers:", sp_inl)
            sp_ran = QDoubleSpinBox(); sp_ran.setRange(0.5, 10.0); sp_ran.setSingleStep(0.5)
            sp_ran.setValue(float(self._adv["ransac_thresh"]))
            fr.addRow("RANSAC threshold (px):", sp_ran)
            sp_ovl = QSpinBox(); sp_ovl.setRange(8, 512); sp_ovl.setSingleStep(8)
            sp_ovl.setValue(int(self._adv["min_overlap_px"]))
            fr.addRow("Min overlap for refine (px):", sp_ovl)
            sp_ncc = QDoubleSpinBox(); sp_ncc.setRange(0.0, 0.9); sp_ncc.setSingleStep(0.05)
            sp_ncc.setValue(float(self._adv["min_overlap_ncc"]))
            fr.addRow("Min overlap NCC (reject ghosts):", sp_ncc)
            chk_sift = QCheckBox("SIFT feature fallback (soft texture)")
            chk_sift.setChecked(bool(self._adv["sift_fallback"]))
            fr.addRow(chk_sift)
            chk_pc = QCheckBox("Correlation-search fallback (no features)")
            chk_pc.setChecked(bool(self._adv["phasecorr_fallback"]))
            fr.addRow(chk_pc)
            chk_limb = QCheckBox("Limb fallback (align disk edge; rotation off only)")
            chk_limb.setChecked(bool(self._adv["limb_fallback"]))
            fr.addRow(chk_limb)
            v.addWidget(gb_reg)

            # seam / local warp
            gb_seam = QGroupBox("Seam / local warp")
            fs = QFormLayout(gb_seam)
            sp_lwms = QDoubleSpinBox(); sp_lwms.setRange(0.0, 5.0); sp_lwms.setSingleStep(0.1)
            sp_lwms.setValue(float(self._adv["local_warp_min_shift"]))
            fs.addRow("Local-warp min shift (px):", sp_lwms)
            v.addWidget(gb_seam)

            # brightness / blending
            gb_bl = QGroupBox("Brightness / Blending")
            fb = QFormLayout(gb_bl)
            sp_bands = QSpinBox(); sp_bands.setRange(1, 10)
            sp_bands.setValue(int(self._adv["multiband_bands"]))
            fb.addRow("Multiband bands:", sp_bands)
            chk_radial = QCheckBox("Radial disk brightness model (auto when limb found)")
            chk_radial.setChecked(bool(self._adv["radial_model"]))
            fb.addRow(chk_radial)
            chk_grad = QCheckBox("Per-tile gradient surface fit (match panel gradients)")
            chk_grad.setChecked(bool(self._adv["gradient_fit"]))
            fb.addRow(chk_grad)
            v.addWidget(gb_bl)

            # danger zone
            gb_danger = QGroupBox("Don't touch unless you know what you're doing")
            fd = QFormLayout(gb_danger)
            sp_ap = QSpinBox(); sp_ap.setRange(16, 256); sp_ap.setSingleStep(8)
            sp_ap.setValue(int(self._adv["ap_size"]))
            fd.addRow("Seam-warp patch size (px):", sp_ap)
            chk_qseam = QCheckBox("Quality-weighted seam (sharper tile wins)")
            chk_qseam.setChecked(bool(self._adv["quality_weighted_seam"]))
            fd.addRow(chk_qseam)
            v.addWidget(gb_danger)

            bb = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.RestoreDefaults
                | QDialogButtonBox.StandardButton.Cancel)
            v.addWidget(bb)

            def _restore():
                d = SurfaceMosaicConfig()
                chk_enh.setChecked(d.enhance_matching)
                sp_layers.setValue(d.match_msd_layers)
                dg = list(d.match_msd_gains)
                for i, sp in enumerate(gain_spins):
                    sp.setValue(float(dg[i]) if i < len(dg) else 1.0)
                _update_layer_rows()
                sp_orb.setValue(d.orb_features)
                sp_fmax.setValue(d.feature_max_dim)
                sp_lowe.setValue(d.lowe_ratio)
                sp_inl.setValue(d.min_inliers)
                sp_ran.setValue(d.ransac_thresh)
                sp_ovl.setValue(d.min_overlap_px)
                sp_ncc.setValue(d.min_overlap_ncc)
                chk_sift.setChecked(d.sift_fallback)
                chk_pc.setChecked(d.phasecorr_fallback)
                chk_limb.setChecked(d.limb_fallback)
                sp_lwms.setValue(d.local_warp_min_shift)
                sp_bands.setValue(d.multiband_bands)
                chk_radial.setChecked(d.radial_model)
                chk_grad.setChecked(d.gradient_fit)
                sp_ap.setValue(d.ap_size)
                chk_qseam.setChecked(d.quality_weighted_seam)

            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            bb.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(_restore)

            if dlg.exec():
                n = sp_layers.value()
                self._adv.update({
                    "enhance_matching": chk_enh.isChecked(),
                    "match_msd_layers": n,
                    "match_msd_gains": tuple(float(gain_spins[i].value()) for i in range(n)),
                    "orb_features": sp_orb.value(),
                    "feature_max_dim": sp_fmax.value(),
                    "lowe_ratio": sp_lowe.value(),
                    "min_inliers": sp_inl.value(),
                    "ransac_thresh": sp_ran.value(),
                    "min_overlap_px": sp_ovl.value(),
                    "min_overlap_ncc": sp_ncc.value(),
                    "sift_fallback": chk_sift.isChecked(),
                    "phasecorr_fallback": chk_pc.isChecked(),
                    "limb_fallback": chk_limb.isChecked(),
                    "local_warp_min_shift": sp_lwms.value(),
                    "multiband_bands": sp_bands.value(),
                    "radial_model": chk_radial.isChecked(),
                    "gradient_fit": chk_grad.isChecked(),
                    "ap_size": sp_ap.value(),
                    "quality_weighted_seam": chk_qseam.isChecked(),
                })

        # ---- run ----
        def _on_run(self):
            sel = self._selected_views()
            if len(sel) < 2:
                QMessageBox.warning(self, "Pick tiles", "Select at least two views.")
                return
            tiles = tiles_from_selection(sel)
            if len(tiles) < 2:
                QMessageBox.warning(self, "No images", "Selected views had no images.")
                return

            self.btn_run.setEnabled(False)
            self.status.setText("Working…")
            self.layout_view.clear()
            self._worker = SurfaceMosaicWorker(tiles, self._cfg_from_ui())
            self._worker.progress.connect(self._on_progress)
            self._worker.finished_ok.connect(self._on_done)
            self._worker.failed.connect(self._on_fail)
            self._worker.start()

        def _on_progress(self, a: int, b: int, label: str):
            pct = int(100 * a / max(1, b))
            self.progress.setValue(min(100, pct))
            self.status.setText(f"{label}… ({a}/{b})")

        def _on_done(self, result):
            self.btn_run.setEnabled(True)
            self.layout_view.set_result(result)
            unplaced = list(getattr(result, "unplaced", []) or [])
            try:
                push_mosaic_to_view(result.image, title="Surface Mosaic",
                                    doc_manager=self.doc_manager)
                if unplaced:
                    self.status.setText(f"Mosaic opened — {len(unplaced)} tile(s) excluded (no match).")
                    names = "\n".join(f"  • {n}" for n in unplaced)
                    QMessageBox.warning(
                        self, "Some tiles could not be matched",
                        "These tiles didn't share a reliable overlap with the rest and were "
                        "left out of the mosaic:\n\n" + names +
                        "\n\nThis usually means too little overlap or too little structure "
                        "(poor seeing). Try more overlap, sharper panels, or the fallback / "
                        "matching-boost options in Advanced.")
                else:
                    self.status.setText("Mosaic opened in a new view.")
            except Exception as e:
                QMessageBox.critical(self, "Open failed", str(e))
                self.status.setText("Mosaic built, but couldn't open a view.")
            self._worker = None

        def _on_fail(self, msg: str):
            self.btn_run.setEnabled(True)
            self.status.setText("Failed.")
            QMessageBox.critical(self, "Mosaic failed", msg)
            self._worker = None