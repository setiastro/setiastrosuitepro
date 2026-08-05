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
import threading
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

    # --- photometric equalisation (Brown-Lowe gain compensation) ---
    photometric: bool = True
    photo_sigma_n: float = 10.0      # intensity error std (in 0..255-ish units)
    photo_sigma_g: float = 0.10      # gain prior std around 1.0

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
    # cached feature detections (filled lazily)
    _kps: Optional[list] = field(default=None, repr=False)
    _des: Optional[np.ndarray] = field(default=None, repr=False)
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


def _ensure_features(tile: Tile, cfg: SurfaceMosaicConfig) -> None:
    if tile._des is not None or cv2 is None:
        return
    u8, scale = _feature_mono_u8(_match_mono(tile, cfg), cfg.feature_max_dim)
    orb = cv2.ORB_create(nfeatures=int(cfg.orb_features))
    kps, des = orb.detectAndCompute(u8, None)
    # scale keypoints back to full-res tile coordinates
    if kps and scale != 1.0:
        inv = 1.0 / scale
        for kp in kps:
            kp.pt = (kp.pt[0] * inv, kp.pt[1] * inv)
    tile._kps = list(kps) if kps else []
    tile._des = des


def _match_pair_features(ti: Tile, tj: Tile, cfg: SurfaceMosaicConfig) -> Optional[PairMatch]:
    """
    Coarse pairwise similarity (rotation+scale+translation) from ORB matches.
    Returns j-relative-to-i translation/rotation, or None if the pair doesn't overlap.
    """
    if cv2 is None:
        return None
    _ensure_features(ti, cfg)
    _ensure_features(tj, cfg)
    if ti._des is None or tj._des is None or len(ti._kps) < 4 or len(tj._kps) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(tj._des, ti._des, k=2)   # query=j, train=i  -> maps j into i
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

    src = np.float32([tj._kps[m.queryIdx].pt for m in good])  # in j
    dst = np.float32([ti._kps[m.trainIdx].pt for m in good])  # in i

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


def build_overlap_graph(tiles: List[Tile], cfg: SurfaceMosaicConfig,
                        progress_cb: Optional[Callable] = None) -> List[PairMatch]:
    pairs: List[PairMatch] = []
    combos = list(combinations(range(len(tiles)), 2))
    for k, (i, j) in enumerate(combos):
        if progress_cb:
            progress_cb(k, len(combos), "Matching tiles")
        coarse = _match_pair_features(tiles[i], tiles[j], cfg)
        if coarse is None:
            continue
        pairs.append(_refine_pair(tiles[i], tiles[j], coarse, cfg))
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


def global_bundle_adjust(tiles: List[Tile], pairs: List[PairMatch], cfg: SurfaceMosaicConfig) -> None:
    if not pairs:
        # nothing overlaps — leave tiles at origin (compose will just stack them)
        return
    comp = _connected_component(len(tiles), pairs, int(np.clip(cfg.anchor_index, 0, len(tiles) - 1)))
    if len(comp) < len(tiles):
        # Tiles outside the anchor's connected component can't be placed reliably.
        # First draft: solve the whole graph anyway; unmatched tiles land at origin.
        pass
    if cfg.solve_rotation:
        _solve_similarity_scipy(tiles, pairs, cfg)
    else:
        _solve_translation(tiles, pairs, cfg)


# ===========================================================================
# Stage 7 — Photometric equalisation (Brown-Lowe gain compensation).  NEW work.
# ===========================================================================
def equalize_photometry(tiles: List[Tile], pairs: List[PairMatch], cfg: SurfaceMosaicConfig) -> None:
    """
    Solve one multiplicative gain per tile so overlap brightness matches.
    Minimises sum_ij N_ij (g_i*Ii - g_j*Ij)^2 / sN^2 + sum_i (1-g_i)^2 / sg^2.
    Writes tile.gain in place. (Bias term left as a TODO extension.)
    """
    if not cfg.photometric or not pairs:
        return
    n = len(tiles)
    sN2 = float(cfg.photo_sigma_n) ** 2
    sg2 = float(cfg.photo_sigma_g) ** 2

    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    mono = [(_to_mono01(t.image).astype(np.float32, copy=False)) for t in tiles]

    for p in pairs:
        rdx, rdy = int(round(p.dx)), int(round(p.dy))
        box = _integer_overlap(tiles[p.i].w, tiles[p.i].h,
                               tiles[p.j].w, tiles[p.j].h, rdx, rdy)
        if box is None:
            continue
        xi0, yi0, xj0, yj0, ow, oh = box
        Ii = float(mono[p.i][yi0:yi0 + oh, xi0:xi0 + ow].mean()) * 255.0
        Ij = float(mono[p.j][yj0:yj0 + oh, xj0:xj0 + ow].mean()) * 255.0
        N = float(ow * oh)
        a = N / sN2
        A[p.i, p.i] += a * Ii * Ii
        A[p.j, p.j] += a * Ij * Ij
        A[p.i, p.j] -= a * Ii * Ij
        A[p.j, p.i] -= a * Ii * Ij

    for i in range(n):
        A[i, i] += 1.0 / sg2
        b[i] += 1.0 / sg2

    try:
        g = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        g = np.linalg.lstsq(A, b, rcond=None)[0]

    g = np.clip(g, 0.2, 5.0)
    # normalise so the brightest-weighted tile isn't crushed/blown
    g = g / float(np.median(g[np.isfinite(g)])) if np.any(np.isfinite(g)) else g
    for k, t in enumerate(tiles):
        t.gain = float(g[k]) if np.isfinite(g[k]) else 1.0


# ===========================================================================
# Stages 5/6/8 — Compose (warp into canvas, local warp, seam + blend)
# ===========================================================================
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


def _measure_local_field(ref_canvas_mono: np.ndarray, cur_mono: np.ndarray,
                         valid: np.ndarray, cfg: SurfaceMosaicConfig):
    """
    Measure the non-rigid seam-correction dense field for a tile against the
    already-composited neighbours. Runs on the SHARPENED match images (ref and
    cur are both sharpened monos) so the AP phase correlation can lock onto
    structure even on soft data; the caller applies the returned field to the
    ORIGINAL pixels. Returns (dxf, dyf), or (None, None) if no correction is
    warranted.
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

    ap_dx, ap_dy, ap_resp = _ap_phase_shifts_per_ap(
        ref_canvas_mono, cur_m, ap_centers=ap_centers,
        ap_size=cfg.ap_size, max_dim=cfg.ap_size,
    )
    keep = _reject_ap_outliers(ap_dx, ap_dy, np.clip(ap_resp, 0.0, 1.0), z=3.5)
    if not np.any(keep):
        return None, None

    # Guard 1 — already well placed: on flat overlap the per-AP shifts are just
    # a broad correlation peak wobbling sub-pixel. If the tile is essentially
    # where it belongs, skip the field entirely instead of baking in that noise.
    mag = np.hypot(ap_dx[keep], ap_dy[keep])
    if float(np.median(mag)) < float(cfg.local_warp_min_shift):
        return None, None

    dxf, dyf = _dense_field_from_ap_shifts(
        cur_m.shape[0], cur_m.shape[1],
        ap_centers[keep], ap_dx[keep], ap_dy[keep], np.clip(ap_resp[keep], 0.0, 1.0),
        grid=cfg.ap_field_grid, power=2.0, conf_floor=0.15,
        radius=float(cfg.ap_size) * 3.0,
    )

    # Guard 2 — confine to the overlap: feather the field to zero at the overlap
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

    for step, k in enumerate(order):
        t = tiles[k]
        if progress_cb:
            progress_cb(step, len(order), "Compositing")

        img = t.image
        if img.ndim == 2 and (is_color or use_mb):
            img = np.repeat(img[:, :, None], 3, axis=2)
        img = (img.astype(np.float32, copy=False) * float(t.gain))

        M = _tile_affine(t, ox, oy)
        warped = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=cfg.border_value)
        ones = np.ones((t.h, t.w), dtype=np.float32)
        valid = cv2.warpAffine(ones, M, (cw, ch), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        # sharpened match copy warped by the SAME pose — used only to measure the
        # local seam field, never composited into the output
        match_warped = cv2.warpAffine(_match_mono(t, cfg) * float(t.gain), M, (cw, ch),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        if warped.ndim == 2:
            warped = warped[:, :, None]

        # local (non-rigid) seam correction: MEASURE on the sharpened match canvas,
        # APPLY the identical field to the original pixels (no shortcuts — the
        # originals ride through the exact same geometry as the sharpened ones).
        if step > 0:
            dxf, dyf = _measure_local_field(running_match, match_warped, valid, cfg)
            if dxf is not None:
                warped = _warp_by_dense_field(warped, dxf, dyf)
                if warped.ndim == 2:   # cv2.remap squeezes (H,W,1) -> (H,W)
                    warped = warped[:, :, None]
                match_warped = _warp_by_dense_field(match_warped, dxf, dyf)

        # blend weight: hard for "none", distance-transform feather otherwise,
        # optionally scaled by tile quality so the sharper panel wins the seam.
        if cfg.blend == "none":
            wt = (valid > 0).astype(np.float32)
        else:
            wt = _feather_weights(valid)
            if cfg.quality_weighted_seam:
                wt = wt * (0.25 + 0.75 * float(np.clip(t.quality, 0.0, 1.0)))

        if use_mb:
            wimg = warped if warped.shape[2] == 3 else np.repeat(warped, 3, axis=2)
            mb_store.append((wimg, (valid > 0), wt.astype(np.float32, copy=False)))
        else:
            wtc = wt[:, :, None]
            accum += warped * wtc
            wsum += wtc

        coverage += (valid > 0).astype(np.float32)
        # update the SHARPENED running reference for the next tile's measurement
        running_match = np.where(valid > 0,
                                 np.maximum(running_match, match_warped),
                                 running_match)

    if use_mb:
        # Hard ownership: each pixel goes to the single valid tile with the
        # highest feather x quality weight (a winner-take-all seam). The blender
        # then does the actual cross-band feathering, so no overlapping soft
        # masks confuse its weight normalisation.
        best = np.full((ch, cw), -1.0, dtype=np.float32)
        owner = np.full((ch, cw), -1, dtype=np.int32)
        for idx, (_wi, vmask, wt) in enumerate(mb_store):
            better = (wt > best) & vmask
            owner = np.where(better, idx, owner)
            best = np.where(better, wt, best)

        blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=int(cfg.multiband_bands))
        blender.prepare((0, 0, cw, ch))
        for idx, (wi, vmask, _wt) in enumerate(mb_store):
            # fill no-data so the Laplacian pyramid doesn't bleed the black cliff
            filled = _fill_nodata(wi, vmask)
            img16 = np.clip(filled * MB_SCALE, -32768.0, 32767.0).astype(np.int16)
            mask8 = np.where(owner == idx, np.uint8(255), np.uint8(0))
            blender.feed(np.ascontiguousarray(img16),
                         np.ascontiguousarray(mask8), (0, 0))
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
    global_bundle_adjust(tiles, pairs, cfg)

    _phase(0, 1, "Equalising brightness")
    equalize_photometry(tiles, pairs, cfg)

    # TODO(sphere_reproject): if cfg.sphere_reproject, reproject each tile onto a
    # common lunar/solar sphere here using _build_lonlat_grids / derotate_stack_lonshift
    # (from setiastro.saspro.derotate) before compositing — needed for full-disk
    # and limb-spanning mosaics. v1 stitches in the image plane.

    result = compose_mosaic(tiles, cfg, progress_cb=progress_cb)
    result.pairs = pairs
    result.anchor = int(np.clip(cfg.anchor_index, 0, len(tiles) - 1))
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

            # translucent tile rectangles (overlaps darken via alpha stacking)
            centres = []
            for i, cc in enumerate(corners):
                r, g, b = self._PALETTE[i % len(self._PALETTE)]
                poly = QPolygonF([tw(pt) for pt in cc])
                fill = QColor(r, g, b, 70)
                p.setBrush(QBrush(fill))
                is_anchor = (i == anchor)
                p.setPen(QPen(QColor(r, g, b), 2.5 if is_anchor else 1.4))
                p.drawPolygon(poly)
                cx = sum(pt.x() for pt in poly) / 4.0
                cy = sum(pt.y() for pt in poly) / 4.0
                centres.append((cx, cy))

            # match-graph edges between overlapping tiles (thickness by confidence)
            for pr in (getattr(res, "pairs", None) or []):
                if 0 <= pr.i < len(centres) and 0 <= pr.j < len(centres):
                    a, b = centres[pr.i], centres[pr.j]
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
                if i == anchor:
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
                "local_warp_min_shift": _d.local_warp_min_shift,
                "multiband_bands": _d.multiband_bands,
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
            v.addWidget(gb_reg)

            # seam / local warp
            gb_seam = QGroupBox("Seam / local warp")
            fs = QFormLayout(gb_seam)
            sp_lwms = QDoubleSpinBox(); sp_lwms.setRange(0.0, 5.0); sp_lwms.setSingleStep(0.1)
            sp_lwms.setValue(float(self._adv["local_warp_min_shift"]))
            fs.addRow("Local-warp min shift (px):", sp_lwms)
            v.addWidget(gb_seam)

            # blending
            gb_bl = QGroupBox("Blending")
            fb = QFormLayout(gb_bl)
            sp_bands = QSpinBox(); sp_bands.setRange(1, 10)
            sp_bands.setValue(int(self._adv["multiband_bands"]))
            fb.addRow("Multiband bands:", sp_bands)
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
                sp_lwms.setValue(d.local_warp_min_shift)
                sp_bands.setValue(d.multiband_bands)
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
                    "local_warp_min_shift": sp_lwms.value(),
                    "multiband_bands": sp_bands.value(),
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
            try:
                push_mosaic_to_view(result.image, title="Surface Mosaic",
                                    doc_manager=self.doc_manager)
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