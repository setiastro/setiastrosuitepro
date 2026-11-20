#saspro/rgbalign.py
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QMessageBox, QProgressBar, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QGridLayout, QWidget
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QCursor


from setiastro.saspro import astroalign
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
from setiastro.saspro.sep_tiled import (
    ensure_pixstack, extract_small_first, waterfall_extract, STAR_LADDER,
)
import sep
sep.set_extract_pixstack(20000000)

import cv2

# try to reuse poly from star_alignment if present
try:
    from setiastro.saspro.star_alignment import PolynomialTransform
except Exception:
    PolynomialTransform = None


def _translate_channel(ch: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if cv2 is None:
        return ch
    H, W = ch.shape[:2]
    A = np.array([[1, 0, dx],
                [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        ch, A, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def _planet_centroid(ch: np.ndarray, signal_floor: float = 0.05):
    """
    Return (cx, cy, effective_area) of the planet disc using an
    intensity-weighted centroid (centre of mass) over all pixels
    above `signal_floor` (fraction of the channel's own max).
 
    This is the same approach used in the planetary stacker — far more
    robust than Otsu thresholding because it:
      • Handles limb darkening naturally (bright limb still dominates)
      • Is not fooled by hot pixels (they add negligible mass)
      • Degrades gracefully when the planet fills most of the frame
      • Requires no morphological cleanup
 
    Returns None if too few signal pixels are found.
    """
    img = np.asarray(ch, dtype=np.float32)
 
    # Normalise to [0, 1] robustly
    ch_max = float(np.percentile(img, 99.5))
    ch_min = float(np.percentile(img,  0.5))
    if ch_max <= ch_min + 1e-9:
        return None
 
    norm = np.clip((img - ch_min) / (ch_max - ch_min), 0.0, 1.0)
 
    # Signal mask — pixels that are part of the planet
    mask = norm > signal_floor
    n_sig = int(np.count_nonzero(mask))
    if n_sig < 16:
        return None
 
    # Intensity-weighted centroid
    weights = norm * mask          # zero outside disc, brightness inside
    total_w = float(weights.sum())
    if total_w < 1e-9:
        return None
 
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
 
    cx = float((weights * xs).sum() / total_w)
    cy = float((weights * ys).sum() / total_w)
 
    return (cx, cy, float(n_sig))



# ─────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────
class RGBAlignCancelled(RuntimeError):
    """The user cancelled an alignment or trial run."""


class RGBAlignWorker(QThread):
    progress = pyqtSignal(int, str)     # (percent, message)
    done = pyqtSignal(np.ndarray)       # aligned RGB image
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()            # stopped on request; nothing to apply
    trial_done = pyqtSignal(str)        # Trial Detect report (trial=True runs)
    EDGE_FRAC = 0.55      # 55% of max radius
    MIN_EDGE_PTS = 6      # want at least 6 out there
    EDGE_INNER_FRAC = 0.38   # toss center 38% radius
    MATCH_MAX_DIST = 10.0    # px, generous for CA
    MIN_MATCHES = 6

    # EDGE star detection steps sigma down this ladder and stops at the first
    # sigma that finds TARGET_STARS in green or TARGET_EDGE_MATCHES edge pairs.
    # The user's sigma is the floor. A fixed low sigma on a deep stack ran for
    # many minutes; a few hundred bright stars are plenty for a CA fit.
    SIGMA_LADDER = (100.0, 50.0, 20.0, 10.0, 5.0, 3.0)
    TARGET_STARS = 500
    TARGET_EDGE_MATCHES = 100
    # SEP holds the GIL for a whole extract() call, so one full-frame call froze
    # the GUI even from this thread (67 s at sigma 1 on a 9 MP frame). A sparse
    # rung is one call; a dense one goes tile by tile, which releases the GIL
    # between tiles and gives Cancel somewhere to land. On a 3008x3008 frame with
    # the GUI process's 20M pixel stack, one call held the GIL 0.8 s at 65k
    # pixels over threshold, 1.8 s at 114k, 4.1 s at 198k and 69 s at 1.76M.
    # Every call also pays ~0.2 s to allocate that stack, so tiles are not free.
    # sep_tiled.extract_strips was tried for the dense case and rejected: its
    # adaptive margin grows toward the whole frame when nebulosity crosses the
    # cuts (122 s, with one 40 s call, at sigma 1 on the same frame).
    ONE_CALL_MAX_ABOVE = 100_000
    DETECT_TILE = 768
    DETECT_PAD = 32       # overlap; each star is kept by the tile owning its centroid

    # Homography / Affine / Poly (and EDGE's fallback) fit the brightest
    # control points with astroalign. The GPU detector finds them in well under
    # a second. Its SEP fallback used a fixed sigma 5 (3 on the second pass):
    # ~40k sources on a 6248x4176 Ha master, 134 s with the GIL held 74 s per
    # pair (345 s / 189 s on the sigma-3 pass). The matcher only uses the
    # brightest max_control_points, so SEP steps down sep_tiled.STAR_LADDER
    # and stops at the first sigma that finds CP_TARGET_FACTOR times that many.
    CP_TARGET_FACTOR = 2


    def __init__(self, img: np.ndarray, model: str, sep_sigma: float = 3.0,
                 planet_floor: float = 0.05, *, trial: bool = False):
        super().__init__()
        self.img = img
        self.model = model
        self.sep_sigma = float(sep_sigma)
        self.planet_floor = float(planet_floor)
        self.trial = bool(trial)


        self.r_xform = None
        self.b_xform = None
        self.r_pairs = None
        self.b_pairs = None

        self._cancel = False
        self._stage_pct = 0
        self._bkg_cache: dict = {}   # id(channel) -> (channel, background-subtracted, global rms)
        self._det_cache: dict = {}   # (id(channel), sigma) -> (channel, (N,2) points brightest first)
        self.waterfall: dict = {}    # label -> {"rungs": [(sigma, n_green, n_src, n_match)], "sigma": used}
        self.tile_failures = 0
        self._cp_cache: dict = {}    # (id(channel), backend, sigma, area, cp) -> (channel, points, sigma used)
        self.detect_notes: dict = {} # label -> how the astroalign control points were found

    def cancel(self):
        self._cancel = True

    def _check_cancel(self):
        if self._cancel or self.isInterruptionRequested():
            raise RGBAlignCancelled("Cancelled.")

    def waterfall_lines(self) -> list:
        lines = []
        for label, info in self.waterfall.items():
            rungs = info.get("rungs") or []
            if not rungs:
                continue
            used = info.get("sigma")
            lines.append(f"{label} → Green star detection (σ waterfall):")
            for sigma, n_ref, n_src, n_match in rungs:
                mark = "   ← used" if used is not None and sigma == used else ""
                lines.append(f"  σ={sigma:g}: {n_ref} green stars, {n_src} {label.lower()} stars, "
                             f"{n_match} edge matches{mark}")
            if used is None:
                lines.append(f"  fewer than {self.MIN_MATCHES} edge matches at every σ: EDGE falls back to astroalign")
        lines.extend(self.detect_notes.values())
        if self.tile_failures:
            lines.append(f"({self.tile_failures} detection tile(s) failed and were skipped)")
        return lines

    def _pts_too_central(self, pts: np.ndarray | None, shape) -> bool:
        """
        Return True if the matched points are all bunched near the center.
        pts: (N, 2) in x,y
        shape: (H, W)
        """
        if pts is None or len(pts) == 0:
            return True
        h, w = shape[:2]
        cx, cy = w * 0.5, h * 0.5
        # distance of each point from center
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        rmax = np.hypot(cx, cy)  # radius to corner
        edge_mask = r > (self.EDGE_FRAC * rmax)
        return edge_mask.sum() < self.MIN_EDGE_PTS

    def _sigma_ladder(self) -> list:
        floor = max(0.5, float(self.sep_sigma))
        rungs = [sg for sg in self.SIGMA_LADDER if sg >= floor]
        if not rungs or rungs[-1] > floor:
            rungs.append(floor)      # a floor between or below the rungs ends the ladder
        return rungs

    def _background_sub(self, img: np.ndarray):
        hit = self._bkg_cache.get(id(img))
        if hit is not None and hit[0] is img:
            return hit[1], hit[2]
        data = np.ascontiguousarray(img, dtype=np.float32)
        bkg = sep.Background(data)
        sub = np.ascontiguousarray(data - bkg, dtype=np.float32)
        rms = float(bkg.globalrms)
        self._bkg_cache[id(img)] = (img, sub, rms)
        return sub, rms

    def _detect_tiled(self, img: np.ndarray, sigma: float):
        """SEP detection: (N,2) x,y points, brightest peak first, or None.

        One full-frame call when the rung is sparse, tile by tile when it is
        dense (ONE_CALL_MAX_ABOVE). Same background and threshold either way
        (global background model and rms); tiles overlap by DETECT_PAD and each
        star is kept only by the tile whose core holds its centroid."""
        key = (id(img), float(sigma))
        hit = self._det_cache.get(key)
        if hit is not None and hit[0] is img:
            return hit[1]
        sub, rms = self._background_sub(img)
        self._check_cancel()
        if int(np.count_nonzero(sub > float(sigma) * rms)) <= self.ONE_CALL_MAX_ABOVE:
            objs = extract_small_first(sep, sub, float(sigma), err=rms)
            pts = None
            if objs is not None and len(objs):
                order = np.argsort(objs["peak"])[::-1]
                pts = np.stack([objs["x"][order], objs["y"][order]], axis=1)
            self._det_cache[key] = (img, pts)
            return pts
        H, W = sub.shape[:2]
        T, P = int(self.DETECT_TILE), int(self.DETECT_PAD)
        xs, ys, pk = [], [], []
        for y0 in range(0, H, T):
            for x0 in range(0, W, T):
                self._check_cancel()
                y1, x1 = min(y0 + T, H), min(x0 + T, W)
                py0, px0 = max(0, y0 - P), max(0, x0 - P)
                tile = np.ascontiguousarray(sub[py0:min(H, y1 + P), px0:min(W, x1 + P)])
                try:
                    ensure_pixstack(sep, tile.size)     # only ever raises it (see sep_tiled)
                    objs = sep.extract(tile, float(sigma), err=rms)
                except Exception:
                    self.tile_failures += 1      # e.g. deblending overflow in a dense tile
                    continue
                if objs is None or len(objs) == 0:
                    continue
                ox = objs["x"] + px0
                oy = objs["y"] + py0
                lox = -np.inf if x0 == 0 else x0 - 0.5
                hix = np.inf if x1 >= W else x1 - 0.5
                loy = -np.inf if y0 == 0 else y0 - 0.5
                hiy = np.inf if y1 >= H else y1 - 0.5
                own = (ox >= lox) & (ox < hix) & (oy >= loy) & (oy < hiy)
                if own.any():
                    xs.append(ox[own])
                    ys.append(oy[own])
                    pk.append(objs["peak"][own])
        if not xs:
            pts = None
        else:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            order = np.argsort(np.concatenate(pk))[::-1]
            pts = np.stack([x[order], y[order]], axis=1)
        self._det_cache[key] = (img, pts)
        return pts

    def _sep_detect_points(self, img: np.ndarray):
        """Return (N,2) points from SEP, brightest first, at the user sigma."""
        if sep is None:
            return None
        return self._detect_tiled(img, self.sep_sigma)

    def _filter_edge_ring(self, pts: np.ndarray, shape, inner_frac=EDGE_INNER_FRAC):
        """Keep only points outside inner_frac * Rmax."""
        if pts is None or pts.size == 0:
            return None
        h, w = shape[:2]
        cx, cy = w * 0.5, h * 0.5
        r = np.hypot(pts[:,0] - cx, pts[:,1] - cy)
        rmax = np.hypot(cx, cy)
        mask = r >= (inner_frac * rmax)
        pts_edge = pts[mask]
        return pts_edge if pts_edge.size else None

    def _match_edge(self, src_pts, ref_pts, shape):
        """Keep REF's edge ring, then nearest-neighbour match into SRC within MATCH_MAX_DIST."""
        if ref_pts is None or src_pts is None:
            return None, None
        ref_edge = self._filter_edge_ring(ref_pts, shape)
        if ref_edge is None:
            return None, None
        src_arr = np.asarray(src_pts, dtype=np.float32)
        ref_arr = np.asarray(ref_edge, dtype=np.float32)
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        if cKDTree is not None:
            # Same nearest neighbour as the old per-star argmin loop, which was
            # O(N^2) and took tens of seconds once N reached the tens of thousands.
            dist, j = cKDTree(src_arr).query(ref_arr, k=1)
            ok = dist <= self.MATCH_MAX_DIST
            return src_arr[j[ok]], ref_arr[ok]
        pairs_src, pairs_dst = [], []
        for (x_ref, y_ref) in ref_arr:
            dxy = src_arr - np.array([x_ref, y_ref], dtype=np.float32)
            dist = np.hypot(dxy[:, 0], dxy[:, 1])
            jj = int(np.argmin(dist))
            if dist[jj] <= self.MATCH_MAX_DIST:
                pairs_src.append(src_arr[jj])
                pairs_dst.append([x_ref, y_ref])
        return (np.array(pairs_src, dtype=np.float32).reshape(-1, 2),
                np.array(pairs_dst, dtype=np.float32).reshape(-1, 2))

    def _pair_edge_points(self, src_img, ref_img, shape, label: str = ""):
        """EDGE pairs via the sigma waterfall.

        At each sigma: detect in BOTH images, keep the edge ring in REF, NN-match
        into SRC. Stop at the first sigma reaching TARGET_STARS green stars or
        TARGET_EDGE_MATCHES pairs (with at least MIN_MATCHES to fit). Returns the
        pairs from the best sigma tried, or (None, None) if none had MIN_MATCHES."""
        name = label or f"Channel {len(self.waterfall) + 1}"
        rungs = []
        best = (None, None, None, -1)      # src_xy, dst_xy, sigma, n_match
        for sigma in self._sigma_ladder():
            self._check_cancel()
            self.progress.emit(self._stage_pct, f"Aligning {name} → Green: detecting stars at σ={sigma:g}…")
            ref_pts = self._detect_tiled(ref_img, sigma)
            src_pts = self._detect_tiled(src_img, sigma)
            n_ref = 0 if ref_pts is None else len(ref_pts)
            n_src = 0 if src_pts is None else len(src_pts)
            s_xy, d_xy = self._match_edge(src_pts, ref_pts, shape)
            n_match = 0 if s_xy is None else len(s_xy)
            rungs.append((float(sigma), n_ref, n_src, n_match))
            if n_match >= best[3]:
                best = (s_xy, d_xy, float(sigma), n_match)
            if n_match >= self.MIN_MATCHES and (n_ref >= self.TARGET_STARS
                                                 or n_match >= self.TARGET_EDGE_MATCHES):
                break
        ok = best[3] >= self.MIN_MATCHES
        self.waterfall[name] = {"rungs": rungs, "sigma": best[2] if ok else None}
        if not ok:
            return None, None
        return best[0], best[1]


    def run(self):
        if self.img is None or self.img.ndim != 3 or self.img.shape[2] < 3:
            self.failed.emit("Image must be RGB (3 channels).")
            return
        if astroalign is None and not self.trial:
            self.failed.emit("astroalign is not available.")
            return

        try:
            self.progress.emit(5, "Preparing channels…")
            R = np.ascontiguousarray(self.img[..., 0].astype(np.float32, copy=False))
            G = np.ascontiguousarray(self.img[..., 1].astype(np.float32, copy=False))
            B = np.ascontiguousarray(self.img[..., 2].astype(np.float32, copy=False))

            if self.trial:
                self._run_trial(R, G, B)
                return

            # R → G
            self._stage_pct = 15
            self.progress.emit(15, "Aligning Red → Green…")
            kind_R, X_R, (r_src, r_dst) = self._estimate_transform(R, G, self.model, label="Red")
            self._check_cancel()
            self.r_xform = (kind_R, X_R)
            self.r_pairs = (r_src, r_dst)
            self.progress.emit(35, f"Red transform = {kind_R}")
            R_aligned = self._warp_channel(R, kind_R, X_R, G.shape)

            # B → G
            self._check_cancel()
            self._stage_pct = 55
            self.progress.emit(55, "Aligning Blue → Green…")
            kind_B, X_B, (b_src, b_dst) = self._estimate_transform(B, G, self.model, label="Blue")
            self._check_cancel()
            self.b_xform = (kind_B, X_B)
            self.b_pairs = (b_src, b_dst)
            self.progress.emit(75, f"Blue transform = {kind_B}")
            B_aligned = self._warp_channel(B, kind_B, X_B, G.shape)

            out = np.stack([R_aligned, G, B_aligned], axis=2).astype(self.img.dtype, copy=False)
            self._check_cancel()
            self.progress.emit(100, "Done.")
            self.done.emit(out)
        except RGBAlignCancelled:
            self.cancelled.emit()
        except Exception as e:
            if self._cancel or self.isInterruptionRequested():
                self.cancelled.emit()
            else:
                self.failed.emit(str(e))

    def _run_trial(self, R, G, B):
        """Trial Detect: the EDGE waterfall for red and blue, reported, no fit or warp."""
        H, W = G.shape[:2]
        self._stage_pct = 15
        self._pair_edge_points(R, G, (H, W), label="Red")
        self._check_cancel()
        self._stage_pct = 55
        self._pair_edge_points(B, G, (H, W), label="Blue")
        self.progress.emit(100, "Trial detect done.")
        self.trial_done.emit("\n".join(["[Trial detect: EDGE σ waterfall]"] + self.waterfall_lines()))

    def _planet_centroid(self, ch: np.ndarray, signal_floor: float = 0.05):
        """
        Intensity-weighted centroid over pixels above signal_floor.
        Same algorithm as the module-level function — kept as a method
        so the worker can call it without importing the module function.
        """
        img = np.asarray(ch, dtype=np.float32)
 
        ch_max = float(np.percentile(img, 99.5))
        ch_min = float(np.percentile(img,  0.5))
        if ch_max <= ch_min + 1e-9:
            return None
 
        norm = np.clip((img - ch_min) / (ch_max - ch_min), 0.0, 1.0)
 
        mask = norm > signal_floor
        n_sig = int(np.count_nonzero(mask))
        if n_sig < 16:
            return None
 
        weights = norm * mask
        total_w = float(weights.sum())
        if total_w < 1e-9:
            return None
 
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
 
        cx = float((weights * xs).sum() / total_w)
        cy = float((weights * ys).sum() / total_w)
 
        return (cx, cy, float(n_sig))

    def _estimate_translation_by_centroid(
        self,
        src: np.ndarray,
        ref: np.ndarray,
        refine_ssd: bool = True,
        ssd_radius: int = 8,
    ):
        """
        Estimate the (dx, dy) shift to align src to ref using:
 
          Pass 1 — Intensity-weighted centroid.
                   Find the planet centre in each channel and compute
                   the shift that moves src's centroid onto ref's centroid.
                   This handles the bulk of atmospheric dispersion.
 
          Pass 2 — SSD gradient refinement (optional, default on).
                   Apply the centroid shift to src, then minimise the
                   Sum of Squared Differences between the gradient images
                   of the shifted src and ref over a small search window.
                   The grey-world hypothesis means that when R, G, B are
                   properly aligned their spatial structure is identical,
                   so the minimum SSD gives the true sub-pixel alignment.
 
        Returns:
            (dx, dy, src_info, ref_info)  or  None if centroid fails.
 
        src_info / ref_info are (cx, cy, n_sig) tuples for diagnostics.
        """
        c_src = self._planet_centroid(src, signal_floor=self.planet_floor)
        c_ref = self._planet_centroid(ref, signal_floor=self.planet_floor)
 
        if c_src is None or c_ref is None:
            return None
 
        sx, sy, sa = c_src
        rx, ry, ra = c_ref
 
        # Centroid shift (coarse)
        dx = rx - sx
        dy = ry - sy
 
        # ── SSD refinement ────────────────────────────────────────────────
        # Apply the centroid shift to src, then find the residual shift
        # that minimises the gradient SSD in a small window around (0,0).
        if refine_ssd and cv2 is not None:
            try:
                # Shift src by the coarse estimate
                H, W = src.shape[:2]
                M_coarse = np.array([[1, 0, dx],
                                     [0, 1, dy]], dtype=np.float32)
                src_shifted = cv2.warpAffine(
                    src.astype(np.float32, copy=False),
                    M_coarse, (W, H),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REPLICATE,
                )
 
                # Gradient images (Sobel) — captures structure, not brightness
                def _grad(im):
                    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
                    g  = cv2.magnitude(gx, gy)
                    g -= float(g.mean())
                    s  = float(g.std()) + 1e-6
                    return (g / s).astype(np.float32)
 
                ref_g = _grad(ref.astype(np.float32, copy=False))
                src_g = _grad(src_shifted)
 
                # Crop centre 80% to avoid border artifacts from warpAffine
                cfy = max(8, int(H * 0.10))
                cfx = max(8, int(W * 0.10))
                ref_gc = ref_g[cfy:H-cfy, cfx:W-cfx]
                src_gc = src_g[cfy:H-cfy, cfx:W-cfx]
 
                # Brute-force SSD scan over [-ssd_radius, +ssd_radius]
                r = int(ssd_radius)
                best_ssd = float("inf")
                best_ddx = 0
                best_ddy = 0
 
                ch_, cw_ = ref_gc.shape[:2]
                for ddy in range(-r, r + 1):
                    for ddx in range(-r, r + 1):
                        # Overlap slices
                        x0r = max(0,  ddx);  x1r = min(cw_,  cw_ + ddx)
                        y0r = max(0,  ddy);  y1r = min(ch_,  ch_ + ddy)
                        x0s = max(0, -ddx);  x1s = min(cw_, cw_  - ddx)
                        y0s = max(0, -ddy);  y1s = min(ch_, ch_  - ddy)
 
                        rr = ref_gc[y0r:y1r, x0r:x1r]
                        ss = src_gc[y0s:y1s, x0s:x1s]
                        if rr.size == 0:
                            continue
                        d = rr - ss
                        ssd = float(np.mean(d * d))
                        if ssd < best_ssd:
                            best_ssd = ssd
                            best_ddx = ddx
                            best_ddy = ddy
 
                # Subpixel quadratic polish (separable)
                def _quad_min(vm, v0, vp):
                    denom = vm - 2.0 * v0 + vp
                    if abs(denom) < 1e-12:
                        return 0.0
                    return float(np.clip(0.5 * (vm - vp) / denom, -0.75, 0.75))
 
                def _ssd_at(ddx_, ddy_):
                    x0r = max(0,  ddx_);  x1r = min(cw_,  cw_ + ddx_)
                    y0r = max(0,  ddy_);  y1r = min(ch_,  ch_ + ddy_)
                    x0s = max(0, -ddx_);  x1s = min(cw_, cw_  - ddx_)
                    y0s = max(0, -ddy_);  y1s = min(ch_, ch_  - ddy_)
                    rr = ref_gc[y0r:y1r, x0r:x1r]
                    ss = src_gc[y0s:y1s, x0s:x1s]
                    if rr.size == 0:
                        return best_ssd
                    d = rr - ss
                    return float(np.mean(d * d))
 
                sub_x = 0.0
                sub_y = 0.0
                if abs(best_ddx - 1) <= r:
                    sub_x = _quad_min(
                        _ssd_at(best_ddx - 1, best_ddy),
                        best_ssd,
                        _ssd_at(best_ddx + 1, best_ddy),
                    )
                if abs(best_ddy - 1) <= r:
                    sub_y = _quad_min(
                        _ssd_at(best_ddx, best_ddy - 1),
                        best_ssd,
                        _ssd_at(best_ddx, best_ddy + 1),
                    )
 
                dx += float(best_ddx) + sub_x
                dy += float(best_ddy) + sub_y
 
            except Exception as e:
                # SSD refinement is best-effort; fall back to centroid-only
                print(f"[RGBAlign] SSD refinement failed (using centroid only): {e}")
 
        return (dx, dy, c_src, c_ref)



    # ───── helpers (basically mini versions of your big star alignment logic) ─────
    def _control_points(self, img: np.ndarray, backend: str, sigma: float, area: int, cp: int):
        """Stars for astroalign: ((N,2) x,y brightest first or None, sigma used).

        "gpu": astroalign's GPU detector at `sigma` (None when it is switched
        off or unavailable). "sep": the SEP waterfall, STAR_LADDER down to
        `sigma`, stopping at CP_TARGET_FACTOR * cp stars."""
        key = (id(img), backend, float(sigma), int(area), int(cp))
        hit = self._cp_cache.get(key)
        if hit is not None and hit[0] is img:
            return hit[1], hit[2]
        self._check_cancel()
        pts, used = None, None
        if backend == "gpu":
            pts = astroalign._find_sources(img, detection_sigma=sigma, min_area=area,
                                           backend="gpu", max_stars=cp)
            used = float(sigma)
        else:
            ladder = [r for r in STAR_LADDER if r[0] >= sigma]
            if not ladder or ladder[-1][0] > sigma:
                ladder.append((float(sigma), int(area)))
            r = waterfall_extract(sep, img, self.CP_TARGET_FACTOR * int(cp), ladder=ladder)
            if r is not None and len(r["cat"]):
                cat = r["cat"]
                order = np.argsort(-np.asarray(cat["flux"]))
                pts = np.stack([cat["x"][order], cat["y"][order]], axis=1).astype(np.float64)
                used = float(r["sigma"])
        self._cp_cache[key] = (img, pts, used)
        return pts, used

    def _astroalign_transform(self, src, ref, cp: int, sigma: float, area: int, label: str):
        """astroalign.find_transform on detected control points, in the order
        its own "auto" detector uses: GPU first; SEP when the GPU detector is
        off, finds fewer than 3 stars, or its match fails.
        Returns (tform, (src_pts, dst_pts), note)."""
        name = label or "Channel"
        err = None
        for backend in ("gpu", "sep"):
            self._check_cancel()
            if backend == "sep":
                self.progress.emit(self._stage_pct,
                                   f"Aligning {name} → Green: detecting stars (SEP, σ waterfall to {sigma:g})…")
            s_pts, s_sig = self._control_points(src, backend, sigma, area, cp)
            r_pts, r_sig = self._control_points(ref, backend, sigma, area, cp)
            if s_pts is None or r_pts is None or len(s_pts) < 3 or len(r_pts) < 3:
                continue
            try:
                tform, (sp, dp) = astroalign.find_transform(s_pts, r_pts, max_control_points=cp)
            except Exception as e:
                err = e
                continue
            if backend == "gpu":
                note = f"{name} → Green: GPU star detection at σ={sigma:g}, {len(sp)} matched stars"
            else:
                note = (f"{name} → Green: SEP star detection, σ={s_sig:g} ({name.lower()}) / "
                        f"σ={r_sig:g} (green), {len(sp)} matched stars")
            return tform, (sp, dp), note
        if err is not None:
            raise err
        raise ValueError(f"Too few stars to align {name} to Green.")

    def _estimate_transform(self, src: np.ndarray, ref: np.ndarray, model: str, label: str = ""):
        H, W = ref.shape[:2]

        # ── Planet centroid ADC mode ─────────────────────────────
        if model == "planet-centroid":
            t = self._estimate_translation_by_centroid(src, ref, refine_ssd=True, ssd_radius=8)
            if t is None:
                # fall through to astroalign if centroid fails
                pass
            else:
                dx, dy, src_info, ref_info = t
                # src_info / ref_info are (cx, cy, n_sig) tuples
                src_xy = np.array([[src_info[0], src_info[1]]], dtype=np.float32)
                dst_xy = np.array([[ref_info[0], ref_info[1]]], dtype=np.float32)
                X = (float(dx), float(dy))
                return ("translate", X, (src_xy, dst_xy))

        # ── 0) edge-only, SEP-based path ─────────────────────────────
        if model == "edge-sep":
            src_xy, dst_xy = self._pair_edge_points(src, ref, (H, W), label=label)
            if src_xy is not None and dst_xy is not None and cv2 is not None:
                # 0a) try homography first (better for corner warp)
                Hh, inliers = cv2.findHomography(
                    src_xy, dst_xy,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2000,
                    confidence=0.999,
                )
                if Hh is not None:
                    return ("homography", Hh, (src_xy, dst_xy))

                # 0b) fallback → affine
                A, inliers = cv2.estimateAffine2D(
                    src_xy, dst_xy,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2000,
                    confidence=0.999,
                )
                if A is not None:
                    return ("affine", A, (src_xy, dst_xy))
            # if SEP failed or cv2 missing → fall through to astroalign normal path

        # ─────────────────────────────────────────────────────────────
        # 1) astroalign normal pass
        # ─────────────────────────────────────────────────────────────
        src = np.ascontiguousarray(src)
        ref = np.ascontiguousarray(ref)
        tform, (src_pts, dst_pts), note = self._astroalign_transform(src, ref, 50, 5.0, 5, label)

        # 2) 'hungry' pass if too central
        if self._pts_too_central(dst_pts, ref.shape):
            tform2, (src_pts2, dst_pts2), note2 = self._astroalign_transform(src, ref, 120, 3.0, 3, label)
            if not self._pts_too_central(dst_pts2, ref.shape):
                tform, src_pts, dst_pts, note = tform2, src_pts2, dst_pts2, note2
        self.detect_notes[label or f"Channel {len(self.detect_notes) + 1}"] = note

        # 3) original branching
        P = np.asarray(tform.params, dtype=np.float64)
        src_xy = np.asarray(src_pts, dtype=np.float32)
        dst_xy = np.asarray(dst_pts, dtype=np.float32)

        # affine
        if model == "affine":
            if cv2 is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            A, _ = cv2.estimateAffine2D(
                src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if A is None:
                return ("affine", P[0:2, :], (src_xy, dst_xy))
            return ("affine", A, (src_xy, dst_xy))

        # homography
        if model == "homography":
            if cv2 is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            Hh, _ = cv2.findHomography(
                src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if Hh is None:
                if P.shape == (3, 3):
                    return ("homography", P, (src_xy, dst_xy))
                A3 = np.vstack([P[0:2, :], [0, 0, 1]])
                return ("homography", A3, (src_xy, dst_xy))
            return ("homography", Hh, (src_xy, dst_xy))

        # poly3 / poly4
        if model in ("poly3", "poly4") and PolynomialTransform is not None and cv2 is not None:
            order = 3 if model == "poly3" else 4
            scale_vec = np.array([W, H], dtype=np.float32)
            src_n = src_xy / scale_vec
            dst_n = dst_xy / scale_vec

            t_poly = PolynomialTransform()
            ok = t_poly.estimate(dst_n, src_n, order=order)  # dst → src
            if not ok:
                Hh, _ = cv2.findHomography(
                    src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                return ("homography", Hh, (src_xy, dst_xy))

            def _warp_poly(img: np.ndarray, out_shape: tuple[int, int]):
                Hh_, Ww_ = out_shape
                yy, xx = np.mgrid[0:Hh_, 0:Ww_].astype(np.float32)
                coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
                coords_n = coords / scale_vec
                mapped_n = t_poly(coords_n)
                mapped = mapped_n * scale_vec
                map_x = mapped[:, 0].reshape(Hh_, Ww_).astype(np.float32)
                map_y = mapped[:, 1].reshape(Hh_, Ww_).astype(np.float32)
                return cv2.remap(
                    img, map_x, map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )

            return (model, _warp_poly, (src_xy, dst_xy))

        # fallback → homography
        if cv2 is None:
            if P.shape == (3, 3):
                return ("homography", P, (src_xy, dst_xy))
            A3 = np.vstack([P[0:2, :], [0, 0, 1]])
            return ("homography", A3, (src_xy, dst_xy))

        Hh, _ = cv2.findHomography(
            src_xy, dst_xy, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        return ("homography", Hh, (src_xy, dst_xy))



    def _pick_edge_stars_with_sep(self, img, tiles=(3,3), per_tile=2):
        if sep is None:
            return []
        data = img.astype(np.float32, copy=False)
        bkg = sep.Background(data)
        data_sub = data - bkg
        objs = sep.extract(data_sub, 1.5, err=bkg.globalrms)
        H, W = data.shape[:2]
        th, tw = H // tiles[0], W // tiles[1]
        picked = []
        for ty in range(tiles[0]):
            for tx in range(tiles[1]):
                y0, y1 = ty*th, min((ty+1)*th, H)
                x0, x1 = tx*tw, min((tx+1)*tw, W)
                box = objs[
                    (objs['y'] >= y0) & (objs['y'] < y1) &
                    (objs['x'] >= x0) & (objs['x'] < x1)
                ]
                if len(box) == 0:
                    continue
                # brightest first
                box = box[np.argsort(box['peak'])][::-1][:per_tile]
                for o in box:
                    picked.append((float(o['x']), float(o['y'])))
        return picked


    def _warp_channel(self, ch: np.ndarray, kind: str, X, ref_shape):
        H, W = ref_shape[:2]

        if kind == "translate":
            dx, dy = X
            A = np.array([[1, 0, dx],
                        [0, 1, dy]], dtype=np.float32)
            return cv2.warpAffine(
                ch, A, (W, H),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        if kind == "affine":
            # Just assume cv2 is available (standard dependency) for perf
            A = np.asarray(X, dtype=np.float32).reshape(2, 3)
            return cv2.warpAffine(ch, A, (W, H), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind == "homography":
            Hm = np.asarray(X, dtype=np.float32).reshape(3, 3)
            return cv2.warpPerspective(ch, Hm, (W, H), flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if kind.startswith("poly"):
            return X(ch, (H, W))

        return ch


# ─────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────
class RGBAlignDialog(QDialog):
    def __init__(self, parent=None, document=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("RGB Align"))
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
        self.parent = parent
        # document could be a view; try to unwrap
        self.doc_view = document
        self.doc = getattr(document, "document", document)
        self.image = getattr(self.doc, "image", None) if self.doc is not None else None

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Align R and B channels to G.\n"
                             "Select model and run."))

        hl = QHBoxLayout()
        hl.addWidget(QLabel(self.tr("Alignment model:")))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Planet Centroid (ADC)",   # NEW
            "EDGE",           # ← first, new default
            "Homography",
            "Affine",
            "Poly 3",
            "Poly 4",
        ])
        self.model_combo.setCurrentIndex(1)
        self.model_combo.setItemData(
            0,
            (
                "Planet Centroid (Atmospheric Dispersion)\n"
                "• Find planet disk in each channel\n"
                "• Compute centroid (moments)\n"
                "• Shift R and B so their centroids match G\n"
                "• Translation only (no warp), ideal for planets"
            ),
            Qt.ItemDataRole.ToolTipRole,
        )
        # tooltips for each mode
        self.model_combo.setItemData(
            1,
            (
                "EDGE (Edge-Detected Guided Estimator)\n"
                "• Detect stars in both channels with SEP\n"
                "• Keep only outer-ring stars (ignore center)\n"
                "• Try homography first for corner CA\n"
                "• If homography fails → try affine\n"
                "• If that fails → fall back to astroalign"
            ),
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            2,
            "Standard homography using astroalign matches (good general-purpose choice).",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            3,
            "Affine (shift + scale + rotate + shear). Good when channels are mostly parallel.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            4,
            "Polynomial (order 3). Use when you have mild field distortion.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.model_combo.setItemData(
            5,
            "Polynomial (order 4). Use for stronger distortion, but needs more/better matches.",
            Qt.ItemDataRole.ToolTipRole,
        )
        hl.addWidget(self.model_combo)
        lay.addLayout(hl)

        # ── SEP controls ─────────────────────────
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel(self.tr("Lowest SEP sigma:")))
        self.sep_spin = QSpinBox()
        self.sep_spin.setRange(1, 100)
        self.sep_spin.setValue(3)
        self.sep_spin.setToolTip("EDGE detects stars at σ = 100, 50, 20, 10, 5, 3 in turn and stops at the\n"
                                 "first that finds 500 stars in green or 100 edge-ring matches.\n"
                                 "This is the lowest σ it may go to; a value below 3 adds one last step.")
        sep_row.addWidget(self.sep_spin)
        self.btn_trial_sep = QPushButton(self.tr("Trial detect stars"))
        self.btn_trial_sep.setToolTip("Run the EDGE star detection for red and blue against green without aligning:\n"
                                      "for each σ tried, the green stars and edge matches, and the σ Align would use.")
        self.btn_trial_sep.clicked.connect(self._trial_sep_detect)
        sep_row.addWidget(self.btn_trial_sep)
        # wrap in a widget so we can show/hide the whole row including its label
        self.sep_row_widget = QWidget()
        self.sep_row_widget.setLayout(sep_row)
        lay.addWidget(self.sep_row_widget)

        # ── Planet controls (planet centroid mode) ────────────────────────────
        from PyQt6.QtWidgets import QDoubleSpinBox
        planet_row = QHBoxLayout()
        planet_row.addWidget(QLabel(self.tr("Signal floor (0..1):")))
        self.planet_floor_spin = QDoubleSpinBox()
        self.planet_floor_spin.setRange(0.01, 0.50)
        self.planet_floor_spin.setSingleStep(0.01)
        self.planet_floor_spin.setDecimals(2)
        self.planet_floor_spin.setValue(0.05)
        self.planet_floor_spin.setToolTip(
            "Pixels with normalised brightness above this value are considered\n"
            "part of the planet disc for centroid calculation.\n"
            "Lower = include more of the disc (better for dim limbs).\n"
            "Higher = ignore faint halo/noise (better for bright planets)."
        )
        planet_row.addWidget(self.planet_floor_spin)
        planet_row.addStretch(1)
        self.planet_row_widget = QWidget()
        self.planet_row_widget.setLayout(planet_row)
        lay.addWidget(self.planet_row_widget)

        self.chk_new_doc = QCheckBox(self.tr("Create new document (keep original)"))
        self.chk_new_doc.setChecked(True)
        lay.addWidget(self.chk_new_doc)

        #sasprogress
        self.progress_label = QLabel("Idle.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_label)
        lay.addWidget(self.progress_bar)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("Transform summary will appear here…")
        self.summary_box.setMinimumHeight(140)
        # optional: monospace
        self.summary_box.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        lay.addWidget(self.summary_box)

        btns = QHBoxLayout()
        self.btn_run = QPushButton(self.tr("Align"))

        self.btn_manual = QPushButton(self.tr("Manual…"))
        self.btn_manual.setToolTip("Manual planetary RGB alignment (nudge channels by 1 px with preview).")

        self.btn_manual.clicked.connect(self._open_manual)

        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop the running alignment or trial detect. Nothing is applied.")
        self.btn_close = QPushButton(self.tr("Close"))
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_manual)
        btns.addWidget(self.btn_close)
        btns.addStretch(1)
        btns.addWidget(make_help_button("rgb_align", self))
        lay.addLayout(btns)

        self.btn_run.clicked.connect(self._start_align)
        self.btn_cancel.clicked.connect(self._cancel_align)
        self.btn_close.clicked.connect(self.close)
        self.model_combo.currentIndexChanged.connect(self._update_mode_ui)
        self._update_mode_ui()   # set initial state

        self.worker: RGBAlignWorker | None = None
        self._closing_after_cancel = False
        self._cancel_requested = False

    def _open_manual(self):
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return

        dlg = RGBAlignManualDialog(self, self.image, planet_roi=("planet" in self.model_combo.currentText().lower()))

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        rdx, rdy, bdx, bdy = dlg.result_offsets()
        self.summary_box.setPlainText(
            "[Manual Planetary]\n"
            f"R shift: dx={rdx}, dy={rdy}\n"
            f"B shift: dx={bdx}, dy={bdy}\n"
        )

        # apply full-res
        img = self.image
        R = img[..., 0].astype(np.float32, copy=False)
        G = img[..., 1].astype(np.float32, copy=False)
        B = img[..., 2].astype(np.float32, copy=False)

        R2 = _translate_channel(R, rdx, rdy)
        B2 = _translate_channel(B, bdx, bdy)

        out = np.stack([R2, G, B2], axis=2)
        if out.dtype != img.dtype:
            out = out.astype(img.dtype, copy=False)

        # same create-new-doc logic you already use
        try:
            if self.chk_new_doc.isChecked():
                dm = getattr(self.parent, "docman", None)
                if dm is not None:
                    dm.open_array(out, {"display_name": "RGB Aligned (Manual)"}, title="RGB Aligned (Manual)")
                else:
                    if hasattr(self.doc, "apply_edit"):
                        self.doc.apply_edit(out, {"step_name": "RGB Align (Manual)"}, step_name="RGB Align (Manual)")
                    else:
                        self.doc.image = out
            else:
                if hasattr(self.doc, "apply_edit"):
                    self.doc.apply_edit(out, {"step_name": "RGB Align (Manual)"}, step_name="RGB Align (Manual)")
                else:
                    self.doc.image = out
        except Exception as e:
            QMessageBox.warning(self, "RGB Align", f"Manual aligned image created, but applying failed:\n{e}")

    def _update_mode_ui(self):
        """Show SEP controls for star-based modes, planet floor for planet mode."""
        is_planet = "planet" in self.model_combo.currentText().lower() \
                    or "centroid" in self.model_combo.currentText().lower() \
                    or "adc" in self.model_combo.currentText().lower()
 
        self.sep_row_widget.setVisible(not is_planet)
        self.planet_row_widget.setVisible(is_planet)


    def _trial_sep_detect(self):
        if self._busy():
            return
        if self.image is None:
            QMessageBox.warning(self, "RGB Align", "No image loaded.")
            return
        if self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return
        if sep is None:
            QMessageBox.warning(self, "RGB Align", "python-sep is not available.")
            return
        # Same detection as Align, on a worker: it used to run on the GUI thread.
        self._launch(
            RGBAlignWorker(self.image, "edge-sep", sep_sigma=float(self.sep_spin.value()), trial=True),
            "Trial detection in progress…",
        )

    # ---- run lifecycle (align and trial share it) ----
    def _busy(self) -> bool:
        return thread_running(self.worker)

    def _set_busy(self, busy: bool):
        for b in (self.btn_run, self.btn_trial_sep, self.btn_manual):
            b.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.btn_cancel.setText(self.tr("Cancel"))

    def _launch(self, worker: RGBAlignWorker, label: str):
        """Start a worker this dialog does not own (thread_keeper does), so closing
        or deleting the dialog mid-run can never destroy a running QThread."""
        self._cancel_requested = False
        self.progress_label.setText(label)
        self.progress_bar.setValue(0)
        worker.progress.connect(self._on_worker_progress)
        worker.done.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.trial_done.connect(self._on_trial_done)
        worker.finished.connect(self._on_worker_finished)
        self.worker = worker
        self._set_busy(True)
        start_owned_thread(worker)

    def _cancel_align(self):
        w = self.worker
        if not thread_running(w):
            return
        self._cancel_requested = True
        try:
            w.cancel()
            w.requestInterruption()
        except RuntimeError:
            pass
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText(self.tr("Cancelling…"))
        self.progress_label.setText("Cancelling…")

    def _on_worker_cancelled(self):
        self.progress_bar.setValue(0)
        self.progress_label.setText("Cancelled. Nothing was applied.")

    def _on_trial_done(self, report: str):
        self.summary_box.setPlainText(report)
        self.progress_label.setText("Trial detect done: see the summary below.")

    def _on_worker_finished(self):
        self._set_busy(False)
        if self._closing_after_cancel:
            self._closing_after_cancel = False
            QTimer.singleShot(0, self.close)

    def closeEvent(self, e):
        if self._busy():
            # Stop the worker first; the window closes itself once it has stopped.
            self._closing_after_cancel = True
            self._cancel_align()
            e.ignore()
            return
        super().closeEvent(e)

    def reject(self):
        # Esc behaves like closing the window.
        if self._busy():
            self._closing_after_cancel = True
            self._cancel_align()
            return
        super().reject()


    def _start_align(self):
        if self._busy():
            return
        if self.image is None:
            QMessageBox.warning(self, "RGB Align", "No image found in active view.")
            return
        if self.image.ndim != 3 or self.image.shape[2] < 3:
            QMessageBox.warning(self, "RGB Align", "Image must be RGB (3 channels).")
            return
        if astroalign is None:
            QMessageBox.warning(self, "RGB Align", "astroalign is not available.")
            return

        model = self._selected_model()
        sep_sigma = float(self.sep_spin.value())
        self.progress_label.setText("Starting…")
        self.progress_bar.setValue(0)
        is_planet = "planet" in self.model_combo.currentText().lower() \
                    or "centroid" in self.model_combo.currentText().lower() \
                    or "adc" in self.model_combo.currentText().lower()
        planet_floor = float(self.planet_floor_spin.value()) if is_planet else 0.05
        self._launch(RGBAlignWorker(
            self.image, model,
            sep_sigma=sep_sigma,
            planet_floor=planet_floor,
        ), "Starting…")

    def _selected_model(self) -> str:
        txt = self.model_combo.currentText().lower()
        if "planet" in txt or "centroid" in txt or "adc" in txt:
            return "planet-centroid"
        if "edge" in txt:
            return "edge-sep"
        if "affine" in txt:
            return "affine"
        if "poly 3" in txt:
            return "poly3"
        if "poly 4" in txt:
            return "poly4"
        if "homography" in txt:
            return "homography"
        return "edge-sep"

    # slots
    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_worker_failed(self, err: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Failed.")
        QMessageBox.critical(self, "RGB Align", err)

    def _on_worker_done(self, out: np.ndarray):
        if self._cancel_requested:
            # Cancel was pressed after the last step: cancel still wins.
            self._on_worker_cancelled()
            return
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Applying…")

        summary_lines = []
        w = self.worker  # type: ignore

        if w is not None:
            def _fmt_mat(M):
                return "\n".join(
                    ["    " + "  ".join(f"{v: .6f}" for v in row) for row in M]
                )

            def _spread_stats(pts, shape):
                if pts is None:
                    return "  points: 0"
                pts = np.asarray(pts, dtype=float)
                if pts.size == 0:
                    return "  points: 0"
                h, w_ = shape[:2]
                cx, cy = w_ * 0.5, h * 0.5
                if pts.ndim != 2 or pts.shape[1] != 2:
                    return f"  points: {len(pts)} (unusual shape {pts.shape})"
                r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                rmax = np.hypot(cx, cy)
                edge = r > (RGBAlignWorker.EDGE_FRAC * rmax)
                return (
                    f"  points: {len(pts)} "
                    f"(edge: {edge.sum()} ≥{RGBAlignWorker.EDGE_FRAC*100:.0f}%Rmax)"
                )

            h_img, w_img = self.image.shape[:2]

            # ── R → G ──
            if w.r_xform is not None:
                kind, X = w.r_xform
                summary_lines.append("Red → Green:")
                if w.r_pairs is not None and len(w.r_pairs) == 2:
                    summary_lines.append(_spread_stats(w.r_pairs[1], (h_img, w_img)))
                summary_lines.append(f"  model: {kind}")
                if kind == "translate":
                    dx, dy = X
                    summary_lines.append(f"  shift: dx={dx:.3f}, dy={dy:.3f}")                
                elif kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    Hm = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(Hm))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

            # ── B → G ──
            if w.b_xform is not None:
                kind, X = w.b_xform
                summary_lines.append("")
                summary_lines.append("Blue → Green:")
                if w.b_pairs is not None and len(w.b_pairs) == 2:
                    summary_lines.append(_spread_stats(w.b_pairs[1], (h_img, w_img)))
                summary_lines.append(f"  model: {kind}")
                if kind == "translate":
                    dx, dy = X
                    summary_lines.append(f"  shift: dx={dx:.3f}, dy={dy:.3f}")                
                elif kind == "affine":
                    A = np.asarray(X, dtype=float).reshape(2, 3)
                    M = np.vstack([A, [0, 0, 1]])
                    summary_lines.append(_fmt_mat(M))
                elif kind == "homography":
                    Hm = np.asarray(X, dtype=float).reshape(3, 3)
                    summary_lines.append(_fmt_mat(Hm))
                else:
                    summary_lines.append("  (non-matrix; warp callable)")

        if w is not None and getattr(w, "waterfall", None):
            summary_lines = list(w.waterfall_lines()) + ([""] if summary_lines else []) + summary_lines
        summary_text = "\n".join(summary_lines) if summary_lines else "No transform info."
        self.summary_box.setPlainText(summary_text)

        if self.parent is not None and hasattr(self.parent, "_log") and callable(self.parent._log):
            self.parent._log("[RGB Align]\n" + summary_text)

        try:
            if self.chk_new_doc.isChecked():
                dm = getattr(self.parent, "docman", None)
                if dm is not None:
                    dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
                else:
                    if hasattr(self.doc, "apply_edit"):
                        self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                    else:
                        self.doc.image = out
            else:
                if hasattr(self.doc, "apply_edit"):
                    self.doc.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
                else:
                    self.doc.image = out

            self.progress_label.setText("Done.")
        except Exception as e:
            self.progress_label.setText("Apply failed.")
            QMessageBox.warning(self, "RGB Align", f"Aligned image created, but applying failed:\n{e}")





def align_rgb_array(img: np.ndarray, model: str = "edge-sep", sep_sigma: float = 3.0) -> np.ndarray:
    """
    Headless core: returns a new RGB image with R,B aligned to G.
    Raises RuntimeError on problems.
    """
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise RuntimeError("Image must be RGB (3 channels).")
    if astroalign is None:
        raise RuntimeError("astroalign is not available.")

    worker = RGBAlignWorker(img, model, sep_sigma=sep_sigma)

    try:
        R = np.ascontiguousarray(img[..., 0].astype(np.float32, copy=False))
        G = np.ascontiguousarray(img[..., 1].astype(np.float32, copy=False))
        B = np.ascontiguousarray(img[..., 2].astype(np.float32, copy=False))

        def _estimate_and_warp(src, ref):
            # NOTE: _estimate_transform now returns 3 values
            kind, X, _pairs = worker._estimate_transform(src, ref, model)
            return worker._warp_channel(src, kind, X, ref.shape)

        R_aligned = _estimate_and_warp(R, G)
        B_aligned = _estimate_and_warp(B, G)

        out = np.stack([R_aligned, G, B_aligned], axis=2)
        if img.dtype != out.dtype:
            out = out.astype(img.dtype, copy=False)
        return out
    except Exception as e:
        raise RuntimeError(str(e))

def _align_rgb_off_gui(img: np.ndarray, model: str, sep_sigma: float, status=None) -> np.ndarray:
    """align_rgb_array on a worker thread, waited for with a local event loop.

    Shortcut and Function Bundle callers expect the result on return, but on the
    GUI thread SEP froze the window for the whole detection. Without a running
    QApplication (or off the GUI thread) this is plain align_rgb_array."""
    from PyQt6.QtCore import QCoreApplication, QEventLoop
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is not app.thread():
        return align_rgb_array(img, model=model, sep_sigma=sep_sigma)
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise RuntimeError("Image must be RGB (3 channels).")
    worker = RGBAlignWorker(img, model, sep_sigma=sep_sigma)
    box = {}
    loop = QEventLoop()
    worker.done.connect(lambda out: box.__setitem__("out", out))
    worker.failed.connect(lambda m: box.__setitem__("err", m))
    worker.cancelled.connect(lambda: box.__setitem__("err", "Cancelled."))
    if callable(status):
        worker.progress.connect(lambda _pct, msg: status(msg))
    worker.finished.connect(loop.quit)
    start_owned_thread(worker)
    loop.exec()
    if "out" in box:
        return box["out"]
    raise RuntimeError(box.get("err") or "RGB Align did not complete.")


def run_rgb_align_headless(main_window, document, preset: dict | None = None):
    if document is None:
        QMessageBox.warning(main_window, "RGB Align", "No active document.")
        return

    img = np.asarray(document.image)
    p = dict(preset or {})
    model = p.get("model", "edge").lower()
    sep_sigma = float(p.get("sep_sigma", 3.0))
    create_new = bool(p.get("new_doc", False))

    sb = getattr(main_window, "statusBar", None)
    if callable(sb):
        sb().showMessage(f"RGB Align ({model})…", 3000)

    try:
        out = _align_rgb_off_gui(
            img, model=model if model != "edge" else "edge-sep", sep_sigma=sep_sigma,
            status=(lambda m: sb().showMessage(f"RGB Align: {m}")) if callable(sb) else None,
        )
    except Exception as e:
        QMessageBox.critical(main_window, "RGB Align (headless)", str(e))
        return

    if create_new:
        dm = getattr(main_window, "docman", None)
        if dm is not None:
            dm.open_array(out, {"display_name": "RGB Aligned"}, title="RGB Aligned")
        else:
            # fallback to replace if we can't create new
            try:
                document.apply_edit(out, {"step_name": "RGB Align"})
            except Exception:
                document.image = out
    else:
        # in-place
        if hasattr(document, "apply_edit"):
            document.apply_edit(out, {"step_name": "RGB Align"}, step_name="RGB Align")
        else:
            document.image = out

    if callable(sb):
        sb().showMessage("RGB Align done.", 3000)


class RGBAlignManualDialog(QDialog):
    """
    Manual channel translation for ADC / dispersion.
    Adjust R and B offsets relative to G with 1px steps and live preview.
    """
    def __init__(self, parent=None, image: np.ndarray | None = None, planet_roi: bool = False):
        super().__init__(parent)
        self.setWindowTitle("RGB Align — Manual (Planetary)")
        self.setModal(True)

        self.image = np.asarray(image) if image is not None else None
        self._timer = None
        self._result = None  # (r_dx, r_dy, b_dx, b_dy)

        # offsets
        self.r_dx = 0
        self.r_dy = 0
        self.b_dx = 0
        self.b_dy = 0

        lay = QVBoxLayout(self)

        self.lbl = QLabel("Nudge Red and Blue so they line up with Green (1 px steps).")
        lay.addWidget(self.lbl)

        # preview label
        self.preview = QLabel()
        self.preview.setMinimumSize(420, 320)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#111; border:1px solid #333;")
        lay.addWidget(self.preview)

        # controls grid
        row = QHBoxLayout()
        lay.addLayout(row)

        # controls
        row = QHBoxLayout()
        lay.addLayout(row)

        self.red_pad = OffsetPad("Red offset (relative to Green):")
        self.blue_pad = OffsetPad("Blue offset (relative to Green):")

        self.red_pad.changed.connect(self._on_change)
        self.blue_pad.changed.connect(self._on_change)

        row.addWidget(self.red_pad)
        row.addWidget(self.blue_pad)


        # buttons
        btns = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_apply = QPushButton("Apply")
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.btn_reset.clicked.connect(self._reset)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_close.clicked.connect(self.reject)

        # build a reasonable ROI for preview (center crop)
        self._roi = None
        if self.image is not None and self.image.ndim == 3 and self.image.shape[2] >= 3:
            H, W = self.image.shape[:2]

            if planet_roi:
                # Use green channel to find the planet (most stable)
                c = _planet_centroid(self.image[..., 1])
                if c is not None:
                    cx, cy, area = c
                    # Estimate radius from area, then choose a padded ROI
                    r = max(32.0, np.sqrt(area / np.pi))
                    s = int(np.clip(r * 3.2, 160, 900))  # ROI size = ~3.2x radius
                    cx_i, cy_i = int(round(cx)), int(round(cy))
                else:
                    cx_i, cy_i = W // 2, H // 2
                    s = int(np.clip(min(H, W) * 0.45, 160, 900))
            else:
                cx_i, cy_i = W // 2, H // 2
                s = int(np.clip(min(H, W) * 0.45, 160, 900))

            x0 = max(0, cx_i - s // 2)
            y0 = max(0, cy_i - s // 2)
            x1 = min(W, x0 + s)
            y1 = min(H, y0 + s)
            self._roi = (x0, y0, x1, y1)


        # debounce timer
        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_preview)
        if self._roi is not None:
            x0,y0,x1,y1 = self._roi
            self.lbl.setText(self.lbl.text() + f"\nPreview ROI: x={x0}:{x1}, y={y0}:{y1}")
        self._render_preview()

    def result_offsets(self):
        return self._result

    def _on_change(self, *_):
        self._timer.start(60)

    def _reset(self):
        self.red_pad.reset()
        self.blue_pad.reset()

    def _apply(self):
        rdx, rdy = self.red_pad.offsets()
        bdx, bdy = self.blue_pad.offsets()
        self._result = (int(rdx), int(rdy), int(bdx), int(bdy))
        self.accept()

    def _render_preview(self):
        if self.image is None or self.image.ndim != 3 or self.image.shape[2] < 3:
            self.preview.setText("No RGB image.")
            return

        x0, y0, x1, y1 = self._roi if self._roi is not None else (0, 0, self.image.shape[1], self.image.shape[0])
        crop = self.image[y0:y1, x0:x1, :3]

        R = crop[..., 0].astype(np.float32, copy=False)
        G = crop[..., 1].astype(np.float32, copy=False)
        B = crop[..., 2].astype(np.float32, copy=False)

        rdx, rdy = self.red_pad.offsets()
        bdx, bdy = self.blue_pad.offsets()

        R2 = _translate_channel(R, rdx, rdy)
        B2 = _translate_channel(B, bdx, bdy)

        out = np.stack([R2, G, B2], axis=2)

        # display: robust scale to uint8
        out8 = self._to_u8_preview(out)
        qimg = QImage(out8.data, out8.shape[1], out8.shape[0], out8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    @staticmethod
    def _to_u8_preview(rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32, copy=False)
        # percentile stretch per-channel, simple + robust
        out = np.empty_like(x, dtype=np.uint8)
        for c in range(3):
            ch = x[..., c]
            p1 = float(np.percentile(ch, 1.0))
            p99 = float(np.percentile(ch, 99.5))
            if p99 <= p1:
                out[..., c] = 0
            else:
                y = (ch - p1) * (255.0 / (p99 - p1))
                out[..., c] = np.clip(y, 0, 255).astype(np.uint8)
        return out

class OffsetPad(QWidget):
    changed = pyqtSignal(int, int)  # dx, dy

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._dx = 0
        self._dy = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        hdr = QLabel(title)
        hdr.setStyleSheet("font-weight:600;")
        outer.addWidget(hdr)

        self.lbl = QLabel("dx=0  dy=0")
        self.lbl.setStyleSheet("color:#bbb;")
        outer.addWidget(self.lbl)

        grid = QGridLayout()
        grid.setSpacing(2)

        self.btn_up = QPushButton("↑")
        self.btn_dn = QPushButton("↓")
        self.btn_lt = QPushButton("←")
        self.btn_rt = QPushButton("→")
        self.btn_mid = QPushButton("⟲")  # reset

        for b in (self.btn_up, self.btn_dn, self.btn_lt, self.btn_rt, self.btn_mid):
            b.setFixedSize(34, 34)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        grid.addWidget(self.btn_up, 0, 1)
        grid.addWidget(self.btn_lt, 1, 0)
        grid.addWidget(self.btn_mid, 1, 1)
        grid.addWidget(self.btn_rt, 1, 2)
        grid.addWidget(self.btn_dn, 2, 1)

        outer.addLayout(grid)

        # connections
        self.btn_up.clicked.connect(lambda: self.nudge(0, -1))
        self.btn_dn.clicked.connect(lambda: self.nudge(0, +1))
        self.btn_lt.clicked.connect(lambda: self.nudge(-1, 0))
        self.btn_rt.clicked.connect(lambda: self.nudge(+1, 0))
        self.btn_mid.clicked.connect(self.reset)

    def set_offsets(self, dx: int, dy: int):
        self._dx = int(dx)
        self._dy = int(dy)
        self._update_label()
        self.changed.emit(self._dx, self._dy)

    def offsets(self):
        return (self._dx, self._dy)

    def reset(self):
        self.set_offsets(0, 0)

    def nudge(self, ddx: int, ddy: int, step: int = 1):
        self._dx += int(ddx) * int(step)
        self._dy += int(ddy) * int(step)
        self._update_label()
        self.changed.emit(self._dx, self._dy)

    def _update_label(self):
        self.lbl.setText(f"dx={self._dx:+d}  dy={self._dy:+d}")
