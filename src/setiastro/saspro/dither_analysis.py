# ============================================================
#  ____       _   _ _    _       _
# / ___|  ___| |_(_) |  / \  ___| |_ _ __ ___
# \___ \ / _ \ __| | | / _ \/ __| __| '__/ _ \
#  ___) |  __/ |_| | |/ ___ \__ \ |_| | | (_) |
# |____/ \___|\__|_|_/_/   \_\___/\__|_|  \___/
#
#  SASpro – Dither Analysis Tool
#  Franklin Marek  |  www.setiastro.com
# ============================================================
"""
DitherAnalysisWindow
====================
Analyzes dither offsets by tracking where the IMAGE CENTER moves after each
frame's alignment transform is applied — not the raw tx/ty, which includes
any meridian-flip rotation and is meaningless as a dither offset.

Method:
  center_pt = (W/2, H/2)  (from REF_SHAPE or reference array)
  For each frame i:
    p_i = M_i @ [cx, cy, 1]^T   (2×3 affine applied to center)
  dither_offset_i = p_i - p_0   (relative to reference frame's center)

This correctly handles meridian flips: the rotated center stays near the
origin of the dither scatter; only the true pointing jitter shows up.

Meridian flips are detected by checking the sign of det(M[:2,:2]).
Flipped frames are flagged in the per-frame table and highlighted on the plot.

THREE ENTRY POINTS:
  1. Stand-alone  – user picks files, StarRegistrationThread runs, shape
                    read from thread.reference_image_2d.shape.
  2. Bolt-on      – load_transforms(transforms, filenames, ref_shape)
  3. .sasd file   – load_sasd(path)  — REF_SHAPE parsed from header.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QSizePolicy, QSplitter,
    QTextEdit, QVBoxLayout, QWidget, QApplication
)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False

# ── palette ──────────────────────────────────────────────────────────────
_BG      = "#1a1a2e"
_PANEL   = "#16213e"
_ACCENT  = "#e94560"
_ACCENT2 = "#0f3460"
_FG      = "#eaeaea"
_DIM     = "#888888"
_GREEN   = "#4caf50"
_YELLOW  = "#ffc107"
_ORANGE  = "#ff9800"


# ═══════════════════════════════════════════════════════════════════════════
#  SASD parser
# ═══════════════════════════════════════════════════════════════════════════
def parse_sasd(path: str) -> tuple[list[str], list[Optional[np.ndarray]], tuple[int, int]]:
    """
    Parse an alignment_transforms.sasd file.
    Returns (basenames, transforms, ref_shape).
      basenames  – list[str] os.path.basename of each FILE: entry
      transforms – list of 2×3 float64 ndarray or None per frame
      ref_shape  – (H, W) from REF_SHAPE: header line, or (0, 0) if absent
    """
    basenames:  list[str] = []
    transforms: list[Optional[np.ndarray]] = []
    ref_shape:  tuple[int, int] = (0, 0)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cur_file:     Optional[str] = None
    cur_kind:     Optional[str] = None
    matrix_lines: list[str] = []
    in_matrix = False

    def flush():
        nonlocal cur_file, cur_kind, matrix_lines, in_matrix
        if cur_file is None:
            return
        M = _parse_matrix_rows(cur_kind or "affine", matrix_lines)
        basenames.append(os.path.basename(cur_file))
        transforms.append(M)
        cur_file = None
        cur_kind = None
        matrix_lines = []
        in_matrix = False

    for line in lines:
        s = line.strip()
        if s.startswith("REF_SHAPE:"):
            try:
                parts = s[10:].strip().split(",")
                ref_shape = (int(parts[0].strip()), int(parts[1].strip()))
            except Exception:
                pass
        elif s.startswith("FILE:"):
            flush()
            cur_file  = s[5:].strip()
            in_matrix = False
        elif s.startswith("KIND:"):
            cur_kind = s[5:].strip().lower()
        elif s.startswith("MATRIX:"):
            in_matrix = True
            rest = s[7:].strip()
            if rest and rest != "UNSUPPORTED":
                matrix_lines.append(rest)
        elif in_matrix and s and not s.startswith("#"):
            if s == "UNSUPPORTED":
                in_matrix = False
            else:
                matrix_lines.append(s)

    flush()
    return basenames, transforms, ref_shape


def _parse_matrix_rows(kind: str, rows: list[str]) -> Optional[np.ndarray]:
    if not rows:
        return None
    try:
        vals = []
        for r in rows:
            vals.extend(float(v) for v in r.replace(",", " ").split())
        if kind in ("affine", "similarity") and len(vals) >= 6:
            return np.array(vals[:6], dtype=np.float64).reshape(2, 3)
        elif kind == "homography" and len(vals) >= 9:
            H = np.array(vals[:9], dtype=np.float64).reshape(3, 3)
            return np.array([[H[0, 0], H[0, 1], H[0, 2]],
                             [H[1, 0], H[1, 1], H[1, 2]]], dtype=np.float64)
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Center-transform helpers
# ═══════════════════════════════════════════════════════════════════════════
def _transform_center(M: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """Apply 2×3 affine M to point (cx, cy). Returns (x', y')."""
    arr = np.asarray(M, dtype=np.float64).reshape(2, 3)
    pt  = np.array([cx, cy, 1.0])
    r   = arr @ pt
    return float(r[0]), float(r[1])


def _rotation_angle_deg(M: np.ndarray) -> float:
    """
    Extract rotation angle (degrees) from the 2×2 part of a 2×3 affine.
    For a similarity M = s*[[cos θ, -sin θ],[sin θ, cos θ]] + t,
    atan2(M[1,0], M[0,0]) gives θ directly.
    Returns value in (-180, 180].
    """
    arr = np.asarray(M, dtype=np.float64).reshape(2, 3)
    return math.degrees(math.atan2(float(arr[1, 0]), float(arr[0, 0])))


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed difference between two angles in degrees, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _detect_flips(transforms: list[Optional[np.ndarray]]) -> np.ndarray:
    """
    Detect meridian flips by looking for rotation-angle jumps of ~180°.

    Strategy:
      1. Extract rotation angle from every valid transform.
      2. Use the median angle of all valid frames as the reference orientation.
      3. Any frame whose angle differs from the reference median by > 90° is a flip.

    This is robust to the stacking pipeline's _project_to_similarity which forces
    det > 0 always (so det-sign detection never works).

    Returns a bool array of length len(transforms), True = flipped.
    """
    angles = []
    for M in transforms:
        if M is None:
            angles.append(float("nan"))
        else:
            angles.append(_rotation_angle_deg(M))

    valid_angles = [a for a in angles if not math.isnan(a)]
    if not valid_angles:
        return np.zeros(len(transforms), dtype=bool)

    # Circular median: use the angle of the circular mean of the majority cluster.
    # Simple approach: sort angles, find the largest gap, the majority cluster is
    # on the other side. For astrophotography sessions this is reliable.
    sorted_a = sorted(valid_angles)
    n_v = len(sorted_a)

    # Find reference angle as circular mean of the most common orientation cluster
    # (handles case where flips are the minority, which is usual)
    # Use the frame-0 angle as the seed; anything within 90° of it is "normal".
    ref_angle = angles[next((i for i, a in enumerate(angles) if not math.isnan(a)), 0)]

    flipped = np.zeros(len(transforms), dtype=bool)
    for i, a in enumerate(angles):
        if math.isnan(a):
            continue
        diff = abs(_angular_diff(a, ref_angle))
        if diff > 90.0:
            flipped[i] = True

    return flipped


def _center_offsets(
    transforms: list[Optional[np.ndarray]],
    ref_shape:  tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each transform compute where the image center (W/2, H/2) maps to,
    then subtract frame-0's mapped center so offsets are relative.

    Returns (dx, dy, flipped_mask) — all length == len(transforms).
    NaN entries for failed (None) frames.
    ref_shape is (H, W).
    """
    H, W = ref_shape
    cx0, cy0 = W / 2.0, H / 2.0

    # Detect flips across the whole population first
    flipped = _detect_flips(transforms)

    # First pass: get raw mapped centers
    raw_x: list[float] = []
    raw_y: list[float] = []
    for M in transforms:
        if M is None:
            raw_x.append(float("nan"))
            raw_y.append(float("nan"))
        else:
            x, y = _transform_center(M, cx0, cy0)
            raw_x.append(x)
            raw_y.append(y)

    # Origin = first valid frame's mapped center
    origin_x, origin_y = cx0, cy0
    for x, y in zip(raw_x, raw_y):
        if not (math.isnan(x) or math.isnan(y)):
            origin_x, origin_y = x, y
            break

    dx = np.array([x - origin_x for x in raw_x])
    dy = np.array([y - origin_y for y in raw_y])

    return dx, dy, flipped


# ═══════════════════════════════════════════════════════════════════════════
#  Stats
# ═══════════════════════════════════════════════════════════════════════════
def _compute_stats(
    dx: np.ndarray,
    dy: np.ndarray,
    flipped: np.ndarray,
) -> dict:
    """dx/dy are the already-masked (valid frames only) center-offset arrays."""
    n = len(dx)
    if n == 0:
        return {}

    radii = np.hypot(dx, dy)

    # Steps between consecutive valid frames
    steps = np.hypot(np.diff(dx), np.diff(dy)) if n > 1 else np.array([0.0])

    # Exclude flip-steps from step statistics (they're not dither steps)
    valid_steps = steps[~flipped[1:n]] if len(flipped) >= n else steps
    if len(valid_steps) == 0:
        valid_steps = steps

    coverage = 0.0
    if n >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.stack([dx, dy], axis=1))
            coverage = hull.volume
        except Exception:
            pass

    if n > 1:
        sa_rad    = np.arctan2(np.diff(dy), np.diff(dx))
        circ_mean = math.degrees(
            math.atan2(np.sin(sa_rad).mean(), np.cos(sa_rad).mean())
        ) % 360
        step_angles = np.degrees(sa_rad) % 360
    else:
        step_angles = np.array([0.0])
        circ_mean   = 0.0

    cs = _cluster_stats(dx, dy, valid_steps)    

    return dict(
        n=n, dx=dx, dy=dy, radii=radii, flipped=flipped,
        steps=steps, valid_steps=valid_steps, step_angles=step_angles,
        rms_offset  = float(np.sqrt((dx**2 + dy**2).mean())),
        mean_radius = float(radii.mean()),
        max_radius  = float(radii.max()),
        mean_step   = float(valid_steps.mean()),
        max_step    = float(valid_steps.max()),
        std_dx      = float(dx.std()),
        std_dy      = float(dy.std()),
        coverage_px = coverage,
        pref_dir    = circ_mean,
        span_x      = float(dx.max() - dx.min()),
        span_y      = float(dy.max() - dy.min()),
        n_flipped   = int(flipped.sum()),
        **cs,
    )

def _cluster_stats(
    dx: np.ndarray,
    dy: np.ndarray,
    steps: np.ndarray,
) -> dict:
    """
    Detect dither clustering — where multiple consecutive frames share
    the same pointing position before the mount dithers again.

    A 'cluster' is a run of consecutive frames where the step to the
    next frame is below step_threshold. We use the valley between the
    two populations in the step distribution as the threshold — if the
    distribution is unimodal we use 20% of the mean step.

    Returns:
        n_clusters       — number of distinct pointing positions
        mean_cluster_size — mean frames per cluster
        max_cluster_size  — worst case (longest run at one position)
        clustered_frac   — fraction of frames that share a position
                           with at least one neighbour (0 = perfect,
                           1 = no dithering at all)
        step_threshold   — threshold used (px)
        is_clustered     — bool: meaningful clustering detected
    """
    if len(steps) == 0:
        return dict(n_clusters=1, mean_cluster_size=1, max_cluster_size=1,
                    clustered_frac=0.0, step_threshold=0.0, is_clustered=False)

    n_frames = len(dx)

    # Find threshold using the valley between small (intra-cluster) and
    # large (inter-cluster) steps. Sort steps and look for the biggest
    # relative jump — that gap is the natural cluster boundary.
    sorted_steps = np.sort(steps)

    # Find the largest gap in the sorted step distribution
    # This is robust to bimodal distributions regardless of scale
    if len(sorted_steps) > 2:
        gaps = np.diff(sorted_steps)
        # Only look for gaps in the lower 80% of the range to avoid
        # outliers skewing the threshold
        upper_idx = int(len(sorted_steps) * 0.85)
        gaps_clipped = gaps[:upper_idx]
        if len(gaps_clipped) > 0 and gaps_clipped.max() > 0:
            gap_idx = int(np.argmax(gaps_clipped))
            step_threshold = float((sorted_steps[gap_idx] + sorted_steps[gap_idx + 1]) / 2.0)
        else:
            step_threshold = float(sorted_steps.max() * 0.2)
    else:
        step_threshold = float(sorted_steps[0] * 1.5) if len(sorted_steps) else 1.0

    # Sanity clamp — threshold should be between 0.5px and 50% of max step
    step_threshold = max(0.5, min(step_threshold, float(steps.max()) * 0.5))

    # Walk frames and count cluster sizes
    cluster_sizes = []
    current_run = 1
    for s in steps:
        if s <= step_threshold:
            current_run += 1
        else:
            cluster_sizes.append(current_run)
            current_run = 1
    cluster_sizes.append(current_run)

    cluster_sizes = np.array(cluster_sizes, dtype=int)
    n_clusters        = len(cluster_sizes)
    mean_cluster_size = float(cluster_sizes.mean())
    max_cluster_size  = int(cluster_sizes.max())

    clustered_frames = int(np.sum(cluster_sizes[cluster_sizes > 1]))
    clustered_frac   = clustered_frames / max(1, n_frames)

    is_clustered = (mean_cluster_size > 1.5) and (clustered_frac > 0.2)

    return dict(
        n_clusters        = n_clusters,
        mean_cluster_size = mean_cluster_size,
        max_cluster_size  = max_cluster_size,
        clustered_frac    = clustered_frac,
        step_threshold    = step_threshold,
        is_clustered      = is_clustered,
        cluster_sizes     = cluster_sizes.tolist(),
    )

def _extract_pixscale(hdr) -> Optional[float]:
    """
    Extract pixel scale in arcsec/px from a FITS header or dict.
    Tries in priority order:
      1) CD matrix (most accurate — includes rotation)
      2) CDELT1/2 (simple TAN projection)
      3) PIXSCALE / SCALE keywords (SGP, PI, NINA etc.)
      4) Compute from FOCALLEN + XPIXSZ (instrumental)
    Returns arcsec/px or None if not determinable.
    """
    if hdr is None:
        return None

    def _get(key, default=None):
        try:
            v = hdr.get(key, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    # 1) CD matrix
    cd11 = _get("CD1_1")
    cd12 = _get("CD1_2", 0.0)
    cd21 = _get("CD2_1", 0.0)
    cd22 = _get("CD2_2")
    if cd11 is not None and cd22 is not None:
        sx = math.sqrt(cd11**2 + cd21**2) * 3600.0
        sy = math.sqrt(cd12**2 + cd22**2) * 3600.0
        return float((sx + sy) / 2.0)

    # 2) CDELT
    cdelt1 = _get("CDELT1")
    cdelt2 = _get("CDELT2")
    if cdelt1 is not None:
        val = abs(cdelt1)
        if val < 1.0:  # degrees
            val *= 3600.0
        return float(val)
    if cdelt2 is not None:
        val = abs(cdelt2)
        if val < 1.0:
            val *= 3600.0
        return float(val)

    # 3) Direct pixscale keywords
    for key in ("PIXSCALE", "SCALE", "PLATESCL", "SECPIX", "SECPIX1"):
        v = _get(key)
        if v is not None and 0.01 < v < 100.0:
            return float(v)

    # 4) Instrumental: focal length + pixel size
    focallen = _get("FOCALLEN")  # mm
    xpixsz   = _get("XPIXSZ")   # μm (effective, includes binning)
    ypixsz   = _get("YPIXSZ")
    if focallen and focallen > 0 and xpixsz and xpixsz > 0:
        px = (xpixsz + (ypixsz or xpixsz)) / 2.0
        return float(206.265 * px / focallen)

    return None


def _find_source_file(directory: str, basename: str) -> Optional[str]:
    """
    Try to find the original source file near the .sasd.
    Searches the sasd directory and one level up for files matching the stem.
    """
    stem = os.path.splitext(basename)[0]
    exts = (".fit", ".fits", ".fts", ".xisf", ".tif", ".tiff")
    search_dirs = [directory, os.path.dirname(directory)]
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for ext in exts:
            candidate = os.path.join(d, stem + ext)
            if os.path.exists(candidate):
                return candidate
    return None

def _dir_label(deg: float) -> str:
    names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    return names[int((deg + 22.5) / 45) % 8]

def _read_sasd_ref_path(path: str) -> Optional[str]:
    """Extract the REF_PATH line from a .sasd file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("REF_PATH:"):
                    return s[9:].strip()
    except Exception:
        pass
    return None

def _hash_sasd(path: str) -> str:
    """SHA1 of first 4KB of the .sasd file — fast, stable identifier."""
    import hashlib
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(4096))
    except Exception:
        return ""
    return h.hexdigest()

def _save_pixscale_for_sasd(settings: QSettings, sasd_path: str, pixscale: float):
    h = _hash_sasd(sasd_path)
    if not h:
        return
    settings.setValue(f"DitherAnalysis/pixscale/{h}", float(pixscale))


def _load_pixscale_for_sasd(settings: QSettings, sasd_path: str) -> Optional[float]:
    h = _hash_sasd(sasd_path)
    if not h:
        return None
    v = settings.value(f"DitherAnalysis/pixscale/{h}", None)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None
    
# ═══════════════════════════════════════════════════════════════════════════
#  Plot widget
# ═══════════════════════════════════════════════════════════════════════════
class _DitherPlot(FigureCanvas if _MPL else QWidget):
    def __init__(self, parent=None):
        if not _MPL:
            super().__init__(parent)
            return
        fig = Figure(figsize=(10, 8), facecolor=_BG)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fig = fig
        self._last_stats: dict = {}
        self._last_unit_scale: float = 1.0
        self._last_unit_label: str = "px"

        # Debounce timer — only rerender after 150ms of no resize events
        from PyQt6.QtCore import QTimer
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._deferred_draw)

    def render(self, stats: dict, unit_scale: float = 1.0, unit_label: str = "px"):
        if not _MPL or not stats:
            return
        # Store for deferred redraws triggered by resize
        self._last_stats      = stats
        self._last_unit_scale = unit_scale
        self._last_unit_label = unit_label
        self._do_render(stats, unit_scale, unit_label)

    def _deferred_draw(self):
        if self._last_stats:
            self._do_render(self._last_stats, self._last_unit_scale, self._last_unit_label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if _MPL and self._last_stats:
            self._resize_timer.start()  # restart the 150ms countdown

    def _do_render(self, stats: dict, unit_scale: float = 1.0, unit_label: str = "px"):
        """Actual rendering — called directly on data change, debounced on resize."""
        self.fig.clear()

        sc = float(unit_scale)
        dx      = stats["dx"] * sc
        dy      = stats["dy"] * sc
        steps   = stats["valid_steps"] * sc
        n       = stats["n"]
        flipped = stats["flipped"]
        ul      = unit_label

        gs = self.fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
        ax_scatter = self.fig.add_subplot(gs[0, 0])
        ax_path    = self.fig.add_subplot(gs[0, 1])
        ax_hist    = self.fig.add_subplot(gs[1, 0])
        ax_rose    = self.fig.add_subplot(gs[1, 1], polar=True)

        for ax in (ax_scatter, ax_path, ax_hist, ax_rose):
            ax.set_facecolor(_PANEL)
            ax.tick_params(colors=_FG, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(_ACCENT2)

        lbl = dict(color=_FG, fontsize=9)
        norm_idx  = np.where(~flipped)[0]
        flip_idx  = np.where(flipped)[0]
        frame_ids = np.arange(n)

        # ── 1. Scatter ───────────────────────────────────────────────
        if len(norm_idx):
            sc_plot = ax_scatter.scatter(
                dx[norm_idx], dy[norm_idx],
                c=frame_ids[norm_idx], cmap="plasma",
                s=30, zorder=3, edgecolors="none", alpha=0.85,
                vmin=0, vmax=n - 1,
            )
            cb = self.fig.colorbar(sc_plot, ax=ax_scatter, pad=0.02)
            cb.set_label("Frame #", color=_FG, fontsize=8)
            cb.ax.yaxis.set_tick_params(color=_FG, labelsize=7)

        if len(flip_idx):
            ax_scatter.scatter(
                dx[flip_idx], dy[flip_idx],
                c=frame_ids[flip_idx], cmap="plasma",
                s=80, zorder=4, edgecolors="none", alpha=0.85,
                vmin=0, vmax=n - 1, marker="o",
            )
            ax_scatter.scatter(
                dx[flip_idx], dy[flip_idx],
                marker="x", s=80, c="white", zorder=5,
                linewidths=1.2, alpha=0.9, label="Flipped side",
            )
            ax_scatter.legend(fontsize=7, facecolor=_PANEL,
                              edgecolor=_ACCENT2, labelcolor=_FG)

        ax_scatter.scatter([0], [0], marker="+", s=120, c=_ACCENT,
                           zorder=5, linewidths=1.5)

        if n <= 80:
            for i, (x, y) in enumerate(zip(dx, dy)):
                ax_scatter.annotate(
                    str(i), (x, y), fontsize=5, color=_DIM,
                    ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points",
                )

        ax_scatter.set_title("Dither Scatter — image centre",
                             color=_ACCENT, fontsize=10)
        ax_scatter.set_xlabel(f"ΔX centre ({ul})", **lbl)
        ax_scatter.set_ylabel(f"ΔY centre ({ul})", **lbl)
        ax_scatter.axhline(0, color=_DIM, lw=0.5, ls="--")
        ax_scatter.axvline(0, color=_DIM, lw=0.5, ls="--")

        # ── 2. Path ──────────────────────────────────────────────────
        for i in range(n - 1):
            col = _ORANGE if (flipped[i] or flipped[i + 1]) else _ACCENT2
            ax_path.plot([dx[i], dx[i + 1]], [dy[i], dy[i + 1]],
                         color=col, lw=0.8, zorder=2)

        if len(norm_idx):
            ax_path.scatter(dx[norm_idx], dy[norm_idx],
                            c=frame_ids[norm_idx], cmap="plasma",
                            s=20, zorder=3, edgecolors="none", alpha=0.85,
                            vmin=0, vmax=n - 1)
        if len(flip_idx):
            ax_path.scatter(dx[flip_idx], dy[flip_idx],
                            c=frame_ids[flip_idx], cmap="plasma",
                            s=60, zorder=4, edgecolors="none", alpha=0.85,
                            vmin=0, vmax=n - 1, marker="o")
            ax_path.scatter(dx[flip_idx], dy[flip_idx],
                            marker="x", s=60, c="white", zorder=5,
                            linewidths=1.2, alpha=0.9)

        ax_path.scatter([dx[0]], [dy[0]], marker="*", s=100, c=_GREEN,
                        zorder=6, label="Frame 0")
        ax_path.set_title("Dither Path", color=_ACCENT, fontsize=10)
        ax_path.set_xlabel(f"ΔX centre ({ul})", **lbl)
        ax_path.set_ylabel(f"ΔY centre ({ul})", **lbl)
        ax_path.axhline(0, color=_DIM, lw=0.5, ls="--")
        ax_path.axvline(0, color=_DIM, lw=0.5, ls="--")
        ax_path.legend(fontsize=7, facecolor=_PANEL,
                       edgecolor=_ACCENT2, labelcolor=_FG, loc="upper right")

        # ── 3. Step histogram ────────────────────────────────────────
        bins = max(8, min(30, len(steps) // 2 + 1))
        ax_hist.hist(steps, bins=bins, color=_ACCENT, edgecolor=_BG, alpha=0.85)
        if len(steps):
            ax_hist.axvline(steps.mean(), color=_YELLOW, lw=1.2, ls="--",
                            label=f"Mean {steps.mean():.1f} {ul}")
        ax_hist.set_title("Dither Step Distribution\n(flip steps excluded)",
                          color=_ACCENT, fontsize=10)
        ax_hist.set_xlabel(f"Step ({ul})", **lbl)
        ax_hist.set_ylabel("Count", **lbl)
        ax_hist.legend(fontsize=7, facecolor=_PANEL,
                       edgecolor=_ACCENT2, labelcolor=_FG)

        # ── 4. Rose plot ─────────────────────────────────────────────
        sa      = stats["step_angles"]
        flip_tr = flipped[1:n] if len(flipped) >= n else np.zeros(len(sa), bool)
        sa_ok   = sa[~flip_tr[:len(sa)]] if len(flip_tr) else sa

        bins16 = np.linspace(0, 360, 17)
        counts, _ = np.histogram(sa_ok if len(sa_ok) else sa, bins=bins16)
        theta  = np.radians(bins16[:-1] + 11.25)
        width  = np.radians(360 / 16)
        ax_rose.set_facecolor(_PANEL)
        ax_rose.tick_params(colors=_FG, labelsize=7)
        ax_rose.bar(theta, counts, width=width,
                    color=_ACCENT, edgecolor=_BG, alpha=0.85)
        if counts.max() > 0:
            pref_rad = math.radians(stats["pref_dir"])
            ax_rose.annotate(
                "", xy=(pref_rad, counts.max() * 1.15), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=_YELLOW, lw=1.5),
            )
        ax_rose.set_title("Step Directions\n(flip steps excluded)",
                          color=_ACCENT, fontsize=10, pad=12)
        ax_rose.set_theta_zero_location("E")
        ax_rose.set_theta_direction(1)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tight_layout.*")
            self.fig.tight_layout()
        self.draw()

# ═══════════════════════════════════════════════════════════════════════════
#  Main window
# ═══════════════════════════════════════════════════════════════════════════
class DitherAnalysisWindow(QWidget):
    """
    Stand-alone Dither Analysis tool.

    Dither offsets are computed as the displacement of the IMAGE CENTRE after
    each frame's alignment transform, relative to the reference frame.  This
    correctly handles meridian flips, which are flagged separately.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dither Analysis")
        self.setMinimumSize(1100, 720)
        self._settings   = QSettings("SetiAstro", "SASpro")
        self._reg_thread = None
        self._transforms: list[Optional[np.ndarray]] = []
        self._filenames:  list[str] = []
        self._ref_shape:  tuple[int, int] = (0, 0)
        self._stats:      dict = {}
        self._save_dir:   str  = ""
        self._pixscale_arcsec: Optional[float] = None
        self._build_ui()
        self._restore_geometry()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        hdr = QLabel(
            '<span style="color:#e94560;font-size:18px;font-weight:700;">'
            '⬡ Dither Analysis</span>'
        )
        hdr.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # ── LEFT ─────────────────────────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(6)

        top_row = QHBoxLayout()
        self._btn_mode_sasd = QPushButton("📄 Load Previous Transform File (.sasd)…")
        self._btn_mode_sasd.setFixedHeight(28)
        self._btn_mode_sasd.setStyleSheet(
            "QPushButton{background:#0f3460;color:#eaeaea;border-radius:4px;}"
            "QPushButton:hover{background:#e94560;}"
        )
        top_row.addWidget(self._btn_mode_sasd)
        top_row.addStretch()
        lv.addLayout(top_row)

        grp_files = QGroupBox("Light Frames  (first file = reference)")
        grp_files.setStyleSheet(self._grp_style())
        gv = QVBoxLayout(grp_files)

        btn_row = QHBoxLayout()
        self._btn_add    = QPushButton("Add Files…")
        self._btn_remove = QPushButton("Remove")
        self._btn_clear  = QPushButton("Clear")
        for b in (self._btn_add, self._btn_remove, self._btn_clear):
            b.setFixedHeight(26)
            btn_row.addWidget(b)
        gv.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_list.setStyleSheet(
            "QListWidget{background:#16213e;color:#eaeaea;"
            "border:1px solid #0f3460;font-size:11px;}"
            "QListWidget::item:selected{background:#e94560;color:#fff;}"
        )
        self._file_list.setMinimumHeight(140)
        gv.addWidget(self._file_list)

        self._chk_save = QCheckBox("Save aligned images")
        self._chk_save.setStyleSheet("color:#eaeaea;")
        gv.addWidget(self._chk_save)

        out_row = QHBoxLayout()
        self._btn_save_dir = QPushButton("Output folder…")
        self._btn_save_dir.setEnabled(False)
        self._btn_save_dir.setFixedHeight(24)
        self._lbl_save_dir = QLabel("(not set)")
        self._lbl_save_dir.setStyleSheet("color:#888;font-size:10px;")
        out_row.addWidget(self._btn_save_dir)
        out_row.addWidget(self._lbl_save_dir, stretch=1)
        gv.addLayout(out_row)
        lv.addWidget(grp_files)

        self._btn_run = QPushButton("▶  Run Registration && Analyze")
        self._btn_run.setFixedHeight(34)
        self._btn_run.setStyleSheet(
            "QPushButton{background:#e94560;color:#fff;font-weight:700;"
            "border-radius:5px;font-size:13px;}"
            "QPushButton:hover{background:#ff6b81;}"
            "QPushButton:disabled{background:#444;color:#888;}"
        )
        lv.addWidget(self._btn_run)

        self._btn_abort = QPushButton("■  Abort")
        self._btn_abort.setFixedHeight(28)
        self._btn_abort.setEnabled(False)
        self._btn_abort.setStyleSheet(
            "QPushButton{background:#333;color:#e94560;border:1px solid #e94560;"
            "border-radius:4px;}"
            "QPushButton:hover{background:#e94560;color:#fff;}"
        )
        lv.addWidget(self._btn_abort)

        self._lbl_status = QLabel("Ready.")
        self._lbl_status.setStyleSheet("color:#888;font-size:10px;")
        self._lbl_status.setWordWrap(True)
        lv.addWidget(self._lbl_status)

        grp_stats = QGroupBox("Statistics")
        grp_stats.setStyleSheet(self._grp_style())
        sv = QVBoxLayout(grp_stats)
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setStyleSheet(
            "QTextEdit{background:#0d0d1a;color:#eaeaea;"
            "border:1px solid #0f3460;font-family:Consolas,monospace;font-size:11px;}"
        )
        self._stats_text.setMinimumHeight(180)
        sv.addWidget(self._stats_text)
        lv.addWidget(grp_stats, stretch=1)

        footer = QLabel(
            '<a href="https://www.setiastro.com" style="color:#e94560;">'
            'Franklin Marek  ·  www.setiastro.com</a>'
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setOpenExternalLinks(True)
        footer.setStyleSheet("font-size:10px;color:#888;padding-top:2px;")
        lv.addWidget(footer)

        splitter.addWidget(left)

        # ── RIGHT ────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        # ── toggle px / arcsec ───────────────────────────────────────
        toggle_row = QHBoxLayout()
        self._btn_px_arcsec = QPushButton("Show in arcsec")
        self._btn_px_arcsec.setCheckable(True)
        self._btn_px_arcsec.setChecked(False)
        self._btn_px_arcsec.setFixedHeight(24)
        self._btn_px_arcsec.setStyleSheet(
            "QPushButton{background:#0f3460;color:#eaeaea;border-radius:4px;font-size:11px;}"
            "QPushButton:checked{background:#e94560;color:#fff;}"
            "QPushButton:hover{background:#e94560;color:#fff;}"
        )
        self._btn_px_arcsec.toggled.connect(self._on_unit_toggle)
        toggle_row.addStretch()
        toggle_row.addWidget(self._btn_px_arcsec)
        rv.addLayout(toggle_row)
        if _MPL:
            self._plot = _DitherPlot(right)
            rv.addWidget(self._plot, stretch=1)
        else:
            lbl = QLabel(
                "matplotlib not found.\nInstall:  pip install matplotlib\n"
                "Statistics are still shown in the left panel."
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#888;font-size:13px;")
            rv.addWidget(lbl, stretch=1)
            self._plot = None

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 770])

        # connections
        self._btn_mode_sasd.clicked.connect(self._load_sasd_dialog)
        self._btn_add.clicked.connect(self._add_files)
        self._btn_remove.clicked.connect(self._remove_files)
        self._btn_clear.clicked.connect(self._clear_files)
        self._chk_save.toggled.connect(self._btn_save_dir.setEnabled)
        self._btn_save_dir.clicked.connect(self._pick_save_dir)
        self._btn_run.clicked.connect(self._run)
        self._btn_abort.clicked.connect(self._abort)

    @staticmethod
    def _grp_style() -> str:
        return (
            "QGroupBox{color:#eaeaea;font-weight:600;font-size:11px;"
            "border:1px solid #0f3460;border-radius:5px;margin-top:8px;padding-top:6px;}"
            "QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top left;"
            "padding:0 4px;color:#e94560;}"
        )

    def _on_unit_toggle(self, checked: bool):
        self._btn_px_arcsec.setText("Show in pixels" if checked else "Show in arcsec")
        # re-render plot and stats with current data
        if self._stats:
            dx_full, dy_full, flipped_full = _center_offsets(
                self._transforms, self._ref_shape
            )
            self._render_stats(self._stats, dx_full, dy_full, flipped_full)
            if self._plot is not None:
                self._plot.render(self._stats, self._unit_scale, self._unit_label)

    @property
    def _unit_scale(self) -> float:
        """Multiplier to apply to pixel values for display."""
        ps = getattr(self, "_pixscale_arcsec", None)
        if self._btn_px_arcsec.isChecked() and ps:
            return float(ps)
        return 1.0

    @property
    def _unit_label(self) -> str:
        ps = getattr(self, "_pixscale_arcsec", None)
        if self._btn_px_arcsec.isChecked() and ps:
            return "\""
        return "px"

    # ----------------------------------------------------------------- file mgmt
    def _add_files(self):
        last_dir = self._settings.value("DitherAnalysis/last_dir", "", type=str)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", last_dir,
            "Images (*.fit *.fits *.fts *.tif *.tiff *.xisf);;All Files (*)"
        )
        if not paths:
            return
        # Save the directory of the first selected file
        self._settings.setValue("DitherAnalysis/last_dir",
                                os.path.dirname(paths[0]))
        existing = {self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self._file_list.count())}
        for p in sorted(paths):
            if p not in existing:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setToolTip(p)
                self._file_list.addItem(item)

    def _remove_files(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))

    def _clear_files(self):
        self._file_list.clear()

    def _pick_save_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Folder", "")
        if d:
            self._save_dir = d
            self._lbl_save_dir.setText(d)

    def _load_sasd_dialog(self):
        last_dir = self._settings.value("DitherAnalysis/last_sasd_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Alignment Transform File", last_dir,
            "SASD Files (*.sasd);;All Files (*)"
        )
        if path:
            self._settings.setValue("DitherAnalysis/last_sasd_dir",
                                    os.path.dirname(path))
            self.load_sasd(path)

    # ----------------------------------------------------------------- run
    def _run(self):
        raw = [
            self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._file_list.count())
        ]
        paths = [p for p in raw if p is not None]
        if len(paths) < 2:
            QMessageBox.warning(self, "Dither Analysis",
                                "Add at least 2 light frames using Add Files.")
            return
        if self._chk_save.isChecked() and not self._save_dir:
            QMessageBox.warning(self, "Dither Analysis",
                                "Please choose an output folder first.")
            return

        self._filenames  = [os.path.basename(p) for p in paths]
        self._transforms = []
        self._ref_shape  = (0, 0)
        self._stats      = {}
        self._stats_text.clear()
        if self._plot:
            self._plot.fig.clear()
            self._plot.draw()

        self._btn_run.setEnabled(False)
        self._btn_abort.setEnabled(True)

        import tempfile
        from astropy.io import fits as _fits
        from PyQt6.QtWidgets import QProgressDialog
        from setiastro.saspro.legacy.image_manager import load_image

        self._temp_dir = tempfile.mkdtemp(prefix="dither_analysis_")
        clean_dir = os.path.join(self._temp_dir, "clean")
        os.makedirs(clean_dir, exist_ok=True)
        out_dir = self._save_dir if self._chk_save.isChecked() else self._temp_dir

        # ── Prepare frames: load via saspro (handles any format),
        #    write clean uint16 FITS with no BZERO/BSCALE so worker
        #    processes can memmap them without error.
        prep_progress = QProgressDialog(
            "Preparing frames…", "Cancel", 0, len(paths), self
        )
        prep_progress.setWindowTitle("Dither Analysis")
        prep_progress.setWindowModality(Qt.WindowModality.WindowModal)
        prep_progress.setMinimumDuration(0)
        prep_progress.setValue(0)
        prep_progress.show()

        clean_paths = []
        for i, p in enumerate(paths):
            prep_progress.setLabelText(
                f"Preparing frame {i + 1} of {len(paths)}\n{os.path.basename(p)}"
            )
            prep_progress.setValue(i)
            QApplication.processEvents()

            if prep_progress.wasCanceled():
                prep_progress.close()
                self._btn_run.setEnabled(True)
                self._btn_abort.setEnabled(False)
                self._lbl_status.setText("Cancelled.")
                return

            try:
                img, hdr, bit_depth, is_mono = load_image(p)
                if img is None:
                    raise ValueError(f"load_image returned None for {os.path.basename(p)}")

                # Extract pixscale from reference frame header (first frame only)
                if i == 0:
                    self._pixscale_arcsec = _extract_pixscale(hdr)
                    # stash it so we can save it once we know the sasd path
                    self._pending_pixscale = self._pixscale_arcsec

                # Build a clean header — preserve useful keywords but strip
                # everything that causes fits.open(memmap=True) to fail in
                # worker subprocesses (BZERO, BSCALE, BLANK, structural keys).
                clean_hdr = _fits.Header()
                if hdr is not None:
                    skip = {"BZERO", "BSCALE", "BLANK", "SIMPLE", "BITPIX",
                            "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
                            "EXTEND", "PCOUNT", "GCOUNT"}
                    for card in hdr.cards:
                        if card.keyword in skip:
                            continue
                        try:
                            clean_hdr.append(card)
                        except Exception:
                            pass

                # Write as float32 — BITPIX=-32 never gets BZERO/BSCALE added
                arr_f32 = np.clip(img, 0.0, 1.0).astype(np.float32)
                if arr_f32.ndim == 3 and arr_f32.shape[2] == 3:
                    arr_f32 = np.transpose(arr_f32, (2, 0, 1))  # HWC → CHW for FITS

                out = os.path.join(
                    clean_dir,
                    os.path.splitext(os.path.basename(p))[0] + ".fit"
                )
                _fits.PrimaryHDU(data=arr_f32, header=clean_hdr).writeto(
                    out, overwrite=True
                )
                clean_paths.append(out)

            except Exception as e:
                prep_progress.close()
                QMessageBox.warning(
                    self, "Dither Analysis",
                    f"Failed to prepare {os.path.basename(p)}:\n{e}"
                )
                self._btn_run.setEnabled(True)
                self._btn_abort.setEnabled(False)
                return

        prep_progress.setValue(len(paths))
        prep_progress.close()

        self._lbl_status.setText("Starting registration…")
        QApplication.processEvents()

        from setiastro.saspro.star_alignment import StarRegistrationThread, _align_prefs
        from PyQt6.QtCore import QSettings

        prefs = _align_prefs(QSettings())

        self._reg_thread = StarRegistrationThread(
            reference_image_path_or_view=clean_paths[0],
            files_to_align=clean_paths[1:],
            output_directory=out_dir,
            max_refinement_passes=1,
            shift_tolerance=0.5,
            parent_window=self.parent(),
            align_prefs=prefs,
        )
        from PyQt6.QtWidgets import QProgressDialog

        self._align_progress = QProgressDialog(
            "Running stellar alignment…", "Abort", 0, 0, self
        )
        self._align_progress.setWindowTitle("Dither Analysis — Registration")
        self._align_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._align_progress.setMinimumDuration(0)
        self._align_progress.setMinimumWidth(420)
        self._align_progress.setValue(0)
        self._align_progress.canceled.connect(self._abort)
        self._align_progress.show()

        self._reg_thread.progress_update.connect(self._on_reg_progress)
        self._reg_thread.progress_step.connect(self._on_reg_step)
        self._reg_thread.registration_complete.connect(self._on_reg_complete)
        self._reg_thread.start()

    def _on_reg_complete(self, success: bool, message: str):
        import shutil
        if hasattr(self, "_align_progress") and self._align_progress is not None:
            self._align_progress.close()
            self._align_progress = None
        self._btn_run.setEnabled(True)
        self._btn_abort.setEnabled(False)

        if not success:
            self._lbl_status.setText(f"Registration failed: {message}")
            QMessageBox.warning(self, "Dither Analysis",
                                f"Registration did not complete:\n{message}")
            try:
                if hasattr(self, "_temp_dir") and self._temp_dir:
                    shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            return

        am        = getattr(self._reg_thread, "alignment_matrices", {})
        ref2d     = getattr(self._reg_thread, "reference_image_2d", None)
        ref_shape = ref2d.shape[:2] if ref2d is not None else (0, 0)

        print(f"[DitherAnalysis] Registration complete: {message}")
        print(f"[DitherAnalysis] am keys: {len(am)}, ref_shape: {ref_shape}")

        # Reconstruct transform list aligned to the original UI file order.
        # clean_paths[0] was the reference (identity), clean_paths[1:] were aligned.
        # am keys are os.path.normpath(clean_path) for each file in files_to_align.
        raw_paths = [
            self._file_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._file_list.count())
        ]
        all_orig_paths = [p for p in raw_paths if p is not None]

        clean_dir = os.path.join(self._temp_dir, "clean")

        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        transforms = [identity]  # frame 0 is the reference

        for p in all_orig_paths[1:]:
            # stem only — clean copies are always .fit regardless of source format
            stem = os.path.splitext(os.path.basename(p))[0]
            clean_p = os.path.normpath(os.path.join(clean_dir, stem + ".fit"))
            T = am.get(clean_p)
            transforms.append(T)

        filenames = [os.path.basename(p) for p in all_orig_paths]

        print(f"[DitherAnalysis] transforms non-None: {sum(1 for t in transforms if t is not None)}")

        # Prompt to save the .sasd before cleanup
        sasd_src = os.path.join(
            getattr(self._reg_thread, "output_directory",
                    getattr(self, "_temp_dir", "")),
            "alignment_transforms.sasd"
        )

        if os.path.exists(sasd_src):
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Alignment Transforms (.sasd)",
                os.path.join(os.path.expanduser("~"), "dither_alignment.sasd"),
                "SASD Files (*.sasd);;All Files (*)"
            )
            if save_path:
                try:
                    shutil.copy2(sasd_src, save_path)
                    # persist pixscale keyed to this sasd
                    if getattr(self, "_pending_pixscale", None):
                        _save_pixscale_for_sasd(
                            self._settings, save_path, self._pending_pixscale
                        )
                except Exception as e:
                    QMessageBox.warning(self, "Save .sasd", f"Could not save:\n{e}")

            # also save against the temp sasd so bolt-on path works
            if getattr(self, "_pending_pixscale", None):
                _save_pixscale_for_sasd(
                    self._settings, sasd_src, self._pending_pixscale
                )

        # Clean up temp dir
        try:
            if hasattr(self, "_temp_dir") and self._temp_dir:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

        self._lbl_status.setText("Registration complete.  Analysing offsets…")
        self.load_transforms(transforms, filenames, ref_shape=ref_shape)
    def _abort(self):
        if self._reg_thread and self._reg_thread.isRunning():
            self._reg_thread.terminate()
            self._reg_thread.wait(3000)
        self._btn_run.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._lbl_status.setText("Aborted.")

    # ----------------------------------------------------------------- thread slots
    def _on_reg_progress(self, msg: str):
        print(f"[DA progress] {msg}")
        # Update label on the alignment progress dialog if it exists
        if hasattr(self, "_align_progress") and self._align_progress is not None:
            short = msg.splitlines()[-1] if "\n" in msg else msg
            self._align_progress.setLabelText(short[:120])
        self._lbl_status.setText((msg.splitlines()[-1] if "\n" in msg else msg)[:120])

    def _on_reg_step(self, done: int, total: int):
        if hasattr(self, "_align_progress") and self._align_progress is not None:
            if total > 0:
                self._align_progress.setMaximum(total)
                self._align_progress.setValue(done)
            else:
                self._align_progress.setMaximum(0)
                self._align_progress.setValue(0)


    # ─────────────────────────────────────────────────────────────────
    #  Public entry points
    # ─────────────────────────────────────────────────────────────────
    def load_transforms(
        self,
        transforms: list,
        filenames: list[str],
        ref_shape: tuple[int, int] = (0, 0),
    ):
        """
        Load pre-computed 2×3 transforms (or None for failed frames) and analyze.
        ref_shape = (H, W) of the reference/output image.
        """
        self._transforms = list(transforms)
        self._filenames  = list(filenames)
        self._ref_shape  = ref_shape

        self._file_list.clear()
        for fn in filenames:
            self._file_list.addItem(QListWidgetItem(fn))

        self._analyze()

    def load_sasd(self, path: str):
        try:
            basenames, transforms, ref_shape = parse_sasd(path)
        except Exception as e:
            QMessageBox.critical(self, "Load .sasd", f"Failed to parse:\n{e}")
            return
        if not basenames:
            QMessageBox.warning(self, "Load .sasd", "No frame entries found.")
            return

        self._pixscale_arcsec = None

        # 1) Check QSettings cache first — fastest and works even when
        #    source files are on a different drive or have moved
        ps = _load_pixscale_for_sasd(self._settings, path)
        if ps:
            self._pixscale_arcsec = ps

        # 2) Try to find a source file near the sasd and read its header
        if not self._pixscale_arcsec:
            sasd_dir = os.path.dirname(path)
            for basename in basenames[:5]:
                try:
                    src = _find_source_file(sasd_dir, basename)
                    if src:
                        from setiastro.saspro.legacy.image_manager import load_image
                        _, hdr, _, _ = load_image(src)
                        ps = _extract_pixscale(hdr)
                        if ps:
                            self._pixscale_arcsec = ps
                            # cache it for next time
                            _save_pixscale_for_sasd(self._settings, path, ps)
                            break
                except Exception:
                    continue

        # 3) Try the REF_PATH stored in the sasd
        if not self._pixscale_arcsec:
            try:
                ref_path = _read_sasd_ref_path(path)
                if ref_path and os.path.exists(ref_path):
                    from setiastro.saspro.legacy.image_manager import load_image
                    _, hdr, _, _ = load_image(ref_path)
                    ps = _extract_pixscale(hdr)
                    if ps:
                        self._pixscale_arcsec = ps
                        _save_pixscale_for_sasd(self._settings, path, ps)
            except Exception:
                pass

        self._lbl_status.setText(
            f"Loaded {len(basenames)} frames from {os.path.basename(path)}"
            + (f"  |  {self._pixscale_arcsec:.3f}\"/px" if self._pixscale_arcsec else "")
        )
        self.load_transforms(transforms, basenames, ref_shape=ref_shape)

    # ----------------------------------------------------------------- analyze
    def _analyze(self):
        H, W = self._ref_shape
        if H == 0 or W == 0:
            # No shape info — fall back to raw tx/ty with a warning
            self._lbl_status.setText(
                "⚠  No REF_SHAPE available — showing raw tx/ty offsets "
                "(meridian flips will not be corrected)."
            )
            self._analyze_raw()
            return

        # Compute image-centre offsets through each transform
        dx_full, dy_full, flipped_full = _center_offsets(self._transforms, (H, W))

        failed  = int(np.sum(np.isnan(dx_full)))
        mask    = ~np.isnan(dx_full)
        n_good  = int(mask.sum())
        n_total = len(self._transforms)

        if n_good < 2:
            msg = (
                f"Only {n_good} frame(s) registered successfully "
                f"({failed} failed out of {n_total}).  "
                "Need at least 2 valid frames."
            )
            self._lbl_status.setText(msg)
            self._stats_text.setPlainText(msg)
            return

        stats = _compute_stats(dx_full[mask], dy_full[mask], flipped_full[mask])
        stats["n_total"]  = n_total
        stats["n_failed"] = failed
        self._stats = stats

        self._render_stats(stats, dx_full, dy_full, flipped_full)
        if self._plot is not None:
            self._plot.render(stats, self._unit_scale, self._unit_label)

        ps = getattr(self, "_pixscale_arcsec", None)
        ps_suffix = f" ({stats['rms_offset']*ps:.2f}\")" if ps else ""
        step_suffix = f" ({stats['mean_step']*ps:.2f}\")" if ps else ""
        self._lbl_status.setText(
            f"Done — {n_good}/{n_total} frames  |  "
            f"RMS dither: {stats['rms_offset']:.2f} px{ps_suffix}  |  "
            f"Mean step: {stats['mean_step']:.2f} px{step_suffix}"
            + (f"  |  Flipped frames: {stats['n_flipped']}" if stats["n_flipped"] else "")
        )
        # Enable/disable arcsec toggle based on pixscale availability
        ps = getattr(self, "_pixscale_arcsec", None)
        self._btn_px_arcsec.setEnabled(bool(ps))
        if not ps:
            self._btn_px_arcsec.setToolTip("No pixel scale found in header")
        else:
            self._btn_px_arcsec.setToolTip(f"Pixel scale: {ps:.3f}\"/px")


    def _analyze_raw(self):
        """Fallback: use raw tx/ty when no REF_SHAPE is available."""
        dx_list, dy_list = [], []
        failed = 0
        for M in self._transforms:
            if M is None:
                failed += 1
                dx_list.append(np.nan)
                dy_list.append(np.nan)
            else:
                arr = np.asarray(M, dtype=np.float64).reshape(2, 3)
                dx_list.append(arr[0, 2])
                dy_list.append(arr[1, 2])

        dx = np.array(dx_list)
        dy = np.array(dy_list)
        mask   = ~(np.isnan(dx) | np.isnan(dy))
        n_good = int(mask.sum())

        if n_good < 2:
            return

        flipped = _detect_flips(self._transforms)
        stats = _compute_stats(dx[mask], dy[mask], flipped[mask])
        stats["n_total"]  = len(self._transforms)
        stats["n_failed"] = failed
        self._stats = stats

        self._render_stats(stats, dx, dy, flipped)
        if self._plot is not None:
            self._plot.render(stats)

        # Enable/disable arcsec toggle based on pixscale availability
        ps = getattr(self, "_pixscale_arcsec", None)
        self._btn_px_arcsec.setEnabled(bool(ps))
        if not ps:
            self._btn_px_arcsec.setToolTip("No pixel scale found in header")
        else:
            self._btn_px_arcsec.setToolTip(f"Pixel scale: {ps:.3f}\"/px")

    # ----------------------------------------------------------------- stats text
    def _render_stats(
        self,
        s: dict,
        dx_full: np.ndarray,
        dy_full: np.ndarray,
        flipped_full: np.ndarray,
    ):
        pref       = _dir_label(s["pref_dir"])
        cov        = s["coverage_px"]
        H, W       = self._ref_shape
        shape_str  = f"{W} × {H} px" if W and H else "unknown"
        ps         = getattr(self, "_pixscale_arcsec", None)
        unit_scale = self._unit_scale
        ul         = self._unit_label

        def _px(val_px: float) -> str:
            if ps and ps > 0:
                return f"{val_px:.2f} px  ({val_px * ps:.2f}\")"
            return f"{val_px:.2f} px"

        def _px2(val_px: float) -> str:
            if ps and ps > 0:
                return f"{val_px:.1f} px²  ({val_px * ps * ps:.1f} arcsec²)"
            return f"{val_px:.1f} px²"

        cov_str = _px2(cov) if cov > 0 else "N/A (install scipy)"
        ps_str  = f"{ps:.3f}\"/px" if ps else "unknown (no WCS/pixscale in header)"

        # ── Clustering ───────────────────────────────────────────────
        is_clustered = s.get("is_clustered", False)
        n_cl  = s.get("n_clusters", 1)
        mc    = s.get("mean_cluster_size", 1.0)
        mx    = s.get("max_cluster_size", 1)
        cfrac = s.get("clustered_frac", 0.0)
        cthr  = s.get("step_threshold", 0.0) * unit_scale

        if is_clustered:
            cluster_note = (
                f"  ⚠  Clustering detected — {cfrac*100:.0f}% of frames share a\n"
                f"     pointing position with neighbors.\n"
                f"     Within-cluster frames share fixed-pattern noise and\n"
                f"     may not be fully separated by sigma rejection."
            )
        else:
            cluster_note = f"  ✓  No significant clustering  (mean cluster size: {mc:.1f})"

        lines = [
            "═" * 52,
            f"  Reference size   : {shape_str}",
            f"  Pixel scale      : {ps_str}",
            f"  Frames analyzed  : {s['n']} / {s['n_total']}",
            f"  Failed / skipped : {s['n_failed']}",
            f"  Flipped frames   : {s['n_flipped']}  (frames on opposite meridian side)",
            "─" * 52,
            "  Image-centre dither offsets:",
            f"  RMS offset       : {_px(s['rms_offset'])}",
            f"  Mean offset      : {_px(s['mean_radius'])}",
            f"  Max offset       : {_px(s['max_radius'])}",
            f"  Span X           : {_px(s['span_x'])}",
            f"  Span Y           : {_px(s['span_y'])}",
            f"  Std(ΔX)          : {_px(s['std_dx'])}",
            f"  Std(ΔY)          : {_px(s['std_dy'])}",
            "─" * 52,
            "  Step statistics  (flip transitions excluded):",
            f"  Mean step        : {_px(s['mean_step'])}",
            f"  Max step         : {_px(s['max_step'])}",
            f"  Preferred dir    : {s['pref_dir']:.1f}°  ({pref})",
            f"  Coverage area    : {cov_str}",
            "─" * 52,
            "  Clustering analysis:",
            f"  Distinct positions  : {n_cl}",
            f"  Mean frames/position: {mc:.1f}",
            f"  Largest cluster     : {mx} frames",
            f"  Clustered fraction  : {cfrac*100:.1f}%",
            f"  Step threshold used : {cthr:.2f} {ul}",
            "",
            cluster_note,
            "═" * 52,
            "",
            "  Per-frame  (ΔX, ΔY = image-centre offset from ref,  r = radius):",
            "─" * 52,
        ]

        for i, (fn, M) in enumerate(zip(self._filenames, self._transforms)):
            if M is None:
                lines.append(f"  [{i:>4}] {fn}  →  FAILED")
            else:
                dx_v = dx_full[i]
                dy_v = dy_full[i]
                r    = math.hypot(dx_v, dy_v) if not math.isnan(dx_v) else float("nan")
                flip_tag = "  ⟳ FLIP" if flipped_full[i] else ""
                if ps and not math.isnan(r):
                    lines.append(
                        f"  [{i:>4}] {fn}  →  "
                        f"ΔX={dx_v:+7.2f}px ({dx_v*ps:+7.2f}\")  "
                        f"ΔY={dy_v:+7.2f}px ({dy_v*ps:+7.2f}\")  "
                        f"r={r:6.2f}px ({r*ps:.2f}\"){flip_tag}"
                    )
                else:
                    lines.append(
                        f"  [{i:>4}] {fn}  →  "
                        f"ΔX={dx_v:+7.2f}  ΔY={dy_v:+7.2f}  r={r:6.2f} px{flip_tag}"
                    )

        self._stats_text.setPlainText("\n".join(lines))

    # ----------------------------------------------------------------- geometry
    def _restore_geometry(self):
        g = self._settings.value("DitherAnalysis/geometry")
        if g:
            self.restoreGeometry(g)

    def closeEvent(self, event):
        self._settings.setValue("DitherAnalysis/geometry", self.saveGeometry())
        if self._reg_thread and self._reg_thread.isRunning():
            self._reg_thread.terminate()
            self._reg_thread.wait(3000)
        super().closeEvent(event)