#saspro/continuum_subtract.py
from __future__ import annotations
from setiastro.saspro.main_helpers import non_blocking_sleep
import os
import numpy as np

# Optional deps used by the processing threads
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pywt
except Exception:
    pywt = None

from PyQt6.QtCore import (
    Qt, QSize, QPoint, QEvent, QThread, pyqtSignal, QTimer,
    QCoreApplication, QSettings
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QScrollArea, QDialog, QInputDialog, QFileDialog,
    QMessageBox, QCheckBox, QApplication, QMainWindow, QCheckBox, QComboBox, QSlider, QSizePolicy, QGraphicsScene,QGraphicsPixmapItem,
    QTabWidget, QSplitter, QGridLayout
)
from PyQt6.QtGui import (
    QPixmap, QImage, QCursor, QWheelEvent
)

# register QImage for cross-thread signals
#qRegisterMetaType(QImage)

from .doc_manager import ImageDocument  # add this import
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image  # CHANGED
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.imageops.starbasedwhitebalance import (
    apply_star_based_white_balance,
    # Same MAD-sigma-clipped iteration the WB slope solve uses. Private by
    # name, shared on purpose -- the diagnostic must draw the line the same
    # way the fit would, or it misreports the fit it is diagnosing.
    _robust_slope_fit as wb_robust_slope_fit,
)
from setiastro.saspro.legacy.numba_utils import apply_curves_numba
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
# At the top of continuum_subtract.py, with the other imports:
try:
    from setiastro.saspro.cosmicclarity_headless import run_cosmicclarity_on_array as _cc_denoise
except Exception:
    _cc_denoise = None

# Star detection for the WB fit is automatic: sep.extract runs at each sigma
# in turn, strictest first, and the first rung that finds at least
# AUTO_STARS_DEFAULT stars wins. One fixed sigma cannot serve both a dense
# Milky Way field and a sparse high-latitude one -- the same number either
# returns tens of thousands of detections or a handful.
AUTO_SIGMA_LADDER = (200.0, 100.0, 50.0, 20.0, 10.0, 5.0, 3.0)
AUTO_STARS_DEFAULT = 500

# Where the Q Preview dialog opens. After a correct star-based white
# balance Q = 1 is the pure value -- stars cancel exactly and line emission
# survives untouched -- so that is where to start from and tune away from,
# rather than from a hedge.
DEFAULT_Q = 1.00

# Every user-facing option in this tool is written here on change and read
# back on open, so the dialog comes up the way it was left.
_SKEY = "continuum_subtract/"


def apply_curves_adjustment(image, target_median, curves_boost):
    """
    Original signature unchanged, but now uses a Numba helper
    to do the pixel-by-pixel interpolation.

    'image' can be 2D (H,W) or 3D (H,W,3).
    """
    # Build the curve array as before
    curve = [
        [0.0, 0.0],
        [0.5 * target_median, 0.5 * target_median],
        [target_median, target_median],
        [
            (1/4 * (1 - target_median) + target_median),
            np.power((1/4 * (1 - target_median) + target_median), (1 - curves_boost))
        ],
        [
            (3/4 * (1 - target_median) + target_median),
            np.power(np.power((3/4 * (1 - target_median) + target_median), (1 - curves_boost)), (1 - curves_boost))
        ],
        [1.0, 1.0]
    ]
    # Convert to arrays
    xvals = np.array([p[0] for p in curve], dtype=np.float32)
    yvals = np.array([p[1] for p in curve], dtype=np.float32)

    # Ensure 'image' is float32
    image_32 = image.astype(np.float32, copy=False)

    # Now apply the piecewise linear function in Numba
    adjusted_image = apply_curves_numba(image_32, xvals, yvals)
    return adjusted_image

class QPreviewDialog(QDialog):
    """
    Non-blocking dialog shown after calibration, before final subtraction.
    Lets the user tweak Q per filter with a slider + live autostretch preview.
    """

    # emitted when user accepts: dict of {filter_name: q_value}
    accepted_q = pyqtSignal(dict)

    def __init__(self, calibration_entries: list[dict], parent=None):
        """
        calibration_entries: list of dicts, one per filter, each containing:
            {
                "name":        str,           # "Ha", "OIII", etc.
                "balanced":    np.ndarray,    # full-res float32 HxWx3 balanced RGB
                "green_med":   float,
                "recipe":      dict,          # full recipe for final pass
                "preview":     np.ndarray,    # downsampled balanced RGB for UI
            }
        """
        super().__init__(parent)
        self.setWindowTitle("Interactive Q Preview")
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.resize(900, 700)

        self._entries     = {e["name"]: e for e in calibration_entries}
        self._q_values    = {e["name"]: DEFAULT_Q for e in calibration_entries}
        self._current     = calibration_entries[0]["name"] if calibration_entries else ""
        self._autostretch = True

        # debounce timer
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._update_preview)

        self._build_ui()
        self._update_preview()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
        from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

        outer = QVBoxLayout(self)
        outer.setSpacing(8)
        outer.setContentsMargins(10, 10, 10, 10)

        # ── filter selector header (prominent, centered) ──────────────────────
        filter_header = QHBoxLayout()
        filter_header.addStretch(1)

        filter_title = QLabel("Filter:")
        filter_title.setStyleSheet("font-size:13px; font-weight:600;")
        filter_header.addWidget(filter_title)

        self._filter_combo = QComboBox()
        self._filter_combo.setStyleSheet("font-size:13px; font-weight:600; padding: 3px 10px;")
        self._filter_combo.setMinimumWidth(140)
        for name in self._entries:
            self._filter_combo.addItem(name)
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_header.addWidget(self._filter_combo)

        filter_header.addStretch(1)
        outer.addLayout(filter_header)

        # thin separator under the filter header
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: palette(mid);")
        outer.addWidget(sep)

        # ── Q slider + autostretch toolbar row ────────────────────────────────
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Q:"))

        self._q_slider = QSlider(Qt.Orientation.Horizontal)
        self._q_slider.setMinimum(10)
        self._q_slider.setMaximum(200)
        self._q_slider.setValue(int(round(DEFAULT_Q * 100)))
        self._q_slider.setTickInterval(10)
        self._q_slider.setFixedWidth(260)
        self._q_slider.valueChanged.connect(self._on_slider_changed)
        toolbar.addWidget(self._q_slider)

        self._q_label = QLabel(f"{DEFAULT_Q:.2f}")
        self._q_label.setFixedWidth(36)
        toolbar.addWidget(self._q_label)

        toolbar.addSpacing(20)
        self._stretch_check = QCheckBox("Autostretch")
        self._stretch_check.setChecked(True)
        self._stretch_check.toggled.connect(self._on_stretch_toggled)
        toolbar.addWidget(self._stretch_check)

        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        # ── zoom toolbar row ──────────────────────────────────────────────────
        zoom_row = QHBoxLayout()

        btn_zoom_out  = themed_toolbtn("zoom-out",      "Zoom Out")
        btn_zoom_in   = themed_toolbtn("zoom-in",       "Zoom In")
        btn_zoom_1to1 = themed_toolbtn("zoom-original", "1:1 (100%)")
        btn_zoom_fit  = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        zoom_row.addWidget(btn_zoom_out)
        zoom_row.addWidget(btn_zoom_in)
        zoom_row.addWidget(btn_zoom_1to1)
        zoom_row.addWidget(btn_zoom_fit)
        zoom_row.addStretch(1)
        outer.addLayout(zoom_row)

        # ── graphics view ─────────────────────────────────────────────────────
        self._scene = QGraphicsScene(self)
        self._view  = ZoomableGraphicsView(self._scene)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        outer.addWidget(self._view, stretch=1)

        # wire zoom buttons
        btn_zoom_out.clicked.connect(self._view.zoom_out)
        btn_zoom_in.clicked.connect(self._view.zoom_in)
        btn_zoom_1to1.clicked.connect(
            self._view.one_to_one if hasattr(self._view, "one_to_one") else (lambda: None)
        )
        btn_zoom_fit.clicked.connect(lambda: self._view.fit_to_item(self._pix_item))

        # ── status + buttons ──────────────────────────────────────────────────
        self._status = QLabel("Adjust Q and click Accept to run final subtraction.")
        self._status.setStyleSheet("font-size:11px; color: palette(placeholderText);")
        outer.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_accept = QPushButton("Accept — Run Final")
        self._btn_accept.setDefault(True)
        self._btn_accept.clicked.connect(self._on_accept)
        btn_row.addWidget(self._btn_accept)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        outer.addLayout(btn_row)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_filter_changed(self, name: str):
        self._current = name
        q = self._q_values.get(name, DEFAULT_Q)
        self._q_slider.blockSignals(True)
        self._q_slider.setValue(int(round(q * 100)))
        self._q_slider.blockSignals(False)
        self._q_label.setText(f"{q:.2f}")
        self._debounce.start()

    def _on_slider_changed(self, value: int):
        q = value / 100.0
        self._q_label.setText(f"{q:.2f}")
        self._q_values[self._current] = q
        self._debounce.start()   # debounced — won't fire until slider rests 150ms

    def _on_stretch_toggled(self, checked: bool):
        self._autostretch = checked
        self._debounce.start()

    def _on_accept(self):
        self.accepted_q.emit(dict(self._q_values))
        self.accept()

    # ── preview rendering ─────────────────────────────────────────────────────
    def _update_preview(self):
        entry = self._entries.get(self._current)
        if entry is None:
            return

        q       = self._q_values.get(self._current, DEFAULT_Q)
        preview = entry["preview"]
        gmed    = entry["green_med"]

        r = preview[..., 0]
        g = preview[..., 1]
        result = np.clip(r - q * (g - gmed), 0.0, 1.0).astype(np.float32, copy=False)

        if self._autostretch:
            from setiastro.saspro.imageops.stretch import stretch_mono_image
            result = stretch_mono_image(result, target_median=0.25, no_black_clip=True)
            result = np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

        u8 = (result * 255.0 + 0.5).astype(np.uint8, copy=False)
        h, w = u8.shape[:2]
        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pm   = QPixmap.fromImage(qimg)

        self._pix_item.setPixmap(pm)
        self._scene.setSceneRect(self._pix_item.boundingRect())

        # fit on first render; subsequent slider updates leave zoom alone
        if not getattr(self, "_initial_fit_done", False):
            self._initial_fit_done = True
            self._view.fit_to_item(self._pix_item)

        stars = entry.get("star_count", 0)
        self._status.setText(
            f"Filter: {self._current}   Q: {q:.2f}   "
            f"Stars used for WB: {stars}   (preview is downsampled)"
        )


class _CalibrationRunner(QThread):
    """Thin wrapper: calls thread.run_calibration_only(name) off the UI thread."""
    def __init__(self, thread: ContinuumProcessingThread, name: str, parent=None):
        super().__init__(parent=None)
        self._t    = thread
        self._name = name

    def run(self):
        self._t.run_calibration_only(self._name)

# Photometric clip bounds, shared by the fit and the plot. A star
# saturated in either axis has lost its photometry; one at the noise floor
# never had any.
FIT_CLIP_HI = 0.95

# Absolute fallback floor, used only when the caller cannot supply a
# measured one. It is deliberately tiny: a FIXED floor in [0,1] units is
# meaningless on linear data, where the entire frame may sit below 0.02.
# The real floor is background + FIT_FLOOR_K sigma, measured per frame --
# see ContinuumProcessingThread._axis_floor.
FIT_CLIP_LO = 1e-5

# How much signal a star needs before its flux ratio means anything,
# expressed in per-pixel background sigma.
#
# The quantity being tested is not a pixel: it is the circular MEDIAN over
# an r=3 aperture, about 28 pixels, whose noise is ~1.2533*sigma/sqrt(N)
# ~= 0.24 sigma. So an aperture SNR of 5 is 5 * 0.24 = 1.2 per-pixel
# sigma. Quoting the threshold in per-pixel sigma and setting it to 5 --
# as the first cut of this did -- actually demands an aperture SNR of ~21
# and throws away most of a normal field.
FIT_FLOOR_SNR = 5.0
FIT_FLOOR_APERTURE_PX = 28.0
FIT_FLOOR_K = FIT_FLOOR_SNR * 1.2533 / (FIT_FLOOR_APERTURE_PX ** 0.5)
FIT_SIGMA = 2.5

# Spatial stratification of the fit sample. The frame is divided into
# FIT_GRID x FIT_GRID cells and no cell may contribute more than
# FIT_CELL_SHARE times its fair share of the fit, so a crowded corner --
# or the one region with nebulosity in it -- cannot quietly become the
# whole fit.
FIT_GRID = 4
FIT_CELL_SHARE = 2.0


def _stratify(keep: np.ndarray, xy, shape, flux, *,
              grid: int = FIT_GRID,
              max_share: float = FIT_CELL_SHARE) -> tuple[np.ndarray, dict]:
    """Thin over-represented cells out of `keep`, and describe the spread.

    Within a cell the survivors are taken evenly across the flux range
    rather than brightest-first: a cell reduced to its brightest stars
    would hand the fit exactly the non-linear tail the sigma clip is
    there to remove.

    Returns (keep, coverage) where coverage describes the FINAL sample.
    """
    # cells == 0 means "not measured": without centroids there is nothing
    # to say about coverage, which is different from covering none of it.
    cov = {"grid": int(grid), "cells": 0,
           "cells_used": 0, "max_share": float("nan"),
           "thinned": 0}
    idx = np.where(keep)[0]
    if idx.size == 0 or xy is None or shape is None:
        return keep, cov

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != keep.size or xy.shape[1] < 2:
        return keep, cov

    h, w = float(shape[0]), float(shape[1])
    if not (h > 0 and w > 0):
        return keep, cov

    cx = np.clip((xy[idx, 0] / w * grid).astype(int), 0, grid - 1)
    cy = np.clip((xy[idx, 1] / h * grid).astype(int), 0, grid - 1)
    cell = cy * grid + cx

    occupied, counts = np.unique(cell, return_counts=True)
    # Cap against the TYPICAL cell, not the mean: one crowded cell drags
    # the mean up toward itself, so a mean-based cap lets it keep most of
    # what made it an outlier in the first place. A floor of 10 keeps the
    # rule from biting on genuinely sparse fields.
    cap = max(10, int(np.ceil(max_share * float(np.median(counts)))))

    out = np.zeros_like(keep)
    thinned = 0
    for c in occupied:
        members = idx[cell == c]
        if members.size > cap:
            order = members[np.argsort(np.asarray(flux)[members])]
            pick = order[np.linspace(0, order.size - 1, cap).astype(int)]
            thinned += members.size - pick.size
            members = pick
        out[members] = True

    final = np.where(out)[0]
    fcx = np.clip((xy[final, 0] / w * grid).astype(int), 0, grid - 1)
    fcy = np.clip((xy[final, 1] / h * grid).astype(int), 0, grid - 1)
    fcell = fcy * grid + fcx
    counts = np.bincount(fcell, minlength=grid * grid)
    cov["cells"] = int(grid * grid)
    cov["cells_used"] = int(np.count_nonzero(counts))
    cov["max_share"] = (float(counts.max()) / float(final.size)
                        if final.size else float("nan"))
    cov["thinned"] = int(thinned)
    return out, cov


def fit_star_flux(nb_flux, cont_flux, *, sigma: float = FIT_SIGMA,
                  clip_lo=FIT_CLIP_LO, xy=None, shape=None) -> dict:
    """Fit NB star flux against broadband star flux, robustly.

    Stage 1 is a hard clip: drop anything saturated (> FIT_CLIP_HI) or at
    the noise floor in EITHER axis. `clip_lo` is the floor -- a scalar, or
    (nb_floor, bb_floor) measured from the frame itself, which is what the
    caller should pass: a fixed number in [0,1] units cuts most of a linear
    frame's stars for no photometric reason.

    Stage 1b is spatial: `xy` (star centroids) and `shape` (image H, W)
    stratify the survivors so no part of the frame dominates the fit.

    Stage 2 is an iterative MAD-sigma clip in slope space -- fit, measure
    the robust scatter of residual/flux, drop what falls outside
    y = (m +/- sigma*scale)x, refit, repeat until nothing more drops. It
    is stage 2 that removes the bright non-linear tail: those stars pass
    the hard clip, and one unweighted least-squares pass follows them up.

    Returns a dict of the fit and its quality. `keep` and `hard` are
    boolean masks over the full input, so a caller can draw what was
    rejected and why.
    """
    nb = np.asarray(nb_flux, dtype=np.float64).reshape(-1)
    bb = np.asarray(cont_flux, dtype=np.float64).reshape(-1)
    n_total = int(nb.size)
    out = {
        "ok": False, "slope": float("nan"), "intercept": float("nan"),
        "n_total": n_total, "n_hard": 0, "n_fit": 0,
        "n_saturated": 0, "n_floor": 0,
        "scatter": float("nan"), "r2": float("nan"),
        "x_lo": float("nan"), "x_hi": float("nan"),
        "keep": np.zeros(n_total, bool), "hard": np.zeros(n_total, bool),
        "band": None, "coverage": {}, "floor": None,
    }
    if n_total < 3:
        return out

    try:
        lo_nb, lo_bb = (float(clip_lo[0]), float(clip_lo[1]))
    except (TypeError, IndexError):
        lo_nb = lo_bb = float(clip_lo)
    out["floor"] = (lo_nb, lo_bb)

    finite = np.isfinite(nb) & np.isfinite(bb)
    sat = finite & ((nb >= FIT_CLIP_HI) | (bb >= FIT_CLIP_HI))
    floor = finite & ~sat & ((nb <= lo_nb) | (bb <= lo_bb))
    hard = finite & ~sat & ~floor
    out["hard"] = hard
    out["n_hard"] = int(hard.sum())
    out["n_saturated"] = int(sat.sum())
    out["n_floor"] = int(floor.sum())

    # Thin over-represented regions BEFORE fitting, so the line is not
    # decided by whichever corner happens to be crowded.
    hard, cov = _stratify(hard, xy, shape, bb)
    out["hard"] = hard
    out["n_hard"] = int(hard.sum())

    try:
        if int(hard.sum()) >= 3:
            m, c, inlier = wb_robust_slope_fit(
                bb[hard], nb[hard], sigma=sigma, max_iter=8,
                min_points=10, relative=True)
            keep = np.zeros(n_total, bool)
            keep[np.where(hard)[0][np.asarray(inlier, bool)]] = True
        else:
            m, c = np.polyfit(bb[finite], nb[finite], 1)
            keep = finite.copy()
    except Exception:
        return out

    out.update({"ok": True, "slope": float(m), "intercept": float(c),
                "keep": keep, "n_fit": int(keep.sum())})

    xk, yk = bb[keep], nb[keep]
    if xk.size >= 3:
        out["x_lo"] = float(xk.min())
        out["x_hi"] = float(xk.max())
        resid = yk - (m * xk + c)
        # Scatter is quoted RELATIVE, because that is the criterion the
        # clip uses and because "stars sit within x% of the line" is the
        # readable form.
        xref = max(float(np.median(np.abs(xk))), 1e-9)
        rel = resid / np.maximum(np.abs(xk), xref)
        mad = float(np.median(np.abs(rel - np.median(rel))))
        scale = 1.4826 * mad if mad > 0 else float(np.std(rel))
        out["scatter"] = float(scale)
        if scale > 1e-9:
            out["band"] = (sigma * scale, xref)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yk - np.mean(yk)) ** 2))
        out["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Coverage is reported for the stars that ended up IN the fit, which is
    # what the question "where did this line come from" is really asking.
    _, cov_final = _stratify(keep, xy, shape, bb, max_share=1e9)
    cov_final["thinned"] = cov.get("thinned", 0)
    out["coverage"] = cov_final
    return out


def star_residual_at_unit_q(stats: dict) -> float:
    """Fraction of its own flux an average star keeps at Q = 1.

    Positive: stars are left bright. Negative: they are dug dark.

    This is NOT a recommendation to move Q. Q scales the continuum across
    the whole image, so buying a cleaner star by raising it costs the same
    fraction of continuum everywhere else, line emission included. With a
    correct star-based white balance Q = 1 is right by construction; a
    residual here means the white balance did not fully land, and that is
    where to look.

    It is also only the bright-star limit: the residual after subtraction
    is (m - Q)*C + c + Q*median(C), so the exact cancelling Q is
    m + c/(C - median) -- flux-dependent unless the intercept is zero.
    """
    if not stats or not stats.get("ok"):
        return float("nan")
    m = float(stats.get("slope", float("nan")))
    if not np.isfinite(m):
        return float("nan")
    return float(m - 1.0)


def wb_fit_verdict(stats: dict) -> tuple[str, str]:
    """A plain-language read on a post-WB fit: (verdict, why).

    Two terms, because they fail differently:

      * how far the balanced slope sits from 1.0 -- how much of itself an
        average star keeps at Q = 1, i.e. how well the white balance
        landed;
      * the scatter about that line -- the spread no single Q can fix.

    Sample size and spatial spread qualify the result rather than grade
    it. The thresholds are a judgement call, not a standard -- they are
    stated here so the number on screen can be argued with. They are
    deliberately not tight: a few percent of residual on a real frame is
    ordinary, and was being called "Marginal" by an earlier cut of this.
    """
    if not stats or not stats.get("ok") or stats.get("n_fit", 0) < 10:
        return ("Not enough stars",
                "Fewer than 10 stars survived clipping; the fit is not "
                "worth reading.")

    sct = float(stats.get("scatter", float("nan")))
    m = float(stats.get("slope", float("nan")))
    if not np.isfinite(sct) or not np.isfinite(m):
        return ("Unknown", "The fit did not converge to usable numbers.")

    res = abs(m - 1.0)

    if m < 0.5 or m > 1.6:
        return ("Check the pairing",
                f"The balanced stars sit on a slope of {m:.2f}. At that "
                "distance from 1.0 the white balance has not landed at "
                "all -- usually a narrowband frame matched to the wrong "
                "continuum channel, or a heavy uncorrected gradient.")

    cov = stats.get("coverage") or {}
    notes = []
    if stats["n_fit"] < 50:
        notes.append(f"only {stats['n_fit']} stars")
    share = cov.get("max_share", float("nan"))
    cells, used = cov.get("cells", 0), cov.get("cells_used", 0)
    if cells and used and used <= max(1, cells // 3):
        notes.append(f"all from {used} of {cells} regions")
    elif np.isfinite(share) and share > 0.35:
        notes.append(f"{share * 100:.0f}% from one region")
    qual = ("" if not notes
            else " Treat it with care: " + ", ".join(notes) + ".")

    if res <= 0.03 and sct <= 0.05:
        grade, why = ("Excellent",
                      "The white balance landed: stars sit on one tight "
                      "line through y = x, and cancel at Q = 1.")
    elif res <= 0.08 and sct <= 0.12:
        grade, why = ("Good",
                      "The white balance landed close enough that stars "
                      "very nearly cancel at Q = 1.")
    elif res <= 0.15 and sct <= 0.25:
        grade, why = ("Fair",
                      "Stars will leave a visible residual at Q = 1, and "
                      "the spread means no single Q removes all of them.")
    else:
        grade, why = ("Loose",
                      "The balanced stars neither sit on y = x nor follow "
                      "one line. Check for a gradient, a mismatched "
                      "continuum, or heavy emission-line contamination in "
                      "the star sample.")
    _r = star_residual_at_unit_q(stats)
    if np.isfinite(_r) and abs(_r) >= 0.01:
        why += (f" At Q = 1 an average star keeps {_r * 100:+.1f}% of "
                "itself.")
    return (grade, why + qual)


class _FitPixmapLabel(QLabel):
    """A label that always draws its pixmap as large as it will fit.

    The WB diagnostics used to be 256 px thumbnails in a fixed grid, which
    is far too small to judge a scatter plot. This label owns the source
    pixmap and rescales on every resize, so the plot grows with the dialog
    instead of the dialog growing around it.
    """

    clicked = pyqtSignal()

    def __init__(self, pix, parent=None):
        super().__init__(parent)
        self._src = pix
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)
        self.setStyleSheet(
            "background: palette(base); border: 1px solid palette(mid);")
        if pix is None or pix.isNull():
            self.setText("n/a")
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rescale()

    def _rescale(self):
        if self._src is None or self._src.isNull():
            return
        avail = self.size() - QSize(6, 6)      # stay inside the border
        if avail.width() < 16 or avail.height() < 16:
            return
        super().setPixmap(self._src.scaled(
            avail,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseReleaseEvent(self, event):
        if self._src is not None and not self._src.isNull():
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _WBDiagnosticsDialog(QDialog):
    """White balance diagnostics, one tab per filter: the star flux fit,
    and the numbers that say whether it landed.

    Only starry frames get an entry -- a starless frame is handed the
    starry frame's recipe and has no fit of its own to show.

    There used to be a second panel here showing the detected stars marked
    on the frame. It existed to troubleshoot star detection back when that
    was the weak link; with the auto-sigma waterfall and a clipped fit it
    told you nothing the statistics do not, and it cost more than half of
    every white-balance call to produce.
    """

    # Working copy cap for anything displayed in a tab. The entry keeps the
    # original pixmap for the detail viewer.
    DISPLAY_MAX = 1800

    def __init__(self, entries, tab):
        super().__init__(tab)
        self._tab = tab
        self.setWindowTitle("WB Diagnostics")
        self.setSizeGripEnabled(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._tabs = QTabWidget(self)
        self._shown = []
        for entry in entries:
            self.add_entry(entry)
        outer.addWidget(self._tabs, 1)

        foot = QHBoxLayout()
        hint = QLabel("Click the plot to open it in the detail viewer.")
        hint.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        foot.addWidget(hint)
        foot.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        foot.addWidget(btn_close)
        outer.addLayout(foot)

        self._restore_geometry()

    def add_entry(self, entry):
        """Add a filter's card, unless it is already on screen.

        Called both at construction and later, as the remaining filters
        finish -- the window is modeless, so it can be open while the rest
        of the run is still going.
        """
        name = str(entry.get("filter", ""))
        if name in self._shown:
            return
        self._shown.append(name)
        self._tabs.addTab(self._build_page(entry), name)

    # -- geometry ---------------------------------------------------------
    def _restore_geometry(self):
        geom = None
        try:
            geom = QSettings().value(_SKEY + "wb_diag_geometry", None)
        except Exception:
            geom = None
        if geom is not None:
            try:
                if self.restoreGeometry(geom):
                    return
            except Exception:
                pass
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            self.resize(min(1400, int(avail.width() * 0.85)),
                        min(900, int(avail.height() * 0.85)))
        except Exception:
            self.resize(1200, 800)

    def closeEvent(self, event):
        try:
            QSettings().setValue(_SKEY + "wb_diag_geometry",
                                 self.saveGeometry())
        except Exception:
            pass
        super().closeEvent(event)

    def done(self, result):
        try:
            QSettings().setValue(_SKEY + "wb_diag_geometry",
                                 self.saveGeometry())
        except Exception:
            pass
        super().done(result)

    # -- one filter -------------------------------------------------------
    def _display_copy(self, pix):
        if pix is None or pix.isNull():
            return None
        if max(pix.width(), pix.height()) <= self.DISPLAY_MAX:
            return pix
        return pix.scaled(self.DISPLAY_MAX, self.DISPLAY_MAX,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)

    def _build_page(self, entry):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        sub = QLabel(str(entry.get("subtitle", "")))
        sub.setStyleSheet("font-size:11px; color: palette(placeholderText);")
        lay.addWidget(sub)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        # left: the plot
        full = entry.get("scatter_pix")
        col = QWidget()
        col_l = QVBoxLayout(col)
        col_l.setContentsMargins(0, 0, 0, 0)
        col_l.setSpacing(4)
        head = QLabel("Star flux fit")
        head.setStyleSheet("font-size:11px; font-weight:500;")
        col_l.addWidget(head)
        img = _FitPixmapLabel(self._display_copy(full), col)
        if full is not None and not full.isNull():
            img.clicked.connect(lambda _=False, p=full: self._enlarge(p))
        col_l.addWidget(img, 1)
        cap = QLabel(
            "Measured NB flux against broadband flux, one point per star, "
            "before white balance. The red line is the fit; the pale wedge "
            "is the ±σ boundary it converged on; grey rings sit outside "
            "it, or were clipped high or at the noise floor.")
        cap.setWordWrap(True)
        cap.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        col_l.addWidget(cap)
        split.addWidget(col)

        # right: the numbers
        split.addWidget(self._build_stats(entry))
        split.setSizes([1100, 700])
        lay.addWidget(split, 1)
        return page

    # -- the statistics panel ---------------------------------------------
    @staticmethod
    def _pct(x, digits=1):
        return "—" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"

    @staticmethod
    def _num(x, digits=4):
        return "—" if not np.isfinite(x) else f"{x:.{digits}f}"

    def _build_stats(self, entry):
        """Everything the removed star-overlay panel was supposed to tell
        you, as numbers that can actually be read: did the white balance
        land, how tight is the star population, and how much of it was
        usable."""
        raw = entry.get("fit") or {}
        wb = entry.get("wb_fit") or {}

        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(4, 0, 0, 0)
        outer.setSpacing(6)

        head = QLabel("Fit quality")
        head.setStyleSheet("font-size:11px; font-weight:500;")
        outer.addWidget(head)

        verdict, why = (wb_fit_verdict(wb) if wb
                        else ("No balanced samples", ""))
        vb = QLabel(verdict)
        vb.setWordWrap(True)
        vb.setStyleSheet(
            "font-size:13px; font-weight:600; padding:4px 0px;")
        vb.setToolTip(why)
        outer.addWidget(vb)
        if why:
            wl = QLabel(why)
            wl.setWordWrap(True)
            wl.setStyleSheet(
                "font-size:10px; color: palette(placeholderText);")
            outer.addWidget(wl)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        grid.setColumnStretch(1, 1)
        row = [0]

        def section(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "font-size:10px; font-weight:600; color: "
                "palette(placeholderText); padding-top:6px;")
            grid.addWidget(lbl, row[0], 0, 1, 2)
            row[0] += 1

        def item(name, value, tip=""):
            k = QLabel(name)
            k.setStyleSheet("font-size:11px;")
            v = QLabel(str(value))
            v.setStyleSheet("font-size:11px; font-weight:600;")
            v.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            if tip:
                k.setToolTip(tip)
                v.setToolTip(tip)
            grid.addWidget(k, row[0], 0)
            grid.addWidget(v, row[0], 1, Qt.AlignmentFlag.AlignRight)
            row[0] += 1

        section("AFTER WHITE BALANCE")
        item("Slope", self._num(wb.get("slope", float("nan")), 4),
             "Balanced NB flux against balanced broadband flux. 1.0000 "
             "means the white balance landed exactly, and stars then "
             "cancel at Q = 1 by construction.")
        _r = star_residual_at_unit_q(wb) if wb else float("nan")
        item("Star residual at Q = 1",
             "\u2014" if not np.isfinite(_r) else f"{_r * 100:+.1f}%",
             "What an average star keeps of itself after subtraction at "
             "Q = 1 \u2014 positive means stars are left bright, negative "
             "means they are dug dark.\n\n"
             "This is a consequence, not an instruction to move Q. Q "
             "scales the continuum across the WHOLE image, so buying a "
             "cleaner star by raising it costs the same fraction of "
             "continuum everywhere else, line emission included. A "
             "residual here means the white balance did not fully land, "
             "and that is where to look.")
        item("Offset", self._num(wb.get("intercept", float("nan")), 5),
             "Intercept of the balanced fit. Should be ~0; a real offset "
             "means the two backgrounds did not neutralise to the same "
             "level.")
        item("Scatter", self._pct(wb.get("scatter", float("nan")), 2),
             "Robust (MAD) spread of the balanced stars about the line, "
             "as a fraction of flux. This is the floor on how cleanly "
             "stars can cancel: at 3% scatter, no single Q cancels every "
             "star.")
        item("R²", self._num(wb.get("r2", float("nan")), 4),
             "Coefficient of determination on the surviving stars. Reads "
             "high on any tight line -- use it to catch a fit that has "
             "gone loose, not as proof that it is right.")

        section("AS MEASURED (BEFORE WB)")
        item("Slope", self._num(raw.get("slope", float("nan")), 4),
             "The field's raw NB/BB star flux ratio -- how much dimmer "
             "your narrowband frame is than the continuum for an ordinary "
             "star. This is what the plot shows, and it is NOT expected "
             "to be 1.")
        item("Scatter", self._pct(raw.get("scatter", float("nan")), 2),
             "Robust spread about the raw line. Much larger than the "
             "balanced scatter usually means a colour term the WB has "
             "since taken out.")
        _lo = raw.get("x_lo", float("nan"))
        _hi = raw.get("x_hi", float("nan"))
        item("Flux span",
             "—" if not (np.isfinite(_lo) and np.isfinite(_hi))
             else f"{_lo:.3f} – {_hi:.3f}",
             "Broadband flux range the surviving stars cover. A fit "
             "anchored over a decade of flux is worth far more than one "
             "anchored over the faintest 5%.")

        section("STAR SAMPLE")
        n_tot = int(raw.get("n_total", entry.get("detected", 0)) or 0)
        n_fit = int(raw.get("n_fit", 0) or 0)
        item("Detected", f"{entry.get('detected', n_tot)}",
             "Stars SEP found and that survived the shape and spatial "
             "sampling filters.")
        sig = entry.get("sigma")
        item("Detection σ", "—" if not sig else f"{float(sig):g}",
             "The rung of the auto-σ ladder that produced them. A low "
             "value means the field was sparse enough to need a sensitive "
             "threshold.")
        item("Saturated", f"{int(raw.get('n_saturated', 0) or 0)}",
             f"Clipped at or above {FIT_CLIP_HI:g} in either axis -- no "
             "colour information left.")
        _fl = raw.get("floor")
        item("At noise floor", f"{int(raw.get('n_floor', 0) or 0)}",
             "Below this frame's own measured floor in either axis -- "
             f"background + {FIT_FLOOR_SNR:g}-sigma on the sampled "
             "aperture, which works out at "
             + ("not measured" if not _fl else
                f"{_fl[0]:.5f} (NB) and {_fl[1]:.5f} (broadband)")
             + ". Below it the flux ratio is noise. The floor is measured "
             "rather than fixed: a constant threshold in [0,1] units "
             "cuts almost every star on linear narrowband data and "
             "almost none on stretched data.")
        _rej = max(0, int(raw.get("n_hard", 0) or 0) - n_fit)
        item("σ-clipped outliers", f"{_rej}",
             "Stars with usable photometry that still fell outside the "
             "wedge: non-linear bright stars, blends, or genuine "
             "emission-line objects. A large number here is the "
             "interesting case.")
        item("In the fit",
             f"{n_fit}" + (f"  ({n_fit / n_tot * 100:.0f}% of detected)"
                           if n_tot else ""),
             "What the line was actually fitted to.")

        cov = raw.get("coverage") or {}
        _cells = int(cov.get("cells", 0) or 0)
        _used = int(cov.get("cells_used", 0) or 0)
        item("Regions covered",
             "not recorded" if not _cells else f"{_used} of {_cells}",
             f"The frame is divided into a {cov.get('grid', FIT_GRID)}x"
             f"{cov.get('grid', FIT_GRID)} grid. This is how many of those "
             "cells the fitted stars actually came from. A fit drawn from "
             "a corner of the frame can be perfectly tight and still not "
             "describe the rest of it.")
        _share = cov.get("max_share", float("nan"))
        item("Busiest region", self._pct(_share, 0),
             "Share of the fitted stars contributed by the single most "
             "crowded cell. Even coverage of a "
             f"{cov.get('grid', FIT_GRID)}x{cov.get('grid', FIT_GRID)} "
             f"grid would be about {100.0 / max(_cells, 1):.0f}%.")
        _thin = int(cov.get("thinned", 0) or 0)
        if _thin:
            item("Thinned for balance", f"{_thin}",
                 "Stars dropped from over-represented cells before "
                 "fitting, so no one region could decide the line. Within "
                 "a cell the ones kept are spread across the flux range.")

        outer.addLayout(grid)
        outer.addStretch(1)
        return panel

    def _enlarge(self, pix):
        try:
            self._tab._showEnlarged(pix)
        except Exception:
            pass


class ContinuumSubtractTab(QWidget):
    def __init__(self, doc_manager, document=None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.doc_manager = doc_manager
        self._loading_settings = False
        self.initUI()
        self._load_settings()
        self._threads = []
        # — initialize every loadable image to None —
        self.ha_image    = None
        self.sii_image   = None
        self.oiii_image  = None
        self.red_image   = None
        self.green_image = None
        self.osc_image   = None
        # NEW: composite HaO3 / S2O3 "source" images (optional)
        self.hao3_image          = None
        self.s2o3_image          = None
        self.hao3_starless_image = None
        self.s2o3_starless_image = None

        # NEW: OIII components extracted from composites (for averaging)
        self._o3_from_hao3          = None
        self._o3_from_s2o3          = None
        self._o3_from_hao3_starless = None
        self._o3_from_s2o3_starless = None
        self.filename = None
        self.is_mono = True
        self.combined_image = None
        self.processing_thread = None
        self.original_header = None
        self._clickable_images = {}
        self._wb_diag_entries = []   # list of dicts built in _onOneResult
        # Run-scoped state, reset by startContinuumSubtraction. Defined
        # here too so a result arriving outside a run (or a caller poking
        # _onOneResult directly) finds the attributes it expects.
        self._calib_entries = {}
        self._calib_threads = {}
        self._calib_sigma = {}
        self._results = []
        self._final_threads = []
        self._expected_results = 0
        self._wb_diag_dlg = None

    def initUI(self):
        self.spinnerLabel = QLabel("")
        self.spinnerLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinnerLabel.setStyleSheet("color:#999; font-style:italic;")
        self.spinnerLabel.hide()

        # starless image slots
        self.ha_starless_image    = None
        self.sii_starless_image   = None
        self.oiii_starless_image  = None
        self.red_starless_image   = None
        self.green_starless_image = None
        self.osc_starless_image   = None

        self.statusLabel = QLabel("")
        self.statusLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from PyQt6.QtWidgets import QGridLayout, QSizePolicy

        main_layout = QVBoxLayout()
        top_cols = QHBoxLayout()
        top_cols.setSpacing(8)

        # ── helper: build one filter block (starry + starless rows) ──────────
        def _filter_block(tag, attr, parent_layout, is_composite=False):
            """
            Adds two rows to parent_layout (a QVBoxLayout):
            row 1: [tag lbl] [filename lbl] [View btn] [File btn]   ← starry
            row 2: [+sl lbl] [filename lbl] [View btn] [File btn]   ← starless
            Wires View/File buttons to loadImage().
            Stores refs as self.<attr>Label, self.<attr>StarlessLabel etc.
            """
            for starless in (False, True):
                row = QHBoxLayout()
                row.setSpacing(4)
                row.setContentsMargins(0, 0, 0, 0)

                tag_lbl = QLabel("+starless" if starless else tag)
                tag_lbl.setFixedWidth(56)
                tag_lbl.setStyleSheet(
                    "font-size:10px; color: palette(placeholderText);" if starless
                    else "font-size:11px; font-weight:500;"
                )
                row.addWidget(tag_lbl)

                fname_lbl = QLabel("no file")
                fname_lbl.setStyleSheet("font-size:11px; font-style:italic; color: palette(placeholderText);")
                fname_lbl.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
                )
                fname_lbl.setMinimumWidth(0)
                row.addWidget(fname_lbl, 1)

                suffix = " (Starless)" if starless else ""
                channel = f"{tag}{suffix}"

                btn_view = QPushButton("View")
                btn_file = QPushButton("File")
                for b in (btn_view, btn_file):
                    b.setFixedWidth(36)
                    b.setStyleSheet("font-size:10px; padding: 1px 4px;")
                btn_view.clicked.connect(lambda _, c=channel: self._load_from_view(c))
                btn_file.clicked.connect(lambda _, c=channel: self._load_from_file(c))
                row.addWidget(btn_view)
                row.addWidget(btn_file)

                parent_layout.addLayout(row)

                # store refs
                attr_name = f"{attr}{'Starless' if starless else ''}Label"
                setattr(self, attr_name, fname_lbl)

            # thin divider after each block (caller adds it)

        # ── Narrowband group ─────────────────────────────────────────────────
        nb_group = QGroupBox("Narrowband Filters")
        nb_l = QVBoxLayout()
        nb_l.setSpacing(3)
        nb_l.setContentsMargins(8, 8, 8, 8)

        for tag, attr in [("Ha","ha"),("SII","sii"),("OIII","oiii"),("HaO3","hao3"),("S2O3","s2o3")]:
            _filter_block(tag, attr, nb_l)
            sep = QWidget(); sep.setFixedHeight(1)
            sep.setStyleSheet("background: palette(dark);")
            nb_l.addWidget(sep)

        nb_l.addStretch(1)
        nb_group.setLayout(nb_l)

        # ── Middle column: Continuum + Processing Options ─────────────────────
        mid_col = QVBoxLayout()
        mid_col.setSpacing(8)

        cont_group = QGroupBox("Continuum Sources")
        cont_l = QVBoxLayout()
        cont_l.setSpacing(3)
        cont_l.setContentsMargins(8, 8, 8, 8)

        for tag, attr in [("Red","red"),("Green","green"),("OSC","osc")]:
            _filter_block(tag, attr, cont_l)
            sep = QWidget(); sep.setFixedHeight(1)
            sep.setStyleSheet("background: palette(mid);")
            cont_l.addWidget(sep)

        cont_l.addStretch(1)
        cont_group.setLayout(cont_l)
        mid_col.addWidget(cont_group)

        # ── Processing Options ────────────────────────────────────────────────
        opts_group = QGroupBox("Processing Options")
        opts_l = QVBoxLayout()
        opts_l.setSpacing(2)
        opts_l.setContentsMargins(8, 8, 8, 8)

        self.linear_output_checkbox = QCheckBox("Output linear only")
        opts_l.addWidget(self.linear_output_checkbox)

        self.denoise_checkbox = QCheckBox("Denoise with Cosmic Clarity")
        self.denoise_checkbox.setChecked(True)
        self.denoise_checkbox.setToolTip(
            "Runs Cosmic Clarity denoise on the linear continuum-subtracted image "
            "before any non-linear stretch."
        )
        opts_l.addWidget(self.denoise_checkbox)

        # strength row (indented)
        str_row = QHBoxLayout()
        str_row.setContentsMargins(19, 0, 0, 0)
        str_row.setSpacing(6)
        self.denoise_strength_label = QLabel("Strength")
        self.denoise_strength_label.setStyleSheet("font-size:11px;")
        self.denoise_strength_btn = QPushButton("0.90")
        self.denoise_strength_btn.setFixedWidth(46)
        self.denoise_strength_btn.setStyleSheet("font-size:11px; padding:1px 4px;")
        self.denoise_strength_btn.clicked.connect(self._change_denoise_strength)
        self.denoise_strength = 0.90
        str_row.addWidget(self.denoise_strength_label)
        str_row.addWidget(self.denoise_strength_btn)
        str_row.addStretch(1)
        opts_l.addLayout(str_row)

        gpu_row = QHBoxLayout()
        gpu_row.setContentsMargins(19, 0, 0, 0)
        self.denoise_gpu_checkbox = QCheckBox("Use GPU")
        self.denoise_gpu_checkbox.setChecked(True)
        self.denoise_gpu_checkbox.setStyleSheet("font-size:11px;")
        self.denoise_gpu_checkbox.setToolTip("Uncheck to force CPU denoising (slower but avoids GPU memory issues)")
        gpu_row.addWidget(self.denoise_gpu_checkbox)
        gpu_row.addStretch(1)
        opts_l.addLayout(gpu_row)

        # wire enable/disable
        def _sync_denoise(checked):
            self.denoise_strength_btn.setEnabled(checked)
            self.denoise_strength_label.setEnabled(checked)
            self.denoise_gpu_checkbox.setEnabled(checked)
        self.denoise_checkbox.toggled.connect(_sync_denoise)
        _sync_denoise(True)

        # separator
        sep2 = QWidget(); sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: palette(mid);")
        opts_l.addWidget(sep2)
        opts_l.addSpacing(2)

        self.stretch_checkbox = QCheckBox("Auto-stretch")
        self.stretch_checkbox.setChecked(True)
        self.stretch_checkbox.setToolTip(
            "Stretch the linear result to non-linear. "
            "When unchecked, always outputs linear regardless of the output linear checkbox."
        )
        opts_l.addWidget(self.stretch_checkbox)

        # stretch sub-options widget (indented, disabled when stretch off)
        self.stretch_subopts = QWidget()
        sub_l = QVBoxLayout(self.stretch_subopts)
        sub_l.setContentsMargins(19, 0, 0, 0)
        sub_l.setSpacing(2)

        self.median_sub_checkbox = QCheckBox("Median background subtract")
        self.median_sub_checkbox.setChecked(True)
        self.median_sub_checkbox.setToolTip(
            "Subtracts 0.7× median after stretch. "
            "Uncheck to keep a cleaner background."
        )
        sub_l.addWidget(self.median_sub_checkbox)

        self.curves_checkbox = QCheckBox("Curves boost")
        self.curves_checkbox.setChecked(False)
        self.curves_checkbox.setToolTip("Applies a gentle S-curve boost after stretching.")
        sub_l.addWidget(self.curves_checkbox)

        self.no_black_clip_checkbox = QCheckBox("No black clip")
        self.no_black_clip_checkbox.setChecked(True)
        self.no_black_clip_checkbox.setToolTip(
            "Uses image minimum as black point instead of median − σ. "
            "Preserves faint background signal."
        )
        sub_l.addWidget(self.no_black_clip_checkbox)
        sub_l.addStretch(1)

        opts_l.addWidget(self.stretch_subopts)

        def _sync_stretch(checked):
            self.stretch_subopts.setEnabled(checked)
        self.stretch_checkbox.toggled.connect(_sync_stretch)
        _sync_stretch(True)

        # separator
        sep3 = QWidget(); sep3.setFixedHeight(1)
        sep3.setStyleSheet("background: palette(mid);")
        opts_l.addWidget(sep3)
        opts_l.addSpacing(2)

        self.target_stars = AUTO_STARS_DEFAULT

        # Star detection target. Q used to sit beside this behind an
        # "Advanced" disclosure; it is set in the Q Preview dialog now, and
        # one control does not need hiding.
        thr_row = QHBoxLayout()
        thr_row.setContentsMargins(0, 0, 0, 0)
        thr_row.setSpacing(6)
        self.threshold_label = QLabel("WB star target")
        self.threshold_label.setStyleSheet("font-size:11px;")
        self.threshold_btn = QPushButton(str(self.target_stars))
        self.threshold_btn.setFixedWidth(52)
        self.threshold_btn.setStyleSheet("font-size:11px; padding:1px 4px;")
        self.threshold_btn.clicked.connect(self._change_target_stars)
        _ladder = " → ".join(f"{v:g}" for v in AUTO_SIGMA_LADDER)
        _tip = (
            "How many stars the white-balance fit aims to work from.\n\n"
            "Detection sigma is not fixed: SEP is run down the ladder\n"
            f"σ = {_ladder} and the first rung that finds at least this\n"
            "many stars is the one used. If no rung reaches it, the most\n"
            "sensitive one is used.\n\n"
            "Raise it for a tighter fit on rich fields; lower it if a\n"
            "sparse field is driving detection down into the noise."
        )
        self.threshold_label.setToolTip(_tip)
        self.threshold_btn.setToolTip(_tip)
        thr_row.addWidget(self.threshold_label)
        thr_row.addWidget(self.threshold_btn)
        thr_row.addStretch(1)
        opts_l.addLayout(thr_row)
        opts_l.addStretch(1)

        opts_group.setLayout(opts_l)
        mid_col.addWidget(opts_group, 1)

        # ── WB Diagnostics (right column) ────────────────────────────────────


        # ── Assemble top columns ──────────────────────────────────────────────
        top_cols.addWidget(nb_group, 2)

        mid_widget = QWidget()
        mid_widget.setLayout(mid_col)
        top_cols.addWidget(mid_widget, 2)

        # ── Bottom row ────────────────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.startContinuumSubtraction)
        self.clear_button = QPushButton("Clear Images")
        self.clear_button.clicked.connect(self.clear_loaded_images)

        bottom_row.addWidget(self.execute_button)
        bottom_row.addWidget(self.clear_button)
        bottom_row.addWidget(self.spinnerLabel)
        bottom_row.addWidget(self.statusLabel, 1)
        # In bottom_row, after clear_button:
        self.wb_diag_button = QPushButton("WB Diagnostics…")
        self.wb_diag_button.clicked.connect(self._show_wb_diagnostics)
        self.wb_diag_button.setEnabled(False)  # enabled after first result
        bottom_row.addWidget(self.wb_diag_button)
        self.btn_help = make_help_button("continuum_subtract", self)
        bottom_row.addWidget(self.btn_help)        
        main_layout.addLayout(top_cols)
        main_layout.addLayout(bottom_row)
        self.setLayout(main_layout)
        self.installEventFilter(self)

        for _cb in (self.linear_output_checkbox, self.denoise_checkbox,
                    self.denoise_gpu_checkbox, self.stretch_checkbox,
                    self.median_sub_checkbox, self.curves_checkbox,
                    self.no_black_clip_checkbox):
            _cb.toggled.connect(lambda _=False: self._save_settings())

    # ── settings persistence ─────────────────────────────────────────────
    #
    # Every option in this panel is a per-user preference, not a per-image
    # one, so the tool comes back up the way it was left rather than at the
    # defaults. The checkboxes save on toggle; the three value controls save
    # in their own handlers; geometry and the Advanced disclosure save on
    # close.
    _OPT_CHECKS = (
        ("linear_output", "linear_output_checkbox"),
        ("denoise",       "denoise_checkbox"),
        ("denoise_gpu",   "denoise_gpu_checkbox"),
        ("stretch",       "stretch_checkbox"),
        ("median_sub",    "median_sub_checkbox"),
        ("curves",        "curves_checkbox"),
        ("no_black_clip", "no_black_clip_checkbox"),
    )

    def _load_settings(self):
        s = QSettings()
        self._loading_settings = True
        try:
            for key, attr in self._OPT_CHECKS:
                cb = getattr(self, attr, None)
                if cb is None:
                    continue
                # setChecked fires toggled, which runs the enable/disable
                # sync for the denoise and stretch sub-options.
                cb.setChecked(bool(s.value(_SKEY + key, cb.isChecked(),
                                           type=bool)))

            self.denoise_strength = float(
                s.value(_SKEY + "denoise_strength", self.denoise_strength,
                        type=float))
            self.denoise_strength_btn.setText(f"{self.denoise_strength:.2f}")

            self.target_stars = int(
                s.value(_SKEY + "target_stars", self.target_stars, type=int))
            self.threshold_btn.setText(str(self.target_stars))

            geom = s.value(_SKEY + "geometry", None)
            if geom is not None:
                try:
                    self.restoreGeometry(geom)
                except Exception:
                    pass
        except Exception:
            # A corrupt or half-written settings store must not stop the
            # tool from opening -- the defaults are already in place.
            pass
        finally:
            self._loading_settings = False

    def _save_settings(self):
        if getattr(self, "_loading_settings", False):
            return
        try:
            s = QSettings()
            for key, attr in self._OPT_CHECKS:
                cb = getattr(self, attr, None)
                if cb is not None:
                    s.setValue(_SKEY + key, bool(cb.isChecked()))
            s.setValue(_SKEY + "denoise_strength", float(self.denoise_strength))
            s.setValue(_SKEY + "target_stars", int(self.target_stars))
        except Exception:
            pass

    def _apply_loaded_image(self, channel: str, image, header, bit_depth, is_mono, name_or_path):
        """Common handler after loading from either View or File."""
        label_text = str(name_or_path) if name_or_path else "From View"
        try:
            if isinstance(name_or_path, str) and os.path.isabs(name_or_path):
                label_text = os.path.basename(name_or_path)
        except Exception:
            pass

        is_starless = "(Starless)" in channel
        base = channel.replace(" (Starless)", "")

        if base == "Ha":
            if is_starless:
                self.ha_starless_image = image
                self.haStarlessLabel.setText(label_text)
            else:
                self.ha_image = image
                self.haLabel.setText(label_text)

        elif base == "SII":
            if is_starless:
                self.sii_starless_image = image
                self.siiStarlessLabel.setText(label_text)
            else:
                self.sii_image = image
                self.siiLabel.setText(label_text)

        elif base == "OIII":
            if is_starless:
                self.oiii_starless_image = image
                self.oiiiStarlessLabel.setText(label_text)
            else:
                self.oiii_image = image
                self.oiiiLabel.setText(label_text)

        elif base == "Red":
            if is_starless:
                self.red_starless_image = image
                self.redStarlessLabel.setText(label_text)
            else:
                self.red_image = image
                self.redLabel.setText(label_text)

        elif base == "Green":
            if is_starless:
                self.green_starless_image = image
                self.greenStarlessLabel.setText(label_text)
            else:
                self.green_image = image
                self.greenLabel.setText(label_text)

        elif base == "OSC":
            if is_starless:
                self.osc_starless_image = image
                self.oscStarlessLabel.setText(label_text)
            else:
                self.osc_image = image
                self.oscLabel.setText(label_text)

        elif base == "HaO3":
            if is_starless:
                self.hao3_starless_image = image
                self.hao3StarlessLabel.setText(label_text)
            else:
                self.hao3_image = image
                self.hao3Label.setText(label_text)

            if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 2):
                QMessageBox.warning(self, "HaO3 Load",
                    "HaO3 expects a 3-channel color image (R=Ha, G=OIII). "
                    "Loaded image is not 3-channel; cannot extract Ha/OIII.")
                return

            img32 = image.astype(np.float32, copy=False)
            ha_from_r = img32[..., 0]
            o3_from_g = img32[..., 1]

            if is_starless:
                self.ha_starless_image = ha_from_r
                self.haStarlessLabel.setText(label_text + " [R → Ha (starless)]")
                self._o3_from_hao3_starless = o3_from_g
                self._update_oiii_from_composites(starless=True)
            else:
                self.ha_image = ha_from_r
                self.haLabel.setText(label_text + " [R → Ha]")
                self._o3_from_hao3 = o3_from_g
                self._update_oiii_from_composites(starless=False)

        elif base == "S2O3":
            if is_starless:
                self.s2o3_starless_image = image
                self.s2o3StarlessLabel.setText(label_text)
            else:
                self.s2o3_image = image
                self.s2o3Label.setText(label_text)

            if not (isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] >= 2):
                QMessageBox.warning(self, "S2O3 Load",
                    "S2O3 expects a 3-channel color image (R=SII, G=OIII). "
                    "Loaded image is not 3-channel; cannot extract SII/OIII.")
                return

            img32 = image.astype(np.float32, copy=False)
            s2_from_r = img32[..., 0]
            o3_from_g = img32[..., 1]

            if is_starless:
                self.sii_starless_image = s2_from_r
                self.siiStarlessLabel.setText(label_text + " [R → SII (starless)]")
                self._o3_from_s2o3_starless = o3_from_g
                self._update_oiii_from_composites(starless=True)
            else:
                self.sii_image = s2_from_r
                self.siiLabel.setText(label_text + " [R → SII]")
                self._o3_from_s2o3 = o3_from_g
                self._update_oiii_from_composites(starless=False)

        else:
            QMessageBox.critical(self, "Error", f"Unknown channel '{channel}'.")
            return

        self.original_header = header
        self.is_mono         = is_mono

    def _load_from_view(self, channel: str):
        result = self.loadImageFromView(channel)
        if result:
            self._apply_loaded_image(channel, *result)

    def _load_from_file(self, channel: str):
        result = self.loadImageFromFile(channel)
        if result:
            self._apply_loaded_image(channel, *result)

    def _change_denoise_strength(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Denoise Strength",
            "Cosmic Clarity denoise strength (0.0–1.0):",
            self.denoise_strength,
            0.0, 1.0, 2
        )
        if ok:
            self.denoise_strength = float(val)
            self.denoise_strength_btn.setText(f"{self.denoise_strength:.2f}")
            self._save_settings()

    def _change_target_stars(self):
        val, ok = QInputDialog.getInt(
            self,
            "WB Star Detection",
            "Stop lowering the detection σ once this many stars are found:",
            self.target_stars,
            20, 20000, 50
        )
        if ok:
            self.target_stars = int(val)
            self.threshold_btn.setText(str(self.target_stars))
            self._save_settings()


    def _main_window(self) -> QMainWindow | None:
        # 1) explicit parent the tool may have been created with
        mw = self.parent_window
        if mw and hasattr(mw, "mdi"):
            return mw
        # 2) walk up the parent chain
        p = self.parent()
        while p is not None:
            if hasattr(p, "mdi"):
                return p  # main window
            p = p.parent()
        # 3) search top-level widgets
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "mdi"):
                return w
        return None

    def _iter_open_docs(self):
        """Yield (doc, title) for all open subwindows."""
        mw = self._main_window()
        if not mw or not hasattr(mw, "mdi"):
            return []
        out = []
        for sw in mw.mdi.subWindowList():
            w = sw.widget()
            d = getattr(w, "document", None)
            if d is not None:
                out.append((d, sw.windowTitle()))
        return out


    def refresh(self):
        if self.image_manager:
            # You might have a way to retrieve the current image and metadata.
            # For example, if your image_manager stores the current image,
            # you could do something like:
            return



    def clear_loaded_images(self):
        for attr in (
            "ha_image","sii_image","oiii_image","red_image","green_image","osc_image",
            "ha_starless_image","sii_starless_image","oiii_starless_image",
            "red_starless_image","green_starless_image","osc_starless_image",
            # NEW composite attrs
            "hao3_image","s2o3_image",
            "hao3_starless_image","s2o3_starless_image"
        ):
            setattr(self, attr, None)

        # Reset NB labels
        self.haLabel.setText("No Ha")
        self.siiLabel.setText("No SII")
        self.oiiiLabel.setText("No OIII")
        # NEW composite labels
        self.hao3Label.setText("No HaO3")
        self.s2o3Label.setText("No S2O3")
        self._wb_diag_entries = []
        if hasattr(self, "wb_diag_button"):
            self.wb_diag_button.setEnabled(False)
        self._close_wb_diag()
        # Reset continuum labels
        self.redLabel.setText("No Red")
        self.greenLabel.setText("No Green")
        self.oscLabel.setText("No OSC")

        self.haStarlessLabel.setText("No Ha (starless)")
        self.siiStarlessLabel.setText("No SII (starless)")
        self.oiiiStarlessLabel.setText("No OIII (starless)")
        self.redStarlessLabel.setText("No Red (starless)")
        self.greenStarlessLabel.setText("No Green (starless)")
        self.oscStarlessLabel.setText("No OSC (starless)")

        # NEW: clear OIII-from-composite caches
        self._o3_from_hao3 = None
        self._o3_from_s2o3 = None
        self._o3_from_hao3_starless = None
        self._o3_from_s2o3_starless = None

        self.combined_image = None
        self.statusLabel.setText("All loaded images cleared.")


    # --- NEW: helper to combine OIII from HaO3 and S2O3 ---
    def _update_oiii_from_composites(self, starless: bool):
        """
        Average all available composite-derived green channels into a single OIII NB image.
        Only averages HaO3+S2O3 (does NOT touch any manually loaded OIII).
        """
        sources = []
        labels = []

        if starless:
            if self._o3_from_hao3_starless is not None:
                sources.append(self._o3_from_hao3_starless)
                labels.append("HaO3")
            if self._o3_from_s2o3_starless is not None:
                sources.append(self._o3_from_s2o3_starless)
                labels.append("S2O3")
        else:
            if self._o3_from_hao3 is not None:
                sources.append(self._o3_from_hao3)
                labels.append("HaO3")
            if self._o3_from_s2o3 is not None:
                sources.append(self._o3_from_s2o3)
                labels.append("S2O3")

        if not sources:
            return

        try:
            combo = np.mean(np.stack(sources, axis=0), axis=0).astype(np.float32, copy=False)
        except ValueError:
            # shape mismatch – fall back to last one
            combo = sources[-1]

        if starless:
            self.oiii_starless_image = combo
            base_label = "OIII from " + "+".join(labels) + " (starless)"
            self.oiiiStarlessLabel.setText(base_label)
        else:
            self.oiii_image = combo
            base_label = "OIII from " + "+".join(labels)
            self.oiiiLabel.setText(base_label)


    def _collect_open_documents(self):
        # kept for compatibility with callers; returns only docs
        return [d for d, _ in self._iter_open_docs()]

    def _select_document_via_dropdown(self, title: str):
        items = self._iter_open_docs()
        if not items:
            QMessageBox.information(self, f"Select View — {title}", "No open views/documents found.")
            return None

        # default to active view if present
        mw = self._main_window()
        active_doc = None
        if mw and mw.mdi.activeSubWindow():
            active_doc = getattr(mw.mdi.activeSubWindow().widget(), "document", None)

        if len(items) == 1:
            return items[0][0]

        names = [t for _, t in items]
        default_index = next((i for i, (d, _) in enumerate(items) if d is active_doc), 0)

        choice, ok = QInputDialog.getItem(
            self, f"Select View — {title}", "Choose:", names, default_index, False
        )
        if not ok:
            return None
        return items[names.index(choice)][0]

    def _image_from_doc(self, doc):
        """(np.ndarray, header, bit_depth, is_mono, file_path) from an ImageDocument."""
        arr = getattr(doc, "image", None)
        if arr is None:
            QMessageBox.warning(self, "No image", "Selected view has no image.")
            return None
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = False
        try:
            import numpy as np
            is_mono = isinstance(arr, np.ndarray) and (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1))
        except Exception:
            pass
        return arr, header, bit_depth, is_mono, meta.get("file_path")

    def loadImageFromView(self, channel: str):
        doc = self._select_document_via_dropdown(channel)
        if not doc:
            return None
        res = self._image_from_doc(doc)
        if not res:
            return None

        img, header, bit_depth, is_mono, _ = res

        # Build a human-friendly name for the label (view/subwindow title)
        title = ""
        try:
            title = doc.display_name()
        except Exception:
            mw = self._main_window()
            if mw and mw.mdi.activeSubWindow():
                title = mw.mdi.activeSubWindow().windowTitle()

        # Return with the "path" field set to the title so the caller can label it
        return img, header, bit_depth, is_mono, title


    def loadImageFromFile(self, channel: str):
        file_filter = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {channel} Image", "", file_filter)
        if not path:
            return None
        try:
            image, header, bit_depth, is_mono = legacy_load_image(path)  # ← use the alias
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {channel} image:\n{e}")
            return None
        return image, header, bit_depth, is_mono, path

    def startContinuumSubtraction(self):
        cont_red_starry     = self.red_image   if self.red_image   is not None else (self.osc_image[..., 0]         if self.osc_image         is not None else None)
        cont_green_starry   = self.green_image if self.green_image is not None else (self.osc_image[..., 1]         if self.osc_image         is not None else None)
        cont_red_starless   = self.red_starless_image   if self.red_starless_image   is not None else (self.osc_starless_image[..., 0] if self.osc_starless_image is not None else None)
        cont_green_starless = self.green_starless_image if self.green_starless_image is not None else (self.osc_starless_image[..., 1] if self.osc_starless_image is not None else None)

        pairs = []
        def add_pair(name, nb_starry, cont_starry, nb_starless, cont_starless):
            has_starry   = (nb_starry   is not None and cont_starry   is not None)
            has_starless = (nb_starless is not None and cont_starless is not None)
            if has_starry or has_starless:
                pairs.append({
                    "name": name,
                    "nb": nb_starry, "cont": cont_starry,
                    "nb_sl": nb_starless, "cont_sl": cont_starless,
                    "starless_only": (has_starless and not has_starry),
                })

        add_pair("Ha",   self.ha_image,   cont_red_starry,   self.ha_starless_image,   cont_red_starless)
        add_pair("SII",  self.sii_image,  cont_red_starry,   self.sii_starless_image,  cont_red_starless)
        add_pair("OIII", self.oiii_image, cont_green_starry, self.oiii_starless_image, cont_green_starless)

        if not pairs:
            self.statusLabel.setText("Load at least one NB + matching continuum channel (or OSC).")
            return

        self.showSpinner()
        self._threads          = []
        self._calib_entries    = {}   # name -> entry dict
        self._calib_threads    = {}   # name -> ContinuumProcessingThread
        self._calib_pending    = 0
        self._final_threads    = []   # kept alive
        self._results          = []
        self._pushed_results   = False
        self._wb_diag_entries  = []
        self._calib_sigma      = {}   # name -> sigma the waterfall settled on
        self._close_wb_diag()         # its cards describe the previous run

        # How many final result signals to expect (set after Q accepted)
        self._expected_results = 0

        denoise_linear   = self.denoise_checkbox.isChecked()
        denoise_strength = self.denoise_strength
        denoise_gpu      = self.denoise_gpu_checkbox.isChecked()
        do_stretch       = self.stretch_checkbox.isChecked()
        do_median_sub    = self.median_sub_checkbox.isChecked()
        do_curves        = self.curves_checkbox.isChecked()
        no_black_clip    = self.no_black_clip_checkbox.isChecked()

        for p in pairs:
            name = p["name"]
            t = ContinuumProcessingThread(
                p["nb"], p["cont"], self.linear_output_checkbox.isChecked(),
                starless_nb=p["nb_sl"], starless_cont=p["cont_sl"],
                starless_only=p["starless_only"],
                auto_min_stars=self.target_stars,
                denoise_linear=denoise_linear,
                denoise_strength=denoise_strength,
                denoise_gpu=denoise_gpu,
                do_stretch=do_stretch,
                do_median_sub=do_median_sub,
                do_curves=do_curves,
                no_black_clip=no_black_clip,
            )

            t.calibration_ready.connect(self._onCalibrationReady)
            t.processing_complete.connect(
                lambda img, stars, raw, after, n=name:
                    self._onOneResult(n, img, stars, raw, after)
            )
            t.processing_complete_starless.connect(
                lambda img, stars, raw, after, n=f"{name} (starless)":
                    self._onOneResult(n, img, stars, raw, after)
            )
            t.status_update.connect(self.update_status_label)

            self._calib_threads[name] = t
            self._calib_pending += 1

            # kick off calibration via a dedicated thread
            calib_runner = _CalibrationRunner(t, name)
            calib_runner.finished.connect(lambda r=calib_runner: self._threads.remove(r) if r in self._threads else None)
            self._threads.append(calib_runner)
            calib_runner.start()

    def _onAllCalibrationsReady(self):
        self.hideSpinner()
        entries = list(self._calib_entries.values())

        dlg = QPreviewDialog(entries, parent=self)
        dlg.accepted_q.connect(self._onQAccepted)
        self._q_preview_dlg = dlg   # prevent GC
        dlg.show()

        # Open the diagnostics beside it. Choosing Q without them means
        # judging a subtraction by eye while the numbers that describe the
        # fit sit behind a button you have not pressed yet.
        if self._wb_diag_entries:
            self._show_wb_diagnostics()
            self._tile_with_preview(dlg, getattr(self, "_wb_diag_dlg", None))

    @staticmethod
    def _tile_with_preview(q_dlg, diag_dlg):
        """Put the Q dialog and the diagnostics side by side.

        Only on this paired open: a diagnostics window the user has
        already sized and placed is left where it is when opened on its
        own from the button.
        """
        if q_dlg is None or diag_dlg is None:
            return
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            gap = 8
            total = avail.width() - 3 * gap
            if total < 900:
                return                      # too narrow to tile usefully
            q_w = max(520, min(q_dlg.width(), int(total * 0.52)))
            d_w = total - q_w
            if d_w < 420:
                return
            h = min(max(q_dlg.height(), diag_dlg.height()),
                    avail.height() - 2 * gap)
            y = avail.y() + gap
            q_dlg.setGeometry(avail.x() + gap, y, q_w, h)
            diag_dlg.setGeometry(avail.x() + gap + q_w + gap, y, d_w, h)
        except Exception:
            pass

    def _onQAccepted(self, q_values: dict):
        """User clicked Accept in QPreviewDialog — run final subtracts."""
        self.showSpinner()
        self._results          = []
        self._pushed_results   = False
        self._wb_diag_entries  = []

        # count expected signals
        self._expected_results = 0
        for name, entry in self._calib_entries.items():
            recipe = entry["recipe"]
            is_sl  = recipe.get("starless_only", False)
            t      = self._calib_threads.get(name)
            if t is None:
                continue
            # starry result (or starless-only result)
            self._expected_results += 1
            # paired starless
            if not is_sl and t.starless_nb is not None and t.starless_cont is not None:
                self._expected_results += 1

        for name, entry in self._calib_entries.items():
            q = q_values.get(name, DEFAULT_Q)
            t = self._calib_threads.get(name)
            if t is None:
                continue
            ft = FinalSubtractThread(t, name, entry, q)
            ft.finished.connect(lambda f=ft: self._final_threads.remove(f) if f in self._final_threads else None)
            self._final_threads.append(ft)
            ft.start()

    def _onCalibrationReady(self, entry: dict):
        name = entry["name"]
        self._calib_entries[name] = entry
        if not hasattr(self, "_calib_sigma"):
            self._calib_sigma = {}
        self._calib_sigma[name] = entry.get("used_sigma")

        # Cards now, not after the final subtract -- they are what the Q
        # decision should be made against.
        _rec = entry.get("recipe") or {}
        self._build_diag_card(name, int(entry.get("star_count", 0) or 0),
                              _rec.get("fit_raw"), _rec.get("fit_after"))

        self._calib_pending -= 1
        self.update_status_label(f"Calibration done for {name}.")
        if self._calib_pending == 0:
            self._onAllCalibrationsReady()

    def _onOneResult(self, filt, img, star_count, raw_pixels, after_pixels):
        try:
            # quick sip check — if self has been destroyed, any attr access will raise
            _ = self.wb_diag_button
        except RuntimeError:
            return
        self._results.append({
            "filter": filt, "image": img, "stars": star_count,
            "raw": raw_pixels, "after": after_pixels
        })

        if (not getattr(self, "_pushed_results", False)
                and len(self._results) == getattr(self, "_expected_results", 0)):
            self._pushed_results = True
            self.hideSpinner()
            self._pushResultsToDocs(self._results)

    def _build_diag_card(self, filt, star_count, raw_pixels,
                         after_pixels):
        """Build one filter's diagnostics card and register it.

        Called from _onCalibrationReady, i.e. BEFORE the Q Preview
        dialog opens -- the whole point of these numbers is to be
        readable while Q is being chosen. The final subtract adds
        nothing to them: the star samples, centroids and measured
        floors are all fixed by calibration.
        """
        try:
            _ = self.wb_diag_button
        except RuntimeError:
            return
        if any(e.get("filter") == filt
               for e in self._wb_diag_entries):
            return                      # already have this filter

        # store full-res pixmaps for the diagnostics dialog
        make_scatter = (
            isinstance(raw_pixels, np.ndarray) and
            raw_pixels.ndim == 2 and raw_pixels.shape[1] >= 2 and
            raw_pixels.shape[0] >= 3 and cv2 is not None
        )

        # The recipe holds what the sample arrays cannot carry: where each
        # star was, and the floor measured on that frame.
        _rec = (((getattr(self, "_calib_entries", None) or {}).get(
            filt.replace(" (starless)", "")) or {}).get("recipe") or {})
        _xy = _rec.get("fit_xy")
        _shape = _rec.get("fit_shape")
        _floor = _rec.get("fit_floor", FIT_CLIP_LO)
        _floor_after = _rec.get("fit_floor_after", FIT_CLIP_LO)

        scatter_pix = None
        fit_info = {}
        if make_scatter:
            nb_flux   = raw_pixels[:, 0].astype(np.float32, copy=False)
            cont_flux = raw_pixels[:, 1].astype(np.float32, copy=False)
            # 512, not 256: at 256 the points sat on top of one another and
            # the axis type was unreadable once the card scaled it down.
            h, w = 512, 512
            _sc = w / 256.0          # scale every drawn feature with it
            # NOTE: this buffer is handed to QImage.Format_RGB888 further
            # down, so these triples are read as RGB -- not as cv2's usual
            # BGR. Write them in RGB order or the plot lies about itself.
            _C_FIT  = (210, 40, 40)     # the fit line
            _C_KEPT = (40, 80, 200)     # a star that went into the fit
            _C_DROP = (150, 150, 150)   # rejected: clipped or noise floor
            _C_AXIS = (0, 0, 0)
            scatter_img = np.ones((h, w, 3), np.uint8) * 255

            raw_stats = fit_star_flux(nb_flux, cont_flux, clip_lo=_floor,
                                      xy=_xy, shape=_shape)
            # Guide lines drawn at the bounds the fit actually applied.
            CLIP_HI = FIT_CLIP_HI
            CLIP_LO = (raw_stats.get("floor") or (FIT_CLIP_LO,))[0]
            fit_info = raw_stats
            keep_mask = raw_stats["keep"]
            band = raw_stats["band"]
            m, c = raw_stats["slope"], raw_stats["intercept"]
            try:
                if not raw_stats["ok"]:
                    raise ValueError("fit did not converge")
                print(f"[ContSub/{filt}] scatter fit: {raw_stats['n_fit']}/"
                      f"{raw_stats['n_hard']} stars after {FIT_SIGMA}-sigma "
                      f"clipping ({raw_stats['n_total'] - raw_stats['n_hard']}"
                      f" hard-clipped), slope={m:.4f} intercept={c:.4f}, "
                      f"scatter={raw_stats['scatter'] * 100:.2f}%")

                x0 = 0;      y0 = int((1 - float(np.clip(c,     0, 1))) * (h-1))
                x1 = w - 1;  y1 = int((1 - float(np.clip(m + c, 0, 1))) * (h-1))
                cv2.line(scatter_img, (x0, y0), (x1, y1), _C_FIT,
                         max(1, int(round(2 * _sc))), cv2.LINE_AA)
                # Draw clip boundaries so the rejected zone is obvious.
                clip_y = int((1 - CLIP_HI) * (h-1))
                clip_x = int(CLIP_HI * (w-1))
                cv2.line(scatter_img, (0, clip_y), (w-1, clip_y),
                        (205, 205, 205), max(1, int(round(_sc))), cv2.LINE_AA)
                cv2.line(scatter_img, (clip_x, 0), (clip_x, h-1),
                        (205, 205, 205), max(1, int(round(_sc))), cv2.LINE_AA)
                if band is not None:
                    _k, _xref = band
                    # Not a straight line: constant width below x_ref, a
                    # wedge above it. Drawn as a polyline so it is the real
                    # boundary and not an approximation of it.
                    _xs = np.linspace(0.0, 1.0, 48)
                    _off = _k * np.maximum(_xs, _xref)
                    for sgn in (-1.0, 1.0):
                        _ys = m * _xs + c + sgn * _off
                        pts = np.stack([
                            (_xs * (w - 1)).astype(np.int32),
                            ((1 - np.clip(_ys, 0, 1)) * (h - 1)).astype(
                                np.int32)], axis=1)
                        cv2.polylines(scatter_img, [pts], False,
                                      (165, 165, 195),
                                      max(1, int(round(_sc))), cv2.LINE_AA)
            except Exception:
                keep_mask = np.ones_like(nb_flux, dtype=bool)

            xs = (np.clip(cont_flux, 0, 1) * (w-1)).astype(int)
            ys = ((1 - np.clip(nb_flux, 0, 1)) * (h-1)).astype(int)
            _r_keep = max(2, int(round(1.6 * _sc)))
            _r_drop = max(2, int(round(2.2 * _sc)))
            for x, y, kept in zip(xs, ys, keep_mask):
                if 0 <= x < w and 0 <= y < h:
                    if kept:
                        cv2.circle(scatter_img, (x, y), _r_keep, _C_KEPT, -1,
                                   cv2.LINE_AA)
                    else:
                        cv2.circle(scatter_img, (x, y), _r_drop, _C_DROP,
                                   max(1, int(round(_sc))), cv2.LINE_AA)
            _ax = max(1, int(round(_sc)))
            cv2.line(scatter_img, (0, h-1), (w-1, h-1), _C_AXIS, _ax)
            cv2.line(scatter_img, (0, 0),   (0, h-1),   _C_AXIS, _ax)
            # Axis type is sized for the plot, not scaled with it -- at a
            # full 2x it swamped the data.
            font = cv2.FONT_HERSHEY_SIMPLEX
            _fs = 0.62
            _ft = 2
            ((tw, th), _) = cv2.getTextSize("BB Flux", font, _fs, _ft)
            cv2.putText(scatter_img, "BB Flux", ((w-tw)//2, h - int(8*_sc)),
                        font, _fs, _C_AXIS, _ft, cv2.LINE_AA)
            _step = int(th * 1.45)
            for i, ch in enumerate("NB Flux"):
                cv2.putText(scatter_img, ch,
                            (int(4*_sc), int(10*_sc) + th + i*_step),
                            font, _fs, _C_AXIS, _ft, cv2.LINE_AA)
            qscatter = QImage(scatter_img.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
            scatter_pix = QPixmap.fromImage(qscatter)

        # The plot above is drawn from the RAW samples, so its slope is the
        # field's measured NB/BB ratio. The balanced samples are the ones
        # that should land on y = x, and that is the quality number.
        wb_stats = None
        if (isinstance(after_pixels, np.ndarray) and after_pixels.ndim == 2
                and after_pixels.shape[1] >= 2 and after_pixels.shape[0] >= 3):
            wb_stats = fit_star_flux(after_pixels[:, 0].astype(np.float32),
                                     after_pixels[:, 1].astype(np.float32),
                                     clip_lo=_floor_after,
                                     xy=_xy, shape=_shape)

        # Only the starry frame has a star-based white balance to diagnose.
        # A starless frame is handed the starry frame's recipe, and a
        # starless-ONLY run never fits stars at all -- both produced a card
        # with a blank overlay, an "n/a" scatter and nothing to look at.
        is_starless = "(starless)" in filt
        if not is_starless and int(star_count) > 0:
            base = filt.replace(" (starless)", "")
            sig = (getattr(self, "_calib_sigma", {}) or {}).get(base)
            if wb_stats is not None and wb_stats["ok"]:
                sub = (f"{wb_stats['n_fit']} of {wb_stats['n_total']} stars "
                       f"in the fit  ·  balanced slope "
                       f"{wb_stats['slope']:.3f}  ·  "
                       f"{wb_fit_verdict(wb_stats)[0]}")
            elif fit_info and fit_info.get("ok"):
                sub = (f"{fit_info['n_fit']} of {fit_info['n_total']} stars "
                       f"in the fit")
            else:
                sub = f"{star_count} stars detected"
            if sig:
                sub += f"  ·  σ = {float(sig):g}"
            self._wb_diag_entries.append({
                "filter":      filt,
                "subtitle":    sub,
                "fit":         fit_info,
                "wb_fit":      wb_stats,
                "sigma":       sig,
                "detected":    int(star_count),
                "scatter_pix": scatter_pix,   # 512×512, or None
            })

        try:
            if hasattr(self, "wb_diag_button"):
                self.wb_diag_button.setEnabled(bool(self._wb_diag_entries))
        except RuntimeError:
            return  # widget was deleted

        # If the panel is already open, hand it this filter rather than
        # making the user close and reopen to see it.
        _dlg = getattr(self, "_wb_diag_dlg", None)
        if _dlg is not None and self._wb_diag_entries:
            try:
                _dlg.add_entry(self._wb_diag_entries[-1])
            except RuntimeError:
                self._wb_diag_dlg = None


    def _show_wb_diagnostics(self):
        if not self._wb_diag_entries:
            QMessageBox.information(
                self, "WB Diagnostics",
                "Nothing to show yet — run a subtraction on a starry frame "
                "first. Starless-only runs never fit stars, so they have no "
                "white balance to diagnose."
            )
            return
        dlg = getattr(self, "_wb_diag_dlg", None)
        if dlg is not None:
            try:
                # Already open: bring it forward rather than stacking a
                # second copy of the same cards.
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
                return
            except RuntimeError:
                self._wb_diag_dlg = None

        # Modeless, so the rest of SASpro stays usable while it is open --
        # the point of the panel is to compare it against the result.
        # WA_DeleteOnClose would be wrong here for the same reason: the tab
        # keeps a reference and clears it on destroyed().
        dlg = _WBDiagnosticsDialog(self._wb_diag_entries, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._wb_diag_dlg = dlg
        dlg.destroyed.connect(self._on_wb_diag_destroyed)
        dlg.show()
        dlg.raise_()

    def _on_wb_diag_destroyed(self, *_):
        self._wb_diag_dlg = None

    def _close_wb_diag(self):
        """Close the diagnostics panel if one is open.

        Called when the cards it is showing stop being true: a new run, a
        Clear Images, or the tool closing.
        """
        dlg = getattr(self, "_wb_diag_dlg", None)
        self._wb_diag_dlg = None
        if dlg is not None:
            try:
                dlg.close()
            except RuntimeError:
                pass

    def eventFilter(self, source, event):
        # catch mouse releases on any of our clickable labels
        if event.type() == QEvent.Type.MouseButtonRelease and source in self._clickable_images:
            pix = self._clickable_images[source]
            self._showEnlarged(pix)
            return True
        return super().eventFilter(source, event)

    def _showEnlarged(self, pixmap: QPixmap):
        """
        Detail View dialog with themed zoom controls, autostretch, and 'push to document'.
        Uses ZoomableGraphicsView + QGraphicsScene (consistent with CLAHE).
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
            QLabel, QPushButton, QGraphicsScene, QGraphicsPixmapItem
        )


        from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
        from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
        from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image

        # ---------- helpers ----------
        def _float01_from_qimage(qimg: QImage) -> np.ndarray:
            """QImage -> float32 [0..1]. Handles Grayscale8 and RGB888; converts others to RGB888."""
            if qimg is None or qimg.isNull():
                return np.zeros((1, 1), dtype=np.float32)

            fmt = qimg.format()
            if fmt not in (QImage.Format.Format_Grayscale8, QImage.Format.Format_RGB888):
                qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
                fmt = QImage.Format.Format_RGB888

            h = qimg.height()
            w = qimg.width()
            bpl = qimg.bytesPerLine()

            ptr = qimg.bits()
            ptr.setsize(h * bpl)
            buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))

            if fmt == QImage.Format.Format_Grayscale8:
                return (buf[:, :w].astype(np.float32) / 255.0).clip(0.0, 1.0)

            # RGB888
            rgb = buf[:, :w * 3].reshape((h, w, 3)).astype(np.float32) / 255.0
            return rgb.clip(0.0, 1.0)

        def _qimage_from_float01(arr: np.ndarray) -> QImage:
            """float32 [0..1] -> QImage (RGB888 or Grayscale8), deep-copied."""
            a = np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)

            if a.ndim == 2:
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w = u8.shape
                q = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
                return q.copy()

            if a.ndim == 3 and a.shape[2] == 1:
                return _qimage_from_float01(a[..., 0])

            if a.ndim == 3 and a.shape[2] == 3:
                u8 = (a * 255.0 + 0.5).astype(np.uint8, copy=False)
                h, w, _ = u8.shape
                q = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                return q.copy()

            # fallback
            raise ValueError(f"Unexpected image shape: {a.shape}")

        def _pixmap_from_float01(arr: np.ndarray) -> QPixmap:
            return QPixmap.fromImage(_qimage_from_float01(arr))

        # ---------- dialog ----------
        dlg = QDialog(self)
        dlg.setWindowTitle("Detail View")
        dlg.resize(980, 820)

        outer = QVBoxLayout(dlg)

        # Convert input pixmap -> float01 working buffer
        try:
            base_qimg = pixmap.toImage()
            current_arr = _float01_from_qimage(base_qimg)
        except Exception:
            current_arr = np.zeros((1, 1), dtype=np.float32)

        # Ensure "display" is always RGB for the pixmap item (GraphicsView)
        def _ensure_rgb(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 2:
                return np.stack([a, a, a], axis=-1)
            if a.ndim == 3 and a.shape[2] == 1:
                return np.repeat(a, 3, axis=2)
            return a

        # Graphics view
        scene = QGraphicsScene(dlg)
        view = ZoomableGraphicsView(scene)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pix_item = QGraphicsPixmapItem()
        scene.addItem(pix_item)
        outer.addWidget(view, stretch=1)

        def _set_scene_from_arr(arr01: np.ndarray):
            rgb = _ensure_rgb(arr01)
            pm = _pixmap_from_float01(rgb)
            pix_item.setPixmap(pm)
            scene.setSceneRect(pix_item.boundingRect())

        _set_scene_from_arr(current_arr)

        # Toolbar row (themed zoom helper)
        row = QHBoxLayout()

        btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        btn_zoom_in = themed_toolbtn("zoom-in", "Zoom In")
        btn_zoom_1to1 = themed_toolbtn("zoom-original", "1:1 (100%)")
        btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        btn_zoom_out.clicked.connect(view.zoom_out)
        btn_zoom_in.clicked.connect(view.zoom_in)
        btn_zoom_1to1.clicked.connect(view.one_to_one if hasattr(view, "one_to_one") else (lambda: None))
        btn_zoom_fit.clicked.connect(lambda: view.fit_to_item(pix_item))

        row.addWidget(btn_zoom_out)
        row.addWidget(btn_zoom_in)
        row.addWidget(btn_zoom_1to1)
        row.addWidget(btn_zoom_fit)

        row.addStretch(1)

        btn_autostretch = QPushButton("Autostretch")
        row.addWidget(btn_autostretch)

        btn_push = QPushButton("Push to New Document")
        row.addWidget(btn_push)

        btn_close = QPushButton("Close")
        row.addWidget(btn_close)

        outer.addLayout(row)

        # Actions
        no_black_clip = self.no_black_clip_checkbox.isChecked()

        def _do_autostretch():
            nonlocal current_arr
            try:
                a = np.asarray(current_arr, dtype=np.float32)
                if a.ndim == 2:
                    stretched = stretch_mono_image(a, target_median=0.25,
                                                   no_black_clip=no_black_clip)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
                elif a.ndim == 3 and a.shape[2] == 1:
                    stretched = stretch_mono_image(a[..., 0], target_median=0.25,
                                                   no_black_clip=no_black_clip)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
                else:
                    stretched = stretch_color_image(a, target_median=0.25, linked=False,
                                                    no_black_clip=no_black_clip)
                    current_arr = np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)
                _set_scene_from_arr(current_arr)
            except Exception as e:
                QMessageBox.warning(dlg, "Detail View", f"Autostretch failed:\n{e}")

        def _do_push_to_doc():
            dm = getattr(self, "doc_manager", None)
            mw = self._main_window() if hasattr(self, "_main_window") else None

            if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
                QMessageBox.critical(dlg, "Detail View", "Cannot create document: missing DocManager or MainWindow.")
                return

            try:
                img = np.asarray(current_arr, dtype=np.float32)

                # Preserve mono where appropriate
                is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

                counter = getattr(self, "_detail_doc_counter", 0) + 1
                self._detail_doc_counter = counter
                name = f"DetailView_{counter}"

                meta = {
                    "display_name": name,
                    "file_path": name,
                    "bit_depth": "32-bit floating point",
                    "is_mono": bool(is_mono),
                    "original_header": getattr(self, "original_header", None),
                    "source": "Continuum Subtract — Detail View",
                }

                doc = dm.create_document(img, metadata=meta, name=name)
                mw._spawn_subwindow_for(doc)

                try:
                    if hasattr(self, "statusLabel") and self.statusLabel is not None:
                        self.statusLabel.setText(f"Pushed detail view → '{name}'.")
                except Exception:
                    pass

            except Exception as e:
                QMessageBox.critical(dlg, "Detail View", f"Failed to create document:\n{e}")

        btn_autostretch.clicked.connect(_do_autostretch)
        btn_push.clicked.connect(_do_push_to_doc)
        btn_close.clicked.connect(dlg.accept)

        # Initial fit
        try:
            view.fit_to_item(pix_item)
        except Exception:
            pass

        dlg.exec()


    def _onThreadFinished(self):
        self._pending -= 1
        if self._pending == 0:
            self.hideSpinner()
            self._pushResultsToDocs(self._results) 

    def _pushResultsToDocs(self, results):
        dm = getattr(self, "doc_manager", None)
        mw = self._main_window()
        if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
            QMessageBox.critical(self, "Continuum Subtract",
                                "Cannot create documents: missing DocManager or MainWindow.")
            return

        created = 0
        for entry in results:
            filt = entry["filter"]
            img  = np.asarray(entry["image"], dtype=np.float32)  # keep everything float32
            name = f"{filt}_ContSub"

            meta = {
                "display_name": name,                 # nice title in the UI
                "file_path": name,                    # placeholder path until user saves
                "bit_depth": "32-bit floating point",
                "is_mono": (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)),
                "original_header": self.original_header,
                "source": "Continuum Subtract",
            }

            try:
                # create a proper ImageDocument and register it
                doc = dm.create_document(img, metadata=meta, name=name)
                # show it as an MDI subwindow
                mw._spawn_subwindow_for(doc)
                created += 1
            except Exception as e:
                QMessageBox.critical(self, "Continuum Subtract",
                                    f"Failed to create document '{name}':\n{e}")

        self.statusLabel.setText(f"Created {created} document(s).")


    def _onThreadFinished(self):
        self._pending -= 1
        # do NOT push here if you already push in _onOneResult

    def update_status_label(self, message):
        self.statusLabel.setText(message)

    def showSpinner(self):
        self.spinnerLabel.setText("Processing…")
        self.spinnerLabel.show()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(False)

    def hideSpinner(self):
        self.spinnerLabel.hide()
        self.spinnerLabel.clear()
        if hasattr(self, "execute_button"):
            self.execute_button.setEnabled(True)

    def closeEvent(self, event):
        try:
            self._save_settings()
            QSettings().setValue(_SKEY + "geometry", self.saveGeometry())
        except Exception:
            pass
        # Parented here, so Qt would destroy it with us anyway -- close it
        # explicitly so it saves its own geometry first.
        self._close_wb_diag()
        # Disconnect all threads to prevent signals firing into a dead widget
        for t in list(getattr(self, '_calib_threads', {}).values()):
            try:
                t.calibration_ready.disconnect()
                t.processing_complete.disconnect()
                t.processing_complete_starless.disconnect()
                t.status_update.disconnect()
            except Exception:
                pass
        for t in list(getattr(self, '_final_threads', [])):
            try:
                t.finished.disconnect()
            except Exception:
                pass
        super().closeEvent(event)

class ContinuumProcessingThread(QThread):
    # emitted once per filter when calibration is done (pre-subtraction)
    calibration_ready = pyqtSignal(dict)          # one entry dict per filter
    # emitted once per filter for the final result (post Q-accept)
    # (result, star_count, raw_star_pixels, after_star_pixels)
    processing_complete          = pyqtSignal(np.ndarray, int, np.ndarray, np.ndarray)
    processing_complete_starless = pyqtSignal(np.ndarray, int, np.ndarray, np.ndarray)
    status_update = pyqtSignal(str)

    # Max pixels on the long axis for the interactive preview
    PREVIEW_MAX = 1024

    def __init__(self, nb_image, continuum_image, output_linear, *,
                 starless_nb=None, starless_cont=None, starless_only=False,
                 auto_min_stars: int = AUTO_STARS_DEFAULT,
                 auto_ladder=AUTO_SIGMA_LADDER,
                 denoise_linear: bool = False, denoise_strength: float = 0.90,
                 denoise_gpu: bool = True,
                 do_stretch: bool = True, do_median_sub: bool = True,
                 do_curves: bool = True, no_black_clip: bool = False):
        super().__init__(parent=None)
        self.nb_image        = nb_image
        self.continuum_image = continuum_image
        self.output_linear   = output_linear
        self.starless_nb     = starless_nb
        self.starless_cont   = starless_cont
        self.starless_only   = starless_only
        self.background_reference = None
        self.auto_min_stars  = int(auto_min_stars)
        self.auto_ladder     = tuple(float(v) for v in auto_ladder)
        # NOTE: no q_factor here. Q comes from the Q Preview dialog, per
        # filter, via set_accepted_q / run_final_subtract.
        self.used_sigma      = None     # resolved by the waterfall, for the UI
        self.denoise_linear  = bool(denoise_linear)
        self.denoise_strength = float(denoise_strength)
        self.denoise_gpu     = bool(denoise_gpu)
        self.do_stretch      = bool(do_stretch)
        self.do_median_sub   = bool(do_median_sub)
        self.do_curves       = bool(do_curves)
        self.no_black_clip   = bool(no_black_clip)

        # set by the tab after the user accepts Q values
        self._accepted_q: float | None = None

    def set_accepted_q(self, q: float):
        self._accepted_q = q

    # ── helpers (unchanged from original) ─────────────────────────────────────

    @staticmethod
    def _to_mono(img):
        a = np.asarray(img)
        if a.ndim == 3:
            if a.shape[2] == 3: return a[..., 0]
            if a.shape[2] == 1: return a[..., 0]
        return a

    @staticmethod
    def _as_rgb(nb, cont):
        r = np.asarray(nb,   dtype=np.float32)
        g = np.asarray(cont, dtype=np.float32)
        if r.ndim == 3: r = r[..., 0]
        if g.ndim == 3: g = g[..., 0]
        if r.dtype.kind in "ui":
            r = r / (255.0 if r.dtype == np.uint8 else 65535.0)
        if g.dtype.kind in "ui":
            g = g / (255.0 if g.dtype == np.uint8 else 65535.0)
        b = g
        return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _axis_floor(plane: np.ndarray) -> float:
        """Background + FIT_FLOOR_K sigma for one channel.

        This is the level a star has to clear before its flux means
        anything -- an aperture SNR of FIT_FLOOR_SNR on the circular
        median that is actually sampled. Measured per frame because the
        alternative -- a fixed number in [0,1] -- is only ever right for
        one exposure scale: on linear narrowband data the whole frame can
        sit below 0.02, and a 0.005 floor then throws away most of the
        field.
        """
        a = np.asarray(plane, dtype=np.float32).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float(FIT_CLIP_LO)
        if a.size > 400_000:            # MAD on a sample is plenty
            a = a[:: max(1, a.size // 400_000)]
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(a))
        return float(max(FIT_CLIP_LO, med + FIT_FLOOR_K * sigma))

    @classmethod
    def _plane_floors(cls, rgb: np.ndarray) -> tuple[float, float]:
        """(NB floor, broadband floor) for an (H,W,3) working image."""
        return (cls._axis_floor(rgb[..., 0]), cls._axis_floor(rgb[..., 1]))

    @staticmethod
    def _fit_ab(x, y):
        x = x.reshape(-1).astype(np.float32)
        y = y.reshape(-1).astype(np.float32)
        N = min(x.size, 100_000)
        if x.size > N:
            idx = np.random.choice(x.size, N, replace=False)
            x = x[idx]; y = y[idx]
        # Reject samples saturated in EITHER axis — a star clipped at 1.0 in
        # both input and output gives a (1,1) point that falsely anchors the
        # fit at the top of the dynamic range.
        keep = (x < 0.95) & (y < 0.95) & (x > 0.005) & (y > 0.005) \
            & np.isfinite(x) & np.isfinite(y)
        if keep.sum() >= 10:
            x, y = x[keep], y[keep]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(a), float(b)

    def _nonlinear_finalize_with_opts(self, lin_img: np.ndarray) -> np.ndarray:
        if not self.do_stretch:
            return np.clip(lin_img, 0.0, 1.0).astype(np.float32, copy=False)
        target_median = 0.25
        stretched = stretch_color_image(lin_img, target_median, True, False,
                                        no_black_clip=self.no_black_clip)
        if self.do_median_sub:
            stretched = stretched - 0.7 * np.median(stretched)
            stretched = np.clip(stretched, 0.0, 1.0)
        if self.do_curves:
            stretched = apply_curves_adjustment(stretched, np.median(stretched), 0.5)
        return np.clip(stretched, 0.0, 1.0).astype(np.float32, copy=False)

    def _compute_bg_pedestal(self, rgb):
        height, width, _ = rgb.shape
        num_boxes, box_size, iterations = 200, 25, 25
        boxes = [(np.random.randint(0, height - box_size),
                  np.random.randint(0, width - box_size)) for _ in range(num_boxes)]
        best = np.full(num_boxes, np.inf, dtype=np.float32)
        for _ in range(iterations):
            for i, (y, x) in enumerate(boxes):
                if y + box_size <= height and x + box_size <= width:
                    patch = rgb[y:y+box_size, x:x+box_size]
                    med = np.median(patch) if patch.size else np.inf
                    best[i] = min(best[i], med)
                    sv = []
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy*box_size, x + dx*box_size
                            if 0 <= yy < height-box_size and 0 <= xx < width-box_size:
                                p2 = rgb[yy:yy+box_size, xx:xx+box_size]
                                if p2.size: sv.append(np.median(p2))
                    if sv:
                        k = int(np.argmin(sv))
                        y += (k // 3 - 1) * box_size
                        x += (k %  3 - 1) * box_size
                        boxes[i] = (y, x)
        darkest = np.inf; ref = None
        for y, x in boxes:
            if y+box_size <= height and x+box_size <= width:
                patch = rgb[y:y+box_size, x:x+box_size]
                med = np.median(patch) if patch.size else np.inf
                if med < darkest:
                    darkest, ref = med, patch
        ped = np.zeros(3, dtype=np.float32)
        if ref is not None:
            self.background_reference = np.median(ref.reshape(-1, 3), axis=0)
            chan_meds = np.median(rgb, axis=(0, 1))
            ped = np.maximum(0.0, chan_meds - self.background_reference)
            r_ref = float(self.background_reference[0])
            for ch in (1, 2):
                if self.background_reference[ch] < r_ref:
                    ped[ch] += (r_ref - self.background_reference[ch])
        return ped

    @staticmethod
    def _apply_pedestal(rgb, ped):
        return np.clip(rgb + ped.reshape(1, 1, 3), 0.0, 1.0)

    @staticmethod
    def _normalize_red_to_green(rgb):
        r = rgb[..., 0]; g = rgb[..., 1]
        mad_r = float(np.mean(np.abs(r - np.mean(r))))
        mad_g = float(np.mean(np.abs(g - np.mean(g))))
        med_r = float(np.median(r))
        med_g = float(np.median(g))
        g_gain = mad_g / max(mad_r, 1e-9)
        g_offs = -g_gain * med_r + med_g
        rgb2 = rgb.copy()
        rgb2[..., 0] = np.clip(r * g_gain + g_offs, 0.0, 1.0)
        return rgb2, g_gain, g_offs

    def _linear_subtract(self, rgb, Q, green_median):
        r = rgb[..., 0]; g = rgb[..., 1]
        return np.clip(r - Q * (g - green_median), 0.0, 1.0)

    def _denoise_linear_image(self, img: np.ndarray) -> np.ndarray:
        try:
            if img is None or _cc_denoise is None:
                return img
            preset = {
                "mode": "denoise",
                "denoise_luma": self.denoise_strength,
                "denoise_color": self.denoise_strength,
                "denoise_mode": "full",
                "separate_channels": False,
                "denoise_lite": False,
                "denoise_walking": False,
                "gpu": self.denoise_gpu,
                "chunk_size": 256,
                "overlap": 64,
                "temp_stretch": False,
                "target_median": 0.25,
            }
            arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
            result = _cc_denoise(arr, preset)
            return np.clip(np.asarray(result, dtype=np.float32), 0.0, 1.0)
        except Exception as e:
            try: self.status_update.emit(f"Cosmic Clarity denoise failed: {e}")
            except Exception: pass
            return img

    @staticmethod
    def _downsample(arr: np.ndarray, max_px: int) -> np.ndarray:
        """Downsample arr so its long axis is ≤ max_px."""
        h, w = arr.shape[:2]
        long = max(h, w)
        if long <= max_px:
            return arr
        scale = max_px / long
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        # simple block-average via reshape when divisible, otherwise slice stride
        try:
            if cv2 is not None:
                return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
        # fallback: nearest-neighbour via index stride
        sy = max(1, h // nh); sx = max(1, w // nw)
        return arr[::sy, ::sx][: nh, :nw]

    # ── calibration pass (runs in thread) ─────────────────────────────────────

    def _calibrate(self, nb, cont, name: str) -> dict | None:
        """
        Run everything up to (but not including) the linear subtract.
        Returns a dict that can be handed to QPreviewDialog and later
        to _final_subtract().
        """
        try:
            rgb = self._as_rgb(nb, cont)

            self.status_update.emit(f"[{name}] Background neutralization…")
            ped = self._compute_bg_pedestal(rgb)
            rgb = self._apply_pedestal(rgb, ped)

            self.status_update.emit(f"[{name}] Normalizing red to green…")
            rgb, g_gain, g_offs = self._normalize_red_to_green(rgb)

            self.status_update.emit(
                f"[{name}] Star-based white balance (auto σ)…")

            def _resolved(sig, _n=name):
                self.used_sigma = float(sig)
                self.status_update.emit(
                    f"[{_n}] star detection settled at σ={float(sig):g}")

            # reuse_cached_sources is deliberately OFF here. The cache in
            # starbasedwhitebalance is module-global and, in auto-σ mode, its
            # gate is image SHAPE alone -- so Ha, SII and OIII (same shape,
            # and calibrated in parallel threads) would silently share
            # whichever filter detected first, as would any same-size image
            # left behind by the White Balance tool. Each filter resolves its
            # own σ against its own data.
            balanced_rgb, star_count, _overlay, raw_pix, after_pix, star_xy = \
                apply_star_based_white_balance(
                    # with_overlay=False: nothing displays the star overlay
                    # any more, and building it costs a full-resolution
                    # autostretch plus one ellipse per detected star.
                    rgb, with_overlay=False,
                    reuse_cached_sources=False, return_star_colors=True,
                    # positions: so the diagnostics can say WHERE the
                    # fitted stars were, and so the fit can refuse to be
                    # decided by one crowded region.
                    return_positions=True,
                    auto_sigma=True,
                    auto_sigma_ladder=self.auto_ladder,
                    auto_sigma_min_stars=self.auto_min_stars,
                    on_sigma_resolved=_resolved,
                )

            wb_a = np.zeros(3, np.float32)
            wb_b = np.zeros(3, np.float32)
            for c in range(3):
                a, b = self._fit_ab(rgb[..., c], balanced_rgb[..., c])
                wb_a[c], wb_b[c] = a, b

            green_med = float(np.median(balanced_rgb[..., 1]))

            preview = self._downsample(balanced_rgb, self.PREVIEW_MAX)

            recipe = {
                "pedestal":    ped,
                "rnorm_gain":  g_gain,
                "rnorm_offs":  g_offs,
                "wb_a":        wb_a,
                "wb_b":        wb_b,
                "green_median": green_med,
                "fit_star_count": int(star_count),
                "fit_sigma": (float(self.used_sigma)
                              if self.used_sigma is not None else None),
                "fit_raw":   np.array(raw_pix,   copy=True),
                "fit_after": np.array(after_pix, copy=True),
                "fit_xy":    (None if star_xy is None
                              else np.array(star_xy, copy=True)),
                "fit_shape": tuple(rgb.shape[:2]),
                "fit_floor":       self._plane_floors(rgb),
                "fit_floor_after": self._plane_floors(balanced_rgb),
            }

            return {
                "name":       name,
                "balanced":   balanced_rgb,   # full-res, kept for final subtract
                "green_med":  green_med,
                "recipe":     recipe,
                "preview":    preview,         # downsampled
                "star_count": int(star_count),
                "used_sigma": (float(self.used_sigma)
                               if self.used_sigma is not None else None),
            }

        except Exception as e:
            self.status_update.emit(f"[{name}] Calibration failed: {e}")
            return None

    def run_calibration_only(self, name: str):
        """Entry point: calibrate and emit calibration_ready. Called from run()."""
        if self.starless_only:
            self._run_starless_only_calibrate(name)
            return

        entry = self._calibrate(self.nb_image, self.continuum_image, name)
        if entry is not None:
            self.calibration_ready.emit(entry)

    def _run_starless_only_calibrate(self, name: str):
        """Starless-only path: no WB, just BG+norm, then emit calibration_ready."""
        try:
            rgb = self._as_rgb(self.starless_nb, self.starless_cont)
            self.status_update.emit(f"[{name}] Background neutralization…")
            ped = self._compute_bg_pedestal(rgb)
            rgb = self._apply_pedestal(rgb, ped)
            self.status_update.emit(f"[{name}] Normalizing…")
            rgb, _, _ = self._normalize_red_to_green(rgb)
            green_med = float(np.median(rgb[..., 1]))
            preview   = self._downsample(rgb, self.PREVIEW_MAX)
            recipe    = {
                "pedestal": np.zeros(3, np.float32),
                "rnorm_gain": 1.0, "rnorm_offs": 0.0,
                "wb_a": np.ones(3, np.float32), "wb_b": np.zeros(3, np.float32),
                "green_median": green_med,
                "fit_star_count": 0,
                "fit_sigma": None,
                "fit_raw":   np.empty((0, 2), float),
                "fit_after": np.empty((0, 2), float),
                "fit_xy":    None,
                "fit_shape": tuple(rgb.shape[:2]),
                "fit_floor":       (FIT_CLIP_LO, FIT_CLIP_LO),
                "fit_floor_after": (FIT_CLIP_LO, FIT_CLIP_LO),
                "starless_only": True,
            }
            entry = {
                "name":       name,
                "balanced":   rgb,
                "green_med":  green_med,
                "recipe":     recipe,
                "preview":    preview,
                "star_count": 0,
                "used_sigma": None,
            }
            self.calibration_ready.emit(entry)
        except Exception as e:
            self.status_update.emit(f"[{name}] Starless calibration failed: {e}")

    def run_final_subtract(self, name: str, entry: dict, q: float):
        """
        Called from a second QThread (FinalSubtractThread) after Q is accepted.
        Does the actual subtraction + optional denoise + stretch and emits results.
        """
        try:
            balanced  = entry["balanced"]
            green_med = entry["green_med"]
            recipe    = entry["recipe"]
            is_sl     = recipe.get("starless_only", False)

            self.status_update.emit(f"[{name}] Applying Q={q:.2f} subtraction…")
            linear = self._linear_subtract(balanced, q, green_med)

            if self.denoise_linear:
                self.status_update.emit(f"[{name}] Denoising…")
                linear = self._denoise_linear_image(linear)

            star_count = recipe["fit_star_count"]
            raw_pix    = recipe["fit_raw"]
            after_pix  = recipe["fit_after"]

            if self.output_linear:
                result = np.clip(linear, 0, 1)
            else:
                self.status_update.emit(f"[{name}] Stretching…")
                result = self._nonlinear_finalize_with_opts(linear)

            sig = self.processing_complete_starless if is_sl else self.processing_complete
            sig.emit(result, star_count,
                     np.array(raw_pix), np.array(after_pix))

            # paired starless pass (apply same recipe + new Q)
            if (not is_sl and
                    self.starless_nb is not None and
                    self.starless_cont is not None):
                self.status_update.emit(f"[{name}] Applying recipe to starless…")
                rgb_sl = self._as_rgb(self.starless_nb, self.starless_cont)
                rgb_sl = self._apply_pedestal(rgb_sl, recipe["pedestal"])
                rgb_sl[..., 0] = np.clip(
                    rgb_sl[..., 0] * recipe["rnorm_gain"] + recipe["rnorm_offs"],
                    0.0, 1.0
                )
                for c in range(3):
                    rgb_sl[..., c] = np.clip(
                        rgb_sl[..., c] * recipe["wb_a"][c] + recipe["wb_b"][c],
                        0.0, 1.0
                    )
                lin_sl = self._linear_subtract(rgb_sl, q, green_med)
                if self.denoise_linear:
                    lin_sl = self._denoise_linear_image(lin_sl)
                if self.output_linear:
                    res_sl = np.clip(lin_sl, 0, 1)
                else:
                    res_sl = self._nonlinear_finalize_with_opts(lin_sl)
                self.processing_complete_starless.emit(
                    res_sl, star_count,
                    np.array(raw_pix, copy=True),
                    np.array(after_pix, copy=True)
                )

        except Exception as e:
            self.status_update.emit(f"[{name}] Final subtract failed: {e}")

    def run(self):
        # This run() is only used for the calibration phase now.
        # The tab calls run_calibration_only() by kicking off one thread per filter.
        pass


class FinalSubtractThread(QThread):
    """Thin wrapper that calls thread.run_final_subtract() off the UI thread."""
    def __init__(self, thread: ContinuumProcessingThread,
                 name: str, entry: dict, q: float, parent=None):
        super().__init__(parent=None)
        self._t     = thread
        self._name  = name
        self._entry = entry
        self._q     = q

    def run(self):
        self._t.run_final_subtract(self._name, self._entry, self._q)