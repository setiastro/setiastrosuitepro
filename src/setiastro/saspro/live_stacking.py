from __future__ import annotations
import os
import glob
import shutil
import tempfile
import datetime as _dt
import numpy as np
import time
import threading
from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QPointF, QThread, QCoreApplication, QT_TRANSLATE_NOOP
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QPen, QColor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox,
                             QGraphicsRectItem)

import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# optional deps used in your code; guard if not installed
import rawpy

import exifread

import sep


# your helpers/utilities
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.legacy.numba_utils import apply_flat_division_numba, debayer_fits_fast   # adjust names if different
from setiastro.saspro.legacy.image_manager import load_image
from setiastro.saspro.star_alignment import StarRegistrationWorker, StarRegistrationThread, IDENTITY_2x3
from setiastro.saspro.widgets.spinboxes import CustomSpinBox, CustomDoubleSpinBox
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.widgets.factory_reset import make_reset_button
from collections import deque
from setiastro.saspro import live_stack_worker as _lsw
from setiastro.saspro.live_stack_worker import (   # also re-exported here
    as_hwc as _as_hwc, linear_channel_stats, normalize_linear, prepare_frame)

class LiveStackSettingsDialog(QDialog):
    """
    Combined dialog for:
      • Live‐stack parameters (bootstrap frames, σ‐clip threshold)
      • Culling thresholds (max FWHM, max eccentricity, min star count)
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Live Stack & Culling Settings"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        # — Live Stack Settings —
        # Bootstrap frames (int)
        self.bs_spin = CustomSpinBox(
            minimum=1,
            maximum=100,
            initial=parent.bootstrap_frames,
            step=1
        )
        self.bs_spin.valueChanged.connect(lambda v: None)

        # Sigma threshold (float)
        self.sigma_spin = CustomDoubleSpinBox(
            minimum=0.1,
            maximum=10.0,
            initial=parent.clip_threshold,
            step=0.1
        )
        self.sigma_spin.valueChanged.connect(lambda v: None)

        # — Culling Thresholds —
        # Max FWHM (float)
        self.fwhm_spin = CustomDoubleSpinBox(
            minimum=0.1,
            maximum=50.0,
            initial=parent.max_fwhm,
            step=0.1
        )
        self.fwhm_spin.valueChanged.connect(lambda v: None)

        # Max eccentricity (float)
        self.ecc_spin = CustomDoubleSpinBox(
            minimum=0.0,
            maximum=1.0,
            initial=parent.max_ecc,
            step=0.01
        )
        self.ecc_spin.valueChanged.connect(lambda v: None)

        # Min star count (int)
        self.star_spin = CustomSpinBox(
            minimum=0,
            maximum=5000,
            initial=parent.min_star_count,
            step=1
        )
        self.star_spin.valueChanged.connect(lambda v: None)
        
        self.delay_spin = CustomDoubleSpinBox(
            minimum=0.0,
            maximum=60.0,
            initial=parent.FILE_STABLE_SECS,
            step=0.5
        )

        # Stars to aim for: the detection threshold is chosen by star count
        self.star_target_spin = CustomSpinBox(
            minimum=10,
            maximum=2000,
            initial=int(getattr(parent, "star_target", _lsw.STAR_TARGET)),
            step=10
        )
        self.star_target_spin.setToolTip(self.tr(
            "How many stars star detection aims for. On a stack's first good frame the\n"
            "threshold steps down (100σ, 50σ, 25σ … 3.5σ) until it finds at least this\n"
            "many, and that threshold is then kept for the rest of the stack, so the star\n"
            "count stays comparable from frame to frame (a cloudy frame shows fewer stars).\n"
            "Each filter gets its own threshold. A change applies after Reset."))
        # Ignore non-FITS files
        self.fits_only_check = QCheckBox(self.tr("Ignore non-FITS files (skip TIFF/PNG/JPG/RAW/XISF)"))
        self.fits_only_check.setChecked(parent.fits_only)
        self.fits_only_check.setToolTip(self.tr(
            "When enabled, only *.fit and *.fits files are picked up from the watch folder.\n"
            "Useful when your capture software also writes preview JPGs, autostretch TIFFs,\n"
            "or other derived images alongside the raw FITS frames."
        ))
        # Frame weighting (Stacking Suite 'Balanced' score)
        self.weight_check = QCheckBox(self.tr("Weight frames by quality"))
        self.weight_check.setChecked(bool(getattr(parent, "weight_frames", True)))
        self.weight_check.setToolTip(self.tr(
            "Each frame counts in proportion to its quality, scored like the\n"
            "Stacking Suite's 'Balanced' weighting: star count, roundness, sky darkness and\n"
            "sharpness. Poor frames still add signal, just less of it."))

        # SNR goal for the frames-needed estimate
        self.snr_goal_spin = CustomDoubleSpinBox(
            minimum=1.0,
            maximum=1000.0,
            initial=float(getattr(parent, "snr_goal", 10.0)),
            step=1.0
        )
        self.snr_goal_spin.setToolTip(self.tr(
            "Signal-to-noise per pixel you want in the SNR region (drag it on the\n"
            "preview with 'SNR Region'). Live Stacking estimates how many more frames\n"
            "it takes to get there."))

        # Build form layout
        form = QFormLayout()
        form.addRow(self.tr("Switch to μ–σ clipping after:"), self.bs_spin)
        form.addRow(self.tr("Clip threshold:"), self.sigma_spin)
        form.addRow(QLabel(""))  # blank row for separation
        form.addRow(self.tr("Max FWHM (px):"), self.fwhm_spin)
        form.addRow(self.tr("Max Eccentricity:"), self.ecc_spin)
        form.addRow(self.tr("Min Star Count:"), self.star_spin)
        form.addRow(self.tr("Acquisition Delay:"), self.delay_spin)
        form.addRow(self.tr("Stars to detect:"), self.star_target_spin)
        form.addRow(self.fits_only_check)
        form.addRow(self.weight_check)
        form.addRow(self.tr("SNR goal (per pixel):"), self.snr_goal_spin)

        self.mapping_combo = QComboBox()
        opts = ["Natural", "SHO", "HSO", "OSH", "SOH", "HOS", "OHS"]
        self.mapping_combo.addItems(opts)
        # preselect current
        idx = opts.index(parent.narrowband_mapping) \
              if parent.narrowband_mapping in opts else 0
        self.mapping_combo.setCurrentIndex(idx)
        form.addRow(self.tr("Narrowband Mapping:"), self.mapping_combo)

        # Demosaic for OSC / Bayer frames: the Stacking Suite's methods
        # (Strict CFA left out: it only prepares CFA drizzle)
        self.debayer_combo = QComboBox()
        for label, tok in DEBAYER_METHODS:
            self.debayer_combo.addItem(QCoreApplication.translate("LiveStacking", label), tok)
        cur = str(getattr(parent, "debayer_method", "rcd")).lower()
        self.debayer_combo.setCurrentIndex(
            next((i for i in range(self.debayer_combo.count())
                  if self.debayer_combo.itemData(i) == cur), 0))
        self.debayer_combo.setToolTip(self.tr(
            "Demosaic used for OSC / Bayer frames (FITS with BAYERPAT and camera RAW),\n"
            "the same methods as the Stacking Suite's Debayer setting:\n"
            "• RCD: noise-robust, keeps faint colour; best for astro, also OSC narrowband.\n"
            "• AHD: adaptive homogeneity-directed; slower.\n"
            "• VNG: sharp on bright subjects, speckles colour in dim noisy areas.\n"
            "• Edge-aware / Bilinear: fastest, lower quality.\n"
            "Applies to frames stacked after the change; Reset to restack with it.\n"
            "X-Trans RAW always uses LibRaw's demosaic."))
        form.addRow(self.tr("Debayer:"), self.debayer_combo)

        # OK / Cancel buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.btn_reset = make_reset_button(self)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        btns.addButton(self.btn_reset, QDialogButtonBox.ButtonRole.ResetRole)

        # Assemble dialog layout
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btns)
        self.setLayout(layout)

    def getValues(self):
        """
        Returns a tuple in order:
            (bootstrap_frames, clip_threshold,
            max_fwhm, max_ecc, min_star_count, mapping, delay,
            star_target, fits_only, weight_frames, snr_goal,
            debayer_method)
        """
        bs            = int(self.bs_spin.value())
        sigma         = self.sigma_spin.value()
        fwhm          = self.fwhm_spin.value()
        ecc           = self.ecc_spin.value()
        stars         = int(self.star_spin.value())
        mapping       = self.mapping_combo.currentText()
        delay         = self.delay_spin.value()
        star_target   = int(self.star_target_spin.value())
        fits_only     = self.fits_only_check.isChecked()
        weight_frames = self.weight_check.isChecked()
        snr_goal      = float(self.snr_goal_spin.value())
        debayer_method = str(self.debayer_combo.currentData() or "rcd")
        return (bs, sigma, fwhm, ecc, stars, mapping, delay, star_target, fits_only,
                weight_frames, snr_goal, debayer_method)

    def _reset_to_factory(self):
        """Every field back to its factory default. OK applies and saves them,
        Cancel keeps the current settings, as for any edit here."""
        D = LIVE_STACK_DEFAULTS
        self.bs_spin.setValue(D["bootstrap_frames"])
        self.sigma_spin.setValue(D["clip_threshold"])
        self.fwhm_spin.setValue(D["max_fwhm"])
        self.ecc_spin.setValue(D["max_ecc"])
        self.star_spin.setValue(D["min_star_count"])
        self.delay_spin.setValue(D["file_stable_secs"])
        self.star_target_spin.setValue(D["star_target"])
        self.fits_only_check.setChecked(D["fits_only"])
        self.weight_check.setChecked(D["weight_frames"])
        self.snr_goal_spin.setValue(D["snr_goal"])
        self.mapping_combo.setCurrentIndex(max(0, self.mapping_combo.findText(D["narrowband_mapping"])))
        # Debayer has no default of its own: it follows the Stacking Suite's choice
        dm = str(QSettings().value("stacking/debayer_method", "rcd") or "rcd").lower()
        i = self.debayer_combo.findData(dm)
        self.debayer_combo.setCurrentIndex(i if i >= 0 else 0)

# Debayer methods offered in Settings (the Stacking Suite's, minus Strict CFA)
DEBAYER_METHODS = (("RCD", "rcd"), (QT_TRANSLATE_NOOP("LiveStacking", "Edge-aware"), "edge"), ("AHD", "ahd"),
                   ("VNG", "vng"), (QT_TRANSLATE_NOOP("LiveStacking", "Bilinear"), "bilinear"))

# Factory defaults of the Settings dialog: LiveStackWindow falls back to them
# when nothing is saved, and the dialog's Reset restores them.
LIVE_STACK_DEFAULTS = {
    "bootstrap_frames":   24,
    "clip_threshold":     3.5,
    "max_fwhm":           15.0,
    "max_ecc":            0.9,
    "min_star_count":     5,
    "narrowband_mapping": "Natural",
    "file_stable_secs":   3.0,
    "star_target":        _lsw.STAR_TARGET,
    "fits_only":          False,
    "weight_frames":      True,
    "snr_goal":           10.0,
}


def _qget(settings: QSettings, key: str, default, typ):
    try:
        return settings.value(key, default, type=typ)
    except TypeError:
        # Key contains junk (likely from earlier method-object save). Reset it.
        try:
            settings.remove(key)
        except Exception:
            pass
        settings.setValue(key, default)
        return default



class LiveMetricsPanel(QWidget):
    """
    3×2 grid of PyQtGraph plots, one point per frame:
      FWHM (px)              · Eccentricity
      Star count             · Frame weight (× this stack's median weight)
      Stack noise (log–log)  · SNR per pixel in the SNR region
    Every filter is its own series, in its own colour, and its x counts that
    filter's frames, so in Mono → Color the filters build up side by side
    from frame 1 (the noise plot too, on log axes).
    Pure random noise falls with slope −½ on the log–log noise plot.
    Culled frames are a red × and appear on the per-frame plots only.
    """
    TITLES = ["FWHM (px)", QT_TRANSLATE_NOOP("LiveMetricsPanel", "Eccentricity"),
              QT_TRANSLATE_NOOP("LiveMetricsPanel", "Star Count"),
              QT_TRANSLATE_NOOP("LiveMetricsPanel", "Frame weight (× median)"),
              QT_TRANSLATE_NOOP("LiveMetricsPanel", "Stack noise (log–log)"),
              QT_TRANSLATE_NOOP("LiveMetricsPanel", "SNR / px (region)")]
    _NOISE = 4
    # filter key → (legend label, colour). "__main__" = the single stack.
    SERIES = {
        "__main__": ("", (100, 100, 255)),
        "L": ("L", (215, 215, 215)),
        "R": ("R", (255, 85, 85)),
        "G": ("G", (90, 210, 90)),
        "B": ("B", (80, 140, 255)),
        "H": ("Ha", (255, 130, 170)),
        "O": ("OIII", (50, 215, 230)),
        "S": ("SII", (255, 190, 50)),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        grid = pg.GraphicsLayoutWidget()
        layout.addWidget(grid)

        n = len(self.TITLES)
        self.plots = []
        self._series = [dict() for _ in range(n)]   # key → {"x","y","f","item"}
        self.legend = None
        # all points of a plot, every series (in arrival order)
        self._data_x = [[] for _ in range(n)]
        self._data_y = [[] for _ in range(n)]
        self._flags = [[] for _ in range(n)]   # True = culled frame

        for idx, title in enumerate(self.TITLES):
            pw = grid.addPlot(row=idx // 2, col=idx % 2)
            pw.setTitle(QCoreApplication.translate("LiveMetricsPanel", title))
            pw.showGrid(x=True, y=True, alpha=0.3)
            if idx == self._NOISE:
                # x = frame number like the other plots, so filters line up;
                # the trend slope is still fitted against N_eff
                pw.setLabel('bottom', self.tr("Frame # (per filter)"))
                # Camera units: a filter's first frame keeps gain 1 and later
                # frames are scaled to it, so filters share units. Absolute
                # noise rises with brightness (shot noise); noise relative to
                # signal is the SNR plot.
                pw.setLabel('left', self.tr("noise (camera units)"))
                pw.setLogMode(x=True, y=True)
            else:
                pw.setLabel('bottom', self.tr("Frame # (per filter)"))
            self.plots.append(pw)

    def _series_for(self, i, key):
        s = self._series[i].get(key)
        if s is not None:
            return s
        label, (r, g, b) = self.SERIES.get(key, (str(key), (200, 200, 200)))
        pw = self.plots[i]
        if i == self._NOISE:
            # PlotDataItem, not a bare ScatterPlotItem: only data items
            # follow the plot's log mode (a scatter item plots raw values
            # on log-labelled axes)
            item = pw.plot([], [], pen=None, symbol='o', symbolSize=6,
                           symbolPen=None, symbolBrush=pg.mkBrush(r, g, b, 220))
        else:
            item = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None),
                                      brush=pg.mkBrush(r, g, b, 220))
            pw.addItem(item)
        if label and i == 0:
            if self.legend is None:
                self.legend = pw.addLegend(offset=(-10, 10))
            self.legend.addItem(item, label)
        s = {"x": [], "y": [], "f": [], "item": item, "color": (r, g, b)}
        self._series[i][key] = s
        return s

    def add_point(self, frame_idx: int, values, flagged: bool = False, key="__main__"):
        """values: one entry per plot, in TITLES order; None = no point there.
        An entry may be an (x, y) pair to plot against something other than
        the frame number. key: the filter."""
        for i, v in enumerate(values):
            if i >= len(self.plots) or v is None:
                continue
            x = frame_idx
            if isinstance(v, (tuple, list)):
                x, v = v
            try:
                x, v = float(x), float(v)
            except Exception:
                continue
            if not (np.isfinite(v) and np.isfinite(x)):
                continue
            if i == self._NOISE and (v <= 0 or x <= 0):
                continue
            s = self._series_for(i, key)
            s["x"].append(x)
            s["y"].append(v)
            s["f"].append(bool(flagged))
            self._data_x[i].append(x)
            self._data_y[i].append(v)
            self._flags[i].append(bool(flagged))
            self._redraw(i, s)

    def _redraw(self, i, s):
        if i == self._NOISE:
            s["item"].setData(s["x"], s["y"])      # culled frames never get here
            return
        r, g, b = s["color"]
        good = pg.mkBrush(r, g, b, 220)
        bad = pg.mkBrush(255, 40, 40, 230)
        s["item"].setData(s["x"], s["y"],
                          brush=[bad if f else good for f in s["f"]],
                          symbol=['x' if f else 'o' for f in s["f"]],
                          size=[9 if f else 6 for f in s["f"]],
                          pen=pg.mkPen(None))

    def clear_all(self):
        """Clear data from all plots."""
        for i, series in enumerate(self._series):
            for s in series.values():
                try:
                    self.plots[i].removeItem(s["item"])
                except Exception:
                    pass
            series.clear()
            self._data_x[i].clear()
            self._data_y[i].clear()
            self._flags[i].clear()
        if self.legend is not None:
            try:
                self.legend.clear()
            except Exception:
                pass

class LiveMetricsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # its own window, but owned by Live Stacking: it stays above it and is
        # never a primary window that keeps SASpro running after exit
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(self.tr("Live Stack Metrics"))
        self.resize(760, 820)

        layout = QVBoxLayout(self)
        self.metrics_panel = LiveMetricsPanel(self)
        layout.addWidget(self.metrics_panel)

def compute_frame_star_metrics(image_2d, thresh_sigma: float = 7.0):
    """(star_count, fwhm, ecc) with Blink's SEP settings (7σ, minarea 16,
    deblend 32, median aggregation); see live_stack_worker.star_metrics."""
    data = np.asarray(image_2d)
    if data.ndim == 3:
        data = data.mean(axis=2)
    if data.dtype == np.uint8:
        data = data.astype(np.float32) / 255.0
    elif data.dtype == np.uint16:
        data = data.astype(np.float32) / 65535.0
    else:
        data = data.astype(np.float32, copy=False)
    return _lsw.star_metrics(data, float(thresh_sigma))

# ─────────────────────────────────────────────────────────────────────────────
# Linear accumulation
#
# Frames are normalized LINEARLY (per channel, the Stacking Suite's
# min-subtracted-median gain), aligned, and accumulated as a weighted running
# mean + variance per pixel. The statistical stretch is applied to the preview
# only, so the stack that opens in a new view is real linear data.
# 0.0 is no-data everywhere (warp borders, dead pixels): it never enters the
# mean, the variance or the weights.
# ─────────────────────────────────────────────────────────────────────────────

_BAND_ROWS = 256   # row band for per-pixel updates; bounds the temporaries


def _fmt_hms(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_dur(seconds: float) -> str:
    m = int(round(max(0.0, float(seconds)) / 60.0))
    return f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m} min"


def spatial_noise(lum, max_side: int = 2048) -> float:
    """Pixel-to-pixel noise of a 2-D image: 1.4826 * MAD of horizontal first
    differences / sqrt(2), over a central crop, no-data excluded. A smooth
    target (galaxy, nebula, gradient) barely moves a first difference, so this
    follows the noise rather than the object."""
    lum = np.asarray(lum, dtype=np.float32)
    H, W = lum.shape
    y0 = max(0, (H - max_side) // 2)
    x0 = max(0, (W - max_side) // 2)
    c = lum[y0:y0 + max_side, x0:x0 + max_side]
    a, b = c[:, :-1], c[:, 1:]
    ok = (a != 0.0) & (b != 0.0) & np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    if d.size < 1000:
        return float("nan")
    med = float(np.median(d))
    return float(1.4826 * np.median(np.abs(d - med)) / np.sqrt(2.0))


def find_sky_box(lum, box: int = 50, margin: int = 100):
    """Background sample box (x0, y0, x1, y1) for the SNR, found exactly the
    way Background Neutralization's Find Background does (auto_rect_box: the
    darkest of 10x10 tile medians, a downhill walk to the local minimum, a
    50x50 refine). No-data (0.0) pixels are lifted to the image maximum first
    so the walk never settles on a warp border. None if it can't be found."""
    a = np.asarray(lum, dtype=np.float32)
    if a.ndim != 2:
        return None
    H, W = a.shape
    valid = np.isfinite(a) & (a != 0.0)
    if not valid.any():
        return None
    margin = int(min(margin, max(0, (min(H, W) - box) // 4)))
    a = np.where(valid, a, np.float32(a[valid].max()))
    try:
        from setiastro.saspro.backgroundneutral import auto_rect_box
        x, y, w, h = auto_rect_box(np.broadcast_to(a[..., None], (H, W, 3)),
                                   box=box, margin=margin)
    except Exception:
        return None
    return (int(x), int(y), int(x + w), int(y + h))


def box_sky(lum, rect):
    """Median of the valid pixels in rect, or None when too few are valid."""
    x0, y0, x1, y1 = rect
    v = np.asarray(lum[y0:y1, x0:x1], dtype=np.float32)
    v = v[np.isfinite(v) & (v != 0.0)]
    return float(np.median(v)) if v.size >= 100 else None


def sky_level(lum, block: int = 16) -> float:
    """Fallback sky estimate when no background box can be found: median of
    the darkest 10% of block medians (blocks of 4*block px), no-data excluded.
    Biased somewhat low (it picks the low tail of noisy medians)."""
    import warnings
    sub = np.asarray(lum, dtype=np.float32)[::4, ::4]
    H, W = sub.shape
    hb, wb = H // block, W // block
    if hb < 2 or wb < 2:
        v = sub[(sub != 0.0) & np.isfinite(sub)]
        return float(np.percentile(v, 10)) if v.size else 0.0
    t = sub[:hb * block, :wb * block].reshape(hb, block, wb, block).swapaxes(1, 2)
    t = t.reshape(hb, wb, block * block)
    t = np.where(t == 0.0, np.nan, t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        meds = np.nanmedian(t, axis=2).ravel()
    meds = meds[np.isfinite(meds)]
    if meds.size == 0:
        return 0.0
    k = max(1, int(round(0.10 * meds.size)))
    return float(np.median(np.sort(meds)[:k]))


def noise_trend_slope(n_eff_hist, noise_hist, min_points: int = 8):
    """Slope of log(stack noise) against log(N_eff) over the recent half of
    the history. Pure random noise gives -0.5; a fixed pattern that stays put
    from frame to frame (no dither) flattens it toward 0."""
    pts = [(float(n), float(s)) for n, s in zip(n_eff_hist, noise_hist)
           if n and n > 0 and s is not None and np.isfinite(s) and s > 0]
    if len(pts) < min_points:
        return None
    pts = pts[len(pts) // 2:] if len(pts) >= 2 * min_points else pts[-min_points:]
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    if float(x.max() - x.min()) < float(np.log(1.5)):
        return None
    return float(np.polyfit(x, y, 1)[0])


class LiveAccumulator:
    """Weighted running mean and variance per pixel (West's update), with
    sigma rejection by EXCLUSION and 0.0 = no-data.

    mu : the stack itself (0.0 where no frame has contributed)
    S  : weighted sum of squared deviations
    W  : sum of weights of the samples kept at each pixel

    Frames before `bootstrap` are kept unconditionally, but the brightest
    sample at each pixel (and its weight) is remembered. When clipping
    starts, that sample is taken back out wherever it is an outlier against
    the remaining frames, so a satellite or plane in the first frames does
    not stay in the stack. No bootstrap frame is held in memory.
    """

    def __init__(self, shape, bootstrap: int, clip_k: float):
        H, W, C = (tuple(shape) + (1,))[:3] if len(shape) == 2 else tuple(shape)
        self.shape = (int(H), int(W), int(C))
        self.bootstrap = max(1, int(bootstrap))
        self.k = float(clip_k)
        self.mu = np.zeros(self.shape, np.float32)
        self.S = np.zeros(self.shape, np.float32)
        self.W = np.zeros(self.shape, np.float32)
        if self.bootstrap > 1:
            self._max = np.full(self.shape, -np.inf, np.float32)
            self._maxw = np.zeros(self.shape, np.float32)
        else:
            self._max = self._maxw = None
        self.n_frames = 0
        self.sum_w = 0.0
        self.sum_w2 = 0.0
        self.clipping = False
        self.sum_w_clipped = 0.0      # weight added while clipping was on
        self.last_reject_frac = 0.0
        self.bootstrap_removed_frac = 0.0

    @property
    def n_eff(self) -> float:
        """Effective number of equal-weight frames, (sum w)^2 / sum w^2."""
        return (self.sum_w * self.sum_w / self.sum_w2) if self.sum_w2 > 0 else 0.0

    @staticmethod
    def _truncation_factor(k: float) -> float:
        """Variance of a normal truncated at +-k sigma, relative to the full
        variance: 1 - 2k phi(k) / (2 Phi(k) - 1). 0.973 at k = 3."""
        import math
        k = float(k)
        if not (k > 0.5):
            return 1.0
        phi = math.exp(-0.5 * k * k) / math.sqrt(2.0 * math.pi)
        mass = math.erf(k / math.sqrt(2.0))
        return max(0.5, 1.0 - 2.0 * k * phi / mass)

    def _var(self, S, W):
        """Unbiased weighted variance per pixel and per-pixel N_eff. The
        per-pixel sum of squared weights is approximated from the global
        ratio, exact when weights are equal. inf where < 2 samples.

        Samples rejected at +-k sigma never enter S, so S describes a
        truncated distribution and runs low (~2% in sigma at k = 3). That is
        divided back out in proportion to the weight stacked while clipping,
        so rejection doesn't tighten on itself and the SNR isn't optimistic."""
        ratio = (self.sum_w / self.sum_w2) if self.sum_w2 > 0 else 0.0
        neff = W * np.float32(ratio)
        f = (self.sum_w_clipped / self.sum_w) if self.sum_w > 0 else 0.0
        corr = 1.0 - (1.0 - self._truncation_factor(self.k)) * f
        with np.errstate(divide="ignore", invalid="ignore"):
            var = S / (W * (1.0 - 1.0 / neff) * np.float32(corr))
        var[~(neff > 1.5)] = np.inf
        return var, neff

    def add(self, frame, w: float) -> None:
        x = _as_hwc(frame)
        if x.shape != self.shape:
            raise ValueError(f"frame shape {x.shape} != stack shape {self.shape}")
        w = float(w)
        if not (w > 0.0) or not np.isfinite(w):
            return
        wf = np.float32(w)
        clip = self.clipping
        rejected = 0
        valid_total = 0
        for y0 in range(0, self.shape[0], _BAND_ROWS):
            sl = slice(y0, y0 + _BAND_ROWS)
            mu, S, Wb = self.mu[sl], self.S[sl], self.W[sl]
            xb = x[sl]
            valid = np.isfinite(xb) & (xb != 0.0)
            xb = np.where(valid, xb, mu)                 # no-data: zero deviation
            keep = valid
            if clip:
                var, _ = self._var(S, Wb)
                with np.errstate(invalid="ignore"):
                    out = np.abs(xb - mu) > (np.float32(self.k) * np.sqrt(var))
                keep = valid & ~out
                rejected += int(np.count_nonzero(valid & out))
            valid_total += int(np.count_nonzero(valid))
            Wn = Wb + np.where(keep, wf, np.float32(0.0))
            d = xb - mu
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(keep, wf / Wn, np.float32(0.0))
            mu_new = mu + r * d
            S += np.where(keep, wf * d * (xb - mu_new), np.float32(0.0))
            mu[...] = mu_new
            Wb[...] = Wn
            if self._max is not None and not clip:
                mx = self._max[sl]
                upd = keep & (xb > mx)
                mx[upd] = xb[upd]
                self._maxw[sl][upd] = wf
        self.n_frames += 1
        self.sum_w += w
        self.sum_w2 += w * w
        if clip:
            self.sum_w_clipped += w
        self.last_reject_frac = (rejected / valid_total) if valid_total else 0.0
        if not self.clipping and self.n_frames >= self.bootstrap:
            self._finish_bootstrap()
            self.clipping = True

    @staticmethod
    def _max_outlier_crit(n_rest: int, alpha: float = 1e-3) -> float:
        """Critical deviation (in sigmas of the other samples) for the MAXIMUM
        of n_rest + 1 samples to be an outlier, at a false-removal rate of
        `alpha` per pixel.

        Testing each pixel's brightest bootstrap sample against plain k sigma
        is a multiple comparison with sigma estimated from few frames: on pure
        noise it removed a sample from ~13% of pixels at B = 8, k = 3, biasing
        the stack low. Bonferroni over the n + 1 candidates, Student t with
        n_rest - 1 degrees of freedom, scaled for predicting a new sample."""
        return _lsw.max_outlier_crit(n_rest, alpha)

    def _finish_bootstrap(self) -> None:
        """Remove each pixel's brightest bootstrap sample where it is a clear
        outlier against the remaining bootstrap frames (at least k sigma, and
        at least the multiple-comparison critical value)."""
        if self._max is None:
            return
        removed = 0
        total = 0
        k = np.float32(max(self.k, self._max_outlier_crit(self.n_frames - 1)))
        for y0 in range(0, self.shape[0], _BAND_ROWS):
            sl = slice(y0, y0 + _BAND_ROWS)
            mu, S, Wb = self.mu[sl], self.S[sl], self.W[sl]
            m, wm = self._max[sl], self._maxw[sl]
            has = (wm > 0) & np.isfinite(m)
            W1 = Wb - wm
            with np.errstate(divide="ignore", invalid="ignore"):
                mu1 = (Wb * mu - wm * m) / W1
            S1 = np.maximum(S - wm * (m - mu1) * (m - mu), np.float32(0.0))
            var1, neff1 = self._var(S1, W1)
            with np.errstate(invalid="ignore"):
                out = has & (W1 > 0) & (neff1 >= 2.5) & ((m - mu1) > k * np.sqrt(var1))
            mu[out] = mu1[out]
            S[out] = S1[out]
            Wb[out] = W1[out]
            removed += int(np.count_nonzero(out))
            total += int(np.count_nonzero(Wb > 0))
        self.bootstrap_removed_frac = (removed / total) if total else 0.0
        self._max = self._maxw = None

    def noise_of_mean(self, ys, xs):
        """Per-pixel noise of the stacked mean, sigma / sqrt(N_eff), from the
        frame-to-frame variance. NaN where it can't be estimated yet."""
        var, neff = self._var(self.S[ys, xs], self.W[ys, xs])
        with np.errstate(divide="ignore", invalid="ignore"):
            nm = np.sqrt(var / neff)
        nm[~np.isfinite(nm)] = np.nan
        return nm


def region_snr(acc: "LiveAccumulator", rect, sky: float):
    """Per-pixel SNR in rect = (x0, y0, x1, y1): (median stacked level - sky)
    / median per-pixel noise of the mean. Colour stacks use the channel mean.
    Returns (snr, signal, noise) or (None, None, None)."""
    x0, y0, x1, y1 = rect
    ys, xs = slice(y0, y1), slice(x0, x1)
    mu = acc.mu[ys, xs]
    nm = acc.noise_of_mean(ys, xs)
    C = mu.shape[2]
    lum = mu.mean(axis=2)
    with np.errstate(invalid="ignore"):
        lnoise = np.sqrt(np.sum(nm * nm, axis=2)) / C     # independent channel noise
    ok = (acc.W[ys, xs].min(axis=2) > 0) & np.isfinite(lnoise) & (lnoise > 0)
    if int(np.count_nonzero(ok)) < 9:
        return None, None, None
    signal = float(np.median(lum[ok])) - float(sky)
    noise = float(np.median(lnoise[ok]))
    if not (noise > 0):
        return None, None, None
    return signal / noise, signal, noise


_LIVE_THREADS = set()   # keeps a watch thread alive until it has finished


class _LiveWatchThread(QThread):
    """Scans the watch folder, hands new frames to the worker processes
    (live_stack_worker.prepare_frame) and stacks what they return.

    It never touches a widget: each frame's result goes back to the window
    through a queued signal and is applied there (_apply_result)."""
    frameDone = pyqtSignal(object)
    message = pyqtSignal(str)

    def __init__(self, win):
        super().__init__(None)          # QThread workers use parent=None
        self._win = win

    def run(self):
        win = self._win
        try:
            while True:
                try:
                    win._watch_loop(self.frameDone.emit, self.message.emit)
                    break
                except Exception as e:
                    self.message.emit(self.tr("⚠ Live stacking error: {0}").format(e))
                    if win._should_stop():
                        break
                    win._stop_event.wait(1.5)
        finally:
            self._win = None


class LiveStackWindow(QDialog):
    def __init__(self, parent=None, doc_manager=None, wrench_path=None, spinner_path=None):
        super().__init__(parent)
        self.parent = parent
        self._docman = doc_manager
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self.setWindowTitle(self.tr("Live Stacking"))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(900, 600)

        # ─── State Variables ─────────────────────────────────────
        self.watch_folder = None
        self.processed_files = set()
        self.master_dark = None
        self.master_flat = None
        self.master_flats  = {}

        # Linear stacks: key → LiveAccumulator. "__main__" normally; in
        # Mono → Color mode one per filter letter. Plus each stack's
        # per-channel normalization targets and running histories.
        self.stacks = {}
        self.norm_targets = {}
        self._fwhm_hist = {}      # key → accepted FWHMs (sharpness reference)
        self._weight_hist = {}    # key → accepted weights
        self._noise_hist = {}     # key → [(N_eff, stack noise)]
        self._exp_by_key = {}     # key → [total seconds, frames with EXPTIME]
        self._first_header = {}   # key → header of the first stacked frame
        self._frames_seen = 0
        self._last_key = None
        self._preview_cache = None
        self._notes_shown = set()
        self.reference_image_2d = None
        self.snr_region = None    # (x0, y0, x1, y1) in image pixels
        self._snr_rect_item = None
        self._snr_pick = False
        self._rb_last = None

        # Frames are processed on a watch thread. It is the only writer of
        # the stacks; it writes under _lock and GUI readers take _lock too.
        # _gen changes on Reset / mode switch so a frame still in flight is
        # dropped; _display_rev changes with the preview settings so a
        # preview rendered with old settings is redrawn instead of shown.
        self._lock = threading.RLock()
        self._gen = 0
        self._display_rev = 0
        self._notes_pending = []
        self._worker = None
        self._sky_box = {}        # key → (x0, y0, x1, y1) SNR background sample
        self._seen_by_key = {}    # key → frames measured (the metrics plots' x)
        self._star_sigma = {}     # key → star detection threshold, set by its first good frame
        # Worker processes that prepare frames (load, calibrate, measure,
        # normalize, align), and the .npy files that carry the masters and
        # the reference frame to them.
        self._pool = None
        self._pool_n = 0
        self._pool_failed = False
        self._xfer = {}           # name → (array, path)
        self._xfer_dir = None
        self._xfer_seq = 0
        self._sky_rect_item = None

        self.cull_folder = None

        self.is_running = False
        self.frame_count = 0
        self.current_stack = None

        self._probe = {}  # path -> {"size": int, "mtime": float, "since": float, "penalty_until": float}
        # Tunables:
        self.FILE_STABLE_SECS = 3.0          # how long size+mtime must stay unchanged
        self.OPEN_RETRY_PENALTY_SECS = 10.0  # cool-down after a read/permission failure
        self.MAX_FILE_WAIT_SECS = 600.0      # optional safety cap (unused by default logic)

        self._stop_event = threading.Event()

        # ── Load persisted settings ───────────────────────────────
        s = QSettings()
        D = LIVE_STACK_DEFAULTS
        self.bootstrap_frames   = _qget(s, "LiveStack/bootstrap_frames",   D["bootstrap_frames"],   int)
        self.clip_threshold     = _qget(s, "LiveStack/clip_threshold",     D["clip_threshold"],     float)
        self.max_fwhm           = _qget(s, "LiveStack/max_fwhm",           D["max_fwhm"],           float)
        self.max_ecc            = _qget(s, "LiveStack/max_ecc",            D["max_ecc"],            float)
        self.min_star_count     = _qget(s, "LiveStack/min_star_count",     D["min_star_count"],     int)
        self.narrowband_mapping = _qget(s, "LiveStack/narrowband_mapping", D["narrowband_mapping"], str)
        self.star_trail_mode    = _qget(s, "LiveStack/star_trail_mode",    False, bool)
        self.FILE_STABLE_SECS     = _qget(s, "LiveStack/file_stable_secs", D["file_stable_secs"],   float)
        self.star_target        = _qget(s, "LiveStack/star_target",        D["star_target"],        int)
        self.fits_only          = _qget(s, "LiveStack/fits_only",          D["fits_only"],          bool)
        self.weight_frames      = _qget(s, "LiveStack/weight_frames",      D["weight_frames"],      bool)
        self.snr_goal           = _qget(s, "LiveStack/snr_goal",           D["snr_goal"],           float)
        self.auto_stretch       = _qget(s, "LiveStack/auto_stretch",      True,  bool)
        # Until set here, follow the Stacking Suite's Debayer choice (RCD default)
        _ss_dm = str(s.value("stacking/debayer_method", "rcd") or "rcd").lower()
        self.debayer_method     = str(_qget(s, "LiveStack/debayer_method", _ss_dm, str)).lower()
        if self.debayer_method not in dict((t, l) for l, t in DEBAYER_METHODS):
            self.debayer_method = "rcd"


        self.total_exposure = 0.0  # seconds
        self.exposure_label = QLabel(self.tr("Total Exp: {0}").format("00:00:00"))
        self.exposure_label.setStyleSheet("color: #cccccc; font-weight: bold;")

        self.brightness = 0.0   # [-1.0..+1.0]
        self.contrast   = 1.0   # [0.1..3.0]



        # ─── Create Separate Metrics Window (initially hidden) ─────
        # We do NOT embed this in the stacking dialog’s layout!
        self.metrics_window = LiveMetricsWindow(self)
        self.metrics_window.hide()

        # ─── UI ELEMENTS FOR STACKING DIALOG ───────────────────────
        # 1) Folder selection
        self.folder_label = QLabel(self.tr("Folder: (none)"))
        self.select_folder_btn = QPushButton(self.tr("Select Folder…"))
        self.select_folder_btn.clicked.connect(self.select_folder)

        # 2) Load master dark/flat
        self.load_darks_btn = QPushButton(self.tr("Load Master Dark…"))
        self.load_darks_btn.clicked.connect(self.load_masters)
        self.load_flats_btn = QPushButton(self.tr("Load Master Flat…"))
        self.load_flats_btn.clicked.connect(self.load_masters)
        self.load_filter_flats_btn = QPushButton(self.tr("Load MonoFilter Flats…"))
        self.load_filter_flats_btn.clicked.connect(self.load_filter_flats)        

        # 2b) Cull folder selection
        self.cull_folder_label = QLabel(self.tr("Cull Folder: (none)"))
        self.select_cull_btn = QPushButton(self.tr("Select Cull Folder…"))
        self.select_cull_btn.clicked.connect(self.select_cull_folder)

        self.dark_status_label = QLabel(self.tr("Dark: ❌"))
        self.flat_status_label = QLabel(self.tr("Flat: ❌"))
        for lbl in (self.dark_status_label, self.flat_status_label):
            lbl.setStyleSheet("color: #cccccc; font-weight: bold;")
        # 3) “Process & Monitor” / “Monitor Only” / “Stop” / “Reset”
        self.mono_color_checkbox = QCheckBox(self.tr("Mono → Color Stacking"))
        self.mono_color_checkbox.setToolTip(self.tr(
            "When checked, bucket mono frames by FILTER and composite R, G, B, Ha, OIII, SII."
        ))
        # **Connect the toggled(bool) signal** before we ever call it
        self.mono_color_checkbox.toggled.connect(self._on_mono_color_toggled)

        # ** new: Star-Trail mode checkbox **
        self.star_trail_checkbox = QCheckBox(self.tr("★★ Star-Trail Mode ★★"))
        self.star_trail_checkbox.setChecked(self.star_trail_mode)
        self.star_trail_checkbox.setToolTip(self.tr("If checked, build a max-value trail instead of a running stack"))
        self.star_trail_checkbox.toggled.connect(self._on_star_trail_toggled)

        self.process_and_monitor_btn = QPushButton(self.tr("Process && Monitor"))
        self.process_and_monitor_btn.clicked.connect(self.start_and_process)
        self.monitor_only_btn = QPushButton(self.tr("Monitor Only"))
        self.monitor_only_btn.clicked.connect(self.start_monitor_only)
        self.stop_btn = QPushButton(self.tr("Stop"))
        self.stop_btn.clicked.connect(self.stop_live)
        self.reset_btn = QPushButton(self.tr("Reset"))
        self.reset_btn.clicked.connect(self.reset_live)

        self.frame_count_label = QLabel(self.tr("Frames: {0}").format(0))

        # 4) Live‐stack preview area (QGraphicsView)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self._did_initial_fit = False
        self.view.rubberBandChanged.connect(self._on_rubber_band)

        # 5) Zoom toolbar + Settings icon
        tb = QToolBar()

        zi = QAction(QIcon.fromTheme("zoom-in"), self.tr("Zoom In"), self)
        zo = QAction(QIcon.fromTheme("zoom-out"), self.tr("Zoom Out"), self)
        fit = QAction(QIcon.fromTheme("zoom-fit-best"), self.tr("Fit to Window"), self)

        tb.addAction(zi)
        tb.addAction(zo)
        tb.addAction(fit)

        # Preview: statistical stretch (default) or the linear stack as it is
        tb.addSeparator()
        self.act_autostretch = QAction(self.tr("Auto Stretch"), self)
        self.act_autostretch.setCheckable(True)
        self.act_autostretch.setChecked(bool(self.auto_stretch))
        self.act_autostretch.setToolTip(self.tr(
            "Show the stack with a statistical stretch. This only changes the preview;\n"
            "the stack itself stays linear. Uncheck to see the linear stack as it is."))
        self.act_autostretch.toggled.connect(self._on_autostretch_toggled)
        tb.addAction(self.act_autostretch)

        # SNR region: drag a box over faint signal → SNR and frames-to-goal
        tb.addSeparator()
        self.act_snr_region = QAction(self.tr("SNR Region"), self)
        self.act_snr_region.setCheckable(True)
        self.act_snr_region.setToolTip(self.tr(
            "Drag a box over the faint signal you care about. Live Stacking reports its\n"
            "signal-to-noise per pixel and how many more frames reach the SNR goal\n"
            "(set in Settings)."))
        self.act_snr_region.toggled.connect(self._on_snr_pick_toggled)
        tb.addAction(self.act_snr_region)
        self.act_snr_clear = QAction(self.tr("Clear Region"), self)
        self.act_snr_clear.setToolTip(self.tr("Remove the SNR region"))
        self.act_snr_clear.triggered.connect(self._clear_snr_region)
        tb.addAction(self.act_snr_clear)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        # In-tool documentation button, left of the Settings wrench
        self.btn_help = make_help_button("live_stacking", self)
        tb.addWidget(self.btn_help)

        # — Replace the QAction “wrench” with a styled QToolButton —
        self.wrench_button = QToolButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.wrench_button.setToolTip(self.tr("Settings"))
        # Apply your stylesheet to the QToolButton
        self.wrench_button.setStyleSheet("""
            QToolButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #FF6347;
            }
        """)
        # Connect the clicked signal to open_settings()
        self.wrench_button.clicked.connect(self.open_settings)

        # Add the styled QToolButton into the toolbar
        tb.addWidget(self.wrench_button)

        zi.triggered.connect(self.zoom_in)
        zo.triggered.connect(self.zoom_out)
        fit.triggered.connect(self.fit_to_window)


        # 6) Brightness & Contrast sliders
        bright_slider = QSlider(Qt.Orientation.Horizontal)
        bright_slider.setRange(-100, 100)
        bright_slider.setValue(0)
        bright_slider.setToolTip(self.tr("Brightness"))
        bright_slider.valueChanged.connect(self.on_brightness_changed)

        contrast_slider = QSlider(Qt.Orientation.Horizontal)
        contrast_slider.setRange(10, 1000)
        contrast_slider.setValue(100)
        contrast_slider.setToolTip(self.tr("Contrast"))
        contrast_slider.valueChanged.connect(self.on_contrast_changed)

        bc_layout = QHBoxLayout()
        bc_layout.addWidget(QLabel(self.tr("Brightness")))
        bc_layout.addWidget(bright_slider)
        bc_layout.addWidget(QLabel(self.tr("Contrast")))
        bc_layout.addWidget(contrast_slider)

        # 7) “Send to Slot” button
        self.open_linear_btn = QPushButton(self.tr("Open Linear Stack ▶"))
        self.open_linear_btn.setToolTip(self.tr(
            "Open the linear 32-bit stack as a new image, for processing\n"
            "(colour calibration, gradient removal, your own stretch)."))
        self.open_linear_btn.clicked.connect(lambda _=False: self.send_to_new_view("linear"))
        self.open_stretched_btn = QPushButton(self.tr("Open Stretched ▶"))
        self.open_stretched_btn.setToolTip(self.tr(
            "Open the Auto Stretch image with your brightness/contrast as a new image.\n"
            "Stretched data: not for further linear processing."))
        self.open_stretched_btn.clicked.connect(lambda _=False: self.send_to_new_view("stretched"))
        self.open_filters_btn = QPushButton(self.tr("Open Each Filter ▶"))
        self.open_filters_btn.setToolTip(self.tr(
            "Mono → Color: open every filter's linear stack as its own mono image."))
        self.open_filters_btn.clicked.connect(lambda _=False: self.send_to_new_view("filters"))
        open_row = QHBoxLayout()
        for b in (self.open_linear_btn, self.open_stretched_btn, self.open_filters_btn):
            open_row.addWidget(b)

        # 8) “Show Metrics” button
        self.show_metrics_btn = QPushButton(self.tr("Show Metrics"))
        self.show_metrics_btn.clicked.connect(self.show_metrics_window)

        # ─── ASSEMBLE MAIN LAYOUT (exactly one setLayout call!) ─────
        main_layout = QVBoxLayout()

        # A) Top‐row controls
        controls = QHBoxLayout()
        controls.addWidget(self.select_folder_btn)
        controls.addWidget(self.load_darks_btn)
        controls.addWidget(self.load_flats_btn)
        controls.addWidget(self.load_filter_flats_btn)
        controls.addWidget(self.select_cull_btn)
        controls.addStretch()
        controls.addWidget(self.mono_color_checkbox)
        controls.addWidget(self.star_trail_checkbox)
        controls.addWidget(self.process_and_monitor_btn)
        controls.addWidget(self.monitor_only_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.reset_btn)
        main_layout.addLayout(controls)

        # B) Status line: folder label + frame count
        status_line = QHBoxLayout()
        status_line.addWidget(self.folder_label)
        status_line.addWidget(self.dark_status_label)
        status_line.addWidget(self.flat_status_label)
        status_line.addWidget(self.cull_folder_label)
        status_line.addStretch()        
        status_line.addWidget(self.frame_count_label)
        status_line.addWidget(self.exposure_label)
        main_layout.addLayout(status_line)

        # C) Zoom toolbar
        main_layout.addWidget(tb)

        # D) Show Metrics button (separate window)
        main_layout.addWidget(self.show_metrics_btn)

        # E) Live‐stack preview area
        main_layout.addWidget(self.view)

        # F) Brightness/Contrast sliders
        main_layout.addLayout(bc_layout)

        # G) “Send to Slot” + mode/idle labels
        main_layout.addLayout(open_row)
        self.mode_label = QLabel(self.tr("Mode: Weighted Average"))
        self.mode_label.setStyleSheet("color: #a0a0a0;")
        main_layout.addWidget(self.mode_label)
        self.status_label = QLabel(self.tr("Idle"))
        self.status_label.setStyleSheet("color: #a0a0a0;")
        main_layout.addWidget(self.status_label)
        self.snr_label = QLabel("")
        self.snr_label.setWordWrap(True)
        self.snr_label.setStyleSheet("color: #e0c060;")
        main_layout.addWidget(self.snr_label)

        # Finalize
        self.setLayout(main_layout)

        self._on_mono_color_toggled(self.mono_color_checkbox.isChecked())
        self._update_open_buttons()

        app = QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(self._on_app_quit)
            except Exception:
                pass

    def _should_stop(self) -> bool:
        # stop requested or not running anymore
        return self._stop_event.is_set() or (not self.is_running)


    # ─────────────────────────────────────────────────────────────────────────
    def _on_star_trail_toggled(self, checked: bool):
        """Enable/disable star-trail mode."""
        self.star_trail_mode = checked
        QSettings().setValue("LiveStack/star_trail_mode", checked)
        self.mode_label.setText(self.tr("Mode: Star-Trail") if checked else self.tr("Mode: Weighted Average"))
        # if you want, disable mono/color checkbox when star-trail is on:
        self.mono_color_checkbox.setEnabled(not checked)
        self._update_open_buttons()

    def _on_mono_color_toggled(self, checked: bool):
        with self._lock:
            self.mono_color_mode = checked
            self._gen += 1        # a frame in flight was keyed for the old mode
            # per-filter stacks only exist in Mono → Color mode; drop them on a switch
            for k in [k for k in self.stacks if k != "__main__"]:
                self.stacks.pop(k, None)
                self.norm_targets.pop(k, None)
            self._display_rev += 1
            self._preview_cache = None

        msg = self.tr("Enabled") if checked else self.tr("Disabled")
        self.status_label.setText(self.tr("Mono→Color Mode {0}").format(msg))
        self._update_open_buttons()

    def _update_open_buttons(self):
        if not hasattr(self, "open_linear_btn"):
            return
        trail = bool(self.star_trail_mode)
        mono_color = bool(getattr(self, "mono_color_mode", False))
        if trail:
            self.open_linear_btn.setText(self.tr("Open Star-Trail Image ▶"))
        elif mono_color:
            self.open_linear_btn.setText(self.tr("Open Linear Color Composite ▶"))
        else:
            self.open_linear_btn.setText(self.tr("Open Linear Stack ▶"))
        self.open_stretched_btn.setVisible(not trail)
        self.open_filters_btn.setVisible(mono_color and not trail)

    def show_metrics_window(self):
        """Pop up the separate metrics window (never embed it here)."""
        self.metrics_window.show()
        self.metrics_window.raise_()


    def select_cull_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Cull Folder"))
        if folder:
            self.cull_folder = folder
            self.cull_folder_label.setText(self.tr("Cull: {0}").format(os.path.basename(folder)))

    def _cull_frame(self, path: str, reason: str = "") -> str:
        """
        Move a flagged frame into the cull folder (if set). Returns the status
        message; runs on the watch thread, so it touches no widgets.
        """
        name = os.path.basename(path)
        why = f" ({reason})" if reason else ""
        if self.cull_folder:
            try:
                os.makedirs(self.cull_folder, exist_ok=True)
                dst = os.path.join(self.cull_folder, name)
                shutil.move(path, dst)
                return self.tr("⚠ Culled {0}{1} → {2}").format(name, why, self.cull_folder)
            except Exception:
                return self.tr("⚠ Failed to cull {0}{1}").format(name, why)
        return self.tr("⚠ Not stacked: {0}{1}").format(name, why)

    def open_settings(self):
        dlg = LiveStackSettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            (bs, sigma, fwhm, ecc, stars, mapping, delay, star_target, fits_only,
             weight_frames, snr_goal, debayer_method) = dlg.getValues()

            # 1) Persist into QSettings
            s = QSettings()
            s.setValue("LiveStack/bootstrap_frames",   bs)
            s.setValue("LiveStack/clip_threshold",     sigma)
            s.setValue("LiveStack/max_fwhm",           fwhm)
            s.setValue("LiveStack/max_ecc",            ecc)
            s.setValue("LiveStack/min_star_count",     stars)
            s.setValue("LiveStack/narrowband_mapping", mapping)
            s.setValue("LiveStack/file_stable_secs",   delay)
            s.setValue("LiveStack/star_target",        star_target)
            s.setValue("LiveStack/fits_only",          fits_only)
            s.setValue("LiveStack/weight_frames",      weight_frames)
            s.setValue("LiveStack/snr_goal",           snr_goal)
            s.setValue("LiveStack/debayer_method",     debayer_method)

            # 2) Apply to this live‐stack session
            self.bootstrap_frames   = bs
            self.clip_threshold     = sigma
            self.max_fwhm           = fwhm
            self.max_ecc            = ecc
            self.min_star_count     = stars
            self.narrowband_mapping = mapping
            self.FILE_STABLE_SECS   = delay
            self.star_target        = star_target
            self.fits_only          = fits_only
            self.weight_frames      = weight_frames
            self.snr_goal           = snr_goal
            self.debayer_method     = debayer_method
            with self._lock:
                for acc in self.stacks.values():
                    acc.k = float(sigma)   # bootstrap length applies to new stacks
                self._display_rev += 1     # mapping may have changed

            self.status_label.setText(
                self.tr("↺ Settings saved: BS={0}, σ={1:.1f}, "
                        "FWHM≤{2:.1f}, ECC≤{3:.2f}, Stars≥{4}, "
                        "Mapping={5}, Stars to detect={6}, "
                        "FITS-only={7}, "
                        "Weights={8}, SNR goal={9:g}, "
                        "Debayer={10}").format(
                    bs, sigma, fwhm, ecc, stars, mapping, star_target,
                    self.tr('on') if fits_only else self.tr('off'),
                    self.tr('on') if weight_frames else self.tr('off'),
                    snr_goal, debayer_method.upper())
            )
            self._preview_cache = None
            self._refresh_preview()
            self._refresh_snr_readout()
            QApplication.processEvents()

    def _watch_extensions(self):
        """Glob patterns to scan for in the watch folder, honoring 'FITS only' setting."""
        if getattr(self, "fits_only", False):
            return ("*.fit", "*.fits")
        return (
            "*.fit", "*.fits", "*.tif", "*.tiff",
            "*.cr2", "*.cr3", "*.nef", "*.arw",
            "*.dng", "*.raf", "*.orf", "*.rw2", "*.pef", "*.xisf",
            "*.png", "*.jpg", "*.jpeg",
        )

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)

    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # — Brightness / Contrast —

    def _refresh_preview(self):
        """
        Re-show the preview. The display image is cached, so brightness /
        contrast changes don't re-stretch the stack.
        """
        if self._preview_cache is None:
            self._preview_cache = self._display_image()
        if self._preview_cache is not None:
            self.update_preview(self._preview_cache)

    def _on_autostretch_toggled(self, on: bool):
        self.auto_stretch = bool(on)
        QSettings().setValue("LiveStack/auto_stretch", self.auto_stretch)
        with self._lock:
            self._display_rev += 1
        self._preview_cache = None
        self._refresh_preview()
        self.status_label.setText(
            self.tr("Preview: auto stretch (display only)") if on else
            self.tr("Preview: the linear stack, no stretch"))

    def on_brightness_changed(self, val: int):
        self.brightness = val / 100.0  # map to [-1,1]
        self._refresh_preview()

    def on_contrast_changed(self, val: int):
        self.contrast = val / 100.0   # map to [0.1,10.0]
        self._refresh_preview()

    # — Sending out —

    def _is_multi_filter(self) -> bool:
        return bool(self.mono_color_mode and any(k != "__main__" for k in list(self.stacks)))

    def send_to_new_view(self, kind: str = "linear"):
        """
        Open the live result as new view(s).

        kind "linear"    – the linear 32-bit stack (Mono → Color: the filters
                           combined with the narrowband mapping, still linear)
        kind "stretched" – the Auto Stretch image with brightness/contrast,
                           whether or not Auto Stretch is on for the preview
        kind "filters"   – Mono → Color: each filter as its own linear mono master
        Star-Trail mode always opens the trail image as shown.
        """
        dm = self._docman
        if dm is None:
            mw = self.parent
            dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        if dm is None or not hasattr(dm, "open_array"):
            QMessageBox.warning(self, self.tr("Open Failed"),
                                self.tr("No document manager is available to open the result."))
            return

        stamp = f"{_dt.datetime.now():%Y%m%d_%H%M%S}"
        outs = []   # (image, title, header, frames)
        with self._lock:
            if self.star_trail_mode:
                if self.current_stack is not None:
                    outs.append((np.clip(self.current_stack, 0.0, 1.0).astype(np.float32),
                                 f"LiveStack_StarTrail_{stamp}_{self.frame_count}f", None,
                                 self.frame_count))
            else:
                multi = self._is_multi_filter()
                stacks = dict(self.stacks)
                if multi:
                    use = {k: a for k, a in stacks.items() if k != "__main__" and a.shape[2] == 1}
                else:
                    use = {k: a for k, a in stacks.items() if k == "__main__"}
                n_all = sum(a.n_frames for a in use.values())
                if kind == "filters" and multi:
                    for key, acc in stacks.items():
                        img = np.clip(acc.mu, 0.0, 1.0).astype(np.float32)   # a copy
                        if img.shape[2] == 1:
                            img = img[..., 0]
                        label = "" if key == "__main__" else f"_{self._key_label(key)}"
                        outs.append((img, f"LiveStack{label}_{stamp}_{acc.n_frames}f",
                                     self._result_header(key, acc), acc.n_frames))
                elif kind == "stretched":
                    disp = self._display_image(stretch=True)
                    if disp is not None and use:
                        outs.append((self._apply_bc(disp),
                                     f"LiveStack_Stretched_{stamp}_{n_all}f",
                                     self._composite_header(use, stretched=True), n_all))
                elif use:
                    if multi:
                        img = self._build_color_composite(
                            {k: a.mu[..., 0] for k, a in use.items()})
                    else:
                        img = use["__main__"].mu
                        img = img[..., 0] if img.shape[2] == 1 else img
                    outs.append((np.clip(img, 0.0, 1.0).astype(np.float32),
                                 f"LiveStack_{stamp}_{n_all}f",
                                 self._composite_header(use), n_all))

        if not outs:
            self.status_label.setText(self.tr("⚠ Nothing to open"))
            return
        for img, title, hdr, n in outs:
            meta = {"source": "LiveStack", "frames_stacked": int(n),
                    "bit_depth": "32-bit floating point", "is_mono": img.ndim == 2}
            if hdr is not None:
                meta["original_header"] = hdr
            try:
                dm.open_array(img, metadata=meta, title=title)
            except Exception as e:
                QMessageBox.warning(self, self.tr("Open Failed"), self.tr("Could not open {0}:\n{1}").format(title, e))
                return
        what = ("" if self.star_trail_mode else
                self.tr(" (stretched)") if kind == "stretched" else self.tr(" (linear)"))
        self.status_label.setText(self.tr("Opened {0} new view(s){1}").format(len(outs), what))

    def _composite_header(self, use, stretched: bool = False):
        """Header for a result built from one stack or several filter stacks."""
        keys = list(use)
        h = self._result_header(keys[0], use[keys[0]])
        if len(keys) > 1:
            h["NCOMBINE"] = (int(sum(a.n_frames for a in use.values())), "frames stacked")
            tot = sum(self._exp_by_key.get(k, (0.0, 0))[0] for k in keys)
            cnt = sum(self._exp_by_key.get(k, (0.0, 0))[1] for k in keys)
            if cnt > 0:
                h["EXPTIME"] = (float(tot / cnt), "mean exposure per frame (s)")
                h["TOTALEXP"] = (float(tot), "total integration (s)")
            if "FILTER" in h:
                del h["FILTER"]
            parts = ", ".join(f"{self._key_label(k)} {use[k].n_frames}f" for k in keys)
            h["HISTORY"] = f"Live Stacking composite ({self.narrowband_mapping}): {parts}"
        if stretched:
            h["HISTORY"] = "Stretched for display (statistical stretch): not linear data"
        return h

    def _result_header(self, key, acc):
        h = self._first_header.get(key)
        h = h.copy() if h is not None else fits.Header()
        h["IMAGETYP"] = ("Master Light", "SASpro Live Stacking")
        h["NCOMBINE"] = (int(acc.n_frames), "frames stacked")
        e = self._exp_by_key.get(key)
        if e and e[1] > 0:
            h["EXPTIME"] = (float(e[0] / e[1]), "mean exposure per frame (s)")
            h["TOTALEXP"] = (float(e[0]), "total integration (s)")
        if key != "__main__":
            h["FILTER"] = self._key_label(key)
        h["HISTORY"] = "SASpro Live Stacking: linear, weighted average, sigma rejection"
        return h


    # ── New helper: map header["FILTER"] to a single letter key
    def _get_filter_key(self, header):
        """FITS FILTER -> 'L','R','G','B','H' (Ha),'O' (OIII),'S' (SII), or None."""
        return _lsw.filter_key(header)

    def _build_color_composite(self, planes):
        """
        Composite per-filter display planes (key → stretched H×W) into an RGB
        preview according to self.narrowband_mapping:

        • "Natural":
            – If SII present:
                R = 0.5*(Ha + SII)
                G = 0.5*(SII + OIII)
                B = OIII
            – Elif any R/G/B loaded:
                R = R_filter
                G = G_filter + OIII
                B = B_filter + OIII
            – Else (no SII, no R/G/B):
                R = Ha
                G = OIII
                B = OIII

        • Any 3-letter code (e.g. "SHO", "OHS"):
            R = planes[mapping[0]]
            G = planes[mapping[1]]
            B = planes[mapping[2]]

        Missing channels default to zero.
        """
        # 1) Determine H, W
        if not planes:
            return None
        H, W = next(iter(planes.values())).shape[:2]

        # helper: get plane or zeros
        def getf(k):
            return planes.get(k, np.zeros((H, W), np.float32))

        mode = self.narrowband_mapping.upper()
        if mode == "NATURAL":
            Ha = getf('H')
            O3 = getf('O')
            S2 = planes.get('S', None)
            Rf = planes.get('R', None)
            Gf = planes.get('G', None)
            Bf = planes.get('B', None)

            if S2 is not None:
                # narrowband SII branch
                R = 0.5 * (Ha + S2)
                G = 0.5 * (S2 + O3)
                B = O3.copy()

            elif any(x is not None for x in (Rf, Gf, Bf)):
                # broadband branch: Rf/Gf/Bf with OIII boost
                R = Rf if Rf is not None else np.zeros((H, W), np.float32)
                G = (Gf if Gf is not None else np.zeros((H, W), np.float32)) + O3
                B = (Bf if Bf is not None else np.zeros((H, W), np.float32)) + O3

            else:
                # fallback HOO
                R = Ha
                G = O3
                B = O3

        else:
            # direct mapping: e.g. "SHO" → R=S, G=H, B=O
            letters = list(mode)
            if len(letters) != 3 or any(l not in ("S","H","O") for l in letters):
                # fallback to natural
                self.narrowband_mapping = "Natural"
                return self._build_color_composite(planes)

            R = getf(letters[0])
            G = getf(letters[1])
            B = getf(letters[2])

        return np.stack([R, G, B], axis=2)


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Folder to Watch"))
        if folder:
            self.watch_folder = folder
            self.folder_label.setText(self.tr("Folder: {0}").format(os.path.basename(folder)))

    def load_masters(self):
        """
        When the user picks “Load Master Dark…” or “Load Master Flat…”, we load exactly one file
        (the first in the dialog).  We simply store it in `self.master_dark` or `self.master_flat`,
        but we also check its dimensions against any existing master so that the user can’t load
        a 2D flat while the dark is 3D (for example).
        """
        sender = self.sender()
        dlg = QFileDialog(self, self.tr("Select Master Files"),
                         filter="FITS TIFF or XISF (*.fit *.fits *.tif *.tiff *.xisf)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if not dlg.exec():
            return

        chosen = dlg.selectedFiles()[0]
        img, hdr, bit_depth, is_mono = load_image(chosen)
        if img is None:
            QMessageBox.warning(self, self.tr("Load Error"),
                                self.tr("Failed to load master file:\n{0}").format(chosen))
            return

        # Convert everything to float32 for consistency
        img = img.astype(np.float32)

        # Which button, by identity: its text is translated.
        if sender is self.load_darks_btn:
            # If a flat is already loaded, ensure shape‐compatibility
            if self.master_flat is not None:
                if not self._shapes_compatible(master=img, other=self.master_flat):
                    QMessageBox.warning(
                        self, self.tr("Shape Mismatch"),
                        self.tr("Cannot load this master dark: it has incompatible shape "
                                "vs. the already‐loaded master flat.")
                    )
                    return

            self.master_dark = img
            self.dark_status_label.setText(self.tr("Dark: ✅"))
            self.dark_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")            
            QMessageBox.information(
                self, self.tr("Master Dark Loaded"),
                self.tr("Loaded master dark:\n{0}").format(os.path.basename(chosen))
            )
        else:
            # "Flat" was clicked
            if self.master_dark is not None:
                if not self._shapes_compatible(master=self.master_dark, other=img):
                    QMessageBox.warning(
                        self, self.tr("Shape Mismatch"),
                        self.tr("Cannot load this master flat: it has incompatible shape "
                                "vs. the already‐loaded master dark.")
                    )
                    return

            self.master_flat = img
            self.flat_status_label.setText(self.tr("Flat: ✅"))
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")            
            QMessageBox.information(
                self, self.tr("Master Flat Loaded"),
                self.tr("Loaded master flat:\n{0}").format(os.path.basename(chosen))
            )

    def load_filter_flats(self):
        """
        Let the user pick one or more flat files.
        We try to read the FITS header FILTER key to decide which filter
        each flat belongs to; otherwise fall back to the filename.
        """
        dlg = QFileDialog(self, self.tr("Select Filter Flats"),
                          filter="FITS or TIFF (*.fit *.fits *.tif *.tiff)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if not dlg.exec():
            return

        files = dlg.selectedFiles()
        loaded = []
        for path in files:
            img, hdr, bit_depth, is_mono = load_image(path)
            if img is None:
                continue
            # guess filter key from header, else from filename
            key = None
            if hdr and hdr.get("FILTER"):
                key = self._get_filter_key(hdr)
            if not key:
                # fallback: basename before extension
                key = os.path.splitext(os.path.basename(path))[0]

            # store it
            self.master_flats[key] = img.astype(np.float32)
            loaded.append(key)

        # update the flat status label to list loaded filters
        if loaded:
            names = ", ".join(loaded)
            self.flat_status_label.setText(self.tr("Flats: {0}").format(names))
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")
            QMessageBox.information(
                self, self.tr("Filter Flats Loaded"),
                self.tr("Loaded flats for filters: {0}").format(names)
            )
        else:
            QMessageBox.warning(self, self.tr("No Flats Loaded"),
                                self.tr("No flats could be loaded."))

    def _shapes_compatible(self, master: np.ndarray, other: np.ndarray) -> bool:
        """
        Return True if `master` and `other` can be used together in calibration:
          - Exactly the same shape, OR
          - master is 2D (H×W) and other is 3D (H×W×3), OR
          - vice versa.
        """
        if master.shape == other.shape:
            return True

        # If one is 2D and the other is H×W×3, check the first two dims
        if master.ndim == 2 and other.ndim == 3 and other.shape[:2] == master.shape:
            return True
        if other.ndim == 2 and master.ndim == 3 and master.shape[:2] == other.shape:
            return True

        return False

    def start_and_process(self):
        """Process everything currently in folder, then begin monitoring."""
        if not self.watch_folder:
            self.status_label.setText(self.tr("❗ No folder selected"))
            return
        self._wait_worker()
        self.processed_files.clear()
        self._start_worker()
        self.status_label.setText(self.tr("▶ Processing & Monitoring: {0}").format(os.path.basename(self.watch_folder)))

    def start_monitor_only(self):
        """Mark existing files as seen and only process new arrivals."""
        if not self.watch_folder:
            self.status_label.setText(self.tr("❗ No folder selected"))
            return
        self._wait_worker()
        # Populate processed_files with all existing files so they won't be re-processed
        exts = self._watch_extensions()
        all_paths = []
        for ext in exts:
            all_paths += glob.glob(os.path.join(self.watch_folder, "**", ext), recursive=True)
        self.processed_files = {self._norm_path(p) for p in all_paths}

        # Start monitoring
        self._start_worker()
        self.status_label.setText(self.tr("▶ Monitoring Only: {0}").format(os.path.basename(self.watch_folder)))

    def start_live(self):
        if not self.watch_folder:
            self.status_label.setText(self.tr("❗ No folder selected"))
            return
        self._wait_worker()
        self._start_worker()
        self.status_label.setText(self.tr("▶ Monitoring: {0}").format(os.path.basename(self.watch_folder)))
        self.mode_label.setText(self.tr("Mode: Weighted Average"))

    def _wait_worker(self, msecs: int = -1) -> bool:
        """Wait for a stopping watch thread to finish its current frame."""
        w = self._worker
        if w is None or not w.isRunning():
            return True
        self._stop_event.set()
        self.status_label.setText(self.tr("Finishing the current frame…"))
        QApplication.processEvents()
        return bool(w.wait() if msecs < 0 else w.wait(msecs))

    def _start_worker(self):
        self._stop_event.clear()
        self.is_running = True
        th = _LiveWatchThread(self)
        th.frameDone.connect(self._apply_result)
        th.message.connect(self._set_status)
        th.finished.connect(self._on_worker_finished)
        th.finished.connect(lambda t=th: _LIVE_THREADS.discard(t))
        _LIVE_THREADS.add(th)
        self._worker = th
        th.start()

    def _on_worker_finished(self):
        if not self.is_running:
            self.status_label.setText(self.tr("■ Stopped"))

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_app_quit(self):
        self.stop_live()
        self._wait_worker(15000)
        self._shutdown_pool()
        self._cleanup_xfer()

    def stop_live(self):
        self._stop_event.set()      # request cancellation immediately
        self.is_running = False
        w = self._worker
        if w is not None and w.isRunning():
            self.status_label.setText(self.tr("■ Stopping after the current frame…"))
        else:
            self.status_label.setText(self.tr("■ Stopped"))

    def reset_live(self):
        self.stop_live()
        with self._lock:
            self._reset_state()
        self._notes_pending.clear()
        self.snr_label.setText("")
        self._draw_sky_box(None)

        # NEW: clear the metrics panel
        self.metrics_window.metrics_panel.clear_all()

        # Update labels
        self.frame_count_label.setText(self.tr("Frames: {0}").format(0))
        self.exposure_label.setText(self.tr("Total Exp: {0}").format("00:00:00"))
        self.status_label.setText(self.tr("↺ Reset"))
        self.mode_label.setText(self.tr("Mode: Weighted Average"))

        # Clear the displayed image
        self.pixmap_item.setPixmap(QPixmap())

        # Reset zoom/pan fit flag
        self._did_initial_fit = False

    def _reset_state(self):
        # a frame still in flight on the watch thread is dropped (_gen)
        self._gen += 1
        self._sky_box.clear()
        self._seen_by_key.clear()
        self._star_sigma.clear()
        self.processed_files.clear()
        self.frame_count = 0
        self.current_stack = None

        self.total_exposure = 0.0

        self.stacks.clear()
        self.norm_targets.clear()
        self._fwhm_hist.clear()
        self._weight_hist.clear()
        self._noise_hist.clear()
        self._exp_by_key.clear()
        self._first_header.clear()
        self._frames_seen = 0
        self._last_key = None
        self._preview_cache = None
        self.reference_image_2d = None
        #self.master_dark = None
        #self.master_flat = None
        #self.dark_status_label.setText("Dark: ❌")
        #self.flat_status_label.setText("Flat: ❌")
        #self.dark_status_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        #self.flat_status_label.setStyleSheet("color: #cccccc; font-weight: bold;")        


    def _update_probe(self, path: str) -> dict:
        orig_path = path
        path = self._norm_path(path)
        try:
            st = os.stat(path)
        except FileNotFoundError:
            self._probe.pop(path, None)
            return None
        now = time.time()
        size, mtime = st.st_size, st.st_mtime
        info = self._probe.get(path)
        if info is None:
            info = {"size": size, "mtime": mtime, "since": now, "penalty_until": 0.0}
            self._probe[path] = info
            return info

        # If size or mtime changed, reset stability timer
        if size != info["size"] or mtime != info["mtime"]:
            info["size"] = size
            info["mtime"] = mtime
            info["since"] = now
        return info

    def _can_open_for_read(self, path: str) -> bool:
        """
        Try a tiny open+read to ensure the writer has released the handle.
        If we hit PermissionError / OSError, we mark a penalty and say 'not ready'.
        """
        orig_path = path
        path = self._norm_path(path)


        try:
            with open(path, "rb") as f:
                _ = f.read(1)
            return True
        except (PermissionError, OSError):
            # mark a penalty window so we don't hammer the file immediately
            info = self._probe.get(path) or self._update_probe(path)
            if info:
                info["penalty_until"] = time.time() + self.OPEN_RETRY_PENALTY_SECS
            return False

    def _file_ready(self, path: str) -> bool:
        """
        A file is 'ready' when:
        - it exists,
        - we are not inside a penalty window,
        - size + mtime have been unchanged for FILE_STABLE_SECS,
        - and we can actually open it for reading.
        """
        # Normalize once and use consistently
        norm = self._norm_path(path)

        # Update or fetch probe info (keyed by normalized path)
        info = self._update_probe(norm)
        if info is None:
            return False  # file disappeared

        now = time.time()

        # Respect penalty window (e.g. previous permission / read failure)
        penalty_until = info.get("penalty_until", 0.0)
        if now < penalty_until:
            return False

        # Require size + mtime stability
        stable_for = now - info.get("since", now)
        if stable_for < self.FILE_STABLE_SECS:
            return False

        # Final check: confirm we can actually open it
        # (_can_open_for_read will set a new penalty if this fails)
        return self._can_open_for_read(norm)

    def check_for_new_frames(self):
        """Scan once and stack what is ready, on the calling thread."""
        if getattr(self, "_poll_busy", False):
            return
        self._poll_busy = True
        try:
            self._poll_once(self._apply_result, self._set_status)
        finally:
            self._poll_busy = False

    def _poll_once(self, emit_frame, emit_msg):
        """One scan of the watch folder, every ready frame prepared and stacked
        on the calling thread (check_for_new_frames)."""
        if self._should_stop() or not self.watch_folder:
            return
        candidates = self._scan_candidates(set())
        if candidates:
            emit_msg(self.tr("➜ New/updated files: {0}").format(len(candidates)))
        for path in candidates:
            if self._should_stop():
                break
            self._inline_frame(path, emit_frame, emit_msg)

    def _scan_candidates(self, busy):
        """Files in the watch folder not yet stacked and not being prepared."""
        if not self.watch_folder:
            return []
        all_paths = []
        for ext in self._watch_extensions():
            all_paths += glob.glob(os.path.join(self.watch_folder, '**', ext), recursive=True)
        all_norm = [(self._norm_path(p), p) for p in all_paths]
        return [orig for (norm, orig) in sorted(all_norm)
                if norm not in self.processed_files and norm not in busy]

    def _strike(self, norm, path, e, emit_msg):
        """A failed frame: cool down, retry, and give up after three tries."""
        info = self._probe.get(norm) or {"fails": 0, "penalty_until": 0.0}
        info["fails"] = info.get("fails", 0) + 1
        info["penalty_until"] = time.time() + self.OPEN_RETRY_PENALTY_SECS
        self._probe[norm] = info
        if info["fails"] >= 3:
            self.processed_files.add(norm)
            emit_msg(self.tr("⚠ Skipping {0} after {1} failures: {2}").format(os.path.basename(path), info['fails'], e))
        else:
            emit_msg(self.tr("⚠ Error on {0} (try {1}): {2}").format(os.path.basename(path), info['fails'], e))

    def _inline_frame(self, path, emit_frame, emit_msg):
        """Prepare and stack one frame on this thread. A frame is marked done
        only when it produced a result (stacked, culled or unreadable): a
        frame that wasn't ready, was stopped, or was dropped by a Reset is
        picked up again."""
        norm = self._norm_path(path)
        info = self._probe.get(norm)
        if info and time.time() < info.get("penalty_until", 0.0):
            return
        if not self._file_ready(path):
            return
        try:
            res = self._process_core(path)
        except Exception as e:
            self._strike(norm, path, e, emit_msg)
            return
        if res is not None:
            self.processed_files.add(norm)
            emit_frame(res)

    # ── worker processes ─────────────────────────────────────────────────
    def _watch_loop(self, emit_frame, emit_msg):
        """The watch thread's loop. Scans the folder every 1.5 s. Frames are
        prepared in worker processes (or on this thread when those are
        unavailable, and always in Star-Trail mode) and committed here in
        file order, so the stack is the same as stacking them one by one.
        A session's first frame is prepared alone: it becomes the reference
        every later frame is aligned to."""
        from concurrent.futures import wait as _fwait
        pool = self._get_pool(emit_msg)
        inflight = deque()         # (future, path, gen) in file order
        busy = set()
        cands = []
        next_scan = 0.0
        said_stopping = False
        while True:
            # 1) commit what has finished, in order
            while inflight and inflight[0][0].done():
                fut, path, gen = inflight.popleft()
                busy.discard(path)
                pool = self._finish_prepared(fut, path, gen, pool, emit_frame, emit_msg)
            if self._should_stop():
                if not inflight:
                    break
                if not said_stopping:
                    said_stopping = True
                    emit_msg(self.tr("■ Stopping: finishing {0} frame(s) already being prepared…").format(len(inflight)))
                _fwait([inflight[0][0]], timeout=0.2)
                continue

            # 2) scan
            now = time.time()
            if now >= next_scan:
                next_scan = now + 1.5
                cands = self._scan_candidates(busy)

            # 3) hand frames out
            while cands and not self._should_stop():
                if pool is None or self.star_trail_mode:
                    if inflight:
                        break                     # let the workers' frames land first
                    self._inline_frame(cands.pop(0), emit_frame, emit_msg)
                    continue
                if len(inflight) >= self._inflight_cap():
                    break
                if self.reference_image_2d is None and inflight:
                    break                         # the reference frame goes alone
                path = self._norm_path(cands.pop(0))
                info = self._probe.get(path)
                if info and time.time() < info.get("penalty_until", 0.0):
                    continue
                if not self._file_ready(path):
                    continue
                try:
                    fut = pool.submit(prepare_frame, self._frame_task(path))
                except Exception as e:
                    pool = self._drop_pool(emit_msg, e)
                    cands.insert(0, path)
                    continue
                inflight.append((fut, path, self._gen))
                busy.add(path)

            # 4) wait for the oldest frame or the next scan
            if inflight:
                _fwait([inflight[0][0]], timeout=0.2)
            else:
                self._stop_event.wait(0.2)

    def _finish_prepared(self, fut, path, gen, pool, emit_frame, emit_msg):
        from concurrent.futures.process import BrokenProcessPool
        try:
            prep = fut.result()
        except BrokenProcessPool as e:
            return self._drop_pool(emit_msg, e)   # not marked: retried on this thread
        except Exception as e:
            self._strike(path, path, e, emit_msg)
            return pool
        try:
            res = self._commit_prepared(prep, gen)
        except Exception as e:
            self._strike(path, path, e, emit_msg)
            return pool
        if res is not None:                        # None: dropped by Reset / mode switch
            self.processed_files.add(path)
            emit_frame(res)
        return pool

    def _n_workers(self) -> int:
        # A live session needs one frame every few minutes; more workers only
        # help catching up on a folder (Process & Monitor). Each worker holds
        # a few full-size frames, so keep the count modest.
        return max(1, min(3, (os.cpu_count() or 2) // 4))

    def _inflight_cap(self) -> int:
        n = max(1, int(self._pool_n))
        acc = next(iter(self.stacks.values()), None)
        if acc is None:
            return n
        try:
            import psutil
            frame_bytes = int(np.prod(acc.shape)) * 4
            avail = int(psutil.virtual_memory().available)
            # a frame in flight: ~6 working copies in its worker + the result here
            return max(1, min(n, int(avail * 0.6) // max(1, 8 * frame_bytes)))
        except Exception:
            return n

    def _get_pool(self, emit_msg):
        """The worker processes, started once per window and kept warm across
        Stop / Start. None when they can't run here; frames are then prepared
        on the watch thread with the same code."""
        if self._pool is not None or self._pool_failed:
            return self._pool
        try:
            from setiastro.saspro.worker_env import (
                check_process_pools, process_pools_ok, process_pools_reason)
            check_process_pools(log_fn=lambda m: print(f"[LiveStack] {m}"))
            if not process_pools_ok():
                raise RuntimeError(process_pools_reason() or "start-up check failed")
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            n = self._n_workers()
            emit_msg(self.tr("Starting {0} worker process(es)…").format(n))
            pool = ProcessPoolExecutor(
                max_workers=n, mp_context=_mp.get_context("spawn"),
                initializer=_lsw.init_worker,
                initargs=(QCoreApplication.organizationName(),
                          QCoreApplication.applicationName()))
            try:
                for f in [pool.submit(_lsw.warm) for _ in range(n)]:
                    f.result(timeout=180)       # a worker that can't import fails here
            except Exception:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            self._pool, self._pool_n = pool, n
            print(f"[LiveStack] {n} worker process(es) prepare frames")
        except Exception as e:
            self._pool_failed = True
            self._pool = None
            print(f"[LiveStack] worker processes unavailable: {type(e).__name__}: {e}")
            emit_msg(self.tr("ℹ Worker processes unavailable; frames are prepared on a "
                             "background thread instead (the window may respond more slowly)."))
        return self._pool

    def _drop_pool(self, emit_msg, e=None):
        """Workers died or can't take work: carry on on the watch thread."""
        self._pool_failed = True
        p, self._pool = self._pool, None
        if p is not None:
            try:
                p.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            print(f"[LiveStack] worker processes stopped: {type(e).__name__ if e else ''} {e or ''}")
            emit_msg(self.tr("⚠ Worker processes stopped; continuing on a background thread."))
        return None

    def _shutdown_pool(self):
        p, self._pool = self._pool, None
        if p is not None:
            try:
                p.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _frame_task(self, path, inline: bool = False) -> dict:
        """Everything prepare_frame needs. For a worker the masters and the
        reference travel as .npy files written once; in-process, as arrays."""
        put = (lambda name, a: a) if inline else self._xfer_file
        flats = {}
        if self.mono_color_mode:
            flats = {k: put(f"flat_{k}", v) for k, v in list(self.master_flats.items())
                     if v is not None}
        with self._lock:
            targets = {k: np.asarray(v, np.float32).copy() for k, v in self.norm_targets.items()}
            ref = self.reference_image_2d
        return {"path": path, "mono_color_mode": bool(self.mono_color_mode),
                "debayer_method": str(self.debayer_method),
                "star_target": int(self.star_target),
                "star_sigmas": dict(self._star_sigma),
                "max_fwhm": float(self.max_fwhm), "max_ecc": float(self.max_ecc),
                "min_star_count": int(self.min_star_count),
                "dark": put("dark", self.master_dark), "flat": put("flat", self.master_flat),
                "flats": flats, "ref": put("ref", ref), "targets": targets}

    def _xfer_file(self, name, arr):
        """Path of an .npy holding arr, written the first time this array is
        sent. A replaced array gets a new file name, so a worker never reads
        a stale cached copy."""
        if arr is None:
            return None
        ent = self._xfer.get(name)
        if ent is not None and ent[0] is arr:
            return ent[1]
        if self._xfer_dir is None or not os.path.isdir(self._xfer_dir):
            self._xfer_dir = tempfile.mkdtemp(prefix="saspro_live_")
        self._xfer_seq += 1
        p = os.path.join(self._xfer_dir, f"{name}_{self._xfer_seq}.npy")
        np.save(p, np.asarray(arr))
        if ent is not None:
            try:
                os.remove(ent[1])
            except OSError:
                pass
        self._xfer[name] = (arr, p)
        return p

    def _cleanup_xfer(self):
        self._xfer = {}
        d, self._xfer_dir = self._xfer_dir, None
        if d:
            shutil.rmtree(d, ignore_errors=True)

    def _match_master_to_image(self, master: np.ndarray, img: np.ndarray) -> np.ndarray:
        return _lsw.match_master_to_image(master, img)

    def _norm_path(self, p: str) -> str:
        # normpath fixes slashes, normcase fixes Windows case-insensitive matching
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))


    # ─────────────────────────────────────────────────────────────────────────
    # Frame pipeline (linear)
    # ─────────────────────────────────────────────────────────────────────────
    _KEY_LABELS = {"H": "Ha", "O": "OIII", "S": "SII"}

    def _key_label(self, key) -> str:
        return self._KEY_LABELS.get(key, str(key))

    def _note_once(self, tag: str, text: str):
        # may run on the watch thread: queue the note for _apply_result
        if tag in self._notes_shown:
            return
        self._notes_shown.add(tag)
        print(f"[LiveStack] {text}")
        self._notes_pending.append(text)

    def _load_calibrated(self, path):
        """Load one frame linear, calibrated on the mosaic, then debayered
        (live_stack_worker.load_calibrated). Returns (img, header, mono_key,
        None) or (None, None, None, reason)."""
        img, header, mono_key, why, notes = _lsw.load_calibrated(
            path, mono_color_mode=self.mono_color_mode, dark=self.master_dark,
            flat=self.master_flat, flats=self.master_flats,
            debayer_method=self.debayer_method)
        for tag, text in notes:
            self._note_once(tag, text)
        return img, header, mono_key, why

    def _frame_weight(self, key, sc, ecc, bg, fwhm) -> float:
        """Stacking Suite 'Balanced' weight: star count x roundness x sky
        darkness x sharpness (median FWHM of this stack / this FWHM), so live
        and offline stacks weigh a frame the same way."""
        hist = list(self._fwhm_hist.get(key, []))
        try:
            fwhm_ref = float(np.median(hist + [float(fwhm)])) if np.isfinite(fwhm) else 0.0
        except Exception:
            fwhm_ref = 0.0
        bg = float(min(max(float(bg), 0.0), 1.0)) if np.isfinite(bg) else 0.0
        try:
            from setiastro.saspro.stacking_suite import StackingSuiteDialog as _SS
            w, _dbg = _SS._score_frame_terms(
                count=sc, ecc=ecc, bg=bg, fwhm=fwhm, noise=None,
                fwhm_ref=fwhm_ref, noise_ref=None,
                exps=_SS._weight_mode_exponents("Balanced"))
            return float(w)
        except Exception:
            c = max(float(sc), 0.0)
            e = float(min(max(float(ecc), 0.0), 1.0))
            return float(c * (1.0 - e) * ((2.0 / (np.sqrt(bg) + 1.0)) - 1.0))

    def _add_exposure(self, key, header):
        exp_val = None
        if header is not None:
            exp_val = header.get("EXPOSURE", header.get("EXPTIME", None))
        try:
            secs = float(exp_val)
        except Exception:
            return
        if not np.isfinite(secs) or secs <= 0:
            return
        self.total_exposure += secs
        e = self._exp_by_key.setdefault(key, [0.0, 0])
        e[0] += secs
        e[1] += 1

    def _remember_header(self, key, header):
        if key in self._first_header or header is None:
            return
        try:
            h = fits.Header(header).copy()
        except Exception:
            return
        for kw in ("BAYERPAT", "XBAYROFF", "YBAYROFF", "BZERO", "BSCALE", "BITPIX",
                   "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "SIMPLE"):
            try:
                while kw in h:
                    del h[kw]
            except Exception:
                pass
        self._first_header[key] = h

    def process_frame(self, path):
        """Stack one frame and update the window, on the calling thread. The
        watch thread calls _process_core and hands the result back through a
        queued signal instead."""
        res = self._process_core(path)
        if res is not None:
            self._apply_result(res)

    def _result(self, gen, **kw):
        kw["gen"] = gen
        kw["frame_count"] = self.frame_count
        kw["total_exposure"] = self.total_exposure
        return kw

    def _apply_result(self, res):
        """GUI side of a processed frame: labels, metrics plots, preview."""
        if not res or res.get("gen") != self._gen:
            return      # from before a Reset / mode switch
        self.frame_count_label.setText(self.tr("Frames: {0}").format(res['frame_count']))
        self.exposure_label.setText(self.tr("Total Exp: {0}").format(_fmt_hms(res['total_exposure'])))
        if res.get("mode"):
            self.mode_label.setText(res["mode"])
        if res.get("snr_text") is not None:
            self.snr_label.setText(res["snr_text"])
        if "sky_box" in res:
            self._draw_sky_box(res["sky_box"])
        status = res.get("status") or ""
        notes, self._notes_pending[:] = list(self._notes_pending), []
        if notes:
            status = "  ·  ".join([s for s in [status] if s] + [f"ℹ {n}" for n in notes])
        if status:
            self.status_label.setText(status)
        pt = res.get("point")
        if pt is not None:
            self.metrics_window.metrics_panel.add_point(
                pt[0], pt[1], flagged=pt[2], key=(pt[3] if len(pt) > 3 else "__main__"))
        if "display" in res:
            if res.get("display_rev") != self._display_rev:
                self._preview_cache = None      # preview settings changed meanwhile
                self._refresh_preview()
            else:
                self._show_display(res["display"])

    def _process_core(self, path):
        """Stack one frame on the calling thread: prepare it in-process (the
        same prepare_frame a worker process runs), then commit it. Used by
        process_frame and when worker processes are unavailable. Returns a
        result for _apply_result, or None (not ready, stopped, or dropped by
        a Reset)."""
        if self._should_stop():
            return None
        if not self._file_ready(path):
            # do not mark as processed here; monitor will retry after cool-down
            return None
        path = self._norm_path(path)
        gen = self._gen

        # star-trail mode keeps its own (display-space) max accumulator
        if self.star_trail_mode:
            return self._star_trail_core(path, gen)

        prep = prepare_frame(self._frame_task(path, inline=True),
                             should_stop=self._should_stop)
        return self._commit_prepared(prep, gen)

    def _commit_prepared(self, prep, gen):
        """The window's side of a prepared frame: frame weight, accumulate,
        histories, noise and SNR, preview image. Runs on the watch thread (or
        the caller of process_frame); touches no widget. Returns a result for
        _apply_result, or None when the frame was stopped or a Reset / mode
        switch happened while it was being prepared."""
        if prep is None or prep.get("status") == "stopped" or gen != self._gen:
            return None
        path = prep["path"]
        name = os.path.basename(path)
        for tag, text in prep.get("notes") or []:
            self._note_once(tag, text)
        if prep["status"] == "error":
            return self._result(gen, status=f"⚠ {name}: {prep.get('why')}")

        key = prep["key"]
        sc, fwhm, ecc, bg = prep["sc"], prep["fwhm"], prep["ecc"], prep["bg"]
        header = prep.get("header")
        # The star threshold is chosen by count on the stack's first frames and
        # then held. A frame prepared before it was held chose its own; use its
        # measurement at the held threshold when it tried that one.
        held = self._star_sigma.get(key)
        own_sig = prep.get("star_sigma")
        if held is not None and own_sig is not None and float(own_sig) != float(held):
            alt = (prep.get("star_rungs") or {}).get(float(held))
            if alt is not None:
                sc, fwhm, ecc = alt
                if prep["status"] == "ok":
                    why = _lsw.cull_reason(sc, fwhm, ecc, self.max_fwhm, self.max_ecc,
                                           self.min_star_count)
                    if why is not None:
                        prep["status"], prep["why"] = "cull", why
        self._frames_seen += 1
        frame_no = self._seen_by_key.get(key, 0) + 1     # x counts this filter's frames
        self._seen_by_key[key] = frame_no

        # Cull thresholds and alignment were checked in prepare_frame; the
        # weight needs this stack's history, so it is decided here.
        reason = prep.get("why") if prep["status"] == "cull" else None
        weight = 1.0
        if reason is None and self.weight_frames:
            weight = self._frame_weight(key, sc, ecc, bg, fwhm)
            if not (weight > 0.0):
                reason = self.tr("no usable stars to weight it by")
        frame = prep.get("frame")
        acc = self.stacks.get(key)
        if reason is None and acc is not None and acc.shape != frame.shape:
            reason = self.tr("size {0}x{1}x{2} differs from "
                             "this stack's {3}x{4}x{5}").format(
                frame.shape[1], frame.shape[0], frame.shape[2],
                acc.shape[1], acc.shape[0], acc.shape[2])
        if reason is None and not prep.get("aligned") and self.reference_image_2d is not None:
            reason = self.tr("prepared before the reference frame was chosen")
        if reason is not None:
            return self._result(gen, status=self._cull_frame(path, reason),
                                point=(frame_no, [fwhm, ecc, sc, None, None, None], True, key))

        # Accumulate: weighted mean + variance, sigma rejection by exclusion.
        # Under the lock: GUI readers (export, preview, SNR) never see a
        # half-updated stack.
        with self._lock:
            if gen != self._gen:
                return None          # Reset or mode switch while this frame was prepared
            targets = self.norm_targets.get(key)
            used = prep["targets_used"]
            if targets is not None and prep.get("targets_own"):
                # Prepared before this stack had its levels, so normalized to
                # its own. The normalization is a per-channel gain after the
                # floor, so moving it to the stack's levels is one more gain.
                for c in range(frame.shape[2]):
                    if c < len(used) and c < len(targets) and float(used[c]) > 0.0:
                        frame[..., c] *= np.float32(float(targets[c]) / float(used[c]))
            acc = self.stacks.get(key)
            if acc is None:
                acc = LiveAccumulator(frame.shape, self.bootstrap_frames, self.clip_threshold)
                self.stacks[key] = acc
                self.norm_targets[key] = targets if targets is not None else used
            if self.reference_image_2d is None:
                self.reference_image_2d = np.ascontiguousarray(
                    frame[..., 0] if frame.shape[2] == 1 else frame.mean(axis=2))
            acc.k = float(self.clip_threshold)
            acc.add(frame, weight)
            if key not in self._star_sigma and prep.get("star_sigma") is not None:
                # this stack's first good frame chose the star threshold: hold it
                self._star_sigma[key] = float(prep["star_sigma"])
                self._note_once(f"starsig_{self._gen}_{key}",
                                f"Star detection{'' if key == '__main__' else ' (' + str(key) + ')'}: "
                                f"{self._star_sigma[key]:g}σ finds {sc} stars; held for this stack.")
            self._last_key = key
            self.frame_count += 1
            self._fwhm_hist.setdefault(key, []).append(float(fwhm))
            whist = self._weight_hist.setdefault(key, [])
            whist.append(float(weight))
            w_rel = float(weight) / float(np.median(whist)) if whist else 1.0
            self._add_exposure(key, header)
            self._remember_header(key, header)
        del frame, prep

        # Stack noise and the SNR readout (reads only: this thread is the
        # only writer of the stacks)
        lum = acc.mu[..., 0] if acc.shape[2] == 1 else acc.mu.mean(axis=2)
        noise = spatial_noise(lum)
        with self._lock:
            self._noise_hist.setdefault(key, []).append((acc.n_eff, noise))
        snr_text, snr = self._snr_text(key, acc, lum)
        del lum

        tag = f" ({self._key_label(key)})" if key != "__main__" else ""
        if acc.clipping:
            mode = self.tr("Mode: Weighted σ-Clipping Average, k={0:g}{1}").format(acc.k, tag)
        else:
            mode = self.tr("Mode: Weighted Average, bootstrap {0}/{1}{2}").format(
                acc.n_frames, acc.bootstrap, tag)
        msg = self.tr("✔ {0}{1}: weight {2:.2f}× median").format(name, tag, w_rel)
        if acc.clipping and acc.n_frames > acc.bootstrap:
            msg += self.tr(", rejected {0:.3f}% of pixels").format(100.0 * acc.last_reject_frac)
        elif acc.clipping and acc.bootstrap >= 3:
            msg += self.tr(", σ-clipping on: an outlier was taken back out of "
                           "{0:.3f}% of pixels").format(100.0 * acc.bootstrap_removed_frac)

        # Preview image (Auto Stretch or linear; display only)
        rev = self._display_rev
        disp = self._display_image()
        return self._result(gen, status=msg, mode=mode, snr_text=snr_text,
                            point=(frame_no, [fwhm, ecc, sc, w_rel, noise, snr], False, key),
                            display=disp, display_rev=rev, sky_box=self._sky_box.get(key))

    # ─────────────────────────────────────────────────────────────────────────
    # SNR goal / noise trend
    # ─────────────────────────────────────────────────────────────────────────
    def _region_for(self, shape):
        if self.snr_region is None:
            return None
        H, W = shape[:2]
        x0, y0, x1, y1 = self.snr_region
        x0, x1 = max(0, min(W, x0)), max(0, min(W, x1))
        y0, y1 = max(0, min(H, y0)), max(0, min(H, y1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return (x0, y0, x1, y1)

    def _exp_per_frame(self, key):
        e = self._exp_by_key.get(key)
        if not e or e[1] <= 0:
            return None
        return e[0] / e[1]

    def _update_snr_readout(self, key, acc, lum):
        """Set the SNR line under the preview; returns the region SNR or None."""
        text, snr = self._snr_text(key, acc, lum)
        self.snr_label.setText(text)
        return snr

    def _sky_for(self, key, lum):
        """Sky level for the SNR from this stack's background box. The box is
        found once per stack (Background Neutralization's finder) and then
        kept, so the SNR series isn't moved by a wandering sample."""
        box = self._sky_box.get(key)
        if box is None:
            box = find_sky_box(lum)
            if box is not None:
                self._sky_box[key] = box
                self._note_once("skybox", "SNR sky sample: the cyan box, found like "
                                          "Background Neutralization's Find Background.")
        sky = box_sky(lum, box) if box is not None else None
        return sky if sky is not None else sky_level(lum)

    def _snr_text(self, key, acc, lum):
        """The SNR line and the region SNR (or None). No widget access.

        SNR per pixel = (median stacked level in the region - darkest sky) /
        median noise of the mean, where the noise comes from the frame-to-frame
        variance (sigma / sqrt(N_eff)), so the target's own structure is never
        counted as noise. Sky-limited frames add SNR as sqrt(N), which gives the
        frames-to-goal estimate. The trend line checks that assumption: the
        stack's pixel-to-pixel noise should fall as N^-0.5."""
        tag = f"[{self._key_label(key)}] " if key != "__main__" else ""
        hist = list(self._noise_hist.get(key, []))
        slope = noise_trend_slope([h[0] for h in hist], [h[1] for h in hist])
        floor = slope is not None and acc.n_frames >= 16 and slope > -0.25
        if slope is None:
            trend = self.tr("noise trend: needs more frames")
        else:
            trend = self.tr("noise ∝ N^{0:+.2f} (ideal −0.50)").format(slope)
        if floor:
            trend += self.tr(" ⚠ noise has stopped falling as it should: a fixed pattern "
                             "dominates, more frames help little; dither between frames")

        snr = None
        rect = self._region_for(acc.shape)
        if self.snr_region is None:
            text = self.tr("{0}{1}  ·  Use 'SNR Region' and drag a box over faint "
                           "signal to estimate the frames needed for an SNR of {2:g}").format(
                tag, trend, self.snr_goal)
        elif rect is None:
            text = self.tr("{0}The SNR region is outside this stack  ·  {1}").format(tag, trend)
        elif acc.n_frames < 3:
            text = self.tr("{0}SNR starts after 3 frames  ·  {1}").format(tag, trend)
        else:
            sky = self._sky_for(key, lum)
            snr, sig, _nz = region_snr(acc, rect, sky)
            if snr is None:
                text = self.tr("{0}SNR: not enough data in the region yet  ·  {1}").format(tag, trend)
            elif sig <= 0:
                snr = None
                text = self.tr("{0}SNR: the region is not brighter than the sky  ·  {1}").format(tag, trend)
            else:
                goal = float(self.snr_goal)
                if snr >= goal:
                    est = self.tr("goal reached")
                else:
                    n_eff = acc.n_eff
                    per_frame = n_eff / max(acc.n_frames, 1)
                    need_eff = n_eff * (goal / snr) ** 2
                    more = int(np.ceil((need_eff - n_eff) / max(per_frame, 1e-9)))
                    exp = self._exp_per_frame(key)
                    est = self.tr("≈{0} more frames").format(more)
                    if exp:
                        est += f" (≈{_fmt_dur(more * exp)})"
                    if floor:
                        est += self.tr(", optimistic")
                text = self.tr("{0}SNR/px {1:.1f} → goal {2:g}: {3}  ·  {4}").format(
                    tag, snr, goal, est, trend)
        return text, snr

    def _refresh_snr_readout(self):
        key = self._last_key
        acc = self.stacks.get(key) if key is not None else None
        if acc is None:
            self.snr_label.setText(
                self.tr("SNR region set; the estimate starts with the first frames.")
                if self.snr_region is not None else "")
            return
        with self._lock:
            lum = acc.mu[..., 0] if acc.shape[2] == 1 else acc.mu.mean(axis=2)
            self._update_snr_readout(key, acc, lum)
        self._draw_sky_box(self._sky_box.get(key))

    def _on_snr_pick_toggled(self, on: bool):
        self._snr_pick = bool(on)
        self._rb_last = None
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag if on
                              else QGraphicsView.DragMode.ScrollHandDrag)
        if on:
            self.status_label.setText(self.tr(
                "Drag a box over the faint signal you want to reach the SNR goal "
                "(click SNR Region again to cancel)."))

    def _on_rubber_band(self, rect, from_scene, to_scene):
        if not self._snr_pick:
            return
        if rect.isNull() and from_scene.isNull() and to_scene.isNull():
            # drag released: Qt reports the end with null values
            last, self._rb_last = self._rb_last, None
            if last is not None:
                self._set_snr_region(*last)
            self.act_snr_region.setChecked(False)
            return
        self._rb_last = (QPointF(from_scene), QPointF(to_scene))

    def _set_snr_region(self, p0, p1):
        xa, xb = sorted((p0.x(), p1.x()))
        ya, yb = sorted((p0.y(), p1.y()))
        x0, y0 = int(np.floor(max(0.0, xa))), int(np.floor(max(0.0, ya)))
        x1, y1 = int(np.ceil(xb)), int(np.ceil(yb))
        if x1 - x0 < 3 or y1 - y0 < 3:
            self.status_label.setText(self.tr("SNR region too small; drag a larger box."))
            return
        self.snr_region = (x0, y0, x1, y1)
        if self._snr_rect_item is None:
            item = QGraphicsRectItem()
            pen = QPen(QColor(255, 200, 0))
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(10)
            self.scene.addItem(item)
            self._snr_rect_item = item
        self._snr_rect_item.setRect(x0, y0, x1 - x0, y1 - y0)
        self.status_label.setText(self.tr("SNR region: {0}×{1} px at ({2}, {3})").format(x1 - x0, y1 - y0, x0, y0))
        self._refresh_snr_readout()

    def _draw_sky_box(self, box):
        """Show the SNR's sky sample (cyan) while an SNR region is set."""
        if box is None or self.snr_region is None:
            if self._sky_rect_item is not None:
                try:
                    self.scene.removeItem(self._sky_rect_item)
                except Exception:
                    pass
                self._sky_rect_item = None
            return
        if self._sky_rect_item is None:
            item = QGraphicsRectItem()
            pen = QPen(QColor(0, 220, 255))
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setZValue(10)
            item.setToolTip(self.tr("SNR sky sample (found like Background Neutralization's Find Background)"))
            self.scene.addItem(item)
            self._sky_rect_item = item
        x0, y0, x1, y1 = box
        self._sky_rect_item.setRect(x0, y0, x1 - x0, y1 - y0)

    def _clear_snr_region(self):
        self.snr_region = None
        self._draw_sky_box(None)
        if self._snr_rect_item is not None:
            try:
                self.scene.removeItem(self._snr_rect_item)
            except Exception:
                pass
            self._snr_rect_item = None
        self._refresh_snr_readout()

    # ─────────────────────────────────────────────────────────────────────────
    # Display (statistical stretch of the linear stacks; never fed back)
    # ─────────────────────────────────────────────────────────────────────────
    def _display_image(self, stretch=None):
        """The preview image: the statistical stretch of the linear stacks
        (Auto Stretch) or the linear stacks as they are. Never fed back."""
        if stretch is None:
            stretch = self.auto_stretch

        def show(a2d):
            if stretch:
                return stretch_mono_image(a2d, target_median=0.3)
            return np.clip(a2d, 0.0, 1.0)

        with self._lock:     # the watch thread writes the stacks under this lock
            if self.star_trail_mode:
                return self.current_stack
            stacks = dict(self.stacks)
            if self.mono_color_mode and any(k != "__main__" for k in stacks):
                planes = {k: show(a.mu[..., 0]) for k, a in stacks.items()
                          if k != "__main__" and a.shape[2] == 1}
                return self._build_color_composite(planes)
            acc = stacks.get("__main__")
            if acc is None:
                return None
            if acc.shape[2] == 1:
                return show(acc.mu[..., 0])
            if stretch:
                return stretch_color_image(acc.mu, target_median=0.3, linked=False)
            return np.clip(acc.mu, 0.0, 1.0)

    def _show_display(self, disp):
        self._preview_cache = disp
        if disp is not None:
            self.update_preview(disp)

    def _star_trail_core(self, path: str, gen):
        """
        Load/calibrate a single frame (RAW or FITS/TIFF), debayer if needed,
        stretch, then build a max‐value “star trail” in self.current_stack.
        Runs on the watch thread; returns a result for _apply_result.
        """
        name = os.path.basename(path)
        img, header, _mono_key, why = self._load_calibrated(path)
        if img is None:
            return self._result(gen, status=f"⚠ {name}: {why}")
        if self._should_stop():
            return None

        # for star-trail we want a visible, stretched version:
        if img.ndim == 2:
            plane = stretch_mono_image(img, target_median=0.3)
            norm_color = np.stack([plane]*3, axis=2)
        else:
            norm_color = stretch_color_image(img, target_median=0.3, linked=False)
        if self._should_stop():
            return None

        # elementwise max over all frames so far
        with self._lock:
            if gen != self._gen:
                return None
            if self.current_stack is None:
                self.current_stack = norm_color.copy()
            else:
                self.current_stack = np.maximum(self.current_stack, norm_color)
            self.frame_count += 1
            self._add_exposure("__main__", header)
            disp = self.current_stack
            n = self.frame_count
        return self._result(gen, status=self.tr("★ Star-Trail frame {0}: {1}").format(n, name),
                            display=disp, display_rev=self._display_rev)

    def closeEvent(self, event):
        # request stop, then let the watch thread finish its current frame
        self.stop_live()
        self._wait_worker(30000)

        # also close the metrics window so it doesn't keep stuff alive
        try:
            if self.metrics_window is not None:
                self.metrics_window.close()
        except Exception:
            pass

        try:
            self._release_memory()
        except Exception:
            pass
        event.accept()

    def _release_memory(self):
        """Drop stack buffers / preview frame so RAM can be reclaimed on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        # Break QApplication → bound-method → self retention.
        try:
            app = QApplication.instance()
            if app is not None:
                try:
                    app.aboutToQuit.disconnect(self._on_app_quit)
                except (TypeError, RuntimeError):
                    pass
        except Exception:
            pass

        self.current_stack = None
        self.master_dark = None
        self.master_flat = None
        try:
            self.stacks.clear()
            self.norm_targets.clear()
            self._star_sigma.clear()
            self.master_flats.clear()
        except Exception:
            pass
        self._last_frame_bytes = None
        self._preview_cache = None
        self.reference_image_2d = None
        self._snr_rect_item = None
        self._sky_rect_item = None
        self._docman = None
        self._shutdown_pool()
        self._cleanup_xfer()

        try:
            self.pixmap_item.setPixmap(QPixmap())
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


    def _apply_bc(self, array: np.ndarray) -> np.ndarray:
        """Preview brightness / contrast (pivot 0.3), float32 in [0, 1]."""
        arr = np.clip(array, 0.0, 1.0).astype(np.float32)
        pivot = 0.3
        arr = ((arr - pivot) * self.contrast + pivot) + self.brightness
        return np.clip(arr, 0.0, 1.0)

    def update_preview(self, array: np.ndarray):
        # 1) normalize, apply contrast/brightness
        arr = self._apply_bc(array)

        # 2) convert to uint8 and KEEP a reference on self
        self._last_frame_bytes = (arr * 255).astype(np.uint8)
        h, w = self._last_frame_bytes.shape[:2]

        # 3) build QImage from the kept buffer
        if self._last_frame_bytes.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8
            bytespp = w
        else:
            fmt = QImage.Format.Format_RGB888
            bytespp = 3 * w
        qimg = QImage(self._last_frame_bytes.data, w, h, bytespp, fmt)

        # 4) update scene
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        if not self._did_initial_fit:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._did_initial_fit = True

