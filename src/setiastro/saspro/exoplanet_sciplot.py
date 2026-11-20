# src/setiastro/saspro/exoplanet_sciplot.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Publication-style light-curve figures for the exoplanet / variable-star tool.

The interactive pyqtgraph plots are for exploring.  These are the figures a
paper, an AAVSO or an ETD submission expects: white background, inward ticks
on all four sides, serif type, every point with its error bar, binned points
with propagated errors, the transit model over the data with a residual panel
underneath, and a broken time axis (one panel per night) for multi-night sets.

matplotlib's object API only -- never pyplot -- so nothing here touches global
figure state inside the Qt application.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

from setiastro.saspro.exoplanet_photometry import bin_curve

# Categorical series colours in a fixed order (never cycled; a colour-blind
# validated set).  A single light curve is drawn in ink, not a series colour.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948")
MAX_SERIES = len(SERIES_COLORS)
INK = "#1a1a1a"
INK_2 = "#52514e"
POINT = "#8a8a8a"
MODEL = "#d62f2f"
REJECT = "#e34948"
SHADE = "#eda100"
LOGC = 2.5 / math.log(10.0)

STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.0,
    "axes.labelsize": 11.0,
    "axes.titlesize": 11.0,
    "figure.titlesize": 11.5,
    "axes.linewidth": 0.9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.facecolor": "white",
    "axes.formatter.useoffset": False,
    "figure.facecolor": "white",
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "errorbar.capsize": 0.0,
    "savefig.facecolor": "white",
}


# ═══════════════════════════════ widget ═══════════════════════════════════

class _Toolbar(NavigationToolbar2QT):
    # Zoom / pan / home only.  Saving goes through the tool's own Save Plot
    # (300 dpi PNG, PDF, SVG), not the toolbar's screen-resolution save.
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in (None, "Home", "Back", "Forward", "Pan", "Zoom")]


class SciPlotWidget(QWidget):
    """A matplotlib figure with a zoom/pan toolbar, in the publication style."""

    def __init__(self, parent=None, figsize=(8.0, 5.0)):
        super().__init__(parent)
        with mpl.rc_context(STYLE):
            self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = _Toolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    def refresh(self):
        self.canvas.draw_idle()

    def save(self, path: str, dpi: int = 300):
        """PNG at `dpi`, or vector PDF / SVG (unbinned points rasterised)."""
        with mpl.rc_context(STYLE):
            self.figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")


# ═══════════════════════════════ helpers ══════════════════════════════════

def draw_message(fig: Figure, text: str):
    with mpl.rc_context(STYLE):
        fig.clear()
        fig.text(0.5, 0.5, text, ha="center", va="center", color=INK_2, fontsize=11)


def _to_mag(y, e=None):
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = -2.5 * np.log10(y)
        me = None if e is None else LOGC * np.abs(np.asarray(e, dtype=np.float64) / y)
    return m, me


def _span(y, e=None):
    """(lo, hi) of points with their error bars; one huge error bar can't stretch it."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if not np.any(ok):
        return None
    if e is None:
        return float(np.min(y[ok])), float(np.max(y[ok]))
    e = np.asarray(e, dtype=np.float64)
    ee = np.where(np.isfinite(e) & (e > 0), e, 0.0)[ok]
    if np.any(ee > 0):
        ee = np.minimum(ee, 3.0 * float(np.median(ee[ee > 0])))
    return float(np.min(y[ok] - ee)), float(np.max(y[ok] + ee))


def _set_ylim(ax, spans, invert=False, pad=0.08):
    spans = [s for s in spans if s is not None]
    if not spans:
        return
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    r = hi - lo
    if not (np.isfinite(r) and r > 0):
        r = max(abs(hi), 1.0) * 1e-3
    lo, hi = lo - pad * r, hi + pad * r
    ax.set_ylim((hi, lo) if invert else (lo, hi))


def _errorbar_points(ax, x, y, e, color, alpha, label=None):
    if x.size == 0:
        return
    ax.errorbar(x, y, yerr=e, fmt="o", ms=2.4, color=color, mec="none", ecolor=color,
                elinewidth=0.6, alpha=alpha, zorder=1, rasterized=True, label=label)


def _errorbar_bins(ax, tb, yb, eb, color, label=None, ms=4.6):
    if tb.size == 0:
        return
    ax.errorbar(tb, yb, yerr=eb, fmt="o", ms=ms, color=color, mec="white", mew=0.5,
                ecolor=color, elinewidth=1.0, capsize=2.0, capthick=1.0, zorder=3,
                label=label)


# ═════════════════════════════ light curves ═══════════════════════════════

def draw_light_curves(fig: Figure, series: Sequence[dict], *, panels=None,
                      units: str = "flux", bin_w: float = 0.0, bin_label: str = "",
                      xlabel: str = "", title: str = "", spans=(),
                      ylabel: Optional[str] = None):
    """
    series   dicts: label, x, y (relative flux), err (or None), group (night
             index per point; bins never straddle one), rej_x, rej_y
    panels   None for one axis, else [(x0, x1, label)]: one panel per night,
             a broken axis with a shared y scale
    units    'flux' (relative flux) or 'mag' (differential magnitude, y down)
    bin_w    bin width in x units, 0 for none
    spans    [(x0, x1, label)] shaded intervals (candidate dip)

    Several stars are stacked with vertical offsets (named in the legend)
    instead of overplotted.
    """
    with mpl.rc_context(STYLE):
        fig.clear()
        mag = units == "mag"
        ser = []
        for s in list(series)[:MAX_SERIES]:
            x = np.asarray(s["x"], dtype=np.float64)
            y = np.asarray(s["y"], dtype=np.float64)
            e = None if s.get("err") is None else np.asarray(s["err"], dtype=np.float64)
            rx = np.asarray(s.get("rej_x", ()), dtype=np.float64)
            ry = np.asarray(s.get("rej_y", ()), dtype=np.float64)
            if mag:
                y, e = _to_mag(y, e)
                ry, _ = _to_mag(ry)
            g = s.get("group")
            g = np.zeros(x.size, dtype=int) if g is None else np.asarray(g)
            ser.append({"label": str(s.get("label", "")), "x": x, "y": y, "e": e,
                        "g": g, "rx": rx, "ry": ry, "off": 0.0})
        n = len(ser)
        if n == 0:
            draw_message(fig, "Select a star in the list")
            return

        # Stack several stars: each one's band sits below the previous one's.
        if n > 1:
            bands = []
            for s in ser:
                v = s["y"][np.isfinite(s["y"])]
                bands.append((float(np.percentile(v, 1)), float(np.percentile(v, 99)))
                             if v.size >= 5 else (0.0, 0.0))
            width = max([b[1] - b[0] for b in bands] + [1e-3])
            gap = 0.3 * width
            for k in range(1, n):
                p, c = bands[k - 1], bands[k]
                if mag:     # inverted axis: larger magnitude is lower
                    off = (p[1] + ser[k - 1]["off"]) - c[0] + gap
                else:
                    off = (p[0] + ser[k - 1]["off"]) - c[1] - gap
                ser[k]["off"] = float(np.round(off, 3))

        multi_panel = bool(panels) and len(panels) > 1
        if multi_panel:
            dur = np.array([max(float(p[1]) - float(p[0]), 1e-6) for p in panels])
            ratios = np.maximum(dur, 0.25 * dur.max())
            axes = list(np.atleast_1d(fig.subplots(1, len(panels), sharey=True,
                                                   gridspec_kw={"width_ratios": ratios})))
            try:
                fig.get_layout_engine().set(wspace=0.0, w_pad=0.01)
            except Exception:
                pass
        else:
            axes = [fig.add_subplot(111)]
            panels = [(None, None, "")]

        y_spans = []
        span_labeled = False
        for ai, (ax, (x0, x1, plabel)) in enumerate(zip(axes, panels)):
            first_ax = ai == 0
            for k, s in enumerate(ser):
                col = INK if n == 1 else SERIES_COLORS[k]
                pcol = POINT if n == 1 else col
                inp = (np.ones(s["x"].size, bool) if x0 is None
                       else (s["x"] >= x0 - 1e-9) & (s["x"] <= x1 + 1e-9))
                fin = inp & np.isfinite(s["y"])
                x = s["x"][fin]
                y = s["y"][fin] + s["off"]
                e = s["e"][fin] if s["e"] is not None else None
                g = s["g"][fin]
                _errorbar_points(ax, x, y, e, pcol, 0.45 if n == 1 else 0.3,
                                 label=("data (±1σ)" if n == 1 and first_ax else None))
                y_spans.append(_span(y, e))
                if bin_w and bin_w > 0 and x.size:
                    first = True
                    for lab in np.unique(g):
                        gm = g == lab
                        tb, yb, eb = bin_curve(x[gm], y[gm], bin_w,
                                               e=(e[gm] if e is not None else None))
                        _errorbar_bins(ax, tb, yb, eb, col, ms=(4.6 if n == 1 else 3.8),
                                       label=(bin_label if n == 1 and first_ax and first
                                              and tb.size else None))
                        if tb.size:
                            first = False
                            y_spans.append(_span(yb, eb))
                if n == 1 and s["rx"].size:
                    rin = np.isfinite(s["ry"]) & (np.ones(s["rx"].size, bool) if x0 is None else
                                                  (s["rx"] >= x0 - 1e-9) & (s["rx"] <= x1 + 1e-9))
                    if np.any(rin):
                        ax.plot(s["rx"][rin], s["ry"][rin] + s["off"], ls="none", marker="x",
                                ms=3.5, mew=0.8, color=REJECT, alpha=0.75, zorder=2,
                                label=("rejected" if first_ax else None))
            for (s0, s1, slab) in spans:
                if x0 is not None and (s1 < x0 or s0 > x1):
                    continue
                ax.axvspan(s0, s1, color=SHADE, alpha=0.16, lw=0, zorder=0,
                           label=(slab if not span_labeled else None))
                span_labeled = True
            if x0 is not None:
                pad = 0.04 * max(x1 - x0, 1e-3)
                ax.set_xlim(x0 - pad, x1 + pad)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if len(panels) > 2 else 4))
                if plabel:
                    ax.set_title(str(plabel), fontsize=8.5, color=INK_2, pad=3)

        _set_ylim(axes[0], y_spans, invert=mag)

        if multi_panel:
            d = 0.6
            kw = dict(marker=[(-1, -d), (1, d)], markersize=7, linestyle="none",
                      color=INK, mec=INK, mew=0.9, clip_on=False, zorder=10)
            for i, ax in enumerate(axes):
                if i < len(axes) - 1:
                    ax.spines["right"].set_visible(False)
                    ax.tick_params(axis="y", which="both", right=False)
                    ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kw)
                if i > 0:
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                    ax.plot([0, 0], [0, 1], transform=ax.transAxes, **kw)
            fig.supxlabel(xlabel)
        else:
            axes[0].set_xlabel(xlabel)

        if ylabel is None:
            ylabel = r"$\Delta m$ (mag)" if mag else "Relative flux"
            if n > 1:
                ylabel += " + offset"
        axes[0].set_ylabel(ylabel)
        if title:
            fig.suptitle(title, wrap=True)   # wraps instead of running off a narrow canvas

        if n == 1:
            # entries can live on any panel (the dip shading is on its night's)
            h, lab = [], []
            for a in axes:
                for hh, ll in zip(*a.get_legend_handles_labels()):
                    if ll not in lab:
                        h.append(hh)
                        lab.append(ll)
            if h:
                axes[0].legend(h, lab, loc="best", handletextpad=0.4, borderaxespad=0.6)
        else:
            unit = "mag" if mag else ""
            handles = [Line2D([], [], ls="none", marker="o", ms=5, color=SERIES_COLORS[k])
                       for k in range(n)]
            labels = [s["label"] + (f"  ({s['off']:+.3f}{(' ' + unit) if unit else ''})"
                                    if s["off"] else "") for s in ser]
            fig.legend(handles, labels, loc="outside right upper", handletextpad=0.3)


# ═════════════════════════════ transit fit ════════════════════════════════

def draw_transit_fit(fig: Figure, *, x, y, err, used, model_x, model_y, resid,
                     resid_err, bin_w: float = 0.0, bin_label: str = "",
                     xlabel: str = "", title: str = "", info: Sequence[str] = (),
                     pred_x: Optional[float] = None):
    """
    The classic transit figure: detrended flux with error bars, binned points
    and the model on top; residuals (ppt) underneath on the same time axis.
    """
    with mpl.rc_context(STYLE):
        fig.clear()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        err = np.asarray(err, dtype=np.float64)
        used = np.asarray(used, bool)
        resid = np.asarray(resid, dtype=np.float64)
        resid_err = np.asarray(resid_err, dtype=np.float64)
        gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.15))
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1], sharex=ax)
        try:
            fig.get_layout_engine().set(hspace=0.0, h_pad=0.01)
        except Exception:
            pass
        ax.tick_params(labelbottom=False)
        u, c = used, ~used

        _errorbar_points(ax, x[u], y[u], err[u], POINT, 0.45, label="data (±1σ)")
        if np.any(c):
            ax.plot(x[c], y[c], ls="none", marker="x", ms=3.5, mew=0.8, color=REJECT,
                    alpha=0.75, zorder=2, label="clipped")
        spans = [_span(y[u], err[u])]
        if bin_w and bin_w > 0:
            tb, yb, eb = bin_curve(x[u], y[u], bin_w, e=err[u])
            _errorbar_bins(ax, tb, yb, eb, INK, label=bin_label or None)
        ax.plot(model_x, model_y, color=MODEL, lw=1.6, zorder=4, label="transit model")
        spans.append(_span(model_y))
        if pred_x is not None and np.isfinite(pred_x):
            ax.axvline(pred_x, color=INK_2, ls="--", lw=0.8, zorder=0,
                       label="predicted mid-transit")
        _set_ylim(ax, spans)
        ax.set_ylabel("Relative flux (detrended)")
        ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.6)
        if info:
            ax.text(0.02, 0.04, "\n".join(info), transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=8.5, color=INK, linespacing=1.35, zorder=6)

        _errorbar_points(axr, x[u], resid[u], resid_err[u], POINT, 0.45)
        if bin_w and bin_w > 0:
            tb, rb, eb = bin_curve(x[u], resid[u], bin_w, e=resid_err[u])
            _errorbar_bins(axr, tb, rb, eb, INK)
        axr.axhline(0.0, color=MODEL, lw=1.0, zorder=4)
        _set_ylim(axr, [_span(resid[u], resid_err[u])])
        lo, hi = axr.get_ylim()
        m = max(abs(lo), abs(hi))
        axr.set_ylim(-m, m)
        axr.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        axr.set_ylabel("Residual (ppt)")
        axr.set_xlabel(xlabel)
        fig.align_ylabels([ax, axr])
        if title:
            fig.suptitle(title, wrap=True)   # wraps instead of running off a narrow canvas
