#saspro/whitebalance.py
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QWidget, QGroupBox,
    QGridLayout, QSlider, QCheckBox, QPushButton, QMessageBox, QDoubleSpinBox
)

# imageops
from setiastro.saspro.imageops.starbasedwhitebalance import (
    apply_star_based_white_balance,
    DEFAULT_AUTO_SIGMA_LADDER,
    DEFAULT_AUTO_SIGMA_MIN_STARS,
)
from setiastro.saspro.imageops.stretch import stretch_color_image

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)

_orphaned_workers: list = []  # keeps threads alive after dialog close

def _mpl_no_tex_guard():
    # Prefer the centralized policy if available (frozen builds + single source of truth)
    try:
        from setiastro.saspro.gui_entry import _force_mpl_no_tex
        _force_mpl_no_tex()
        return
    except Exception:
        pass

    # Fallback: force-disable TeX directly
    try:
        import matplotlib
        matplotlib.rcParams["text.usetex"] = False
    except Exception:
        pass

def _np_trapezoid_compat(y, x):
    """
    NumPy compatibility helper:
    - prefer np.trapezoid when available
    - fall back to np.trapz on older NumPy builds
    """
    fn = getattr(np, "trapezoid", None)
    if callable(fn):
        return fn(y, x)
    return np.trapz(y, x)
# ----------------------------
# Core WB implementations
# ----------------------------
def build_star_color_ratios_figure(raw_pixels: np.ndarray, after_pixels: np.ndarray):
    """
    Build the SAS-style diagnostic plot and return a matplotlib Figure.

    Axes:
        x = R / B
        y = G / B

    Adds:
      - RGB ratio background
      - star scatter before/after WB
      - best-fit line
      - blackbody locus in the same ratio space
      - labeled temperature tick marks

    Notes:
      - The blackbody locus is plotted in *ratio space* (R/B, G/B), not CIE x,y.
      - This is intentionally matched to the background/plot semantics used here.
    """
    _mpl_no_tex_guard()

    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    # Ratios past this are both far outside the 0..2 view and a sign the
    # denominator has collapsed — no real stellar sample reaches it.
    _RATIO_MAX = 100.0

    def compute_ratios(pixels: np.ndarray):
        """
        Star colors as (R/B, G/B), with degenerate points returned as nan.

        The old form was `pixels[:, 0] / (pixels[:, 2] + 1e-8)`. That `+ eps`
        guard stops the ZeroDivisionError but produces something worse: a
        blue of 0.0 becomes a perfectly FINITE 8e7 instead of an inf, so
        every np.isfinite() check downstream waved it through to the artists.
        It is the same defect that crashed the locus tick labels (see
        _blackbody_ratio_point), latent here for any star whose blue sample
        sits at the noise floor — and those coordinates overflow a 32-bit
        FT_Pos on Windows once anything text-shaped is placed at them.

        Dividing straight through and filtering afterwards is both simpler
        and honest: a star with no blue has no meaningful R/B, so it is not
        a point on an R/B plot. Callers already drop non-finite pairs via
        _finite_pair, so nan is all that is needed here.
        """
        p = np.asarray(pixels, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = p[:, 0] / p[:, 2]
            gb = p[:, 1] / p[:, 2]
        bad = (
            ~np.isfinite(rb) | ~np.isfinite(gb) |
            (np.abs(rb) > _RATIO_MAX) | (np.abs(gb) > _RATIO_MAX)
        )
        return np.where(bad, np.nan, rb), np.where(bad, np.nan, gb)

    def _finite(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return x[m], y[m]

    def _blackbody_rgb_approx(temp_k: float) -> np.ndarray:
        """
        Smooth blackbody RGB approximation via Planck spectrum -> approximate CIE XYZ
        -> linear sRGB.

        Returns linear RGB in [0,1], normalized by max channel.
        This is much smoother than the common piecewise Kelvin->RGB approximation
        and avoids the visible kink near ~6500K.
        """
        T = float(max(1000.0, min(40000.0, temp_k)))

        # Wavelengths in nm
        wl = np.arange(380.0, 781.0, 5.0, dtype=np.float64)

        # --- Planck spectral radiance (relative only) ---
        # Constants
        h = 6.62607015e-34
        c = 2.99792458e8
        k = 1.380649e-23

        lam = wl * 1e-9  # nm -> m

        # Relative spectral power; absolute scale does not matter here
        c1 = 2.0 * h * c * c
        c2 = (h * c) / k
        spd = c1 / ((lam ** 5) * (np.exp(c2 / (lam * T)) - 1.0))

        # Normalize for numerical stability
        spd /= np.max(spd)

        # --- Approximate CIE 1931 2° color matching functions ---
        # Wyman, Sloan, Shirley-style analytic fits
        def x_bar(l):
            t1 = (l - 442.0) * (0.0624 if l < 442.0 else 0.0374)
            t2 = (l - 599.8) * (0.0264 if l < 599.8 else 0.0323)
            t3 = (l - 501.1) * (0.0490 if l < 501.1 else 0.0382)
            return (
                0.362 * np.exp(-0.5 * t1 * t1)
                + 1.056 * np.exp(-0.5 * t2 * t2)
                - 0.065 * np.exp(-0.5 * t3 * t3)
            )

        def y_bar(l):
            t1 = (l - 568.8) * (0.0213 if l < 568.8 else 0.0247)
            t2 = (l - 530.9) * (0.0613 if l < 530.9 else 0.0322)
            return (
                0.821 * np.exp(-0.5 * t1 * t1)
                + 0.286 * np.exp(-0.5 * t2 * t2)
            )

        def z_bar(l):
            t1 = (l - 437.0) * (0.0845 if l < 437.0 else 0.0278)
            t2 = (l - 459.0) * (0.0385 if l < 459.0 else 0.0725)
            return (
                1.217 * np.exp(-0.5 * t1 * t1)
                + 0.681 * np.exp(-0.5 * t2 * t2)
            )

        x = np.array([x_bar(v) for v in wl], dtype=np.float64)
        y = np.array([y_bar(v) for v in wl], dtype=np.float64)
        z = np.array([z_bar(v) for v in wl], dtype=np.float64)

        # Integrate SPD * CMFs
        X = _np_trapezoid_compat(spd * x, wl)
        Y = _np_trapezoid_compat(spd * y, wl)
        Z = _np_trapezoid_compat(spd * z, wl)

        XYZ = np.array([X, Y, Z], dtype=np.float64)
        if not np.all(np.isfinite(XYZ)) or np.max(XYZ) <= 0:
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Normalize XYZ
        XYZ /= np.max(XYZ)

        # --- XYZ -> linear sRGB ---
        M = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570],
        ], dtype=np.float64)

        rgb = M @ XYZ

        # Clamp negatives (out-of-gamut for sRGB)
        rgb = np.clip(rgb, 0.0, None)

        # Normalize by max channel for ratio-space plotting
        mx = float(np.max(rgb))
        if mx <= 1e-12 or not np.isfinite(mx):
            return np.array([1.0, 1.0, 1.0], dtype=np.float32)

        rgb /= mx
        return rgb.astype(np.float32)

    # Below this the blue channel carries no usable signal and the R/B, G/B
    # ratios are meaningless. 1500 K clips blue to EXACTLY 0.0 after the
    # out-of-gamut clamp in _blackbody_rgb_approx; 2000 K is ~7.6e-3, which
    # is small but still a real number, so this floor only rejects the
    # genuinely degenerate end of the curve.
    _BB_MIN_BLUE = 1e-4

    def _blackbody_ratio_point(temp_k: float):
        """
        Blackbody point in (R/B, G/B) ratio space, or (nan, nan) when the
        blue channel is degenerate.

        The old code did `rgb[0] / max(rgb[2], 1e-8)`, which turns a blue of
        exactly 0.0 (true at 1500 K) into a ratio of 1e8 rather than an inf
        or a nan — so every finite-check downstream passed it through. A text
        label placed at data coordinate 1e8 lands around 2.7e10 display
        pixels, and matplotlib's Agg text path hands FreeType
        `round(0x40 * px)` ≈ 1.7e12 as the transform delta.

        FT_Pos is a C long: 64-bit on Linux, but 32-bit on Windows (LLP64).
        Since matplotlib 3.10 ft2font is a pybind11 module with strict
        integer conversion, so on Windows that value is rejected with
        "TypeError: _set_transform(): incompatible function arguments"
        — raised from inside the Qt paint event, i.e. AFTER the figure is
        handed to the result dialog, where _on_apply's try/except can no
        longer see it. Hence a hard traceback on a WB that actually succeeded.

        Returning nan lets the callers filter these points out instead.
        """
        rgb = _blackbody_rgb_approx(temp_k).astype(np.float32)
        b = float(rgb[2])
        if not np.isfinite(b) or b <= _BB_MIN_BLUE:
            return float("nan"), float("nan")
        rb = float(rgb[0] / b)
        gb = float(rgb[1] / b)
        if not (np.isfinite(rb) and np.isfinite(gb)):
            return float("nan"), float("nan")
        return rb, gb

    def _add_blackbody_locus(ax):
        """
        Plot blackbody locus and labeled temperature tick marks in ratio space.
        """
        # Smooth curve temperatures
        temps_curve = np.concatenate([
            np.linspace(1500, 4000, 120),
            np.linspace(4100, 12000, 120),
        ])

        pts = np.array([_blackbody_ratio_point(t) for t in temps_curve], dtype=np.float32)
        x = pts[:, 0]
        y = pts[:, 1]

        # Keep finite points, and drop the cool end of the curve where the
        # ratios run away to huge values. The bound is deliberately generous
        # (the view is 0..2) so the drawn locus still crosses the whole
        # visible box and the local-tangent estimate below still has
        # neighbours at the edges — it only removes coordinates far enough
        # out to be both invisible and numerically hazardous.
        _CURVE_LIMIT = 10.0
        m = (np.isfinite(x) & np.isfinite(y) &
             (x > -1.0) & (x < _CURVE_LIMIT) &
             (y > -1.0) & (y < _CURVE_LIMIT))
        x = x[m]
        y = y[m]

        if len(x) < 2:
            return

        ax.plot(x, y, color="black", linewidth=2.0, label="Blackbody locus")

        # Labeled temperature ticks
        tick_temps = [1500, 2000, 2500, 3000, 4000, 6000, 10000]

        # Only label ticks that actually land inside the plotted box. The
        # cool end of the locus sits far outside it — 1500 K is at
        # (1e8, 1.5e7), 2000 K at (131, 35), 2500 K at (14.8, 5.6) — so those
        # labels were being created as artists and then drawn off-screen
        # where nobody could see them anyway. Skipping them is both the fix
        # for the FreeType overflow above and the behaviour the plot already
        # appeared to have.
        tick_pts = []
        for t in tick_temps:
            px, py = _blackbody_ratio_point(t)
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            if not (rmin <= px <= rmax and gmin <= py <= gmax):
                continue
            tick_pts.append((t, px, py))

        if len(tick_pts) < 1:
            return

        # For local tangent estimation, use neighboring curve samples
        curve_pts = np.column_stack([x, y])

        for t, px, py in tick_pts:
            # Find nearest point on curve
            d2 = (curve_pts[:, 0] - px) ** 2 + (curve_pts[:, 1] - py) ** 2
            i = int(np.argmin(d2))

            i0 = max(0, i - 2)
            i1 = min(len(curve_pts) - 1, i + 2)

            if i1 == i0:
                continue

            dx = curve_pts[i1, 0] - curve_pts[i0, 0]
            dy = curve_pts[i1, 1] - curve_pts[i0, 1]

            norm = float(np.hypot(dx, dy))
            if norm < 1e-8:
                continue

            # Unit normal to the curve
            nx = -dy / norm
            ny =  dx / norm

            # Tick size in ratio-space units
            tick_len = 0.035
            x0 = px - nx * tick_len
            y0 = py - ny * tick_len
            x1 = px + nx * tick_len
            y1 = py + ny * tick_len

            ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.5)

            # Label offset a little farther along the normal
            lx = px + nx * 0.065
            ly = py + ny * 0.065
            ax.text(lx, ly, f"{t}", fontsize=9, color="black", ha="center", va="center")

    # ------------------------------------------------------------
    # Compute ratios
    # ------------------------------------------------------------
    rb_before, gb_before = compute_ratios(raw_pixels)
    rb_after,  gb_after  = compute_ratios(after_pixels)

    # Saturation mask, computed from the RAW (N,3) pixels rather than from the
    # ratios — a star clipped at 1.0 in R with B=0.5 still yields a perfectly
    # finite R/B, so a ratio-space filter cannot see it. Thresholds mirror
    # starbasedwhitebalance._photometric_keep_mask so the dashed fit on this
    # plot agrees with the slope the WB engine actually measured. Without
    # this, the clipped stars cluster near (1,1) and drag the Before fit up,
    # which is exactly the bias the engine now rejects.
    _CLIP_HI, _CLIP_LO = 0.95, 0.005

    def _unsat(pix):
        a = np.asarray(pix, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] < 3:
            return np.ones(len(a), dtype=bool)
        return (np.all(np.isfinite(a[:, :3]), axis=1) &
                np.all(a[:, :3] < _CLIP_HI, axis=1) &
                np.all(a[:, :3] > _CLIP_LO, axis=1))

    def _finite_pair(x, y):
        return np.isfinite(x) & np.isfinite(y)

    keep_before = _unsat(raw_pixels)   & _finite_pair(rb_before, gb_before)
    keep_after  = _unsat(after_pixels) & _finite_pair(rb_after,  gb_after)

    # Drop only non-finite points (they cannot be drawn); clipped stars are
    # KEPT so they still appear on the scatter, just flagged and excluded
    # from the fit.
    fin_before = _finite_pair(rb_before, gb_before)
    fin_after  = _finite_pair(rb_after,  gb_after)
    rb_before, gb_before, keep_before = (rb_before[fin_before],
                                         gb_before[fin_before],
                                         keep_before[fin_before])
    rb_after,  gb_after,  keep_after  = (rb_after[fin_after],
                                         gb_after[fin_after],
                                         keep_after[fin_after])

    # Limit plotted stars
    max_plot_stars = 2000
    total_available = min(len(rb_before), len(rb_after))
    n = min(total_available, max_plot_stars)

    if n > 0:
        rb_before = rb_before[:n]
        gb_before = gb_before[:n]
        rb_after  = rb_after[:n]
        gb_after  = gb_after[:n]
        keep_before = keep_before[:n]
        keep_after  = keep_after[:n]

    # ------------------------------------------------------------
    # Plot bounds / ratio background
    # ------------------------------------------------------------
    rmin, rmax = 0.0, 2.0
    gmin, gmax = 0.0, 2.0
    res = 250

    rb_vals = np.linspace(rmin, rmax, res)
    gb_vals = np.linspace(gmin, gmax, res)
    rb_grid, gb_grid = np.meshgrid(rb_vals, gb_vals)

    # Blue normalized to 1.0 in this background
    rgb_image = np.stack([rb_grid, gb_grid, np.ones_like(rb_grid)], axis=-1)
    rgb_image /= np.maximum(rgb_image.max(axis=2, keepdims=True), 1e-8)

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig = Figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

    def plot_panel(ax, rb_data, gb_data, title, keep=None):
        ax.imshow(
            rgb_image,
            extent=(rmin, rmax, gmin, gmax),
            origin="lower",
            aspect="auto",
            alpha=0.9,
        )

        if keep is None:
            keep = np.ones(rb_data.shape, dtype=bool)
        drop = ~keep

        # Stars used for the fit
        ax.scatter(
            rb_data[keep],
            gb_data[keep],
            alpha=0.45,
            edgecolors="k",
            linewidths=0.25,
            s=12,
            label=f"Stars ({int(keep.sum())})",
        )

        # Clipped stars — shown so the population is still visible, but
        # excluded from the fit. Hollow gray so they read as "not counted".
        if int(drop.sum()) > 0:
            ax.scatter(
                rb_data[drop],
                gb_data[drop],
                facecolors="none",
                edgecolors="0.45",
                linewidths=0.5,
                s=14,
                alpha=0.55,
                label=f"Clipped, excluded ({int(drop.sum())})",
            )

        # Fit on unclipped stars only, so this line matches the slope the WB
        # engine reports in the console.
        fit_rb, fit_gb = rb_data[keep], gb_data[keep]
        if fit_rb.size < 2:
            fit_rb, fit_gb = rb_data, gb_data
        if fit_rb.size >= 2:
            m, b = np.polyfit(fit_rb, fit_gb, 1)
            xs = np.linspace(rmin, rmax, 200)
            ax.plot(xs, m * xs + b, "r--", linewidth=1.5, label=f"Best Fit\ny = {m:.2f}x + {b:.2f}")

        # Neutral marker only
        ax.plot([1.0], [1.0], marker="+", markersize=10, markeredgewidth=2.0,
                color="blue", label="Neutral (1,1)")

        # Blackbody locus
        _add_blackbody_locus(ax)

        ax.set_xlim(rmin, rmax)
        ax.set_ylim(gmin, gmax)
        ax.set_title(f"{title} White Balance")
        ax.set_xlabel("Red / Blue Ratio")
        ax.set_ylabel("Green / Blue Ratio")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
        ax.grid(True, alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plot_panel(ax1, rb_before, gb_before, "Before", keep=keep_before)
    plot_panel(ax2, rb_after,  gb_after,  "After",  keep=keep_after)

    if total_available > max_plot_stars:
        fig.suptitle(
            f"Star Color Ratios with RGB Mapping (showing first {max_plot_stars} of {total_available} stars)",
            fontsize=14
        )
    else:
        fig.suptitle("Star Color Ratios with RGB Mapping", fontsize=14)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.84,
        wspace=0.18,
    )
    return fig

def apply_manual_white_balance(img: np.ndarray, r_gain: float, g_gain: float, b_gain: float) -> np.ndarray:
    """Simple per-channel gain, clipped to [0,1]."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Manual WB requires RGB image.")
    out = _to_float01(img).copy()
    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32).reshape((1, 1, 3))
    out = np.clip(out * gains, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def apply_auto_white_balance(img: np.ndarray) -> np.ndarray:
    """
    Gray-world auto WB: scale each channel so its mean equals the overall mean.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Auto WB requires RGB image.")
    rgb = _to_float01(img)
    means = np.mean(rgb, axis=(0, 1))
    overall = float(np.mean(means))
    means = np.where(means <= 1e-8, 1e-8, means)
    scale = (overall / means).reshape((1, 1, 3))
    out = np.clip(rgb * scale, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


# --------------------------------
# Headless entry point for DnD
# --------------------------------
def apply_white_balance_to_doc(doc, preset: Optional[Dict] = None):
    """
    Preset schema:
      {
        "mode": "star" | "manual" | "auto",   # default "star"
        # star mode:
        "threshold": float (default 50),      # ignored when auto_sigma=True
        "auto_sigma": bool (default True),    # waterfall-based threshold pick
        "auto_sigma_min_stars": int (default 1200),
        "reuse_cached_sources": bool (default True),
        "use_color_matrix": bool (default False),
        "matrix_strength": float 0..1 (default 0.70),  # only if use_color_matrix
                                              # 0 = no rotation, 1 = full send
        # manual mode:
        "r_gain": float (default 1.0), "g_gain": float (default 1.0), "b_gain": float (default 1.0)
      }
    """
    import numpy as np

    p = dict(preset or {})
    mode = (p.get("mode") or "star").lower()

    base = np.asarray(doc.image).astype(np.float32, copy=False)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("White Balance requires an RGB image.")

    base_n = _to_float01(base)

    try:
        if mode == "manual":
            r = float(p.get("r_gain", 1.0))
            g = float(p.get("g_gain", 1.0))
            b = float(p.get("b_gain", 1.0))
            out = apply_manual_white_balance(base_n, r, g, b)

        elif mode == "auto":
            out = apply_auto_white_balance(base_n)

        else:  # "star"
            thr = float(p.get("threshold", 50.0))
            use_cm = bool(p.get("use_color_matrix", False))
            # Default 0.7 matches the dialog's conservative default, so an
            # old frozen shortcut with no matrix_strength key behaves exactly
            # as it did before this setting existed.
            cm_strength = float(p.get("matrix_strength", 0.70))
            # Auto-σ waterfall is on by default — removes the manual guess
            # and the "find too many stars" blowups at low σ.
            auto_sig = bool(p.get("auto_sigma", True))
            auto_min = int(p.get("auto_sigma_min_stars", DEFAULT_AUTO_SIGMA_MIN_STARS))
            out, _count, _overlay = apply_star_based_white_balance(
                base_n,
                threshold=thr,
                autostretch=False,
                reuse_cached_sources=False,
                return_star_colors=False,
                use_color_matrix=use_cm,
                matrix_strength=cm_strength,
                auto_sigma=auto_sig,
                auto_sigma_min_stars=auto_min,
            )
    except Exception as e:
        # Fallback: if SEP missing or star detection fails, try Auto WB
        if mode == "star":
            try:
                out = apply_auto_white_balance(base_n)
            except Exception:
                raise e
        else:
            raise

    # Destination-mask blend (if any)
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        if out.ndim == 3:
            m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
        else:
            m3 = m.astype(np.float32, copy=False)
        base_for_blend = _to_float01(np.asarray(doc.image).astype(np.float32, copy=False))
        out = base_for_blend * (1.0 - m3) + out * m3

    doc.apply_edit(
        out.astype(np.float32, copy=False),
        metadata={"step_name": "White Balance", "preset": p},
        step_name="White Balance",
    )

def apply_pivot_gain(img: np.ndarray, med: np.ndarray, gains: np.ndarray) -> np.ndarray:
    # img: HxWx3 float32 in [0,1]
    med3 = med.reshape(1, 1, 3).astype(np.float32)
    g3   = gains.reshape(1, 1, 3).astype(np.float32)

    # pivot around median; do not scale negative deltas
    d = img - med3
    d = np.maximum(d, 0.0)
    out = d * g3 + med3
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def apply_soft_protect(img: np.ndarray, out_pivot: np.ndarray, k: float = 0.02) -> np.ndarray:
    # luminance-based fade-in above median luminance
    L = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    Lm = float(np.median(L))
    w = smoothstep(Lm, Lm + k, L).astype(np.float32)
    w3 = w[..., None]
    out = img * (1.0 - w3) + out_pivot * w3
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

class WhiteBalanceResultDialog(QDialog):
    def __init__(self, parent, figure, star_count: int):
        super().__init__(parent)
        self.setWindowTitle(self.tr("White Balance Result"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setModal(False)
        self.resize(1100, 700)

        layout = QVBoxLayout(self)

        info = QLabel(self.tr("Star-Based WB applied successfully.\nDetected {0} stars.").format(int(star_count)))
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

        self.canvas = FigureCanvas(figure)
        self.toolbar = NavToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton(self.tr("Close"))
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

# -------------------------
# Interactive dialog (UI)
# -------------------------
class WhiteBalanceDialog(QDialog):
    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self._main = parent
        self.doc = doc
        self._active_doc_conn = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._active_doc_conn = True
            except Exception:
                self._active_doc_conn = False
        if icon:
            self.setWindowIcon(icon)
        self.setWindowTitle(self.tr("White Balance"))
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
        self.resize(900, 600)

        self._build_ui()
        self._wire_events()

        # default to Star-Based, like SASv2
        self.type_combo.setCurrentText("Star-Based")
        self._update_mode_widgets()
        # kick off a first detection preview
        QTimer.singleShot(200, self._update_star_preview)



    # ---- UI construction ------------------------------------------------
    def _build_ui(self):
        self.main_layout = QVBoxLayout(self)

        # Type selector
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("White Balance Type:")))
        self.type_combo = QComboBox()
        self.type_combo.addItems([self.tr("Star-Based"), self.tr("Manual"), self.tr("Auto")])
        row.addWidget(self.type_combo); row.addStretch()
        self.main_layout.addLayout(row)

        # Manual group
        self.manual_group = QGroupBox(self.tr("Manual Gains"))
        g = QGridLayout(self.manual_group)
        self.r_spin = QDoubleSpinBox(); self._cfg_gain(self.r_spin, 1.0)
        self.g_spin = QDoubleSpinBox(); self._cfg_gain(self.g_spin, 1.0)
        self.b_spin = QDoubleSpinBox(); self._cfg_gain(self.b_spin, 1.0)
        g.addWidget(QLabel(self.tr("Red gain:")),   0, 0); g.addWidget(self.r_spin, 0, 1)
        g.addWidget(QLabel(self.tr("Green gain:")), 1, 0); g.addWidget(self.g_spin, 1, 1)
        g.addWidget(QLabel(self.tr("Blue gain:")),  2, 0); g.addWidget(self.b_spin, 2, 1)
        self.main_layout.addWidget(self.manual_group)

        # Star-based controls + preview
        self.star_group = QGroupBox(self.tr("Star-Based Settings"))
        sg = QVBoxLayout(self.star_group)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel(self.tr("Detection threshold (σ):")))
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setMinimum(1); self.thr_slider.setMaximum(200)
        self.thr_slider.setValue(50); self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thr_label = QLabel("50")
        thr_row.addWidget(self.thr_slider); thr_row.addWidget(self.thr_label)
        sg.addLayout(thr_row)

        # ── Auto σ waterfall ──────────────────────────────────────────
        # Removes manual guess-and-check. The worker runs σ = 200, 100, 50, 25,
        # 12, 8, 5, 3 in order and stops at the first value that returns
        # ≥ 1200 detections. The slider is driven to the chosen σ so the
        # user can see what Auto picked (and uncheck Auto to override).
        auto_row = QHBoxLayout()
        self.auto_sigma_chk = QCheckBox(self.tr("Auto σ (waterfall)"))
        self.auto_sigma_chk.setChecked(True)
        self.auto_sigma_chk.setToolTip(self.tr(
            "Auto σ waterfall:\n"
            "  Tries σ = 200, 100, 50, 25, 12, 8, 5, 3 in order and stops at the\n"
            "  first value that yields ≥ 1200 detections. Removes the need to\n"
            "  guess a threshold for different fields.\n\n"
            "  Uncheck to drive the slider manually."
        ))
        auto_row.addWidget(self.auto_sigma_chk)
        auto_row.addStretch(1)
        sg.addLayout(auto_row)

        # Slider disabled when Auto is on — it becomes a read-out of the
        # resolved σ instead of an input.
        self.thr_slider.setEnabled(not self.auto_sigma_chk.isChecked())
        self.chk_color_matrix = QCheckBox(self.tr(
            "Advanced Color Matrix WB  "
            "(cross-channel matrix aligned to blackbody locus)"
        ))
        self.chk_color_matrix.setChecked(False)
        self.chk_color_matrix.setStyleSheet("color: #f0a000;")
        self.chk_color_matrix.setToolTip(self.tr(
            "Applies a 3×3 color matrix solved from the detected stellar population.\n"
            "Unlike standard per-channel scaling, the matrix mixes channels to rotate\n"
            "the stellar color scatter onto the blackbody locus — producing physically\n"
            "accurate star and galaxy colors in broadband RGB data.\n\n"
            "NOT recommended for narrowband or duoband data (Hα, OIII, SII etc.) —\n"
            "emission line colors will be shifted away from their physically accurate values.\n\n"
            "The amber color is a reminder that this changes color relationships differently\n"
            "than standard diagonal white balance scaling."
        ))
        sg.addWidget(self.chk_color_matrix)

        # ── Matrix strength (indented sub-control of the matrix checkbox) ──
        # How far each star is blended toward its blackbody-locus projection
        # before the 3x3 is solved. 70% is the conservative default: the
        # per-star targets are built from each star's own measured color, so
        # a pathological population steers the targets as well as the solve.
        # Holding back means such a population bends the result instead of
        # dictating it. Users with clean broadband data can push it.
        self.matrix_strength = 0.70
        cm_row = QHBoxLayout()
        cm_row.setContentsMargins(19, 0, 0, 0)
        cm_row.setSpacing(6)
        self.cm_strength_caption = QLabel(self.tr("Rotation strength:"))
        self.cm_strength_caption.setStyleSheet("font-size:11px;")
        cm_row.addWidget(self.cm_strength_caption)

        self.cm_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.cm_strength_slider.setMinimum(0)
        self.cm_strength_slider.setMaximum(100)
        self.cm_strength_slider.setValue(70)
        self.cm_strength_slider.setTickInterval(10)
        self.cm_strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cm_strength_slider.setFixedWidth(200)
        self.cm_strength_slider.setToolTip(self.tr(
            "How hard the color matrix rotates the stellar scatter onto the\n"
            "blackbody locus.\n\n"
            "  0%    no rotation (diagonal white balance only)\n"
            "  70%   default — conservative, resists pathological stars\n"
            "  100%  full send: every star aimed straight at the locus\n\n"
            "Each star's target is built from its OWN measured color, so a\n"
            "population full of nebulosity, emission-line objects or junk\n"
            "left over by a loose clip threshold steers the targets as well\n"
            "as the solve. The default holds back so such a population bends\n"
            "the result rather than dictating it.\n\n"
            "Push it up on clean broadband fields with plenty of unclipped\n"
            "stars. Keep it at or below the default on narrowband, duoband\n"
            "or sparse data.\n\n"
            "This is what is ASKED for, not what lands — the matrix realizes\n"
            "roughly three quarters of the request even at 100%."
        ))
        cm_row.addWidget(self.cm_strength_slider)

        self.cm_strength_label = QLabel("70%  default")
        self.cm_strength_label.setFixedWidth(88)
        self.cm_strength_label.setStyleSheet("font-size:11px;")
        cm_row.addWidget(self.cm_strength_label)
        cm_row.addStretch(1)
        sg.addLayout(cm_row)

        # Strength is meaningless unless the matrix is on.
        def _sync_matrix_strength(checked: bool):
            self.cm_strength_slider.setEnabled(bool(checked))
            self.cm_strength_label.setEnabled(bool(checked))
            self.cm_strength_caption.setEnabled(bool(checked))
        self.chk_color_matrix.toggled.connect(_sync_matrix_strength)
        _sync_matrix_strength(self.chk_color_matrix.isChecked())

        self.chk_reuse = None

        self.chk_autostretch_overlay = QCheckBox(self.tr("Autostretch overlay preview"))
        self.chk_autostretch_overlay.setChecked(True)
        sg.addWidget(self.chk_autostretch_overlay)

        self.star_count = QLabel(self.tr("Detecting stars..."))
        sg.addWidget(self.star_count)

        # Show Preview button — the preview QLabel is hidden by default
        # because on big RGB frames the overlay autostretch dominates open
        # time and the preview is too small for the ellipses to be useful
        # anyway. Disabled until the first detection completes so the
        # button never kicks off an overlay against an empty source list.
        show_row = QHBoxLayout()
        show_row.addStretch(1)
        self.btn_show_preview = QPushButton(self.tr("Show Preview"))
        self.btn_show_preview.setEnabled(False)
        self.btn_show_preview.setToolTip(self.tr(
            "Reveal the star-detection overlay. Hidden by default on big\n"
            "frames because building it needs an autostretch of the full\n"
            "image. Enables once the SEP waterfall has settled."
        ))
        show_row.addWidget(self.btn_show_preview)
        show_row.addStretch(1)
        sg.addLayout(show_row)

        self.preview = QLabel()
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("border: 1px solid #333;")
        self.preview.setVisible(False)
        sg.addWidget(self.preview)
        self.main_layout.addWidget(self.star_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        self.main_layout.addLayout(btn_row)

        # --- preset drag handle (grip) ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            from setiastro.saspro.help_support import make_help_button
            try:
                from setiastro.saspro.resources import whitebalance_path
                _grip_icon = QIcon(whitebalance_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "white_balance", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a White Balance shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            self.btn_help = make_help_button("white_balance", self)
            drag_row.addWidget(self.btn_help)
            self.main_layout.addLayout(drag_row)
        except Exception:
            pass

        self.setLayout(self.main_layout)

        # Debounce: wait for slider to settle before firing detection
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._update_star_preview)

        # Worker state
        self._detection_worker = None
        self._pending_threshold = None
        self._pending_autostretch = None
        self._closing = False

        # Lazy-preview state. The worker runs in cheap-mode (detection only,
        # no overlay) until the user clicks Show Preview; after that every
        # subsequent re-detection (e.g. slider move) builds the overlay too.
        self._preview_shown = False

    def _cfg_gain(self, box: QDoubleSpinBox, val: float):
        box.setRange(0.5, 1.5); box.setDecimals(3); box.setSingleStep(0.01); box.setValue(val)

    # ---- events ---------------------------------------------------------
    def _wire_events(self):
        self.type_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._on_apply)
        self.thr_slider.valueChanged.connect(self._on_threshold_changed)
        self.chk_autostretch_overlay.toggled.connect(lambda _=None: self._debounce.start())
        self.auto_sigma_chk.toggled.connect(self._on_auto_sigma_toggled)
        self.btn_show_preview.clicked.connect(self._on_show_preview_clicked)
        self.cm_strength_slider.valueChanged.connect(self._on_matrix_strength_changed)
        self.finished.connect(lambda *_: self._stop_detection_worker())

        # Restore last matrix strength (default 0.70 — the conservative value)
        try:
            s = QSettings()
            v = int(s.value("WhiteBalance/MatrixStrength", 70, type=int))
            v = max(0, min(100, v))
            self.cm_strength_slider.blockSignals(True)
            self.cm_strength_slider.setValue(v)
            self.cm_strength_slider.blockSignals(False)
            self._on_matrix_strength_changed(v)
        except Exception:
            pass

        # Restore last Auto σ preference (default: ON — removes guess-and-check)
        try:
            s = QSettings()
            auto = s.value("WhiteBalance/AutoSigma", True, type=bool)
            self.auto_sigma_chk.setChecked(bool(auto))
            self.thr_slider.setEnabled(not bool(auto))
        except Exception:
            pass


    def _on_threshold_changed(self, v: int):
        self.thr_label.setText(str(v))
        self._debounce.start()  # restart debounce — slider still moving

    def _on_matrix_strength_changed(self, v: int):
        """Matrix rotation strength. Does NOT re-trigger detection — it only
        affects the WB solve at Apply time, not star detection, so there is
        nothing to re-detect and no debounce needed."""
        self.matrix_strength = float(v) / 100.0
        if v == 0:
            word = self.tr("off")
        elif v < 45:
            word = self.tr("gentle")
        elif v <= 75:
            word = self.tr("default")
        elif v < 100:
            word = self.tr("strong")
        else:
            word = self.tr("full")
        self.cm_strength_label.setText(f"{v}%  {word}")
        try:
            QSettings().setValue("WhiteBalance/MatrixStrength", int(v))
        except Exception:
            pass

    def _on_auto_sigma_toggled(self, on: bool):
        try:
            QSettings().setValue("WhiteBalance/AutoSigma", bool(on))
        except Exception:
            pass
        try:
            # Slider is read-only (shows the auto-picked value) in auto mode,
            # and user-driven in manual mode.
            self.thr_slider.setEnabled(not bool(on))
        except Exception:
            pass
        # Kick a fresh preview so the mode change is reflected right away.
        self._debounce.start()

    def _update_mode_widgets(self):
        t = self.type_combo.currentText()
        self.manual_group.setVisible(t == "Manual")
        self.star_group.setVisible(t == "Star-Based")

    # ---- active document change ------------------------------------
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        # Qt fires currentDocumentChanged for focus churn as well as real
        # doc swaps — if the dialog grabs activation on show, we get a
        # None→doc round-trip for the SAME doc, which was starting a
        # second detection worker on top of the one QTimer.singleShot had
        # already kicked off. .cancel() only flags the result emission —
        # the SEP waterfall inside the old worker still runs to completion.
        # So no-op on same-doc changes.
        if doc is self.doc:
            return
        self.doc = doc
        self._update_star_preview()

    # ---- preview --------------------------------------------------------

    def _update_star_preview(self):
        if getattr(self, "_closing", False):
            return
        if self.type_combo.currentText() != "Star-Based":
            return

        auto_sigma = bool(self.auto_sigma_chk.isChecked()) if hasattr(self, "auto_sigma_chk") else False
        thr = float(self.thr_slider.value())
        auto_stretch = bool(self.chk_autostretch_overlay.isChecked())

        # Full set of params the worker will act on — used to short-circuit
        # redundant restarts. doc identity matters because cancel() only
        # flags result emission; SEP keeps running in the cancelled worker.
        want = (
            id(self.doc),
            thr,
            auto_sigma,
            auto_stretch,
            bool(self._preview_shown),
        )

        # If a worker is already running, cancel it and remember what we want next.
        # The _on_worker_done slot will re-fire with the pending values.
        if self._detection_worker is not None and self._detection_worker.isRunning():
            # Short-circuit: the running worker is doing exactly what we'd
            # ask of a new one. Starting a replacement would run a second
            # SEP waterfall in parallel (the first can't be stopped mid-call)
            # for an identical result — this is where the duplicate
            # [Auto-σ] prints were coming from.
            if getattr(self, "_running_params", None) == want:
                return
            self._pending_threshold = thr
            self._pending_autostretch = auto_stretch
            self._detection_worker.cancel()
            # Don't wait — let it finish naturally and _on_worker_done will restart
            return

        self._pending_threshold = None
        self._pending_autostretch = None
        self._running_params = want
        if auto_sigma:
            self.star_count.setText(self.tr("Auto σ: detecting…"))
        else:
            self.star_count.setText(self.tr("Detecting…"))

        from setiastro.saspro.imageops.starbasedwhitebalance import StarDetectionWorker
        img = _to_float01(np.asarray(self.doc.image))
        # In Auto mode the worker ignores `thr` and runs the σ ladder itself,
        # emitting the resolved σ back via finished(overlay_or_None, count, used_σ).
        # with_overlay controls whether the slow stretch + ellipse pass runs;
        # it stays off until the user has opted into the preview.
        worker = StarDetectionWorker(
            img, thr, auto_stretch, parent=self,
            auto_sigma=auto_sigma,
            with_overlay=self._preview_shown,
        )
        worker.finished.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_failed)
        self._detection_worker = worker
        worker.start()

    def _on_worker_done(self, overlay: np.ndarray, count: int, used_sigma: float = 0.0):
        if getattr(self, "_closing", False):
            self._detection_worker = None
            return
        worker = self._detection_worker
        self._detection_worker = None

        if self._pending_threshold is not None:
            # Let the old worker finish before starting a new one
            if worker is not None:
                worker.wait(2000)  # wait up to 2s — already done since signal fired
            QTimer.singleShot(0, self._update_star_preview)
            return

        auto_on = bool(self.auto_sigma_chk.isChecked()) if hasattr(self, "auto_sigma_chk") else False

        # Drive the slider from the resolved σ when Auto picked it, so the
        # user can see what the waterfall landed on. (Clamp into slider range.)
        if auto_on and used_sigma > 0.0:
            try:
                iv = int(round(float(used_sigma)))
                iv = max(self.thr_slider.minimum(), min(self.thr_slider.maximum(), iv))
                self.thr_slider.blockSignals(True)
                self.thr_slider.setValue(iv)
                self.thr_label.setText(str(iv))
            finally:
                try:
                    self.thr_slider.blockSignals(False)
                except Exception:
                    pass
            self.star_count.setText(self.tr(
                "Auto σ={0}: detected {1} stars."
            ).format(int(round(used_sigma)), int(count)))
        else:
            self.star_count.setText(self.tr("Detected {0} stars.").format(int(count)))

        # Detection succeeded — Show Preview is now safe to click. Stays
        # enabled across subsequent re-detections even once the preview is
        # visible, as a no-op (handler guards on _preview_shown).
        try:
            self.btn_show_preview.setEnabled(True)
        except Exception:
            pass

        # No overlay unless the user asked for the preview. Nothing to draw
        # in cheap-mode, and the preview QLabel is hidden anyway.
        if overlay is None:
            return

        try:
            overlay8 = np.ascontiguousarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8))
            h, w, _ = overlay8.shape
            qimg = QImage(overlay8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg.copy()).scaled(
                self.preview.width(), self.preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview.setPixmap(pm)
        except Exception:
            self.preview.clear()

    def _on_worker_failed(self, error: str):
        if getattr(self, "_closing", False):
            self._detection_worker = None
            return
        worker = self._detection_worker
        self._detection_worker = None
        if self._pending_threshold is not None:
            if worker is not None:
                worker.wait(2000)
            QTimer.singleShot(0, self._update_star_preview)
            return
        self.star_count.setText(self.tr("Detection failed."))
        self.preview.clear()

    def _on_show_preview_clicked(self):
        """Flip the dialog into preview-shown mode and kick off a detection
        pass that will build the overlay. One-shot — once the preview is
        visible further slider moves rebuild it via the normal debounce
        path (which reads self._preview_shown), so the button hides itself."""
        if self._preview_shown:
            return
        self._preview_shown = True
        try:
            self.preview.setVisible(True)
            self.btn_show_preview.setVisible(False)
        except Exception:
            pass
        # Fire a fresh detection — this one builds the overlay. The result
        # handler is the usual _on_worker_done.
        self._update_star_preview()

    # ---- preset emit (grip) ---------------------------------------------
    def get_preset(self) -> dict:
        """Emit current state as the WB preset schema (inverse of seed_from_preset).
        Mirrors exactly what _on_apply records for Replay Last."""
        mode = self.type_combo.currentText()
        if mode == "Manual":
            return {
                "mode": "manual",
                "r_gain": float(self.r_spin.value()),
                "g_gain": float(self.g_spin.value()),
                "b_gain": float(self.b_spin.value()),
            }
        if mode == "Auto":
            return {"mode": "auto"}
        return {
            "mode": "star",
            "threshold": float(self.thr_slider.value()),
            "use_color_matrix": bool(self.chk_color_matrix.isChecked()),
            "matrix_strength": float(self.cm_strength_slider.value()) / 100.0,
            "auto_sigma": (
                bool(self.auto_sigma_chk.isChecked())
                if hasattr(self, "auto_sigma_chk") else True
            ),
        }

    # ---- preset seed (double-click open) --------------------------------
    def seed_from_preset(self, p: dict | None):
        """Inverse of get_preset. Block signals around all sets, then re-sync
        the threshold label and mode-group visibility by hand."""
        p = dict(p or {})
        mode = (p.get("mode") or "star").lower()

        widgets = [
            self.type_combo, self.thr_slider,
            self.r_spin, self.g_spin, self.b_spin,
            self.chk_color_matrix, self.cm_strength_slider,
        ]
        if hasattr(self, "auto_sigma_chk"):
            widgets.append(self.auto_sigma_chk)
        for wdg in widgets:
            try:
                wdg.blockSignals(True)
            except Exception:
                pass
        try:
            if mode == "manual":
                self.type_combo.setCurrentText("Manual")
                self.r_spin.setValue(float(p.get("r_gain", 1.0)))
                self.g_spin.setValue(float(p.get("g_gain", 1.0)))
                self.b_spin.setValue(float(p.get("b_gain", 1.0)))
            elif mode == "auto":
                self.type_combo.setCurrentText("Auto")
            else:  # star
                self.type_combo.setCurrentText("Star-Based")
                iv = int(round(float(p.get("threshold", 50.0))))
                iv = max(self.thr_slider.minimum(), min(self.thr_slider.maximum(), iv))
                self.thr_slider.setValue(iv)
                self.thr_label.setText(str(iv))
                self.chk_color_matrix.setChecked(bool(p.get("use_color_matrix", False)))
                ms = int(round(float(p.get("matrix_strength", 0.70)) * 100.0))
                self.cm_strength_slider.setValue(max(0, min(100, ms)))
                if hasattr(self, "auto_sigma_chk"):
                    # Default: True — the whole point is to turn guess-and-check off.
                    self.auto_sigma_chk.setChecked(bool(p.get("auto_sigma", True)))
        finally:
            for wdg in widgets:
                try:
                    wdg.blockSignals(False)
                except Exception:
                    pass

        # Re-apply dependent state that the blocked signals would have set.
        self._update_mode_widgets()
        # Blocked signals skipped the strength label + enable sync.
        try:
            self._on_matrix_strength_changed(self.cm_strength_slider.value())
            on_cm = self.chk_color_matrix.isChecked()
            self.cm_strength_slider.setEnabled(on_cm)
            self.cm_strength_label.setEnabled(on_cm)
            self.cm_strength_caption.setEnabled(on_cm)
        except Exception:
            pass
        # Re-sync slider enabled state with the restored Auto flag.
        if hasattr(self, "auto_sigma_chk"):
            try:
                self.thr_slider.setEnabled(not self.auto_sigma_chk.isChecked())
            except Exception:
                pass

    # ---- apply ----------------------------------------------------------
    def _on_apply(self):
        mode = self.type_combo.currentText()

        # Find the main window that carries the replay machinery
        main = self.parent()
        while main is not None and not hasattr(main, "replay_last_action_on_base"):
            main = main.parent()

        def _record_preset_for_replay(preset: dict):
            """Helper to stash the WB preset on the main window for Replay Last Action."""
            if main is None:
                return
            try:
                main._last_headless_command = {
                    "command_id": "white_balance",
                    "preset": preset,
                }
                if hasattr(main, "_log"):
                    mode_str = str(preset.get("mode", "star")).lower()
                    if mode_str == "manual":
                        r = float(preset.get("r_gain", 1.0))
                        g = float(preset.get("g_gain", 1.0))
                        b = float(preset.get("b_gain", 1.0))
                        main._log(
                            f"[Replay] Recorded White Balance preset "
                            f"(mode=manual, R={r:.3f}, G={g:.3f}, B={b:.3f})"
                        )
                    elif mode_str == "auto":
                        main._log("[Replay] Recorded White Balance preset (mode=auto)")
                    else:
                        thr = float(preset.get("threshold", 50.0))
                        auto = bool(preset.get("auto_sigma", False))
                        resolved = preset.get("resolved_threshold")
                        if preset.get("use_color_matrix"):
                            main._log(
                                f"[Replay] White Balance color matrix ON "
                                f"(strength "
                                f"{float(preset.get('matrix_strength', 0.70))*100:.0f}%)"
                            )
                        if auto and resolved is not None:
                            main._log(
                                f"[Replay] Recorded White Balance preset "
                                f"(mode=star, auto σ → {float(resolved):.1f})"
                            )
                        elif auto:
                            main._log(
                                f"[Replay] Recorded White Balance preset "
                                f"(mode=star, auto σ)"
                            )
                        else:
                            main._log(
                                f"[Replay] Recorded White Balance preset "
                                f"(mode=star, threshold={thr:.1f})"
                            )
            except Exception:
                # Logging/recording must never break the dialog
                pass

        try:
            if mode == "Manual":
                preset = {
                    "mode": "manual",
                    "r_gain": float(self.r_spin.value()),
                    "g_gain": float(self.g_spin.value()),
                    "b_gain": float(self.b_spin.value()),
                }

                _record_preset_for_replay(preset)
                apply_white_balance_to_doc(self.doc, preset)
                self._finish_and_close()
                return

            elif mode == "Auto":
                preset = {"mode": "auto"}

                _record_preset_for_replay(preset)
                apply_white_balance_to_doc(self.doc, preset)
                self._finish_and_close()
                return

            else:  # Star-Based
                thr = float(self.thr_slider.value())
                use_cm = bool(self.chk_color_matrix.isChecked())
                cm_strength = float(self.cm_strength_slider.value()) / 100.0
                auto_sigma = (
                    bool(self.auto_sigma_chk.isChecked())
                    if hasattr(self, "auto_sigma_chk") else False
                )

                preset = {
                    "mode": "star",
                    "threshold": thr,
                    "use_color_matrix": use_cm,
                    "matrix_strength": cm_strength,
                    "auto_sigma": auto_sigma,
                }

                base = _to_float01(
                    np.asarray(self.doc.image).astype(np.float32, copy=False)
                )

                # Captured by on_sigma_resolved so we can log / drive the UI
                # with the σ the waterfall actually settled on.
                resolved: dict = {"sigma": thr}
                def _capture_sigma(sig: float):
                    resolved["sigma"] = float(sig)

                # reuse_cached_sources=True lets the dialog's open-time
                # detection (either Preview or Show-Preview, which both ran
                # via detect_stars_for_wb and populated the module cache)
                # satisfy this call without a second SEP waterfall. The
                # backend only honors the cache when image shape + σ match,
                # and in auto mode it keys off the resolved σ, so this is a
                # safe reuse — not a stale-cache hazard.
                result = apply_star_based_white_balance(
                    base,
                    threshold=thr,
                    autostretch=False,
                    reuse_cached_sources=True,
                    return_star_colors=True,
                    use_color_matrix=use_cm,     # ← pass through
                    matrix_strength=cm_strength,
                    auto_sigma=auto_sigma,
                    on_sigma_resolved=_capture_sigma,
                )
                # Record the σ actually used — matters for Replay Last when
                # Auto was on, so the replay reproduces the same waterfall-
                # picked value even if the user is on a different field now.
                preset["resolved_threshold"] = float(resolved["sigma"])

                # Expected: (out, count, overlay, raw_colors, after_colors)
                if len(result) < 5:
                    raise RuntimeError(
                        "Star-based WB did not return star color arrays. "
                        "Ensure return_star_colors=True is supported."
                    )

                out, star_count, _overlay, raw_colors, after_colors = result

                # Optional destination-mask blend, same as headless path
                m = _active_mask_array_from_doc(self.doc)
                if m is not None:
                    m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
                    base_for_blend = _to_float01(
                        np.asarray(self.doc.image).astype(np.float32, copy=False)
                    )
                    out = base_for_blend * (1.0 - m3) + out * m3

                # Commit to the document, including the preset in metadata
                self.doc.apply_edit(
                    out.astype(np.float32, copy=False),
                    metadata={"step_name": "White Balance", "preset": preset},
                    step_name="White Balance",
                )

                # Record for Replay Last Action (using the same preset we just stored in metadata)
                _record_preset_for_replay(preset)

                # Plot is OPTIONAL and must never make the whole WB look like it failed
                plot_error = None
                try:
                    if (
                        raw_colors is not None
                        and after_colors is not None
                        and raw_colors.size
                        and after_colors.size
                    ):
                        fig = build_star_color_ratios_figure(raw_colors, after_colors)
                        dlg = WhiteBalanceResultDialog(self._main or self.parent(), fig, int(star_count))
                        dlg.show()
                        dlg.raise_()
                        dlg.activateWindow()

                        # Keep alive so Python doesn't garbage collect it immediately
                        if not hasattr(self, "_result_dialogs"):
                            self._result_dialogs = []
                        self._result_dialogs.append(dlg)

                        try:
                            dlg.finished.connect(
                                lambda *_: self._result_dialogs.remove(dlg)
                                if dlg in self._result_dialogs else None
                            )
                        except Exception:
                            pass
                    else:
                        QMessageBox.information(
                            self,
                            self.tr("White Balance"),
                            self.tr("Star-Based WB applied.\nDetected {0} stars.").format(int(star_count)),
                        )
                except Exception as e_plot:
                    plot_error = e_plot

                if plot_error is not None:
                    try:
                        if main is not None and hasattr(main, "_log"):
                            main._log(f"[White Balance] Summary plot failed after successful apply: {type(plot_error).__name__}: {plot_error}")
                    except Exception:
                        pass

                    QMessageBox.warning(
                        self,
                        self.tr("White Balance"),
                        self.tr(
                            "White Balance was applied successfully.\n"
                            "Detected {0} stars.\n\n"
                            "The summary plot could not be shown:\n{1}"
                        ).format(int(star_count), str(plot_error)),
                    )

                self._finish_and_close()
                return

        except Exception as e:
            QMessageBox.critical(
                self,
                self.tr("White Balance"),
                self.tr("Failed to apply White Balance:\n{0}").format(e)
            )

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
        except Exception:
            pass

    def _finish_and_close(self):
        """
        Close this dialog after a successful apply.
        Use accept() so it behaves like a successful completion.
        """
        try:
            self.accept()
        except Exception:
            self.close()

    def _stop_detection_worker(self):
        """Cancel + detach any running detection worker so its QThread is never
        destroyed while the OS thread is still running (that hard-crashes Qt)."""
        self._closing = True
        try:
            self._debounce.stop()
        except Exception:
            pass
        worker = getattr(self, "_detection_worker", None)
        self._detection_worker = None
        if worker is None:
            return
        # Stop it calling back into this (soon-destroyed) dialog.
        for _sig in ("finished", "failed"):
            try:
                getattr(worker, _sig).disconnect()
            except Exception:
                pass
        try:
            worker.cancel()
        except Exception:
            pass
        try:
            running = worker.isRunning()
        except Exception:
            running = False
        if running:
            # Detach from the dialog and keep it alive at module scope so it can
            # finish on its own; it removes itself when done. No blocking wait
            # (SEP can't be interrupted mid-call) and no terminate.
            try:
                worker.setParent(None)
            except Exception:
                pass
            _orphaned_workers.append(worker)
            def _cleanup_orphan(*_a, w=worker):
                try:
                    _orphaned_workers.remove(w)
                except ValueError:
                    pass
            try:
                worker.finished.connect(_cleanup_orphan)
                worker.failed.connect(_cleanup_orphan)
            except Exception:
                pass

    def closeEvent(self, ev):
        # Stop debounce so no new workers can start
        try:
            self._debounce.stop()
        except Exception:
            pass

        # Detach active-doc signal
        try:
            if self._active_doc_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._active_doc_conn = False

        self._stop_detection_worker()

        super().closeEvent(ev)

    def _cleanup(self):
        try:
            if self._main and hasattr(self._main, "currentDocumentChanged") and self._active_doc_conn:
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._active_doc_conn = False

        try:
            if getattr(self, "_debounce", None) is not None:
                self._debounce.stop()
        except Exception:
            pass

        # macOS: must cancel + wait before the thread object is destroyed
        try:
            worker = getattr(self, "_detection_worker", None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(3000)   # give it up to 3s to finish
            self._detection_worker = None
        except Exception:
            pass


def open_white_balance_with_preset(main_window, preset: dict | None = None):
    from PyQt6.QtGui import QIcon

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
        return

    try:
        from setiastro.saspro.resources import whitebalance_path
        _icon = QIcon(whitebalance_path)
    except Exception:
        _icon = QIcon()

    dlg = WhiteBalanceDialog(main_window, doc, _icon)

    # No on-show reset in this dialog -> seed directly before show.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    # Retain against GC (dialog has WA_DeleteOnClose).
    try:
        main_window._wb_preset_dialog = dlg
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg