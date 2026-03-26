# pro/whitebalance.py
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QWidget, QGroupBox,
    QGridLayout, QSlider, QCheckBox, QPushButton, QMessageBox, QDoubleSpinBox
)

# imageops
from setiastro.saspro.imageops.starbasedwhitebalance import apply_star_based_white_balance
from setiastro.saspro.imageops.stretch import stretch_color_image

# Shared utilities
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)

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
    def compute_ratios(pixels: np.ndarray):
        eps = 1e-8
        rb = pixels[:, 0] / (pixels[:, 2] + eps)  # R/B
        gb = pixels[:, 1] / (pixels[:, 2] + eps)  # G/B
        return rb, gb

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

    def _blackbody_ratio_point(temp_k: float):
        rgb = _blackbody_rgb_approx(temp_k).astype(np.float32)
        eps = 1e-8
        rb = float(rgb[0] / max(rgb[2], eps))
        gb = float(rgb[1] / max(rgb[2], eps))
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

        # Keep only finite/in-range-ish points
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        if len(x) < 2:
            return

        ax.plot(x, y, color="black", linewidth=2.0, label="Blackbody locus")

        # Labeled temperature ticks
        tick_temps = [1500, 2000, 2500, 3000, 4000, 6000, 10000]

        tick_pts = []
        for t in tick_temps:
            px, py = _blackbody_ratio_point(t)
            if np.isfinite(px) and np.isfinite(py):
                tick_pts.append((t, px, py))

        if len(tick_pts) < 2:
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

    rb_before, gb_before = _finite(rb_before, gb_before)
    rb_after,  gb_after  = _finite(rb_after,  gb_after)

    # Limit plotted stars
    max_plot_stars = 2000
    total_available = min(len(rb_before), len(rb_after))
    n = min(total_available, max_plot_stars)

    if n > 0:
        rb_before = rb_before[:n]
        gb_before = gb_before[:n]
        rb_after  = rb_after[:n]
        gb_after  = gb_after[:n]

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

    def plot_panel(ax, rb_data, gb_data, title):
        ax.imshow(
            rgb_image,
            extent=(rmin, rmax, gmin, gmax),
            origin="lower",
            aspect="auto",
            alpha=0.9,
        )

        ax.scatter(
            rb_data,
            gb_data,
            alpha=0.45,
            edgecolors="k",
            linewidths=0.25,
            s=12,
            label="Stars",
        )

        if rb_data.size >= 2:
            m, b = np.polyfit(rb_data, gb_data, 1)
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

    plot_panel(ax1, rb_before, gb_before, "Before")
    plot_panel(ax2, rb_after,  gb_after,  "After")

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
        "threshold": float (default 50),
        "reuse_cached_sources": bool (default True),
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
            out, _count, _overlay = apply_star_based_white_balance(
                base_n,
                threshold=thr,
                autostretch=False,
                reuse_cached_sources=False,
                return_star_colors=False
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
        self.finished.connect(lambda *_: self._cleanup())


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
        # threshold slider
        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel(self.tr("Detection threshold (σ):")))
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setMinimum(1); self.thr_slider.setMaximum(100)
        self.thr_slider.setValue(50); self.thr_slider.setTickInterval(10)
        self.thr_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thr_label = QLabel("50")
        thr_row.addWidget(self.thr_slider); thr_row.addWidget(self.thr_label)
        sg.addLayout(thr_row)

        # Reusing cached detections is intentionally disabled for White Balance.
        # The threshold slider must always trigger a fresh star detection pass.
        self.chk_reuse = None

        self.chk_autostretch_overlay = QCheckBox(self.tr("Autostretch overlay preview")); self.chk_autostretch_overlay.setChecked(True)
        sg.addWidget(self.chk_autostretch_overlay)

        # star count + image preview
        self.star_count = QLabel(self.tr("Detecting stars..."))
        sg.addWidget(self.star_count)

        self.preview = QLabel(); self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("border: 1px solid #333;")
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

        self.setLayout(self.main_layout)

        # debounce timer for star preview
        self._debounce = QTimer(self); self._debounce.setSingleShot(True); self._debounce.setInterval(600)
        self._debounce.timeout.connect(self._update_star_preview)

    def _cfg_gain(self, box: QDoubleSpinBox, val: float):
        box.setRange(0.5, 1.5); box.setDecimals(3); box.setSingleStep(0.01); box.setValue(val)

    # ---- events ---------------------------------------------------------
    def _wire_events(self):
        self.type_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self._on_apply)

        self.thr_slider.valueChanged.connect(lambda v: (self.thr_label.setText(str(v)), self._debounce.start()))
        self.chk_autostretch_overlay.toggled.connect(lambda _=None: self._debounce.start())

    def _update_mode_widgets(self):
        t = self.type_combo.currentText()
        self.manual_group.setVisible(t == "Manual")
        self.star_group.setVisible(t == "Star-Based")

    # ---- active document change ------------------------------------
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._update_star_preview()

    # ---- preview --------------------------------------------------------
    def _update_star_preview(self):
        if self.type_combo.currentText() != "Star-Based":
            return
        try:
            img = _to_float01(np.asarray(self.doc.image))
            thr = float(self.thr_slider.value())
            auto = bool(self.chk_autostretch_overlay.isChecked())

            _, count, overlay = apply_star_based_white_balance(
                img,
                threshold=thr,
                autostretch=auto,
                reuse_cached_sources=False,
                return_star_colors=False
            )
            self.star_count.setText(self.tr("Detected {0} stars.").format(count))
            # to pixmap
            overlay8 = np.ascontiguousarray(np.clip(overlay * 255.0, 0, 255).astype(np.uint8))
            h, w, _ = overlay8.shape
            qimg = QImage(overlay8.data, w, h, 3 * w, QImage.Format.Format_RGB888)

            # Make sure the numpy buffer stays alive until QPixmap is created
            pm = QPixmap.fromImage(qimg.copy()).scaled(
                self.preview.width(), self.preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview.setPixmap(pm)
        except Exception as e:
            self.star_count.setText(self.tr("Detection failed."))
            self.preview.clear()

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

            else:  # --- Star-Based: compute here so we can plot like SASv2 ---
                thr = float(self.thr_slider.value())

                preset = {
                    "mode": "star",
                    "threshold": thr,
                }

                base = _to_float01(
                    np.asarray(self.doc.image).astype(np.float32, copy=False)
                )

                # Ask for star colors so we can plot
                result = apply_star_based_white_balance(
                    base,
                    threshold=thr,
                    autostretch=False,
                    reuse_cached_sources=False,
                    return_star_colors=True,
                )

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

    def _cleanup(self):
        # Disconnect active-doc signal so the main window doesn't keep us alive
        try:
            if self._active_doc_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._active_doc_conn = False

        # Stop debounce timer
        try:
            if getattr(self, "_debounce", None) is not None:
                self._debounce.stop()
        except Exception:
            pass


    def closeEvent(self, ev):
        self._cleanup()
        super().closeEvent(ev)