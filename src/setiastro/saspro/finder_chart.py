from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QFileDialog, QMessageBox, QSpinBox, QSlider, QApplication
)

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord

from setiastro.saspro.bright_stars import BRIGHT_STARS

if TYPE_CHECKING:
    from astropy.wcs import WCS as AstropyWCS
    from astropy.coordinates import SkyCoord as AstropySkyCoord
else:
    AstropyWCS = object
    AstropySkyCoord = object


@dataclass
class FinderChartRequest:
    survey: str
    scale_mult: int
    show_grid: bool
    show_star_names: bool = False
    star_mag_limit: float = 2.0   # optional, for later
    out_px: int = 900
    overlay_opacity: float = 0.35




def get_doc_wcs(meta: dict) -> Optional["AstropyWCS"]:
    """Prefer prebuilt WCS object; else build from header. Always return *celestial* (2D) WCS."""
    if WCS is None:
        return None

    w = meta.get("wcs")
    if w is None:
        hdr = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        if hdr is None:
            return None

        # normalize to fits.Header if needed
        if fits is not None and not isinstance(hdr, fits.Header):
            try:
                h2 = fits.Header()
                for k, v in dict(hdr).items():
                    try:
                        h2[k] = v
                    except Exception:
                        pass
                hdr = h2
            except Exception:
                return None

        try:
            w = WCS(hdr, relax=True)
        except Exception:
            return None

    # --- CRITICAL FIX: if WCS has extra axes, use celestial slice ---
    try:
        if hasattr(w, "celestial"):
            wc = w.celestial
            if wc is not None:
                return wc
    except Exception:
        pass

    return w


def image_footprint_sky(wcs: "WCS", w: int, h: int):
    """Return (corners SkyCoord[4], center SkyCoord)."""
    wc = wcs
    try:
        if hasattr(wcs, "celestial") and wcs.celestial is not None:
            wc = wcs.celestial
    except Exception:
        pass

    xs = np.array([0.5, w - 0.5, w - 0.5, 0.5], dtype=np.float64)
    ys = np.array([0.5, 0.5, h - 0.5, h - 0.5], dtype=np.float64)

    corners = wc.pixel_to_world(xs, ys)
    center = wc.pixel_to_world(np.array([w / 2.0]), np.array([h / 2.0]))
    return corners, center


def _ang_sep_deg(a1, d1, a2, d2) -> float:
    """Small helper; inputs degrees."""
    ra1 = math.radians(a1); dec1 = math.radians(d1)
    ra2 = math.radians(a2); dec2 = math.radians(d2)
    return math.degrees(math.acos(
        max(-1.0, min(1.0, math.sin(dec1)*math.sin(dec2)
                      + math.cos(dec1)*math.cos(dec2)*math.cos(ra1-ra2)))
    ))


def estimate_fov_deg(corners: "AstropySkyCoord") -> Tuple[float, float]:
    """Approx FOV width/height in degrees using corner separations."""
    ra = corners.ra.deg
    dec = corners.dec.deg

    w1 = _ang_sep_deg(ra[0], dec[0], ra[1], dec[1])
    w2 = _ang_sep_deg(ra[3], dec[3], ra[2], dec[2])
    width = 0.5 * (w1 + w2)

    h1 = _ang_sep_deg(ra[0], dec[0], ra[3], dec[3])
    h2 = _ang_sep_deg(ra[1], dec[1], ra[2], dec[2])
    height = 0.5 * (h1 + h2)

    return float(width), float(height)


def _survey_to_hips_id(label: str) -> str:
    key = (label or "").strip().lower()
    if "dss" in key:
        return "CDS/P/DSS2/color"
    if "pan" in key:
        return "CDS/P/PanSTARRS/DR1/color"
    if "gaia" in key:
        return "CDS/P/Gaia/DR3/flux-color"
    return "CDS/P/DSS2/color"

def try_fetch_hips_cutout(center: "SkyCoord", fov_deg: float, out_px: int, survey_label: str):
    """
    Returns (rgb_float01, bg_wcs_celestial, err)
      - rgb_float01: (H,W,3) float32 [0..1] or None
      - bg_wcs_celestial: WCS(2D) or None
      - err: str or None
    """
    hips_id = _survey_to_hips_id(survey_label)

    try:
        from astroquery.hips2fits import hips2fits
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    def _decode(hdul):
        data = np.array(hdul[0].data, dtype=np.float32)

        bg_wcs = None
        try:
            bg_wcs = WCS(hdul[0].header, relax=True)
            if hasattr(bg_wcs, "celestial") and bg_wcs.celestial is not None:
                bg_wcs = bg_wcs.celestial
        except Exception:
            bg_wcs = None

        # Normalize to RGB
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        elif data.ndim == 3 and data.shape[0] in (3, 4):
            data = np.transpose(data[:3, ...], (1, 2, 0))
        elif data.ndim == 3 and data.shape[2] >= 3:
            data = data[..., :3]

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        lo, hi = np.percentile(data, [1.0, 99.5])
        if hi > lo:
            data = (data - lo) / (hi - lo)

        return np.clip(data, 0.0, 1.0), bg_wcs

    out_px = int(out_px)
    ra_deg = float(center.ra.deg)
    dec_deg = float(center.dec.deg)

    # Prefer quantity for fov, but fall back if this build wants float
    try:
        import astropy.units as u
        fov = float(fov_deg) * u.deg
    except Exception:
        fov = float(fov_deg)

    last_err = None

    # ---- Correct signature for YOUR astroquery 0.4.11 ----
    # query(hips, width, height, projection, ra, dec, fov, *, coordsys='icrs', ...)
    import astropy.units as u
    from astropy.coordinates import Angle

    out_px = int(out_px)

    # IMPORTANT: pass Angle/Quantity, not floats
    ra  = center.ra.to(u.deg)          # Angle
    dec = center.dec.to(u.deg)         # Angle
    fov = Angle(float(fov_deg), unit=u.deg)

    try:
        hdul = hips2fits.query(
            hips_id,
            out_px, out_px,
            "TAN",
            ra, dec,
            fov,
            coordsys="icrs",
            format="fits",
        )
        rgb01, bg_wcs = _decode(hdul)
        return rgb01, bg_wcs, None

    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    return None, None, last_err


def render_finder_chart(doc_image: np.ndarray, meta: dict, req: FinderChartRequest) -> Optional[np.ndarray]:
    if WCS is None:
        return None

    doc_wcs = get_doc_wcs(meta)
    if doc_wcs is None:
        return None

    H, Wimg = doc_image.shape[:2]
    corners, center = image_footprint_sky(doc_wcs, Wimg, H)
    fov_w, fov_h = estimate_fov_deg(corners)
    fov = max(fov_w, fov_h) * float(req.scale_mult)

    # Fetch background (+ WCS + error string)
    bg, bg_wcs, err = try_fetch_hips_cutout(center[0], fov_deg=fov, out_px=req.out_px, survey_label=req.survey)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(req.out_px / 100.0, req.out_px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # --- Draw background OR error message ---
    if bg is None:
        ax.set_facecolor((0, 0, 0))
        msg = "No HiPS background.\n" + (err or "Unknown error")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    else:
        # If you want doc overlay, do it here
        if bg_wcs is not None and doc_wcs is not None:
            try:
                overlay_u8 = _overlay_doc_on_bg(bg, bg_wcs, doc_image, doc_wcs, alpha=req.overlay_opacity)
                ax.imshow(overlay_u8, origin="lower")
            except Exception:
                # fallback to plain bg if overlay fails
                ax.imshow(bg, origin="lower")
        else:
            ax.imshow(bg, origin="lower")

        # Optional: draw WCS-correct footprint polygon
        if bg_wcs is not None:
            try:
                xs, ys = bg_wcs.world_to_pixel(corners)
                ax.plot(
                    [xs[0], xs[1], xs[2], xs[3], xs[0]],
                    [ys[0], ys[1], ys[2], ys[3], ys[0]],
                    linewidth=2
                )
            except Exception:
                pass

    # center crosshair (axes coords)
    ax.plot([0.5], [0.5], marker="+", markersize=20, transform=ax.transAxes)

    # labels
    ra = float(center[0].ra.deg)
    dec = float(center[0].dec.deg)
    ax.text(
        0.02, 0.98,
        f"{req.survey}  |  {req.scale_mult}×FOV\nRA {ra:.6f}°  Dec {dec:.6f}°\nFOV ~ {fov*60:.1f}′",
        transform=ax.transAxes, va="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.45), edgecolor=(1, 1, 1, 0.12))
    )

    if req.show_grid:
        # keep axis ON so grid can render
        ax.set_axis_on()
        ax.set_xticks(np.linspace(0, req.out_px, 7))
        ax.set_yticks(np.linspace(0, req.out_px, 7))
        ax.grid(True, alpha=0.35)
        # optionally hide tick labels but keep grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()

    plt.close(fig)
    return rgb

def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] > 3:
        a = a[..., :3]
    a = a.astype(np.float32)
    # simple robust normalize for display
    lo, hi = np.percentile(a, [1.0, 99.5])
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

def _overlay_doc_on_bg(bg_rgb01: np.ndarray, bg_wcs: "WCS", doc_img: np.ndarray, doc_wcs: "WCS", alpha=0.35) -> np.ndarray:
    import cv2

    Hbg, Wbg = bg_rgb01.shape[:2]
    bg_u8 = (np.clip(bg_rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    doc_u8 = _to_u8_rgb(doc_img)
    H, W = doc_u8.shape[:2]

    # doc pixel corners -> sky -> bg pixels
    src = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype=np.float32)
    sky = doc_wcs.pixel_to_world(src[:,0], src[:,1])   # SkyCoord
    xbg, ybg = bg_wcs.world_to_pixel(sky)
    dst = np.stack([xbg, ybg], axis=1).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc_u8, M, (Wbg, Hbg), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # alpha blend where warped has content
    mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    out = bg_u8.astype(np.float32) * (1 - alpha*mask) + warped.astype(np.float32) * (alpha*mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rgb_u8_to_qimage(rgb_u8: np.ndarray) -> QImage:
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    h, w, _ = rgb_u8.shape
    bpl = rgb_u8.strides[0]
    # QImage uses the buffer; to be safe, copy via .copy() when making pixmap
    return QImage(rgb_u8.data, w, h, bpl, QImage.Format.Format_RGB888)

def _draw_star_names(ax, bg_wcs: "WCS", center: "SkyCoord", fov_deg: float, *, max_labels: int = 30):
    if bg_wcs is None:
        return

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # 1) Cull by a simple spherical radius from center
    ra0 = float(center.ra.deg)
    dec0 = float(center.dec.deg)

    # generous radius: half-diagonal-ish
    radius = float(fov_deg) * 0.75

    rows = []
    for (name, ra, dec, vmag) in BRIGHT_STARS:
        # quick dec prefilter
        if abs(dec - dec0) > radius + 2.0:
            continue
        # use SkyCoord sep for correctness
        c0 = SkyCoord(ra0*u.deg, dec0*u.deg, frame="icrs")
        c1 = SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame="icrs")
        if c0.separation(c1).deg <= radius:
            rows.append((name, float(ra), float(dec), float(vmag)))

    if not rows:
        return

    # 2) Sort by brightness (lowest mag first), then cap
    rows.sort(key=lambda r: r[3])
    rows = rows[:max_labels]

    # 3) Project to pixels
    coords = SkyCoord([r[1] for r in rows]*u.deg, [r[2] for r in rows]*u.deg, frame="icrs")
    xs, ys = bg_wcs.world_to_pixel(coords)

    # 4) Anti-clutter: keep only one label per cell in a coarse grid
    #    (fast, good enough, avoids “wall of text”)
    kept = []
    cell = 28  # px, tweak to taste
    used = set()

    for (row, x, y) in zip(rows, xs, ys):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        gx = int(x // cell)
        gy = int(y // cell)
        key = (gx, gy)
        if key in used:
            continue
        used.add(key)
        kept.append((row[0], float(x), float(y), row[3]))

    # 5) Draw
    for (name, x, y, vmag) in kept:
        # a tiny marker + label
        ax.plot([x], [y], marker="o", markersize=2.5, alpha=0.8, transform=ax.get_transform("pixel"))
        ax.text(
            x + 6, y + 4,
            name,
            fontsize=9,
            alpha=0.9,
            transform=ax.get_transform("pixel"),
        )


def render_finder_chart_cached(
    *,
    doc_image: np.ndarray,
    doc_wcs: WCS,
    corners: SkyCoord,
    center: SkyCoord,
    fov_deg: float,
    req: FinderChartRequest,
    bg: Optional[np.ndarray],
    bg_wcs: Optional[WCS],
    err: Optional[str],
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(req.out_px / 100.0, req.out_px / 100.0), dpi=100)

    # Use WCSAxes when we have bg_wcs so we can draw RA/Dec labels & grid properly
    if bg_wcs is not None:
        ax = fig.add_subplot(111, projection=bg_wcs)
    else:
        ax = fig.add_axes([0, 0, 1, 1])

    # ---- background (or error) ----
    if bg is None:
        ax.set_facecolor((0, 0, 0))
        msg = "No HiPS background.\n" + (err or "Unknown error")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    else:
        # overlay (optional)
        if bg_wcs is not None and doc_wcs is not None and req.overlay_opacity > 0.0:
            try:
                overlay_u8 = _overlay_doc_on_bg(bg, bg_wcs, doc_image, doc_wcs, alpha=req.overlay_opacity)
                ax.imshow(overlay_u8, origin="lower")
            except Exception:
                ax.imshow(bg, origin="lower")
        else:
            ax.imshow(bg, origin="lower")

        # footprint polygon in pixel coords
        if bg_wcs is not None:
            try:
                xs, ys = bg_wcs.world_to_pixel(corners)
                ax.plot(
                    [xs[0], xs[1], xs[2], xs[3], xs[0]],
                    [ys[0], ys[1], ys[2], ys[3], ys[0]],
                    linewidth=2,
                    transform=ax.get_transform("pixel"),
                )
            except Exception:
                pass

    # center crosshair
    ax.plot([0.5], [0.5], marker="+", markersize=20, transform=ax.transAxes)

    # top-left info text
    ra = float(center[0].ra.deg)
    dec = float(center[0].dec.deg)
    ax.text(
        0.02, 0.98,
        f"{req.survey}  |  {req.scale_mult}×FOV\nRA {ra:.6f}°  Dec {dec:.6f}°\nFOV ~ {fov_deg*60:.1f}′",
        transform=ax.transAxes, va="top"
    )

    # ---- grid + RA/Dec labels ----
    if bg_wcs is not None:
        # RA/Dec edge labels
        try:
            ax.coords[0].set_axislabel("RA")
            ax.coords[1].set_axislabel("Dec")
        except Exception:
            pass

        # toggle grid lines
        try:
            ax.coords.grid(bool(req.show_grid), alpha=0.35)
        except Exception:
            pass

        # If grid is off, you may still want edge tick labels; keep axis visible.
        # WCSAxes handles ticks/labels automatically.
    else:
        # fallback: pixel grid only
        if req.show_grid:
            ax.set_axis_on()
            ax.set_xticks(np.linspace(0, req.out_px, 7))
            ax.set_yticks(np.linspace(0, req.out_px, 7))
            ax.grid(True, alpha=0.35)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

    # ---- star names overlay ----
    if getattr(req, "show_star_names", False) and (bg_wcs is not None):
        try:
            _draw_star_names(ax, bg_wcs, center[0], fov_deg)
        except Exception:
            pass

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


class FinderChartDialog(QDialog):
    """
    Minimal v1 Finder Chart dialog:
    - Survey dropdown
    - Size multiplier dropdown
    - Show grid checkbox
    - Render preview
    - Save PNG
    - Send to New Document (push into SASpro)
    """
    def __init__(self, doc, settings, parent=None):
        super().__init__(parent)
        self._doc = doc
        self._settings = settings
        self._last_rgb_u8: Optional[np.ndarray] = None
        # ---- HiPS cache (avoid refetching for UI-only changes) ----
        self._hips_cache_key = None
        self._hips_bg = None          # float01 RGB background
        self._hips_wcs = None         # celestial WCS for background
        self._hips_err = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_debounced_fire)
        self._pending_force_refetch = False
        # Cached geometry derived from the current doc WCS (used for overlays/labels)
        self._doc_wcs_cached = None
        self._corners_cached = None
        self._center_cached = None
        self._fov_deg_cached = None
        self.setWindowTitle("Finder Chart…")
        self.setModal(False)
        self.resize(920, 980)

        root = QVBoxLayout(self)

        # controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Survey:"))
        self.cmb_survey = QComboBox()
        self.cmb_survey.addItems(["DSS2", "Pan-STARRS", "Gaia"])
        row.addWidget(self.cmb_survey)

        row.addSpacing(12)
        row.addWidget(QLabel("Size:"))
        self.cmb_size = QComboBox()
        self.cmb_size.addItems(["2× FOV", "4× FOV", "8× FOV"])
        row.addWidget(self.cmb_size)

        row.addSpacing(12)
        self.chk_grid = QCheckBox("Show grid")
        row.addWidget(self.chk_grid)
        row.addSpacing(6)
        self.chk_stars = QCheckBox("Star names")
        row.addWidget(self.chk_stars)

        row.addSpacing(12)
        row.addWidget(QLabel("Image opacity:"))
        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(35)   # matches default 0.35
        self.sld_opacity.setFixedWidth(140)
        row.addWidget(self.sld_opacity)

        self.lbl_opacity = QLabel("35%")
        self.lbl_opacity.setFixedWidth(40)
        row.addWidget(self.lbl_opacity)

        row.addStretch(1)

        row.addWidget(QLabel("Output px:"))
        self.sb_px = QSpinBox()
        self.sb_px.setRange(300, 2400)
        self.sb_px.setSingleStep(100)
        self.sb_px.setValue(900)
        row.addWidget(self.sb_px)

        self.btn_render = QPushButton("Render")
        row.addWidget(self.btn_render)

        root.addLayout(row)

        # preview
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setMinimumHeight(700)
        self.lbl.setStyleSheet("QLabel { background:#111; border:1px solid #333; }")
        root.addWidget(self.lbl, 1)

        # buttons
        brow = QHBoxLayout()
        self.btn_send = QPushButton("Send to New Document")
        self.btn_save = QPushButton("Save PNG…")
        self.btn_close = QPushButton("Close")
        brow.addWidget(self.btn_send)
        brow.addStretch(1)
        brow.addWidget(self.btn_save)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

        self.btn_render.clicked.connect(lambda: self._render_now(force_refetch=True))
        self.btn_save.clicked.connect(self._save_png)
        self.btn_send.clicked.connect(self._send_to_new_doc)
        self.btn_close.clicked.connect(self.close)
        # button: immediate + force refetch
        self.btn_render.clicked.connect(lambda: self._render_now(force_refetch=True))

        # grid: debounced (no refetch)
        self.chk_grid.toggled.connect(lambda _=False: self._schedule_render(force_refetch=False, delay_ms=150))

        # survey/size/px: immediate refetch (or debounce if you want, but refetch is required)
        self.cmb_survey.currentIndexChanged.connect(lambda _=0: self._render_now(force_refetch=True))
        self.cmb_size.currentIndexChanged.connect(lambda _=0: self._render_now(force_refetch=True))
        self.sb_px.valueChanged.connect(lambda _=0: self._render_now(force_refetch=True))

        # opacity: update label + debounce render (no refetch)
        self.sld_opacity.valueChanged.connect(self._on_opacity_changed)
        self.sld_opacity.sliderReleased.connect(lambda: self._render_now(force_refetch=False))
        # auto render once
        # placeholder so the user sees *something* immediately
        self.lbl.setText("Fetching survey background…")
        self.lbl.setStyleSheet("QLabel { background:#111; border:1px solid #333; color:#ccc; }")

        self.chk_stars.toggled.connect(lambda _=False: self._schedule_render(force_refetch=False, delay_ms=150))

        # kick initial render AFTER the dialog has had a chance to show/paint
        QTimer.singleShot(0, self._initial_render)

    def _set_busy(self, busy: bool, msg: str = "Working…"):
        # lightweight busy state (no threads)
        self.btn_render.setEnabled(not busy)
        self.btn_send.setEnabled((not busy) and (self._last_rgb_u8 is not None))
        self.btn_save.setEnabled((not busy) and (self._last_rgb_u8 is not None))


    def _initial_render(self):
        self._set_busy(True, "Fetching survey background…")
        # schedule again so the UI paints the busy message + cursor first
        QTimer.singleShot(0, lambda: self._render_now(force_refetch=True))


    def _schedule_render(self, *, force_refetch: bool = False, delay_ms: int = 200):
        # Accumulate "force" requests until the next fire
        self._pending_force_refetch = self._pending_force_refetch or bool(force_refetch)

        # Restart debounce timer
        self._render_timer.stop()
        self._render_timer.start(int(delay_ms))

    def _render_debounced_fire(self):
        force = bool(self._pending_force_refetch)
        self._pending_force_refetch = False
        self._render(force_refetch=force)

    def _render_now(self, *, force_refetch: bool = False):
        # Cancel any pending debounced render and render immediately
        self._render_timer.stop()
        self._pending_force_refetch = False
        self._render(force_refetch=force_refetch)


    def _compute_doc_geometry(self, img: np.ndarray, meta: dict, req: FinderChartRequest):
        doc_wcs = get_doc_wcs(meta)
        if doc_wcs is None:
            return None, None, None, None

        H, Wimg = img.shape[:2]
        corners, center = image_footprint_sky(doc_wcs, Wimg, H)
        fov_w, fov_h = estimate_fov_deg(corners)
        fov = max(fov_w, fov_h) * float(req.scale_mult)
        return doc_wcs, corners, center, float(fov)

    def _ensure_hips_background(self, req: FinderChartRequest, center: SkyCoord, fov_deg: float, *, force: bool = False):
        # Key only on fetch-driving inputs
        key = (
            str(req.survey),
            int(req.out_px),
            round(float(center.ra.deg), 8),
            round(float(center.dec.deg), 8),
            round(float(fov_deg), 8),
        )

        if (not force) and (self._hips_cache_key == key) and (self._hips_bg is not None):
            return  # cache hit

        bg, bg_wcs, err = try_fetch_hips_cutout(center, fov_deg=fov_deg, out_px=req.out_px, survey_label=req.survey)

        self._hips_cache_key = key
        self._hips_bg = bg
        self._hips_wcs = bg_wcs
        self._hips_err = err


    def _req(self) -> FinderChartRequest:
        survey = str(self.cmb_survey.currentText())
        mult = {0: 2, 1: 4, 2: 8}.get(int(self.cmb_size.currentIndex()), 2)
        show_grid = bool(self.chk_grid.isChecked())
        out_px = int(self.sb_px.value())
        overlay_opacity = float(getattr(self, "sld_opacity", None).value() if hasattr(self, "sld_opacity") else 35) / 100.0
        show_star_names = bool(self.chk_stars.isChecked())
        return FinderChartRequest(
            survey=survey,
            scale_mult=mult,
            show_grid=show_grid,
            show_star_names=show_star_names,
            out_px=out_px,
            overlay_opacity=overlay_opacity,
        )

    def _render(self, *, force_refetch: bool = False):
        self._set_busy(True, "Rendering finder chart…")
        try:
            img = np.asarray(self._doc.image)
            meta = dict(getattr(self._doc, "metadata", None) or {})
            req = self._req()

            # 1) compute geometry from doc WCS
            doc_wcs, corners, center, fov_deg = self._compute_doc_geometry(img, meta, req)
            if doc_wcs is None:
                QMessageBox.warning(self, "Finder Chart", "Could not render finder chart (missing WCS).")
                return

            # cache these for reuse (overlay / footprint / labels)
            self._doc_wcs_cached = doc_wcs
            self._corners_cached = corners
            self._center_cached = center
            self._fov_deg_cached = fov_deg

            # 2) fetch background only if needed
            self._ensure_hips_background(req, center[0], fov_deg, force=force_refetch)

            # 3) render using cached background (NO network)
            rgb = render_finder_chart_cached(
                doc_image=img,
                doc_wcs=doc_wcs,
                corners=corners,
                center=center,
                fov_deg=fov_deg,
                req=req,
                bg=self._hips_bg,
                bg_wcs=self._hips_wcs,
                err=self._hips_err,
            )

            self._last_rgb_u8 = rgb
            qimg = _rgb_u8_to_qimage(rgb).copy()
            self.lbl.setPixmap(QPixmap.fromImage(qimg))

        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", str(e))
        finally:
            self._set_busy(False)

    def _on_opacity_changed(self, v: int):
        self.lbl_opacity.setText(f"{int(v)}%")
        self._schedule_render(delay_ms=200)  # no force refetch


    def _save_png(self):
        if self._last_rgb_u8 is None:
            QMessageBox.information(self, "Finder Chart", "Nothing rendered yet.")
            return

        start_dir = ""
        try:
            start_dir = self._settings.value("finder_chart/last_dir", "", type=str) or ""
        except Exception:
            start_dir = ""

        fn, _ = QFileDialog.getSaveFileName(self, "Save Finder Chart", start_dir, "PNG Image (*.png)")
        if not fn:
            return

        try:
            if not fn.lower().endswith(".png"):
                fn += ".png"
            qimg = _rgb_u8_to_qimage(self._last_rgb_u8).copy()
            ok = qimg.save(fn, "PNG")
            if not ok:
                raise RuntimeError("QImage.save() failed.")
            try:
                self._settings.setValue("finder_chart/last_dir", fn)
                self._settings.sync()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", str(e))

    def _send_to_new_doc(self):
        if self._last_rgb_u8 is None:
            QMessageBox.information(self, "Finder Chart", "Nothing rendered yet.")
            return

        img01 = self._last_rgb_u8.astype(np.float32) / 255.0

        req = self._req()
        meta = {
            "step_name": "Finder Chart",
            "finder_chart": {
                "survey": req.survey,
                "scale_mult": req.scale_mult,
                "show_grid": req.show_grid,
                "out_px": req.out_px,
                "overlay_opacity": req.overlay_opacity,
            },
        }

        dm = self._get_doc_manager()
        if dm is None:
            QMessageBox.warning(self, "Finder Chart", "DocManager not found.")
            return

        title = f"Finder Chart ({req.survey})"

        try:
            if hasattr(dm, "open_array"):
                # matches PerfectPalettePicker
                dm.open_array(img01, metadata=meta, title=title)
                return

            if hasattr(dm, "create_document"):
                # PPP fallback
                doc = dm.create_document(image=img01, metadata=meta, name=title)
                if hasattr(dm, "add_document"):
                    dm.add_document(doc)
                return

            raise RuntimeError("DocManager lacks open_array/create_document")

        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", f"Failed to open new view:\n{e}")


    def _get_doc_manager(self):
        mw = self.parent()
        if mw is None:
            return None
        return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)

    