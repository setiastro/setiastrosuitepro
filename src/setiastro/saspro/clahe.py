#saspro/clahe.py
from __future__ import annotations
import json
import time
import numpy as np
import cv2

from PyQt6.QtCore import (
    Qt, QTimer, QSettings, QObject, QRunnable, QThreadPool, QEvent, QPointF,
    pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QImage, QPixmap, QIcon, QCursor, QPen, QBrush, QColor, QPainterPath, QTransform
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QLabel, QPushButton, QSlider, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsItem,
    QMessageBox, QSplitter, QWidget, QScrollArea,
    QDoubleSpinBox, QSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QButtonGroup, QToolButton, QApplication, QSizePolicy,
    QFrame
)

# Import centralized widgets
from setiastro.saspro.widgets.graphics_views import ZoomableGraphicsView
from setiastro.saspro.widgets.image_utils import extract_mask_resized as _get_active_mask_resized
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button

# Patch revision marker — used by the patcher for idempotency. Do not remove.
_SASPRO_CLAHE_UI_REV = 4

# Bit depth of the plane handed to cv2.CLAHE. 16 keeps 32-bit float data out of
# an 8-bit round trip (no posterisation on linear astro data). Set to 8 to
# restore the pre-rev2 behaviour exactly.
DEFAULT_PRECISION = 16

# ----------------------- Core -----------------------
def _clahe_plane(plane01: np.ndarray, clip_limit: float, tile_grid_size: tuple,
                 precision: int) -> np.ndarray:
    """Run cv2 CLAHE on a single 0..1 float plane at 8- or 16-bit precision."""
    c = cv2.createCLAHE(clipLimit=float(clip_limit),
                        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])))
    p = np.clip(np.asarray(plane01, dtype=np.float32), 0.0, 1.0)
    if int(precision) >= 16:
        q = (p * 65535.0 + 0.5).astype(np.uint16)
        return c.apply(q).astype(np.float32) / 65535.0
    q = (p * 255.0 + 0.5).astype(np.uint8)
    return c.apply(q).astype(np.float32) / 255.0


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8),
                precision: int | None = None) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    prec = DEFAULT_PRECISION if precision is None else int(precision)
    prec = 16 if prec >= 16 else 8

    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    was_hw1 = (arr.ndim == 3 and arr.shape[2] == 1)

    if arr.ndim == 3 and arr.shape[2] not in (1, 3):
        raise ValueError(f"CLAHE expects mono or 3-channel RGB, got shape {arr.shape}")

    if arr.ndim == 3 and arr.shape[2] == 3:
        if prec >= 16:
            # float32 LAB keeps a/b untouched and L at full precision
            lab = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0] * (1.0 / 100.0)
            lab[:, :, 0] = _clahe_plane(L, clip_limit, tile_grid_size, 16) * 100.0
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            lab = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        return np.clip(enhanced, 0.0, 1.0)

    mono = arr if arr.ndim == 2 else arr[:, :, 0]
    cl = _clahe_plane(mono, clip_limit, tile_grid_size, prec)
    cl = np.clip(cl, 0.0, 1.0)
    if was_hw1:
        cl = cl[..., None]
    return cl


def tile_grid_from_px(tile_px: int, hw: tuple) -> tuple[int, int]:
    """
    Convert desired tile size (pixels) into OpenCV tileGridSize=(n,n) where n is
    the number of tiles across the *min dimension*.

    n is scale invariant: computing it from the full-resolution shape and reusing
    it on a downsampled proxy keeps the preview faithful to the applied result.
    """
    h, w = int(hw[0]), int(hw[1])
    s = float(min(h, w))
    tile_px = max(8, int(tile_px))
    n = int(round(s / float(tile_px)))
    n = max(2, min(n, 128))
    return (n, n)


def resolve_grid_from_preset(p: dict, hw: tuple) -> tuple[int, int]:
    """Preset -> tileGridSize, honouring legacy tile-count presets unchanged."""
    p = p or {}
    if "tile_px" in p:
        return tile_grid_from_px(int(p.get("tile_px", 128)), hw)
    tile = int(p.get("tile", 8))
    tile = max(2, min(tile, 128))
    return (tile, tile)


def preset_precision(p: dict) -> int:
    try:
        v = int((p or {}).get("precision", DEFAULT_PRECISION))
    except Exception:
        v = DEFAULT_PRECISION
    return 16 if v >= 16 else 8


def preset_blend(p: dict) -> float:
    """0..1 blend, accepting 0..1 or percent-style presets."""
    try:
        blend = float((p or {}).get("blend", 1.0))
    except Exception:
        blend = 1.0
    if blend > 1.0:
        blend = blend / 100.0
    return float(np.clip(blend, 0.0, 1.0))


# ---------------- display-only screen transfer function ----------------
def stf_params(a: np.ndarray, target_bg: float = 0.25, shadow_sigma: float = 2.8):
    """
    Midtone-transfer autostretch parameters (shadow clip, midtone m).

    Returned so the SAME parameters can be applied to both the original and the
    processed frame — otherwise a before/after comparison re-stretches each side
    and the comparison is meaningless.
    """
    v = np.asarray(a, dtype=np.float32).ravel()
    if v.size > 500_000:
        v = v[::(v.size // 500_000) + 1]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 0.5
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if not np.isfinite(mad) or mad <= 0.0:
        mad = 1e-6
    lo = float(np.clip(med - shadow_sigma * mad, 0.0, 0.999))
    x0 = float(np.clip((med - lo) / max(1e-6, 1.0 - lo), 1e-6, 1.0 - 1e-6))
    y0 = float(np.clip(target_bg, 1e-3, 1.0 - 1e-3))
    m = (x0 * (1.0 - y0)) / max(1e-9, (x0 - 2.0 * x0 * y0 + y0))
    return lo, float(np.clip(m, 1e-4, 1.0 - 1e-4))


def stf_apply(a: np.ndarray, lo: float, m: float) -> np.ndarray:
    x = (np.asarray(a, dtype=np.float32) - float(lo)) / max(1e-6, 1.0 - float(lo))
    np.clip(x, 0.0, 1.0, out=x)
    num = (float(m) - 1.0) * x
    den = (2.0 * float(m) - 1.0) * x - float(m)
    out = np.zeros_like(x)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-12)
    return np.clip(out, 0.0, 1.0)


# Note: _get_active_mask_resized imported from setiastro.saspro.widgets.image_utils

def apply_clahe_to_doc(doc, preset: dict | None):
    """
    Apply CLAHE to doc.image using a preset.

    Backward compatible:
      - old presets: {"clip_limit": 2.0, "tile": 8}  # tile count across min dimension
      - new presets: {"clip_limit": 2.0, "tile_px": 128}  # tile size in pixels
      - new: {"blend": 1.0}  # 0..1
      - new: {"precision": 16}  # 8 or 16 bit CLAHE plane
    """
    if doc is None or getattr(doc, "image", None) is None:
        raise RuntimeError("Document has no image.")

    img = np.asarray(doc.image)

    # --- preset decode (supports old + new) ---
    p = preset or {}
    clip = float(p.get("clip_limit", 2.0))
    blend = preset_blend(p)
    prec = preset_precision(p)
    tile_grid = resolve_grid_from_preset(p, img.shape[:2])

    # Normalize into [0..1] first, then run CLAHE on the normalized data.
    # NOTE: the integer branch used to test base.dtype *after* the float32 cast,
    # so it could never fire; it now tests the source dtype. Previously an
    # integer-typed document was handed straight to apply_clahe, which clips to
    # 0..1 and so saturated the whole frame.
    base = img.astype(np.float32, copy=False)
    if img.dtype.kind in "ui":
        base = base / max(1.0, float(np.iinfo(img.dtype).max))
    else:
        base = np.clip(base, 0.0, 1.0)

    out = apply_clahe(base, clip_limit=clip, tile_grid_size=tile_grid, precision=prec)
    out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    # Global blend (before mask)
    if out.ndim == 3:
        if base.ndim == 2:
            base = base[:, :, None].repeat(out.shape[2], axis=2)
        elif base.ndim == 3 and base.shape[2] == 1:
            base = base.repeat(out.shape[2], axis=2)
        out = np.clip(base * (1.0 - blend) + out * blend, 0.0, 1.0)
    else:
        if base.ndim == 3 and base.shape[2] == 1:
            base = base.squeeze(axis=2)
        out = np.clip(base * (1.0 - blend) + out * blend, 0.0, 1.0)

    # Blend with active mask if present
    H, W = out.shape[:2]
    m = _get_active_mask_resized(doc, H, W)
    if m is not None:
        if out.ndim == 3:
            M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
            out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
        else:
            out = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

    # Commit
    out = out.astype(np.float32, copy=False)
    _meta = {
        "step_name": "CLAHE",
        "command_id": "clahe",
        "preset": {
            "clip_limit": float(clip),
            **({"tile_px": int(p.get("tile_px", 128))} if "tile_px" in p
               else {"tile": int(p.get("tile", 8))}),
            "blend": float(blend),
            "precision": int(prec),
        },
    }
    doc.set_image(out, metadata=_meta, step_name="CLAHE")


# ----------------------- preview worker -----------------------
class _PreviewSignals(QObject):
    done = pyqtSignal(int, object, float)
    failed = pyqtSignal(int, str)


class _PreviewJob(QRunnable):
    """One CLAHE preview render. Stale jobs drop out at entry via gen_box."""

    def __init__(self, gen, gen_box, signals, base, clip, grid, blend, precision, mask):
        super().__init__()
        self.setAutoDelete(True)
        self._gen = int(gen)
        self._gen_box = gen_box
        self._sig = signals
        self._base = base
        self._clip = float(clip)
        self._grid = grid
        self._blend = float(blend)
        self._prec = int(precision)
        self._mask = mask

    @pyqtSlot()
    def run(self):
        if self._gen != self._gen_box[0]:
            return
        t0 = time.perf_counter()
        try:
            base = self._base
            out = apply_clahe(base, clip_limit=self._clip, tile_grid_size=self._grid,
                              precision=self._prec)
            if out.ndim == 3 and base.ndim == 2:
                base = base[:, :, None].repeat(out.shape[2], axis=2)
            b = self._blend
            if b < 1.0:
                out = np.clip(base * (1.0 - b) + out * b, 0.0, 1.0)
            m = self._mask
            if m is not None:
                if out.ndim == 3:
                    M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                else:
                    M = m.astype(np.float32)
                out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
            if self._gen != self._gen_box[0]:
                return
            self._sig.done.emit(self._gen, out.astype(np.float32, copy=False),
                                time.perf_counter() - t0)
        except Exception as e:
            self._sig.failed.emit(self._gen, f"{type(e).__name__}: {e}")


# ----------------------- Dialog -----------------------
# The preview is always rendered at full resolution. A downscaled proxy is NOT
# a valid preview of CLAHE: INTER_AREA averages away read noise, which narrows
# every tile histogram, so equalisation stretches far harder. Measured on a
# 4144x2822 mono frame at tile 64 px, a 1:2 proxy shows 2.9x and a 1:4 proxy
# 6.6x the local contrast of the true result. No tile-size correction fixes it.
_PREVIEW_DEBOUNCE_MS = 250

_MODE_AFTER, _MODE_BEFORE, _MODE_SPLIT, _MODE_SBS = 0, 1, 2, 3


def _to_rgb(a: np.ndarray) -> np.ndarray:
    """Any supported layout -> HxWx3 float32 for display."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        return a[..., None].repeat(3, axis=2)
    if a.ndim == 3 and a.shape[2] == 1:
        return a.repeat(3, axis=2)
    return a


class CLAHEDialogPro(QDialog):
    _SKEY = "clahe_pro"

    def __init__(self, parent, doc, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("CLAHE"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        if icon:
            try: self.setWindowIcon(icon)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions

        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        self._disp_base = _to_rgb(self.orig)   # kept for backward compatibility
        self._preview = None

        # preview plumbing
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._gen_box = [0]
        self._sig = _PreviewSignals()          # unparented on purpose: outlives self
        self._sig.done.connect(self._on_preview_ready)
        self._sig.failed.connect(self._on_preview_failed)
        self._mask = None
        self._mask_ready = False
        self._stf = None
        self._last_ms = 0.0
        self._rendering = False
        self._hold_before = False
        self._split_frac = 0.5
        self._split_horizontal = True      # True = vertical line, wipes left/right
        self._drag_split = False
        self._pm_before = None
        self._pm_after = None
        self._pm_sbs = None
        self._refit_pending = False

        root = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(self.splitter, 1)

        self.splitter.addWidget(self._build_controls())
        self.splitter.addWidget(self._build_preview())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setCollapsible(0, True)
        self.splitter.setCollapsible(1, False)

        self._timer = QTimer(self); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._update_preview)
        self._zoom_timer = QTimer(self); self._zoom_timer.setInterval(300)
        self._zoom_timer.timeout.connect(self._refresh_zoom_label)
        self._zoom_timer.start()

        self._restore_settings()
        self._set_pix(self._disp_base)
        self.view.fit_to_item(self.pix)
        self.view.setFocus()
        self._update_preview()

    # ---------------- construction ----------------
    def _build_controls(self) -> QWidget:
        panel = QWidget(self)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox(self.tr("CLAHE Parameters")); grid = QGridLayout(grp)

        self.s_clip = QSlider(Qt.Orientation.Horizontal)
        self.s_clip.setRange(1, 200); self.s_clip.setValue(20)
        self.s_clip.setPageStep(5)
        self.sp_clip = QDoubleSpinBox()
        self.sp_clip.setRange(0.1, 20.0); self.sp_clip.setSingleStep(0.1)
        self.sp_clip.setDecimals(1); self.sp_clip.setValue(2.0)
        self.lbl_clip = QLabel("2.0")   # kept for backward compatibility
        self.lbl_clip.setVisible(False)
        grid.addWidget(QLabel(self.tr("Clip Limit:")), 0, 0)
        grid.addWidget(self.s_clip, 0, 1); grid.addWidget(self.sp_clip, 0, 2)

        self.s_tile = QSlider(Qt.Orientation.Horizontal)
        self.s_tile.setRange(8, 1024); self.s_tile.setSingleStep(8)
        self.s_tile.setPageStep(64); self.s_tile.setValue(128)
        self.s_tile.setToolTip(self.tr("CLAHE tile size in pixels (larger = coarser, smaller = finer)."))
        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(8, 1024); self.sp_tile.setSingleStep(8)
        self.sp_tile.setSuffix(" px"); self.sp_tile.setValue(128)
        self.sp_tile.setToolTip(self.s_tile.toolTip())
        self.lbl_tile = QLabel("128 px")
        self.lbl_tile.setVisible(False)
        grid.addWidget(QLabel(self.tr("Tile Size:")), 1, 0)
        grid.addWidget(self.s_tile, 1, 1); grid.addWidget(self.sp_tile, 1, 2)

        self.s_blend = QSlider(Qt.Orientation.Horizontal)
        self.s_blend.setRange(0, 100); self.s_blend.setValue(100)
        self.sp_blend = QSpinBox()
        self.sp_blend.setRange(0, 100); self.sp_blend.setSuffix(" %")
        self.sp_blend.setValue(100)
        self.lbl_blend = QLabel("100%")
        self.lbl_blend.setVisible(False)
        grid.addWidget(QLabel(self.tr("Blend Amount:")), 2, 0)
        grid.addWidget(self.s_blend, 2, 1); grid.addWidget(self.sp_blend, 2, 2)

        self.chk_16bit = QCheckBox(self.tr("16-bit CLAHE plane"))
        self.chk_16bit.setChecked(DEFAULT_PRECISION >= 16)
        self.chk_16bit.setToolTip(self.tr(
            "Run CLAHE on a 16-bit plane instead of 8-bit.\n"
            "Avoids posterisation on 32-bit float data. Uncheck to reproduce\n"
            "results from SASpro builds before this rework."))
        grid.addWidget(self.chk_16bit, 3, 0, 1, 3)
        self.lbl_grid = QLabel("")
        self.lbl_grid.setEnabled(False)
        grid.addWidget(self.lbl_grid, 4, 0, 1, 3)
        v.addWidget(grp)

        self._link(self.s_clip, self.sp_clip, 10.0)
        self._link(self.s_tile, self.sp_tile, 1.0)
        self._link(self.s_blend, self.sp_blend, 1.0)
        self.chk_16bit.toggled.connect(lambda _on: self._debounce_preview())

        # ---- preview options ----
        pg = QGroupBox(self.tr("Preview")); pgl = QGridLayout(pg)
        self.chk_stretch = QCheckBox(self.tr("Autostretch preview (display only)"))
        self.chk_stretch.setToolTip(self.tr(
            "Screen transfer function so linear data is judgeable.\n"
            "Before and after share one stretch, so the comparison stays honest.\n"
            "Never written to the document."))
        self.chk_stretch.toggled.connect(self._on_stretch_toggled)
        pgl.addWidget(self.chk_stretch, 0, 0, 1, 2)
        v.addWidget(pg)

        # ---- favorites ----
        cg = QGroupBox(self.tr("Favorites")); cgl = QVBoxLayout(cg)
        self.lst_fav = QListWidget()
        self.lst_fav.setToolTip(self.tr("Double-click a favorite to load it back into the sliders."))
        self.lst_fav.setMaximumHeight(120)
        self.lst_fav.itemDoubleClicked.connect(self._recall_favorite)
        cgl.addWidget(self.lst_fav)
        crow = QHBoxLayout()
        self.btn_fav_add = QPushButton(self.tr("Add current"))
        self.btn_fav_del = QPushButton(self.tr("Remove"))
        self.btn_fav_add.clicked.connect(self._remember_favorite)
        self.btn_fav_del.clicked.connect(self._remove_favorite)
        for b in (self.btn_fav_add, self.btn_fav_del):
            b.setAutoDefault(False); crow.addWidget(b)
        cgl.addLayout(crow)
        v.addWidget(cg)

        v.addStretch(1)

        row = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply"));  self.btn_apply.clicked.connect(self._apply)
        self.btn_reset = QPushButton(self.tr("Reset"));  self.btn_reset.clicked.connect(self._reset)
        self.btn_close = QPushButton(self.tr("Cancel")); self.btn_close.clicked.connect(self.reject)
        for b in (self.btn_apply, self.btn_reset, self.btn_close):
            b.setAutoDefault(False); b.setDefault(False)
        row.addStretch(1); row.addWidget(self.btn_apply); row.addWidget(self.btn_reset); row.addWidget(self.btn_close)
        v.addLayout(row)

        # --- preset drag handle (grip), pinned lower-left ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import clahe_path
                _grip_icon = QIcon(clahe_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "clahe", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a CLAHE shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            self.btn_help = make_help_button("clahe", self)
            drag_row.addWidget(self.btn_help)
            v.addLayout(drag_row)
        except Exception:
            pass

        panel.setMinimumWidth(300)
        scroll = QScrollArea(self)
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(460)
        return scroll

    def _build_preview(self) -> QWidget:
        box = QWidget(self)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.grp_mode = QButtonGroup(self)
        self.grp_mode.setExclusive(True)
        for mid, text, tip in (
            (_MODE_AFTER,  self.tr("After"),        self.tr("Show the CLAHE result")),
            (_MODE_BEFORE, self.tr("Before"),       self.tr("Show the untouched image")),
            (_MODE_SPLIT,  self.tr("Split"),        self.tr("One frame, divider you can drag")),
            (_MODE_SBS,    self.tr("Side by side"), self.tr("Before and after, left and right")),
        ):
            b = QToolButton(); b.setText(text); b.setCheckable(True); b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.grp_mode.addButton(b, mid)
            bar.addWidget(b)
        self.grp_mode.button(_MODE_SPLIT).setChecked(True)
        self.grp_mode.idClicked.connect(self._on_mode_changed)

        self.btn_hold = QToolButton()
        self.btn_hold.setText(self.tr("Hold to compare"))
        self.btn_hold.setToolTip(self.tr("Press and hold (or hold Space) to peek at the original"))
        self.btn_hold.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_hold.pressed.connect(lambda: self._set_hold(True))
        self.btn_hold.released.connect(lambda: self._set_hold(False))
        bar.addSpacing(12); bar.addWidget(self.btn_hold)

        self.lbl_split = QLabel(self.tr("Divider:"))
        self.s_split = QSlider(Qt.Orientation.Horizontal)
        self.s_split.setRange(0, 1000); self.s_split.setValue(500)
        self.s_split.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.s_split.setMaximumWidth(160)
        self.s_split.valueChanged.connect(self._on_split_slider)
        self.btn_split_axis = QToolButton()
        self.btn_split_axis.setCheckable(True)
        self.btn_split_axis.setText(self.tr("Vertical"))
        self.btn_split_axis.setToolTip(self.tr("Divider orientation — vertical wipes left/right, "
                                               "horizontal wipes top/bottom"))
        self.btn_split_axis.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_split_axis.toggled.connect(self._on_split_axis)
        bar.addSpacing(12); bar.addWidget(self.lbl_split); bar.addWidget(self.s_split)
        bar.addWidget(self.btn_split_axis)
        bar.addStretch(1)
        v.addLayout(bar)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Base layer. In split mode it holds "before"; the "after" pixmap sits on
        # top inside a clipping rect, so dragging the divider is a setRect() and
        # costs no pixel work at all.
        self.pix = QGraphicsPixmapItem()
        self.scene.addItem(self.pix)

        self._clip = QGraphicsRectItem()
        self._clip.setPen(QPen(Qt.PenStyle.NoPen))
        self._clip.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._clip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self._clip.setZValue(1)
        self.scene.addItem(self._clip)
        self._pix_after = QGraphicsPixmapItem(self._clip)

        # Cosmetic pens (width 0) stay one screen pixel at any zoom; the dark line
        # under the light one keeps the divider visible on bright and dark frames.
        self._divider_bg = QGraphicsLineItem()
        self._divider_bg.setPen(QPen(QColor(0, 0, 0, 190), 3))
        self._divider_bg.setZValue(2)
        self._divider = QGraphicsLineItem()
        self._divider.setPen(QPen(QColor(255, 255, 255, 230), 0))
        self._divider.setZValue(3)
        for it in (self._divider_bg, self._divider):
            self.scene.addItem(it)

        self._grip = QGraphicsEllipseItem(-13.0, -13.0, 26.0, 26.0)
        self._grip.setBrush(QBrush(QColor(255, 255, 255, 235)))
        self._grip.setPen(QPen(QColor(0, 0, 0, 190), 1))
        self._grip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self._grip.setZValue(4)
        self.scene.addItem(self._grip)
        _chev = QGraphicsPathItem(self._grip)
        _path = QPainterPath()
        _path.moveTo(-8.0, 0.0); _path.lineTo(-3.0, -5.0); _path.lineTo(-3.0, 5.0)
        _path.closeSubpath()
        _path.moveTo(8.0, 0.0); _path.lineTo(3.0, -5.0); _path.lineTo(3.0, 5.0)
        _path.closeSubpath()
        _chev.setPath(_path)
        _chev.setBrush(QBrush(QColor(25, 25, 25)))
        _chev.setPen(QPen(Qt.PenStyle.NoPen))
        for _it in (self._clip, self._divider, self._divider_bg, self._grip):
            _it.setVisible(False)

        self.view.viewport().installEventFilter(self)
        v.addWidget(self.view, 1)

        z = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setEnabled(False)
        z.addWidget(self.lbl_status)
        z.addStretch(1)
        self.lbl_zoom = QLabel("")
        z.addWidget(self.lbl_zoom)

        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")
        self.btn_zoom_1x  = QToolButton(); self.btn_zoom_1x.setText("1:1")
        self.btn_zoom_1x.setToolTip(self.tr("One image pixel per screen pixel"))
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_fit, self.btn_zoom_1x):
            try: b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            except Exception: pass

        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        self.btn_zoom_fit.clicked.connect(lambda: self.view.fit_to_item(self.pix))
        self.btn_zoom_1x.clicked.connect(self._zoom_actual)

        z.addWidget(self.btn_zoom_in); z.addWidget(self.btn_zoom_out)
        z.addWidget(self.btn_zoom_fit); z.addWidget(self.btn_zoom_1x)
        v.addLayout(z)

        self._sync_split_visibility()
        return box

    def _link(self, slider: QSlider, spin, scale: float):
        """Two-way slider/spin binding; one preview request per user change."""
        def from_slider(val):
            want = val / scale
            if abs(spin.value() - want) > 1e-9:
                spin.blockSignals(True)
                spin.setValue(want if isinstance(spin, QDoubleSpinBox) else int(round(want)))
                spin.blockSignals(False)
            self._sync_param_labels()
            self._debounce_preview()

        def from_spin(val):
            want = int(round(float(val) * scale))
            if slider.value() != want:
                slider.blockSignals(True)
                slider.setValue(want)
                slider.blockSignals(False)
            self._sync_param_labels()
            self._debounce_preview()

        slider.valueChanged.connect(from_slider)
        spin.valueChanged.connect(from_spin)

    # ---------------- settings ----------------
    def _settings(self) -> QSettings:
        return QSettings()

    def _restore_settings(self):
        try:
            s = self._settings()
            geo = s.value(f"{self._SKEY}/geometry")
            if geo is not None:
                self.restoreGeometry(geo)
            else:
                scr = self.screen() or QApplication.primaryScreen()
                if scr is not None:
                    ag = scr.availableGeometry()
                    self.resize(int(ag.width() * 0.82), int(ag.height() * 0.82))
                    self.move(ag.center().x() - self.width() // 2,
                              ag.center().y() - self.height() // 2)
            st = s.value(f"{self._SKEY}/splitter")
            if st is not None:
                self.splitter.restoreState(st)
            self.chk_stretch.blockSignals(True)
            self.chk_stretch.setChecked(s.value(f"{self._SKEY}/autostretch", False, type=bool))
            self.chk_stretch.blockSignals(False)
            btn = self.grp_mode.button(int(s.value(f"{self._SKEY}/mode", _MODE_SPLIT)))
            if btn is not None:
                btn.setChecked(True)
            self._split_frac = max(0.0, min(1.0, float(s.value(f"{self._SKEY}/split", 0.5))))
            self.s_split.blockSignals(True)
            self.s_split.setValue(int(round(self._split_frac * 1000)))
            self.s_split.blockSignals(False)
            horiz = s.value(f"{self._SKEY}/split_horizontal", True, type=bool)
            self._split_horizontal = bool(horiz)
            self.btn_split_axis.blockSignals(True)
            self.btn_split_axis.setChecked(not self._split_horizontal)
            self.btn_split_axis.setText(self.tr("Vertical") if self._split_horizontal
                                        else self.tr("Horizontal"))
            self.btn_split_axis.blockSignals(False)
        except Exception:
            pass
        self._sync_split_visibility()
        self._load_favorites()
        self._sync_param_labels()

    def _save_settings(self):
        try:
            s = self._settings()
            s.setValue(f"{self._SKEY}/geometry", self.saveGeometry())
            s.setValue(f"{self._SKEY}/splitter", self.splitter.saveState())
            s.setValue(f"{self._SKEY}/autostretch", bool(self.chk_stretch.isChecked()))
            s.setValue(f"{self._SKEY}/mode", int(self.grp_mode.checkedId()))
            s.setValue(f"{self._SKEY}/split", float(self._split_frac))
            s.setValue(f"{self._SKEY}/split_horizontal", bool(self._split_horizontal))
        except Exception:
            pass

    def closeEvent(self, ev):
        self._gen_box[0] += 1          # in-flight renders become stale and self-discard
        try: self._pool.clear()        # drop anything still queued
        except Exception: pass
        self._save_settings()
        try: self._zoom_timer.stop()
        except Exception: pass
        try:
            self._sig.done.disconnect(self._on_preview_ready)
            self._sig.failed.disconnect(self._on_preview_failed)
        except Exception:
            pass
        super().closeEvent(ev)

    # ---------------- favorites ----------------
    def _fav_label(self, p: dict) -> str:
        bits = 16 if int(p.get("precision", 16)) >= 16 else 8
        return (f"clip {float(p.get('clip_limit', 2.0)):.1f}  ·  "
                f"tile {int(p.get('tile_px', 128))} px  ·  "
                f"blend {int(round(float(p.get('blend', 1.0)) * 100))}%  ·  {bits}-bit")

    def _load_favorites(self):
        try:
            s = self._settings()
            raw = s.value(f"{self._SKEY}/favorites", "")
            if not raw:                      # carry over the old key name
                raw = s.value(f"{self._SKEY}/candidates", "")
            items = json.loads(raw) if raw else []
        except Exception:
            items = []
        self.lst_fav.clear()
        for p in items if isinstance(items, list) else []:
            if isinstance(p, dict):
                self._add_favorite_item(p)

    def _add_favorite_item(self, p: dict):
        it = QListWidgetItem(self._fav_label(p))
        it.setData(Qt.ItemDataRole.UserRole, p)
        self.lst_fav.addItem(it)

    def _store_favorites(self):
        out = []
        for i in range(self.lst_fav.count()):
            d = self.lst_fav.item(i).data(Qt.ItemDataRole.UserRole)
            if isinstance(d, dict):
                out.append(d)
        try:
            self._settings().setValue(f"{self._SKEY}/favorites", json.dumps(out[-24:]))
        except Exception:
            pass

    def _remember_favorite(self):
        p = self.get_preset()
        for i in range(self.lst_fav.count()):
            if self.lst_fav.item(i).data(Qt.ItemDataRole.UserRole) == p:
                self.lst_fav.setCurrentRow(i)
                return
        self._add_favorite_item(p)
        self.lst_fav.setCurrentRow(self.lst_fav.count() - 1)
        self._store_favorites()

    def _remove_favorite(self):
        r = self.lst_fav.currentRow()
        if r >= 0:
            self.lst_fav.takeItem(r)
            self._store_favorites()

    def _recall_favorite(self, item: QListWidgetItem):
        d = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(d, dict):
            self.seed_from_preset(d)

    # ---------------- preview scheduling ----------------
    def _debounce_preview(self):
        self._timer.start(_PREVIEW_DEBOUNCE_MS)

    def _full_hw(self) -> tuple[int, int]:
        return (int(self.orig.shape[0]), int(self.orig.shape[1]))

    def _mask_for(self):
        if not self._mask_ready:
            h, w = self._full_hw()
            try:
                self._mask = _get_active_mask_resized(self.doc, h, w)
            except Exception:
                self._mask = None
            self._mask_ready = True
        return self._mask

    def _stf_for(self):
        if self._stf is None:
            self._stf = stf_params(self._disp_base)
        return self._stf

    def _invalidate_caches(self):
        self._mask = None
        self._mask_ready = False
        self._stf = None
        self._last_sig = None
        self._pm_before = self._pm_after = self._pm_sbs = None
        self._refit_pending = False

    def _update_preview(self):
        """Render the preview. Also the entry point for external callers."""
        self._request_preview()

    def _request_preview(self):
        clip = float(self.sp_clip.value())
        tile_px = int(self.sp_tile.value())
        blend = float(self.sp_blend.value()) / 100.0
        prec = 16 if self.chk_16bit.isChecked() else 8

        sig = (clip, tile_px, blend, prec, id(self.doc))
        if sig == getattr(self, "_last_sig", None) and self._preview is not None:
            self._update_status()
            return
        self._last_sig = sig

        grid = tile_grid_from_px(tile_px, self._full_hw())
        self._gen_box[0] += 1          # any render still in flight becomes stale
        gen = self._gen_box[0]
        self._rendering = True
        self._update_status()
        job = _PreviewJob(gen, self._gen_box, self._sig, self.orig, clip, grid, blend,
                          prec, self._mask_for())
        self._pool.start(job)

    @pyqtSlot(int, object, float)
    def _on_preview_ready(self, gen: int, out, secs: float):
        if gen != self._gen_box[0]:
            return
        self._preview = out
        self._rendering = False
        self._last_ms = float(secs) * 1000.0
        self._rebuild_pixmaps()
        self._compose_and_show()
        if getattr(self, "_refit_pending", False):
            self._refit_pending = False
            try: self.view.fit_to_item(self.pix)
            except Exception: pass
        self._update_status()

    @pyqtSlot(int, str)
    def _on_preview_failed(self, gen: int, msg: str):
        if gen != self._gen_box[0]:
            return
        self._rendering = False
        self.lbl_status.setText(self.tr("preview failed: ") + msg)

    # ---------------- composition ----------------
    def _mode(self) -> int:
        if self._hold_before:
            return _MODE_BEFORE
        return int(self.grp_mode.checkedId())

    def _view_scale(self) -> float:
        try:
            return float(self.view.transform().m11()) or 1.0
        except Exception:
            return 1.0

    def _pixmap_from(self, rgb) -> QPixmap:
        arr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q)      # fromImage copies, so arr can go

    def _rebuild_pixmaps(self):
        """Convert the current before/after pair once.

        Mode switches, hold-to-compare and divider drags then touch no pixels.
        """
        if self._preview is None:
            self._pm_before = self._pm_after = self._pm_sbs = None
            return
        before = self._disp_base
        after = _to_rgb(self._preview)
        if after.shape[:2] != before.shape[:2]:
            return
        if self.chk_stretch.isChecked():
            lo, m = self._stf_for()
            before = stf_apply(before, lo, m)
            after = stf_apply(after, lo, m)
        self._pm_before = self._pixmap_from(before)
        self._pm_after = self._pixmap_from(after)
        self._pm_sbs = None
        self._pix_after.setPixmap(self._pm_after)

    def _set_pix(self, rgb):
        """Set the base layer straight from an array (used before the first render)."""
        self.pix.setPixmap(self._pixmap_from(rgb))
        self.pix.setTransform(QTransform())
        self.scene.setSceneRect(self.pix.sceneBoundingRect())

    def _apply_layer_transform(self):
        """Keep the scene in image coordinates and pick a sane sampling mode.

        Pixmaps are always the full-resolution result, so the item transform is
        identity and the view transform means the same thing at every moment.
        Sampling follows the zoom: smooth while zoomed out (star fields alias
        badly under nearest neighbour), nearest at 100% and above so you are
        looking at real pixels.
        """
        if self._pm_before is None:
            return
        smooth = self._view_scale() < 0.999
        for it in (self.pix, self._pix_after):
            it.setTransform(QTransform())
            it.setTransformationMode(Qt.TransformationMode.SmoothTransformation if smooth
                                     else Qt.TransformationMode.FastTransformation)

    def _compose_and_show(self):
        if self._pm_before is None or self._pm_after is None:
            return
        mode = self._mode()
        split = (mode == _MODE_SPLIT)
        if mode == _MODE_SBS:
            if self._pm_sbs is None and self._preview is not None:
                before = self._disp_base
                after = _to_rgb(self._preview)
                if self.chk_stretch.isChecked():
                    lo, m = self._stf_for()
                    before = stf_apply(before, lo, m)
                    after = stf_apply(after, lo, m)
                gap = np.zeros((before.shape[0], 4, 3), dtype=np.float32)
                self._pm_sbs = self._pixmap_from(np.concatenate([before, gap, after], axis=1))
            if self._pm_sbs is not None:
                self.pix.setPixmap(self._pm_sbs)
        elif mode == _MODE_AFTER:
            self.pix.setPixmap(self._pm_after)
        else:
            self.pix.setPixmap(self._pm_before)

        self._apply_layer_transform()
        for it in (self._clip, self._divider, self._divider_bg, self._grip):
            it.setVisible(split)
        rect = self.pix.sceneBoundingRect()
        if rect != self.scene.sceneRect():
            self.scene.setSceneRect(rect)
        if split:
            self._update_divider()

    def _update_divider(self):
        """Geometry only — no pixel work, so dragging stays smooth at any size."""
        if self._pm_before is None:
            return
        ih, iw = self._full_hw()
        w, h = float(iw), float(ih)
        f = max(0.0, min(1.0, float(self._split_frac)))
        if self._split_horizontal:
            x = f * w
            self._clip.setRect(x, 0.0, max(0.0, w - x), h)
            self._divider.setLine(x, 0.0, x, h)
            self._divider_bg.setLine(x, 0.0, x, h)
            self._grip.setPos(x, h * 0.5)
            self._grip.setRotation(0.0)
        else:
            y = f * h
            self._clip.setRect(0.0, y, w, max(0.0, h - y))
            self._divider.setLine(0.0, y, w, y)
            self._divider_bg.setLine(0.0, y, w, y)
            self._grip.setPos(w * 0.5, y)
            self._grip.setRotation(90.0)

    def _update_status(self):
        h, w = self._full_hw()
        bits = 16 if self.chk_16bit.isChecked() else 8
        tail = ("  ·  " + self.tr("rendering…")) if self._rendering else ""
        self.lbl_status.setText(
            f"{w} × {h}  ·  {self._last_ms:.0f} ms  ·  {bits}-bit{tail}")
        self.lbl_status.setToolTip(self.tr(
            "The preview is always the full-resolution result, never a downscaled\n"
            "approximation, so what you see is what Apply will commit."))

    def _sync_param_labels(self):
        try:
            self.lbl_clip.setText(f"{self.sp_clip.value():.1f}")
            self.lbl_tile.setText(f"{int(self.sp_tile.value())} px")
            self.lbl_blend.setText(f"{int(self.sp_blend.value())}%")
            n, _ = tile_grid_from_px(int(self.sp_tile.value()), self._full_hw())
            self.lbl_grid.setText(self.tr("tile grid: ") + f"{n} × {n}")
        except Exception:
            pass

    def _refresh_zoom_label(self):
        try:
            self.lbl_zoom.setText(f"{self._view_scale() * 100.0:.0f}%")
            want = (Qt.TransformationMode.SmoothTransformation if self._view_scale() < 0.999
                    else Qt.TransformationMode.FastTransformation)
            if self.pix.transformationMode() != want:
                for it in (self.pix, self._pix_after):
                    it.setTransformationMode(want)
        except Exception:
            pass

    def _zoom_actual(self):
        # The scene is in image coordinates, so identity == one image pixel per
        # screen pixel whatever the preview is currently rendered at.
        try:
            self.view.resetTransform()
            self._refresh_zoom_label()
        except Exception:
            pass

    # ---------------- comparison controls ----------------
    def _on_mode_changed(self, _mid: int):
        self._sync_split_visibility()
        self._compose_and_show()

    def _sync_split_visibility(self):
        on = (int(self.grp_mode.checkedId()) == _MODE_SPLIT)
        self.lbl_split.setVisible(on)
        self.s_split.setVisible(on)
        self.btn_split_axis.setVisible(on)

    def _on_split_slider(self, val: int):
        self._split_frac = float(val) / 1000.0
        if int(self.grp_mode.checkedId()) == _MODE_SPLIT:
            self._update_divider()

    def _on_split_axis(self, horizontal_line: bool):
        self._split_horizontal = not bool(horizontal_line)
        self.btn_split_axis.setText(self.tr("Vertical") if self._split_horizontal
                                    else self.tr("Horizontal"))
        if int(self.grp_mode.checkedId()) == _MODE_SPLIT:
            self._update_divider()

    def _on_stretch_toggled(self, _on: bool):
        self._rebuild_pixmaps()
        self._compose_and_show()

    def _set_hold(self, on: bool):
        on = bool(on)
        if on == self._hold_before:
            return
        self._hold_before = on
        self._compose_and_show()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._set_hold(True); ev.accept(); return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space and not ev.isAutoRepeat():
            self._set_hold(False); ev.accept(); return
        super().keyReleaseEvent(ev)

    def eventFilter(self, obj, ev):
        """Drag the split divider directly in the preview.

        Runs on the viewport before ZoomableGraphicsView sees the event, so the
        view's pan handling is untouched unless the cursor is on the divider.
        """
        try:
            if obj is self.view.viewport() and self._pm_before is not None \
                    and int(self.grp_mode.checkedId()) == _MODE_SPLIT:
                et = ev.type()
                if et in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove,
                          QEvent.Type.MouseButtonRelease):
                    ih, iw = self._full_hw()
                    w, h = float(iw), float(ih)
                    horiz = self._split_horizontal
                    anchor = (QPointF(self._split_frac * w, h * 0.5) if horiz
                              else QPointF(w * 0.5, self._split_frac * h))
                    va = self.view.mapFromScene(anchor)
                    pos = ev.position()
                    dx, dy = pos.x() - va.x(), pos.y() - va.y()
                    d_line = abs(dx) if horiz else abs(dy)
                    near = (d_line <= 10.0) or ((dx * dx + dy * dy) ** 0.5 <= 18.0)
                    if et == QEvent.Type.MouseButtonPress:
                        if ev.button() == Qt.MouseButton.LeftButton and near:
                            self._drag_split = True
                            return True
                    elif et == QEvent.Type.MouseMove:
                        if self._drag_split:
                            sp = self.view.mapToScene(pos.toPoint())
                            f = (sp.x() / w) if horiz else (sp.y() / h)
                            f = max(0.0, min(1.0, float(f)))
                            self._split_frac = f
                            self.s_split.blockSignals(True)
                            self.s_split.setValue(int(round(f * 1000)))
                            self.s_split.blockSignals(False)
                            self._update_divider()
                            return True
                        self.view.viewport().setCursor(QCursor(
                            (Qt.CursorShape.SplitHCursor if horiz
                             else Qt.CursorShape.SplitVCursor)
                            if near else Qt.CursorShape.ArrowCursor))
                    elif et == QEvent.Type.MouseButtonRelease and self._drag_split:
                        self._drag_split = False
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------------- apply ----------------
    def _apply(self):
        try:
            clip = float(self.sp_clip.value())
            tile_px = int(self.sp_tile.value())
            blend = float(self.sp_blend.value()) / 100.0
            prec = 16 if self.chk_16bit.isChecked() else 8

            # --- CLAHE params ---
            tile_grid = tile_grid_from_px(tile_px, self.orig.shape[:2])

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                # --- Run CLAHE on the original (normalized) image ---
                out = apply_clahe(self.orig, clip_limit=clip, tile_grid_size=tile_grid,
                                  precision=prec)
                out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

                # --- Build base from CURRENT doc state (normalized to 0..1) ---
                src = np.asarray(self.doc.image)
                base = src.astype(np.float32, copy=False)
                if src.dtype.kind in "ui":
                    base = base / max(1.0, float(np.iinfo(src.dtype).max))
                else:
                    base = np.clip(base, 0.0, 1.0)

                # --- Align base channels to out channels ---
                if out.ndim == 3:
                    if base.ndim == 2:
                        base = base[:, :, None].repeat(out.shape[2], axis=2)
                    elif base.ndim == 3 and base.shape[2] == 1:
                        base = base.repeat(out.shape[2], axis=2)
                else:
                    if base.ndim == 3 and base.shape[2] == 1:
                        base = base.squeeze(axis=2)

                # --- Global blend ALWAYS (even with no mask) ---
                out = np.clip(base * (1.0 - blend) + out * blend, 0.0, 1.0)

                # --- Apply active mask gating (if present) ---
                H, W = out.shape[:2]
                m = _get_active_mask_resized(self.doc, H, W)
                if m is not None:
                    if out.ndim == 3:
                        M = np.repeat(m[:, :, None], out.shape[2], axis=2).astype(np.float32)
                        out = np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
                    else:
                        out = np.clip(base * (1.0 - m) + out * m, 0.0, 1.0)

                out = out.astype(np.float32, copy=False)
            finally:
                QApplication.restoreOverrideCursor()

            # --- Commit to document ---
            preset = {
                "clip_limit": float(clip),
                "tile_px": int(tile_px),
                "blend": float(blend),
                "precision": int(prec),
            }
            _meta = {"step_name": "CLAHE", "command_id": "clahe", "preset": dict(preset)}
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(out, metadata=_meta, step_name="CLAHE")
            else:
                self.doc.image = out

            # ── Register as last_headless_command for replay ─────────────
            try:
                main = self.parent()
                while main is not None and not hasattr(main, "_remember_last_headless_command"):
                    main = main.parent()
                if main is not None:
                    main._remember_last_headless_command(
                        "clahe", dict(preset),
                        description=f"CLAHE (clip={clip:.1f}, tile={tile_px}px, blend={blend:.2f})"
                    )
            except Exception:
                pass
            # ─────────────────────────────────────────────────────────────

            # Dialog stays open so user can apply to other images
            self._refresh_document_from_active()
            try: self.view.setFocus()
            except Exception: pass

        except Exception as e:
            try: QApplication.restoreOverrideCursor()
            except Exception: pass
            QMessageBox.critical(self, "CLAHE", f"Failed to apply:\n{e}")

    def _rebind_doc(self, doc):
        """Point the dialog at doc and rebuild every preview cache."""
        prev_hw = self._full_hw()
        self.doc = doc
        self.orig = np.clip(np.asarray(doc.image, dtype=np.float32), 0.0, 1.0)
        self._disp_base = _to_rgb(self.orig)
        self._preview = None
        self._invalidate_caches()
        self._sync_param_labels()
        # Only a genuine change of image size justifies moving the user's view.
        self._refit_pending = (self._full_hw() != prev_hw)
        self._update_preview()

    def _refresh_document_from_active(self):
        """
        Re-point the dialog at the currently active document, so the same dialog
        can be reused on other images. Also re-reads the source image from the
        document we just committed to, so a second Apply stacks on the committed
        result instead of re-running CLAHE on the stale original.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self._rebind_doc(new_doc)
                    return
        except Exception:
            pass
        try:
            self._rebind_doc(self.doc)
        except Exception:
            pass

    def _tile_grid_from_px(self, tile_px: int, hw: tuple[int, int]) -> tuple[int, int]:
        """Kept for callers outside this dialog; see module-level tile_grid_from_px."""
        return tile_grid_from_px(tile_px, hw)

    # -------- preset emit (grip) --------
    def get_preset(self) -> dict:
        """Emit current slider state as the CLAHE preset schema (inverse of seed_from_preset).
        Mirrors exactly what _apply records for Replay Last."""
        return {
            "clip_limit": float(self.sp_clip.value()),
            "tile_px": int(self.sp_tile.value()),
            "blend": float(self.sp_blend.value()) / 100.0,
            "precision": 16 if self.chk_16bit.isChecked() else 8,
        }

    # -------- preset seed (double-click open) --------
    def seed_from_preset(self, p: dict | None):
        """Inverse of get_preset. Supports legacy 'tile' (tile-count) presets and
        both 0..1 and 0..100 blend conventions. Block signals around sets, then
        fire one preview by hand."""
        p = dict(p or {})

        try:
            clip = float(p.get("clip_limit", 2.0))
            clip = max(self.sp_clip.minimum(), min(self.sp_clip.maximum(), clip))
        except Exception:
            clip = float(self.sp_clip.value())

        try:
            if "tile_px" in p:
                tp = int(p.get("tile_px", 128))
            else:
                legacy = max(2, min(int(p.get("tile", 8)), 128))
                tp = int(round(2048 / float(legacy)))
            tp = max(self.sp_tile.minimum(), min(self.sp_tile.maximum(), tp))
        except Exception:
            tp = int(self.sp_tile.value())

        try:
            bi = int(round(preset_blend(p) * 100.0))
            bi = max(self.sp_blend.minimum(), min(self.sp_blend.maximum(), bi))
        except Exception:
            bi = int(self.sp_blend.value())

        prec16 = preset_precision(p) >= 16

        widgets = (self.s_clip, self.s_tile, self.s_blend,
                   self.sp_clip, self.sp_tile, self.sp_blend, self.chk_16bit)
        for w in widgets:
            try: w.blockSignals(True)
            except Exception: pass
        try:
            self.s_clip.setValue(int(round(clip * 10.0)))
            self.s_tile.setValue(int(tp))
            self.s_blend.setValue(int(bi))
            self.sp_clip.setValue(float(clip))
            self.sp_tile.setValue(int(tp))
            self.sp_blend.setValue(int(bi))
            self.chk_16bit.setChecked(bool(prec16))
        finally:
            for w in widgets:
                try: w.blockSignals(False)
                except Exception: pass

        self._sync_param_labels()
        try:
            self._update_preview()
        except Exception:
            pass

    def _reset(self):
        self.seed_from_preset({"clip_limit": 2.0, "tile_px": 128, "blend": 1.0,
                               "precision": DEFAULT_PRECISION})
        self.view.fit_to_item(self.pix)
        try: self.view.setFocus()
        except Exception: pass


def open_clahe_with_preset(main_window, preset: dict | None = None):
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
        return None

    _icon = QIcon()
    try:
        from setiastro.saspro.resources import clahe_path
        _icon = QIcon(clahe_path)
    except Exception:
        pass

    # Constructor takes a QIcon (not a path).
    dlg = CLAHEDialogPro(main_window, doc, icon=_icon)

    # No on-show reset -> seed directly before show.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    try:
        main_window._clahe_dialog = dlg   # retain against GC (WA_DeleteOnClose)
        dlg.destroyed.connect(
            lambda *_: setattr(main_window, "_clahe_dialog", None))
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg