# src/setiastro/saspro/cosmetic_correction.py
"""
Standalone Cosmetic Correction tool.

Detects and repairs hot / cold pixels via a two-pass 5-neighbor median
comparison (with an anti-star-core guard).  Backed by
setiastro.saspro.torch_rejection.cosmetic_correction_gpu, which
transparently falls back to CPU when CUDA/MPS is unavailable.

Standard tool shape (mirrors remove_green.py / stat_stretch.py):
  - CosmeticCorrectionDialog: interactive UI with a side-by-side preview
    that always uses autostretch so the user can see faint noise / stars
    at the same relative brightness regardless of the underlying data.
  - CosmeticCorrectionBatchDialog: batch processor — pick a list of files,
    pick an output directory (or overwrite in place), run cosmetic
    correction on each and save back out.  Reuses the same detection
    knobs as the interactive dialog.
  - cosmetic_correction_headless(doc, ...): the actual apply-to-doc
    path used by both the Apply button and by preset drops.
  - open_cosmetic_correction_dialog / apply_cosmetic_correction_preset_to_doc /
    open_cosmetic_correction_with_preset: entry points wired from main.
"""
from __future__ import annotations
import os
import platform
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox,
    QPushButton, QCheckBox, QMessageBox, QGroupBox, QFormLayout,
    QDoubleSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSplitter, QSizePolicy, QListWidget, QListWidgetItem, QFileDialog,
    QProgressBar, QAbstractItemView, QRadioButton, QButtonGroup, QLineEdit,
    QWidget, QGridLayout, QToolButton,
)

# ---------- shared helpers ----------
from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc,
)
try:
    from setiastro.saspro.imageops.stretch import (
        stretch_color_image, stretch_mono_image,
    )
except Exception:
    stretch_color_image = None
    stretch_mono_image = None

try:
    import cv2  # only used for mask resize
except Exception:
    cv2 = None


# =====================================================================
# Cosmetic correction wrappers
# =====================================================================
def _run_cosmetic(
    image: np.ndarray,
    *,
    hot_sigma: float,
    cold_sigma: float,
    bayer_pattern: str | None = None,
) -> np.ndarray:
    """Route to the GPU/CPU cosmetic corrector.  Accepts:
       - (H, W) mono
       - (H, W, 3) RGB
       - (H, W, C) with C > 3 (only the first 3 channels get corrected;
         extras pass through)
    Returns an array with the same shape and dtype float32.
    """
    from setiastro.saspro.torch_rejection import cosmetic_correction_gpu

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return cosmetic_correction_gpu(
            arr, hot_sigma=float(hot_sigma), cold_sigma=float(cold_sigma),
            bayer_pattern=(bayer_pattern or None),
        )

    if arr.ndim == 3:
        # If Bayer is specified, the input MUST be 2-D mosaic — silently
        # ignore the flag for already-debayered arrays.
        bp = None if arr.shape[2] > 1 else (bayer_pattern or None)
        if arr.shape[2] <= 3:
            return cosmetic_correction_gpu(
                arr, hot_sigma=float(hot_sigma),
                cold_sigma=float(cold_sigma), bayer_pattern=bp,
            )
        # extra channels (e.g. RGBA) — process the first 3, pass rest through
        out = arr.copy()
        out[..., :3] = cosmetic_correction_gpu(
            arr[..., :3], hot_sigma=float(hot_sigma),
            cold_sigma=float(cold_sigma), bayer_pattern=bp,
        )
        return out

    raise ValueError(f"Unsupported image shape for cosmetic correction: {arr.shape}")


def _bayer_from_doc(doc) -> str | None:
    """Extract a Bayer pattern hint from doc metadata / FITS header.  Only
    returns a value when the image is a raw 2-D mosaic (i.e. hasn't been
    debayered yet) — otherwise applying Bayer stride-2 to an RGB frame
    would be wrong.
    """
    if doc is None:
        return None
    try:
        img = np.asarray(getattr(doc, "image", None))
        if img.ndim != 2:
            return None
    except Exception:
        return None

    for src_name in ("metadata", "meta", "header"):
        m = getattr(doc, src_name, None)
        if not m:
            continue
        try:
            for key in ("BAYERPAT", "bayerpat", "bayer_pattern", "CFA"):
                if hasattr(m, "get"):
                    v = m.get(key)
                else:
                    v = getattr(m, key, None)
                if v:
                    s = str(v).strip().upper()
                    if s in ("RGGB", "BGGR", "GRBG", "GBRG"):
                        return s
        except Exception:
            continue
    return None


# =====================================================================
# Headless entry point
# =====================================================================
def cosmetic_correction_headless(
    doc,
    *,
    hot_sigma: float = 3.0,
    cold_sigma: float = 3.0,
    bayer_pattern: str | None = None,
    correct_hot: bool = True,
    correct_cold: bool = True,
    use_hw_accel: bool = True,   # informational; corrector picks internally
):
    """Run cosmetic correction on doc.image and push as an undoable edit.

    correct_hot/cold=False effectively disables that pass by setting the
    sigma threshold to a very large value (the underlying kernel doesn't
    have a per-pass switch, so we make the threshold unreachable).
    """
    if doc is None or getattr(doc, "image", None) is None:
        return

    src = np.asarray(doc.image)
    src_f = _to_float01(src).astype(np.float32, copy=False)

    hs = float(hot_sigma) if bool(correct_hot) else 1e9
    cs = float(cold_sigma) if bool(correct_cold) else 1e9

    # Auto-detect Bayer if the caller didn't specify AND the image is 2-D
    bp = (bayer_pattern or "").strip().upper() or None
    if bp not in ("RGGB", "BGGR", "GRBG", "GBRG"):
        bp = None
    if bp is None and src_f.ndim == 2:
        bp = _bayer_from_doc(doc)

    processed = _run_cosmetic(
        src_f, hot_sigma=hs, cold_sigma=cs, bayer_pattern=bp,
    )

    # Preserve alpha / extra channels if the source had them
    if src_f.ndim == 3 and src_f.shape[2] > 3 and processed.shape == src_f.shape[:2] + (3,):
        out = src_f.astype(np.float32, copy=True)
        out[..., :3] = processed
    else:
        out = processed

    # Mask-aware blend (same pattern as remove_green.py)
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        h, w = out.shape[:2]
        if m.shape != (h, w):
            if cv2 is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                yi = np.linspace(0, m.shape[0] - 1, h).astype(np.int32)
                xi = np.linspace(0, m.shape[1] - 1, w).astype(np.int32)
                m = m[yi][:, xi]
        if out.ndim == 3:
            m3 = np.repeat(m[:, :, None], out.shape[2], axis=2)
        else:
            m3 = m
        out = src_f * (1.0 - m3) + out * m3

    out = np.clip(out.astype(np.float32, copy=False), 0.0, 1.0)

    step_label = "Cosmetic Correction"
    passes = []
    if correct_hot:  passes.append("hot")
    if correct_cold: passes.append("cold")
    meta = {
        "step_name": step_label,
        "cosmetic_correction": {
            "hot_sigma": float(hot_sigma),
            "cold_sigma": float(cold_sigma),
            "correct_hot": bool(correct_hot),
            "correct_cold": bool(correct_cold),
            "bayer_pattern": bp,
            "passes": passes,
        },
        "bit_depth": "32-bit floating point",
        "is_mono": (out.ndim == 2),
    }
    doc.apply_edit(out, metadata=meta, step_name=step_label)


# =====================================================================
# Preview worker
# =====================================================================
class _CosmeticPreviewWorker(QThread):
    """Runs cosmetic correction on a tile so the preview stays responsive."""
    finished_ok = pyqtSignal(object)     # np.ndarray (corrected tile)
    failed      = pyqtSignal(str)

    def __init__(
        self,
        tile: np.ndarray,
        *,
        hot_sigma: float,
        cold_sigma: float,
        bayer_pattern: str | None,
        parent=None,
    ):
        super().__init__(parent)
        self._tile = tile.astype(np.float32, copy=True)
        self._hot = float(hot_sigma)
        self._cold = float(cold_sigma)
        self._bp = bayer_pattern

    def run(self):
        try:
            out = _run_cosmetic(
                self._tile,
                hot_sigma=self._hot,
                cold_sigma=self._cold,
                bayer_pattern=self._bp,
            )
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))


# =====================================================================
# Preview widget: side-by-side before/after, always autostretched
# =====================================================================
def _to_hwc3_float01(arr: np.ndarray) -> np.ndarray:
    """Normalize input to (H, W, 3) float32 in [0, 1] — matches the
    display-buffer convention used by blemish_blaster.  Mono inputs
    are broadcast across all three channels so downstream stretch code
    can treat everything as RGB."""
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        maxv = float(np.nanmax(a)) or 1.0
        a = a.astype(np.float32) / max(1.0, maxv)
    else:
        a = a.astype(np.float32, copy=False)
    a = np.clip(a, 0.0, 1.0)

    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[:, :, :3].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported preview shape: {a.shape}")


def _to_display_rgb(arr: np.ndarray, is_mono_source: bool,
                    target_median: float = 0.25) -> np.ndarray:
    """Autostretch to 8-bit RGB for on-screen display.  Matches
    blemish_blaster._update_display_autostretch: always work on an
    (H, W, 3) buffer, dispatch to stretch_mono_image (for originally
    mono sources) or stretch_color_image (for true colour)."""
    src3 = _to_hwc3_float01(arr)

    if stretch_color_image is not None and stretch_mono_image is not None:
        try:
            if is_mono_source:
                mono = src3[..., 0]  # all channels identical, pick one
                mono_st = stretch_mono_image(
                    mono, target_median=target_median,
                    normalize=False, apply_curves=False,
                )
                disp = np.stack([mono_st] * 3, axis=-1)
            else:
                disp = stretch_color_image(
                    src3, target_median=target_median, linked=False,
                    normalize=False, apply_curves=False,
                )
            disp = np.clip(disp.astype(np.float32, copy=False), 0.0, 1.0)
            return (disp * 255.0 + 0.5).astype(np.uint8)
        except Exception:
            pass

    # Fallback if the shared stretch module isn't importable — simple
    # per-channel gamma to land the median at target_median.
    outp = []
    channels = [src3[..., 0]] if is_mono_source else [src3[..., c] for c in range(3)]
    for p in channels:
        med = float(np.median(p))
        if med <= 1e-6:
            outp.append(np.clip(p, 0.0, 1.0))
            continue
        gamma = np.log(target_median) / np.log(med + 1e-9)
        outp.append(np.clip(np.power(p, gamma, dtype=np.float32), 0.0, 1.0))
    if is_mono_source:
        outp = [outp[0], outp[0], outp[0]]
    s3 = np.stack(outp[:3], axis=-1)
    return (s3 * 255.0 + 0.5).astype(np.uint8)


class _PreviewPane(QGraphicsView):
    """Zoom/pan view holding a single QPixmap.

    Two behaviours worth calling out:

    1. **Initial fit is deferred until the widget has a real size.**
       Calling fitInView() before Qt has laid the widget out means we
       fit to the placeholder size (~640×480), then the widget grows to
       fill the splitter and the picture ends up tiny in the middle of
       the view. We defer the fit until the first paint/resize event
       when the widget actually has its final size.

    2. **View state is preserved across image replacements.** When a
       preview refresh replaces the pixmap (e.g. the user nudges a
       slider), we keep the current transform + scrollbar positions
       so the user isn't yanked back to fit-to-view every time.

    3. **Panes can be linked** so panning/zooming one drives its peer.
       set_peer(other) establishes the two-way link.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Zoom around the point under the cursor (feels natural).
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._zoom = 1.0
        self._did_initial_fit = False
        self._peer: "_PreviewPane | None" = None
        self._syncing = False  # re-entry guard for the peer link

    # ------------------------------------------------------------------
    # Peer linking
    # ------------------------------------------------------------------
    def set_peer(self, other: "_PreviewPane | None"):
        """Link this pane's pan+zoom to `other` (two-way).  Pass None to unlink."""
        self._peer = other
        if other is None:
            return
        # Make the peer reference symmetric.
        other._peer = self
        # Wire scrollbars on BOTH panes so pan works in either direction.
        # The _syncing guard on each pane prevents ping-pong.
        self.horizontalScrollBar().valueChanged.connect(self._sync_h_to_peer)
        self.verticalScrollBar().valueChanged.connect(self._sync_v_to_peer)
        other.horizontalScrollBar().valueChanged.connect(other._sync_h_to_peer)
        other.verticalScrollBar().valueChanged.connect(other._sync_v_to_peer)

    def _sync_h_to_peer(self, value: int):
        if self._syncing or self._peer is None:
            return
        try:
            self._peer._syncing = True
            self._peer.horizontalScrollBar().setValue(value)
        finally:
            self._peer._syncing = False

    def _sync_v_to_peer(self, value: int):
        if self._syncing or self._peer is None:
            return
        try:
            self._peer._syncing = True
            self._peer.verticalScrollBar().setValue(value)
        finally:
            self._peer._syncing = False

    def _apply_zoom_to_peer(self, factor: float):
        """Called after a wheel zoom to keep the peer in sync (and using
        the same anchor point so the two views stay pixel-aligned)."""
        if self._syncing or self._peer is None:
            return
        try:
            self._peer._syncing = True
            self._peer.scale(factor, factor)
            self._peer._zoom = self._zoom
        finally:
            self._peer._syncing = False

    # ------------------------------------------------------------------
    # Image replacement — PRESERVE view state
    # ------------------------------------------------------------------
    def set_image(self, rgb_u8: np.ndarray):
        h, w = rgb_u8.shape[:2]
        rgb_u8 = np.ascontiguousarray(rgb_u8)
        qimg = QImage(rgb_u8.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pm = QPixmap.fromImage(qimg)

        first_time = (self._item is None)

        if first_time:
            self._item = self._scene.addPixmap(pm)
            self._scene.setSceneRect(0, 0, w, h)
            # Defer the initial fit until we have a real viewport size —
            # see resizeEvent / showEvent below.
            self._did_initial_fit = False
            return

        # Subsequent updates: keep the exact transform + scrollbar values.
        # Capturing them isn't strictly required because setPixmap doesn't
        # touch them, but scene rect changes CAN reset the scrollbars, so
        # we play it safe.
        h_before = self.horizontalScrollBar().value()
        v_before = self.verticalScrollBar().value()

        self._item.setPixmap(pm)
        # setSceneRect only if the image dimensions actually change
        if (self._scene.sceneRect().width() != w
                or self._scene.sceneRect().height() != h):
            self._scene.setSceneRect(0, 0, w, h)

        self.horizontalScrollBar().setValue(h_before)
        self.verticalScrollBar().setValue(v_before)

    # ------------------------------------------------------------------
    # Deferred initial fit
    # ------------------------------------------------------------------
    def _maybe_initial_fit(self):
        """Fit the image to view exactly once, after Qt has given the widget
        its real size.  Everything after that is up to the user."""
        if self._did_initial_fit or self._item is None:
            return
        vp = self.viewport().size()
        if vp.width() < 20 or vp.height() < 20:
            return  # still laying out — try again on the next resize
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        # Capture the effective zoom so later scale() calls stay in scale.
        # fitInView applies a uniform scale; grab it from the transform matrix.
        try:
            self._zoom = float(self.transform().m11()) or 1.0
        except Exception:
            self._zoom = 1.0
        self._did_initial_fit = True
        # Mirror the fit into the peer so both start pixel-aligned
        if self._peer is not None and not self._peer._did_initial_fit:
            self._peer.resetTransform()
            self._peer.fitInView(self._peer._item, Qt.AspectRatioMode.KeepAspectRatio) \
                if self._peer._item is not None else None
            self._peer._zoom = self._zoom
            self._peer._did_initial_fit = True

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Try the initial fit here — this is where we first get a real size.
        if not self._did_initial_fit:
            self._maybe_initial_fit()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._did_initial_fit:
            # Defer to next event loop cycle so the parent's layout has run.
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self._maybe_initial_fit)

    def fit_to_view(self):
        """Public: forget any user-adjusted view state and re-fit."""
        self._did_initial_fit = False
        self._maybe_initial_fit()

    # ------------------------------------------------------------------
    # Wheel zoom — mirror to peer
    # ------------------------------------------------------------------
    def wheelEvent(self, ev):
        if self._item is None:
            return
        delta = ev.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = self._zoom * factor
        if 0.05 <= new_zoom <= 40.0:
            self._zoom = new_zoom
            self.scale(factor, factor)
            self._apply_zoom_to_peer(factor)


# =====================================================================
# Main dialog
# =====================================================================
class CosmeticCorrectionDialog(QDialog):
    """Interactive cosmetic-correction tool.

    Shows a before/after preview of a centred tile of the active image
    (up to 1024 px on the long side) with autostretch, so the user can
    see whether the current sigma settings are eating star cores.

    UI knobs:
      - Hot sigma (0.5 – 10.0, default 3.0)
      - Cold sigma (0.5 – 10.0, default 3.0)
      - Bayer pattern combo (Auto / None / RGGB / BGGR / GRBG / GBRG)
      - "Correct hot" / "Correct cold" checkboxes (per-pass on/off)
      - Preview refresh (Preview button, or auto on parameter change)
      - Apply / Cancel + preset drag handle
    """
    _PREVIEW_MAX_LONG_SIDE = 1024

    def __init__(self, main, doc, parent=None, *, tuning_mode: bool = False,
                 sample_source_label: str | None = None):
        """Interactive cosmetic-correction dialog.

        Normal mode: Apply modifies the active document (undoable).

        tuning_mode=True: dialog runs on a sample light frame supplied by
        the caller (e.g. the Stacking Suite).  The Apply button becomes
        "Save σ Settings" and just pushes hot/cold sigma to the Stacking
        Suite's QSettings before closing.  The sample frame is used
        exclusively for the preview — it is never written back.
        """
        super().__init__(parent)
        self.main = main
        self.doc = doc
        self._tuning_mode = bool(tuning_mode)
        self._sample_source_label = sample_source_label or ""
        self.setWindowTitle(
            self.tr("Cosmetic Correction — Tune Sigmas") if self._tuning_mode
            else self.tr("Cosmetic Correction")
        )
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        # Detect Bayer once so the combo starts on the right entry
        self._auto_bayer = _bayer_from_doc(doc)

        # Track mono vs colour source for autostretch dispatch (matches
        # blemish_blaster's _orig_mono handling).
        try:
            _src = np.asarray(doc.image)
            self._orig_mono = (_src.ndim == 2) or (_src.ndim == 3 and _src.shape[2] == 1)
        except Exception:
            self._orig_mono = False

        # Cached tile for preview (computed on demand from the full doc image)
        self._preview_tile_src: np.ndarray | None = None
        self._preview_worker: _CosmeticPreviewWorker | None = None
        self._preview_pending = False
        # Restore last-used preview region (defaults to centre)
        try:
            from PyQt6.QtCore import QSettings
            self._preview_region = str(
                QSettings().value("cosmetic_correction/preview_region", "MC", type=str)
                or "MC"
            )
        except Exception:
            self._preview_region = "MC"
        if self._preview_region not in self._PREVIEW_REGIONS:
            self._preview_region = "MC"

        self._build_ui()
        self._grab_preview_tile()
        self._render_before_preview()
        self._request_preview_refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # ---- top controls in a groupbox ----
        gb = QGroupBox(self.tr("Detection"), self)
        form = QFormLayout(gb)

        self.sp_hot = QDoubleSpinBox()
        self.sp_hot.setRange(0.5, 10.0)
        self.sp_hot.setSingleStep(0.1); self.sp_hot.setDecimals(2)
        self.sp_hot.setValue(3.0)
        self.sp_hot.setToolTip(self.tr(
            "Sigma threshold for hot-pixel detection.  Lower = more aggressive.\n"
            "Bright pixels flagged as hot are replaced by the median of their\n"
            "un-flagged neighbours.  A 3x3 star guard prevents star cores from\n"
            "being clipped."
        ))
        form.addRow(self.tr("Hot pixels σ:"), self.sp_hot)

        self.sp_cold = QDoubleSpinBox()
        self.sp_cold.setRange(0.5, 10.0)
        self.sp_cold.setSingleStep(0.1); self.sp_cold.setDecimals(2)
        self.sp_cold.setValue(3.0)
        self.sp_cold.setToolTip(self.tr(
            "Sigma threshold for cold-pixel detection.  Lower = more aggressive."
        ))
        form.addRow(self.tr("Cold pixels σ:"), self.sp_cold)

        self.cb_hot = QCheckBox(self.tr("Correct hot pixels"))
        self.cb_hot.setChecked(True)
        self.cb_cold = QCheckBox(self.tr("Correct cold pixels"))
        self.cb_cold.setChecked(True)
        pass_row = QHBoxLayout()
        pass_row.addWidget(self.cb_hot)
        pass_row.addWidget(self.cb_cold)
        pass_row.addStretch(1)
        form.addRow(self.tr("Passes:"), pass_row)

        self.cmb_bayer = QComboBox()
        self.cmb_bayer.addItem(self.tr("Auto-detect"), userData="")
        self.cmb_bayer.addItem(self.tr("None (mono / debayered)"), userData="__none__")
        for pat in ("RGGB", "BGGR", "GRBG", "GBRG"):
            self.cmb_bayer.addItem(pat, userData=pat)
        # If auto-detected, tack the actual value into the "Auto" label so it's visible
        if self._auto_bayer:
            self.cmb_bayer.setItemText(0, self.tr(f"Auto-detect  ({self._auto_bayer})"))
        self.cmb_bayer.setToolTip(self.tr(
            "How to treat the image for neighbour comparison.\n"
            "• Auto-detect: read BAYERPAT from FITS header if the image is a 2-D mosaic.\n"
            "• None: standard stride-1 neighbours (use for mono or already-debayered RGB).\n"
            "• RGGB/BGGR/GRBG/GBRG: force stride-2 same-colour neighbours."
        ))
        form.addRow(self.tr("Bayer pattern:"), self.cmb_bayer)

        # Push current sigma settings to Stacking Suite's QSettings so the
        # per-frame cosmetic pass during light calibration uses the same
        # numbers the user just tuned interactively here.
        self.btn_push_stacking = QPushButton(self.tr("Push σ to Stacking Suite"))
        self.btn_push_stacking.setToolTip(self.tr(
            "Copy the current Hot σ and Cold σ values into the Stacking Suite\n"
            "settings.  The per-frame cosmetic pass during light calibration\n"
            "will then use these numbers on future runs."
        ))
        self.btn_push_stacking.clicked.connect(self._on_push_to_stacking)
        push_row = QHBoxLayout()
        push_row.addWidget(self.btn_push_stacking)
        push_row.addStretch(1)
        form.addRow("", push_row)

        root.addWidget(gb)

        # ---- preview area (before | after) ----
        prev_bar = QHBoxLayout()
        prev_bar.addWidget(QLabel(self.tr(
            "<b>Preview</b> — %d px tile, always autostretched"
            % self._PREVIEW_MAX_LONG_SIDE
        )))
        prev_bar.addStretch(1)

        # 3x3 region selector — lets the user pick which corner / edge /
        # centre of the frame to preview.  A bright centred DSO can hide
        # the hot/cold pixel activity, so being able to shift to a corner
        # is essential for tuning sigmas that don't clip star cores.
        prev_bar.addWidget(QLabel(self.tr("Region:")))
        region_grid = QGridLayout()
        region_grid.setSpacing(1)
        region_grid.setContentsMargins(0, 0, 0, 0)
        self._region_btns: dict[str, QToolButton] = {}
        # Layout: (row, col, key)  matching a 3x3 grid.  The letters
        # follow the top/middle/bottom + left/center/right convention.
        _grid_positions = [
            (0, 0, "TL"), (0, 1, "TC"), (0, 2, "TR"),
            (1, 0, "ML"), (1, 1, "MC"), (1, 2, "MR"),
            (2, 0, "BL"), (2, 1, "BC"), (2, 2, "BR"),
        ]
        _region_names = {
            "TL": self.tr("Top-Left"),      "TC": self.tr("Top-Centre"),   "TR": self.tr("Top-Right"),
            "ML": self.tr("Middle-Left"),   "MC": self.tr("Centre"),       "MR": self.tr("Middle-Right"),
            "BL": self.tr("Bottom-Left"),   "BC": self.tr("Bottom-Centre"),"BR": self.tr("Bottom-Right"),
        }
        for r, c, key in _grid_positions:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setAutoExclusive(False)  # we manage exclusivity manually
            btn.setFixedSize(14, 14)
            btn.setToolTip(_region_names[key])
            btn.clicked.connect(lambda _=False, k=key: self._on_region_clicked(k))
            self._region_btns[key] = btn
            region_grid.addWidget(btn, r, c)
        # Wrap the grid in a container widget so it sits nicely in the bar
        _region_holder = QWidget()
        _region_holder.setLayout(region_grid)
        prev_bar.addWidget(_region_holder)
        self._sync_region_buttons()

        self.btn_refresh = QPushButton(self.tr("Refresh preview"))
        self.btn_refresh.clicked.connect(self._request_preview_refresh)
        prev_bar.addWidget(self.btn_refresh)
        root.addLayout(prev_bar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.view_before = _PreviewPane(self)
        self.view_after  = _PreviewPane(self)
        # Link pan + zoom two-way so scrolling / zooming one drives the other.
        self.view_before.set_peer(self.view_after)
        # Small labels above each pane
        pane_before = QVBoxLayout()
        pane_before.addWidget(QLabel(self.tr("Before")))
        pane_before.addWidget(self.view_before)
        pane_after = QVBoxLayout()
        pane_after.addWidget(QLabel(self.tr("After")))
        pane_after.addWidget(self.view_after)

        # Wrap into container widgets for the splitter
        w_before = QWidget(); w_before.setLayout(pane_before)
        w_after  = QWidget(); w_after.setLayout(pane_after)
        self.splitter.addWidget(w_before)
        self.splitter.addWidget(w_after)
        self.splitter.setSizes([500, 500])
        root.addWidget(self.splitter, 1)  # stretch

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        # Buttons
        row = QHBoxLayout()
        if not self._tuning_mode:
            self.btn_batch = QPushButton(self.tr("Batch…"))
            self.btn_batch.setToolTip(self.tr(
                "Open the batch processor.  Pick a list of files, choose an\n"
                "output location, and run cosmetic correction on all of them\n"
                "using these settings."
            ))
            self.btn_batch.clicked.connect(self._open_batch)
            row.addWidget(self.btn_batch)
        else:
            self.btn_batch = None
            # In tuning mode the "Push σ to Stacking Suite" button is
            # redundant — the primary Apply button already does that.
            try:
                self.btn_push_stacking.hide()
            except Exception:
                pass

        # Apply button behaves differently in tuning mode
        if self._tuning_mode:
            self.btn_apply = QPushButton(self.tr("Save σ Settings"))
            self.btn_apply.setToolTip(self.tr(
                "Save Hot σ and Cold σ to Stacking Suite settings and close.\n"
                "The sample frame is NOT modified — it is only used for the\n"
                "preview so you can dial in values that don't clip star cores."
            ))
        else:
            self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_apply.clicked.connect(self._apply)
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(self.btn_apply); row.addWidget(self.btn_cancel)
        root.addLayout(row)

        # Preset drag handle (grip) — skipped in tuning mode (no doc target)
        if not self._tuning_mode:
            try:
                from setiastro.saspro.shortcuts import PresetDragHandle
                from PyQt6.QtGui import QIcon
                try:
                    from setiastro.saspro.resources import cosmeticcorrection_path
                    grip_icon = QIcon(cosmeticcorrection_path)
                except Exception:
                    grip_icon = QIcon()
                drag_row = QHBoxLayout()
                drag_row.setContentsMargins(0, 0, 0, 0)
                self.preset_drag_handle = PresetDragHandle(
                    "cosmetic_correction", self.get_preset, icon=grip_icon,
                    tooltip=self.tr(
                        "Drag to the canvas to create a Cosmetic Correction shortcut with these\n"
                        "settings.  Drop directly on an image to apply them headlessly."
                    ),
                    parent=self,
                )
                drag_row.addWidget(self.preset_drag_handle)
                drag_row.addStretch(1)
                root.addLayout(drag_row)
            except Exception:
                pass

        # Wire parameter changes → schedule a preview refresh + clear status
        for w in (self.sp_hot, self.sp_cold):
            w.valueChanged.connect(lambda _=None: self._on_params_changed())
        for w in (self.cb_hot, self.cb_cold):
            w.toggled.connect(lambda _=None: self._on_params_changed())
        self.cmb_bayer.currentIndexChanged.connect(lambda _=None: self._on_params_changed())

        self.resize(1050, 700)

    # ------------------------------------------------------------------
    # Preset get/seed
    # ------------------------------------------------------------------
    def get_preset(self) -> dict:
        return {
            "hot_sigma": float(self.sp_hot.value()),
            "cold_sigma": float(self.sp_cold.value()),
            "correct_hot": bool(self.cb_hot.isChecked()),
            "correct_cold": bool(self.cb_cold.isChecked()),
            "bayer_pattern": self._selected_bayer_for_preset(),
        }

    def _selected_bayer_for_preset(self) -> str:
        """Persist the raw combo choice; opener re-applies auto-detect."""
        data = self.cmb_bayer.currentData()
        if data == "" or data is None:
            return ""            # "Auto"
        if data == "__none__":
            return "__none__"
        return str(data)

    def seed_from_preset(self, p: dict | None):
        p = dict(p or {})
        if "hot_sigma" in p:
            try: self.sp_hot.setValue(float(p["hot_sigma"]))
            except Exception: pass
        if "cold_sigma" in p:
            try: self.sp_cold.setValue(float(p["cold_sigma"]))
            except Exception: pass
        if "correct_hot" in p:
            self.cb_hot.setChecked(bool(p["correct_hot"]))
        if "correct_cold" in p:
            self.cb_cold.setChecked(bool(p["correct_cold"]))
        bp = str(p.get("bayer_pattern", "") or "").strip()
        idx = 0  # Auto default
        if bp == "__none__":
            idx = 1
        elif bp.upper() in ("RGGB", "BGGR", "GRBG", "GBRG"):
            idx = self.cmb_bayer.findData(bp.upper())
            if idx < 0: idx = 0
        self.cmb_bayer.setCurrentIndex(idx)
        try: self.status_label.clear()
        except Exception: pass
        self._request_preview_refresh()

    # ------------------------------------------------------------------
    # Preview management
    # ------------------------------------------------------------------
    # 3×3 grid of preview regions.  Values are (y_frac, x_frac) tuples
    # locating the CENTER of the tile within the image; a fraction of
    # 0.0 = top/left edge, 0.5 = center, 1.0 = bottom/right edge.
    _PREVIEW_REGIONS = {
        "TL": (0.0, 0.0),  "TC": (0.0, 0.5),  "TR": (0.0, 1.0),
        "ML": (0.5, 0.0),  "MC": (0.5, 0.5),  "MR": (0.5, 1.0),
        "BL": (1.0, 0.0),  "BC": (1.0, 0.5),  "BR": (1.0, 1.0),
    }

    def _grab_preview_tile(self):
        """Extract a tile up to _PREVIEW_MAX_LONG_SIDE per side from the
        current doc image, positioned per the selected 3x3 region.  A
        bright centred DSO can hide the hot/cold pixel activity you're
        trying to see, so the user can shift the preview off-target.
        """
        try:
            img = np.asarray(self.doc.image)
        except Exception:
            self._preview_tile_src = None
            return
        if img is None:
            self._preview_tile_src = None
            return

        arr = _to_float01(img).astype(np.float32, copy=False)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[..., :3]
        H, W = arr.shape[:2]
        S = self._PREVIEW_MAX_LONG_SIDE

        region = getattr(self, "_preview_region", "MC")
        yf, xf = self._PREVIEW_REGIONS.get(region, (0.5, 0.5))

        if max(H, W) > S:
            th = min(H, S); tw = min(W, S)
            # Anchor: yf=0 puts the tile at the top; yf=1 puts it at the
            # bottom; yf=0.5 centres.  Same for xf horizontally.
            y0 = int(round((H - th) * yf))
            x0 = int(round((W - tw) * xf))
            # Guard against 0-pixel images just in case
            y0 = max(0, min(H - th, y0))
            x0 = max(0, min(W - tw, x0))
            arr = arr[y0:y0 + th, x0:x0 + tw]

        self._preview_tile_src = arr

    def _sync_region_buttons(self):
        """Visually mark the currently-selected region button as checked."""
        for key, btn in getattr(self, "_region_btns", {}).items():
            btn.setChecked(key == self._preview_region)

    def _on_region_clicked(self, key: str):
        """Switch the preview tile to a different 3x3 region."""
        if key == self._preview_region:
            # Re-clicking the same region shouldn't unselect it (autoexclusive
            # is off so this can happen without special handling); just make
            # sure the visual state stays correct.
            self._sync_region_buttons()
            return
        self._preview_region = key
        self._sync_region_buttons()
        try:
            from PyQt6.QtCore import QSettings
            QSettings().setValue("cosmetic_correction/preview_region", key)
        except Exception:
            pass

        # Reset the peer-linked panes' fit state so the new tile fills the
        # viewport (rather than inheriting the previous tile's zoom+pan).
        for pane in (getattr(self, "view_before", None),
                     getattr(self, "view_after", None)):
            if pane is not None:
                try:
                    pane._did_initial_fit = False
                    pane._zoom = 1.0
                    pane.resetTransform()
                except Exception:
                    pass

        self._grab_preview_tile()
        self._render_before_preview()
        # Fit the before pane now that a fresh pixmap is loaded — the peer
        # link handles the after pane on the first render.
        try:
            self.view_before._maybe_initial_fit()
        except Exception:
            pass
        self._request_preview_refresh()

    def _render_before_preview(self):
        if self._preview_tile_src is None:
            return
        rgb_u8 = _to_display_rgb(self._preview_tile_src, self._orig_mono)
        self.view_before.set_image(rgb_u8)

    def _resolved_bayer(self) -> str | None:
        data = self.cmb_bayer.currentData()
        if data == "__none__":
            return None
        if data == "" or data is None:
            # Auto — only meaningful for a 2-D tile
            if self._preview_tile_src is not None and self._preview_tile_src.ndim == 2:
                return self._auto_bayer
            return None
        return str(data)

    def _on_params_changed(self):
        try:
            self.status_label.clear()
        except Exception:
            pass
        self._request_preview_refresh()

    def _on_push_to_stacking(self):
        """Copy Hot σ / Cold σ into the Stacking Suite's QSettings keys.

        The suite reads these on every calibrate_lights() run:
          stacking/cosmetic/hot_sigma
          stacking/cosmetic/cold_sigma
        so this button lets the user tune them here (with the preview) and
        have the same values applied to per-frame calibration.
        """
        from PyQt6.QtCore import QSettings
        hs = float(self.sp_hot.value())
        cs = float(self.sp_cold.value())
        try:
            s = QSettings()
            s.setValue("stacking/cosmetic/hot_sigma", hs)
            s.setValue("stacking/cosmetic/cold_sigma", cs)
            s.sync()
        except Exception as e:
            QMessageBox.warning(self, self.tr("Push to Stacking Suite"),
                                self.tr(f"Could not update settings:\n{e}"))
            return

        if hasattr(self.main, "_log"):
            try:
                self.main._log(
                    f"Pushed cosmetic σ to Stacking Suite: hot={hs:.2f}, cold={cs:.2f}"
                )
            except Exception:
                pass

        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setText(
            self.tr(f"✓ Pushed σ to Stacking Suite (hot={hs:.2f}, cold={cs:.2f})")
        )

    def _request_preview_refresh(self):
        if self._preview_tile_src is None:
            return
        # If a worker is already running, remember that we need another pass
        if self._preview_worker is not None and self._preview_worker.isRunning():
            self._preview_pending = True
            return
        self._launch_preview_worker()

    def _launch_preview_worker(self):
        self._preview_pending = False
        if self._preview_tile_src is None:
            return

        hs = float(self.sp_hot.value()) if self.cb_hot.isChecked() else 1e9
        cs = float(self.sp_cold.value()) if self.cb_cold.isChecked() else 1e9
        bp = self._resolved_bayer()

        self.btn_refresh.setEnabled(False)
        self.btn_apply.setEnabled(False)

        self._preview_worker = _CosmeticPreviewWorker(
            self._preview_tile_src,
            hot_sigma=hs, cold_sigma=cs, bayer_pattern=bp,
            parent=self,
        )
        self._preview_worker.finished_ok.connect(self._on_preview_ok)
        self._preview_worker.failed.connect(self._on_preview_fail)
        self._preview_worker.start()

    def _on_preview_ok(self, corrected: np.ndarray):
        try:
            rgb_u8 = _to_display_rgb(corrected, self._orig_mono)
            self.view_after.set_image(rgb_u8)
        finally:
            self.btn_refresh.setEnabled(True)
            self.btn_apply.setEnabled(True)
        # If parameters changed mid-run, launch another pass now
        if self._preview_pending:
            self._launch_preview_worker()

    def _on_preview_fail(self, msg: str):
        self.status_label.setStyleSheet("color: #d05050; font-weight: bold;")
        self.status_label.setText(self.tr(f"Preview failed: {msg}"))
        self.btn_refresh.setEnabled(True)
        self.btn_apply.setEnabled(True)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------
    def _apply(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            QMessageBox.warning(self, self.tr("Cosmetic Correction"),
                                self.tr("No image."))
            return

        hs = float(self.sp_hot.value())
        cs = float(self.sp_cold.value())
        ch = bool(self.cb_hot.isChecked())
        cc = bool(self.cb_cold.isChecked())
        if not (ch or cc):
            QMessageBox.information(self, self.tr("Cosmetic Correction"),
                self.tr("Nothing to do — enable at least one of Hot / Cold."))
            return

        # Tuning mode (Stacking Suite): save σ settings and close — do NOT
        # modify the sample frame.  The sample is only a preview aid.
        if self._tuning_mode:
            try:
                from PyQt6.QtCore import QSettings
                s = QSettings()
                s.setValue("stacking/cosmetic/hot_sigma", hs)
                s.setValue("stacking/cosmetic/cold_sigma", cs)
                s.sync()
            except Exception as e:
                QMessageBox.warning(self, self.tr("Cosmetic Correction"),
                    self.tr(f"Could not save settings:\n{e}"))
                return
            if hasattr(self.main, "_log"):
                try:
                    self.main._log(
                        f"Cosmetic σ tuned in Stacking Suite: "
                        f"hot={hs:.2f}, cold={cs:.2f}"
                    )
                except Exception:
                    pass
            elif hasattr(self.main, "update_status"):
                try:
                    self.main.update_status(
                        f"✓ Cosmetic σ saved: hot={hs:.2f}, cold={cs:.2f}"
                    )
                except Exception:
                    pass
            self.accept()
            return

        bp_raw = self._selected_bayer_for_preset()
        if bp_raw == "" or bp_raw is None:
            bp = None  # None here → headless auto-detects
        elif bp_raw == "__none__":
            bp = "__none__"  # sentinel meaning "explicitly none"
        else:
            bp = bp_raw

        preset = {
            "hot_sigma": hs, "cold_sigma": cs,
            "correct_hot": ch, "correct_cold": cc,
            "bayer_pattern": bp_raw,
        }

        try:
            cosmetic_correction_headless(
                self.doc,
                hot_sigma=hs, cold_sigma=cs,
                bayer_pattern=(None if bp == "__none__" else bp),
                correct_hot=ch, correct_cold=cc,
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("Cosmetic Correction"),
                                 self.tr(f"Apply failed:\n{e}"))
            return

        # Refresh the preview tile from the corrected image
        try:
            self._grab_preview_tile()
            self._render_before_preview()
            self._request_preview_refresh()
        except Exception:
            pass

        # Log + replay bookkeeping
        try:
            if hasattr(self.main, "_log"):
                self.main._log(
                    f"Cosmetic Correction applied — hot σ={hs:.2f} ({'on' if ch else 'off'}), "
                    f"cold σ={cs:.2f} ({'on' if cc else 'off'}), bayer={bp_raw or 'auto'}"
                )
            self.main._last_headless_command = {
                "command_id": "cosmetic_correction",
                "preset": dict(preset),
            }
        except Exception:
            pass

        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        try:
            name = self.doc.display_name() if hasattr(self.doc, "display_name") else ""
        except Exception:
            name = ""
        self.status_label.setText(
            self.tr(f"✓ Applied to “{name}”") if name else self.tr("✓ Applied")
        )
        self._refresh_document_from_active()

    def _refresh_document_from_active(self):
        try:
            if self.main and hasattr(self.main, "_active_doc"):
                new_doc = self.main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    self._auto_bayer = _bayer_from_doc(self.doc)
                    if self._auto_bayer:
                        self.cmb_bayer.setItemText(0, self.tr(f"Auto-detect  ({self._auto_bayer})"))
                    else:
                        self.cmb_bayer.setItemText(0, self.tr("Auto-detect"))
                    try:
                        _src = np.asarray(self.doc.image)
                        self._orig_mono = (
                            _src.ndim == 2
                            or (_src.ndim == 3 and _src.shape[2] == 1)
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Batch launcher
    # ------------------------------------------------------------------
    def _open_batch(self):
        """Open the batch processor, seeded with the current settings."""
        preset = self.get_preset()
        dlg = CosmeticCorrectionBatchDialog(self.main, initial_preset=preset,
                                            parent=self)
        try:
            from setiastro.saspro.resources import cosmeticcorrection_path
            from PyQt6.QtGui import QIcon
            dlg.setWindowIcon(QIcon(cosmeticcorrection_path))
        except Exception:
            pass
        dlg.show(); dlg.raise_(); dlg.activateWindow()


# =====================================================================
# Batch worker + dialog
# =====================================================================
class _CosmeticBatchWorker(QThread):
    """Loads each file, runs cosmetic correction, saves it back to disk.

    Runs on a worker thread so the UI stays responsive.  Emits per-file
    progress signals and honours a cancel request.
    """
    file_started = pyqtSignal(int, str)          # (index, source path)
    file_done    = pyqtSignal(int, str, str)     # (index, output path, message)
    file_failed  = pyqtSignal(int, str, str)     # (index, source path, error message)
    all_done     = pyqtSignal(int, int)          # (n_ok, n_fail)

    def __init__(
        self,
        paths: list[str],
        *,
        hot_sigma: float,
        cold_sigma: float,
        correct_hot: bool,
        correct_cold: bool,
        bayer_pattern: str,     # "" auto, "__none__" force off, "RGGB"/... force
        out_mode: str,          # "overwrite" | "suffix" | "dir"
        out_dir: str = "",
        suffix: str = "_cc",
        parent=None,
    ):
        super().__init__(parent)
        self._paths = list(paths)
        self._hot = float(hot_sigma) if bool(correct_hot) else 1e9
        self._cold = float(cold_sigma) if bool(correct_cold) else 1e9
        self._bp = bayer_pattern
        self._out_mode = out_mode
        self._out_dir = out_dir or ""
        self._suffix = suffix or ""
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # -- output filename resolution ------------------------------------
    def _resolve_output_path(self, src_path: str) -> str:
        base_dir, base_name = os.path.split(src_path)
        stem, ext = os.path.splitext(base_name)
        if self._out_mode == "overwrite":
            return src_path
        if self._out_mode == "suffix":
            return os.path.join(base_dir, f"{stem}{self._suffix}{ext}")
        # "dir"
        if not self._out_dir:
            return os.path.join(base_dir, f"{stem}{self._suffix}{ext}")
        # Preserve the source's stem+ext, drop it into the target dir.
        # If the target dir equals the source dir AND no suffix was
        # given, that would overwrite the source — refuse gracefully.
        target = os.path.join(self._out_dir, base_name)
        if os.path.abspath(target) == os.path.abspath(src_path):
            if self._suffix:
                target = os.path.join(self._out_dir, f"{stem}{self._suffix}{ext}")
            else:
                # append a default marker to avoid clobbering the source
                target = os.path.join(self._out_dir, f"{stem}_cc{ext}")
        return target

    # -- Bayer resolution ----------------------------------------------
    def _bayer_for(self, header, is_mono: bool, image_ndim: int) -> str | None:
        """Only meaningful for 2-D mono mosaics; ignored otherwise."""
        if image_ndim != 2:
            return None
        if self._bp == "__none__":
            return None
        if self._bp and self._bp.upper() in ("RGGB", "BGGR", "GRBG", "GBRG"):
            return self._bp.upper()
        # Auto: read from FITS header if present
        if header is not None:
            try:
                if hasattr(header, "get"):
                    v = header.get("BAYERPAT")
                else:
                    v = getattr(header, "BAYERPAT", None)
                if v:
                    s = str(v).strip().upper()
                    if s in ("RGGB", "BGGR", "GRBG", "GBRG"):
                        return s
            except Exception:
                pass
        return None

    # -- main loop ------------------------------------------------------
    def run(self):
        from setiastro.saspro.legacy.image_manager import load_image, save_image

        n_ok = n_fail = 0
        for i, src in enumerate(self._paths):
            if self._cancel:
                break
            self.file_started.emit(i, src)
            try:
                image, header, bit_depth, is_mono = load_image(src)
                if image is None:
                    self.file_failed.emit(i, src, "load returned None")
                    n_fail += 1
                    continue

                arr = np.asarray(image, dtype=np.float32)
                bp = self._bayer_for(header, is_mono, arr.ndim)

                corrected = _run_cosmetic(
                    arr, hot_sigma=self._hot, cold_sigma=self._cold,
                    bayer_pattern=bp,
                )

                # Determine format from source extension
                _, src_ext = os.path.splitext(src)
                fmt = (src_ext or ".fits").lstrip(".").lower()

                # Preserve the source's bit depth if the loader reported one;
                # fall back to a 32-bit float FITS-safe default otherwise.
                out_bit_depth = bit_depth or "32-bit floating point"

                out_path = self._resolve_output_path(src)
                # Make sure the output directory exists (dir-mode)
                out_parent = os.path.dirname(out_path)
                if out_parent and not os.path.isdir(out_parent):
                    try:
                        os.makedirs(out_parent, exist_ok=True)
                    except Exception:
                        pass

                # Add HISTORY line if this is a FITS header we can annotate
                try:
                    if header is not None:
                        note = (
                            f"Cosmetic Correction: hot σ={self._hot:g}, "
                            f"cold σ={self._cold:g}, bayer={bp or 'none'}"
                        )
                        if hasattr(header, "add_history"):
                            header.add_history(note)
                        elif hasattr(header, "__setitem__"):
                            header["HISTORY"] = note
                except Exception:
                    pass

                save_image(
                    img_array=corrected,
                    filename=out_path,
                    original_format=fmt,
                    bit_depth=out_bit_depth,
                    original_header=header,
                    is_mono=(corrected.ndim == 2),
                )

                self.file_done.emit(i, out_path,
                    f"saved → {os.path.basename(out_path)}"
                    + (f"  ({bp})" if bp else ""))
                n_ok += 1
            except Exception as e:
                self.file_failed.emit(i, src, str(e))
                n_fail += 1

        self.all_done.emit(n_ok, n_fail)


class CosmeticCorrectionBatchDialog(QDialog):
    """Batch cosmetic correction across a user-picked file list.

    Layout:
      - File list (add / remove / clear buttons)
      - Detection knobs (same as interactive dialog, prefilled from initial_preset)
      - Output group: Overwrite / Add suffix / Save to directory (with picker)
      - Progress bar + status label + Run / Cancel buttons
    """
    _SUPPORTED_EXTS = (
        "*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz "
        "*.xisf *.tif *.tiff *.png *.jpg *.jpeg *.cr2 *.cr3 *.nef "
        "*.arw *.dng *.raf *.pef *.orf *.rw2"
    )

    def __init__(self, main, initial_preset: dict | None = None, parent=None):
        super().__init__(parent)
        self.main = main
        self.setWindowTitle(self.tr("Cosmetic Correction — Batch"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self._worker: _CosmeticBatchWorker | None = None
        self._n_ok = 0
        self._n_fail = 0

        self._build_ui(initial_preset or {})
        self.resize(760, 620)

    def _build_ui(self, preset: dict):
        root = QVBoxLayout(self)

        # ---- file list ----
        gb_files = QGroupBox(self.tr("Files to process"), self)
        files_v = QVBoxLayout(gb_files)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setToolTip(self.tr(
            "Drag files here, or use the buttons.  Files are processed in the\n"
            "listed order.  Duplicates are ignored."
        ))
        self.list.setAcceptDrops(True)
        # Wire drag-drop directly on the widget via event filter
        self.list.installEventFilter(self)
        files_v.addWidget(self.list, 1)

        files_btn_row = QHBoxLayout()
        self.btn_add = QPushButton(self.tr("Add files…"))
        self.btn_add.clicked.connect(self._on_add_files)
        self.btn_add_dir = QPushButton(self.tr("Add folder…"))
        self.btn_add_dir.clicked.connect(self._on_add_folder)
        self.btn_remove = QPushButton(self.tr("Remove selected"))
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_clear = QPushButton(self.tr("Clear"))
        self.btn_clear.clicked.connect(self._on_clear)
        files_btn_row.addWidget(self.btn_add)
        files_btn_row.addWidget(self.btn_add_dir)
        files_btn_row.addStretch(1)
        files_btn_row.addWidget(self.btn_remove)
        files_btn_row.addWidget(self.btn_clear)
        files_v.addLayout(files_btn_row)
        root.addWidget(gb_files, 1)

        # ---- detection knobs (mirror the interactive dialog) ----
        gb_det = QGroupBox(self.tr("Detection"), self)
        det_form = QFormLayout(gb_det)

        self.sp_hot = QDoubleSpinBox()
        self.sp_hot.setRange(0.5, 10.0); self.sp_hot.setSingleStep(0.1)
        self.sp_hot.setDecimals(2)
        self.sp_hot.setValue(float(preset.get("hot_sigma", 3.0)))
        det_form.addRow(self.tr("Hot pixels σ:"), self.sp_hot)

        self.sp_cold = QDoubleSpinBox()
        self.sp_cold.setRange(0.5, 10.0); self.sp_cold.setSingleStep(0.1)
        self.sp_cold.setDecimals(2)
        self.sp_cold.setValue(float(preset.get("cold_sigma", 3.0)))
        det_form.addRow(self.tr("Cold pixels σ:"), self.sp_cold)

        self.cb_hot = QCheckBox(self.tr("Correct hot pixels"))
        self.cb_hot.setChecked(bool(preset.get("correct_hot", True)))
        self.cb_cold = QCheckBox(self.tr("Correct cold pixels"))
        self.cb_cold.setChecked(bool(preset.get("correct_cold", True)))
        pass_row = QHBoxLayout()
        pass_row.addWidget(self.cb_hot); pass_row.addWidget(self.cb_cold)
        pass_row.addStretch(1)
        det_form.addRow(self.tr("Passes:"), pass_row)

        self.cmb_bayer = QComboBox()
        self.cmb_bayer.addItem(self.tr("Auto-detect (from FITS header)"), userData="")
        self.cmb_bayer.addItem(self.tr("None (mono / debayered)"), userData="__none__")
        for pat in ("RGGB", "BGGR", "GRBG", "GBRG"):
            self.cmb_bayer.addItem(pat, userData=pat)
        bp = str(preset.get("bayer_pattern", "") or "")
        if bp == "__none__":
            idx = 1
        elif bp.upper() in ("RGGB", "BGGR", "GRBG", "GBRG"):
            idx = self.cmb_bayer.findData(bp.upper())
            if idx < 0: idx = 0
        else:
            idx = 0
        self.cmb_bayer.setCurrentIndex(idx)
        det_form.addRow(self.tr("Bayer pattern:"), self.cmb_bayer)

        self.btn_push_stacking = QPushButton(self.tr("Push σ to Stacking Suite"))
        self.btn_push_stacking.setToolTip(self.tr(
            "Copy the current Hot σ and Cold σ into the Stacking Suite settings,\n"
            "so per-frame cosmetic correction during light calibration uses them."
        ))
        self.btn_push_stacking.clicked.connect(self._on_push_to_stacking)
        push_row = QHBoxLayout()
        push_row.addWidget(self.btn_push_stacking)
        push_row.addStretch(1)
        det_form.addRow("", push_row)

        root.addWidget(gb_det)

        # ---- output selection ----
        gb_out = QGroupBox(self.tr("Output"), self)
        out_v = QVBoxLayout(gb_out)
        self._out_group = QButtonGroup(self)
        self.rb_overwrite = QRadioButton(self.tr("Overwrite source files (danger!)"))
        self.rb_suffix    = QRadioButton(self.tr("Save alongside source with suffix"))
        self.rb_dir       = QRadioButton(self.tr("Save to directory"))
        self.rb_suffix.setChecked(True)
        for rb in (self.rb_overwrite, self.rb_suffix, self.rb_dir):
            self._out_group.addButton(rb)
            out_v.addWidget(rb)

        suf_row = QHBoxLayout()
        suf_row.addSpacing(20)
        suf_row.addWidget(QLabel(self.tr("Suffix:")))
        self.ed_suffix = QLineEdit("_cc")
        self.ed_suffix.setMaximumWidth(120)
        suf_row.addWidget(self.ed_suffix)
        suf_row.addStretch(1)
        out_v.addLayout(suf_row)

        dir_row = QHBoxLayout()
        dir_row.addSpacing(20)
        dir_row.addWidget(QLabel(self.tr("Directory:")))
        self.ed_dir = QLineEdit("")
        self.btn_pick_dir = QPushButton(self.tr("Browse…"))
        self.btn_pick_dir.clicked.connect(self._on_pick_dir)
        dir_row.addWidget(self.ed_dir, 1)
        dir_row.addWidget(self.btn_pick_dir)
        out_v.addLayout(dir_row)

        # Enable/disable the sub-controls based on radio selection
        def _sync_out_enabled():
            self.ed_suffix.setEnabled(self.rb_suffix.isChecked()
                                      or self.rb_dir.isChecked())
            self.ed_dir.setEnabled(self.rb_dir.isChecked())
            self.btn_pick_dir.setEnabled(self.rb_dir.isChecked())
        for rb in (self.rb_overwrite, self.rb_suffix, self.rb_dir):
            rb.toggled.connect(lambda _=None: _sync_out_enabled())
        _sync_out_enabled()

        root.addWidget(gb_out)

        # ---- progress + status ----
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.status_label)

        # ---- run / cancel / close ----
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_run = QPushButton(self.tr("Run"))
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton(self.tr("Cancel"))
        self.btn_stop.clicked.connect(self._on_cancel)
        self.btn_stop.setEnabled(False)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Push σ to Stacking Suite settings
    # ------------------------------------------------------------------
    def _on_push_to_stacking(self):
        """See CosmeticCorrectionDialog._on_push_to_stacking."""
        from PyQt6.QtCore import QSettings
        hs = float(self.sp_hot.value())
        cs = float(self.sp_cold.value())
        try:
            s = QSettings()
            s.setValue("stacking/cosmetic/hot_sigma", hs)
            s.setValue("stacking/cosmetic/cold_sigma", cs)
            s.sync()
        except Exception as e:
            QMessageBox.warning(self, self.tr("Push to Stacking Suite"),
                                self.tr(f"Could not update settings:\n{e}"))
            return

        if hasattr(self.main, "_log"):
            try:
                self.main._log(
                    f"Pushed cosmetic σ to Stacking Suite: hot={hs:.2f}, cold={cs:.2f}"
                )
            except Exception:
                pass

        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setText(
            self.tr(f"✓ Pushed σ to Stacking Suite (hot={hs:.2f}, cold={cs:.2f})")
        )

    # ------------------------------------------------------------------
    # File list management
    # ------------------------------------------------------------------
    def _current_paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]

    def _add_paths(self, paths):
        existing = set(self._current_paths())
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if not os.path.isfile(p):
                continue
            if p in existing:
                continue
            it = QListWidgetItem(p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.list.addItem(it)
            existing.add(p)
            added += 1
        if added:
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.status_label.setText(self.tr(f"Added {added} file(s)."))

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Add files"), "",
            self.tr(f"Images ({self._SUPPORTED_EXTS});;All files (*)"),
        )
        if paths:
            self._add_paths(paths)

    def _on_add_folder(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Add folder"))
        if not d:
            return
        # Recursively find supported images
        exts = {"." + e.strip("*.").lower() for e in self._SUPPORTED_EXTS.split()}
        found = []
        for root, _dirs, files in os.walk(d):
            for f in files:
                low = f.lower()
                if any(low.endswith(e) for e in exts):
                    found.append(os.path.join(root, f))
        if found:
            self._add_paths(sorted(found))
        else:
            self.status_label.setStyleSheet("color: #ffb020; font-weight: bold;")
            self.status_label.setText(self.tr("No supported images found in that folder."))

    def _on_remove_selected(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _on_clear(self):
        self.list.clear()

    def _on_pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Output directory"),
                                             self.ed_dir.text() or "")
        if d:
            self.ed_dir.setText(d)

    # ------------------------------------------------------------------
    # Drag & drop onto the file list
    # ------------------------------------------------------------------
    def eventFilter(self, src, ev):
        if src is self.list:
            et = ev.type()
            from PyQt6.QtCore import QEvent
            if et == QEvent.Type.DragEnter or et == QEvent.Type.DragMove:
                if ev.mimeData().hasUrls():
                    ev.acceptProposedAction()
                    return True
            elif et == QEvent.Type.Drop:
                if ev.mimeData().hasUrls():
                    paths = [u.toLocalFile() for u in ev.mimeData().urls()
                             if u.toLocalFile()]
                    # Expand any dropped folders
                    expanded = []
                    exts = {"." + e.strip("*.").lower()
                            for e in self._SUPPORTED_EXTS.split()}
                    for p in paths:
                        if os.path.isdir(p):
                            for root, _dirs, files in os.walk(p):
                                for f in files:
                                    if any(f.lower().endswith(e) for e in exts):
                                        expanded.append(os.path.join(root, f))
                        else:
                            expanded.append(p)
                    if expanded:
                        self._add_paths(expanded)
                    ev.acceptProposedAction()
                    return True
        return super().eventFilter(src, ev)

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------
    def _selected_out_mode(self) -> str:
        if self.rb_overwrite.isChecked():
            return "overwrite"
        if self.rb_dir.isChecked():
            return "dir"
        return "suffix"

    def _on_run(self):
        paths = self._current_paths()
        if not paths:
            QMessageBox.information(self, self.tr("Cosmetic Correction — Batch"),
                                    self.tr("Add some files first."))
            return

        out_mode = self._selected_out_mode()
        out_dir = self.ed_dir.text().strip()
        suffix = self.ed_suffix.text().strip()

        if out_mode == "dir" and not out_dir:
            QMessageBox.warning(self, self.tr("Cosmetic Correction — Batch"),
                                self.tr("Pick an output directory."))
            return
        if out_mode == "suffix" and not suffix:
            QMessageBox.warning(self, self.tr("Cosmetic Correction — Batch"),
                                self.tr("Provide a filename suffix (e.g. \"_cc\")."))
            return

        ch = bool(self.cb_hot.isChecked())
        cc = bool(self.cb_cold.isChecked())
        if not (ch or cc):
            QMessageBox.information(self, self.tr("Cosmetic Correction — Batch"),
                self.tr("Nothing to do — enable at least one of Hot / Cold."))
            return

        if out_mode == "overwrite":
            ret = QMessageBox.question(
                self, self.tr("Overwrite source files?"),
                self.tr(f"This will overwrite {len(paths)} source file(s) in place. "
                        "This cannot be undone.  Continue?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        # Kick off the worker
        self._n_ok = 0
        self._n_fail = 0
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setText(self.tr(f"Processing {len(paths)} file(s)…"))

        bp_raw = str(self.cmb_bayer.currentData() or "")

        self._set_controls_enabled(False)
        self._worker = _CosmeticBatchWorker(
            paths,
            hot_sigma=float(self.sp_hot.value()),
            cold_sigma=float(self.sp_cold.value()),
            correct_hot=ch, correct_cold=cc,
            bayer_pattern=bp_raw,
            out_mode=out_mode, out_dir=out_dir, suffix=suffix,
            parent=self,
        )
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.start()

    def _on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.status_label.setStyleSheet("color: #ffb020; font-weight: bold;")
            self.status_label.setText(self.tr("Cancelling…"))
            self.btn_stop.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool):
        for w in (self.btn_add, self.btn_add_dir, self.btn_remove, self.btn_clear,
                  self.sp_hot, self.sp_cold, self.cb_hot, self.cb_cold,
                  self.cmb_bayer, self.btn_push_stacking,
                  self.rb_overwrite, self.rb_suffix, self.rb_dir,
                  self.ed_suffix, self.ed_dir, self.btn_pick_dir, self.btn_run,
                  self.btn_close):
            w.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

    # ------------------------------------------------------------------
    # Worker signals
    # ------------------------------------------------------------------
    def _on_file_started(self, idx: int, src: str):
        try:
            it = self.list.item(idx)
            if it is not None:
                it.setText(f"⟳  {src}")
                self.list.scrollToItem(it)
        except Exception:
            pass

    def _on_file_done(self, idx: int, out_path: str, message: str):
        try:
            it = self.list.item(idx)
            if it is not None:
                src = it.data(Qt.ItemDataRole.UserRole)
                it.setText(f"✓  {src}   →   {os.path.basename(out_path)}")
        except Exception:
            pass
        self._n_ok += 1
        self.progress.setValue(self._n_ok + self._n_fail)
        if hasattr(self.main, "_log"):
            try:
                self.main._log(f"[Batch CC] {message}")
            except Exception:
                pass

    def _on_file_failed(self, idx: int, src: str, err: str):
        try:
            it = self.list.item(idx)
            if it is not None:
                it.setText(f"✗  {src}   —   {err}")
                it.setForeground(Qt.GlobalColor.red)
        except Exception:
            pass
        self._n_fail += 1
        self.progress.setValue(self._n_ok + self._n_fail)
        if hasattr(self.main, "_log"):
            try:
                self.main._log(f"[Batch CC] FAILED: {src}: {err}")
            except Exception:
                pass

    def _on_all_done(self, n_ok: int, n_fail: int):
        self._set_controls_enabled(True)
        if n_fail == 0:
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.status_label.setText(self.tr(f"✓ Done — {n_ok} file(s) processed."))
        else:
            self.status_label.setStyleSheet("color: #ffb020; font-weight: bold;")
            self.status_label.setText(
                self.tr(f"Done — {n_ok} ok, {n_fail} failed.  See list for details.")
            )
        if hasattr(self.main, "_log"):
            try:
                self.main._log(f"[Batch CC] finished: {n_ok} ok, {n_fail} failed")
            except Exception:
                pass

    def closeEvent(self, ev):
        # Don't leave a worker running behind our back
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(ev)


# =====================================================================
# Entry points used by main / shortcuts
# =====================================================================
def open_cosmetic_correction_dialog(main, doc=None, preset: dict | None = None):
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        # No image → open batch mode instead of nagging the user.  The batch
        # dialog operates on files, so it doesn't need an active view.
        return open_cosmetic_correction_batch(main, preset=preset)

    dlg = CosmeticCorrectionDialog(main, doc, parent=main)
    try:
        from setiastro.saspro.resources import cosmeticcorrection_path
        from PyQt6.QtGui import QIcon
        dlg.setWindowIcon(QIcon(cosmeticcorrection_path))
    except Exception:
        pass
    if preset:
        dlg.seed_from_preset(preset)
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    try:
        main._cosmetic_correction_dialog = dlg
    except Exception:
        pass
    return dlg


def apply_cosmetic_correction_preset_to_doc(main, doc, preset: dict):
    """Headless-apply from a preset (grip drop onto image, replay, etc.)."""
    hs = float(preset.get("hot_sigma", 3.0))
    cs = float(preset.get("cold_sigma", 3.0))
    ch = bool(preset.get("correct_hot", True))
    cc = bool(preset.get("correct_cold", True))
    bp_raw = str(preset.get("bayer_pattern", "") or "").strip()
    if bp_raw == "" or bp_raw is None:
        bp = None                                # None here → auto-detect
    elif bp_raw == "__none__":
        bp = None                                # explicit none == also None
    else:
        bp = bp_raw.upper() if bp_raw.upper() in ("RGGB", "BGGR", "GRBG", "GBRG") else None

    cosmetic_correction_headless(
        doc,
        hot_sigma=hs, cold_sigma=cs,
        bayer_pattern=bp,
        correct_hot=ch, correct_cold=cc,
    )
    if hasattr(main, "_log"):
        try:
            name = doc.display_name() if hasattr(doc, "display_name") else "Image"
            main._log(
                f"Cosmetic Correction (headless) on '{name}': "
                f"hot σ={hs:.2f} ({'on' if ch else 'off'}), "
                f"cold σ={cs:.2f} ({'on' if cc else 'off'}), bayer={bp_raw or 'auto'}"
            )
        except Exception:
            pass


def open_cosmetic_correction_with_preset(main_window, preset: dict | None = None):
    """Double-click a shortcut → open the dialog seeded from the preset."""
    from PyQt6.QtGui import QIcon

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
        # No image → open batch mode seeded with the shortcut's preset.
        return open_cosmetic_correction_batch(main_window, preset=preset)

    dlg = CosmeticCorrectionDialog(main_window, doc, parent=main_window)
    try:
        from setiastro.saspro.resources import cosmeticcorrection_path
        dlg.setWindowIcon(QIcon(cosmeticcorrection_path))
    except Exception:
        pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    try:
        main_window._cosmetic_correction_dialog = dlg
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg


def open_cosmetic_correction_batch(main, preset: dict | None = None):
    """Open the batch cosmetic-correction dialog directly (no image
    context required — the batch tool works on files, not on the active
    view).  Optionally seed detection knobs from a preset."""
    from PyQt6.QtGui import QIcon
    dlg = CosmeticCorrectionBatchDialog(main, initial_preset=(preset or {}),
                                        parent=main)
    try:
        from setiastro.saspro.resources import cosmeticcorrection_path
        dlg.setWindowIcon(QIcon(cosmeticcorrection_path))
    except Exception:
        pass
    try:
        main._cosmetic_correction_batch_dialog = dlg
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg


# =====================================================================
# Tuning-mode opener (Stacking Suite)
# =====================================================================
class _SampleFrameDoc:
    """Minimal doc-like stub wrapping a raw image + FITS header.

    Used only as an input to CosmeticCorrectionDialog in tuning mode,
    where the dialog needs:
      - `image`    (ndarray) for preview
      - `header`   (FITS header, optional) for Bayer auto-detect
      - `display_name()` for status labels
    Everything else on a real Document isn't touched, because tuning
    mode never calls apply_edit / masks / undo.
    """
    def __init__(self, image, header=None, display_name: str = "Sample light"):
        self.image = image
        self.header = header
        self.metadata = header    # _bayer_from_doc checks either name
        self._display_name = str(display_name)
        self.active_mask_id = None
        self.masks = {}

    def display_name(self) -> str:
        return self._display_name


def open_cosmetic_correction_tune(main, sample_light_path: str,
                                  preset: dict | None = None):
    """Open Cosmetic Correction in Stacking-Suite tuning mode.

    Loads `sample_light_path` as the preview source, seeds the sigma
    spin-boxes from stacking/cosmetic/* QSettings (or from `preset` if
    given), and — on Apply — writes those sigmas back to the same
    QSettings keys without modifying the sample file.

    Returns the dialog instance (already show()'n) or None if the sample
    could not be loaded.
    """
    from PyQt6.QtCore import QSettings
    from PyQt6.QtGui import QIcon
    import os
    try:
        from setiastro.saspro.legacy.image_manager import load_image
    except Exception as e:
        QMessageBox.critical(main, "Cosmetic Correction",
                             f"Could not import image loader:\n{e}")
        return None

    if not sample_light_path or not os.path.isfile(sample_light_path):
        QMessageBox.warning(main, "Cosmetic Correction",
                            "Could not find a sample light frame to load.")
        return None

    # Load the sample light
    try:
        image, header, _bit_depth, _is_mono = load_image(sample_light_path)
    except Exception as e:
        QMessageBox.critical(main, "Cosmetic Correction",
                             f"Failed to load sample light:\n{e}")
        return None
    if image is None:
        QMessageBox.warning(main, "Cosmetic Correction",
                            f"Sample load returned no image:\n{sample_light_path}")
        return None

    # Wrap it as a stub doc for the dialog
    stub = _SampleFrameDoc(
        image=np.asarray(image, dtype=np.float32),
        header=header,
        display_name=os.path.basename(sample_light_path),
    )

    # Seed preset from existing stacking settings if the caller didn't
    # supply one — this way opening the dialog shows the sigmas that are
    # currently active in the calibration pipeline.
    if not preset:
        try:
            s = QSettings()
            preset = {
                "hot_sigma": float(s.value("stacking/cosmetic/hot_sigma", 3.0, type=float)),
                "cold_sigma": float(s.value("stacking/cosmetic/cold_sigma", 3.0, type=float)),
                "correct_hot": True,
                "correct_cold": True,
                "bayer_pattern": "",
            }
        except Exception:
            preset = {}

    dlg = CosmeticCorrectionDialog(
        main, stub, parent=main,
        tuning_mode=True,
        sample_source_label=os.path.basename(sample_light_path),
    )
    try:
        from setiastro.saspro.resources import cosmeticcorrection_path
        dlg.setWindowIcon(QIcon(cosmeticcorrection_path))
    except Exception:
        pass
    if preset:
        dlg.seed_from_preset(preset)
    try:
        main._cosmetic_correction_tune_dialog = dlg
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg