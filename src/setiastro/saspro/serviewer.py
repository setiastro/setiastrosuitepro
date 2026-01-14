# src/setiastro/saspro/serviewer.py
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, QEvent, QPoint, QRect, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QScrollArea, QSlider, QCheckBox, QGroupBox, QFormLayout, QSpinBox,
    QMessageBox, QRubberBand, QComboBox, QDoubleSpinBox
)

from setiastro.saspro.imageops.serloader import open_planetary_source, PlanetaryFrameSource
import threading
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QProgressDialog


from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.ser_stack_config import SERStackConfig
from setiastro.saspro.ser_stacker import stack_ser
from setiastro.saspro.ser_stacker_dialog import SERStackerDialog

# Use your stretch functions for DISPLAY
try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None

class _TrimExportWorker(QObject):
    progress = pyqtSignal(int, int)   # done, total
    finished = pyqtSignal(str)        # out_path
    failed = pyqtSignal(str)          # error text
    canceled = pyqtSignal()

    def __init__(
        self,
        src: PlanetaryFrameSource,
        out_path: str,
        start: int,
        end: int,
        *,
        bayer_pattern: str | None,
        store_raw_mosaic_if_forced: bool,
        progress_every: int = 10,
    ):
        super().__init__()
        self._src = src
        self._out_path = out_path
        self._start = int(start)
        self._end = int(end)
        self._bp = bayer_pattern
        self._store_raw = bool(store_raw_mosaic_if_forced)
        self._progress_every = int(progress_every)

        self._cancel_evt = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_evt.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            from setiastro.saspro.imageops.serloader import export_trimmed_to_ser

            # Progress callback executed from worker thread.
            # It emits a Qt signal (thread-safe), and checks cancel.
            def _cb(done: int, total: int) -> None:
                if self._cancel_evt.is_set():
                    # Abort export ASAP (exporter swallows callback errors,
                    # so we also raise a hard exception to stop loops).
                    raise RuntimeError("CANCELLED_BY_USER")
                self.progress.emit(int(done), int(total))

            export_trimmed_to_ser(
                self._src,
                self._out_path,
                self._start,
                self._end,
                bayer_pattern=self._bp,
                store_raw_mosaic_if_forced=self._store_raw,
                progress_cb=_cb,
                progress_every=self._progress_every,
            )

            # Ensure UI sees final state even if last callback was throttled
            self.progress.emit(int(self._end - self._start + 1), int(self._end - self._start + 1))
            self.finished.emit(self._out_path)

        except Exception as e:
            msg = str(e) if e is not None else "Unknown error"
            if "CANCELLED_BY_USER" in msg:
                self.canceled.emit()
            else:
                self.failed.emit(msg)


class SERViewer(QDialog):
    """
    Minimal SER viewer:
    - Open SER
    - Slider to scrub frames
    - Play/pause
    - ROI controls (x,y,w,h + enable)
    - Debayer toggle (for Bayer SER)
    - Linked autostretch toggle (preview only)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Planetary Stacker Viewer")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self._panning = False
        self._pan_start_pos = None          # QPoint in viewport coords
        self._pan_start_h = 0
        self._pan_start_v = 0
        self.reader: PlanetaryFrameSource | None = None
        self._cur = 0
        self._playing = False
        self._roi_dragging = False
        self._roi_start = None       # QPoint (viewport coords)
        self._roi_end = None         # QPoint (viewport coords)
        self._rubber = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps scrub/play
        self._timer.timeout.connect(self._tick_playback)
        self._drag_mode = None  # None / "roi" / "anchor"
        self._surface_anchor = None  # (x,y,w,h) in ROI-space
        self._source_spec = None  # str or list[str]
        self._zoom = 1.0
        self._fit_mode = True
        self._last_qimg: QImage | None = None
        self._last_disp_arr: np.ndarray | None = None   # the float [0..1] image we displayed (after stretch + tone)
        self._last_overlay = None                       # dict with overlay info for _render_last()

        self._build_ui()


    # ---------------- UI ----------------

    def _build_ui(self):
        # Root: left (viewer) + right (controls)
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---------- LEFT: playback + scrubber + preview + zoom ----------
        left = QVBoxLayout()
        left.setSpacing(8)
        root.addLayout(left, 1)

        # Top controls (left)
        top = QHBoxLayout()
        self.btn_open = QPushButton("Open SER/AVI/Frames…", self)
        self.btn_play = QPushButton("Play", self)
        self.btn_play.setEnabled(False)

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_play)
        top.addStretch(1)
        left.addLayout(top)

        self.lbl_info = QLabel("No SER loaded.", self)
        self.lbl_info.setStyleSheet("color:#888;")
        self.lbl_info.setWordWrap(True)
        left.addWidget(self.lbl_info)

        # Scrubber (left)
        scrub = QHBoxLayout()
        self.sld = QSlider(Qt.Orientation.Horizontal, self)
        self.sld.setRange(0, 0)
        self.sld.setEnabled(False)
        self.lbl_frame = QLabel("0 / 0", self)
        scrub.addWidget(self.sld, 1)
        scrub.addWidget(self.lbl_frame, 0)
        left.addLayout(scrub)

        # Trim Options (right)
        trim = QGroupBox("Trim", self)
        tform = QFormLayout(trim)

        self.spin_trim_start = QSpinBox(self)
        self.spin_trim_end = QSpinBox(self)
        self.spin_trim_start.setRange(0, 0)
        self.spin_trim_end.setRange(0, 0)

        self.btn_save_trimmed = QPushButton("Save Trimmed SER…", self)
        self.btn_save_trimmed.setEnabled(False)

        tform.addRow("Start frame", self.spin_trim_start)
        tform.addRow("End frame", self.spin_trim_end)
        tform.addRow("", self.btn_save_trimmed)

        


        # Preview area (left)
        self.scroll = QScrollArea(self)
        # IMPORTANT: for sane zoom + scrollbars, do NOT let the scroll area auto-resize the widget
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.scroll.viewport().setMouseTracking(True)
        self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)

        # Rubber band for Shift+drag ROI (thick, bright green, always visible)
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.scroll.viewport())
        self._rubber.setStyleSheet(
            "QRubberBand {"
            "  border: 3px solid #00ff00;"
            "  background: rgba(0,255,0,30);"
            "}"
        )
        self._rubber.hide()

        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.scroll.setWidget(self.preview)
        left.addWidget(self.scroll, 1)

        # Zoom buttons (NOW under preview, centered)
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_1_1  = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit  = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addStretch(1)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_1_1, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        left.addLayout(zoom_row)

        # ---------- RIGHT: options + stacking ----------
        right = QVBoxLayout()
        right.setSpacing(8)
        root.addLayout(right, 0)

        # Preview Options (right)
        opts = QGroupBox("Preview Options", self)
        form = QFormLayout(opts)
        right.addWidget(trim, 0)
        self.chk_roi = QCheckBox("Use ROI (crop for preview)", self)

        self.chk_debayer = QCheckBox("Debayer (Bayer SER)", self)
        self.chk_debayer.setChecked(True)
        self.cmb_bayer = QComboBox(self)
        self.cmb_bayer.addItems(["AUTO", "RGGB", "GRBG", "GBRG", "BGGR"])
        self.cmb_bayer.setCurrentText("AUTO")  # ✅ default for raw mosaic AVI

        self.chk_autostretch = QCheckBox("Autostretch preview (linked)", self)
        self.chk_autostretch.setChecked(False)

        # ROI controls
        self.spin_x = QSpinBox(self); self.spin_x.setRange(0, 999999)
        self.spin_y = QSpinBox(self); self.spin_y.setRange(0, 999999)
        self.spin_w = QSpinBox(self); self.spin_w.setRange(1, 999999); self.spin_w.setValue(512)
        self.spin_h = QSpinBox(self); self.spin_h.setRange(1, 999999); self.spin_h.setValue(512)

        form.addRow("", self.chk_roi)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.addWidget(QLabel("x:", self)); row1.addWidget(self.spin_x)
        row1.addWidget(QLabel("y:", self)); row1.addWidget(self.spin_y)
        form.addRow("ROI origin", row1)

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(QLabel("w:", self)); row2.addWidget(self.spin_w)
        row2.addWidget(QLabel("h:", self)); row2.addWidget(self.spin_h)
        form.addRow("ROI size", row2)

        form.addRow("", self.chk_debayer)
        form.addRow("Bayer pattern", self.cmb_bayer)        
        form.addRow("", self.chk_autostretch)

        # --- Preview tone controls (DISPLAY ONLY) ---
        self.sld_brightness = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_brightness.setRange(-100, 100)   # maps to -0.25 .. +0.25
        self.sld_brightness.setValue(0)
        self.sld_brightness.setToolTip("Preview brightness (display only)")

        self.sld_gamma = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_gamma.setRange(30, 300)          # 0.30 .. 3.00
        self.sld_gamma.setValue(100)              # 1.00
        self.sld_gamma.setToolTip("Preview gamma (display only)")

        form.addRow("Brightness", self.sld_brightness)
        form.addRow("Gamma", self.sld_gamma)

        right.addWidget(opts, 0)

        # Stacking Options (right)
        stack = QGroupBox("Stacking Options", self)
        sform = QFormLayout(stack)

        self.cmb_track = QComboBox(self)
        self.cmb_track.addItems(["Planetary", "Surface", "Off"])   # map to config
        self.cmb_track.setCurrentText("Planetary")

        self.spin_keep = QDoubleSpinBox(self)
        self.spin_keep.setRange(0.1, 100.0)
        self.spin_keep.setDecimals(1)
        self.spin_keep.setSingleStep(1.0)
        self.spin_keep.setValue(20.0)

        self.lbl_anchor = QLabel("Surface anchor: (not set)", self)
        self.lbl_anchor.setStyleSheet("color:#888;")
        self.lbl_anchor.setWordWrap(True)
        self.lbl_anchor.setToolTip(
            "Surface tracking needs an anchor patch.\n"
            "Ctrl+Shift+drag to define it (within ROI)."
        )

        self.btn_stack = QPushButton("Open Stacker…", self)
        self.btn_stack.setEnabled(False)   # enabled once SER loaded

        sform.addRow("Tracking", self.cmb_track)
        sform.addRow("Keep %", self.spin_keep)
        sform.addRow("", self.lbl_anchor)
        sform.addRow("", self.btn_stack)

        right.addWidget(stack, 0)

        right.addStretch(1)

        # Keep the right panel from getting too wide
        for gb in (opts, stack):
            gb.setMinimumWidth(360)

        # ---------- Signals ----------
        self.btn_open.clicked.connect(self._open_source)
        self.btn_play.clicked.connect(self._toggle_play)
        self.sld.valueChanged.connect(self._on_slider_changed)

        self.btn_zoom_out.clicked.connect(lambda: self._zoom_step(1/1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_step(1.25))
        self.btn_zoom_1_1.clicked.connect(lambda: self._set_zoom(1.0, anchor=self._viewport_center_anchor()))
        self.btn_zoom_fit.clicked.connect(self._set_fit_mode)

        for w in (self.chk_roi, self.chk_debayer, self.chk_autostretch,
                self.spin_x, self.spin_y, self.spin_w, self.spin_h,
                self.sld_brightness, self.sld_gamma):
            if hasattr(w, "toggled"):
                w.toggled.connect(self._refresh)
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._refresh)

        self.cmb_track.currentIndexChanged.connect(self._on_track_mode_changed)
        self.btn_stack.clicked.connect(self._open_stacker_clicked)
        self.cmb_bayer.currentIndexChanged.connect(self._refresh)
        self.chk_debayer.toggled.connect(lambda v: self.cmb_bayer.setEnabled(bool(v)))
        self.cmb_bayer.setEnabled(self.chk_debayer.isChecked())
        self.spin_trim_start.valueChanged.connect(self._on_trim_changed)
        self.spin_trim_end.valueChanged.connect(self._on_trim_changed)
        self.btn_save_trimmed.clicked.connect(self._save_trimmed_ser)

        self.resize(1200, 800)


    #-----qsettings
    def _settings(self) -> QSettings:
        # Prefer app-wide QSettings if your main window provides it
        if hasattr(self.parent(), "settings"):
            s = getattr(self.parent(), "settings")
            if isinstance(s, QSettings):
                return s
        # Fallback: app-global QSettings (uses org/app set in main())
        return QSettings()

    def _last_open_dir(self) -> str:
        s = self._settings()
        return s.value("serviewer/last_open_dir", "", type=str) or ""

    def _set_last_open_dir(self, path: str) -> None:
        try:
            d = os.path.dirname(os.path.abspath(path))
        except Exception:
            return
        if d:
            s = self._settings()
            s.setValue("serviewer/last_open_dir", d)


    # ---------------- actions ----------------

    def _apply_preview_tone(self, img: np.ndarray) -> np.ndarray:
        """
        Preview-only brightness + gamma.
        - brightness: adds offset in [-0.25..+0.25]
        - gamma: power curve in [0.30..3.00] (1.0 = no change)
        Works on mono or RGB float32 [0..1].
        """
        if img is None:
            return img

        # Brightness: -100..100 -> -0.25..+0.25
        b = float(self.sld_brightness.value()) / 100.0 * 0.25

        # Gamma: 30..300 -> 0.30..3.00
        g = float(self.sld_gamma.value()) / 100.0
        if g <= 0:
            g = 1.0

        out = img

        if abs(b) > 1e-6:
            out = np.clip(out + b, 0.0, 1.0)

        if abs(g - 1.0) > 1e-6:
            # gamma > 1 darkens, gamma < 1 brightens
            out = np.clip(np.power(np.clip(out, 0.0, 1.0), g), 0.0, 1.0)

        return out

    def _viewport_center_anchor(self):
        vp = self.scroll.viewport()
        return vp.rect().center()

    def _mouse_anchor(self):
        # Anchor zoom to mouse position if mouse is over the viewport, else center.
        vp = self.scroll.viewport()
        p = vp.mapFromGlobal(self.cursor().pos())
        if vp.rect().contains(p):
            return p
        return vp.rect().center()

    def _set_fit_mode(self):
        self._fit_mode = True
        self._render_last()  # rerender in fit mode

    def _set_zoom(self, z: float, anchor=None):
        self._fit_mode = False
        self._zoom = float(max(0.05, min(20.0, z)))
        self._render_last(anchor=anchor)

    def _zoom_step(self, factor: float):
        # Anchor zoom to mouse
        anchor = self._mouse_anchor()

        # If coming from fit, start from the fit zoom (prevents snapping)
        self._ensure_manual_zoom_from_fit()

        self._set_zoom(self._zoom * factor, anchor=anchor)

    def _fit_zoom_factor(self) -> float:
        """
        If we are in fit mode and a pixmap is displayed, return the effective zoom
        relative to the *original* frame size. This is what the user is visually seeing.
        """
        if self._last_qimg is None:
            return 1.0

        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return 1.0

        ow = max(1, self._last_qimg.width())
        oh = max(1, self._last_qimg.height())
        fw = max(1, pm.width())
        fh = max(1, pm.height())

        # KeepAspectRatio means either width or height matches; take the smaller ratio to be safe.
        return min(fw / ow, fh / oh)

    def _ensure_manual_zoom_from_fit(self):
        """
        If we are currently in fit mode, switch to manual zoom using the current
        effective fit zoom as the starting point (prevents snapping to ~1:1).
        """
        if self._fit_mode:
            self._zoom = self._fit_zoom_factor()
            self._fit_mode = False

    def _roi_rect_vp(self):
        """ROI QRect in viewport coords from start/end points."""
        if self._roi_start is None or self._roi_end is None:
            return None
        x1, y1 = self._roi_start.x(), self._roi_start.y()
        x2, y2 = self._roi_end.x(), self._roi_end.y()
        left, right = (x1, x2) if x1 <= x2 else (x2, x1)
        top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
        # enforce minimum size
        if (right - left) < 4 or (bottom - top) < 4:
            return None
        from PyQt6.QtCore import QRect
        return QRect(left, top, right - left, bottom - top)

    def _viewport_rect_to_display_image(self, r_vp):
        """
        Convert a viewport QRect (rubberband geometry) into coords in the CURRENT DISPLAYED IMAGE.
        That image is exactly self._last_qimg (ROI-sized if ROI is enabled).
        Returns (x,y,w,h) in _last_qimg pixel space.
        """
        if self._last_qimg is None:
            return None
        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return None

        # preview widget top-left inside viewport coords
        wp = self.preview.pos()
        lbl_left = int(wp.x())
        lbl_top  = int(wp.y())

        lbl_w = int(self.preview.width())
        lbl_h = int(self.preview.height())
        if lbl_w < 2 or lbl_h < 2:
            return None

        # rect corners in preview-widget coords
        x1 = int(r_vp.left()   - lbl_left)
        y1 = int(r_vp.top()    - lbl_top)
        x2 = int(r_vp.right()  - lbl_left)
        y2 = int(r_vp.bottom() - lbl_top)

        # clamp to widget bounds
        x1 = max(0, min(lbl_w - 1, x1))
        y1 = max(0, min(lbl_h - 1, y1))
        x2 = max(0, min(lbl_w - 1, x2))
        y2 = max(0, min(lbl_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        # map widget coords -> displayed image coords (_last_qimg space)
        ow = max(1, self._last_qimg.width())
        oh = max(1, self._last_qimg.height())

        scale_x = ow / float(lbl_w)
        scale_y = oh / float(lbl_h)

        ix = int(round(x1 * scale_x))
        iy = int(round(y1 * scale_y))
        iw = int(round((x2 - x1) * scale_x))
        ih = int(round((y2 - y1) * scale_y))

        # clamp to image bounds
        ix = max(0, min(ow - 1, ix))
        iy = max(0, min(oh - 1, iy))
        iw = max(1, min(ow - ix, iw))
        ih = max(1, min(oh - iy, ih))

        return (ix, iy, iw, ih)


    def _viewport_rect_to_image_roi(self, r_vp):
        """
        Convert a viewport-rect (viewport coords) into an ROI in IMAGE coords:
        returns (x,y,w,h) in original frame pixel space.
        Works in both fit mode and manual zoom mode, with scrollbars and centering.
        """
        if self._last_qimg is None:
            return None
        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return None

        # Where the preview widget actually is inside the viewport (accounts for scroll + centering)
        wp = self.preview.pos()  # QPoint in viewport coords
        lbl_left = int(wp.x())
        lbl_top  = int(wp.y())

        lbl_w = int(self.preview.width())
        lbl_h = int(self.preview.height())
        if lbl_w < 2 or lbl_h < 2:
            return None

        # ROI corners in widget coords
        x1 = int(r_vp.left()   - lbl_left)
        y1 = int(r_vp.top()    - lbl_top)
        x2 = int(r_vp.right()  - lbl_left)
        y2 = int(r_vp.bottom() - lbl_top)

        # Clamp to widget bounds
        x1 = max(0, min(lbl_w - 1, x1))
        y1 = max(0, min(lbl_h - 1, y1))
        x2 = max(0, min(lbl_w - 1, x2))
        y2 = max(0, min(lbl_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Map widget coords -> original image coords
        ow = max(1, self._last_qimg.width())
        oh = max(1, self._last_qimg.height())

        scale_x = ow / float(lbl_w)
        scale_y = oh / float(lbl_h)

        ix = int(round(x1 * scale_x))
        iy = int(round(y1 * scale_y))
        iw = int(round((x2 - x1) * scale_x))
        ih = int(round((y2 - y1) * scale_y))

        # clamp to image bounds
        ix = max(0, min(ow - 1, ix))
        iy = max(0, min(oh - 1, iy))
        iw = max(1, min(ow - ix, iw))
        ih = max(1, min(oh - iy, ih))

        return (ix, iy, iw, ih)

    def _update_anchor_label(self):
        a = getattr(self, "_surface_anchor", None)
        if a is None:
            self.lbl_anchor.setText("Surface anchor: (not set)  •  Ctrl+Shift+drag to set")
            self.lbl_anchor.setStyleSheet("color:#888;")
        else:
            x, y, w, h = a
            self.lbl_anchor.setText(f"Surface anchor: x={x}, y={y}, w={w}, h={h}  •  Ctrl+Shift+drag to change")
            self.lbl_anchor.setStyleSheet("color:#4a4;")

    def _on_track_mode_changed(self):
        mode = self._track_mode_value()

        # ✅ always reflect current anchor state
        self._update_anchor_label()

        if mode == "surface" and self._surface_anchor is None:
            self.lbl_anchor.setText("Surface anchor: REQUIRED  •  Ctrl+Shift+drag to set")
            self.lbl_anchor.setStyleSheet("color:#c66;")

        self._refresh()
    

    def _track_mode_value(self) -> str:
        t = self.cmb_track.currentText().strip().lower()
        if t.startswith("planet"):
            return "planetary"
        if t.startswith("surface"):
            return "surface"
        return "off"


    def eventFilter(self, obj, event):
        vp = self.scroll.viewport()
        try:
            if obj is vp:
                et = event.type()

                # ---- Ctrl+Wheel zoom ----
                if et == QEvent.Type.Wheel:
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        dy = event.angleDelta().y()
                        if dy != 0:
                            factor = 1.25 if dy > 0 else (1 / 1.25)
                            anchor = event.position().toPoint()  # viewport coords
                            self._ensure_manual_zoom_from_fit()
                            self._set_zoom(self._zoom * factor, anchor=anchor)
                        event.accept()
                        return True
                    return False

                # ---- Left-drag pan and ROI ----
                if et == QEvent.Type.MouseButtonPress:
                    if event.button() == Qt.MouseButton.LeftButton:
                        mods = event.modifiers()

                        is_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
                        is_ctrl  = bool(mods & Qt.KeyboardModifier.ControlModifier)

                        if is_shift:
                            # Shift+Drag = ROI, Ctrl+Shift+Drag = Anchor
                            self._roi_dragging = True
                            self._roi_start = event.position().toPoint()
                            self._drag_mode = "anchor" if is_ctrl else "roi"

                            if self._rubber is not None:
                                self._rubber.setGeometry(QRect(self._roi_start, QSize(1, 1)))
                                self._rubber.show()
                                self._rubber.raise_()

                                # Optional: different color for anchor
                                if self._drag_mode == "anchor":
                                    self._rubber.setStyleSheet(
                                        "QRubberBand { border: 3px solid #00aaff; background: rgba(0,170,255,30); }"
                                    )
                                else:
                                    self._rubber.setStyleSheet(
                                        "QRubberBand { border: 3px solid #00ff00; background: rgba(0,255,0,30); }"
                                    )

                            vp.setCursor(Qt.CursorShape.CrossCursor)
                            event.accept()
                            return True

                        # Normal left-drag pan
                        self._panning = True
                        self._pan_start_pos = event.position().toPoint()
                        self._pan_start_h = self.scroll.horizontalScrollBar().value()
                        self._pan_start_v = self.scroll.verticalScrollBar().value()
                        vp.setCursor(Qt.CursorShape.ClosedHandCursor)
                        event.accept()
                        return True
                if et == QEvent.Type.MouseMove:
                    if self._roi_dragging and self._roi_start is not None:
                        cur = event.position().toPoint()
                        if self._rubber is not None:
                            self._rubber.setGeometry(QRect(self._roi_start, cur).normalized())
                            self._rubber.raise_()
                        event.accept()
                        return True

                    if self._panning and self._pan_start_pos is not None:
                        cur = event.position().toPoint()
                        delta = cur - self._pan_start_pos
                        hbar = self.scroll.horizontalScrollBar()
                        vbar = self.scroll.verticalScrollBar()
                        hbar.setValue(self._pan_start_h - delta.x())
                        vbar.setValue(self._pan_start_v - delta.y())
                        event.accept()
                        return True
                    
                if et == QEvent.Type.MouseButtonRelease:
                    if event.button() == Qt.MouseButton.LeftButton:

                        # --- finish ROI/anchor rubberband drag ---
                        if self._roi_dragging:
                            self._roi_dragging = False
                            vp.setCursor(Qt.CursorShape.ArrowCursor)

                            r_vp = None
                            if self._rubber is not None:
                                r_vp = self._rubber.geometry()
                                self._rubber.hide()

                            self._roi_start = None

                            if r_vp is not None and r_vp.width() >= 4 and r_vp.height() >= 4:
                                rect_disp = self._viewport_rect_to_display_image(r_vp)  # coords in _last_qimg space (ROI-sized if ROI enabled)
                                if rect_disp is not None:
                                    if self._drag_mode == "roi":
                                        # If ROI is already enabled, the displayed image is ROI-space.
                                        # The user is drawing a NEW ROI inside that ROI -> convert to full-frame.
                                        if self.chk_roi.isChecked():
                                            rx, ry, rw, rh = self._roi_bounds()
                                            x, y, w, h = rect_disp
                                            x_full = int(rx + x)
                                            y_full = int(ry + y)
                                            self.spin_x.setValue(x_full)
                                            self.spin_y.setValue(y_full)
                                            self.spin_w.setValue(int(w))
                                            self.spin_h.setValue(int(h))
                                        else:
                                            x, y, w, h = rect_disp
                                            self.spin_x.setValue(int(x))
                                            self.spin_y.setValue(int(y))
                                            self.spin_w.setValue(int(w))
                                            self.spin_h.setValue(int(h))

                                        self.chk_roi.setChecked(True)
                                        self._refresh()

                                    elif self._drag_mode == "anchor":
                                        # Anchor is ALWAYS stored in ROI-space.
                                        # When ROI is enabled, displayed image == ROI-space, so rect_disp is already correct.
                                        # When ROI is disabled, ROI-space == full-frame, so rect_disp is still correct.
                                        self._surface_anchor = tuple(int(v) for v in rect_disp)
                                        self._update_anchor_label()
                                        self._render_last()

                            self._drag_mode = None
                            event.accept()
                            return True

                        # --- finish panning ---
                        if self._panning:
                            self._panning = False
                            self._pan_start_pos = None
                            vp.setCursor(Qt.CursorShape.ArrowCursor)
                            event.accept()
                            return True


                if et == QEvent.Type.Leave:
                    if self._panning or self._roi_dragging:
                        self._panning = False
                        self._roi_dragging = False
                        self._pan_start_pos = None
                        self._roi_start = None
                        if self._rubber is not None:
                            self._rubber.hide()
                            self._drag_mode = None
                        vp.setCursor(Qt.CursorShape.ArrowCursor)
                        event.accept()
                        return True

        except Exception:
            pass

        return super().eventFilter(obj, event)

    def _open_stacker_clicked(self):
        if self.reader is None:
            return

        source = self.get_source_spec()
        if not source:
            return

        # Only meaningful for single-file sources; OK to pass None for sequences
        ser_path = source if isinstance(source, str) else None

        roi = self.get_roi()
        anchor = self.get_surface_anchor()

        main = self.parent() or self
        current_doc = None
        try:
            if hasattr(main, "active_document"):
                current_doc = main.active_document()
            elif hasattr(main, "currentDocument"):
                current_doc = main.currentDocument()
            elif hasattr(main, "docman") and hasattr(main.docman, "current_document"):
                current_doc = main.docman.current_document()
            elif hasattr(main, "docman") and hasattr(main.docman, "current"):
                current_doc = main.docman.current()
        except Exception:
            current_doc = None

        debayer = bool(self.chk_debayer.isChecked())

        # Normalize: "AUTO" means "let the loader decide"
        bp = self.cmb_bayer.currentText().strip().upper()
        if not debayer or bp == "AUTO":
            bp = None

        dlg = SERStackerDialog(
            parent=self,
            main=main,
            source_doc=current_doc,
            ser_path=ser_path,
            source=source,
            roi=roi,
            track_mode=self._track_mode_value(),
            surface_anchor=anchor,
            debayer=debayer,
            bayer_pattern=bp,                 # ✅ THIS IS THE FIX
            keep_percent=float(self.spin_keep.value()),
        )


        dlg.stackProduced.connect(self._on_stacker_produced)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_stacker_produced(self, out: np.ndarray, diag: dict):
        """
        Viewer should NOT decide “save vs new view” long-term.
        For now, we just hand it to the parent/main if it supports a hook.
        """
        # Try common patterns without hard dependency:
        # - main window has doc manager: main.push_array_to_new_view(...)
        # - or a generic method: main.open_image_from_array(...)
        main = self.parent()

        # 1) Example hook you can implement in MainWindow/DocManager:
        if main is not None and hasattr(main, "push_array_to_new_view"):
            try:
                title = f"Stacked SER ({diag.get('frames_kept')}/{diag.get('frames_total')})"
                main.push_array_to_new_view(out, title=title, meta={"ser_diag": diag})
                return
            except Exception:
                pass

        # 2) Fallback: show it in this viewer preview (temporary)
        try:
            qimg = self._to_qimage(out)
            self._last_qimg = qimg
            self._fit_mode = True
            self._render_last()
            self.lbl_info.setText(
                self.lbl_info.text()
                + f"<br><b>Stacked (from stacker):</b> kept {diag.get('frames_kept')} / {diag.get('frames_total')}"
            )
        except Exception:
            pass

    def _compute_planet_com_px(self, img01: np.ndarray) -> tuple[float, float] | None:
        """
        Compute a quick center-of-mass in *image pixel coords* of the currently displayed image (ROI-space).
        Uses a simple brightness-weighted COM with background subtraction.
        """
        try:
            if img01 is None:
                return None
            if img01.ndim == 3:
                # simple luma (no extra deps)
                m = 0.2126 * img01[..., 0] + 0.7152 * img01[..., 1] + 0.0722 * img01[..., 2]
            else:
                m = img01

            m = np.asarray(m, dtype=np.float32)
            H, W = m.shape[:2]
            if H < 2 or W < 2:
                return None

            # Robust-ish background subtraction to focus on the planet
            bg = float(np.percentile(m, 60))  # helps ignore dark background
            w = np.clip(m - bg, 0.0, None)

            s = float(w.sum())
            if s <= 1e-8:
                return None

            ys = np.arange(H, dtype=np.float32)[:, None]
            xs = np.arange(W, dtype=np.float32)[None, :]
            cy = float((w * ys).sum() / s)
            cx = float((w * xs).sum() / s)
            return (cx, cy)
        except Exception:
            return None


    def _img_xy_to_pixmap_xy(self, x: float, y: float) -> tuple[int, int] | None:
        """
        Map a point in ORIGINAL IMAGE pixel coords (of _last_qimg) into current pixmap coords.
        In this viewer, pixmap size == preview label size in both fit and manual modes.
        """
        if self._last_qimg is None:
            return None
        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return None

        ow = max(1, self._last_qimg.width())
        oh = max(1, self._last_qimg.height())
        pw = max(1, pm.width())
        ph = max(1, pm.height())

        px = int(round((x / ow) * pw))
        py = int(round((y / oh) * ph))
        return (px, py)


    def _roi_rect_to_pixmap_rect(self, rect_roi: tuple[int, int, int, int]) -> QRect | None:
        """
        rect_roi is ROI-space (0..roi_w,0..roi_h) but the displayed image is also ROI-sized
        whenever ROI checkbox is ON (because get_frame(roi=roi) crops).
        So ROI-space == displayed image pixel space. Great.
        """
        if rect_roi is None:
            return None

        x, y, w, h = [int(v) for v in rect_roi]
        p1 = self._img_xy_to_pixmap_xy(x, y)
        p2 = self._img_xy_to_pixmap_xy(x + w, y + h)
        if p1 is None or p2 is None:
            return None
        x1, y1 = p1
        x2, y2 = p2
        left, right = (x1, x2) if x1 <= x2 else (x2, x1)
        top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
        return QRect(left, top, max(1, right - left), max(1, bottom - top))


    def _paint_overlays_on_current_pixmap(self):
        """
        Draw overlays (COM crosshair and/or anchor rectangle) onto the CURRENT pixmap.
        Call this at the end of _render_last().
        """
        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return
        if self._last_disp_arr is None:
            return

        mode = self._track_mode_value()

        # Make a paintable copy
        pm2 = pm.copy()
        p = QPainter(pm2)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # --- Surface anchor box overlay ---
        if mode == "surface" and self._surface_anchor is not None:
            r = self._roi_rect_to_pixmap_rect(self._surface_anchor)
            if r is not None:
                pen = QPen(QColor(0, 170, 255))
                pen.setWidth(3)
                p.setPen(pen)
                p.setBrush(QColor(0, 170, 255, 30))
                p.drawRect(r)

        # --- Planetary COM crosshair overlay ---
        if mode == "planetary":
            com = self._compute_planet_com_px(self._last_disp_arr)
            if com is not None:
                cx, cy = com
                qpt = self._img_xy_to_pixmap_xy(cx, cy)
                if qpt is not None:
                    px, py = qpt

                    pen = QPen(QColor(255, 220, 0))  # bright yellow
                    pen.setWidth(3)
                    p.setPen(pen)

                    # crosshair size in pixmap pixels (constant visibility)
                    r = 10
                    p.drawLine(px - r, py, px + r, py)
                    p.drawLine(px, py - r, px, py + r)

                    # small center dot
                    p.setBrush(QColor(255, 220, 0))
                    p.drawEllipse(px - 2, py - 2, 4, 4)

        p.end()

        self.preview.setPixmap(pm2)


    def _render_last(self, anchor=None):
        if self._last_qimg is None:
            return

        pm = QPixmap.fromImage(self._last_qimg)
        if pm.isNull():
            return

        if self._fit_mode:
            # Fit: scale pixmap to viewport, and size the label to the scaled pixmap.
            vp = self.scroll.viewport().size()
            if vp.width() < 5 or vp.height() < 5:
                return
            pm2 = pm.scaled(vp, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pm2)
            self.preview.resize(pm2.size())     # in fit mode, label == pixmap size
            self._paint_overlays_on_current_pixmap()
            return

        # Manual zoom: label becomes the scaled size so scrollbars are correct/stable
        w = max(1, int(pm.width() * self._zoom))
        h = max(1, int(pm.height() * self._zoom))
        pm2 = pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)

        # Preserve current view position (anchor before/after)
        # Preserve current view position (anchor before/after)
        if anchor is None:
            anchor = self._viewport_center_anchor()

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # old content size
        old_w = max(1, self.preview.width())
        old_h = max(1, self.preview.height())

        # anchor point in CONTENT coords before change
        ax = hbar.value() + anchor.x()
        ay = vbar.value() + anchor.y()

        # fractional anchor position in old content
        fx = ax / old_w
        fy = ay / old_h

        self.preview.setPixmap(pm2)
        self.preview.resize(pm2.size())

        # new content size
        new_w = max(1, self.preview.width())
        new_h = max(1, self.preview.height())

        # restore scrollbars so anchor stays put
        hbar.setValue(int(fx * new_w - anchor.x()))
        vbar.setValue(int(fy * new_h - anchor.y()))
        self._paint_overlays_on_current_pixmap()



    def _open_source(self):
        start_dir = self._last_open_dir()

        # Let user either:
        # - pick a single SER/AVI
        # - OR multi-select images for a sequence
        dlg = QFileDialog(self, "Open Planetary Frames")
        if start_dir:
            dlg.setDirectory(start_dir)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dlg.setNameFilters([
            "Planetary Sources (*.ser *.avi *.mp4 *.mov *.mkv *.png *.tif *.tiff *.jpg *.jpeg *.bmp *.webp)",
            "SER Videos (*.ser)",
            "AVI/Video (*.avi *.mp4 *.mov *.mkv)",
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp *.webp)",
            "All Files (*)",
        ])

        if not dlg.exec():
            return

        files = dlg.selectedFiles()
        if not files:
            return

        # Heuristic:
        # - If exactly one file and it's .ser/.avi/etc -> open as that
        # - If multiple files -> treat as image sequence (sorted)
        files = [os.fspath(f) for f in files]
        files_sorted = sorted(files, key=lambda p: os.path.basename(p).lower())
        self._source_spec = files_sorted[0] if len(files_sorted) == 1 else files_sorted

        try:
            if self.reader is not None:
                self.reader.close()
        except Exception:
            pass
        self.reader = None

        try:
            if len(files_sorted) == 1:
                src = open_planetary_source(files_sorted[0], cache_items=10)
                self._set_last_open_dir(files_sorted[0])
            else:
                src = open_planetary_source(files_sorted, cache_items=10)
                self._set_last_open_dir(files_sorted[0])

            self.reader = src

        except Exception as e:
            QMessageBox.critical(self, "SER Viewer", f"Failed to open:\n{e}")
            self.reader = None
            return

        m = self.reader.meta
        base = os.path.basename(m.path or (files_sorted[0] if files_sorted else ""))

        # Nice info string
        src_kind = getattr(m, "source_kind", "unknown")
        extra = ""
        if src_kind == "sequence":
            extra = f" • sequence={m.frames}"
        elif src_kind == "avi":
            extra = f" • video={m.frames}"
        elif src_kind == "ser":
            extra = f" • frames={m.frames}"
        else:
            extra = f" • frames={m.frames}"

        self.lbl_info.setText(
            f"<b>{base}</b><br>"
            f"{m.width}×{m.height}{extra} • depth={m.pixel_depth}-bit • format={m.color_name}"
            + (" • timestamps" if getattr(m, "has_timestamps", False) else "")
        )

        self._cur = 0
        self.sld.setEnabled(True)
        self.sld.setRange(0, max(0, m.frames - 1))
        self.sld.setValue(0)

        self.spin_trim_start.blockSignals(True)
        self.spin_trim_end.blockSignals(True)
        self.spin_trim_start.setRange(0, max(0, m.frames - 1))
        self.spin_trim_end.setRange(0, max(0, m.frames - 1))
        self.spin_trim_start.setValue(0)
        self.spin_trim_end.setValue(max(0, m.frames - 1))
        self.spin_trim_start.blockSignals(False)
        self.spin_trim_end.blockSignals(False)

        self.btn_save_trimmed.setEnabled(m.frames > 0)


        # Set ROI defaults to centered box
        cx = max(0, (m.width // 2) - 256)
        cy = max(0, (m.height // 2) - 256)
        self.spin_x.setValue(cx)
        self.spin_y.setValue(cy)
        self.spin_w.setValue(min(512, m.width))
        self.spin_h.setValue(min(512, m.height))

        # Debayer only makes sense for SER Bayer; but leaving enabled is fine (no-op elsewhere)
        self.btn_play.setEnabled(True)
        self.btn_stack.setEnabled(True)  # (see note below about stacker input)
        self._surface_anchor = None
        self._update_anchor_label()
        self.btn_play.setText("Play")
        self._playing = False

        self._refresh()


    def _toggle_play(self):
        if self.reader is None:
            return
        self._playing = not self._playing
        self.btn_play.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _tick_playback(self):
        if self.reader is None:
            return
        if self._cur >= self.reader.meta.frames - 1:
            self._cur = 0
        else:
            self._cur += 1
        self.sld.blockSignals(True)
        self.sld.setValue(self._cur)
        self.sld.blockSignals(False)
        self._refresh()

    def _on_slider_changed(self, v: int):
        self._cur = int(v)
        self._refresh()

    # ---------------- rendering ----------------

    def _roi_tuple(self):
        if not self.chk_roi.isChecked():
            return None
        return (int(self.spin_x.value()), int(self.spin_y.value()),
                int(self.spin_w.value()), int(self.spin_h.value()))

    def _on_trim_changed(self):
        if self.reader is None:
            return
        n = max(0, int(self.reader.meta.frames) - 1)
        a = int(self.spin_trim_start.value())
        b = int(self.spin_trim_end.value())
        a = max(0, min(n, a))
        b = max(0, min(n, b))
        if a > b:
            # keep it intuitive: clamp end to start
            b = a
            self.spin_trim_end.blockSignals(True)
            self.spin_trim_end.setValue(b)
            self.spin_trim_end.blockSignals(False)

    def _save_trimmed_ser(self):
        if self.reader is None:
            return

        start = int(self.spin_trim_start.value())
        end = int(self.spin_trim_end.value())
        if end < start:
            end = start

        src = self.get_source_spec()
        if isinstance(src, str) and src:
            base_dir = os.path.dirname(src)
            base_name = os.path.splitext(os.path.basename(src))[0]
        else:
            base_dir = self._last_open_dir() or os.getcwd()
            base_name = "trimmed"

        default_path = os.path.join(base_dir, f"{base_name}_trim_{start:05d}-{end:05d}.ser")

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trimmed SER",
            default_path,
            "SER Videos (*.ser)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".ser"):
            out_path += ".ser"

        # Use the user's current debayer selection to decide output format
        debayer = bool(self.chk_debayer.isChecked())
        bp = self.cmb_bayer.currentText().strip().upper()
        if (not debayer) or (bp == "AUTO"):
            bp = None  # means: don't force, export RGB
        else:
            # match serloader's accepted forms: "RGGB" -> "BAYER_RGGB" etc handled there
            bp = bp  # keep short name; serloader normalizes it

        total = int(end - start + 1)

        # Disable UI controls during export (prevents state changes mid-write)
        self.btn_save_trimmed.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_stack.setEnabled(False)

        # Progress dialog
        pd = QProgressDialog("Exporting trimmed SER…", "Cancel", 0, total, self)
        pd.setWindowTitle("Saving Trimmed SER")
        pd.setWindowModality(Qt.WindowModality.WindowModal)
        pd.setAutoClose(False)
        pd.setAutoReset(False)
        pd.setMinimumDuration(0)
        pd.setValue(0)
        pd.show()

        # Thread + worker
        thread = QThread(self)
        worker = _TrimExportWorker(
            self.reader,
            out_path,
            start,
            end,
            bayer_pattern=bp,
            store_raw_mosaic_if_forced=True,   # key: makes Bayer SER if bp is set
            progress_every=100,
        )
        worker.moveToThread(thread)

        # Keep references so they don't get GC'd
        self._trim_thread = thread
        self._trim_worker = worker
        self._trim_progress = pd

        # Cancel hook
        def _on_cancel():
            try:
                worker.request_cancel()
                pd.setLabelText("Canceling… (finishing current frame)")
                pd.setCancelButtonText("Canceling…")
                pd.setEnabled(False)  # prevents repeated clicks
            except Exception:
                pass
        pd.canceled.connect(_on_cancel)

        # Progress updates (runs on GUI thread)
        @pyqtSlot(int, int)
        def _on_progress(done: int, tot: int):
            try:
                pd.setMaximum(int(tot))
                pd.setValue(int(done))
                pd.setLabelText(f"Exporting trimmed SER… {done}/{tot}")
            except Exception:
                pass
        worker.progress.connect(_on_progress)

        # Finish / fail / canceled cleanup
        def _cleanup_ui():
            try:
                pd.close()
            except Exception:
                pass
            self.btn_save_trimmed.setEnabled(True)
            self.btn_open.setEnabled(True)
            self.btn_play.setEnabled(self.reader is not None)
            self.btn_stack.setEnabled(self.reader is not None)

            # release refs
            self._trim_progress = None
            self._trim_worker = None
            self._trim_thread = None

        @pyqtSlot(str)
        def _on_finished(path: str):
            try:
                _cleanup_ui()
                QMessageBox.information(
                    self,
                    "Trim",
                    f"Saved trimmed SER:\n{path}\n\nFrames: {start}..{end} ({total})"
                )
            finally:
                try:
                    thread.quit()
                    thread.wait(2000)
                except Exception:
                    pass

        @pyqtSlot(str)
        def _on_failed(err: str):
            try:
                _cleanup_ui()
                QMessageBox.critical(self, "Trim", f"Failed to save trimmed SER:\n{err}")
            finally:
                try:
                    thread.quit()
                    thread.wait(2000)
                except Exception:
                    pass

        @pyqtSlot()
        def _on_canceled():
            try:
                _cleanup_ui()
                QMessageBox.information(self, "Trim", "Export canceled.")
            finally:
                try:
                    thread.quit()
                    thread.wait(2000)
                except Exception:
                    pass

        worker.finished.connect(_on_finished)
        worker.failed.connect(_on_failed)
        worker.canceled.connect(_on_canceled)

        thread.started.connect(worker.run)
        thread.start()


    def _refresh(self):
        if self.reader is None:
            return

        m = self.reader.meta
        self.lbl_frame.setText(f"{self._cur+1} / {m.frames}")

        roi = self._roi_tuple()
        debayer = bool(self.chk_debayer.isChecked())

        try:
            img = self.reader.get_frame(
                self._cur,
                roi=roi,
                debayer=debayer,
                to_float01=True,
                force_rgb=False,
                bayer_pattern=self.cmb_bayer.currentText(),  # ✅ NEW
            )
        except Exception as e:
            QMessageBox.warning(self, "SER Viewer", f"Frame read failed:\n{e}")
            return

        # Autostretch preview (linked)
        if self.chk_autostretch.isChecked():
            try:
                if img.ndim == 2 and stretch_mono_image is not None:
                    img = np.clip(stretch_mono_image(img, target_median=0.25), 0.0, 1.0)
                elif img.ndim == 3 and img.shape[2] == 3 and stretch_color_image is not None:
                    # linked=True for planetary preview (you requested this)
                    img = np.clip(stretch_color_image(img, target_median=0.25, linked=True), 0.0, 1.0)
            except Exception:
                # if stretch fails, fall back to raw preview
                pass

        try:
            img = self._apply_preview_tone(img)
        except Exception:
            pass

        # store for overlay calculations (ROI-sized if ROI is on)
        self._last_disp_arr = img

        qimg = self._to_qimage(img)
        self._last_qimg = qimg
        self._render_last(anchor=self._viewport_center_anchor() if not self._fit_mode else None)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_qimg is None:
            return
        if self._fit_mode:
            self._render_last()

    def _to_qimage(self, arr: np.ndarray) -> QImage:
        a = np.clip(arr, 0.0, 1.0)
        if a.ndim == 2:
            u = (a * 255.0).astype(np.uint8)
            h, w = u.shape
            return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

        if a.ndim == 3 and a.shape[2] >= 3:
            u = (a[..., :3] * 255.0).astype(np.uint8)
            h, w, _ = u.shape
            return QImage(u.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

        raise ValueError(f"Unexpected image shape: {a.shape}")

    def _roi_bounds(self):
        """
        Returns (rx, ry, rw, rh) in full-frame coords if ROI enabled,
        else (0,0, full_w, full_h) if we can infer it.
        """
        if self.reader is None:
            return (0, 0, 0, 0)

        if self.chk_roi.isChecked():
            return (int(self.spin_x.value()), int(self.spin_y.value()),
                    int(self.spin_w.value()), int(self.spin_h.value()))
        # ROI disabled: treat whole frame as ROI
        m = self.reader.meta
        return (0, 0, int(m.width), int(m.height))


    def _full_to_roi_space(self, rect_full):
        """
        rect_full: (x,y,w,h) in full-frame coords
        returns: (x,y,w,h) in ROI-space (0..rw,0..rh)
        """
        if rect_full is None:
            return None

        fx, fy, fw, fh = rect_full
        rx, ry, rw, rh = self._roi_bounds()

        # convert full -> roi space
        x = fx - rx
        y = fy - ry
        w = fw
        h = fh

        # clamp to ROI-space bounds
        x = max(0, min(rw - 1, x))
        y = max(0, min(rh - 1, y))
        w = max(1, min(rw - x, w))
        h = max(1, min(rh - y, h))
        return (int(x), int(y), int(w), int(h))


    def get_source_path(self) -> str | None:
        return getattr(self.reader, "path", None) if self.reader is not None else None

    def get_roi(self):
        return self._roi_tuple()  # already returns (x,y,w,h) or None

    def get_surface_anchor(self):
        return getattr(self, "_surface_anchor", None)

    def get_source_spec(self):
        if self.reader is None:
            return None

        m = getattr(self.reader, "meta", None)
        if m is not None:
            # ✅ If this is an image sequence, use the full file list
            fl = getattr(m, "file_list", None)
            if isinstance(fl, (list, tuple)) and len(fl) > 0:
                return list(fl)

            # Otherwise fall back to the meta path (SER/AVI)
            p = getattr(m, "path", None)
            if isinstance(p, str) and p:
                return p

        # Fallback
        return getattr(self.reader, "path", None)

