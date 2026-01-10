# src/setiastro/saspro/serviewer.py
from __future__ import annotations

import os
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, QEvent, QPoint, QRect, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QScrollArea, QSlider, QCheckBox, QGroupBox, QFormLayout, QSpinBox,
    QMessageBox, QRubberBand
)

from setiastro.saspro.imageops.serloader import SERReader

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


# Use your stretch functions for DISPLAY
try:
    from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
except Exception:
    stretch_mono_image = None
    stretch_color_image = None


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
        self.setWindowTitle("SER Viewer")
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
        self.reader: SERReader | None = None
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
        self._zoom = 1.0
        self._fit_mode = True
        self._last_qimg: QImage | None = None

        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # Top controls
        top = QHBoxLayout()
        self.btn_open = QPushButton("Open SER…", self)
        self.btn_play = QPushButton("Play", self)
        self.btn_play.setEnabled(False)

        self.lbl_info = QLabel("No SER loaded.", self)
        self.lbl_info.setStyleSheet("color:#888;")
        self.lbl_info.setWordWrap(True)

        top.addWidget(self.btn_open)
        top.addWidget(self.btn_play)
        top.addStretch(1)
        outer.addLayout(top)

        outer.addWidget(self.lbl_info)

        # Scrubber
        scrub = QHBoxLayout()
        self.sld = QSlider(Qt.Orientation.Horizontal, self)
        self.sld.setRange(0, 0)
        self.sld.setEnabled(False)
        self.lbl_frame = QLabel("0 / 0", self)
        scrub.addWidget(self.sld, 1)
        scrub.addWidget(self.lbl_frame, 0)
        outer.addLayout(scrub)

        # Options panel
        opts = QGroupBox("Preview Options", self)
        form = QFormLayout(opts)

        self.chk_roi = QCheckBox("Use ROI (crop for preview)", self)
        self.chk_debayer = QCheckBox("Debayer (Bayer SER)", self)
        self.chk_debayer.setChecked(True)

        self.chk_autostretch = QCheckBox("Autostretch preview (linked)", self)
        self.chk_autostretch.setChecked(True)

        # ROI controls
        self.spin_x = QSpinBox(self); self.spin_x.setRange(0, 999999)
        self.spin_y = QSpinBox(self); self.spin_y.setRange(0, 999999)
        self.spin_w = QSpinBox(self); self.spin_w.setRange(1, 999999); self.spin_w.setValue(512)
        self.spin_h = QSpinBox(self); self.spin_h.setRange(1, 999999); self.spin_h.setValue(512)

        form.addRow("", self.chk_roi)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("x:", self)); row1.addWidget(self.spin_x)
        row1.addWidget(QLabel("y:", self)); row1.addWidget(self.spin_y)
        form.addRow("ROI origin", row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("w:", self)); row2.addWidget(self.spin_w)
        row2.addWidget(QLabel("h:", self)); row2.addWidget(self.spin_h)
        form.addRow("ROI size", row2)

        form.addRow("", self.chk_debayer)
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
        outer.addWidget(opts, 0)

        # Zoom buttons (use SASpro themed tool buttons like Statistical Stretch)
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_1_1  = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit  = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addStretch(1)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_1_1, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        outer.addLayout(zoom_row)


        # Preview area
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
        outer.addWidget(self.scroll, 1)

        # Signals
        self.btn_open.clicked.connect(self._open_ser)
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

        self.resize(1100, 800)

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


    def _viewport_rect_to_image_roi(self, r_vp):
        """
        Convert a viewport-rect (in viewport coords) into an ROI in IMAGE coords:
        returns (x,y,w,h) in original frame pixel space.
        Works in both fit mode and manual zoom mode, with centered label.
        """
        if self._last_qimg is None:
            return None
        pm = self.preview.pixmap()
        if pm is None or pm.isNull():
            return None

        # Label rect inside viewport (centered)
        vp = self.scroll.viewport()
        vp_w, vp_h = vp.width(), vp.height()

        lbl_w, lbl_h = self.preview.width(), self.preview.height()
        # Where the label sits inside the viewport (centered)
        lbl_left = (vp_w - lbl_w) // 2
        lbl_top  = (vp_h - lbl_h) // 2

        # ROI corners in label coords
        x1 = r_vp.left() - lbl_left
        y1 = r_vp.top() - lbl_top
        x2 = r_vp.right() - lbl_left
        y2 = r_vp.bottom() - lbl_top

        # Clamp to label bounds
        x1 = max(0, min(lbl_w - 1, x1))
        y1 = max(0, min(lbl_h - 1, y1))
        x2 = max(0, min(lbl_w - 1, x2))
        y2 = max(0, min(lbl_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Map label coords -> original image coords
        ow = max(1, self._last_qimg.width())
        oh = max(1, self._last_qimg.height())

        # label displays scaled pixmap; label size == pixmap size in both modes
        scale_x = ow / float(lbl_w)
        scale_y = oh / float(lbl_h)

        ix = int(round(x1 * scale_x))
        iy = int(round(y1 * scale_y))
        iw = int(round((x2 - x1) * scale_x))
        ih = int(round((y2 - y1) * scale_y))

        # clamp to image bounds, enforce >=1
        ix = max(0, min(ow - 1, ix))
        iy = max(0, min(oh - 1, iy))
        iw = max(1, min(ow - ix, iw))
        ih = max(1, min(oh - iy, ih))

        return (ix, iy, iw, ih)


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
                                rect_full = self._viewport_rect_to_image_roi(r_vp)  # FULL coords
                                if rect_full is not None:
                                    if self._drag_mode == "roi":
                                        x, y, w, h = rect_full
                                        self.chk_roi.setChecked(True)
                                        self.spin_x.setValue(x)
                                        self.spin_y.setValue(y)
                                        self.spin_w.setValue(w)
                                        self.spin_h.setValue(h)

                                        # ROI change invalidates anchor
                                        self._surface_anchor = None
                                        try:
                                            self._settings().setValue("serviewer/surface_anchor", None)
                                        except Exception:
                                            pass

                                        self._refresh()

                                    elif self._drag_mode == "anchor":
                                        anchor = self._full_to_roi_space(rect_full)
                                        if anchor is not None:
                                            self._surface_anchor = anchor
                                            try:
                                                self._settings().setValue("serviewer/surface_anchor", list(anchor))
                                            except Exception:
                                                pass

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



    def _open_ser(self):
        start_dir = self._last_open_dir()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open SER Video",
            start_dir if start_dir else "",
            "SER Videos (*.ser);;All Files (*)"
        )
        if not path:
            return

        try:
            if self.reader is not None:
                self.reader.close()
            self.reader = SERReader(path, cache_items=10)
        except Exception as e:
            QMessageBox.critical(self, "SER Viewer", f"Failed to open:\n{e}")
            self.reader = None
            return

        self._set_last_open_dir(path)

        m = self.reader.meta
        base = os.path.basename(path)
        self.lbl_info.setText(
            f"<b>{base}</b><br>"
            f"{m.width}×{m.height} • frames={m.frames} • depth={m.pixel_depth}-bit • format={m.color_name}"
            + (" • timestamps" if m.has_timestamps else "")
        )

        self._cur = 0
        self.sld.setEnabled(True)
        self.sld.setRange(0, max(0, m.frames - 1))
        self.sld.setValue(0)

        # Set ROI defaults to a centered box (nice for planets)
        cx = max(0, (m.width // 2) - 256)
        cy = max(0, (m.height // 2) - 256)
        self.spin_x.setValue(cx)
        self.spin_y.setValue(cy)
        self.spin_w.setValue(min(512, m.width))
        self.spin_h.setValue(min(512, m.height))
        # restore saved anchor if present
        try:
            v = self._settings().value("serviewer/surface_anchor", None)
            if isinstance(v, (list, tuple)) and len(v) == 4:
                self._surface_anchor = tuple(int(x) for x in v)
        except Exception:
            self._surface_anchor = None
        self.btn_play.setEnabled(True)
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
                to_float01=True,    # float32 [0..1]
                force_rgb=False,
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


    def get_ser_path(self) -> str | None:
        return getattr(self.reader, "path", None) if self.reader is not None else None

    def get_roi(self):
        return self._roi_tuple()  # already returns (x,y,w,h) or None

    def get_surface_anchor(self):
        return getattr(self, "_surface_anchor", None)
