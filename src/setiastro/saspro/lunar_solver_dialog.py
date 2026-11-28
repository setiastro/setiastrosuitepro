# src/setiastro/saspro/lunar_solver_dialog.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""Lunar Solver window: solve the active lunar image, label its features,
show lunar coordinates under the cursor, and store the solution on the image."""
from __future__ import annotations

import datetime as _dt
import math
import traceback

import numpy as np
from PyQt6.QtCore import (
    QCoreApplication, QDate, QDateTime, QPointF, QUrl, QRectF, QSettings, Qt, QThread, QTime, QTimeZone, QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QGuiApplication, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QDateTimeEdit, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMenu,
    QProgressBar, QPushButton, QSizePolicy, QSpinBox, QTextBrowser, QVBoxLayout, QWidget,
)

from setiastro.saspro import lunar_solver as LS
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.widgets.factory_reset import make_reset_button, restore_controls, snapshot_controls

# feature classes the label options switch on
_CRATER_TYPES = ("AA",)
_AREA_TYPES = ("ME", "OC", "SI", "LC", "PA", "MO", "VA", "RU", "PR", "PL")
_LINEAR_TYPES = ("RI", "DO", "CA")          # rilles, ridges, crater chains: size = length
_LETTERED_TYPES = ("SF",)
_AREA_SET = frozenset(_AREA_TYPES + _LINEAR_TYPES)
_PREVIEW_MIN_PX = 9.0                        # smallest crater the preview labels, on screen


def _tr(text):
    return QCoreApplication.translate("LunarSolver", text)


def _utc_zone() -> QTimeZone:
    try:
        return QTimeZone.utc()                    # Qt 6.5+
    except AttributeError:
        return QTimeZone(b"UTC")


def _display_rgb8(image) -> np.ndarray:
    """The image as 8-bit RGB for display and annotation (common percentile
    stretch over all channels, so colour balance is kept)."""
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    a = np.nan_to_num(a.astype(np.float32, copy=False))
    lo, hi = np.percentile(a, 0.05), np.percentile(a, 99.95)
    u8 = np.clip((a - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    if u8.ndim == 2:
        u8 = np.repeat(u8[..., None], 3, axis=2)
    return np.ascontiguousarray(u8[..., :3])


def _same_pixels(a, b) -> bool:
    """The same image buffer: identity, or (an ROI preview builds a new slice of
    its parent on every read) the same memory, shape and strides."""
    if a is b:
        return True
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    return (a.shape == b.shape and a.strides == b.strides
            and a.__array_interface__["data"][0] == b.__array_interface__["data"][0])


def _qimage_rgb(u8: np.ndarray) -> QImage:
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class _SolveWorker(QThread):
    progressed = pyqtSignal(float, str)
    done = pyqtSignal(object, str)          # LunarSolution or None, error ("" = cancelled)

    def __init__(self, image, when, site, scale=None):
        super().__init__(None)              # parent None: Qt lifetime rule
        self._image = image
        self._when = when
        self._site = site
        self._scale = scale
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            sol = LS.solve_moon(self._image, self._when, self._site,
                                progress=lambda f, m: self.progressed.emit(f, m),
                                cancel=lambda: self._cancel, scale_arcsec=self._scale)
            self.done.emit(sol, "")
        except LS.LunarSolveCancelled:
            self.done.emit(None, "")
        except LS.LunarSolveError as ex:
            self.done.emit(None, str(ex))
        except Exception as ex:             # worker owns its errors: log, report in the dialog
            traceback.print_exc()
            self.done.emit(None, _tr("Unexpected error: {0}").format(ex))
        finally:
            self._image = None


class _MapsWorker(QThread):
    """Downloads the detailed maps of the near side (same signals as _SolveWorker)."""
    progressed = pyqtSignal(float, str)
    done = pyqtSignal(object, str)          # None, error ("" = finished or cancelled)

    def __init__(self):
        super().__init__(None)              # parent None: Qt lifetime rule
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            LS.download_detailed_maps(progress=lambda f, m: self.progressed.emit(f, m),
                                      cancel=lambda: self._cancel)
            self.done.emit(None, "")
        except LS.LunarSolveCancelled:
            self.done.emit(None, "")
        except LS.LunarSolveError as ex:
            self.done.emit(None, str(ex))
        except Exception as ex:             # worker owns its errors: log, report in the dialog
            traceback.print_exc()
            self.done.emit(None, _tr("Unexpected error: {0}").format(ex))


class _OverlayView(QWidget):
    """The image with feature outlines and names drawn as vectors on top.
    Wheel zooms about the cursor, drag pans, double-click fits; right-click
    asks the dialog for a context menu. Reports the image pixel under the mouse."""
    hovered = pyqtSignal(float, float)      # image x, y (or -1, -1 when outside)
    context = pyqtSignal(float, float, object)   # image x, y, global QPoint
    zoomed = pyqtSignal(float)              # device px per image px

    MAX_SCALE = 16.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(360, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self._img = None                    # screen-size preview
        self._full = None                   # full resolution, for zooming in
        self._iw = self._ih = 0
        self._items = []
        self._zoom = 1.0                    # times the fit scale
        self._cx = self._cy = 0.0           # image point at the widget centre
        self._drag = None

    def set_image(self, qimg: QImage | None, full_w: int = 0, full_h: int = 0, full: QImage | None = None):
        self._img = qimg
        self._full = full
        self._iw, self._ih = int(full_w), int(full_h)
        self.fit()

    def set_items(self, items):
        self._items = items or []
        self.update()

    # ---- geometry ----
    def _fit_scale(self):
        return min(self.width() / self._iw, self.height() / self._ih)

    def _geom(self):
        if self._img is None or self._iw <= 0:
            return None
        s = self._fit_scale() * self._zoom
        return s, self.width() / 2.0 - self._cx * s, self.height() / 2.0 - self._cy * s

    def scale(self) -> float:
        g = self._geom()
        return g[0] if g else 0.0

    def fit(self):
        self._zoom = 1.0
        self._cx, self._cy = self._iw / 2.0, self._ih / 2.0
        self.update()
        self.zoomed.emit(self.scale())

    def set_scale(self, s_dev, about=None):
        """Zoom to s_dev device px per image px, keeping image point `about`
        (or the view centre) where it is on screen."""
        if self._img is None or self._iw <= 0:
            return
        fit = self._fit_scale()
        s_old = fit * self._zoom
        s_new = min(max(s_dev, fit), max(fit, self.MAX_SCALE))
        if about is not None:
            ax, ay = about
            px = self.width() / 2.0 + (ax - self._cx) * s_old
            py = self.height() / 2.0 + (ay - self._cy) * s_old
            self._cx = ax - (px - self.width() / 2.0) / s_new
            self._cy = ay - (py - self.height() / 2.0) / s_new
        self._zoom = s_new / fit
        self._clamp()
        self.update()
        self.zoomed.emit(self.scale())

    def zoom_to(self, x, y, radius_px):
        """Centre image point (x, y) and fit a circle of radius_px around it."""
        if self._img is None:
            return
        r = max(8.0, float(radius_px))
        s = min(self.width(), self.height()) / (2.6 * r)
        self._cx, self._cy = float(x), float(y)
        self.set_scale(s)

    def _clamp(self):
        if self._zoom <= 1.0 + 1e-9:
            self._zoom = 1.0
            self._cx, self._cy = self._iw / 2.0, self._ih / 2.0
            return
        self._cx = min(max(self._cx, 0.0), float(self._iw))
        self._cy = min(max(self._cy, 0.0), float(self._ih))

    def _to_image(self, pos):
        g = self._geom()
        if g is None:
            return None
        s, ox, oy = g
        return (pos.x() - ox) / s, (pos.y() - oy) / s

    # ---- painting ----
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 16))
        g = self._geom()
        if g is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, _tr("Open a lunar image"))
            return
        s, ox, oy = g
        img_rect = QRectF(ox, oy, self._iw * s, self._ih * s)
        vis = img_rect.intersected(QRectF(self.rect()))
        if vis.isEmpty():
            return
        # the screen-size preview while it has enough pixels, the full image beyond that
        src_img = self._img
        if self._full is not None and s > self._img.width() / float(self._iw) * 1.05:
            src_img = self._full
        k = src_img.width() / float(self._iw)
        src = QRectF((vis.left() - ox) / s * k, (vis.top() - oy) / s * k, vis.width() / s * k, vis.height() / s * k)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 2.0)
        p.drawImage(vis, src_img, src)
        p.setClipRect(vis)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _paint_items(p, self._items, s, ox, oy, font_px=11, min_disp_px=_PREVIEW_MIN_PX, bounds=vis)

    # ---- mouse ----
    def wheelEvent(self, ev):
        pt = self._to_image(ev.position())
        if pt is None:
            return
        steps = ev.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.set_scale(self.scale() * (1.25 ** steps), about=pt)
        ev.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._drag = (ev.position(), self._cx, self._cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._drag is not None:
            self._drag = None
            self.unsetCursor()
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.fit()

    def mouseMoveEvent(self, ev):
        g = self._geom()
        if g is None:
            return
        if self._drag is not None:
            p0, cx0, cy0 = self._drag
            s = g[0]
            self._cx = cx0 - (ev.position().x() - p0.x()) / s
            self._cy = cy0 - (ev.position().y() - p0.y()) / s
            self._clamp()
            self.update()
        x, y = self._to_image(ev.position())
        if 0 <= x < self._iw and 0 <= y < self._ih:
            self.hovered.emit(x, y)
        else:
            self.hovered.emit(-1.0, -1.0)

    def contextMenuEvent(self, ev):
        pt = self._to_image(QPointF(ev.pos()))
        if pt is not None and 0 <= pt[0] < self._iw and 0 <= pt[1] < self._ih:
            self.context.emit(pt[0], pt[1], ev.globalPos())

    def leaveEvent(self, ev):
        self.hovered.emit(-1.0, -1.0)
        super().leaveEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.zoomed.emit(self.scale())


def _html_escape(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _maps_height_km(lat, lon):
    """LOLA height at a point when the reference maps are loaded, else None."""
    maps = getattr(LS, "_MAPS", None)
    if maps is None:
        return None
    try:
        return float(LS._sample_h(maps, [lat], [lon])[0])
    except Exception:
        return None


def _paint_items(p: QPainter, items, s, ox, oy, font_px, min_disp_px=0.0, bounds=None):
    """Shared by the preview and the annotated export. s/ox/oy map image px to
    the paint device. Items are drawn largest first; a label that would
    overlap one already drawn is skipped with its outline, so the big
    features stay readable. Craters smaller than min_disp_px on the device
    are left out."""
    rim = QPen(QColor(255, 196, 40, 230), max(1.0, font_px / 9.0))
    area_pen = QColor(120, 220, 255)
    name_pen = QColor(110, 255, 140)
    font = QFont()
    font.setPixelSize(int(font_px))
    p.setFont(font)
    fm = p.fontMetrics()
    halo = QPen(QColor(0, 0, 0, 200), max(2.0, font_px / 4.0))
    taken = []
    site_pen = QColor(255, 110, 220)
    for it in [d for d in items if d.get("site")]:
        u, v = it["anchor"]
        x, y = ox + u * s, oy + v * s
        m = max(3.0, font_px * 0.32)
        txt = it["name"]
        tw = fm.horizontalAdvance(txt)
        tx, ty = x + m + 3, y + fm.ascent() / 2.0 - 1
        box = QRectF(tx - 2, ty - fm.ascent() - 1, tw + 4, fm.height() + 2)
        if any(box.intersects(b) for b in taken):
            continue
        taken.append(box)
        taken.append(QRectF(x - m - 1, y - m - 1, 2 * m + 2, 2 * m + 2))
        mark = QPainterPath()
        mark.addEllipse(QPointF(x, y), m, m)
        mark.moveTo(x - 1.8 * m, y)
        mark.lineTo(x - 0.5 * m, y)
        mark.moveTo(x + 0.5 * m, y)
        mark.lineTo(x + 1.8 * m, y)
        mark.moveTo(x, y - 1.8 * m)
        mark.lineTo(x, y - 0.5 * m)
        mark.moveTo(x, y + 0.5 * m)
        mark.lineTo(x, y + 1.8 * m)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.strokePath(mark, halo)
        p.strokePath(mark, QPen(site_pen, max(1.2, font_px / 10.0)))
        path = QPainterPath()
        path.addText(QPointF(tx, ty), font, txt)
        p.strokePath(path, halo)
        p.fillPath(path, QBrush(site_pen))
    for it in sorted((d for d in items if not d.get("site")), key=lambda d: -d["r"]):
        if not it["area"] and 2.0 * it["r"] * s < min_disp_px:
            continue
        u, v = it["anchor"]
        x, y = ox + u * s, oy + v * s
        txt = it["name"]
        tw = fm.horizontalAdvance(txt)
        if it["area"]:
            if bounds is not None and not bounds.contains(QPointF(x, y)):
                continue                    # its centre is out of view (zoomed in): no label
            x -= tw / 2.0
            if bounds is not None:          # slide the name so it stays inside the view
                x = min(max(x, bounds.left() + 2), bounds.right() - tw - 2)
            col = area_pen
        else:
            r = it["r"] * s
            x += 0.72 * r + 2
            y -= 0.72 * r + 2
            col = name_pen
        box = QRectF(x - 2, y - fm.ascent() - 1, tw + 4, fm.height() + 2)
        if any(box.intersects(b) for b in taken):
            continue
        taken.append(box)
        poly = it.get("poly")
        if poly is not None:
            p.setPen(rim)
            p.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            first = True
            for (u, v), vis in zip(poly, it["vis"]):
                pt = QPointF(ox + u * s, oy + v * s)
                if not vis:
                    first = True
                    continue
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            if it["vis"].all():
                path.closeSubpath()
            p.drawPath(path)
        path = QPainterPath()
        path.addText(QPointF(x, y), font, txt)
        p.strokePath(path, halo)
        p.fillPath(path, QBrush(col))


class LunarSolverDialog(QDialog):
    """Standalone Lunar Solver. Follows the active view."""

    def __init__(self, parent=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Lunar Solver"))
        self._main = parent
        self._dm = doc_manager
        self._settings = QSettings()
        self.doc = None
        self._solution = None
        self._worker = None
        self._pending_close = False
        self._follow_conn = None
        self._base_u8 = None
        self._lit_cache = None
        self._doc_conn = None           # the document whose edits refresh the preview
        self._edit_timer = QTimer(self)
        self._edit_timer.setSingleShot(True)
        self._edit_timer.setInterval(150)
        self._edit_timer.timeout.connect(self._on_doc_edited)

        # ---------------- controls ----------------
        left = QVBoxLayout()
        self.lbl_target = QLabel()
        self.lbl_target.setWordWrap(True)
        left.addWidget(self.lbl_target)

        gb_time = QGroupBox(self.tr("Capture time (UTC)"))
        tl = QVBoxLayout(gb_time)
        self.dt_edit = QDateTimeEdit()
        self.dt_edit.setDisplayFormat("yyyy-MM-dd  HH:mm:ss")
        try:
            self.dt_edit.setTimeZone(_utc_zone())     # Qt 6.7+
        except AttributeError:
            self.dt_edit.setTimeSpec(Qt.TimeSpec.UTC)
        self.dt_edit.setCalendarPopup(True)
        tl.addWidget(self.dt_edit)
        self.chk_unknown = QCheckBox(self.tr("Date unknown (slower blind search)"))
        self.chk_unknown.setToolTip(self.tr(
            "Solve without a capture time: the Sun angle and libration are searched for.\n"
            "Takes two or three times as long as a dated solve."))
        tl.addWidget(self.chk_unknown)
        self.lbl_time_src = QLabel()
        self.lbl_time_src.setWordWrap(True)
        self.lbl_time_src.setStyleSheet("color: palette(mid); font-size: 11px;")
        tl.addWidget(self.lbl_time_src)
        self.btn_reread = QPushButton(self.tr("Read From Image"))
        self.btn_reread.setToolTip(self.tr("Read the capture time from the image's header or file name again"))
        tl.addWidget(self.btn_reread, 0, Qt.AlignmentFlag.AlignLeft)
        left.addWidget(gb_time)

        self.gb_site = QGroupBox(self.tr("Observing site"))
        self.gb_site.setCheckable(True)
        self.gb_site.setToolTip(self.tr(
            "Your site refines the libration by up to 1 degree.\n"
            "Without it the Earth's centre is used and the fit absorbs the difference."))
        sf = QFormLayout(self.gb_site)
        self.spin_lat = QDoubleSpinBox()
        self.spin_lat.setRange(-90.0, 90.0)
        self.spin_lat.setDecimals(4)
        self.spin_lat.setSuffix(" °")
        self.spin_lon = QDoubleSpinBox()
        self.spin_lon.setRange(-180.0, 180.0)
        self.spin_lon.setDecimals(4)
        self.spin_lon.setSuffix(" °")
        self.spin_lon.setToolTip(self.tr("East positive (west longitudes are negative)"))
        sf.addRow(self.tr("Latitude:"), self.spin_lat)
        sf.addRow(self.tr("Longitude (E+):"), self.spin_lon)
        left.addWidget(self.gb_site)

        self.gb_scale = QGroupBox(self.tr("Image scale (helps small fields)"))
        self.gb_scale.setCheckable(True)
        self.gb_scale.setChecked(False)
        self.gb_scale.setToolTip(self.tr(
            "Arcseconds per pixel. Fields smaller than about 200 km have too few surface features to find\n"
            "by matching alone; with the scale known, every rotation and position on the near side is searched.\n"
            "Read from the image's FITS header when it has one, otherwise taken from your last solve."))
        scl = QVBoxLayout(self.gb_scale)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 200.0)
        self.spin_scale.setDecimals(3)
        self.spin_scale.setSingleStep(0.01)
        self.spin_scale.setSuffix(" ″/px")
        self.spin_scale.setValue(0.5)
        scl.addWidget(self.spin_scale)
        self.lbl_scale_src = QLabel()
        self.lbl_scale_src.setWordWrap(True)
        scl.addWidget(self.lbl_scale_src)
        left.addWidget(self.gb_scale)

        self.gb_maps = QGroupBox(self.tr("Detailed maps (small fields)"))
        self.gb_maps.setToolTip(self.tr(
            "Small, sharp fields are checked against 64 px/° maps (0.47 km per pixel) of their own area.\n"
            "Each area is downloaded the first time a solve needs it and kept. Download them all to solve\n"
            "small fields without an internet connection."))
        mpl = QVBoxLayout(self.gb_maps)
        self.lbl_maps = QLabel()
        self.lbl_maps.setWordWrap(True)
        mpl.addWidget(self.lbl_maps)
        self.btn_maps = QPushButton(self.tr("Download for Offline Use"))
        mpl.addWidget(self.btn_maps, 0, Qt.AlignmentFlag.AlignLeft)

        gb_lab = QGroupBox(self.tr("Labels"))
        ll = QVBoxLayout(gb_lab)
        self.chk_craters = QCheckBox(self.tr("Named craters"))
        self.chk_craters.setChecked(True)
        self.chk_areas = QCheckBox(self.tr("Maria, mountains and valleys"))
        self.chk_areas.setChecked(True)
        self.chk_linear = QCheckBox(self.tr("Rilles, ridges and crater chains"))
        self.chk_linear.setChecked(False)
        self.chk_lettered = QCheckBox(self.tr("Lettered craters (Copernicus A, ...)"))
        self.chk_lettered.setChecked(False)
        self.chk_sites = QCheckBox(self.tr("Spacecraft landing sites"))
        self.chk_sites.setChecked(True)
        self.chk_night = QCheckBox(self.tr("Label the unlit side too"))
        self.chk_night.setChecked(False)
        for w in (self.chk_craters, self.chk_areas, self.chk_linear, self.chk_lettered, self.chk_sites,
                  self.chk_night):
            ll.addWidget(w)
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Smallest feature:")))
        self.spin_min_px = QSpinBox()
        self.spin_min_px.setRange(4, 2000)
        self.spin_min_px.setValue(40)
        self.spin_min_px.setSuffix(" px")
        self.spin_min_px.setToolTip(self.tr("Label only features at least this wide in the image"))
        row.addWidget(self.spin_min_px)
        row.addStretch(1)
        ll.addLayout(row)
        left.addWidget(gb_lab)

        btns = QHBoxLayout()
        self.btn_solve = QPushButton(self.tr("Solve"))
        self.btn_solve.setDefault(True)
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_cancel.setVisible(False)
        self.btn_reset = make_reset_button(self)
        btns.addWidget(self.btn_solve)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_reset)
        left.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        left.addWidget(self.progress)
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        left.addWidget(self.lbl_status)

        gb_res = QGroupBox(self.tr("Solution"))
        self.res_form = QFormLayout(gb_res)
        self._res = {}
        for key, label in (("fit", self.tr("Fit:")), ("confidence", self.tr("Confidence:")),
                           ("scale", self.tr("Scale:")), ("north", self.tr("Lunar north:")),
                           ("mirror", self.tr("Mirrored:")), ("libration", self.tr("Libration:")),
                           ("sun", self.tr("Sun:")), ("phase", self.tr("Phase:")),
                           ("centre", self.tr("Frame centre:")), ("time", self.tr("Solved for:"))):
            v = QLabel("—")
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v.setWordWrap(True)
            self._res[key] = v
            self.res_form.addRow(label, v)

        self.btn_annotate = QPushButton(self.tr("Create Annotated Image"))
        self.btn_annotate.setToolTip(self.tr("Open a copy of the image with the labels drawn in, as a new view"))
        self.btn_annotate.setEnabled(False)
        left.addStretch(1)

        # grip + help: own bottom row of the left column. No addStretch() in
        # this row: a stretch makes the column claim half of the window.
        from setiastro.saspro.shortcuts import ShortcutGrip
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addWidget(ShortcutGrip("lunar_solver", self))
        grip_row.addWidget(make_help_button("lunar_solver", self), 0, Qt.AlignmentFlag.AlignRight)
        left.addLayout(grip_row)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(330)

        centre = QVBoxLayout()
        self.view = _OverlayView(self)
        self.view.setToolTip(self.tr("Wheel: zoom. Drag: pan. Double-click: fit. Right-click a feature for its details."))
        centre.addWidget(self.view, 1)
        zrow = QHBoxLayout()
        self.btn_fit = QPushButton(self.tr("Fit"))
        self.btn_fit.setToolTip(self.tr("Fit the whole image in the view (double-click the image does the same)"))
        self.btn_one = QPushButton(self.tr("1:1"))
        self.btn_one.setToolTip(self.tr("One image pixel per screen pixel"))
        self.lbl_zoom = QLabel("")
        self.lbl_zoom.setMinimumWidth(48)
        for b in (self.btn_fit, self.btn_one):
            b.setFixedWidth(48)
            zrow.addWidget(b)
        zrow.addWidget(self.lbl_zoom)
        self.lbl_hover = QLabel(" ")
        self.lbl_hover.setStyleSheet("font-family: Consolas, monospace;")
        zrow.addWidget(self.lbl_hover, 1)
        centre.addLayout(zrow)

        # results in their own column on the far side of the image: stacked
        # under the controls they made the window too tall for a 1080p screen
        results = QVBoxLayout()
        results.addWidget(gb_res)
        results.addWidget(self.btn_annotate)
        results.addStretch(1)
        results.addWidget(self.gb_maps)       # reference data: bottom of the right column, out of the way
        results_w = QWidget()
        results_w.setLayout(results)
        results_w.setFixedWidth(300)

        main = QHBoxLayout(self)
        main.addWidget(left_w, 0)
        main.addLayout(centre, 1)
        main.addWidget(results_w, 0)
        self.resize(1480, 760)

        # ---------------- wiring ----------------
        self.btn_solve.clicked.connect(self._start_solve)
        self.btn_cancel.clicked.connect(self._cancel_solve)
        self.btn_maps.clicked.connect(self._start_maps_download)
        self.btn_reread.clicked.connect(self._read_time_from_doc)
        self.btn_annotate.clicked.connect(self._create_annotated)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        self.chk_unknown.toggled.connect(lambda on: self.dt_edit.setEnabled(not on))
        for w in (self.chk_craters, self.chk_areas, self.chk_linear, self.chk_lettered, self.chk_sites,
                  self.chk_night):
            w.toggled.connect(self._on_label_options)
        self.spin_min_px.valueChanged.connect(self._on_label_options)
        self.gb_site.toggled.connect(self._save_site)
        self.spin_lat.valueChanged.connect(self._save_site)
        self.spin_lon.valueChanged.connect(self._save_site)
        self.view.hovered.connect(self._on_hover)
        self.view.context.connect(self._on_context)
        self.view.zoomed.connect(lambda s: self.lbl_zoom.setText("%d %%" % round(100.0 * s) if s > 0 else ""))
        self.btn_fit.clicked.connect(lambda _=False: self.view.fit())
        self.btn_one.clicked.connect(lambda _=False: self.view.set_scale(1.0))

        # factory defaults: label options only. The site is the observer's own
        # data and the date belongs to the image, so Reset leaves both alone.
        self._factory = snapshot_controls(self, exclude=(self.gb_site, gb_time, self.gb_scale))
        self._load_settings()
        self._update_maps_status()

        # follow the active view
        if self._main is not None and hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = None
        self._set_doc(self._active_doc())

    # ---------------- documents ----------------
    def _active_doc(self):
        dm = self._dm
        doc = None
        try:
            if dm is not None:
                doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                       else getattr(dm, "active_document", None))
        except Exception:
            doc = None
        return doc

    def _on_active_doc_changed(self, doc):
        if self._worker is not None:
            return                      # a solve is pinned to its document
        if doc is None or getattr(doc, "image", None) is None:
            return
        self._set_doc(doc)

    def _doc_title(self, doc):
        for attr in ("display_name",):
            f = getattr(doc, attr, None)
            if callable(f):
                try:
                    return str(f())
                except Exception:
                    pass
        meta = getattr(doc, "metadata", {}) or {}
        return str(meta.get("display_name") or meta.get("file_path") or self.tr("Untitled"))

    def _watch_doc(self, doc):
        """Refresh the preview when the shown image is edited (a flip or crop
        moves its stored placement with it)."""
        if self._doc_conn is not None and self._doc_conn is not doc:
            try:
                self._doc_conn.changed.disconnect(self._on_doc_changed)
            except Exception:
                pass
            self._doc_conn = None
        if doc is not None and self._doc_conn is None and hasattr(doc, "changed"):
            try:
                doc.changed.connect(self._on_doc_changed)
                self._doc_conn = doc
            except Exception:
                self._doc_conn = None

    def _on_doc_changed(self):
        if self._worker is None and self.isVisible():
            self._edit_timer.start()        # an edit can emit several times: refresh once

    def _on_doc_edited(self):
        doc = self.doc
        if doc is None or self._worker is not None or not self.isVisible():
            return
        img = getattr(doc, "image", None)
        if img is None:
            return
        self._show_doc_image(img)
        sol = self._existing_solution(doc)
        if sol is not None:
            self._apply_solution(sol, store=False)
        else:
            self._solution = None
            self._show_result(None)
            self.btn_annotate.setEnabled(False)
            self.view.set_items([])

    def _show_doc_image(self, img):
        h, w = img.shape[:2]
        self._base_u8 = None
        self._lit_cache = None
        u8 = _display_rgb8(img)
        k = min(1.0, 1600.0 / max(h, w))
        if k < 1.0:
            import cv2
            small = cv2.resize(u8, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        else:
            small = u8
        full = _qimage_rgb(np.ascontiguousarray(u8)) if k < 1.0 else None
        self.view.set_image(_qimage_rgb(np.ascontiguousarray(small)), w, h, full=full)
        self.view.set_items([])

    def _set_doc(self, doc):
        self._watch_doc(doc)
        self.doc = doc
        self._solution = None
        self.lbl_hover.setText(" ")
        self.btn_annotate.setEnabled(False)
        self.lbl_status.setText("")
        self._show_result(None)
        img = getattr(doc, "image", None) if doc is not None else None
        if img is None:
            self.lbl_target.setText(self.tr("No image open."))
            self.view.set_image(None)
            self.btn_solve.setEnabled(False)
            return
        self.btn_solve.setEnabled(True)
        self.lbl_target.setText(self.tr("Image: {0}").format(self._doc_title(doc)))
        self._show_doc_image(img)
        self._read_time_from_doc()
        self._read_scale_from_doc()
        sol = self._existing_solution(doc)
        if sol is not None:
            self._apply_solution(sol, store=False)
            self.lbl_status.setText(self.tr("This image already has a lunar solution. Solve again to replace it."))

    def _read_scale_from_doc(self):
        """Image scale hint: the FITS header first (WCS, a pixel-scale card, or
        pixel size and focal length, the plate solver's reading), else the
        last solve's scale."""
        doc = self.doc
        meta = getattr(doc, "metadata", None) or {}
        val = None
        # a reopened FITS keeps its header as fits_header
        hdr = next((meta.get(k) for k in ("original_header", "fits_header", "header") if meta.get(k) is not None), None)
        if hdr is not None:
            try:
                from astropy.io.fits import Header
                from setiastro.saspro.plate_solver import _estimate_scale_arcsec_from_header
                if isinstance(hdr, Header):
                    val = _estimate_scale_arcsec_from_header(hdr)
            except Exception:
                val = None
        if val:
            self.spin_scale.setValue(float(val))
            self.gb_scale.setChecked(True)
            self.lbl_scale_src.setText(self.tr("From the image header."))
            return
        last = self._settings.value("lunar_solver/last_scale", None)
        try:
            last = float(last) if last is not None else None
        except (TypeError, ValueError):
            last = None
        if last and last > 0:
            self.spin_scale.setValue(last)
            self.gb_scale.setChecked(True)
            self.lbl_scale_src.setText(self.tr("From your last solve. Uncheck if this image was taken "
                                               "with a different telescope or camera."))
        else:
            self.gb_scale.setChecked(False)
            self.lbl_scale_src.setText(self.tr("Not in the image. Enter it if you know it."))

    def _existing_solution(self, doc):
        meta = getattr(doc, "metadata", None) or {}
        # None when stale: resized or cropped outside SASpro's geometry tools since
        return LS.solution_from_metadata(meta, doc.image.shape[:2])

    def _read_time_from_doc(self):
        doc = self.doc
        if doc is None:
            return
        meta = getattr(doc, "metadata", None) or {}
        try:
            when, src = LS.capture_time(meta)
        except Exception:
            when, src = None, "No capture time found in the image."
        if when is not None:
            qdt = QDateTime(QDate(when.year, when.month, when.day),
                            QTime(when.hour, when.minute, when.second), _utc_zone())
            self.dt_edit.setDateTime(qdt)
            self.chk_unknown.setChecked(False)
            self.lbl_time_src.setText(src)
        else:
            self.chk_unknown.setChecked(True)
            self.lbl_time_src.setText(self.tr("No capture time found in the image. Enter it, or leave "
                                              "“Date unknown” checked for a blind search."))

    # ---------------- settings ----------------
    def _load_settings(self):
        s = self._settings
        lat = s.value("lunar_solver/lat", None)
        lon = s.value("lunar_solver/lon", None)
        if lat is None or lon is None:        # first use: the What's In My Sky location
            try:
                lat = float(s.value("latitude", 0.0))
                lon = float(s.value("longitude", 0.0))
            except Exception:
                lat, lon = 0.0, 0.0
            use = not (abs(float(lat)) < 1e-9 and abs(float(lon)) < 1e-9)
        else:
            use = s.value("lunar_solver/use_site", True, type=bool)
        for w in (self.gb_site, self.spin_lat, self.spin_lon):
            w.blockSignals(True)
        self.spin_lat.setValue(float(lat))
        self.spin_lon.setValue(float(lon))
        self.gb_site.setChecked(bool(use))
        for w in (self.gb_site, self.spin_lat, self.spin_lon):
            w.blockSignals(False)
        self.chk_craters.setChecked(s.value("lunar_solver/labels_craters", True, type=bool))
        self.chk_areas.setChecked(s.value("lunar_solver/labels_areas", True, type=bool))
        self.chk_lettered.setChecked(s.value("lunar_solver/labels_lettered", False, type=bool))
        self.chk_linear.setChecked(s.value("lunar_solver/labels_linear", False, type=bool))
        self.chk_night.setChecked(s.value("lunar_solver/labels_night", False, type=bool))
        self.chk_sites.setChecked(s.value("lunar_solver/labels_sites", True, type=bool))
        self.spin_min_px.setValue(int(s.value("lunar_solver/min_px", 40)))

    def _save_site(self, *_):
        s = self._settings
        s.setValue("lunar_solver/use_site", bool(self.gb_site.isChecked()))
        s.setValue("lunar_solver/lat", float(self.spin_lat.value()))
        s.setValue("lunar_solver/lon", float(self.spin_lon.value()))

    def _on_label_options(self, *_):
        s = self._settings
        s.setValue("lunar_solver/labels_craters", bool(self.chk_craters.isChecked()))
        s.setValue("lunar_solver/labels_areas", bool(self.chk_areas.isChecked()))
        s.setValue("lunar_solver/labels_lettered", bool(self.chk_lettered.isChecked()))
        s.setValue("lunar_solver/labels_linear", bool(self.chk_linear.isChecked()))
        s.setValue("lunar_solver/labels_night", bool(self.chk_night.isChecked()))
        s.setValue("lunar_solver/labels_sites", bool(self.chk_sites.isChecked()))
        s.setValue("lunar_solver/min_px", int(self.spin_min_px.value()))
        self._refresh_items()

    def _reset_to_factory(self):
        restore_controls(self._factory)

    # ---------------- solving ----------------
    def _start_solve(self):
        if self._worker is not None or self.doc is None or getattr(self.doc, "image", None) is None:
            return
        when = None
        if not self.chk_unknown.isChecked():
            qdt = self.dt_edit.dateTime().toUTC()
            d, t = qdt.date(), qdt.time()
            when = _dt.datetime(d.year(), d.month(), d.day(), t.hour(), t.minute(), t.second())
        site = (float(self.spin_lat.value()), float(self.spin_lon.value()), 0.0) if self.gb_site.isChecked() else None
        self._solve_doc = self.doc
        self._solve_img = self.doc.image    # an edit during the solve replaces this array
        scale = float(self.spin_scale.value()) if self.gb_scale.isChecked() else None
        w = _SolveWorker(self.doc.image, when, site, scale)
        # bound methods (not lambdas): queued onto the GUI thread; sender()
        # tells an older run's late signals apart
        w.progressed.connect(self._on_progress)
        w.done.connect(self._on_done)
        w.finished.connect(self._on_thread_finished)
        self._worker = w
        self.btn_solve.setVisible(False)
        self.btn_cancel.setVisible(True)
        self.btn_annotate.setEnabled(False)
        self.btn_maps.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.lbl_status.setText(self.tr("Starting…"))
        w.start()

    def _update_maps_status(self):
        """How much of the near side's detailed maps is on this computer."""
        try:
            have, total = LS.detailed_maps_status()
            ready = LS.detailed_maps_ready()
        except Exception:
            have, total, ready = 0, LS.DETAILED_MAPS_MB, False
        if ready:
            self.lbl_maps.setText(self.tr("All {0} MB are on this computer: small fields solve offline.").format(total))
        elif have <= 0:
            self.lbl_maps.setText(self.tr("None on this computer yet ({0} MB in all). Each area is downloaded "
                                          "when a solve first needs it.").format(total))
        else:
            self.lbl_maps.setText(self.tr("{0} of {1} MB on this computer. Each missing area is downloaded "
                                          "when a solve first needs it.").format(have, total))
        self.btn_maps.setVisible(not ready)
        self.btn_maps.setEnabled(self._worker is None)

    def _start_maps_download(self):
        if self._worker is not None:
            return
        w = _MapsWorker()
        w.progressed.connect(self._on_progress)
        w.done.connect(self._on_maps_done)
        w.finished.connect(self._on_thread_finished)
        self._worker = w
        self.btn_solve.setVisible(False)
        self.btn_cancel.setVisible(True)
        self.btn_maps.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.lbl_status.setText(self.tr("Starting the download…"))
        w.start()

    def _on_maps_done(self, _none, err):
        if self.sender() is not self._worker:
            return
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_solve.setVisible(True)
        if self._pending_close:
            return
        try:
            ready = LS.detailed_maps_ready()
        except Exception:
            ready = False
        if err:
            self.lbl_status.setText(err)
        elif ready:
            self.lbl_status.setText(self.tr("The detailed maps are on this computer."))
        else:
            self.lbl_status.setText(self.tr("Download stopped. What was downloaded is kept: press the button "
                                            "again to continue."))

    def _cancel_solve(self):
        if self._worker is not None:
            self._worker.cancel()
            self.lbl_status.setText(self.tr("Cancelling…"))

    def _on_progress(self, frac, msg):
        if self.sender() is not self._worker:
            return
        self.progress.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.lbl_status.setText(msg)

    def _on_done(self, sol, err):
        if self.sender() is not self._worker:
            return                       # a late signal from an older run
        self.progress.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_solve.setVisible(True)
        if self._pending_close:
            return
        if sol is None:
            self.lbl_status.setText(err or self.tr("Cancelled."))
            return
        doc = getattr(self, "_solve_doc", None)
        solved_img, self._solve_img = getattr(self, "_solve_img", None), None
        if doc is not None and not _same_pixels(getattr(doc, "image", None), solved_img):
            # the image was edited while solving: this placement fits the old pixels
            self.lbl_status.setText(self.tr("The image changed while it was being solved, so the solution was not "
                                            "stored. Solve again."))
            if doc is self.doc:
                self._on_doc_edited()
            return
        self._apply_solution(sol, store=True, doc=doc)
        try:                                 # the next image from the same setup gets it as a hint
            if sol.arcsec_per_px:
                scale = float(sol.arcsec_per_px)
            else:                            # no date: the mean distance, good to a few percent
                scale = math.degrees(math.asin(LS.R_MOON / 384400.0)) * 3600.0 / float(sol.radius_px)
            self._settings.setValue("lunar_solver/last_scale", scale)
        except Exception:
            pass
        took = next((n for n in sol.notes if n.startswith("solved in")), "")
        self.lbl_status.setText(self.tr("Solved ({0}). The solution is stored with the image.").format(
            took.replace("solved in ", "") if took else ""))

    def _on_thread_finished(self):
        if self.sender() is self._worker:
            self._worker = None
        if self._pending_close:
            self._pending_close = False
            self.close()
            return
        if self._worker is None:
            self._update_maps_status()      # a solve may have fetched an area too

    # ---------------- results ----------------
    def _apply_solution(self, sol, store, doc=None):
        doc = doc if doc is not None else self.doc
        if store and doc is not None:
            self._store_on_doc(doc, sol)
        if doc is not self.doc:
            return
        self._solution = sol
        self._show_result(sol)
        self._refresh_items()
        self.btn_annotate.setEnabled(True)

    def _store_on_doc(self, doc, sol):
        meta = getattr(doc, "metadata", None)
        if meta is None:
            return
        try:
            # into the header a save writes (a reopened file keeps it as fits_header)
            LS.store_solution_in_metadata(meta, sol)
        except Exception:
            traceback.print_exc()
            meta["lunar_solution"] = sol
        # same refresh Plate Solve does, so the Header Viewer shows the cards
        try:
            meta.pop("__header_snapshot__", None)
            from setiastro.saspro.doc_manager import _snapshot_header_for_metadata
            _snapshot_header_for_metadata(meta)
        except Exception:
            pass
        if self._main is not None and hasattr(self._main, "_refresh_header_viewer"):
            try:
                self._main._refresh_header_viewer(doc)
            except Exception:
                pass

    def _show_result(self, sol):
        if sol is None:
            for v in self._res.values():
                v.setText("—")
            return
        mode = {"disc": self.tr("full disc"), "closeup": self.tr("close-up"),
                "blind": self.tr("blind search")}.get(sol.mode, sol.mode)
        self._res["fit"].setText(self.tr("{0} matches, {1:.2f} px RMS ({2})").format(sol.n_matches, sol.rms_px, mode))
        conf = getattr(sol, "confidence", None)
        if conf is None:
            self._res["confidence"].setText("—")
        else:
            txt = self.tr("{0:.0f} % of the photo explained").format(100.0 * conf)
            run = getattr(sol, "runner_up", None)
            if run is not None:
                txt += self.tr("; runner-up {0:.1f} %").format(100.0 * run)
            self._res["confidence"].setText(txt)
        self._res["confidence"].setToolTip(self.tr(
            "How much of the band-passed photo the Moon's albedo and slope maps explain at this placement, "
            "under the best-fitting Sun. The runner-up is the best other placement the search found."))
        self._res["phase"].setText(self._phase_text(sol))
        fc = sol.frame_centre() if hasattr(sol, "frame_centre") else None
        if fc is None:
            self._res["centre"].setText(self.tr("off the Moon"))
        else:
            self._res["centre"].setText(self.tr("{0:.2f}° {1}, {2:.2f}° {3}").format(
                abs(fc[0]), "N" if fc[0] >= 0 else "S", abs(fc[1]), "E" if fc[1] >= 0 else "W"))
        if sol.arcsec_per_px:
            self._res["scale"].setText(self.tr("{0:.3f}″/px, {1:.3f} km/px at disc centre").format(
                sol.arcsec_per_px, sol.km_per_px))
        else:
            self._res["scale"].setText(self.tr("{0:.3f} km/px at disc centre").format(sol.km_per_px))
        self._res["north"].setText(self.tr("{0:.1f}° clockwise from up").format(sol.north_up_deg))
        self._res["mirror"].setText(self.tr("Yes (mirror image of the sky)") if sol.mirrored else self.tr("No"))
        la, lo = sol.sub_earth
        self._res["libration"].setText(self.tr("{0:+.2f}° latitude, {1:+.2f}° longitude").format(la, lo))
        sla, slo = sol.sub_solar
        if getattr(sol, "sun_approximate", False):
            self._res["sun"].setText(self.tr("colongitude ≈ {0:.0f}° (approximate: too little shading in "
                                             "view to pin it down)").format(sol.colongitude))
        else:
            self._res["sun"].setText(self.tr("colongitude {0:.1f}°").format(sol.colongitude))
        self._res["sun"].setToolTip(self.tr("Sub-solar point {0:+.2f}° latitude, {1:+.2f}° longitude").format(sla, slo))
        if sol.date_known:
            self._res["time"].setText(sol.utc.replace("T", " ")[:19] + " UTC")
        else:
            self._res["time"].setText(self.tr("no date (Sun angle and libration found by search)"))

    def _phase_text(self, sol):
        k = sol.illuminated_fraction
        if k < 0.03:
            name = self.tr("New Moon")
        elif k > 0.97:
            name = self.tr("Full Moon")
        elif 0.47 <= k <= 0.53:
            name = self.tr("First quarter") if sol.waxing else self.tr("Last quarter")
        elif sol.waxing:
            name = self.tr("Waxing crescent") if k < 0.5 else self.tr("Waxing gibbous")
        else:
            name = self.tr("Waning crescent") if k < 0.5 else self.tr("Waning gibbous")
        return self.tr("{0}, {1:.0f} % lit, phase angle {2:.1f}°").format(name, 100.0 * k, sol.phase_angle_deg)

    def _feature_items(self, sol, min_px):
        types = []
        if self.chk_craters.isChecked():
            types += list(_CRATER_TYPES)
        if self.chk_areas.isChecked():
            types += list(_AREA_TYPES)
        if self.chk_linear.isChecked():
            types += list(_LINEAR_TYPES)
        if self.chk_lettered.isChecked():
            types += list(_LETTERED_TYPES)
        if not types:
            return []
        feats = LS.visible_features(sol, min_px=float(min_px), types=tuple(types), max_count=1500)
        if not feats:
            return []
        F = LS.load_features()
        index = {n: i for i, n in enumerate(F["name"])}
        if not self.chk_night.isChecked():
            # keep features whose centre sees the Sun (or sits on the terminator); with
            # an undated solve's approximate Sun, those the photo itself shows lit
            ids = np.array([index[f[0]] for f in feats])
            la, lo = np.radians(F["lat"][ids]), np.radians(F["lon"][ids])
            vec = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
            lit = self._lit_flags(sol, vec @ np.asarray(sol.S) > -0.01, [(f[2], f[3], f[4]) for f in feats])
            feats = [f for f, ok in zip(feats, lit) if ok]
        craters = [f for f in feats if f[1] not in _AREA_SET]
        items = []
        if craters:
            ids = [index[f[0]] for f in craters]
            polys, vis = LS.feature_outlines(sol, F["lat"][ids], F["lon"][ids], F["diam"][ids])
            for f, poly, v in zip(craters, polys, vis):
                items.append(dict(name=f[0], area=False, anchor=(f[2], f[3]), r=f[4], poly=poly, vis=v))
        H, W = sol.shape
        for f in feats:
            # a region's name sits on its centre: only label regions centred in frame
            if f[1] in _AREA_SET and 0 <= f[2] < W and 0 <= f[3] < H:
                items.append(dict(name=f[0], area=True, anchor=(f[2], f[3]), r=f[4], poly=None, vis=None))
        return items + self._site_items(sol)

    def _site_items(self, sol):
        if not self.chk_sites.isChecked():
            return []
        out = []
        for name, country, year, kind, approx, lat, lon, u, v in LS.visible_sites(sol):
            if not self.chk_night.isChecked():
                la, lo = math.radians(lat), math.radians(lon)
                vec = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
                if not self._lit_flags(sol, np.array([float(vec @ np.asarray(sol.S)) >= -0.01]), [(u, v, 0.0)])[0]:
                    continue
            out.append(dict(name="%s (%d)" % (name, year), area=False, site=True, anchor=(u, v), r=0.0,
                            poly=None, vis=None))
        return out

    def _image_lit(self, sol):
        """Where the photo itself shows sunlit surface: (integral image of the
        lit mask on a <= 1024 px copy, scale), or (None, 1.0). Cached per solution."""
        img = getattr(self.doc, "image", None) if self.doc is not None else None
        c = self._lit_cache
        if c is not None and c[0] is sol and c[1] is img:
            return c[2], c[3]
        if img is None:
            return None, 1.0
        import cv2
        a = np.asarray(img)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        elif a.ndim == 3 and a.shape[2] > 4:
            a = a[..., :3]
        H, W = a.shape[:2]
        f = min(1.0, 1024.0 / max(H, W))
        a = np.nan_to_num(a.astype(np.float32, copy=False))
        if f < 1.0:
            a = cv2.resize(a, (max(1, int(round(W * f))), max(1, int(round(H * f)))), interpolation=cv2.INTER_AREA)
        if a.ndim == 3:
            a = a.mean(axis=2)
        lo, hi = float(np.percentile(a, 1.0)), float(np.percentile(a, 99.5))
        lit = cv2.GaussianBlur(a, (0, 0), 2.0) > lo + 0.08 * max(hi - lo, 1e-12)
        ii = cv2.integral(lit.astype(np.uint8))
        self._lit_cache = (sol, img, ii, f)          # holds sol and img: no id() reuse
        return ii, f

    def _lit_flags(self, sol, by_sun, places):
        """Lit test for labels. With a known Sun: by_sun (bool array, does the
        centre see the Sun). With the approximate Sun angle of an undated solve
        (it can be 10-20 deg off and put the terminator through lit ground):
        the photo decides, at least 10 % lit pixels within each place's
        (u, v, r) core."""
        by_sun = np.asarray(by_sun, bool)
        if not getattr(sol, "sun_approximate", False):
            return by_sun
        ii, f = self._image_lit(sol)
        if ii is None:
            return np.ones(len(places), bool)
        h, w = ii.shape[0] - 1, ii.shape[1] - 1
        out = np.zeros(len(places), bool)
        for i, (u, v, r) in enumerate(places):
            hr = max(0.7 * float(r) * f, 2.0)
            x0, x1 = int(np.clip(math.floor(u * f - hr), 0, w)), int(np.clip(math.ceil(u * f + hr), 0, w))
            y0, y1 = int(np.clip(math.floor(v * f - hr), 0, h)), int(np.clip(math.ceil(v * f + hr), 0, h))
            if x1 <= x0 or y1 <= y0:
                continue
            n = int(ii[y1, x1]) - int(ii[y0, x1]) - int(ii[y1, x0]) + int(ii[y0, x0])
            out[i] = n >= 0.1 * (x1 - x0) * (y1 - y0)
        return out

    def _refresh_items(self):
        sol = self._solution
        if sol is None:
            self.view.set_items([])
            return
        self.view.set_items(self._feature_items(sol, self.spin_min_px.value()))

    def _on_hover(self, x, y):
        sol = self._solution
        if sol is None or x < 0:
            self.lbl_hover.setText(" ")
            return
        lat, lon = sol.pixel_to_latlon(np.array([[x, y]]))
        lat, lon = float(lat[0]), float(lon[0])
        if not math.isfinite(lat):
            self.lbl_hover.setText(self.tr("x {0:.0f}  y {1:.0f}   off the Moon").format(x, y))
            return
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        txt = self.tr("x {0:.0f}  y {1:.0f}   {2:.3f}° {3}  {4:.3f}° {5}").format(
            x, y, abs(lat), ns, abs(lon), ew)
        name = self._feature_at(lat, lon)
        if name:
            txt += "   " + name
        self.lbl_hover.setText(txt)

    _TYPE_NAMES = {
        "AA": "crater", "SF": "satellite crater", "RI": "rille (rima)", "MO": "mountain (mons, montes)",
        "DO": "wrinkle ridge (dorsum)", "ME": "sea (mare)", "LC": "lake (lacus)", "CA": "crater chain (catena)",
        "VA": "valley (vallis)", "SI": "bay (sinus)", "PR": "cape (promontorium)", "RU": "scarp (rupes)",
        "ST": "station (statio)", "PA": "marsh (palus)", "OC": "ocean (oceanus)", "PL": "plain (planitia)",
        "LF": "landing-site name", "AL": "albedo feature",
    }

    def _features_under(self, lat, lon):
        """Named features containing the point, smallest first: (index, distance km)."""
        F = LS.load_features()
        la, lo = np.radians(lat), np.radians(lon)
        fl, fo = np.radians(F["lat"]), np.radians(F["lon"])
        cosd = np.sin(la) * np.sin(fl) + np.cos(la) * np.cos(fl) * np.cos(fo - lo)
        dist_km = np.arccos(np.clip(cosd, -1, 1)) * LS.R_MOON
        inside = dist_km <= np.maximum(F["diam"] / 2.0, 1.0)
        if not self.chk_lettered.isChecked():
            inside &= F["type"] != "SF"
        idx = np.where(inside)[0]
        idx = idx[np.argsort(F["diam"][idx])]
        return [(int(i), float(dist_km[i])) for i in idx[:4]]

    def _site_near(self, lat, lon):
        sol = self._solution
        near_km = max(3.0, 6.0 * sol.km_per_px) if sol is not None else 3.0
        la, lo = math.radians(lat), math.radians(lon)
        best = None
        for row in LS.LANDING_SITES:
            a, b = math.radians(row[4]), math.radians(row[5])
            c = math.sin(la) * math.sin(a) + math.cos(la) * math.cos(a) * math.cos(b - lo)
            d = math.acos(max(-1.0, min(1.0, c))) * LS.R_MOON
            if d <= near_km and (best is None or d < best[0]):
                best = (d, row)
        return best[1] if best else None

    @staticmethod
    def _fmt_latlon(lat, lon, nd=3):
        return "%.*f° %s, %.*f° %s" % (nd, abs(lat), "N" if lat >= 0 else "S", nd, abs(lon), "E" if lon >= 0 else "W")

    def _sun_at(self, lat, lon):
        """Sun elevation (deg) at a surface point under the solution's Sun."""
        sol = self._solution
        la, lo = math.radians(lat), math.radians(lon)
        r = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
        S = np.asarray(sol.S, np.float64)
        return math.degrees(math.asin(max(-1.0, min(1.0, float(r @ S) / float(np.linalg.norm(S))))))

    def _on_context(self, x, y, gpos):
        sol = self._solution
        menu = QMenu(self)
        if sol is None:
            a = menu.addAction(self.tr("Solve the image to identify features"))
            a.setEnabled(False)
            menu.exec(gpos)
            return
        lat, lon = sol.pixel_to_latlon(np.array([[x, y]]))
        lat, lon = float(lat[0]), float(lon[0])
        if not math.isfinite(lat):
            a = menu.addAction(self.tr("Off the Moon"))
            a.setEnabled(False)
            menu.exec(gpos)
            return
        F = LS.load_features()
        site = self._site_near(lat, lon) if self.chk_sites.isChecked() else None
        feats = self._features_under(lat, lon)
        targets = []
        if site is not None:
            targets.append(("site", site))
        targets += [("feature", i) for i, _ in feats[:3]]
        for kind, t in targets:
            if kind == "site":
                name, slat, slon = t[0], t[4], t[5]
                title = self.tr("{0} landing site").format(name)
                radius_km = 10.0
            else:
                name, slat, slon = F["name"][t], float(F["lat"][t]), float(F["lon"][t])
                title = "%s — %s" % (name, self.tr(self._TYPE_NAMES.get(str(F["type"][t]), str(F["type"][t]))))
                radius_km = max(5.0, float(F["diam"][t]) / 2.0)
            sub = menu.addMenu(title)
            sub.addAction(self.tr("Details…"), lambda _=False, k=kind, v=t, la_=lat, lo_=lon: self._show_details(k, v, la_, lo_))
            uv, vis = sol.latlon_to_pixel(slat, slon, relief=False)
            if bool(vis[0]):
                r_px = radius_km / sol.km_per_px
                sub.addAction(self.tr("Zoom to it"),
                              lambda _=False, u=float(uv[0, 0]), v=float(uv[0, 1]), r=r_px: self.view.zoom_to(u, v, r))
            sub.addAction(self.tr("Copy name"), lambda _=False, n=name: QGuiApplication.clipboard().setText(n))
            sub.addAction(self.tr("Copy its coordinates"),
                          lambda _=False, a=slat, b=slon: QGuiApplication.clipboard().setText("%.4f, %.4f" % (a, b)))
            sub.addAction(self.tr("Look up on Wikipedia"), lambda _=False, n=name, k=kind, v=t: self._lookup(n, k, v))
        if targets:
            menu.addSeparator()
        here = menu.addAction(self.tr("This point: {0}").format(self._fmt_latlon(lat, lon)))
        here.setEnabled(False)
        menu.addAction(self.tr("Copy this point's coordinates"),
                       lambda _=False: QGuiApplication.clipboard().setText("%.4f, %.4f" % (lat, lon)))
        menu.addAction(self.tr("Zoom in here"), lambda _=False: self.view.set_scale(self.view.scale() * 2.0, about=(x, y)))
        menu.addAction(self.tr("Fit the image"), lambda _=False: self.view.fit())
        menu.exec(gpos)

    def _lookup(self, name, kind, t):
        from urllib.parse import quote
        q = name
        if kind == "feature" and str(LS.load_features()["type"][t]) in ("AA", "SF"):
            q = name + " (lunar crater)"
        elif kind == "site":
            q = name + " Moon landing"
        QDesktopServices.openUrl(QUrl("https://en.wikipedia.org/w/index.php?search=" + quote(q)))

    def _show_details(self, kind, t, lat, lon):
        """A small non-modal window: what the feature is, its size and place, and
        how it is lit in this image."""
        sol = self._solution
        if sol is None:
            return
        rows = []
        if kind == "site":
            name, country, year, mk, slat, slon, approx = t
            title = self.tr("{0} landing site").format(name)
            rows += [(self.tr("Mission"), "%s, %s, %d" % (country, mk, year)),
                     (self.tr("Position"), self._fmt_latlon(slat, slon, 4)),
                     (self.tr("Accuracy"), self.tr("as announced at the time, not confirmed from orbit") if approx
                      else self.tr("measured from Lunar Reconnaissance Orbiter images (tens of metres)"))]
        else:
            F = LS.load_features()
            name, code = F["name"][t], str(F["type"][t])
            slat, slon, diam = float(F["lat"][t]), float(F["lon"][t]), float(F["diam"][t])
            title = name
            rows += [(self.tr("Kind"), self.tr(self._TYPE_NAMES.get(code, code))),
                     (self.tr("Size"), self.tr("{0:.1f} km across ({1:.0f} px in this image)").format(
                         diam, diam / sol.km_per_px) if diam > 0 else "—"),
                     (self.tr("Centre"), self._fmt_latlon(slat, slon, 3))]
        el = self._sun_at(slat, slon)
        if el > 0.5:
            light = self.tr("lit, Sun {0:.1f}° above the horizon").format(el)
        elif el > -0.5:
            light = self.tr("on the terminator (Sun on the horizon)")
        else:
            light = self.tr("in darkness, Sun {0:.1f}° below the horizon").format(-el)
        if getattr(sol, "sun_approximate", False):
            light += " " + self.tr("(from the approximate Sun angle of an undated solve)")
        rows.append((self.tr("Lighting"), light))
        if _maps_height_km(slat, slon) is not None:
            rows.append((self.tr("Height"), self.tr("{0:+.2f} km against the 1737.4 km sphere (LOLA)").format(
                _maps_height_km(slat, slon))))
        rows.append((self.tr("You clicked"), self._fmt_latlon(lat, lon, 3)))
        html = "<h3>%s</h3><table cellspacing='4'>%s</table>" % (
            _html_escape(title), "".join("<tr><td><b>%s</b></td><td>%s</td></tr>" % (_html_escape(a), _html_escape(b))
                                         for a, b in rows))
        w = QDialog(self)
        w.setWindowTitle(title)
        w.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        lay = QVBoxLayout(w)
        tb = QTextBrowser(w)
        tb.setHtml(html)
        tb.setOpenExternalLinks(True)
        lay.addWidget(tb)
        w.resize(460, 260)
        w.show()

    def _feature_at(self, lat, lon):
        """A landing site within a few km, else the smallest named crater or
        area containing the point."""
        if self.chk_sites.isChecked():
            sol = self._solution
            near_km = max(3.0, 6.0 * sol.km_per_px) if sol is not None else 3.0
            la, lo = math.radians(lat), math.radians(lon)
            best = None
            for name, country, year, kind, slat, slon, approx in LS.LANDING_SITES:
                a, b = math.radians(slat), math.radians(slon)
                c = math.sin(la) * math.sin(a) + math.cos(la) * math.cos(a) * math.cos(b - lo)
                d = math.acos(max(-1.0, min(1.0, c))) * LS.R_MOON
                if d <= near_km and (best is None or d < best[0]):
                    best = (d, name, country, year, kind, approx)
            if best is not None:
                d, name, country, year, kind, approx = best
                txt = self.tr("{0} landing site ({1}, {2}, {3})").format(name, country, kind, year)
                if approx:
                    txt += self.tr(", position as announced, not confirmed from orbit")
                return txt
        F = LS.load_features()
        la, lo = np.radians(lat), np.radians(lon)
        fl, fo = np.radians(F["lat"]), np.radians(F["lon"])
        cosd = np.sin(la) * np.sin(fl) + np.cos(la) * np.cos(fl) * np.cos(fo - lo)
        dist_km = np.arccos(np.clip(cosd, -1, 1)) * LS.R_MOON
        inside = dist_km <= F["diam"] / 2.0
        if not self.chk_lettered.isChecked():
            inside &= F["type"] != "SF"
        if not inside.any():
            return ""
        i = np.where(inside)[0]
        return F["name"][int(i[np.argmin(F["diam"][i])])]

    # ---------------- annotated copy ----------------
    def _create_annotated(self):
        sol, doc = self._solution, self.doc
        if sol is None or doc is None or getattr(doc, "image", None) is None:
            return
        u8 = _display_rgb8(doc.image)
        h, w = u8.shape[:2]
        qimg = _qimage_rgb(u8).convertToFormat(QImage.Format.Format_RGB888)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _paint_items(p, self._feature_items(sol, self.spin_min_px.value()), 1.0, 0.0, 0.0,
                     font_px=max(12, int(round(max(w, h) / 110.0))), bounds=QRectF(0, 0, w, h))
        p.end()
        ptr = qimg.constBits()
        ptr.setsize(qimg.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape(h, qimg.bytesPerLine())[:, :3 * w].reshape(h, w, 3)
        out = arr.astype(np.float32) / np.float32(255.0)
        title = self.tr("{0} (lunar labels)").format(self._doc_title(doc))
        dm = self._dm
        try:
            if hasattr(dm, "open_array"):
                dm.open_array(out, metadata={"is_mono": False}, title=title)
            elif hasattr(dm, "create_document"):
                dm.create_document(image=out, metadata={"is_mono": False}, name=title)
        except Exception as ex:
            self.lbl_status.setText(self.tr("Could not open the annotated image: {0}").format(ex))

    # ---------------- lifetime ----------------
    def reject(self):
        # Esc goes through close() so a running solve is never torn down mid-run
        self.close()

    def closeEvent(self, ev):
        if self._worker is not None:
            self._pending_close = True
            self._worker.cancel()
            self.hide()
            ev.ignore()
            return
        try:
            if self._follow_conn and self._main is not None and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = None
        self._edit_timer.stop()
        self._watch_doc(None)
        super().closeEvent(ev)
