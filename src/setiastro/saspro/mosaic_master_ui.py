# src/setiastro/saspro/mosaic_master_ui.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Widgets for the Mosaic Master window.

    PanelTable      the panel list: one row per panel, drag-and-drop files in
    MosaicMapWidget before a build, every plate-solved panel's footprint laid
                    out the way the mosaic will be (the reference panel's
                    frame, with north/east arrows); after a build, the
                    finished mosaic with each panel's outline, its gain and
                    alignment, and every seam coloured by how well it matched

Pure drawing and list handling; the window (mosaic_master.MosaicMasterDialog)
owns the panels and the build.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
from PyQt6.QtCore import (Qt, pyqtSignal, QPointF, QRectF, QSize,
                          QCoreApplication, QT_TRANSLATE_NOOP)
from PyQt6.QtGui import (QPainter, QPen, QColor, QBrush, QPolygonF, QFont,
                         QImage, QFontMetrics, QPixmap, QPalette, QPainterPath)
from PyQt6.QtWidgets import (QTableWidget, QWidget, QAbstractItemView,
                             QHeaderView, QToolTip, QSizePolicy, QAbstractButton)

# Panel outline colours, cycled
PANEL_COLORS = [
    (86, 180, 233), (230, 159, 0), (0, 158, 115), (204, 121, 167),
    (240, 228, 66), (0, 114, 178), (213, 94, 0), (170, 170, 255),
    (120, 220, 160), (255, 140, 140),
]
REF_COLOR = (255, 215, 0)

# Seam quality: leftover mismatch across a seam, as a fraction of the sky's
# pixel noise. A large-scale step well below the pixel noise doesn't show
# after a stretch; one approaching it does.
SEAM_OK = 0.10
SEAM_WARN = 0.30
SEAM_COLORS = {"ok": (60, 200, 90), "warn": (240, 180, 40), "bad": (235, 70, 60)}


def seam_grade(after, sigma):
    """'ok' | 'warn' | 'bad' | None for a seam's leftover mismatch."""
    try:
        r = float(after) / float(sigma)
    except Exception:
        return None
    if not math.isfinite(r):
        return None
    if r < SEAM_OK:
        return "ok"
    if r < SEAM_WARN:
        return "warn"
    return "bad"


def coverage_polygon(mask, max_side: int = 600):
    """Outline of the largest region of a footprint mask, as an (N, 2) array
    of (x, y) in the mask's pixel coordinates, or None."""
    import cv2
    m = np.asarray(mask)
    if m.ndim != 2 or not m.any():
        return None
    H, W = m.shape
    s = min(1.0, float(max_side) / float(max(H, W)))
    small = m.astype(np.uint8)
    if s < 1.0:
        # Area-averaged, not nearest: where a panel's bad pixels are left out
        # one by one (a master's darkest sky clipped to 0), nearest sampling
        # lands on them about as often as not and the outline dips there. A
        # block counts as covered when a quarter of it is; empty areas are 0.
        frac = cv2.resize(m.astype(np.float32), (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                          interpolation=cv2.INTER_AREA)
        small = (frac >= 0.25).astype(np.uint8)
    cs, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float64)
    if c.shape[0] < 3:
        return None
    return (c + 0.5) / s


def _qcolor(rgb, a=255):
    return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))


# Coverage depth colours: few frames -> deep violet, many -> yellow
_HEAT_T = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_HEAT_RGB = np.array([(60, 30, 120), (50, 90, 180), (30, 160, 160),
                      (140, 205, 80), (250, 230, 60)], float)


def heat_rgb(t):
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return np.stack([np.interp(t, _HEAT_T, _HEAT_RGB[:, c]) for c in range(3)], axis=-1)


def heat_qimage(heat, vmax, alpha=150):
    """ARGB image of a coverage-count raster; 0 = transparent."""
    h = np.asarray(heat, float)
    rgb = heat_rgb((h - 1.0) / max(float(vmax) - 1.0, 1.0))
    a = np.where(h > 0.5, alpha, 0).astype(np.uint8)
    bgra = np.dstack([rgb[..., 2], rgb[..., 1], rgb[..., 0], a]).astype(np.uint8)
    bgra = np.ascontiguousarray(bgra)
    H, W = h.shape
    return QImage(bgra.data, W, H, 4 * W, QImage.Format.Format_ARGB32).copy()


def coverage_raster(polys, max_side: int = 360):
    """Count raster of overlapping polygons: (counts, x0, y0, cell) in the
    polygons' units, or None."""
    import cv2
    polys = [np.asarray(p, float) for p in polys if p is not None and len(p) >= 3]
    if not polys:
        return None
    allp = np.concatenate(polys, axis=0)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    cell = max(x1 - x0, y1 - y0, 1e-9) / float(max_side)
    W = int(math.ceil((x1 - x0) / cell)) + 1
    H = int(math.ceil((y1 - y0) / cell)) + 1
    acc = np.zeros((H, W), np.float32)
    tmp = np.zeros((H, W), np.uint8)
    for p in polys:
        tmp[:] = 0
        q = np.round((p - [x0, y0]) / cell).astype(np.int32)
        cv2.fillPoly(tmp, [q], 1)
        acc += tmp
    return acc, float(x0), float(y0), float(cell)


# ─────────────────────────────────────────────────────────────────────────────
# Mode card: a big checkable button with a picture, a title and one line
# ─────────────────────────────────────────────────────────────────────────────
class ModeCardButton(QAbstractButton):
    """Checkable card for choosing between modes: icon on the left, bold
    title, a one-line description. The chosen card gets the palette's
    highlight border and tint; the other's picture is dimmed. Put the cards
    in an exclusive QButtonGroup."""
    ICON = 52

    def __init__(self, icon_path: str, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self._pix = QPixmap(icon_path) if icon_path else QPixmap()
        self._title = title
        self._subtitle = subtitle
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName(title)
        self.setAccessibleDescription(subtitle)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(240, max(self.ICON + 20, 2 * fm.height() + 30))

    def minimumSizeHint(self):
        return QSize(160, self.sizeHint().height())

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        pal = self.palette()
        on = self.isChecked()
        hover = self.underMouse() and self.isEnabled()
        base = pal.color(QPalette.ColorRole.Button)
        hi = pal.color(QPalette.ColorRole.Highlight)
        mid = pal.color(QPalette.ColorRole.Mid)
        if on:
            t = 0.22
            bg = QColor(int(base.red() * (1 - t) + hi.red() * t),
                        int(base.green() * (1 - t) + hi.green() * t),
                        int(base.blue() * (1 - t) + hi.blue() * t))
            border, bw = hi, 2.0
        else:
            bg = base.lighter(112) if hover else base
            border, bw = (hi if hover else mid), 1.0
        if not self.isEnabled():
            p.setOpacity(0.5)
        r = QRectF(self.rect()).adjusted(bw / 2 + 0.5, bw / 2 + 0.5, -bw / 2 - 0.5, -bw / 2 - 0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        p.fillPath(path, bg)
        pen = QPen(border, bw)
        if self.hasFocus() and not on:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setColor(hi)
        p.setPen(pen)
        p.drawPath(path)

        x = 12
        if not self._pix.isNull():
            s = self.ICON
            pm = self._pix.scaled(s * 2, s * 2, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            if not on:
                p.save()
                p.setOpacity(p.opacity() * (0.85 if hover else 0.55))
            p.drawPixmap(QRectF(x, (self.height() - s) / 2, s, s), pm, QRectF(pm.rect()))
            if not on:
                p.restore()
            x += s + 12

        tw = max(10, self.width() - x - (26 if on else 10))
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(max(6.0, self.font().pointSizeF() * 1.12))
        fm1 = QFontMetrics(f)
        fs = QFont(self.font())
        fm2 = QFontMetrics(fs)
        th = fm1.height() + 2 + fm2.height()
        y = (self.height() - th) / 2
        p.setFont(f)
        p.setPen(pal.color(QPalette.ColorRole.ButtonText))
        p.drawText(QRectF(x, y, tw, fm1.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm1.elidedText(self._title, Qt.TextElideMode.ElideRight, int(tw)))
        p.setFont(fs)
        sub = QColor(pal.color(QPalette.ColorRole.ButtonText))
        sub.setAlpha(170)
        p.setPen(sub)
        p.drawText(QRectF(x, y + fm1.height() + 2, tw, fm2.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm2.elidedText(self._subtitle, Qt.TextElideMode.ElideRight, int(tw)))

        if on:
            # a check badge in the top-right corner
            c = QPointF(self.width() - 15, 15)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(hi)
            p.drawEllipse(c, 8, 8)
            pk = QPen(pal.color(QPalette.ColorRole.HighlightedText), 2.0)
            pk.setCapStyle(Qt.PenCapStyle.RoundCap)
            pk.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pk)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(c.x() - 4, c.y()), QPointF(c.x() - 1, c.y() + 3),
                                      QPointF(c.x() + 4, c.y() - 3)]))
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Panel table
# ─────────────────────────────────────────────────────────────────────────────
class PanelTable(QTableWidget):
    """Panel list. Accepts files dropped from the desktop (filesDropped)."""
    filesDropped = pyqtSignal(list)
    COLS = ("#", QT_TRANSLATE_NOOP("PanelTable", "Panel"),
            QT_TRANSLATE_NOOP("PanelTable", "Plate solve"),
            QT_TRANSLATE_NOOP("PanelTable", "Scale"),
            QT_TRANSLATE_NOOP("PanelTable", "Ref"))
    COL_IDX, COL_NAME, COL_SOLVE, COL_SCALE, COL_REF = range(5)

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(
            [QCoreApplication.translate("PanelTable", c) for c in self.COLS])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setIconSize(QSize(40, 30))
        self.verticalHeader().setDefaultSectionSize(34)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        for c in (self.COL_IDX, self.COL_SOLVE, self.COL_SCALE, self.COL_REF):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setAlternatingRowColors(True)
        self.placeholder = self.tr("Drop panel files here\nor use + Files / + Views\n\n"
                                   "Linear, plate-solved panels work best.")

    @contextmanager
    def bulk_update(self):
        """Rewrite many cells at once. A ResizeToContents column re-measures
        every row on each cell change, so filling a 600-row table cell by cell
        cost ~37 ms per cell (110 s for the table). Here those columns are
        fixed while the cells are written and measured once at the end."""
        hh = self.horizontalHeader()
        auto = [c for c in range(self.columnCount())
                if hh.sectionResizeMode(c) == QHeaderView.ResizeMode.ResizeToContents]
        for c in auto:
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        self.setUpdatesEnabled(False)
        try:
            yield
        finally:
            for c in auto:
                hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
            self.setUpdatesEnabled(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
            if paths:
                self.filesDropped.emit(paths)
            e.acceptProposedAction()
            return
        super().dropEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rowCount() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Sky / result map
# ─────────────────────────────────────────────────────────────────────────────
class MosaicMapWidget(QWidget):
    """Plan: footprints on the sky. Result: the mosaic with its panel map."""
    panelClicked = pyqtSignal(int)
    resultActivated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._mode = "empty"
        self._plan = []            # dicts: idx, name, poly (x, y; y down), ref, color
        self._plan_note = ""
        self._frame = None         # north/east unit vectors, degrees per unit
        self._selected = set()
        self._result = None        # dict: image, size, panels, seams, sigma
        self._overlay = True
        self._hit = []             # (QPolygonF, kind, payload) for click / hover
        self._plan_rev = 0         # bumped by set_plan: invalidates the plan picture
        self._plan_cache = None    # (key, QPixmap, hit list)

    # ── data ─────────────────────────────────────────────────────────────
    def set_plan(self, panels, note: str = "", frame=None, dense: bool = False):
        """panels: polygons in one plane, x right and y DOWN (screen-like).
        frame: {"north": (dx, dy), "east": (dx, dy), "deg_per_unit": d}.
        dense: many frames (Seestar): thin outlines over a coverage-depth map
        instead of filled, labelled panels. A panel may carry approx=True
        (position from the header only, drawn dashed) or failed=True (red)."""
        self._plan = list(panels)
        self._plan_rev += 1
        self._plan_note = note
        self._frame = frame
        self._dense = bool(dense)
        self._plan_heat = None
        if self._dense:
            try:
                self._plan_heat = coverage_raster(
                    [d["poly"] for d in self._plan
                     if not d.get("failed") and d.get("poly") is not None])
            except Exception:
                self._plan_heat = None
        if self._mode != "result":
            self._mode = "plan" if self._plan else "empty"
        self.update()

    def show_plan(self):
        self._mode = "plan" if self._plan else "empty"
        self.update()

    def set_result(self, image: QImage, size, panels, seams, sigma, coverage=None):
        """coverage (Seestar): {"heat": 2-D frame counts over the whole canvas
        (downsampled), "max": int, "exptime": s per frame or None,
        "frames": [canvas polygons]}."""
        self._result = {"image": image, "size": tuple(size), "panels": list(panels or []),
                        "seams": list(seams or []), "sigma": sigma, "coverage": coverage}
        self._cov_img = None
        if coverage is not None and coverage.get("heat") is not None:
            self._cov_img = heat_qimage(coverage["heat"], max(float(coverage.get("max") or 1), 1.0),
                                        alpha=120)
        self._mode = "result"
        self.update()

    def show_result(self):
        if self._result is not None:
            self._mode = "result"
            self.update()

    def has_result(self) -> bool:
        return self._result is not None

    def mode(self) -> str:
        return self._mode

    def set_selected(self, idxs):
        self._selected = set(int(i) for i in idxs)
        self.update()

    def set_overlay_visible(self, on: bool):
        self._overlay = bool(on)
        self.update()

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(18, 20, 26))
        self._hit = []
        if self._mode == "plan":
            # Hundreds of Seestar footprints take a quarter of a second to
            # draw, and the window repaints far more often than the plan
            # changes (progress dialog, tooltips, hover): draw it once into a
            # picture and blit that until the plan, size or selection changes.
            dpr = self.devicePixelRatioF()
            key = (self._plan_rev, self.width(), self.height(), dpr,
                   tuple(sorted(self._selected)))
            c = self._plan_cache
            if c is None or c[0] != key:
                pm = QPixmap(max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
                pm.setDevicePixelRatio(dpr)
                pm.fill(QColor(18, 20, 26))
                q = QPainter(pm)
                q.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                q.setFont(self.font())
                self._paint_plan(q)
                q.end()
                c = self._plan_cache = (key, pm, list(self._hit))
            p.drawPixmap(0, 0, c[1])
            self._hit = list(c[2])
        elif self._mode == "result" and self._result is not None:
            self._paint_result(p)
        else:
            p.setPen(QColor(140, 140, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       getattr(self, "empty_text", None)
                       or self.tr("Plate-solved panels appear here\nas footprints on the sky."))
        p.end()

    def _paint_plan(self, p):
        polys = [d for d in self._plan if d.get("poly") is not None and len(d["poly"]) >= 3]
        if not polys:
            p.setPen(QColor(140, 140, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       self.tr("No panel is plate-solved yet.\n"
                               "Panels are solved when you build the mosaic."))
            self._paint_note(p, self._plan_note)
            return
        allp = np.concatenate([np.asarray(d["poly"], float) for d in polys], axis=0)
        x0, y0 = float(allp[:, 0].min()), float(allp[:, 1].min())
        x1, y1 = float(allp[:, 0].max()), float(allp[:, 1].max())
        w_deg, h_deg = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        m = 28
        avail_w = max(10, self.width() - 2 * m)
        avail_h = max(10, self.height() - 2 * m - 22)
        s = min(avail_w / w_deg, avail_h / h_deg)
        ox = m + 0.5 * (avail_w - s * w_deg)
        oy = m + 0.5 * (avail_h - s * h_deg)

        def to_px(x, y):         # y down, like the finished mosaic
            return QPointF(ox + (x - x0) * s, oy + (y - y0) * s)

        dense = getattr(self, "_dense", False)
        heat = getattr(self, "_plan_heat", None)
        if dense and heat is not None:
            acc, hx0, hy0, cell = heat
            vmax = max(float(acc.max()), 1.0)
            img = heat_qimage(acc, vmax, alpha=170)
            tl = to_px(hx0, hy0)
            p.drawImage(QRectF(tl.x(), tl.y(), acc.shape[1] * cell * s, acc.shape[0] * cell * s), img)
            self._heat_legend(p, vmax, self.tr("frames per spot"))

        # fills first, so every outline stays visible on top; overlaps show
        # as the darker/brighter blend of the two fills
        qpolys = []
        for d in polys:
            qp = QPolygonF([to_px(x, y) for x, y in np.asarray(d["poly"], float)])
            qpolys.append((d, qp))
            if dense:
                continue
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(_qcolor(d["color"], 55)))
            p.drawPolygon(qp)
        for d, qp in qpolys:
            sel = d["idx"] in self._selected
            col = (235, 70, 60) if d.get("failed") else (REF_COLOR if d.get("ref") else d["color"])
            if dense:
                pen = QPen(QColor(255, 255, 255) if sel else _qcolor(col, 230 if d.get("failed") else 110),
                           2.5 if sel else 1.0)
            else:
                pen = QPen(QColor(255, 255, 255) if sel else _qcolor(col),
                           3 if (sel or d.get("ref")) else 1.6)
            if d.get("approx"):
                pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(qp)
            self._hit.append((qp, "panel", d))
        if not dense:
            f = QFont(self.font()); f.setBold(True); f.setPointSizeF(max(8.0, f.pointSizeF() + 1))
            for d, qp in qpolys:
                c = qp.boundingRect().center()
                label = f"{d['idx'] + 1}" + (" ★" if d.get("ref") else "")
                self._badge(p, c, label, f, REF_COLOR if d.get("ref") else d["color"])

        fr = self._frame or {}
        dpu = float(fr.get("deg_per_unit") or 1.0)
        self._compass(p, fr.get("north", (0.0, -1.0)), fr.get("east", (-1.0, 0.0)))
        self._scale_bar(p, s / dpu, ox)
        n_solved = sum(1 for d in polys if not d.get("approx") and not d.get("failed"))
        approx = [d for d in polys if d.get("approx")]
        what = self.tr("frame") if dense else self.tr("panel")
        size = f"{w_deg * dpu:.2f}° × {h_deg * dpu:.2f}°"
        if approx and n_solved == 0:
            info = self.tr("estimated {0}").format(size)
        else:
            if dense:
                info = (self.tr("{0} solved frame · {1}") if n_solved == 1
                        else self.tr("{0} solved frames · {1}")).format(n_solved, size)
            else:
                info = (self.tr("{0} solved panel · {1}") if n_solved == 1
                        else self.tr("{0} solved panels · {1}")).format(n_solved, size)
        self._paint_note(p, (info + ("   ·   " + self._plan_note if self._plan_note else "")))
        if approx:
            self._estimate_banner(p, approx, n_solved, what, top=46 if (dense and heat is not None) else 12)

    def _paint_result(self, p):
        r = self._result
        img = r["image"]
        W, H = r["size"]
        m = 10
        avail_w = max(10, self.width() - 2 * m)
        avail_h = max(10, self.height() - 2 * m - 22)
        s = min(avail_w / max(W, 1), avail_h / max(H, 1))
        dw, dh = W * s, H * s
        ox = m + 0.5 * (avail_w - dw)
        oy = m + 0.5 * (avail_h - dh)
        p.drawImage(QRectF(ox, oy, dw, dh), img)

        def to_px(x, y):
            return QPointF(ox + x * s, oy + y * s)

        self._result_geom = (ox, oy, s, W, H)
        cov = r.get("coverage")
        if cov is not None and self._overlay:
            if getattr(self, "_cov_img", None) is not None:
                p.drawImage(QRectF(ox, oy, dw, dh), self._cov_img)
            pen = QPen(QColor(255, 255, 255, 70), 1.0); pen.setCosmetic(True)
            p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in cov.get("frames") or []:
                if poly is not None and len(poly) >= 3:
                    p.drawPolygon(QPolygonF([to_px(x, y) for x, y in np.asarray(poly, float)]))
            self._heat_legend(p, max(float(cov.get("max") or 1), 1.0), self.tr("frames per pixel"),
                              exptime=cov.get("exptime"))

        n_ok = n_warn = n_bad = 0
        sigma = r.get("sigma")
        for sm in r["seams"]:
            g = seam_grade(sm.get("after"), sigma)
            n_ok += g == "ok"; n_warn += g == "warn"; n_bad += g == "bad"
        if self._overlay:
            for sm in r["seams"]:
                poly = sm.get("poly")
                if poly is None or len(poly) < 3:
                    continue
                g = seam_grade(sm.get("after"), sigma) or "warn"
                qp = QPolygonF([to_px(x, y) for x, y in np.asarray(poly, float)])
                p.setBrush(QBrush(_qcolor(SEAM_COLORS[g], 85)))
                pen = QPen(_qcolor(SEAM_COLORS[g], 220), 1.2); pen.setCosmetic(True)
                p.setPen(pen)
                p.drawPolygon(qp)
                self._hit.append((qp, "seam", sm))
            f = QFont(self.font()); f.setBold(True)
            for k, d in enumerate(r["panels"]):
                poly = d.get("poly")
                if poly is None or len(poly) < 3:
                    continue
                qp = QPolygonF([to_px(x, y) for x, y in np.asarray(poly, float)])
                num = d.get("idx", k)
                num = k if num is None else num
                col = REF_COLOR if d.get("ref") else PANEL_COLORS[num % len(PANEL_COLORS)]
                pen = QPen(_qcolor(col, 230), 2.0 if d.get("ref") else 1.4); pen.setCosmetic(True)
                p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(qp)
                self._hit.append((qp, "panel", d))
            for k, d in enumerate(r["panels"]):
                poly = d.get("poly")
                if poly is None or len(poly) < 3:
                    continue
                pts = np.asarray(poly, float)
                c = to_px(float(pts[:, 0].mean()), float(pts[:, 1].mean()))
                num = d.get("idx", k)
                num = k if num is None else num
                col = REF_COLOR if d.get("ref") else PANEL_COLORS[num % len(PANEL_COLORS)]
                g = d.get("gain")
                label = f"{num + 1}" + (" ★" if d.get("ref") else
                                      (f" · ×{g:.3f}" if g is not None else ""))
                self._badge(p, c, label, f, col)
        # legend / summary
        parts = []
        if r["seams"]:
            parts.append(self.tr("seams: {0} matched, {1} small step, {2} visible step").format(
                n_ok, n_warn, n_bad))
        if r.get("coverage") is not None:
            parts.append(self.tr("hover for the depth at a spot"))
        parts.append(self.tr("double-click for a full-size preview"))
        self._paint_note(p, "   ·   ".join(parts), legend=bool(r["seams"]) and self._overlay)

    # ── small drawing helpers ────────────────────────────────────────────
    def _badge(self, p, c, text, font, rgb):
        p.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text) + 10
        th = fm.height() + 2
        rect = QRectF(c.x() - tw / 2, c.y() - th / 2, tw, th)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(rect, 4, 4)
        p.setPen(_qcolor(rgb))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_note(self, p, text, legend=False):
        f = QFont(self.font())
        p.setFont(f)
        y = self.height() - 8
        x = 10
        if legend:
            for key, lab in (("ok", self.tr("matched")), ("warn", self.tr("small step")),
                             ("bad", self.tr("visible step"))):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_qcolor(SEAM_COLORS[key]))
                p.drawRoundedRect(QRectF(x, y - 10, 10, 10), 2, 2)
                p.setPen(QColor(200, 200, 205))
                p.drawText(QPointF(x + 14, y), lab)
                x += 14 + QFontMetrics(f).horizontalAdvance(lab) + 14
            text = ""
        if text:
            p.setPen(QColor(185, 185, 195))
            p.drawText(QRectF(x, y - 16, self.width() - x - 10, 20),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       QFontMetrics(f).elidedText(text, Qt.TextElideMode.ElideRight,
                                                  self.width() - x - 10))

    def _estimate_banner(self, p, approx, n_solved, what, top=12):
        """Say plainly that dashed footprints are an estimate: drawn from each
        file's header pointing (RA/DEC, and the scale from FOCALLEN and
        XPIXSZ), north-up because the rotation isn't known until it is
        plate-solved. When every header gives the same pointing the estimate
        is one box, and the real layout only appears as frames are solved."""
        cs = np.array([np.asarray(d["poly"], float).mean(axis=0) for d in approx])
        ext = np.asarray(approx[0]["poly"], float)
        side = float(min(np.ptp(ext[:, 0]), np.ptp(ext[:, 1]))) or 1.0
        one_spot = len(approx) > 1 and float(np.ptp(cs, axis=0).max()) < 0.05 * side
        if n_solved == 0:
            head = self.tr("Estimated layout from the header pointing, not plate-solved yet")
        else:
            # `what` is the already-translated "frame"/"panel"; pick a whole
            # sentence per noun and number instead of gluing on an English 's'.
            k = len(approx)
            if what == self.tr("frame"):
                head = (self.tr("Partly estimated: {0} frame drawn from the header pointing (dashed) until solved")
                        if k == 1 else
                        self.tr("Partly estimated: {0} frames drawn from the header pointing (dashed) until solved"))
            else:
                head = (self.tr("Partly estimated: {0} panel drawn from the header pointing (dashed) until solved")
                        if k == 1 else
                        self.tr("Partly estimated: {0} panels drawn from the header pointing (dashed) until solved"))
            head = head.format(k)
        if one_spot:
            tail = self.tr("Every header gives the same pointing, so they stack in one spot; "
                           "the real layout appears as they are solved.")
        else:
            tail = self.tr("Positions are approximate and drawn north-up; rotation is known once solved.")
        f1 = QFont(self.font()); f1.setBold(True)
        f2 = QFont(self.font())
        fm1, fm2 = QFontMetrics(f1), QFontMetrics(f2)
        maxw = max(60, self.width() - 2 * 80)
        head = fm1.elidedText(head, Qt.TextElideMode.ElideRight, maxw)
        tail = fm2.elidedText(tail, Qt.TextElideMode.ElideRight, maxw)
        w = max(fm1.horizontalAdvance(head), fm2.horizontalAdvance(tail)) + 24
        h = fm1.height() + fm2.height() + 12
        r = QRectF((self.width() - w) / 2, top, w, h)
        p.save()
        p.setPen(QPen(QColor(240, 180, 40, 200), 1.0))
        p.setBrush(QColor(20, 20, 24, 215))
        p.drawRoundedRect(r, 6, 6)
        p.setFont(f1)
        p.setPen(QColor(245, 195, 70))
        p.drawText(QRectF(r.left(), r.top() + 5, r.width(), fm1.height()),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, head)
        p.setFont(f2)
        p.setPen(QColor(215, 215, 222))
        p.drawText(QRectF(r.left(), r.top() + 6 + fm1.height(), r.width(), fm2.height()),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, tail)
        p.restore()

    def _heat_legend(self, p, vmax, label, exptime=None):
        x, y, w, h = 12, 12, 130, 9
        n = 64
        cols = heat_rgb(np.linspace(0, 1, n))
        for i in range(n):
            p.fillRect(QRectF(x + i * w / n, y, w / n + 0.6, h), _qcolor(cols[i]))
        p.setPen(QColor(220, 220, 225))
        f = QFont(self.font()); p.setFont(f)
        p.drawText(QPointF(x, y + h + 13), "1")
        top = f"{int(round(vmax))}"
        if exptime:
            mins = vmax * float(exptime) / 60.0
            top += f" (≈{mins:.0f} min)" if mins >= 1 else f" (≈{vmax * float(exptime):.0f} s)"
        tw = QFontMetrics(f).horizontalAdvance(top)
        p.drawText(QPointF(x + w - tw, y + h + 13), top)
        p.drawText(QPointF(x + w + 8, y + h), label)

    def _compass(self, p, north, east):
        cx, cy = self.width() - 40, 40
        pen = QPen(QColor(200, 200, 210), 1.5); pen.setCosmetic(True)
        p.setPen(pen)
        for vec, lab in ((north, "N"), (east, "E")):
            dx, dy = float(vec[0]), float(vec[1])
            n = math.hypot(dx, dy) or 1.0
            dx, dy = dx / n, dy / n
            p.drawLine(QPointF(cx, cy), QPointF(cx + 20 * dx, cy + 20 * dy))
            p.drawText(QPointF(cx + 28 * dx - 4, cy + 28 * dy + 5), lab)

    def _scale_bar(self, p, s_px_per_deg, ox):
        target_px = 90.0
        arcmin = target_px / s_px_per_deg * 60.0
        nice = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600]
        val = min(nice, key=lambda v: abs(v - arcmin))
        L = val / 60.0 * s_px_per_deg
        x, y = self.width() - 20 - L, self.height() - 34
        pen = QPen(QColor(200, 200, 210), 2); pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLine(QPointF(x, y), QPointF(x + L, y))
        p.drawText(QPointF(x, y - 5), f"{val}′" if val < 60 else f"{val // 60}°")

    # ── interaction ──────────────────────────────────────────────────────
    def _hit_at(self, pos):
        pt = QPointF(pos)
        found = None
        for qp, kind, payload in self._hit:          # last drawn = on top
            if qp.containsPoint(pt, Qt.FillRule.OddEvenFill):
                if kind == "seam":
                    return kind, payload
                found = (kind, payload)
        return found

    def mousePressEvent(self, e):
        h = self._hit_at(e.position())
        if h and h[0] == "panel" and self._mode == "plan":
            self.panelClicked.emit(int(h[1]["idx"]))
        super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self._mode == "result":
            self.resultActivated.emit()
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e):
        h = self._hit_at(e.position())
        tip = ""
        r = self._result if self._mode == "result" else None
        if r is not None and r.get("coverage") is not None and self._overlay \
                and getattr(self, "_result_geom", None):
            ox, oy, s, W, H = self._result_geom
            cx = (e.position().x() - ox) / s
            cy = (e.position().y() - oy) / s
            heat = r["coverage"].get("heat")
            if heat is not None and 0 <= cx < W and 0 <= cy < H:
                hy = int(cy * heat.shape[0] / H); hx = int(cx * heat.shape[1] / W)
                v = float(heat[min(hy, heat.shape[0] - 1), min(hx, heat.shape[1] - 1)])
                if v > 0:
                    ex = r["coverage"].get("exptime")
                    tip = self.tr("≈{0:.0f} frames here").format(v) + (f" (≈{v * ex / 60:.1f} min)" if ex else "")
                    QToolTip.showText(e.globalPosition().toPoint(), tip, self)
                    super().mouseMoveEvent(e)
                    return
        if h:
            kind, d = h
            if kind == "panel":
                tip = d.get("tip") or d.get("name", "")
            elif kind == "seam":
                tip = d.get("tip", "")
        if tip:
            QToolTip.showText(e.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(e)
