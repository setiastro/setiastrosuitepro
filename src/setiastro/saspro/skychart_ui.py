#saspro/skychart_ui.py
# Copyright (C) Franklin Marek / SetiAstro. Part of Seti Astro Suite Pro (GPLv3).
"""
The drawn-chart page of Finder Chart: layer tree, preview with zoom and a live
RA/Dec readout, the Chart / Contents / Rendering / Output tabs, history, and the
SVG / PDF / PNG / new-image outputs.

Based on the Sky Chart Generator dialog by Jean d'Hooghvorst, contributed to
SASpro. As a panel it lives inside Finder Chart rather than as its own window,
so the script-era window management (focus stealing, Escape guards, the
self-reloading entry point, a global excepthook) is gone; settings live in
QSettings instead of a file next to the data.
"""
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF, QSize, QTimer, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPicture, QTransform
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QComboBox, QCheckBox,
    QTabWidget, QWidget, QTreeWidget, QTreeWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QFileDialog, QMessageBox, QGroupBox,
    QSplitter, QFontComboBox, QSpinBox, QHeaderView,
    QListWidget, QListWidgetItem, QDialogButtonBox, QColorDialog,
)

from setiastro.saspro.skychart_engine import *          # noqa: F401,F403
from setiastro.saspro.skychart_engine import (           # underscore names
    _PROJ_FOVMAX, _PROJ_TMAX,
)

_log = logging.getLogger("setiastro.skychart")
_SETTINGS_KEY = "skychart/params_json"


class _BprpRelay(QObject):
    """Module-lifetime relay for the background online-colour fetch. The worker
    thread emits on this, never on a panel that may have been closed and
    deleted while it ran."""
    done = pyqtSignal(int, str)


_BPRP_RELAY = None


def _bprp_relay() -> _BprpRelay:
    global _BPRP_RELAY
    if _BPRP_RELAY is None:
        _BPRP_RELAY = _BprpRelay()
    return _BPRP_RELAY


def find_main_window(w):
    """Walk up to the window that owns the document manager."""
    while w is not None:
        if getattr(w, "docman", None) is not None or getattr(w, "doc_manager", None) is not None:
            return w
        w = w.parent()
    return None


class AppContext:
    """What the panel needs from SASpro: the open images, the active one, a way
    to open a new document, and the log. (The original script got these from
    SASpro's scripting context.)"""

    def __init__(self, main_window):
        self.app = main_window

    def log(self, msg):
        _log.info(str(msg))

    def _dm(self):
        mw = self.app
        return None if mw is None else (getattr(mw, "docman", None) or getattr(mw, "doc_manager", None))

    def list_image_views(self):
        out = []
        mdi = getattr(self.app, "mdi", None)
        if mdi is None:
            return out
        for sw in mdi.subWindowList():
            w = sw.widget()
            doc = getattr(w, "document", None)
            if doc is not None:
                out.append((sw.windowTitle(), doc))
        return out

    def active_document(self):
        mdi = getattr(self.app, "mdi", None)
        sw = mdi.activeSubWindow() if mdi is not None else None
        return getattr(sw.widget(), "document", None) if sw is not None else None

    def open_new_document(self, arr, metadata=None, name=None):
        dm = self._dm()
        if dm is None or not hasattr(dm, "open_array"):
            raise RuntimeError("SASpro's document manager is not available.")
        return dm.open_array(arr, metadata=metadata, title=name)


# -----------------------------------------------------------------------------
# Preview widgets
# -----------------------------------------------------------------------------
class _PictureItem(QGraphicsItem):
    def __init__(self, pic: QPicture, w: float, h: float):
        super().__init__()
        self.pic, self.w, self.h = pic, w, h
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def boundingRect(self):
        return QRectF(0, 0, self.w, self.h)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawPicture(0, 0, self.pic)


class _ChartView(QGraphicsView):
    K = 10.0     # scene units per mm

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor("#2b2b2b")))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.on_move = None
        self._item = None

    def set_picture(self, pic, w_mm, h_mm, keep_view=False):
        self.scene().clear()
        self._item = _PictureItem(pic, w_mm * self.K, h_mm * self.K)
        self.scene().addItem(self._item)
        self.scene().setSceneRect(self._item.boundingRect())
        if not keep_view:
            self.fit()

    def fit(self):
        if self._item is not None:
            self.fitInView(self._item.boundingRect().adjusted(-20, -20, 20, 20),
                           Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, f):
        s = self.transform().m11() * f
        if 0.02 < s < 60:
            self.scale(f, f)

    def one_to_one(self):
        # true size on screen
        dpi = self.logicalDpiX() or 96
        s = dpi / 25.4 / self.K
        self.setTransform(QTransform().scale(s, s))

    def wheelEvent(self, e):
        self.zoom(1.25 if e.angleDelta().y() > 0 else 0.8)

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        if self.on_move is not None:
            sp = self.mapToScene(e.position().toPoint())
            self.on_move(sp.x() / self.K, sp.y() / self.K)


# -----------------------------------------------------------------------------
# Dialog
# -----------------------------------------------------------------------------
class ColorButton(QPushButton):
    """A button that shows a colour swatch and opens a colour picker."""

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedWidth(70)
        self._color = color
        self.callbacks = []
        self.clicked.connect(self._pick)
        self._paint()

    def color(self):
        return self._color

    def setColor(self, c):
        self._color = QColor(c).name()
        self._paint()

    def _paint(self):
        c = QColor(self._color)
        fg = "#000000" if c.lightness() > 128 else "#ffffff"
        self.setText(self._color)
        self.setStyleSheet(f"QPushButton {{ background:{self._color}; color:{fg}; border:1px solid #777; }}")

    def _pick(self):
        from PyQt6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(QColor(self._color), self, "Colour")
        if c.isValid():
            self.setColor(c.name())
            for cb in self.callbacks:
                cb()


class StarNamesDialog(QDialog):
    """Choose which proper star names are written on the chart."""

    def __init__(self, data, chosen, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Star Names")
        self.resize(420, 560)
        v = QVBoxLayout(self)
        self.ck_auto = QCheckBox("Automatic (bright stars, by magnitude)")
        v.addWidget(self.ck_auto)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Mag.", "Constellation"])
        self.tree.setRootIsDecorated(False)
        names = {}
        for i in np.nonzero(data.s_name != "")[0]:
            nm = str(data.s_name[i])
            if nm not in names or data.s_mag[i] < names[nm][0]:
                names[nm] = (float(data.s_mag[i]), str(data.s_con[i]))
        self.items = []
        for nm, (mg, con) in sorted(names.items(), key=lambda kv: kv[1][0]):
            it = QTreeWidgetItem([nm, f"{mg:.2f}", con])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked if (nm in chosen or (not chosen and mg <= 3.5))
                             else Qt.CheckState.Unchecked)
            self.tree.addTopLevelItem(it)
            self.items.append(it)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.tree, 1)
        self.ck_auto.setChecked(not chosen)
        self.ck_auto.toggled.connect(lambda on: self.tree.setEnabled(not on))
        self.tree.setEnabled(bool(chosen))
        row = QHBoxLayout()
        for txt, fn in (("All", lambda: self._set(True)), ("None", lambda: self._set(False)),
                        ("Brighter than 2.5", lambda: self._by_mag(2.5)),
                        ("Brighter than 4", lambda: self._by_mag(4.0))):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        row.addStretch(1)
        ok, cancel = QPushButton("OK"), QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)

    def _set(self, on):
        for it in self.items:
            it.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def _by_mag(self, lim):
        for it in self.items:
            it.setCheckState(0, Qt.CheckState.Checked if float(it.text(1)) <= lim else Qt.CheckState.Unchecked)

    def chosen(self):
        if self.ck_auto.isChecked():
            return []
        return [it.text(0) for it in self.items if it.checkState(0) == Qt.CheckState.Checked]

    def total(self):
        return len(self.items)


class DsoStylesDialog(QDialog):
    """Line type, line colour and fill colour of every deep-sky object type."""

    def __init__(self, styles, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deep Sky Object Styles")
        v = QVBoxLayout(self)
        lab = QLabel("The styles of the symbols and outlines of deep sky objects. Nebula shapes "
                     "(H-alpha, dust and OpenNGC outlines) use the colours of their types. "
                     "A fill opacity of zero leaves the objects of a type unfilled.")
        lab.setWordWrap(True)
        v.addWidget(lab)
        g = QGridLayout()
        for c, t in enumerate(("Object type", "Line type", "Line color", "Opacity", "Fill color", "Opacity")):
            g.addWidget(QLabel(f"<i>{t}</i>"), 0, c)
        self.rows = {}
        for r, name in enumerate(DSO_STYLE_DEFAULTS, start=1):
            lt, lc, la, fc, fa = (styles or {}).get(name) or DSO_STYLE_DEFAULTS[name]
            cb = QComboBox()
            cb.addItems(LINE_TYPES)
            cb.setCurrentText(lt)
            lcol, fcol = ColorButton(lc), ColorButton(fc)
            las, fas = QSpinBox(), QSpinBox()
            for sp, val in ((las, la), (fas, fa)):
                sp.setRange(0, 255)
                sp.setValue(int(val))
            g.addWidget(QLabel(name), r, 0)
            g.addWidget(cb, r, 1)
            g.addWidget(lcol, r, 2)
            g.addWidget(las, r, 3)
            g.addWidget(fcol, r, 4)
            g.addWidget(fas, r, 5)
            self.rows[name] = (cb, lcol, las, fcol, fas)
        v.addLayout(g)
        row = QHBoxLayout()
        bd = QPushButton("Defaults")
        bd.clicked.connect(self._defaults)
        row.addWidget(bd)
        row.addStretch(1)
        ok, cancel = QPushButton("OK"), QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        v.addLayout(row)

    def _defaults(self):
        for name, (cb, lcol, las, fcol, fas) in self.rows.items():
            lt, lc, la, fc, fa = DSO_STYLE_DEFAULTS[name]
            cb.setCurrentText(lt)
            lcol.setColor(lc)
            las.setValue(la)
            fcol.setColor(fc)
            fas.setValue(fa)

    def styles(self):
        return {name: (cb.currentText(), lcol.color(), las.value(), fcol.color(), fas.value())
                for name, (cb, lcol, las, fcol, fas) in self.rows.items()}


# which layers have markers (M) and / or labels (L)
LAYER_ML = {
    "milkyway": (True, False), "halpha": (True, False), "dust": (True, False), "grid": (True, True),
    "ecliptic": (True, False), "galactic": (True, False), "borders": (True, False), "lines": (True, False),
    "connames": (False, True), "messier": (True, True), "ngc": (True, True), "ic": (True, True),
    "other": (True, True), "sharpless": (True, True), "lbn": (True, True), "ldn": (True, True),
    "abell": (True, True), "png": (True, True), "outlines": (True, False), "stars": (True, True),
    "gaia": (True, False), "starnames": (False, True), "bayer": (False, True), "variable": (True, False),
    "double": (True, False),
}

# parameters that are not preferences (never saved / restored)
_NOT_SAVED = ("outline", "outline_name", "image_mag", "image_stars", "ref_grid")


class SkyChartPanel(QWidget):
    """The drawn sky chart, embedded in Finder Chart."""

    def __init__(self, ctx=None, parent=None):
        super().__init__(parent)
        self._bprp_busy = False
        _bprp_relay().done.connect(self._on_bprp_done)
        self.ctx = ctx
        self.data = None
        self.renderer = None
        self._pic = None
        self._loading = False
        self._history = []
        self._hist_pos = -1
        self._completer_ready = False
        self.params = self._load_settings()
        self._build_ui()
        self._params_to_ui(self.params)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(800)
        self._save_timer.timeout.connect(self._save_settings)

    # ---- logging ---------------------------------------------------------
    def log(self, msg):
        try:
            if self.ctx is not None:
                self.ctx.log(str(msg))
            else:
                _log.info(str(msg))
        except Exception:
            pass

    # ---- settings (all preferences persist between searches and sessions) --
    def _load_settings(self) -> ChartParams:
        p = ChartParams()
        try:
            raw = QSettings().value(_SETTINGS_KEY, "", type=str) or ""
            j = json.loads(raw) if raw else {}
            for k, v in j.items():
                if k in _NOT_SAVED or not hasattr(p, k):
                    continue
                if k == "layers":
                    p.layers.update({a: bool(b) for a, b in v.items() if a in p.layers})
                elif k == "dso_styles":
                    p.dso_styles = {a: tuple(b) for a, b in v.items() if a in DSO_STYLE_DEFAULTS}
                else:
                    setattr(p, k, v)
        except Exception:
            pass
        return p

    def _save_settings(self):
        try:
            j = asdict(self._ui_to_params())
            for k in _NOT_SAVED:
                j.pop(k, None)
            QSettings().setValue(_SETTINGS_KEY, json.dumps(j))
        except Exception as e:
            self.log(f"Settings not saved: {e}")

    def _schedule_save(self):
        if not self._loading and hasattr(self, "_save_timer") and not getattr(self, "_closed", False):
            self._save_timer.start()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # ---- left: layers + preview
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.layer_label = QLabel("Layers")
        lv.addWidget(self.layer_label)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer", "M", "L", "Description"])
        self.tree.setRootIsDecorated(False)
        self.tree.setMaximumHeight(210)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.headerItem().setToolTip(1, "Markers: draw the objects of the layer")
        self.tree.headerItem().setToolTip(2, "Labels: write the names of the objects of the layer")
        self.layer_checks, self.mark_checks, self.label_checks = {}, {}, {}
        for key, name, desc, _on in LAYERS:
            it = QTreeWidgetItem(["", "", "", desc])
            it.setData(0, Qt.ItemDataRole.UserRole, key)
            it.setToolTip(3, desc)
            self.tree.addTopLevelItem(it)
            cb = QCheckBox(name)
            cb.setToolTip(desc)
            cb.toggled.connect(self._layer_toggled)
            self.tree.setItemWidget(it, 0, cb)
            self.layer_checks[key] = cb
            has_m, has_l = LAYER_ML.get(key, (True, False))
            for col, has, store, tip in ((1, has_m, self.mark_checks, "Markers"),
                                         (2, has_l, self.label_checks, "Labels")):
                if has:
                    c = QCheckBox()
                    c.setChecked(True)
                    c.setToolTip(f"{tip} of {name}")
                    c.toggled.connect(self._layer_toggled)
                    self.tree.setItemWidget(it, col, c)
                    store[key] = c
        self.tree.itemClicked.connect(self._layer_row_clicked)
        lv.addWidget(self.tree)
        lrow = QHBoxLayout()
        for txt, on in (("All", True), ("None", False)):
            b = QPushButton(txt)
            b.setFixedWidth(60)
            b.clicked.connect(lambda _=False, v=on: self._set_all_layers(v))
            lrow.addWidget(b)
        self.ck_auto = QCheckBox("Update the preview automatically")
        self.ck_auto.setChecked(True)
        lrow.addWidget(self.ck_auto)
        lrow.addStretch(1)
        lv.addLayout(lrow)
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(700)
        self._auto_timer.timeout.connect(lambda: self.preview(keep_view=True))

        bar = QHBoxLayout()
        for txt, tip, fn in (("+", "Zoom in", lambda: self.view.zoom(1.25)),
                             ("−", "Zoom out", lambda: self.view.zoom(0.8)),
                             ("Fit", "Fit the page in the view", lambda: self.view.fit()),
                             ("1:1", "True size", lambda: self.view.one_to_one())):
            b = QPushButton(txt)
            b.setToolTip(tip)
            b.setFixedWidth(44)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch(1)
        self.chart_label = QLabel("")
        bar.addWidget(self.chart_label)
        bar.addStretch(1)
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setDefault(True)
        self.btn_preview.clicked.connect(lambda: self.preview())
        bar.addWidget(self.btn_preview)
        lv.addLayout(bar)

        self.view = _ChartView()
        self.view.on_move = self._on_mouse
        lv.addWidget(self.view, 1)
        self.status = QLabel(" ")
        lv.addWidget(self.status)
        hrow = QHBoxLayout()
        hrow.addWidget(QLabel("History:"))
        self.cb_hist = QComboBox()
        self.cb_hist.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        hrow.addWidget(self.cb_hist, 1)
        self.btn_hprev, self.btn_hnext = QPushButton("◀"), QPushButton("▶")
        for b, d in ((self.btn_hprev, -1), (self.btn_hnext, 1)):
            b.setFixedWidth(34)
            b.clicked.connect(lambda _=False, d=d: self._history_step(d))
            hrow.addWidget(b)
        self.btn_restore = QPushButton("Restore")
        self.btn_restore.setToolTip("Restore the chart parameters of the selected history entry")
        self.btn_restore.clicked.connect(self._history_restore)
        hrow.addWidget(self.btn_restore)
        lv.addLayout(hrow)
        split.addWidget(left)

        # ---- right: tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_chart(), "Chart")
        self.tabs.addTab(self._tab_contents(), "Contents")
        self.tabs.addTab(self._tab_rendering(), "Rendering")
        self.tabs.addTab(self._tab_output(), "Output")
        self.tabs.setMinimumWidth(430)
        split.addWidget(self.tabs)
        self._wire_auto_preview()
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)

        # ---- bottom
        bot = QHBoxLayout()
        b_reset = QPushButton("Reset")
        b_reset.setToolTip("Restore the default parameters")
        b_reset.clicked.connect(self._reset)
        bot.addWidget(b_reset)
        bot.addStretch(1)
        self.btn_run = QPushButton("Create Outputs")
        self.btn_run.setToolTip("Generate the chart and write the outputs chosen in the Output tab")
        self.btn_run.clicked.connect(self.run)
        bot.addWidget(self.btn_run)
        root.addLayout(bot)

    def _dspin(self, lo, hi, step, dec=2, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _ispin(self, lo, hi, suffix=""):
        s = QSpinBox()
        s.setRange(lo, hi)
        if suffix:
            s.setSuffix(suffix)
        return s

    def _tab_chart(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Center")
        f = QGridLayout(g)
        self.sp_rah, self.sp_ram, self.sp_ras = self._ispin(0, 23), self._ispin(0, 59), self._dspin(0, 59.99, 1, 2)
        self.sp_dcd, self.sp_dcm, self.sp_dcs = self._ispin(0, 90), self._ispin(0, 59), self._dspin(0, 59.99, 1, 2)
        self.ck_south = QCheckBox("S")
        self.ck_south.setToolTip("Southern declination")
        ra_row = QHBoxLayout()
        for wdg, unit in ((self.sp_rah, "h"), (self.sp_ram, "m"), (self.sp_ras, "s")):
            ra_row.addWidget(wdg)
            ra_row.addWidget(QLabel(unit))
        dec_row = QHBoxLayout()
        for wdg, unit in ((self.sp_dcd, "°"), (self.sp_dcm, "′"), (self.sp_dcs, "″")):
            dec_row.addWidget(wdg)
            dec_row.addWidget(QLabel(unit))
        dec_row.addWidget(self.ck_south)
        f.addWidget(QLabel("Right Ascension:"), 0, 0)
        f.addLayout(ra_row, 0, 1)
        f.addWidget(QLabel("Declination:"), 1, 0)
        f.addLayout(dec_row, 1, 1)
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Object or constellation: M 57, NGC 7000, Heart Nebula, Vega, Lyra...")
        self.ed_search.returnPressed.connect(self.search)
        self.ed_search.textEdited.connect(self._ensure_completer)
        b_s = QPushButton("Search")
        b_s.clicked.connect(self.search)
        b_i = QPushButton("From image...")
        b_i.setToolTip("Choose one of the plate-solved images open in SASpro and take its\n"
                       "center, field, orientation and outline (the active one is preselected)")
        b_i.clicked.connect(self.from_image)
        srow = QHBoxLayout()
        srow.addWidget(self.ed_search, 1)
        srow.addWidget(b_s)
        f.addLayout(srow, 2, 0, 1, 2)
        f.addWidget(b_i, 3, 1, alignment=Qt.AlignmentFlag.AlignRight)
        v.addWidget(g)

        g = QGroupBox("Field")
        f = QFormLayout(g)
        self.sp_fov = self._dspin(0.05, 300.0, 1.0, 3, " °")
        self.cb_proj = QComboBox()
        self.cb_proj.addItems(PROJECTIONS)
        self.sp_rot = self._dspin(-360.0, 360.0, 1.0, 2, " °")
        self.sp_rot.setToolTip("Angle of celestial north, counter-clockwise from up")
        self.ck_mir = QCheckBox("Mirrored")
        row = QHBoxLayout()
        row.addWidget(self.sp_rot)
        row.addWidget(self.ck_mir)
        f.addRow("Field of view:", self.sp_fov)
        f.addRow("Projection:", self.cb_proj)
        f.addRow("Rotation:", row)
        v.addWidget(g)

        g = QGroupBox("Page")
        f = QFormLayout(g)
        self.cb_fmt = QComboBox()
        self.cb_fmt.addItems(list(PAGE_FORMATS.keys()))
        self.cb_orient = QComboBox()
        self.cb_orient.addItems(["Landscape", "Portrait"])
        row = QHBoxLayout()
        row.addWidget(self.cb_fmt, 1)
        row.addWidget(self.cb_orient)
        self.sp_w = self._dspin(40.0, 3000.0, 1.0, 1, " mm")
        self.sp_h = self._dspin(40.0, 3000.0, 1.0, 1, " mm")
        self.sp_margin = self._dspin(0.0, 100.0, 1.0, 1, " mm")
        self.ed_title = QLineEdit()
        self.ed_title.setPlaceholderText("(automatic: constellation name)")
        f.addRow("Format:", row)
        f.addRow("Width:", self.sp_w)
        f.addRow("Height:", self.sp_h)
        f.addRow("Margin:", self.sp_margin)
        f.addRow("Title:", self.ed_title)
        self.cb_fmt.currentTextChanged.connect(self._fmt_changed)
        self.cb_orient.currentTextChanged.connect(self._fmt_changed)
        v.addWidget(g)

        g = QGroupBox("Image Outline")
        h = QHBoxLayout(g)
        self.lb_outline = QLabel("No image outline")
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self._remove_outline)
        h.addWidget(self.lb_outline, 1)
        h.addWidget(b_rm)
        v.addWidget(g)

        self.lb_info = QLabel("")
        self.lb_info.setWordWrap(True)
        self.lb_info.setStyleSheet("color: gray;")
        v.addWidget(self.lb_info)
        v.addStretch(1)
        return w

    def _tab_contents(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Stars")
        f = QFormLayout(g)
        self.sp_smag = self._dspin(1.0, 18.0, 0.5, 1)
        self.sp_smag.setToolTip("Only stars of this magnitude or brighter are drawn.\n"
                                "Typing a value switches 'Automatic' off.")
        self.ck_smag_auto = QCheckBox("Automatic")
        self.ck_smag_auto.setToolTip("Automatic: the faintest magnitude visible in the image when the chart\n"
                                     "was made with 'From image'; otherwise the magnitude that gives the\n"
                                     "star density below.")
        row = QHBoxLayout()
        row.addWidget(self.sp_smag)
        row.addWidget(self.ck_smag_auto)
        self.sp_density = self._dspin(0.5, 200.0, 1.0, 1, " stars/cm²")
        self.btn_names = QPushButton("Select...")
        self.btn_names.clicked.connect(self._select_names)
        self.lb_names = QLabel("automatic")
        nrow = QHBoxLayout()
        nrow.addWidget(self.btn_names)
        nrow.addWidget(self.lb_names, 1)
        self.sp_smin = self._dspin(0.05, 5.0, 0.05, 2, " mm")
        self.sp_smax = self._dspin(0.5, 20.0, 0.1, 2, " mm")
        self.ck_halos = QCheckBox("Halos")
        self.sp_halo_w = self._dspin(0.0, 3.0, 0.05, 2, " mm")
        self.bt_halo = ColorButton("#ffffff")
        self.sp_halo_a = self._ispin(0, 255)
        hrow = QHBoxLayout()
        hrow.addWidget(self.bt_halo)
        hrow.addWidget(self.sp_halo_a)
        hrow.addStretch(1)
        self.ck_halo_map = QCheckBox("Map background (halos take the map colour)")
        self.ck_colored = QCheckBox("Colored stars (B−V color index)")
        self.ck_colored.setToolTip("Bright stars use the catalog B−V index. Gaia stars are colored from\n"
                                   "their XP spectra when SASpro's Gaia XP spectral library is installed.")
        self.ck_online = QCheckBox("Online Gaia colours for stars beyond the local XP library")
        self.ck_online.setToolTip(
            "Stars without a local XP spectrum (fainter than the installed XP groups) take their\n"
            "BP−RP colour from the Gaia archive (ESA). The local library always has priority.\n"
            "The preview is drawn at once and repainted when the colours arrive; every piece of\n"
            "sky is downloaded only once and kept in a local cache. Only sky areas are sent.")
        self.ck_outlined = QCheckBox("Outlined stars")
        self.bt_outline = ColorButton("#000000")
        self.sp_outline_w = self._dspin(0.01, 1.0, 0.02, 2, " mm")
        f.addRow("Star magnitude limit:", row)
        f.addRow("Max. density:", self.sp_density)
        f.addRow("Star names:", nrow)
        f.addRow("Smallest star:", self.sp_smin)
        f.addRow("Largest star:", self.sp_smax)
        f.addRow("", self.ck_halos)
        f.addRow("Halo width:", self.sp_halo_w)
        f.addRow("Halo color:", hrow)
        f.addRow("", self.ck_halo_map)
        f.addRow("", self.ck_colored)
        f.addRow("", self.ck_online)
        f.addRow("", self.ck_outlined)
        f.addRow("Outline color:", self.bt_outline)
        f.addRow("Outline width:", self.sp_outline_w)
        v.addWidget(g)
        g = QGroupBox("Deep Sky Objects (galaxies, clusters, nebulae — not stars)")
        f = QFormLayout(g)
        self.sp_dmag = self._dspin(1.0, 22.0, 0.5, 1)
        self.sp_dmag.setToolTip("Applies to galaxies, clusters and nebulae only. The stars use the\n"
                                "'Star magnitude limit' of the Stars box above.")
        self.ck_dmag_auto = QCheckBox("Automatic")
        row = QHBoxLayout()
        row.addWidget(self.sp_dmag)
        row.addWidget(self.ck_dmag_auto)
        self.sp_dmin = self._dspin(0.3, 10.0, 0.1, 2, " mm")
        f.addRow("Object magnitude limit:", row)
        f.addRow("Smallest symbol:", self.sp_dmin)
        v.addWidget(g)
        v.addStretch(1)
        for wdg in (self.ck_smag_auto, self.ck_halos, self.ck_halo_map, self.ck_outlined, self.ck_dmag_auto):
            wdg.toggled.connect(self._update_enabled)
        return w

    def _tab_rendering(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Chart")
        f = QFormLayout(g)
        self.sp_ts = self._dspin(0.4, 4.0, 0.05, 2)
        self.sp_gs = self._dspin(0.4, 4.0, 0.05, 2)
        self.bt_bg, self.bt_map, self.bt_frame = ColorButton(), ColorButton(), ColorButton("#000000")
        self.sp_frame_a = self._ispin(0, 255)
        frow = QHBoxLayout()
        frow.addWidget(self.bt_frame)
        frow.addWidget(self.sp_frame_a)
        frow.addStretch(1)
        self.cb_font = QFontComboBox()
        self.ck_bw = QCheckBox("Black and white")
        self.ck_bw.setToolTip("For visual observation sessions and monochrome printed media")
        self.ck_dupes = QCheckBox("Remove duplicate objects")
        self.ck_labelopt = QCheckBox("Label placement optimization")
        self.ck_overlap = QCheckBox("Overlappable stars")
        self.ck_overlap.setToolTip("Labels may be written over faint stars")
        self.ck_outlabels = QCheckBox("Outlined labels")
        self.ck_leader = QCheckBox("Leader lines")
        self.bt_leader = ColorButton("#303030")
        self.sp_leader_w = self._dspin(0.02, 1.0, 0.02, 2, " mm")
        self.sp_reduction = self._dspin(0.2, 1.0, 0.05, 2)
        self.sp_reduction.setToolTip("Labels that do not fit are retried at this fraction of their size")
        f.addRow("Text scale:", self.sp_ts)
        f.addRow("Graphics scale:", self.sp_gs)
        f.addRow("Background:", self.bt_bg)
        f.addRow("Map background:", self.bt_map)
        f.addRow("Frame and legend:", frow)
        f.addRow("Font:", self.cb_font)
        f.addRow("", self.ck_bw)
        f.addRow("", self.ck_dupes)
        f.addRow("", self.ck_labelopt)
        f.addRow("", self.ck_overlap)
        f.addRow("", self.ck_outlabels)
        f.addRow("", self.ck_leader)
        f.addRow("Leader line color:", self.bt_leader)
        f.addRow("Leader line width:", self.sp_leader_w)
        f.addRow("Reduction factor:", self.sp_reduction)
        v.addWidget(g)
        g = QGroupBox("Deep Sky Objects")
        h = QHBoxLayout(g)
        b = QPushButton("Edit...")
        b.clicked.connect(self._edit_styles)
        self.lb_styles = QLabel("")
        h.addWidget(QLabel("Object styles:"))
        h.addWidget(b)
        h.addWidget(self.lb_styles, 1)
        v.addWidget(g)
        self.gb_legend = QGroupBox("Legend")
        self.gb_legend.setCheckable(True)
        f = QFormLayout(self.gb_legend)
        self.cb_legend_pos = QComboBox()
        self.cb_legend_pos.addItems(["Bottom", "Top"])
        self.cb_tfont, self.cb_xfont = QFontComboBox(), QFontComboBox()
        self.sp_tsize, self.sp_xsize = self._dspin(4, 48, 0.5, 1, " pt"), self._dspin(3, 24, 0.5, 1, " pt")
        self.ck_tbold, self.ck_titalic = QCheckBox("Bold"), QCheckBox("Italic")
        self.ck_xbold, self.ck_xitalic = QCheckBox("Bold"), QCheckBox("Italic")
        for fcb, sp, bb, ii, label in ((self.cb_tfont, self.sp_tsize, self.ck_tbold, self.ck_titalic, "Title font:"),
                                       (self.cb_xfont, self.sp_xsize, self.ck_xbold, self.ck_xitalic, "Text font:")):
            r1 = QHBoxLayout()
            r1.addWidget(fcb, 1)
            r1.addWidget(sp)
            r2 = QHBoxLayout()
            r2.addWidget(bb)
            r2.addWidget(ii)
            r2.addStretch(1)
            f.addRow(label, r1)
            f.addRow("", r2)
        f.insertRow(0, "Position:", self.cb_legend_pos)
        v.addWidget(self.gb_legend)
        v.addStretch(1)
        self.ck_leader.toggled.connect(self._update_enabled)
        return w

    def _tab_output(self):
        w = QWidget()
        v = QVBoxLayout(w)

        def file_row(ext):
            ed = QLineEdit()
            b = QPushButton("...")
            b.setFixedWidth(30)
            b.clicked.connect(lambda: self._browse(ed, ext))
            h = QHBoxLayout()
            h.addWidget(ed, 1)
            h.addWidget(b)
            return ed, h

        self.gb_svg = QGroupBox("SVG")
        self.gb_svg.setCheckable(True)
        f = QFormLayout(self.gb_svg)
        self.ed_svg, h = file_row("svg")
        f.addRow("File:", h)
        self.ck_paths = QCheckBox("Convert text to paths")
        self.ck_paths.setToolTip("Recommended: the chart looks the same in every application.\n"
                                 "Disable to keep selectable text (the fonts must be available to the viewer).")
        f.addRow("", self.ck_paths)
        self.gb_pdf = QGroupBox("PDF")
        self.gb_pdf.setCheckable(True)
        f = QFormLayout(self.gb_pdf)
        self.ed_pdf, h = file_row("pdf")
        f.addRow("File:", h)
        self.gb_png = QGroupBox("PNG file")
        self.gb_png.setCheckable(True)
        f = QFormLayout(self.gb_png)
        self.ed_png, h = file_row("png")
        f.addRow("File:", h)
        self.gb_image = QGroupBox("Image (new document in SASpro)")
        self.gb_image.setCheckable(True)
        f = QFormLayout(self.gb_image)
        self.ed_imgid = QLineEdit()
        self.ed_imgid.setPlaceholderText("(automatic: chart title)")
        f.addRow("Identifier:", self.ed_imgid)
        self.ck_match = QCheckBox("Match the reference image (same size and pixel grid)")
        self.ck_match.setToolTip(
            "The chart is drawn exactly on the pixel grid of the image loaded with 'From Image':\n"
            "same size, scale, orientation and projection, without margins or legend, so that\n"
            "Link Pan/Zoom, blinking or overlays line it up pixel for pixel with the image.\n"
            "Every chart sent to SASpro carries its astrometric solution (WCS) anyway.")
        f.addRow("", self.ck_match)
        for gb in (self.gb_svg, self.gb_pdf, self.gb_png, self.gb_image):
            v.addWidget(gb)
        g = QGroupBox("Raster resolution (PNG and image)")
        f = QFormLayout(g)
        self.sp_res = self._dspin(10.0, 2400.0, 10.0, 2)
        self.cb_resunit = QComboBox()
        self.cb_resunit.addItems(["pixels per inch", "pixels per cm"])
        row = QHBoxLayout()
        row.addWidget(self.sp_res)
        row.addWidget(self.cb_resunit)
        self.lb_px = QLabel("")
        self.lb_px.setStyleSheet("color: gray;")
        f.addRow("Resolution:", row)
        f.addRow("", self.lb_px)
        v.addWidget(g)
        note = QLabel("Files with an empty name are asked for when the chart is run.\n\n"
                      "Data: SASpro's local Gaia DR3 library (stars and Milky Way), HYG 4.1 (bright stars, "
                      "names and colors, CC BY-SA), OpenNGC (catalog and nebula outlines, CC BY-SA), "
                      "Finkbeiner H-alpha and SFD dust maps (NASA LAMBDA), d3-celestial (constellations, BSD), "
                      "SASpro catalogs.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        v.addWidget(note)
        v.addStretch(1)
        self.sp_res.valueChanged.connect(self._update_px)
        self.cb_resunit.currentTextChanged.connect(self._update_px)
        return w

    # ---- helpers --------------------------------------------------------------
    def _browse(self, ed, ext):
        start = ed.text().strip() or str(Path(getattr(self, "_last_dir", str(Path.home()))) / f"chart.{ext}")
        fn, _ = QFileDialog.getSaveFileName(self, f"Sky Chart: {ext.upper()} file", start,
                                            f"{ext.upper()} files (*.{ext})")
        if fn:
            if not fn.lower().endswith("." + ext):
                fn += "." + ext
            ed.setText(fn)
            self._last_dir = str(Path(fn).parent)
            self._schedule_save()

    def _set_center(self, ra, dec):
        ra = ra % 360.0
        tot = round(ra / 15.0 * 3600.0, 2)
        if tot >= 86400:
            tot -= 86400
        self.sp_rah.setValue(int(tot // 3600))
        self.sp_ram.setValue(int((tot % 3600) // 60))
        self.sp_ras.setValue(round(tot % 60, 2))
        t = round(abs(dec) * 3600.0, 2)
        self.sp_dcd.setValue(int(t // 3600))
        self.sp_dcm.setValue(int((t % 3600) // 60))
        self.sp_dcs.setValue(round(t % 60, 2))
        self.ck_south.setChecked(dec < 0)

    def _center(self):
        ra = (self.sp_rah.value() + self.sp_ram.value() / 60.0 + self.sp_ras.value() / 3600.0) * 15.0
        dec = self.sp_dcd.value() + self.sp_dcm.value() / 60.0 + self.sp_dcs.value() / 3600.0
        dec = min(dec, 90.0)
        return ra % 360.0, (-dec if self.ck_south.isChecked() else dec)

    def _update_enabled(self, *_):
        self.sp_density.setEnabled(self.ck_smag_auto.isChecked())
        halos = self.ck_halos.isChecked()
        self.sp_halo_w.setEnabled(halos)
        self.ck_halo_map.setEnabled(halos)
        self.bt_halo.setEnabled(halos and not self.ck_halo_map.isChecked())
        self.sp_halo_a.setEnabled(halos and not self.ck_halo_map.isChecked())
        out = self.ck_outlined.isChecked()
        self.bt_outline.setEnabled(out)
        self.sp_outline_w.setEnabled(out)
        lead = self.ck_leader.isChecked()
        self.bt_leader.setEnabled(lead)
        self.sp_leader_w.setEnabled(lead)

    def _update_px(self, *_):
        try:
            p = self._ui_to_params()
            w, h = p.page_size()
            k = p.dpi() / 25.4
            self.lb_px.setText(f"{int(round(w * k))} × {int(round(h * k))} px")
        except Exception:
            pass

    def _update_styles_label(self):
        n = len(self.params.dso_styles or {})
        self.lb_styles.setText("custom" if n else "defaults")

    def _update_names_label(self):
        names = self.params.star_names or []
        total = getattr(self, "_n_names", None)
        if not names:
            self.lb_names.setText("automatic (by magnitude)")
        else:
            self.lb_names.setText(f"{len(names)} of {total} names" if total else f"{len(names)} names")

    def _select_names(self):
        if not self._ensure_data():
            return
        dlg = StarNamesDialog(self.data, set(self.params.star_names or []), self)
        if dlg.exec():
            self.params.star_names = dlg.chosen()
            self._n_names = dlg.total()
            self._update_names_label()
            self.schedule_preview()
            self._schedule_save()

    def _edit_styles(self):
        dlg = DsoStylesDialog(self.params.dso_styles, self)
        if dlg.exec():
            st = dlg.styles()
            self.params.dso_styles = {} if st == {k: tuple(v) for k, v in DSO_STYLE_DEFAULTS.items()} else st
            self._update_styles_label()
            self.schedule_preview()
            self._schedule_save()

    def _fmt_changed(self, *_):
        custom = self.cb_fmt.currentText() == "Custom"
        self.sp_w.setEnabled(custom)
        self.sp_h.setEnabled(custom)
        self.cb_orient.setEnabled(not custom)
        if not custom:
            w, h = PAGE_FORMATS[self.cb_fmt.currentText()]
            land = self.cb_orient.currentText() == "Landscape"
            self.sp_w.setValue(max(w, h) if land else min(w, h))
            self.sp_h.setValue(min(w, h) if land else max(w, h))

    def _update_layer_label(self, *_):
        n = sum(cb.isChecked() for cb in self.layer_checks.values())
        self.layer_label.setText(f"<b>Layers ({n} of {len(self.layer_checks)} checked)</b>")

    def _layer_row_clicked(self, item, column):
        cb = self.layer_checks.get(item.data(0, Qt.ItemDataRole.UserRole))
        if cb is not None and column == 3:
            cb.toggle()

    def _set_all_layers(self, on):
        self._bulk = True
        try:
            for cb in self.layer_checks.values():
                cb.setChecked(on)
        finally:
            self._bulk = False
        self._layer_toggled()

    def _layer_toggled(self, *_):
        self._update_layer_label()
        for key, cb in self.layer_checks.items():
            for store in (self.mark_checks, self.label_checks):
                if key in store:
                    store[key].setEnabled(cb.isChecked())
        if getattr(self, "_bulk", False) or self._loading:
            return
        self.schedule_preview()

    def _wire_auto_preview(self):
        checks = (self.ck_mir, self.ck_smag_auto, self.ck_halos, self.ck_colored, self.ck_online, self.ck_dmag_auto, self.ck_bw,
                  self.ck_south, self.ck_halo_map, self.ck_outlined, self.ck_dupes, self.ck_labelopt,
                  self.ck_overlap, self.ck_outlabels, self.ck_leader, self.ck_tbold, self.ck_titalic,
                  self.ck_xbold, self.ck_xitalic)
        for cb in checks:
            cb.toggled.connect(lambda *_: self.schedule_preview())
        self.gb_legend.toggled.connect(lambda *_: self.schedule_preview())
        for co in (self.cb_proj, self.cb_fmt, self.cb_orient, self.cb_font, self.cb_legend_pos,
                   self.cb_tfont, self.cb_xfont):
            co.currentTextChanged.connect(lambda *_: self.schedule_preview())
        spins = (self.sp_fov, self.sp_rot, self.sp_w, self.sp_h, self.sp_margin, self.sp_smag, self.sp_smin,
                 self.sp_smax, self.sp_dmag, self.sp_dmin, self.sp_ts, self.sp_gs, self.sp_rah, self.sp_ram,
                 self.sp_ras, self.sp_dcd, self.sp_dcm, self.sp_dcs, self.sp_density, self.sp_halo_w,
                 self.sp_halo_a, self.sp_outline_w, self.sp_frame_a, self.sp_leader_w, self.sp_reduction,
                 self.sp_tsize, self.sp_xsize)
        for sp in spins:
            sp.setKeyboardTracking(True)           # typed values apply after a short pause
            sp.valueChanged.connect(lambda *_: self.schedule_preview())
        # typing a magnitude limit switches its "Automatic" off
        self.sp_smag.valueChanged.connect(lambda *_: self._manual_limit(self.ck_smag_auto))
        self.sp_dmag.valueChanged.connect(lambda *_: self._manual_limit(self.ck_dmag_auto))
        for bt in (self.bt_halo, self.bt_outline, self.bt_bg, self.bt_map, self.bt_frame, self.bt_leader):
            bt.callbacks.append(self.schedule_preview)
        self.ed_title.editingFinished.connect(lambda: self.schedule_preview())
        # output options are preferences too
        for gb in (self.gb_svg, self.gb_pdf, self.gb_png, self.gb_image):
            gb.toggled.connect(lambda *_: self._schedule_save())
        for ed in (self.ed_svg, self.ed_pdf, self.ed_png, self.ed_imgid):
            ed.editingFinished.connect(self._schedule_save)
        self.ck_paths.toggled.connect(lambda *_: self._schedule_save())
        self.sp_res.valueChanged.connect(lambda *_: self._schedule_save())
        self.cb_resunit.currentTextChanged.connect(lambda *_: self._schedule_save())
        self.ck_auto.toggled.connect(lambda *_: self._schedule_save())

    def _manual_limit(self, ck):
        if self._loading or not ck.isChecked():
            return
        self._loading = True
        ck.setChecked(False)
        self._loading = False
        self._update_enabled()

    def _commit_spins(self):
        from PyQt6.QtWidgets import QAbstractSpinBox
        self._loading, was = True, self._loading
        try:
            for sp in self.findChildren(QAbstractSpinBox):
                sp.interpretText()
        finally:
            self._loading = was

    def schedule_preview(self):
        """Regenerate the preview shortly after a change, if one is shown."""
        if self._loading or not hasattr(self, "ck_auto"):
            return
        self._schedule_save()
        if self.renderer is not None and self.ck_auto.isChecked():
            self._auto_timer.start()

    # ---- UI <-> params -----------------------------------------------------
    def _params_to_ui(self, p: ChartParams):
        self._loading = True
        try:
            self._params_to_ui_impl(p)
        finally:
            self._loading = False
        self._update_enabled()
        self._update_px()
        self.schedule_preview()

    def _params_to_ui_impl(self, p: ChartParams):
        keep = getattr(self, "params", None)
        self._set_center(p.ra, p.dec)
        self.sp_fov.setValue(p.fov)
        self.cb_proj.setCurrentText(p.projection)
        self.sp_rot.setValue(p.rotation)
        self.ck_mir.setChecked(p.mirrored)
        self.cb_fmt.setCurrentText(p.page_format if p.page_format in PAGE_FORMATS else "Custom")
        self.cb_orient.setCurrentText("Landscape" if p.landscape else "Portrait")
        self.sp_w.setValue(p.page_w)
        self.sp_h.setValue(p.page_h)
        self._fmt_changed()
        self.sp_margin.setValue(p.margin)
        self.ed_title.setText(p.title)
        for key, cb in self.layer_checks.items():
            cb.setChecked(bool(p.layers.get(key, False)))
        for key, cb in self.mark_checks.items():
            cb.setChecked(bool(p.layer_marks.get(key, True)))
        for key, cb in self.label_checks.items():
            cb.setChecked(bool(p.layer_labels.get(key, True)))
        self._layer_toggled()
        self._update_layer_label()
        self.sp_smag.setValue(p.star_mag)
        self.ck_smag_auto.setChecked(p.star_mag_auto)
        self.sp_density.setValue(p.star_density)
        self.sp_smin.setValue(p.star_min)
        self.sp_smax.setValue(p.star_max)
        self.ck_halos.setChecked(p.halos)
        self.sp_halo_w.setValue(p.halo_width)
        self.bt_halo.setColor(p.halo_color)
        self.sp_halo_a.setValue(int(p.halo_alpha))
        self.ck_halo_map.setChecked(p.halo_map_bg)
        self.ck_colored.setChecked(p.colored_stars)
        self.ck_online.setChecked(p.online_colors)
        self.ck_outlined.setChecked(p.outlined_stars)
        self.bt_outline.setColor(p.star_outline_color)
        self.sp_outline_w.setValue(p.star_outline_width)
        self.sp_dmag.setValue(p.dso_mag)
        self.ck_dmag_auto.setChecked(p.dso_mag_auto)
        self.sp_dmin.setValue(p.dso_min)
        self.sp_ts.setValue(p.text_scale)
        self.sp_gs.setValue(p.graphics_scale)
        self.bt_bg.setColor(p.bg_color)
        self.bt_map.setColor(p.map_color)
        self.bt_frame.setColor(p.frame_color)
        self.sp_frame_a.setValue(int(p.frame_alpha))
        default_family = QApplication.font().family()
        self.cb_font.setCurrentFont(QFont(p.font or default_family))
        self.ck_bw.setChecked(p.black_white)
        self.ck_dupes.setChecked(p.remove_duplicates)
        self.ck_labelopt.setChecked(p.label_opt)
        self.ck_overlap.setChecked(p.overlap_stars)
        self.ck_outlabels.setChecked(p.outlined_labels)
        self.ck_leader.setChecked(p.leader_lines)
        self.bt_leader.setColor(p.leader_color)
        self.sp_leader_w.setValue(p.leader_width)
        self.sp_reduction.setValue(p.reduction)
        self.gb_legend.setChecked(p.legend)
        self.cb_legend_pos.setCurrentText(p.legend_pos)
        for fcb, name in ((self.cb_tfont, p.title_font), (self.cb_xfont, p.text_font)):
            fcb.setCurrentFont(QFont(name or default_family))
        self.sp_tsize.setValue(p.title_size)
        self.sp_xsize.setValue(p.text_size)
        self.ck_tbold.setChecked(p.title_bold)
        self.ck_titalic.setChecked(p.title_italic)
        self.ck_xbold.setChecked(p.text_bold)
        self.ck_xitalic.setChecked(p.text_italic)
        self.gb_svg.setChecked(p.out_svg)
        self.gb_pdf.setChecked(p.out_pdf)
        self.gb_png.setChecked(p.out_png)
        self.gb_image.setChecked(p.out_image)
        self.ed_svg.setText(p.svg_path)
        self.ed_pdf.setText(p.pdf_path)
        self.ed_png.setText(p.png_path)
        self.ed_imgid.setText(p.image_id)
        self.ck_match.setChecked(p.match_image)
        self.sp_res.setValue(p.resolution)
        self.cb_resunit.setCurrentText(p.res_unit)
        self.ck_paths.setChecked(p.text_as_paths)
        if keep is not None:
            p.outline, p.outline_name = keep.outline, keep.outline_name
            p.image_mag, p.image_stars = keep.image_mag, keep.image_stars
            p.ref_grid = keep.ref_grid
        self.params = p
        self._update_outline_label()
        self._update_styles_label()
        self._update_names_label()

    def _ui_to_params(self) -> ChartParams:
        self._commit_spins()
        p = ChartParams()
        p.ra, p.dec = self._center()
        p.fov = self.sp_fov.value()
        p.projection = self.cb_proj.currentText()
        p.rotation = self.sp_rot.value()
        p.mirrored = self.ck_mir.isChecked()
        p.page_format = self.cb_fmt.currentText()
        p.landscape = self.cb_orient.currentText() == "Landscape"
        p.page_w, p.page_h = self.sp_w.value(), self.sp_h.value()
        p.margin = self.sp_margin.value()
        p.title = self.ed_title.text().strip()
        for key, cb in self.layer_checks.items():
            p.layers[key] = cb.isChecked()
        p.layer_marks = {k: cb.isChecked() for k, cb in self.mark_checks.items()}
        p.layer_labels = {k: cb.isChecked() for k, cb in self.label_checks.items()}
        p.star_mag, p.star_mag_auto = self.sp_smag.value(), self.ck_smag_auto.isChecked()
        p.star_density = self.sp_density.value()
        p.star_min, p.star_max = self.sp_smin.value(), self.sp_smax.value()
        p.halos, p.colored_stars = self.ck_halos.isChecked(), self.ck_colored.isChecked()
        p.online_colors = self.ck_online.isChecked()
        p.halo_width, p.halo_color = self.sp_halo_w.value(), self.bt_halo.color()
        p.halo_alpha, p.halo_map_bg = self.sp_halo_a.value(), self.ck_halo_map.isChecked()
        p.outlined_stars = self.ck_outlined.isChecked()
        p.star_outline_color, p.star_outline_width = self.bt_outline.color(), self.sp_outline_w.value()
        p.dso_mag, p.dso_mag_auto = self.sp_dmag.value(), self.ck_dmag_auto.isChecked()
        p.dso_min = self.sp_dmin.value()
        p.text_scale, p.graphics_scale = self.sp_ts.value(), self.sp_gs.value()
        p.bg_color, p.map_color = self.bt_bg.color(), self.bt_map.color()
        p.frame_color, p.frame_alpha = self.bt_frame.color(), self.sp_frame_a.value()
        p.font = self.cb_font.currentFont().family()
        p.black_white = self.ck_bw.isChecked()
        p.remove_duplicates, p.label_opt = self.ck_dupes.isChecked(), self.ck_labelopt.isChecked()
        p.overlap_stars, p.outlined_labels = self.ck_overlap.isChecked(), self.ck_outlabels.isChecked()
        p.leader_lines, p.leader_color = self.ck_leader.isChecked(), self.bt_leader.color()
        p.leader_width, p.reduction = self.sp_leader_w.value(), self.sp_reduction.value()
        p.legend, p.legend_pos = self.gb_legend.isChecked(), self.cb_legend_pos.currentText()
        p.title_font, p.text_font = self.cb_tfont.currentFont().family(), self.cb_xfont.currentFont().family()
        p.title_size, p.text_size = self.sp_tsize.value(), self.sp_xsize.value()
        p.title_bold, p.title_italic = self.ck_tbold.isChecked(), self.ck_titalic.isChecked()
        p.text_bold, p.text_italic = self.ck_xbold.isChecked(), self.ck_xitalic.isChecked()
        p.out_svg, p.out_pdf = self.gb_svg.isChecked(), self.gb_pdf.isChecked()
        p.out_png, p.out_image = self.gb_png.isChecked(), self.gb_image.isChecked()
        p.svg_path, p.pdf_path, p.png_path = self.ed_svg.text().strip(), self.ed_pdf.text().strip(), \
            self.ed_png.text().strip()
        p.image_id = self.ed_imgid.text().strip()
        p.match_image = self.ck_match.isChecked()
        p.resolution, p.res_unit = self.sp_res.value(), self.cb_resunit.currentText()
        p.text_as_paths = self.ck_paths.isChecked()
        cur = self.params
        p.star_names = list(cur.star_names or [])
        p.dso_styles = dict(cur.dso_styles or {})
        p.outline, p.outline_name = cur.outline, cur.outline_name
        p.image_mag, p.image_stars = cur.image_mag, cur.image_stars
        p.ref_grid = cur.ref_grid
        return p

    def _reset(self):
        d = ChartParams()
        cur = self._ui_to_params()
        d.ra, d.dec, d.fov, d.title = cur.ra, cur.dec, cur.fov, cur.title   # keep the field
        self._params_to_ui(d)

    def _update_outline_label(self):
        if self.params.outline:
            self.lb_outline.setText(f"Outline: {self.params.outline_name or 'image'}")
        else:
            self.lb_outline.setText("No image outline")
        try:
            g = self.params.ref_grid
            self.ck_match.setEnabled(bool(g))
            self.ck_match.setText("Match the reference image (same size and pixel grid)" +
                                  (f": {self.params.outline_name or 'image'}, {g['w']} x {g['h']} px" if g else
                                   " - load one with 'From Image'"))
        except Exception:
            pass

    def _remove_outline(self):
        self.params.outline, self.params.outline_name = [], ""
        self.params.image_mag, self.params.image_stars = 0.0, 0
        self.params.ref_grid = {}
        self._update_outline_label()
        if self.renderer is not None:
            self.preview(keep_view=True)

    # ---- history ---------------------------------------------------------------
    def _history_add(self, p):
        snap = asdict(p)
        if self._history and self._history[self._hist_pos][1] == snap:
            return
        del self._history[self._hist_pos + 1:]
        label = (f"{len(self._history) + 1}  {time.strftime('%H:%M:%S')}  "
                 f"{(p.title or fmt_ra(p.ra) + ' ' + fmt_dec(p.dec))}, {p.fov:g}°, "
                 f"{sum(p.layers.values())} layers")
        self._history.append((label, snap))
        self._history = self._history[-30:]
        self._hist_pos = len(self._history) - 1
        self._refresh_history()

    def _refresh_history(self):
        self.cb_hist.blockSignals(True)
        self.cb_hist.clear()
        for label, _snap in self._history:
            self.cb_hist.addItem(label)
        self.cb_hist.setCurrentIndex(self._hist_pos)
        self.cb_hist.blockSignals(False)
        self.btn_hprev.setEnabled(self._hist_pos > 0)
        self.btn_hnext.setEnabled(0 <= self._hist_pos < len(self._history) - 1)

    def _history_step(self, d):
        i = self._hist_pos + d
        if 0 <= i < len(self._history):
            self.cb_hist.setCurrentIndex(i)
            self._history_restore()

    def _history_restore(self):
        i = self.cb_hist.currentIndex()
        if not (0 <= i < len(self._history)):
            return
        snap = dict(self._history[i][1])
        p = ChartParams()
        for k, v in snap.items():
            setattr(p, k, v)
        self._hist_pos = i
        self._restoring = True
        self._params_to_ui(p)
        self.params.outline, self.params.outline_name = snap.get("outline", []), snap.get("outline_name", "")
        self.params.image_mag, self.params.image_stars = snap.get("image_mag", 0.0), snap.get("image_stars", 0)
        self.params.ref_grid = snap.get("ref_grid", {}) or {}
        self._update_outline_label()
        self.preview(keep_view=True)
        self._restoring = False
        self._refresh_history()

    # ---- actions ------------------------------------------------------------
    def _ensure_data(self) -> bool:
        if self.data is not None:
            return True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.data = get_data(self.log)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Sky Chart", f"Could not load the catalogs:\n\n{e}")
            return False
        QApplication.restoreOverrideCursor()
        return True

    def _get_params(self):
        try:
            return self._ui_to_params()
        except Exception as e:
            QMessageBox.warning(self, "Sky Chart", f"Invalid parameters: {e}")
            return None

    def _ensure_completer(self, *_):
        """Suggestions while typing: every name and designation known locally."""
        if self._completer_ready or not self._ensure_data():
            return
        from PyQt6.QtWidgets import QCompleter
        from PyQt6.QtCore import QStringListModel
        names = self.data.search_names()
        comp = QCompleter(self)
        comp.setModel(QStringListModel(names, comp))
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        comp.setFilterMode(Qt.MatchFlag.MatchContains)
        comp.setMaxVisibleItems(14)
        comp.activated[str].connect(self._completer_pick)
        self.ed_search.setCompleter(comp)
        self._completer_ready = True
        comp.setCompletionPrefix(self.ed_search.text())
        comp.complete()

    def _completer_pick(self, text):
        self.ed_search.setText(text)
        self.search()

    def search(self):
        text = self.ed_search.text().strip()
        if not text or not self._ensure_data():
            return
        name = re.sub(r"\s+—\s+.*$", "", text)          # "Lyra — constellation" -> "Lyra"
        view = self.data.view_for(name, self._frame_aspect(), self.cb_proj.currentText())
        if view is None:
            sug = self.data.suggestions(name)
            msg = f"Object not found: {name}"
            if sug:
                msg += "\n\nDid you mean:\n  " + "\n  ".join(sug)
            msg += ("\n\nYou can also type a catalog designation (M 42, NGC 7000, IC 1805, "
                    "Sh2-155, LBN 438, LDN 1622, B 33...) or enter the coordinates.")
            QMessageBox.information(self, "Sky Chart", msg)
            return
        ra, dec, title, fov = view
        self._loading = True
        try:
            self._set_center(ra, dec)
            if fov is not None:
                self.sp_rot.setValue(0.0)
                self.sp_fov.setValue(round(fov, 3))
            elif self.data.object_size(name) * 1.25 > self.sp_fov.value():
                self.sp_fov.setValue(round(self.data.object_size(name) * 1.6, 3))
            if title:
                self.ed_title.setText(title)
        finally:
            self._loading = False
        self.preview()          # new chart: view fitted, centred on the object
        self._schedule_save()

    def _frame_aspect(self):
        try:
            p = self._ui_to_params()
            r = ChartRenderer(p, self.data, lambda m: None) if self.data is not None else None
            if r is not None:
                x0, y0, x1, y1 = r.frame
                return (x1 - x0) / (y1 - y0)
        except Exception:
            pass
        return 1.6

    @staticmethod
    def _base_doc(doc):
        """The real document behind a live view or a preview (ROI) wrapper."""
        if doc is None:
            return None
        base = getattr(doc, "_base", None)
        if base is not None:
            doc = base
        parent = getattr(doc, "_parent_doc", None)
        return parent if parent is not None else doc

    @staticmethod
    def _doc_wcs(doc):
        """Celestial WCS of a SASpro document, or None if it is not plate solved."""
        meta = getattr(doc, "metadata", None) or {}
        wcs = None
        try:
            from setiastro.saspro.finder_chart import get_doc_wcs_from_metadata
            wcs = get_doc_wcs_from_metadata(meta)
        except Exception:
            wcs = None
        if wcs is None:
            try:
                from astropy.wcs import WCS
                hdr = meta.get("wcs_header") or meta.get("original_header") or meta.get("header")
                if hdr is not None:
                    wcs = WCS(hdr, relax=True).celestial
            except Exception:
                wcs = None
        if wcs is None or not getattr(wcs, "has_celestial", True):
            return None
        try:
            if not str(wcs.wcs.ctype[0]).strip():
                return None
        except Exception:
            pass
        return wcs

    def _pick_image_document(self):
        """Ask which open SASpro image to use. Returns (doc, title) or (None, None)."""
        views = []
        try:
            views = list(self.ctx.list_image_views())
        except Exception:
            views = []
        if not views:
            doc = self._base_doc(self.ctx.active_document())
            if doc is None or getattr(doc, "image", None) is None:
                raise RuntimeError("There is no image open in SASpro.")
            return doc, ""
        active = None
        try:
            active = self._base_doc(self.ctx.active_document())
        except Exception:
            active = None
        items, seen = [], set()
        for title, doc in views:
            doc = self._base_doc(doc)
            img = getattr(doc, "image", None)
            if doc is None or img is None or id(doc) in seen:
                continue
            seen.add(id(doc))
            items.append((str(title or "Untitled"), doc, self._doc_wcs(doc) is not None,
                          int(img.shape[1]), int(img.shape[0])))
        if not items:
            raise RuntimeError("There is no image open in SASpro.")
        solved = [it for it in items if it[2]]
        if not solved:
            raise RuntimeError("None of the open images has an astrometric solution.\n"
                               "Plate solve the image first.")
        if len(items) == 1:
            return items[0][1], items[0][0]

        dlg = QDialog(self)
        dlg.setWindowTitle("Choose the reference image")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Image whose centre, field, orientation and outline the chart will use:"))
        lst = QListWidget()
        pick = 0
        for n, (title, doc, ok, w, h) in enumerate(items):
            it = QListWidgetItem(f"{title}    {w} x {h} px    "
                                 f"{'plate solved' if ok else 'no astrometric solution'}")
            if not ok:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable)
            lst.addItem(it)
            if ok and (doc is active or (pick == 0 and not items[0][2])):
                pick = n
        if not items[pick][2]:
            pick = items.index(solved[0])
        lst.setCurrentRow(pick)
        lst.setMinimumWidth(460)
        v.addWidget(lst)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lst.itemDoubleClicked.connect(lambda _it: dlg.accept() if _it.flags() & Qt.ItemFlag.ItemIsEnabled else None)
        v.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None, None
        row = lst.currentRow()
        if row < 0 or not items[row][2]:
            return None, None
        return items[row][1], items[row][0]

    def _fit_fov(self, ra0, dec0, rot, mirrored, ora, odec, fallback):
        """Field of view that makes the image fill the chart frame exactly: its outline
        touches the frame on the limiting side (the chart keeps the page proportions)."""
        try:
            if self.data is None:
                self._ensure_data()             # the frame depends on the full layout
            p = self._ui_to_params()
            p = replace(p, ra=ra0, dec=dec0, rotation=rot, mirrored=bool(mirrored),
                        match_image=False, fov=max(fallback, 0.01))
            r = ChartRenderer(p, self.data) if self.data is not None else None
            if r is not None:
                x0, y0, x1, y1 = r.frame
            else:
                pw, ph = p.page_size()
                x0, y0, x1, y1 = 0.0, 0.0, pw, ph
            fw, fh = x1 - x0, y1 - y0
            pr = Projection(ra0, dec0, p.projection, rot, bool(mirrored))
            pr.scale, pr.cx, pr.cy = 1.0, 0.0, 0.0
            X, Y, _t = pr.project(np.asarray(ora), np.asarray(odec))
            need = max(float(np.max(np.abs(X))), float(np.max(np.abs(Y))) * fw / fh)
            half = float(pr.Rinv(np.array([need]))[0])
            if half == half and half > 0:
                return 2.0 * half * R2D
        except Exception as e:
            self.log(f"Exact image fit not computed: {e}")
        return fallback

    def from_image(self, doc=None, title=""):
        """Centre, field, orientation and outline from a plate-solved image: the
        given document, or one picked from the open images."""
        if doc is None and self.ctx is None:
            QMessageBox.information(self, "Sky Chart", "No SASpro context available.")
            return
        try:
            if doc is None:
                doc, picked_title = self._pick_image_document()
            else:
                doc, picked_title = self._base_doc(doc), title
            if doc is None:
                return
            img = getattr(doc, "image", None)
            if img is None:
                raise RuntimeError("The chosen image has no pixel data.")
            wcs = self._doc_wcs(doc)
            if wcs is None:
                raise RuntimeError(f"'{picked_title or 'The image'}' has no astrometric solution.\n"
                                   "Plate solve it first.")
            h, w = img.shape[0], img.shape[1]

            def sky(px, py):
                c = wcs.pixel_to_world(px, py)
                c = c.icrs if hasattr(c, "icrs") else c
                return np.atleast_1d(c.ra.deg), np.atleast_1d(c.dec.deg)

            n = 24
            t = np.linspace(0, 1, n)
            ex = np.concatenate([t * (w - 1), np.full(n, w - 1), (1 - t) * (w - 1), np.zeros(n)])
            ey = np.concatenate([np.zeros(n), t * (h - 1), np.full(n, h - 1), (1 - t) * (h - 1)])
            ora, odec = sky(ex, ey)
            cra, cdec = sky(np.array([(w - 1) / 2.0]), np.array([(h - 1) / 2.0]))
            ra0, dec0 = float(cra[0]), float(cdec[0])
            # orientation: pixel direction of north and east at the centre
            wr, wd = sky(np.array([(w - 1) / 2.0 + 1.0, (w - 1) / 2.0]), np.array([(h - 1) / 2.0, (h - 1) / 2.0 + 1.0]))
            cd = math.cos(dec0 * D2R)
            J = np.array([[((wr[0] - ra0 + 180) % 360 - 180) * cd, ((wr[1] - ra0 + 180) % 360 - 180) * cd],
                          [wd[0] - dec0, wd[1] - dec0]])          # d(east,north)/d(px,py)
            Ji = np.linalg.inv(J)
            north = Ji @ np.array([0.0, 1.0])      # pixel vector (x right, y down)
            east = Ji @ np.array([1.0, 0.0])
            rot = math.degrees(math.atan2(-north[0], -north[1]))
            mirrored = (east[0] * north[1] - east[1] * north[0]) < 0
            scale_deg = math.sqrt(abs(np.linalg.det(J)))
            fov_img = scale_deg * w
            name = picked_title or ""
            if not name:
                try:
                    name = str(doc.display_name()) if callable(getattr(doc, "display_name", None)) else ""
                except Exception:
                    name = ""
            self._loading = True
            self._set_center(ra0, dec0)
            self.sp_rot.setValue(round(rot, 2))
            self.ck_mir.setChecked(bool(mirrored))
            self.sp_fov.setValue(round(min(self._fit_fov(ra0, dec0, rot, mirrored, ora, odec, fov_img), 240.0), 4))
            self._loading = False
            self.params.outline = [[float(a), float(b)] for a, b in zip(ora, odec)]
            self.params.outline_name = name or f"{w} x {h} px"
            try:
                self.params.ref_grid = ref_grid_from_wcs(wcs, w, h)
            except Exception as e:
                self.params.ref_grid = {}
                self.log(f"Reference image grid not available: {e}")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                lim = self._image_limiting_mag(img, sky, ora, odec)
            finally:
                QApplication.restoreOverrideCursor()
            if lim is not None:
                self.params.image_mag, self.params.image_stars = lim
                self.sp_smag.setValue(round(lim[0], 1))
                self.ck_smag_auto.setChecked(True)
                self.log(f"Image limiting magnitude: {lim[0]:.1f} ({lim[1]} stars detected)")
            else:
                self.params.image_mag, self.params.image_stars = 0.0, 0
            self._update_outline_label()
            self.preview()
        except Exception as e:
            QMessageBox.warning(self, "Sky Chart", str(e))

    def _image_limiting_mag(self, img, sky, ora, odec):
        """Faintest catalog magnitude actually visible in the image: detect the
        stars, match them with Gaia DR3 / HYG and take the magnitude reached by
        98% of the matched stars. Returns (mag, n_detected) or None."""
        if not self._ensure_data():
            return None
        a = np.asarray(img, dtype=np.float32)
        if a.ndim == 3:
            a = a[..., :3].mean(axis=2)
        h, w = a.shape
        k = max(1, int(math.ceil(max(h, w) / 3000.0)))          # work on <= 3000 px
        if k > 1:
            a = a[: h - h % k, : w - w % k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        xs = ys = None
        try:
            import sep
            data = np.ascontiguousarray(a, dtype=np.float32)
            bkg = sep.Background(data)
            sub = data - bkg.back()
            objs = sep.extract(sub, 4.0, err=bkg.globalrms, minarea=4)
            objs = objs[(objs["flag"] & 4) == 0] if len(objs) else objs
            xs, ys = objs["x"], objs["y"]
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter, maximum_filter
                sm = gaussian_filter(a, 1.2)
                med = np.median(sm)
                mad = np.median(np.abs(sm - med)) * 1.4826 + 1e-12
                peaks = (sm == maximum_filter(sm, 7)) & (sm > med + 6.0 * mad)
                ys, xs = np.nonzero(peaks)
            except Exception:
                return None
        n = len(xs)
        if n < 15:
            self.log(f"Only {n} stars detected in the image: magnitude limit not measured.")
            return None
        xs = (np.asarray(xs, dtype=np.float64) + 0.5) * k - 0.5
        ys = (np.asarray(ys, dtype=np.float64) + 0.5) * k - 0.5
        dra, ddec = sky(xs, ys)
        # catalog stars in the image footprint (HYG + Gaia)
        ra_lo, ra_hi = float(np.min(ora)), float(np.max(ora))
        if ra_hi - ra_lo > 180:
            r2 = (np.asarray(ora) + 180) % 360
            ra_lo, ra_hi = float(r2.min()) - 180, float(r2.max()) - 180
        dec_lo, dec_hi = float(np.min(odec)), float(np.max(odec))
        cra, cdec, cmag = self.data.catalog_box(ra_lo % 360, ra_hi % 360, dec_lo, dec_hi, 17.0)
        if len(cra) < 10:
            return None
        # match each detection with the nearest catalog star (radius: 3 image pixels or 4")
        cd = math.cos(math.radians((dec_lo + dec_hi) / 2))
        scale = abs(dec_hi - dec_lo) / max(h, w) * 3600.0 * max(h, w) / h
        tol = max(4.0, 3.0 * scale) / 3600.0
        cell = tol
        grid = {}
        cx = ((cra - ra_lo + 540) % 360 - 180) * cd
        for i, (u, v) in enumerate(zip(cx, cdec)):
            grid.setdefault((int(u // cell), int(v // cell)), []).append(i)
        mags = []
        dx = ((dra - ra_lo + 540) % 360 - 180) * cd
        for u, v in zip(dx, ddec):
            best, bd = None, tol
            gi, gj = int(u // cell), int(v // cell)
            for a1 in (gi - 1, gi, gi + 1):
                for b1 in (gj - 1, gj, gj + 1):
                    for i in grid.get((a1, b1), ()):
                        dd = math.hypot(cx[i] - u, cdec[i] - v)
                        if dd < bd:
                            best, bd = i, dd
            if best is not None:
                mags.append(cmag[best])
        if len(mags) >= 15:
            lim = float(np.percentile(mags, 98))
        else:                         # matching failed: equal-count estimate
            srt = np.sort(cmag)
            lim = float(srt[min(n, len(srt)) - 1])
        return round(min(max(lim, 3.0), 17.0), 2), int(n)


    def preview(self, keep_view=False):
        if hasattr(self, "_auto_timer"):
            self._auto_timer.stop()
        p = self._get_params()
        if p is None or not self._ensure_data():
            return False
        self.params = p
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            t0 = time.time()
            q = ChartParams(**{**asdict(p), "text_as_paths": True})
            r = ChartRenderer(q, self.data, self.log)
            pic = QPicture()
            painter = QPainter(pic)
            try:
                painter.scale(_ChartView.K, _ChartView.K)
                r.render(painter)
            finally:
                painter.end()
            self.renderer = r
            self._pic = pic
            self.view.set_picture(pic, r.pw, r.ph, keep_view=keep_view)
            i = r.info
            if i.get("bprp_pending"):
                self._start_bprp(i["bprp_pending"])
            faint = f", faintest {i['faintest']:.2f}" if i.get("faintest") is not None else ""
            self.chart_label.setText(f"{i['stars']} stars (magnitude ≤ {i['star_mag']:.1f}{faint}), "
                                     f"{i['dsos']} deep-sky objects")
            gaia_txt = ""
            if i.get("gaia"):
                n_xp = i.get("gaia_xp", i.get("gaia_colored", 0))
                n_on = i.get("gaia_colored", 0) - n_xp
                gaia_txt = (f"\nGaia DR3: {i['gaia']} stars, {n_xp} colored from the local XP spectra"
                            + (f", {n_on} from the Gaia archive (BP−RP)" if n_on > 0 else "")
                            + (" — downloading colours..." if self._bprp_busy else ""))
                if not i.get("gaia_colored") and ChartData.xp_dir() is None and p.colored_stars \
                        and not p.black_white:
                    gaia_txt += " (XP library not installed)"
            if not p.star_mag_auto:
                auto_txt = ""
            elif p.image_mag and p.outline and p.image_mag <= i['star_mag'] + 0.05:
                auto_txt = f" (measured on the image: {p.image_stars} stars detected)"
            else:
                auto_txt = f" (automatic, {p.star_density:g} stars/cm²)"
            if i.get("gaia_capped"):
                gaia_txt += f"\nWide field: Gaia stars shown to G = {GAIA_CACHE_MAG}"
            k = p.dpi() / 25.4
            x0, y0, x1, y1 = r.frame
            arcsec_px = 3600.0 / (i['mm_per_deg'] * k)
            self.lb_info.setText(
                f"Field {i['fov_w']:.4g}° × {i['fov_h']:.4g}°, 1° = {i['mm_per_deg']:.3g} mm, "
                f"{arcsec_px:.2f}″/px\n"
                f"Page {r.pw:.0f} × {r.ph:.0f} mm ({int(round(r.pw * k))} × {int(round(r.ph * k))} px), "
                f"map {int(round((x1 - x0) * k))} × {int(round((y1 - y0) * k))} px\n"
                f"Stars to magnitude {i['star_mag']:.1f}"
                f"{auto_txt}, deep-sky objects to {i['dso_mag']:.1f}"
                f"{' (automatic)' if p.dso_mag_auto else ''}" + gaia_txt)
            if p.star_mag_auto:
                self._loading = True
                self.sp_smag.setValue(round(i['star_mag'], 1))
                if p.dso_mag_auto:
                    self.sp_dmag.setValue(round(i['dso_mag'], 1))
                self._loading = False
            self.log(f"Chart: {i['stars']} stars, {i['dsos']} deep-sky objects ({time.time() - t0:.2f} s)")
            if not getattr(self, "_restoring", False):
                self._history_add(p)
            self._schedule_save()
            return True
        except Exception as e:
            import traceback
            self.log(traceback.format_exc())
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Sky Chart", f"Could not generate the chart:\n\n{e}")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _start_bprp(self, want):
        """Fetch the missing Gaia colours in the background; repaint when they arrive."""
        if self._bprp_busy or not want:
            return
        self._bprp_busy = True
        import threading

        data = self.data
        relay = _bprp_relay()

        def job():
            try:
                n = data.fetch_bprp(dict(want))
                relay.done.emit(n, "")
            except Exception as e:
                relay.done.emit(-1, str(e))

        threading.Thread(target=job, daemon=True).start()

    def _on_bprp_done(self, n, err):
        self._bprp_busy = False
        if getattr(self, "_closed", False):
            return
        if n < 0:
            self.log(f"Online Gaia colours not available: {err}")
            return
        self.log(f"Gaia archive: colours for {n} stars downloaded")
        if n > 0:
            was = getattr(self, "_restoring", False)
            self._restoring = True                   # same chart: no new history entry
            try:
                self.preview(keep_view=True)
            finally:
                self._restoring = was

    def _on_mouse(self, x_mm, y_mm):
        r = self.renderer
        if r is None:
            return
        x0, y0, x1, y1 = r.frame
        if not (x0 <= x_mm <= x1 and y0 <= y_mm <= y1):
            self.status.setText(f"x: {x_mm:.1f} mm   y: {y_mm:.1f} mm")
            return
        ra, dec = r.proj.inverse(np.array([x_mm]), np.array([y_mm]))
        if ra[0] != ra[0] or dec[0] != dec[0]:
            return
        zoom = self.view.transform().m11() * _ChartView.K / ((self.view.logicalDpiX() or 96) / 25.4)
        self.status.setText(f"Zoom: {zoom:.2f}   x: {x_mm:.1f} mm   y: {y_mm:.1f} mm      "
                            f"α: {fmt_ra(float(ra[0]), 1)}   δ: {fmt_dec(float(dec[0]), 0)}")

    def _ask_file(self, ext, current, base):
        if current:
            return current
        last = getattr(self, "_last_dir", str(Path.home()))
        fn, _ = QFileDialog.getSaveFileName(self, f"Sky Chart: {ext.upper()} file",
                                            str(Path(last) / f"{base}.{ext}"), f"{ext.upper()} files (*.{ext})")
        if fn and not fn.lower().endswith("." + ext):
            fn += "." + ext
        if fn:
            self._last_dir = str(Path(fn).parent)
        return fn

    def run(self):
        p = self._get_params()
        if p is None or not self._ensure_data():
            return
        self.params = p
        if not (p.out_svg or p.out_pdf or p.out_png or p.out_image):
            QMessageBox.information(self, "Sky Chart", "Select at least one output in the Output tab.")
            return
        base = re.sub(r"[^\w\-]+", "_", (p.title or self.data.constellation_at(p.ra, p.dec))).strip("_") or "chart"
        done = []
        try:
            jobs = []
            if p.out_svg:
                fn = self._ask_file("svg", p.svg_path, base)
                if fn:
                    jobs.append(("svg", fn))
            if p.out_pdf:
                fn = self._ask_file("pdf", p.pdf_path, base)
                if fn:
                    jobs.append(("pdf", fn))
            if p.out_png:
                fn = self._ask_file("png", p.png_path, base)
                if fn:
                    jobs.append(("png", fn))
            if p.out_image:
                jobs.append(("image", None))
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if jobs and p.online_colors and p.colored_stars and not p.black_white:
                    try:                      # complete the colour cache before writing
                        ChartRenderer(p, self.data, self.log, wait_online=True)._select_stars()
                    except Exception as e:
                        self.log(f"Online Gaia colours not available: {e}")
                for kind, fn in jobs:
                    if kind == "svg":
                        render_svg(p, self.data, fn, self.log)
                    elif kind == "pdf":
                        render_pdf(p, self.data, fn, self.log)
                    elif kind == "png":
                        img, _r = render_image(p, self.data, p.dpi(), self.log)
                        if not img.save(fn):
                            raise RuntimeError(f"Could not write {fn}")
                    else:
                        if self.ctx is None:
                            raise RuntimeError("No SASpro context available for the image output.")
                        img, _r = render_image(p, self.data, p.dpi(), self.log)
                        arr = qimage_to_array(img)
                        ident = re.sub(r"[^\w\-]+", "_", p.image_id).strip("_") or base
                        meta = {"is_mono": False}
                        hdr = wcs_obj = None
                        try:
                            from astropy.wcs import WCS
                            hdr = chart_wcs_header(_r, p.dpi() / 25.4, arr.shape[1], arr.shape[0])
                            wcs_obj = WCS(hdr)
                            meta.update({"original_header": hdr.copy(), "wcs_header": hdr,
                                         "is_solved": True, "plate_solve_source": "Sky Chart"})
                        except Exception as e:
                            self.log(f"Chart WCS not attached: {e}")
                        doc = self.ctx.open_new_document(arr, metadata=meta, name=ident)
                        if hdr is not None and doc is not None and hasattr(doc, "set_wcs"):
                            try:
                                doc.set_wcs(hdr, source="Sky Chart", wcs_obj=wcs_obj)
                            except Exception as e:
                                self.log(f"Chart WCS not attached: {e}")
                        fn = (f"new image '{ident}' ({arr.shape[1]} x {arr.shape[0]} px"
                              f"{', astrometric solution attached' if hdr is not None else ''}"
                              f"{', on the reference image grid' if _r._match is not None else ''})")
                    done.append(fn)
                    self.log(f"Chart written: {fn}")
            finally:
                QApplication.restoreOverrideCursor()
            if done:
                self.status.setText("Chart written: " + "; ".join(done))
            self._save_settings()
            self.preview(keep_view=True)
        except Exception as e:
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "Sky Chart", f"Could not write the chart:\n\n{e}")

    # ---- lifetime ---------------------------------------------------------
    def shutdown(self):
        """Called by the host when it closes: stop timers, save the settings, and
        stop listening for an online-colour fetch that may still be running."""
        if getattr(self, "_closed", False):
            return
        for t in ("_save_timer", "_auto_timer"):
            try:
                getattr(self, t).stop()
            except Exception:
                pass
        try:
            _bprp_relay().done.disconnect(self._on_bprp_done)
        except Exception:
            pass
        self._save_settings()
        self._closed = True


class OverlayLayersDialog(QDialog):
    """Which sky-chart layers an overlay draws: Finder Chart over a survey
    image, or an image view's WCS overlay over the photograph.

    Optional: `title` and `note` for the caller's wording; `defaults`, what the
    Defaults button restores (OVERLAY_DEFAULTS otherwise); `options`, a dict
    with any of "dso_mag" (0 = automatic), "text_scale", "graphics_scale" --
    when given, those are editable too and options() returns them, with
    `option_defaults` restored by Defaults."""

    def __init__(self, layers: dict, parent=None, *, title: str | None = None,
                 note: str | None = None, defaults: dict | None = None,
                 options: dict | None = None, option_defaults: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(title or "Sky Chart Overlay Layers")
        defaults = dict(OVERLAY_DEFAULTS if defaults is None else defaults)
        v = QVBoxLayout(self)
        lbl = QLabel(note or "Drawn over the survey image. The photograph already shows the stars, "
                             "nebulae and Milky Way, so those are labelled, never painted.")
        lbl.setWordWrap(True)
        v.addWidget(lbl)
        names = {k: (n, d) for k, n, d, _on in LAYERS}
        self._checks = {}
        grid = QGridLayout()
        for i, key in enumerate(OVERLAY_LAYERS):
            n, d = names.get(key, (key, ""))
            cb = QCheckBox(n)
            cb.setToolTip(d)
            cb.setChecked(bool(layers.get(key, defaults.get(key, False))))
            self._checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        v.addLayout(grid)

        self._opts = {}
        self._opt_defaults = dict(option_defaults or {})
        if options is not None:
            from PyQt6.QtWidgets import QDoubleSpinBox, QFormLayout
            form = QFormLayout()
            if "dso_mag" in options:
                sp = QDoubleSpinBox()
                sp.setRange(0.0, 20.0); sp.setDecimals(1); sp.setSingleStep(0.5)
                sp.setSpecialValueText("Auto")
                sp.setToolTip("Faintest deep-sky object drawn. Auto follows the field of view.")
                sp.setValue(float(options["dso_mag"]))
                self._opts["dso_mag"] = sp
                form.addRow("Faintest deep-sky object, mag ≤", sp)
            for key, label, tip in (
                    ("text_scale", "Label size",
                     "Size of names and labels on screen; they stay this size while you zoom."),
                    ("graphics_scale", "Line weight",
                     "Thickness of constellation lines, borders and object outlines.")):
                if key in options:
                    sp = QDoubleSpinBox()
                    sp.setRange(0.5, 4.0); sp.setDecimals(1); sp.setSingleStep(0.1)
                    sp.setSuffix(" ×")
                    sp.setToolTip(tip)
                    sp.setValue(float(options[key]))
                    self._opts[key] = sp
                    form.addRow(label, sp)
            v.addLayout(form)

        row = QHBoxLayout()
        for txt, on in (("All", True), ("None", False)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, val=on: [c.setChecked(val) for c in self._checks.values()])
            row.addWidget(b)
        b = QPushButton("Defaults")

        def _defaults():
            for k, c in self._checks.items():
                c.setChecked(bool(defaults.get(k, False)))
            for k, sp in self._opts.items():
                if k in self._opt_defaults:
                    sp.setValue(float(self._opt_defaults[k]))
        b.clicked.connect(_defaults)
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def layers(self) -> dict:
        return {k: cb.isChecked() for k, cb in self._checks.items()}

    def options(self) -> dict:
        return {k: float(sp.value()) for k, sp in self._opts.items()}
