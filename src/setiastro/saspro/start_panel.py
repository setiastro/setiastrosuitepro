# src/setiastro/saspro/start_panel.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
The start panel: what an empty workspace offers. Open, process, plan, a stacker
for each kind of imaging (deep sky, smart telescope, planetary), and the last
few images as thumbnails (or, with none yet, a way in for someone
new). It goes when an image opens and comes back when the last one closes.

It never covers a desktop shortcut, the setup reminder or the First Steps card:
it moves aside, then drops its second half, then stays away. The ✕ closes it for
the rest of the session and View > Start Panel brings it back. With "Show on
startup" off (here or in Preferences) it only comes when asked for, that one time.

Thumbnails are small JPEGs in the SASpro data folder, one per recent image,
made from the open image. One that was never open in this version is read from
its file if that is cheap (FITS and XISF through Blink's downsampled reader,
PNG/JPEG/WebP through Qt). All of that, down to asking whether a file exists,
is on a worker thread: a recent image can be on a drive that isn't there.
"""
from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np
from PyQt6.QtCore import (QCoreApplication, QEvent, QObject, QPoint, QPointF, QRect,
                          QRectF, QSize, QStandardPaths, Qt, QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import (QColor, QDesktopServices, QFontMetrics, QIcon, QImage,
                         QImageReader, QKeySequence, QPainter, QPainterPath, QPen, QPixmap)
from PyQt6.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QStyle, QStyleOptionButton, QVBoxLayout, QWidget)

STARTUP_KEY = "ui/start_panel"                   # show on startup
RELEASES_URL = "https://github.com/setiastro/setiastrosuitepro/releases/latest"
TUTORIAL_URL = "https://youtu.be/zBUOZl2OgaY"     # the welcome window's beginner video

RECENTS = 4                      # shown
THUMB = QSize(184, 112)          # logical size; the files hold twice that
WIDTH = 840
GAP = 10                         # kept clear around shortcuts and cards
LINEAR_MEDIAN = 0.08             # as first_steps_tour: below this, unstretched


def _t(text: str) -> str:
    return QCoreApplication.translate("StartPanel", text)


_STYLE = """
#StartPanel { background-color: rgba(15, 15, 27, 232); border: 1px solid #34345c; border-radius: 14px; }
#sp_h { color: #ffffff; font-size: 20px; font-weight: 700; background: transparent; }
#sp_k { color: #8f9be0; font-size: 11px; font-weight: 700; background: transparent; }
#sp_s { color: #9a9abf; font-size: 12px; background: transparent; }
#sp_tile_t { color: #e8e8f6; font-size: 13px; font-weight: 600; background: transparent; }
#sp_tile_s { color: #9a9abf; font-size: 11px; background: transparent; }
#sp_tile_more { color: #8f9be0; font-size: 12px; background: transparent; }
QPushButton#sp_tile { background-color: #1b1b34; border: 1px solid #32325a; border-radius: 9px; }
QPushButton#sp_tile:hover, QPushButton#sp_tile:focus { background-color: #25254a; border-color: #5f6fd8; }
QPushButton#sp_tile[main="true"] { border: none;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3344bb, stop:1 #5566dd); }
QPushButton#sp_tile[main="true"]:hover, QPushButton#sp_tile[main="true"]:focus {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4455cc, stop:1 #6677ee); }
QPushButton#sp_tile[main="true"] #sp_tile_s { color: #cfd6ff; }
QPushButton#sp_recent { background: transparent; border: none; border-radius: 9px; }
QPushButton#sp_recent:hover, QPushButton#sp_recent:focus { background-color: #25254a; }
QPushButton#sp_link { background: transparent; border: none; color: #8fa0ff; font-size: 12px; padding: 2px; }
QPushButton#sp_link:hover, QPushButton#sp_link:focus { color: #ffffff; text-decoration: underline; }
QPushButton#sp_x { background: transparent; border: none; border-radius: 6px; color: #8888bb; font-size: 14px; }
QPushButton#sp_x:hover, QPushButton#sp_x:focus { color: #ffffff; background-color: #25254a; }
QCheckBox#sp_startup { color: #9a9abf; font-size: 12px; background: transparent; spacing: 6px; }
QCheckBox#sp_startup:hover, QCheckBox#sp_startup:focus { color: #ffffff; }
QCheckBox#sp_startup::indicator { width: 13px; height: 13px; border: 1px solid #6c6ca0;
    border-radius: 4px; background-color: #1b1b34; }
QCheckBox#sp_startup::indicator:hover { border-color: #8f9be0; }
QCheckBox#sp_startup::indicator:checked { border-color: #5566dd; background-color: #4455cc; }
"""


# ── thumbnails ──────────────────────────────────────────────────────────────

def thumbnails_dir() -> Path:
    from setiastro.saspro import settings_sandbox
    box = settings_sandbox.active()
    if box:
        return Path(box) / "recent_thumbnails"
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    return Path(base) / "SASpro" / "recent_thumbnails"


def thumbnail_file(path: str) -> Path:
    name = os.path.normcase(os.path.abspath(path))
    return thumbnails_dir() / (hashlib.sha1(name.encode("utf-8", "replace")).hexdigest() + ".jpg")


def render_thumbnail(image) -> QImage | None:
    """A cover-cropped, display-stretched thumbnail of an image array."""
    import cv2
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] not in (3, 4):
        a = a[..., 0]
    if a.ndim == 3:
        a = a[..., :3]
    if a.ndim != 2 and a.ndim != 3 or min(a.shape[:2]) < 8:
        return None
    w, h = THUMB.width() * 2, THUMB.height() * 2
    # Strided first, leaving four times the pixels for the averaging below.
    step = max(1, min(a.shape[0] // (2 * h), a.shape[1] // (2 * w)))
    kind = a.dtype
    a = np.array(a[::step, ::step], dtype=np.float32)
    np.nan_to_num(a, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    if np.issubdtype(kind, np.integer):
        a /= float(np.iinfo(kind).max)
    else:
        top = float(a.max())
        if top > 1.0:
            a /= top
    np.clip(a, 0.0, 1.0, out=a)
    scale = max(w / a.shape[1], h / a.shape[0])
    size = (max(w, round(a.shape[1] * scale)), max(h, round(a.shape[0] * scale)))
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y, x = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    a = a[y:y + h, x:x + w]
    if float(np.median(a)) < LINEAR_MEDIAN:
        from setiastro.saspro.autostretch import autostretch
        a = np.asarray(autostretch(np.ascontiguousarray(a), 0.25), dtype=np.float32)
        if a.max() > 1.0:
            a = a / 255.0
    u = np.ascontiguousarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    if u.ndim == 2:
        return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QImage(u.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _thumbnail_from_file(path: str) -> QImage | None:
    """Cheap formats only; None leaves the placeholder until the image is opened."""
    low = path.lower()
    if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
        r = QImageReader(path)
        r.setAutoTransform(True)
        size = r.size()
        if not size.isValid():
            return None
        box = QSize(THUMB.width() * 2, THUMB.height() * 2)
        r.setScaledSize(size.scaled(box, Qt.AspectRatioMode.KeepAspectRatioByExpanding))
        img = r.read()
        if img.isNull():
            return None
        return img.copy((img.width() - box.width()) // 2, (img.height() - box.height()) // 2,
                        box.width(), box.height())
    from setiastro.saspro.blink_fast_read import read_downsampled
    for step in (16, 8, 4):
        got = read_downsampled(path, step, cfa_mode="luminance")
        if got is None:
            continue
        # Twice what the file will hold, or there is nothing left to average and
        # a star field comes out as hard white dots.
        if step > 4 and min(got[0].shape[0] / THUMB.height(), got[0].shape[1] / THUMB.width()) < 4:
            continue
        return render_thumbnail(got[0])
    return None


class Thumbnails(QObject):
    """Makes and keeps the thumbnails, on one worker thread."""

    made = pyqtSignal(str)                       # path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: queue.Queue = queue.Queue()
        self._thread = None
        self._stopped = False
        self._tried = set()                      # read from file once per session

    def pixmap(self, path: str) -> QPixmap:
        return QPixmap(str(thumbnail_file(path)))

    def from_image(self, path: str, image):
        self._put(("image", path, image))

    def from_file(self, path: str):
        if path not in self._tried:
            self._tried.add(path)
            self._put(("file", path, None))

    def keep_only(self, paths):
        self._put(("prune", list(paths), None))

    def stop(self):
        self._stopped = True
        self._jobs.put(None)

    def _put(self, job):
        if self._stopped:
            return
        self._jobs.put(job)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="recent-thumbnails", daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stopped:
            job = self._jobs.get()
            if job is None or self._stopped:
                return
            kind, what, image = job
            try:
                if kind == "prune":
                    self._prune(what)
                elif self._make(kind, what, image) and not self._stopped:
                    self.made.emit(what)
            except RuntimeError:
                return                           # the window is gone
            except OSError:
                pass                             # the file is gone, or its drive
            except Exception as e:
                print(f"[start panel] no thumbnail for {os.path.basename(str(what))}: {e!r}")
            del job, image

    @staticmethod
    def _make(kind, path, image) -> bool:
        out = thumbnail_file(path)
        taken = os.stat(path).st_mtime           # raises when the file is gone
        if out.exists() and out.stat().st_mtime >= taken:
            return False
        img = render_thumbnail(image) if kind == "image" else _thumbnail_from_file(path)
        if img is None:
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        if not img.save(str(tmp), "JPG", 88):
            return False
        os.replace(tmp, out)
        return True

    @staticmethod
    def _prune(paths):
        keep = {thumbnail_file(p).name for p in paths}
        try:
            names = os.listdir(thumbnails_dir())
        except OSError:
            return                               # none made yet
        for n in names:
            if n not in keep:
                try:
                    os.remove(thumbnails_dir() / n)
                except OSError:
                    pass


# ── pieces ──────────────────────────────────────────────────────────────────

def _label(text, name) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName(name)
    lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    return lab


def _link(text, slot) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("sp_link")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFocusPolicy(Qt.FocusPolicy.TabFocus)
    b.clicked.connect(lambda: slot())
    return b


class _Check(QCheckBox):
    """The theme's own box all but vanishes on the panel, which is dark in every
    theme. The style sheet draws this one, except the tick: that takes a file."""

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if not self.isChecked():
            return
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        r = QRectF(self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, opt, self))
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#ffffff"), 1.8, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPolyline([r.topLeft() + QPointF(x * r.width(), y * r.height())
                        for x, y in ((0.27, 0.53), (0.44, 0.69), (0.74, 0.33))])
        p.end()


def _card_icon(path) -> QIcon:
    """The picture on one of Mosaic Master's mode cards, with a light edge:
    its outlines (and the robotic one's tripod) are near black, and so is a tile."""
    pic = QPixmap(path)
    if pic.isNull():
        return QIcon()
    size, edge = 112, 5                          # drawn at 28, four times finer
    pic = pic.scaled(size - 2 * edge, size - 2 * edge, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
    shape = QPixmap(pic)
    p = QPainter(shape)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(shape.rect(), QColor("#dfe3ff"))
    p.end()
    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)
    x, y = (size - pic.width()) / 2, (size - pic.height()) / 2
    p = QPainter(out)
    for i in range(24):
        a = 2 * math.pi * i / 24
        p.drawPixmap(QPointF(x + edge * math.cos(a), y + edge * math.sin(a)), shape)
    p.drawPixmap(QPointF(x, y), pic)
    p.end()
    return QIcon(out)


class _Tile(QPushButton):
    """Icon, what it does, and the name of the tool that does it. `more` marks
    one that asks which tool first."""

    def __init__(self, icon, title, sub, slot, main=False, more=False):
        super().__init__()
        self.setObjectName("sp_tile")
        self.setProperty("main", "true" if main else "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName(f"{title}. {sub}")
        self.setMinimumSize(170, 60)
        self._texts = []                         # (label, its full text)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 8, 10, 8)
        row.setSpacing(12)
        pic = QLabel()
        pic.setFixedSize(28, 28)
        pic.setStyleSheet("background: transparent;")
        pic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if icon is not None and not icon.isNull():
            pic.setPixmap(icon.pixmap(QSize(28, 28)))
        row.addWidget(pic)
        col = QVBoxLayout()
        col.setSpacing(1)
        col.addStretch(1)
        for text, name in ((title, "sp_tile_t"), (sub, "sp_tile_s")):
            lab = _label(text, name)
            lab.setMinimumWidth(1)               # the tile's width decides
            self._texts.append((lab, text))
            col.addWidget(lab)
        col.addStretch(1)
        row.addLayout(col, 1)
        self._room = 14 + 28 + 12 + 10           # margins, icon and spacing above
        if more:
            chev = _label("▾", "sp_tile_more")
            chev.setFixedWidth(14)
            row.addWidget(chev)
            self._room += 14 + 12
        self.clicked.connect(lambda: slot())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        room = self.width() - self._room
        for lab, text in self._texts:
            lab.ensurePolished()
            lab.setText(QFontMetrics(lab.font()).elidedText(
                text, Qt.TextElideMode.ElideRight, max(1, room)))


class _Choices(QFrame):
    """A small popup under a tile: one tile for each way to do it. Closes on a
    choice, a click anywhere else, or Esc."""

    def __init__(self, anchor, choices, parent):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)   # round corners
        # The click that closes it stops there: on the tile, it would reopen.
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)
        self.setStyleSheet(_STYLE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self.tiles = []
        for icon, title, sub, slot in choices:
            t = _Tile(icon, title, sub, lambda s=slot: self._choose(s))
            t.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            t.setAutoDefault(True)               # Enter as well as Space
            lay.addWidget(t)
            self.tiles.append(t)
        self.setFixedWidth(max(250, anchor.width()))
        self.adjustSize()
        under = anchor.mapToGlobal(QPoint(0, anchor.height() + 6))
        screen = anchor.screen().availableGeometry()
        if under.y() + self.height() > screen.bottom():
            under = anchor.mapToGlobal(QPoint(0, -self.height() - 6))
        under.setX(max(screen.left(), min(under.x(), screen.right() - self.width())))
        self.move(under)

    def paintEvent(self, _ev):
        # Not in the style sheet: Qt leaves the background of a translucent
        # window out, and the choices would float over the panel's text.
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QColor("#5f6fd8"))
        p.setBrush(QColor("#14142a"))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 11, 11)
        p.end()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.tiles[0].setFocus(Qt.FocusReason.PopupFocusReason)

    def _choose(self, slot):
        self.close()
        QTimer.singleShot(0, slot)               # once the popup is gone


class _Recent(QPushButton):
    """One recent image: its thumbnail and file name."""

    def __init__(self, path, pixmap, slot):
        super().__init__()
        self.setObjectName("sp_recent")
        self.path = path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setToolTip(path)
        name = os.path.basename(path) or path
        self.setAccessibleName(_t("Open {0}").format(name))
        self.setFixedSize(THUMB.width() + 12, THUMB.height() + 36)
        col = QVBoxLayout(self)
        col.setContentsMargins(6, 6, 6, 4)
        col.setSpacing(4)
        self._pic = QLabel()
        self._pic.setFixedSize(THUMB)
        self._pic.setStyleSheet("background: transparent;")
        self._pic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        col.addWidget(self._pic)
        text = _label("", "sp_s")
        text.ensurePolished()
        text.setText(QFontMetrics(text.font()).elidedText(
            name, Qt.TextElideMode.ElideMiddle, THUMB.width() - 4))
        col.addWidget(text)
        self.set_picture(pixmap)
        self.clicked.connect(lambda: slot(path))

    def set_picture(self, pixmap: QPixmap):
        dpr = self.devicePixelRatioF()
        pm = QPixmap(round(THUMB.width() * dpr), round(THUMB.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        box = QRectF(0, 0, THUMB.width(), THUMB.height())
        path = QPainterPath()
        path.addRoundedRect(box, 8, 8)
        p.setClipPath(path)
        if pixmap.isNull():
            p.fillRect(box, QColor("#1b1b34"))
            p.setPen(QColor("#6c6ca0"))
            font = p.font()
            font.setPointSize(13)
            font.setBold(True)
            p.setFont(font)
            ext = os.path.splitext(self.path)[1].lstrip(".").upper()
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, ext or "?")
        else:
            p.drawPixmap(box, pixmap, QRectF(pixmap.rect()))
        # A dark sky would otherwise run into the panel.
        p.setClipping(False)
        p.setPen(QColor(255, 255, 255, 34))
        p.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        p.end()
        self._pic.setPixmap(pm)


# ── the panel ───────────────────────────────────────────────────────────────

class StartPanel(QFrame):
    def __init__(self, main_window):
        self._mw = main_window
        self._vp = main_window.mdi.viewport()
        super().__init__(self._vp)
        self.setObjectName("StartPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_STYLE)
        self.setAcceptDrops(True)                # handed to the workspace below
        self.hide()
        self._placed_for = None
        self._shown_recents = None
        self._recent_buttons = []
        self.on = self.on_startup()              # for this session

        self.thumbs = Thumbnails(self)
        self.thumbs.made.connect(self._thumb_made)
        self._build()

        self._vp.installEventFilter(self)
        self._settle = QTimer(self)              # one placement per burst of changes
        self._settle.setSingleShot(True)
        self._settle.setInterval(0)
        self._settle.timeout.connect(self.sync)
        self._watch = QTimer(self)               # shortcuts and cards come and go
        self._watch.setInterval(700)
        self._watch.timeout.connect(self.sync)
        self._capture = QTimer(self)             # after the open has settled
        self._capture.setSingleShot(True)
        self._capture.setInterval(1500)
        self._capture.timeout.connect(self._capture_open_images)
        self._settle.start()

    # ---- contents -----------------------------------------------------------
    def _build(self):
        mw = self._mw
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 18, 26, 16)
        root.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(_label(_t("What would you like to do?"), "sp_h"))
        head.addStretch(1)
        self._startup = _Check(_t("Show on startup"))
        self._startup.setObjectName("sp_startup")
        self._startup.setCursor(Qt.CursorShape.PointingHandCursor)
        self._startup.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._startup.setAccessibleName(_t("Show the start panel on startup"))
        self._startup.setToolTip(_t(
            "Unticked, this panel only comes up from View > Start Panel. "
            "Also in Settings > Preferences."))
        self._startup.setChecked(self.on_startup())
        self._startup.clicked.connect(lambda on: mw._set_start_panel_on_startup(on))
        head.addWidget(self._startup)
        head.addSpacing(8)
        x = QPushButton("✕")
        x.setObjectName("sp_x")
        x.setFixedSize(26, 26)
        x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        x.setAccessibleName(_t("Close the start panel"))
        x.setToolTip(_t("Close for now. View > Start Panel brings it back."))
        x.clicked.connect(lambda: mw._set_start_panel(False))
        head.addWidget(x)
        root.addLayout(head)

        grid = QGridLayout()
        grid.setSpacing(10)
        self._tiles = []                         # (QAction, its tile)
        # The second row: one stacker for each kind of imaging.
        for act, title, sub in (
                ("act_open", _t("Open images…"), _t("FITS, XISF, TIFF, RAW, PNG, JPEG")),
                ("act_workflows", _t("Process step by step"),
                 _t("Open Image with Workflow Assistant")),
                ("act_whats_in_my_sky", _t("Plan tonight"), _t("What's In My Sky")),
                ("deep_sky", _t("Deep-Sky Stacking"), _t("Quick Stack or Stacking Suite")),
                ("mosaics", _t("Deep-Sky Mosaics"), _t("Traditional Panels or Robotic")),
                ("act_planetary_stacker", _t("Planetary Stacking"),
                 _t("Moon, Sun and planets from video"))):
            if act == "deep_sky":                # asks which of the two
                a, slot = None, self._choose_deep_sky
                icon = mw.act_stacking_suite.icon()
            elif act == "mosaics":               # asks which mode of Mosaic Master
                from setiastro.saspro.resources import mosaic_robotic_path
                a, slot = None, self._choose_mosaic
                icon = _card_icon(mosaic_robotic_path)
            else:
                a = getattr(mw, act, None)
                if a is None:
                    continue
                slot, icon = a.trigger, a.icon()
                if icon.isNull():                # Workflow Assistant has none
                    icon = mw.act_function_bundles.icon()
                if act == "act_workflows":
                    slot = self._process_step_by_step
            i = grid.count()
            tile = _Tile(icon, title, sub, slot, main=(i == 0),
                         more=(act in ("deep_sky", "mosaics")))
            if act == "deep_sky":
                self._deep_sky = tile
            elif act == "mosaics":
                self._mosaics = tile
            self._tiles.append((a, tile))
            grid.addWidget(tile, i // 3, i % 3)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        root.addLayout(grid)

        self._more = QWidget()                   # the half that goes when space is short
        self._more.setObjectName("sp_more")
        self._more.setStyleSheet("#sp_more { background: transparent; }")
        self._more_lay = QVBoxLayout(self._more)
        self._more_lay.setContentsMargins(0, 4, 0, 0)
        self._more_lay.setSpacing(8)
        root.addWidget(self._more)

        foot = QHBoxLayout()
        find = QKeySequence("Ctrl+Shift+P").toString(QKeySequence.SequenceFormat.NativeText)
        foot.addWidget(_label(_t("Drop files anywhere here. {0} finds any tool by name.").format(find),
                              "sp_s"))
        foot.addStretch(1)
        foot.setSpacing(16)
        foot.addWidget(_link(_t("Open a project…"), mw.act_project_load.trigger))
        foot.addWidget(_link(_t("What's new"), lambda: QDesktopServices.openUrl(QUrl(RELEASES_URL))))
        root.addLayout(foot)
        self._fill_more()

    def _process_step_by_step(self):
        """Images first, then the Workflow Assistant. This panel is only up with
        no image open, and opening one takes it away; the assistant works on the
        open images. Cancelled, the assistant still opens, to look around in."""
        mw = self._mw
        try:
            mw.open_files()
        finally:
            mw.act_workflows.trigger()

    def _choose_deep_sky(self):
        """Quick Stack or the Stacking Suite, in a popup under the tile."""
        choices = []
        for act, title, sub in (
                ("act_quick_stack", _t("Quick Stack"), _t("Fast and easy to start with")),
                ("act_stacking_suite", _t("Stacking Suite"), _t("The full pipeline, in depth"))):
            a = getattr(self._mw, act, None)
            if a is not None:
                choices.append((a.icon(), title, sub, a.trigger))
        if len(choices) == 1:
            choices[0][3]()
            return
        self._show_choices(self._deep_sky, choices)

    def _choose_mosaic(self):
        """Mosaic Master's two modes, Traditional Panels or Robotic Mosaic, in a
        popup under the tile. Each opens Mosaic Master with that card chosen."""
        from setiastro.saspro.resources import mosaic_panels_path, mosaic_robotic_path
        mw = self._mw
        self._show_choices(self._mosaics, [
            (_card_icon(mosaic_panels_path), _t("Traditional Panels"),
             _t("A few plate-solved panels"), mw._open_panel_mosaic),
            (_card_icon(mosaic_robotic_path), _t("Robotic Mosaic"),
             _t("Seestar and smart telescopes"), mw._open_robotic_mosaic)])

    def _show_choices(self, anchor, choices):
        pop = _Choices(anchor, choices, self)
        pop.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        pop.show()

    def tile_for(self, action):
        """The tile that runs `action`, if it is on screen (First Steps rings it)."""
        for a, tile in self._tiles:
            if a is action and tile.isVisible():
                return tile
        return None

    def _recent_paths(self) -> list:
        return list(getattr(self._mw, "_recent_image_paths", []) or [])

    def _fill_more(self):
        """Recent images; without any, where to start."""
        paths = self._recent_paths()[:RECENTS]
        if self._shown_recents == paths:
            return
        self._shown_recents = paths
        self._placed_for = None                  # the size can differ
        while self._more_lay.count():
            it = self._more_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._recent_buttons = []
        self._tour_link = None
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        head = _label(_t("RECENT IMAGES") if paths else _t("NEW TO SASPRO?"), "sp_k")
        self._more_lay.addWidget(head)
        head.show()
        if paths:
            row.setSpacing(0)
            for p in paths:
                b = _Recent(p, self.thumbs.pixmap(p), self._open_recent)
                self._recent_buttons.append(b)
                row.addWidget(b)
        else:
            from setiastro.saspro.first_steps_tour import sample_image_path
            row.setSpacing(18)
            if sample_image_path() is not None:
                row.addWidget(_link(_t("Open the sample image"), self._open_sample))
            self._tour_link = _link(_t("First Steps tour"),
                                    lambda: self._mw.act_first_steps.trigger())
            row.addWidget(self._tour_link)
            row.addWidget(_link(_t("Beginner video"),
                                lambda: QDesktopServices.openUrl(QUrl(TUTORIAL_URL))))
            row.addWidget(_link(_t("Documentation"), lambda: self._mw.act_docs.trigger()))
        row.addStretch(1)
        self._more_lay.addWidget(box)
        # Added to a visible panel, children stay hidden until the next event
        # loop pass, and a layout measures hidden children as nothing.
        for w in [box, *box.findChildren(QWidget)]:
            w.show()
        self._sync_tour_link()

    def _sync_tour_link(self):
        """Not offered while the tour is already running."""
        tour = getattr(self._mw, "_first_steps_tour", None)
        try:
            running = tour is not None and tour.isVisible()
        except RuntimeError:
            running = False
        if self._tour_link is not None and self._tour_link.isHidden() != running:
            self._tour_link.setVisible(not running)

    def _open_sample(self):
        from setiastro.saspro.first_steps_tour import open_sample_image
        open_sample_image(self._mw)

    def _open_recent(self, path):
        self._mw._open_recent_image(path)

    # ---- recent images ------------------------------------------------------
    def recents_changed(self):
        """The recent list changed: an image was opened, saved or the list cleared."""
        self._fill_more()
        self._settle.start()
        self._capture.start()

    def _capture_open_images(self):
        paths = self._recent_paths()
        self.thumbs.keep_only(paths)
        wanted = {os.path.normcase(p): p for p in paths[:2 * RECENTS]}
        for doc in list(getattr(self._mw.docman, "_docs", []) or []):
            try:
                meta = getattr(doc, "metadata", None) or {}
                # "file.xisf::XISF[0]": one image of several in a file
                path = str(meta.get("file_path") or "").split("::")[0]
                image = getattr(doc, "image", None)
                if not path or image is None:
                    continue
                # pop: one image per file, the first one opened from it
                p = wanted.pop(os.path.normcase(os.path.abspath(path)), None)
                if p is not None:
                    self.thumbs.from_image(p, image)
            except Exception:
                continue

    def read_missing(self):
        """Thumbnails for recents never opened in this version, where cheap."""
        for p in (self._shown_recents or []):
            self.thumbs.from_file(p)             # does nothing if one is there

    def _thumb_made(self, path):
        for b in self._recent_buttons:
            if b.path == path:
                b.set_picture(self.thumbs.pixmap(path))

    # ---- showing ------------------------------------------------------------
    def on_startup(self) -> bool:
        return self._mw.settings.value(STARTUP_KEY, True, type=bool)

    def startup_changed(self):
        self._startup.setChecked(self.on_startup())

    def sync(self):
        """Show, place or hide, for the workspace as it is now."""
        mw = self._mw
        closing = getattr(mw, "_shutting_down", False)
        images = bool(mw.mdi.subWindowList())
        if closing or images or not self.on:
            was_up = not self.isHidden()
            self._watch.stop()
            self._placed_for = None
            self.hide()
            if images and was_up and not closing and not self.on_startup():
                mw._set_start_panel(False)       # it was up for that one time
            return
        if not self._watch.isActive():
            self._watch.start()
        self._sync_tour_link()
        self._place()

    def _in_the_way(self) -> list:
        """Desktop shortcuts and corner cards, in workspace coordinates."""
        out = []
        mgr = getattr(self._mw, "shortcuts", None)
        for w in list(getattr(mgr, "widgets", {}).values()):
            try:
                if w.isVisible():
                    out.append(QRect(w.mapTo(self._vp, QPoint(0, 0)), w.size()))
            except RuntimeError:
                continue
        for name in ("_setup_pill", "_first_steps_tour"):
            w = getattr(self._mw, name, None)
            try:
                if w is not None and w.isVisible():
                    top = self._vp.mapFrom(self._mw, w.pos())
                    out.append(QRect(top, w.size()))
            except RuntimeError:
                continue
        return out

    def _place(self):
        area = self._vp.size()
        blocks = [(r.left() - GAP, r.top() - GAP, r.right() + GAP, r.bottom() + GAP)
                  for r in self._in_the_way()]
        key = (area.width(), area.height(), tuple(blocks))
        if key == self._placed_for:
            return                               # nothing moved (or still no room)
        width = min(WIDTH, area.width() - 40)
        spot = None
        if width >= 600:
            for full in (True, False):
                self._more.setVisible(full)
                if full and self._recent_buttons:
                    fit = max(1, (width - 52) // self._recent_buttons[0].width())
                    for i, b in enumerate(self._recent_buttons):
                        b.setVisible(i < fit)
                self._more_lay.invalidate()      # refilled while hidden, the
                self.layout().invalidate()       # old size would still be cached
                size = QSize(width, self.layout().totalSizeHint().height())
                spot = _free_spot(area, size, blocks)
                if spot is not None:
                    self.setGeometry(QRect(spot, size))
                    break
        self._placed_for = key
        if spot is None:
            self.hide()
            return
        if not self.isVisible():
            self.show()
            self.raise_()

    # ---- the workspace underneath -------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self._vp:
            if ev.type() == QEvent.Type.Resize:
                self._settle.start()
            elif ev.type() in (QEvent.Type.ChildAdded, QEvent.Type.ChildRemoved):
                # A view came or went. subWindowActivated doesn't always say so
                # for the last one to close, and documentRemoved comes before
                # its window is gone.
                self._mw._views_sync_timer.start()
        return False

    def mousePressEvent(self, ev):
        ev.accept()                              # not a click on the workspace

    def mouseReleaseEvent(self, ev):
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        ev.accept()

    def dragEnterEvent(self, ev):
        """Whatever the workspace takes can be dropped here: its shortcut layer
        comes up above the panel for the rest of the drag."""
        layer = getattr(getattr(self._mw, "shortcuts", None), "canvas", None)
        if layer is None:
            ev.ignore()
            return
        layer.dragEnterEvent(ev)

    def dropEvent(self, ev):
        layer = getattr(getattr(self._mw, "shortcuts", None), "canvas", None)
        if layer is None:
            ev.ignore()
            return
        layer.dropEvent(ev)

    def stop(self):
        """SASpro is closing."""
        for t in (self._watch, self._capture, self._settle):
            t.stop()
        self.thumbs.stop()


def _free_spot(area: QSize, size: QSize, blocks):
    """Where `size` fits without touching `blocks`, as near as possible to the
    middle (a little above it); None if nowhere."""
    max_x, max_y = area.width() - size.width() - GAP, area.height() - size.height() - GAP
    if max_x < GAP or max_y < GAP:
        return None
    cx, cy = (area.width() - size.width()) // 2, round((area.height() - size.height()) * 0.45)

    def clear(x, y):
        r, b = x + size.width(), y + size.height()
        return all(x > right or r < left or y > bottom or b < top
                   for left, top, right, bottom in blocks)

    if clear(cx, cy):
        return QPoint(cx, cy)
    xs = sorted({GAP, max_x, *range(cx % 24 or GAP, max_x, 24)} - {0})
    ys = sorted({GAP, max_y, *range(cy % 24 or GAP, max_y, 24)} - {0})
    spots = [(abs(x - cx) + abs(y - cy), x, y) for x in xs for y in ys
             if GAP <= x <= max_x and GAP <= y <= max_y]
    for _d, x, y in sorted(spots):
        if clear(x, y):
            return QPoint(x, y)
    return None


def install(main_window):
    """Create the panel. Called at the end of the main window's __init__."""
    panel = StartPanel(main_window)
    main_window._start_panel = panel
    # Once startup is out of the way: thumbnails for what is already on the list.
    QTimer.singleShot(4000, panel.read_missing)
    return panel
