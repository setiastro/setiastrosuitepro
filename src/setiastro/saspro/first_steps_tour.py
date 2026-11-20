# src/setiastro/saspro/first_steps_tour.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
First Steps: a small card in the corner of the workspace that walks a new
user through a first edit -- open an image, Statistical Stretch, Curves.

It moves on when the user actually does each step, not on a Next button:
an image opening, then a "Statistical Stretch ..." and a "Curves" entry
landing in an image's history. An image that already looks stretched skips
straight to Curves. Step 1 also offers the bundled sample, a stacked linear
16-bit TIFF (images/sample_image.tif), for anyone without an image to hand.

Starts once, after the welcome screen closes, on a brand-new install (no
saved window layout) -- so `--firstrun` shows it every time -- and any time
from About > First Steps Tour. Skipping or finishing marks it done.

Wording lives in _STEPS below, so it can be tuned without touching the logic.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt6.QtCore import (QCoreApplication, QEvent, QPoint, QRect, QSettings, Qt,
                          QTimer, QUrl)
from PyQt6.QtGui import (QColor, QDesktopServices, QFont, QFontMetrics, QPainter,
                         QPen, QRegion)
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QToolBar,
                             QVBoxLayout, QWidget)

DONE_KEY = "ui/first_steps_tour_done_v1"
TUTORIAL_URL = "https://youtu.be/zBUOZl2OgaY"     # the welcome screen's beginner video
# An image whose median is below this is treated as linear (unstretched) data.
LINEAR_MEDIAN = 0.08


def _t(text: str) -> str:
    return QCoreApplication.translate("FirstStepsTour", text)


# step key -> (counter, title, body, button label, button action)
# Actions: "open" = File > Open, "stretch" / "curves" = open that tool,
# "skip_to_curves" = go to the Curves step, "tutorial" = the video.
_STEPS = {
    "open": (
        "1 of 3", "Open an image",
        "Double-click the empty workspace, or use File → Open.\n\n"
        "Any image works. A stacked, unstretched FITS, XISF or TIFF straight "
        "from your stacking software is ideal.",
        "Open an image…", "open"),
    "stretch": (
        "2 of 3", "Stretch it with Statistical Stretch",
        "Stacked data is linear: nearly everything sits close to black, which "
        "is why it looks dark. If Display-Stretch is on in the toolbar, the view "
        "is brightened but the pixels are not.\n\n"
        "Statistical Stretch changes the pixels. It is under Functions → "
        "Stretching & Tone → Statistical Stretch. Click Apply when it looks right.",
        "Open Statistical Stretch", "stretch"),
    "already_stretched": (
        "2 of 3", "This image is already stretched",
        "Its brightness is already spread out, so Statistical Stretch, which is "
        "for linear data, isn't needed here. Go straight to Curves.",
        "Continue to Curves", "skip_to_curves"),
    "curves": (
        "3 of 3", "Add contrast with Curves",
        "Curves is under Functions → Stretching & Tone → Curves Editor. "
        "Lift the middle of the curve a little to bring out faint detail, or "
        "make a gentle S-curve for contrast, then click Apply.",
        "Open Curves Editor", "curves"),
    "done": (
        "Done", "You've processed your first image",
        "Ctrl+Z undoes any step, and the History menu lists them all.\n\n"
        "A full workflow removes gradients and calibrates colour before "
        "stretching. The beginner tutorial walks through it.",
        "Watch the tutorial", "tutorial"),
}

_STYLE = """
#FirstStepsTour {
    background-color: #1a1a2e;
    border: 1px solid #3a3a6a;
    border-radius: 10px;
}
#fst_counter { color: #7788cc; font-size: 9pt; font-weight: bold; letter-spacing: 1px; }
#fst_title   { color: #ffffff; font-size: 12pt; font-weight: bold; }
#fst_body    { color: #c8c8dc; font-size: 10pt; }
#fst_close, #fst_skip {
    background: transparent; border: none; color: #8888bb; font-size: 9pt;
}
#fst_close:hover, #fst_skip:hover { color: #ffffff; }
#fst_sample {
    background: transparent; border: none; color: #8fa0ff; font-size: 10pt;
    text-align: left; padding: 0px;
}
#fst_sample:hover { color: #ffffff; text-decoration: underline; }
#fst_primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3344bb, stop:1 #5566dd);
    color: #ffffff; border: none; border-radius: 7px; padding: 7px 14px;
    font-weight: bold;
}
#fst_primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4455cc, stop:1 #6677ee);
}
"""


# step -> the action whose toolbar icon and menu path the spotlight marks
_STEP_ACTION = {
    "open": "act_open",
    "stretch": "act_stat_stretch",
    "already_stretched": "act_curves",
    "curves": "act_curves",
}
_RING = QColor(110, 130, 255)          # the card's blue


def _menu_paths(menu, act, depth=0):
    """Every [(menu, item to highlight in it), ...] chain from `menu` to `act`."""
    if menu is None or depth > 4:
        return
    for a in menu.actions():
        if a is act:
            yield [(menu, a)]
        elif a.menu() is not None:
            for rest in _menu_paths(a.menu(), act, depth + 1):
                yield [(menu, a)] + rest


def _menu_route(main_window, act):
    """(menubar title action, path) for the route the step text describes.

    A tool can be in two places under one title: its category submenu and the
    "All Functions" flat list, which comes first. The route whose last menu
    is smallest is the category one (Functions > Stretching & Tone)."""
    mb = main_window.menuBar()
    for top in mb.actions():
        paths = list(_menu_paths(top.menu(), act))
        if paths:
            return top, min(paths, key=lambda p: len(p[-1][0].actions()))
    return None, []


class _Spotlight(QWidget):
    """Pulsing rings on where the current step's tool lives: its toolbar icon
    (with the tool's name under it) and its menu title.

    A transparent child spread over the main window that lets every click
    through. Only the ring areas are repainted each tick, so the image views
    underneath are not redrawn 20 times a second. An icon squeezed into a
    toolbar's >> overflow, or on a hidden toolbar, is skipped: the menu ring
    still shows the way."""

    _PAD = 14     # ring growth + pen width around a target

    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window
        self._act = None
        self._shapes: list = []        # [(ring QRect, pill QRect | None, label)]
        self._phase = 0.0
        self._font = QFont(self.font())
        self._font.setPointSize(9)
        self._font.setBold(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def point_at(self, act):
        self._act = act
        if act is None:
            self._timer.stop()
            self._set_shapes([])
            self.hide()
            return
        self.setGeometry(self._mw.rect())
        self.show()
        self.raise_()
        self._timer.start()
        self._tick()

    def stop(self):
        self.point_at(None)

    # ---- where to draw ---------------------------------------------------
    def _targets(self):
        act, mw = self._act, self._mw
        out = []
        if act is None:
            return out
        label = act.text().replace("&", "").rstrip(".… ").strip()
        for tb in mw.findChildren(QToolBar):
            try:
                if not tb.isVisible() or tb.isFloating():
                    continue
                w = tb.widgetForAction(act)
                if w is not None and w.isVisible() and w.width() > 1:
                    out.append((QRect(w.mapTo(mw, QPoint(0, 0)), w.size()), label))
                    break
            except RuntimeError:
                continue
        mb = mw.menuBar()
        try:
            if mb is not None and mb.isVisible() and not mb.isNativeMenuBar():
                top, _path = _menu_route(mw, act)
                if top is not None:
                    g = mb.actionGeometry(top)
                    if g.isValid() and g.width() > 0:
                        out.append((QRect(mb.mapTo(mw, g.topLeft()), g.size()), ""))
        except RuntimeError:
            pass
        return out

    def _shape_for(self, rect: QRect, label: str):
        ring = rect.adjusted(-self._PAD, -self._PAD, self._PAD, self._PAD)
        pill = None
        if label:
            fm = QFontMetrics(self._font)
            w, h = fm.horizontalAdvance(label) + 16, fm.height() + 8
            x = rect.center().x() - w // 2
            x = max(4, min(x, self.width() - w - 4))
            pill = QRect(x, rect.bottom() + 10, w, h)
        return ring, pill, label

    def _region(self, shapes) -> QRegion:
        reg = QRegion()
        for ring, pill, _label in shapes:
            reg = reg.united(QRegion(ring.adjusted(-2, -2, 2, 2)))
            if pill is not None:
                reg = reg.united(QRegion(pill.adjusted(-2, -2, 2, 2)))
        return reg

    def _set_shapes(self, shapes):
        dirty = self._region(self._shapes).united(self._region(shapes))
        self._shapes = shapes
        if not dirty.isEmpty():
            self.update(dirty)

    def _tick(self):
        if self.size() != self._mw.size():
            self.setGeometry(self._mw.rect())
        self._phase = (self._phase + 0.25) % (2 * math.pi)
        self._set_shapes([self._shape_for(r, lab) for r, lab in self._targets()])

    def paintEvent(self, ev):
        if not self._shapes:
            return
        a = 0.5 + 0.5 * math.sin(self._phase)          # 0..1 pulse
        grow = 3 + int(round(4 * a))
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for ring, pill, label in self._shapes:
            r = ring.adjusted(self._PAD - grow, self._PAD - grow,
                              grow - self._PAD, grow - self._PAD)
            glow = QColor(_RING)
            glow.setAlpha(int(35 + 55 * a))
            p.setPen(QPen(glow, 7))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r, 7, 7)
            line = QColor(_RING)
            line.setAlpha(int(170 + 85 * a))
            p.setPen(QPen(line, 2.5))
            p.drawRoundedRect(r, 7, 7)
            if pill is not None:
                p.setPen(QPen(line, 1.5))
                p.setBrush(QColor(26, 26, 46, 235))
                p.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)
                p.setPen(QColor(255, 255, 255))
                p.setFont(self._font)
                p.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)
        p.end()


def _alive(w) -> bool:
    try:
        from PyQt6 import sip
        return w is not None and not sip.isdeleted(w)
    except Exception:
        return w is not None


def looks_linear(image) -> bool:
    """True for unstretched data: the median of a coarse sample sits near black."""
    try:
        a = np.asarray(image)
        if a.size == 0:
            return True
        step = max(1, int(max(a.shape[:2]) // 512))
        s = a[::step, ::step]
        if s.ndim == 3:
            s = s.mean(axis=2)
        med = float(np.nanmedian(s))
        return (not np.isfinite(med)) or med < LINEAR_MEDIAN
    except Exception:
        return True


def sample_image_path() -> str | None:
    """The bundled sample (images/sample_image.tif), or None if this build lacks it."""
    try:
        from setiastro.saspro import resources as _res
        p = _res.sample_image_path
        return p if p and os.path.isfile(p) else None
    except Exception:
        return None


def open_sample_image(main_window):
    """Open the bundled sample as an UNTITLED document.

    Untitled on purpose: opened by its path, Save would write over the
    bundled file -- in a pip install that write succeeds, and the next new
    user would get someone's processed sample. Untitled, Save asks where.
    """
    path = sample_image_path()
    dm = getattr(main_window, "docman", None) or getattr(main_window, "doc_manager", None)
    if path is None or dm is None:
        return None
    from setiastro.saspro.legacy.image_manager import load_image
    img, header, bit_depth, is_mono = load_image(path)[:4]
    if img is None:
        return None
    return dm.open_array(img, {"original_header": header, "bit_depth": bit_depth,
                               "is_mono": is_mono},
                         title=_t("Sample image (linear)"))


def _last_step_name(doc) -> str:
    """Step name of the newest entry in a document's history ('' if none)."""
    try:
        undo = getattr(doc, "_undo", None)
        if undo:
            entry = undo[-1]
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                return str(entry[2] or "")
    except Exception:
        pass
    return ""


class FirstStepsTour(QFrame):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.setObjectName("FirstStepsTour")
        self.setStyleSheet(_STYLE)
        self.setFixedWidth(380)
        self._mw = main_window
        self._dm = getattr(main_window, "docman", None) or getattr(main_window, "doc_manager", None)
        self._watched: list = []
        self._step = ""
        self._torn_down = False
        self._spot = _Spotlight(main_window)
        self._menu_hooks: list = []       # [(QMenu, slot)] while a menu route is guided

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 12, 14)
        lay.setSpacing(6)
        head = QHBoxLayout()
        self._counter = QLabel(objectName="fst_counter")
        btn_close = QPushButton("✕", objectName="fst_close")
        btn_close.setToolTip(_t("Close the tour"))
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(lambda: self.finish())
        head.addWidget(self._counter)
        head.addStretch(1)
        head.addWidget(btn_close)
        lay.addLayout(head)
        self._title = QLabel(objectName="fst_title")
        self._title.setWordWrap(True)
        self._body = QLabel(objectName="fst_body")
        self._body.setWordWrap(True)
        lay.addWidget(self._title)
        lay.addWidget(self._body)
        self._sample = QPushButton(_t("No image handy? Open the sample image"),
                                   objectName="fst_sample")
        self._sample.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sample.setToolTip(_t("A stacked, unstretched 16-bit image bundled with SASpro"))
        self._sample.clicked.connect(lambda: open_sample_image(self._mw))
        lay.addWidget(self._sample)
        lay.addSpacing(6)
        row = QHBoxLayout()
        self._skip = QPushButton(_t("Skip tour"), objectName="fst_skip")
        self._skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip.clicked.connect(lambda: self.finish())
        self._primary = QPushButton(objectName="fst_primary")
        self._primary.setCursor(Qt.CursorShape.PointingHandCursor)
        self._primary.clicked.connect(self._on_primary)
        row.addWidget(self._skip)
        row.addStretch(1)
        row.addWidget(self._primary)
        lay.addLayout(row)

        main_window.installEventFilter(self)
        if self._dm is not None:
            self._dm.documentAdded.connect(self._on_doc_added)
            try:
                for d in list(self._dm.all_documents() or []):
                    self._watch(d)
            except Exception:
                pass

        active = None
        try:
            active = main_window._active_doc()
        except Exception:
            pass
        if active is not None and getattr(active, "image", None) is not None:
            self._go("stretch" if looks_linear(active.image) else "already_stretched")
        else:
            self._go("open")

    # ---- steps ---------------------------------------------------------------
    def _go(self, step: str):
        self._step = step
        counter, title, body, button, _action = _STEPS[step]
        self._counter.setText(_t("FIRST STEPS") + "  ·  " + _t(counter))
        self._title.setText(_t(title))
        self._body.setText(_t(body))
        self._primary.setText(_t(button))
        self._skip.setText(_t("Close") if step == "done" else _t("Skip tour"))
        self._sample.setVisible(step == "open" and sample_image_path() is not None)
        self.adjustSize()
        self._reposition()
        act = getattr(self._mw, _STEP_ACTION[step], None) if step in _STEP_ACTION else None
        self._spot.point_at(act)
        self._guide_menus(act)
        self.raise_()                       # the card stays above the spotlight

    # ---- the menu route ------------------------------------------------------
    def _guide_menus(self, act):
        """While the step's tool is the target, opening its menu highlights the
        route: Functions opens with "Stretching & Tone" active (which opens it),
        and that opens with the tool itself active. Mouse and keys work as usual."""
        self._unguide_menus()
        if act is None:
            return
        try:
            _top, path = _menu_route(self._mw, act)
        except Exception:
            path = []
        for menu, item in path:
            def _slot(m=menu, it=item):
                QTimer.singleShot(0, lambda: _alive(m) and m.isVisible() and m.setActiveAction(it))
            try:
                menu.aboutToShow.connect(_slot)
                self._menu_hooks.append((menu, _slot))
            except Exception:
                pass

    def _unguide_menus(self):
        for menu, slot in self._menu_hooks:
            try:
                menu.aboutToShow.disconnect(slot)
            except Exception:
                pass
        self._menu_hooks.clear()

    def _on_primary(self):
        action = _STEPS[self._step][4]
        if action == "open":
            self._trigger("act_open")
        elif action == "stretch":
            self._trigger("act_stat_stretch")
        elif action == "curves":
            self._trigger("act_curves")
        elif action == "skip_to_curves":
            self._go("curves")
        elif action == "tutorial":
            QDesktopServices.openUrl(QUrl(TUTORIAL_URL))
            self.finish()

    def _trigger(self, attr: str):
        act = getattr(self._mw, attr, None)
        if act is not None:
            act.trigger()

    # ---- what the user did ---------------------------------------------------
    def _watch(self, doc):
        if doc is None or any(d is doc for d in self._watched):
            return
        try:
            doc.changed.connect(self._on_doc_changed)
            self._watched.append(doc)
        except Exception:
            pass

    def _on_doc_added(self, doc):
        self._watch(doc)
        if self._step == "open" and getattr(doc, "image", None) is not None:
            self._go("stretch" if looks_linear(doc.image) else "already_stretched")

    def _on_doc_changed(self):
        name = _last_step_name(self.sender())
        if self._step in ("stretch", "already_stretched", "open") and name.startswith("Statistical Stretch"):
            self._go("curves")
        elif self._step in ("stretch", "already_stretched", "curves") and name.startswith("Curves"):
            self._go("done")

    # ---- placement -----------------------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self._mw and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show,
                                             QEvent.Type.LayoutRequest):
            self._reposition()
        return super().eventFilter(obj, ev)

    def _reposition(self):
        """Bottom-right of the workspace, clear of the panel rails."""
        mw = self._mw
        try:
            vp = mw.mdi.viewport()
            tl = vp.mapTo(mw, QPoint(0, 0))
            right, bottom = tl.x() + vp.width(), tl.y() + vp.height()
        except Exception:
            right, bottom = mw.width(), mw.height()
        self.move(max(0, right - self.width() - 20), max(0, bottom - self.height() - 20))

    # ---- finish --------------------------------------------------------------
    def finish(self):
        """Skipped or completed: never start on its own again."""
        try:
            QSettings().setValue(DONE_KEY, True)
        except Exception:
            pass
        self.close()

    def closeEvent(self, ev):
        self._teardown()
        super().closeEvent(ev)

    def _teardown(self):
        # Every connection goes, or a document change after close would call
        # into a deleted widget.
        if self._torn_down:
            return
        self._torn_down = True
        try:
            self._mw.removeEventFilter(self)
        except Exception:
            pass
        self._unguide_menus()
        try:
            self._spot.stop()
            self._spot.deleteLater()
        except Exception:
            pass
        if self._dm is not None:
            try:
                self._dm.documentAdded.disconnect(self._on_doc_added)
            except Exception:
                pass
        for d in self._watched:
            try:
                d.changed.disconnect(self._on_doc_changed)
            except Exception:
                pass
        self._watched.clear()
        if getattr(self._mw, "_first_steps_tour", None) is self:
            self._mw._first_steps_tour = None


def start_first_steps_tour(main_window):
    """Show the tour now (About > First Steps Tour, or a new install)."""
    old = getattr(main_window, "_first_steps_tour", None)
    if _alive(old):
        old.close()
    tour = FirstStepsTour(main_window)
    tour.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    main_window._first_steps_tour = tour
    tour.show()
    tour.raise_()
    return tour


def maybe_start_first_steps_tour(main_window):
    """After the welcome screen: only on a brand-new install, only once."""
    if not getattr(main_window, "_fresh_install", False):
        return None
    if QSettings().value(DONE_KEY, False, type=bool):
        return None
    return start_first_steps_tour(main_window)


def tour_is_showing(main_window) -> bool:
    t = getattr(main_window, "_first_steps_tour", None)
    return _alive(t) and t.isVisible()
