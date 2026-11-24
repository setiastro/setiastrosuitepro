# src/setiastro/saspro/setup_center.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Welcome and Setup: the first thing a new user sees, and the first thing
someone coming back from 1.x sees.

  New install     Welcome, acceleration, AI models, Gaia, ready.
  Back from 1.x   Your layout (new or the one they had), then the same steps.

Each step shows whether it is already done and does it in place: the installs
run in the background (setup_manager.py), so the window can be closed and the
steps can be visited in any order. Not modal, so SASpro stays usable and a
layout choice shows behind it as it is clicked.

Opened at startup by first_run_dialog.maybe_show_first_run_dialog, from About >
Welcome and Setup, from the reminder in the workspace corner, and from any tool
that needs something not installed yet (offer_setup).
"""
from __future__ import annotations

import os

from PyQt6.QtCore import (QCoreApplication, QEvent, QLocale, QPoint, QPointF, QRectF, QSize, Qt,
                          QTimer, QTranslator, QUrl, pyqtSignal)
from PyQt6.QtGui import (QColor, QDesktopServices, QGuiApplication, QKeySequence,
                         QPainter, QPen, QPixmap)
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QMessageBox, QProgressBar, QPushButton,
                             QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout,
                             QWidget)

from setiastro.saspro import workspace_layout
from setiastro.saspro.setup_manager import (GAIA_SEEN_KEY, REMINDER_OFF_KEY, STEPS,
                                            setup_manager)

TUTORIAL_URL = "https://youtu.be/zBUOZl2OgaY"
DISCORD_URL = "https://discord.gg/vvYH82C82f"


def _t(text: str) -> str:
    return QCoreApplication.translate("Setup", text)


# ── language ────────────────────────────────────────────────────────────────
# The picker speaks the language it offers: the names are each language's own,
# and the restart button reads in the language picked (from that language's
# .qm), so someone who reads no English can still find theirs.

REOPEN_KEY = "ui/setup_reopen_page"      # a language restart brings this window back, on this page


def pick_language(ui_languages, available) -> str | None:
    """The first of the system's UI languages, in its order of preference, that
    SASpro has a translation for. None when English comes first: English is
    what SASpro already speaks. Only Simplified Chinese is translated, so a
    Traditional Chinese system is not offered it."""
    for tag in ui_languages:
        parts = tag.replace("_", "-").lower().split("-")
        base = parts[0]
        if base == "en":
            return None
        if base == "zh" and {"hant", "tw", "hk", "mo"} & set(parts[1:]):
            continue
        if base in available:
            return base
    return None


def system_language() -> str | None:
    from setiastro.saspro.i18n import get_available_languages
    try:
        return pick_language(QLocale.system().uiLanguages(), get_available_languages())
    except Exception:
        return None


_TRANSLATORS: dict = {}


def in_language(code: str, context: str, text: str) -> str:
    """`text` as it reads in `code`, whatever language SASpro is running in."""
    if not code or code == "en":
        return text
    t = _TRANSLATORS.get(code)
    if t is None:
        from setiastro.saspro.i18n import get_translations_dir
        t = QTranslator()
        if not t.load(os.path.join(get_translations_dir(), f"saspro_{code}.qm")):
            t = False
        _TRANSLATORS[code] = t
    return (t.translate(context, text) if t else "") or text


# state -> colour
_COLORS = {
    "ok": QColor("#3dbf9f"), "warn": QColor("#e0a83c"), "todo": QColor("#8899ee"),
    "blocked": QColor("#e0a83c"), "busy": QColor("#6f86ff"), "checking": QColor("#5a5a80"),
}

_STYLE = """
#SetupCenter { background-color: #13131f; }
#sc_header { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0e0e20, stop:1 #1c1c38);
             border-bottom: 2px solid #2e2e5a; }
#sc_title { color: #ffffff; font-size: 20px; font-weight: 700; background: transparent; }
#sc_sub { color: #8888bb; font-size: 12px; background: transparent; }
#sc_version { color: #8888bb; font-size: 12px; background: transparent; }
#sc_rail { background-color: #0f0f1b; border-right: 1px solid #222240; }
#sc_row { background: transparent; border: none; border-radius: 8px; }
#sc_row:hover, #sc_row:focus { background-color: #181830; }
#sc_row[current="true"] { background-color: #20204a; }
#sc_row_title { color: #dde0ff; font-size: 13px; font-weight: 600; background: transparent; }
#sc_row_sub { color: #8888bb; font-size: 11px; background: transparent; }
#sc_scroll, #sc_page { background-color: #13131f; border: none; }
#sc_h1 { color: #ffffff; font-size: 21px; font-weight: 700; }
#sc_lead { color: #c0c0dc; font-size: 13px; }
#sc_small { color: #9a9abf; font-size: 12px; }
#sc_label { color: #7788cc; font-size: 11px; font-weight: 700; }
#sc_sign { color: #7788cc; font-size: 12px; font-style: italic; }
#sc_panel { background-color: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px; }
#sc_state { color: #ffffff; font-size: 15px; font-weight: 600; background: transparent; }
#sc_detail { color: #aaaacc; font-size: 12px; background: transparent; }
#sc_choice { background-color: #1a1a2e; border: 2px solid #2a2a4a; border-radius: 12px; }
#sc_choice:hover, #sc_choice:focus { border-color: #5a5a9c; }
#sc_choice[chosen="true"] { border-color: #6f86ff; background-color: #1f1f3f; }
#sc_choice_title { color: #ffffff; font-size: 15px; font-weight: 700; background: transparent; }
#sc_choice_body { color: #aaaacc; font-size: 12px; background: transparent; }
#sc_footer { background-color: #0e0e1c; border-top: 1px solid #222240; }
#sc_globe { font-size: 16px; background: transparent; }
QComboBox#sc_lang { background-color: #1e1e40; color: #dde0ff; border: 1px solid #3a3a70;
    border-radius: 7px; padding: 5px 24px 5px 10px; font-size: 12px; }
QComboBox#sc_lang:hover, QComboBox#sc_lang:focus { border-color: #5566aa; }
QComboBox#sc_lang::drop-down { subcontrol-origin: padding; subcontrol-position: center right;
    width: 18px; border: none; }
QComboBox#sc_lang QAbstractItemView { background-color: #1a1a2e; color: #dde0ff;
    border: 1px solid #3a3a70; selection-background-color: #28285a; outline: none; }
QPushButton#sc_primary { color: #ffffff; border: none; border-radius: 7px; padding: 9px 20px;
    font-size: 13px; font-weight: 700;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3344bb, stop:1 #5566dd); }
QPushButton#sc_primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4455cc, stop:1 #6677ee); }
QPushButton#sc_primary:disabled { background: #2a2a44; color: #77779a; }
QPushButton#sc_plain { background-color: #1e1e40; color: #a9b8ff; border: 1px solid #3a3a70;
    border-radius: 7px; padding: 8px 16px; font-size: 12px; font-weight: 600; }
QPushButton#sc_plain:hover { background-color: #28285a; color: #ffffff; border-color: #5566aa; }
QPushButton#sc_plain:disabled { color: #66668a; border-color: #2a2a4a; background-color: #171728; }
QPushButton#sc_plain[chosen="true"] { border-color: #6f86ff; color: #ffffff; background-color: #28285a; }
QPushButton#sc_link { background: transparent; border: none; color: #8fa0ff; font-size: 12px;
    text-align: left; padding: 2px 0px; }
QPushButton#sc_link:hover { color: #ffffff; text-decoration: underline; }
QPushButton:focus { outline: none; }
QPushButton#sc_plain:focus, QPushButton#sc_primary:focus { border: 1px solid #aab4ff; }
QProgressBar { background-color: #0f0f1b; border: 1px solid #2a2a4a; border-radius: 5px; max-height: 8px; min-height: 8px; }
QProgressBar::chunk { background-color: #6f86ff; border-radius: 4px; }
QScrollBar:vertical { background: #1a1a2e; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #3a3a6a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""


# ── small pieces ────────────────────────────────────────────────────────────

class _Dot(QWidget):
    """A step's state: tick, number, '!' or a turning arc."""

    def __init__(self, size=26, number=0, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._state, self._number, self._angle = "checking", number, 0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._turn)

    def set_state(self, state: str):
        if state == self._state:
            return
        self._state = state
        self._timer.start() if state in ("busy", "checking") else self._timer.stop()
        self.update()

    def _turn(self):
        if self.isVisible():
            self._angle = (self._angle + 9) % 360
            self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = _COLORS.get(self._state, _COLORS["todo"])
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        if self._state in ("busy", "checking"):
            p.setPen(QPen(QColor("#2a2a4a"), 2.5))
            p.drawEllipse(r)
            p.setPen(QPen(c, 2.5))
            p.drawArc(r, int(-self._angle * 16), 100 * 16)
        elif self._state == "ok":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawEllipse(r)
            p.setPen(QPen(QColor("#0f1f1b"), max(2.0, r.width() / 10),
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin))
            w, h, x, y = r.width(), r.height(), r.x(), r.y()
            p.drawPolyline([QPoint(int(x + .27 * w), int(y + .53 * h)),
                            QPoint(int(x + .44 * w), int(y + .69 * h)),
                            QPoint(int(x + .74 * w), int(y + .35 * h))])
        else:
            p.setPen(QPen(c, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(r)
            f = p.font()
            f.setPixelSize(int(r.height() * 0.55))
            f.setBold(True)
            p.setFont(f)
            mark = "!" if self._state in ("warn", "blocked") else (str(self._number) or "")
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, mark if mark != "0" else "")
        p.end()


class _Combo(QComboBox):
    """The style sheet's drop-down has no arrow of its own: draw a chevron."""

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#a9b8ff"), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                      Qt.PenJoinStyle.RoundJoin))
        x, y = self.width() - 15.0, self.height() / 2.0
        p.drawPolyline([QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2)])
        p.end()


def _label(text, name, wrap=True) -> QLabel:
    lb = QLabel(text)
    lb.setObjectName(name)
    lb.setWordWrap(wrap)
    if wrap:
        lb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lb


def _button(text, name, slot) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName(name)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setAutoDefault(False)
    b.clicked.connect(lambda _=False: slot())
    return b


def _restyle(w):
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


class _Page(QWidget):
    """A page as tall as its text is at the width it has. The default is as
    tall as it would be at its narrowest, which scrolls for no reason."""

    def minimumSizeHint(self):
        lay = self.layout()
        if lay is None:
            return super().minimumSizeHint()
        return QSize(0, lay.heightForWidth(self.width()))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if ev.size().width() != ev.oldSize().width():
            self.updateGeometry()


class _RailRow(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title, number):
        super().__init__()
        self.setObjectName("sc_row")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 9, 10, 9)
        lay.setSpacing(11)
        self.dot = _Dot(26, number)
        lay.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = _label(title, "sc_row_title")
        self.sub = _label("", "sc_row_sub")
        col.addWidget(self.title)
        col.addWidget(self.sub)
        col.addStretch(1)
        lay.addLayout(col, 1)
        self.set_sub("")

    def set_sub(self, text):
        self.sub.setText(text)
        self.sub.setVisible(bool(text))

    def set_current(self, on):
        self.setProperty("current", "true" if on else "false")
        _restyle(self)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(ev)


# ── layout choice ───────────────────────────────────────────────────────────

_THEME_PAINT = {          # window, workspace, bar, line
    "gray": ("#3a3a3a", "#242424", "#333333", "#5a5a5a"),
    "dark": ("#232323", "#171717", "#1c1c1c", "#454545"),
    "light": ("#e4e4e4", "#c4c4c4", "#d8d8d8", "#9a9a9a"),
}


def _layout_picture(kind: str, theme: str, second_window: bool) -> QPixmap:
    """A schematic of a window: the new layout, or a fuller older one."""
    w, h, dpr = 250, 132, 2
    pm = QPixmap(w * dpr, h * dpr)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    win, work, bar, line = (QColor(c) for c in _THEME_PAINT.get(theme, _THEME_PAINT["gray"]))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    icons = [QColor(c) for c in ("#e05252", "#e0a83c", "#3dbf9f", "#4f8fe0", "#b06fe0")]

    def panel(x, y, pw, ph):
        p.setPen(QPen(line, 1))
        p.setBrush(win)
        p.drawRect(QRectF(x, y, pw, ph))
        p.setPen(QPen(line, 2))
        for i in range(int((ph - 8) // 7)):
            p.drawLine(int(x + 5), int(y + 8 + 7 * i), int(x + pw * (.85 if i % 3 else .6)), int(y + 8 + 7 * i))

    def row(y, x0, x1, step=11, start=0):
        p.setPen(Qt.PenStyle.NoPen)
        x, i = x0, start
        while x < x1 - 7:
            p.setBrush(icons[i % 5])
            p.drawRoundedRect(QRectF(x, y, 7, 7), 2, 2)
            x, i = x + step, i + 1

    right = w - 40 if second_window else w
    p.setPen(QPen(line, 1))
    p.setBrush(win)
    p.drawRoundedRect(QRectF(.5, .5, right - 1, h - 1), 5, 5)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bar)
    p.drawRect(QRectF(1, 8, right - 2, 26))
    if kind == "new":
        row(11, 8, right * .92)
        row(23, 8, right * .78, start=2)
        p.setBrush(work)
        p.drawRect(QRectF(13, 35, right - 26, h - 47))
        p.setPen(QPen(QColor("#4a4a86"), 1))     # the start panel
        p.setBrush(QColor("#14142a"))
        p.drawRoundedRect(QRectF(right / 2 - 42, 50, 84, 46), 4, 4)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(6):
            p.setBrush(QColor("#4652cf" if i == 0 else "#2b2b52"))
            p.drawRoundedRect(QRectF(right / 2 - 36 + 25 * (i % 3), 56 + 11 * (i // 3), 22, 8), 2, 2)
        p.setBrush(QColor("#3a3a66"))
        for i in range(4):
            p.drawRoundedRect(QRectF(right / 2 - 36 + 18.6 * i, 80, 16, 11), 2, 2)
        for y in (43, 55, 67, 79):
            row(y, right - 10, right - 2, start=y)
        row(43, 3, 11)
        row(h - 10, 16, 62, start=2)
    else:
        row(11, 8, right * .92)
        row(23, 8, right * .78, start=2)
        p.setBrush(work)
        p.drawRect(QRectF(1, 35, right - 2, h - 37))
        if not second_window:
            panel(1, 35, 44, h - 58)
            panel(right - 52, 35, 50, 44)
            panel(right - 52, 79, 50, h - 102)
            panel(1, h - 23, right - 3, 22)
        else:
            p.setPen(QPen(line, 1))
            p.setBrush(win)
            p.drawRoundedRect(QRectF(right + 5, 14, 34, h - 26), 4, 4)
            panel(right + 8, 22, 28, 36)
            panel(right + 8, 62, 28, 46)
    p.end()
    return pm


class _ChoiceCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, title, body, picture):
        super().__init__()
        self.setObjectName("sc_choice")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)
        pic = QLabel()
        pic.setPixmap(picture)
        pic.setStyleSheet("background: transparent;")
        lay.addWidget(pic, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(_label(title, "sc_choice_title"))
        lay.addWidget(_label(body, "sc_choice_body"))
        lay.addStretch(1)

    def set_chosen(self, on):
        self.setProperty("chosen", "true" if on else "false")
        _restyle(self)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
        else:
            super().keyPressEvent(ev)


def saved_layout_text(settings) -> str:
    """The set-aside layout in a few words, so it is clearly theirs."""
    themes = {"dark": _t("Dark theme"), "gray": _t("Gray theme"), "light": _t("Light theme"),
              "custom": _t("Custom theme")}
    out = []
    for what, n in workspace_layout.describe_saved_layout(settings):
        if what == "theme":
            out.append(themes.get(n, _t("System theme")))
        elif what == "background":
            out.append(_t("your own background"))
        elif what == "moved":
            out.append(_t("{0} moved icons").format(n) if n != 1 else _t("1 moved icon"))
        elif what == "hidden":
            out.append(_t("{0} hidden icons").format(n) if n != 1 else _t("1 hidden icon"))
        elif what == "second_window":
            out.append(_t("panels in a second window"))
    return ", ".join(out)


# ── the state panel of a step ───────────────────────────────────────────────

class _StatePanel(QFrame):
    """Dot, state, detail, buttons, progress."""

    def __init__(self):
        super().__init__()
        self.setObjectName("sc_panel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        top = QHBoxLayout()
        top.setSpacing(14)
        self.dot = _Dot(32)
        top.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(3)
        self.state = _label("", "sc_state")
        self.detail = _label("", "sc_detail")
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col.addWidget(self.state)
        col.addWidget(self.detail)
        top.addLayout(col, 1)
        lay.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.progress = _label("", "sc_detail")
        self.progress.hide()
        lay.addWidget(self.progress)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(10)
        lay.addLayout(self.buttons)

    def add(self, *buttons):
        for b in buttons:
            self.buttons.addWidget(b)
        self.buttons.addStretch(1)

    def show_item(self, item):
        self.dot.set_state(item.state)
        self.state.setText(item.headline or _t("Checking"))
        self.detail.setText(item.detail)
        self.detail.setVisible(bool(item.detail))
        self.bar.hide()
        self.progress.hide()

    def show_job(self, headline, job):
        self.dot.set_state("busy")
        self.state.setText(headline)
        self.detail.hide()
        if job.percent is None:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(job.percent * 10))
        self.bar.show()
        self.progress.setText(f"{job.text}\n" + _t("{0} elapsed").format(job.elapsed()))
        self.progress.show()


# ── the window ──────────────────────────────────────────────────────────────

class SetupCenter(QDialog):
    """Show with open_setup_center()."""

    def __init__(self, main_window, start: str | None = None):
        super().__init__(main_window)
        self._mw = main_window
        self._mgr = setup_manager(main_window)
        s = main_window.settings
        reopen = s.value(REOPEN_KEY, "", type=str)
        if reopen:                               # back from a language restart
            s.remove(REOPEN_KEY)
            start = start or reopen
        self._asking = workspace_layout.choice_pending(s)
        self._has_old = workspace_layout.has_saved_layout(s)
        self._last_error: dict = {}
        self._saw_gaia = False

        self.setObjectName("SetupCenter")
        self.setModal(False)
        self.setWindowTitle(_t("Welcome and Setup"))
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setStyleSheet(_STYLE)

        # Pages, in order. Back from 1.x, the layout question opens instead of
        # the welcome; afterwards it stays available as its own page.
        self._keys = ["layout"] if self._asking else ["welcome"]
        self._keys += list(STEPS) + ["ready"]
        if self._has_old and not self._asking:
            self._keys.insert(1, "layout")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        mid = QHBoxLayout()
        mid.setSpacing(0)
        root.addLayout(mid, 1)
        mid.addWidget(self._build_rail())
        self._stack = QStackedWidget()
        mid.addWidget(self._stack, 1)
        build = {"welcome": self._page_welcome, "layout": self._page_layout,
                 "accel": self._page_accel, "models": self._page_models,
                 "gaia": self._page_gaia, "ready": self._page_ready}
        self._pages = {}
        for k in self._keys:
            scroll = QScrollArea()
            scroll.setObjectName("sc_scroll")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = _Page()
            page.setObjectName("sc_page")
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            lay = QVBoxLayout(page)
            lay.setContentsMargins(30, 26, 30, 22)
            lay.setSpacing(14)
            build[k](lay)
            lay.addStretch(1)
            scroll.setWidget(page)
            self._pages[k] = scroll
            self._stack.addWidget(scroll)
        root.addWidget(self._build_footer())

        avail = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        self.resize(min(960, avail.width() - 60), min(660, avail.height() - 80))

        self._mgr.changed.connect(self._sync)
        self._mgr.finished.connect(self._on_job_finished)
        self._mgr.refresh()
        self._go(start if start in self._keys else self._keys[0])

    # ---- frame -----------------------------------------------------------
    def _build_header(self) -> QWidget:
        head = QWidget()
        head.setObjectName("sc_header")
        head.setFixedHeight(82)
        lay = QHBoxLayout(head)
        lay.setContentsMargins(22, 12, 24, 12)
        lay.setSpacing(16)
        logo = QLabel()
        logo.setStyleSheet("background: transparent;")
        icon = getattr(self._mw, "app_icon", None) or QApplication.windowIcon()
        logo.setPixmap(icon.pixmap(QSize(54, 54)))
        lay.addWidget(logo)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        col.addWidget(_label(_t("Welcome back") if self._asking else _t("Seti Astro Suite Pro"),
                             "sc_title", wrap=False))
        col.addWidget(_label(_t("SASpro has a new look. Your images, presets and tool settings "
                                "are as you left them.") if self._asking else
                             _t("Welcome and setup"), "sc_sub", wrap=False))
        lay.addLayout(col, 1)
        v = str(getattr(self._mw, "_version", "") or "")
        if v:
            lay.addWidget(_label(f"v{v}", "sc_version", wrap=False))
        return head

    def _build_rail(self) -> QWidget:
        rail = QWidget()
        rail.setObjectName("sc_rail")
        rail.setFixedWidth(238)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(4)
        titles = {"welcome": _t("Welcome"), "layout": _t("Your layout"),
                  "accel": _t("Hardware acceleration"), "models": _t("AI models"),
                  "gaia": _t("Gaia catalogs"), "ready": _t("Ready")}
        self._rows = {}
        for i, k in enumerate(self._keys):
            row = _RailRow(titles[k], i + 1)
            row.clicked.connect(lambda key=k: self._go(key))
            self._rows[k] = row
            lay.addWidget(row)
        lay.addStretch(1)
        lay.addWidget(self._build_language())
        return rail

    def _build_language(self) -> QWidget:
        """🌐 and every language in its own name. The welcome opens on the
        system's language when SASpro has it, so it is seen; nothing changes
        until the restart button (in that language) is pressed."""
        from setiastro.saspro import first_run_dialog
        from setiastro.saspro.i18n import get_available_languages, get_current_language
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 0, 4, 0)
        v.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(8)
        globe = QLabel("🌐")
        globe.setObjectName("sc_globe")
        self._lang = _Combo()
        self._lang.setObjectName("sc_lang")
        self._lang.setCursor(Qt.CursorShape.PointingHandCursor)
        for code, name in get_available_languages().items():
            self._lang.addItem(name, code)
        pick = get_current_language()
        if pick == "en" and first_run_dialog.welcome_due(self._mw):
            pick = system_language() or "en"
        self._lang.setCurrentIndex(max(0, self._lang.findData(pick)))
        self._lang.currentIndexChanged.connect(lambda _i: self._sync_language())
        row.addWidget(globe)
        row.addWidget(self._lang, 1)
        v.addLayout(row)
        self._lang_go = _button("", "sc_plain", self._switch_language)
        v.addWidget(self._lang_go)
        self._sync_language()
        return box

    def _sync_language(self):
        from setiastro.saspro.i18n import get_current_language
        code = self._lang.currentData()
        other = bool(code) and code != get_current_language()
        if other:
            self._lang_go.setText(in_language(code, "Setup", "Restart SASpro now"))
        self._lang_go.setVisible(other)

    def _switch_language(self):
        from setiastro.saspro.i18n import save_language
        code = self._lang.currentData()
        if not code:
            return
        save_language(code)
        if not self._can_restart():
            QMessageBox.information(
                self, in_language(code, "SettingsDialog", "Restart required"),
                in_language(code, "SettingsDialog", "Language changed. Please manually restart "
                                                    "the application to apply the new language."))
            return
        s = self._mw.settings
        s.setValue(REOPEN_KEY, self.current())
        s.sync()
        self._mw.restart_app()
        if not getattr(self._mw, "_shutting_down", False):     # the restart was called off
            s.remove(REOPEN_KEY)

    def _build_footer(self) -> QWidget:
        foot = QWidget()
        foot.setObjectName("sc_footer")
        lay = QHBoxLayout(foot)
        lay.setContentsMargins(22, 12, 22, 12)
        lay.setSpacing(10)
        self._later = _button(_t("Finish later"), "sc_link", self.close)
        self._later.setToolTip(_t("Close this window. About > Welcome and Setup opens it again."))
        self._back = _button(_t("Back"), "sc_plain", lambda: self._step(-1))
        self._next = _button(_t("Next"), "sc_primary", lambda: self._step(1))
        self._next.setAutoDefault(True)
        self._next.setDefault(True)
        lay.addWidget(self._later)
        lay.addStretch(1)
        lay.addWidget(self._back)
        lay.addWidget(self._next)
        return foot

    # ---- moving about ----------------------------------------------------
    def current(self) -> str:
        return self._keys[self._stack.currentIndex()]

    def go(self, key: str):
        if key in self._keys:
            self._go(key)

    def _go(self, key: str):
        self._stack.setCurrentIndex(self._keys.index(key))
        for k, row in self._rows.items():
            row.set_current(k == key)
        if key == "gaia":
            self._saw_gaia = True
        self._sync()

    def _step(self, by: int):
        i = self._stack.currentIndex() + by
        if i >= len(self._keys):
            self.close()
        elif i >= 0:
            self._go(self._keys[i])

    def _done(self, key: str) -> bool:
        item = self._mgr.items[key]
        return item.state in ("ok", "warn") or (key == "accel" and self._mgr.restart_needed)

    def _sync(self):
        """Everything that depends on the manager or the current page."""
        mgr, cur = self._mgr, self.current()
        subs = {"accel": _t("About 2.5 GB"), "models": _t("About 2.8 GB"),
                "gaia": _t("Optional")}
        for k, row in self._rows.items():
            if k in STEPS:
                item = mgr.items[k]
                if mgr.is_running(k):
                    row.dot.set_state("busy")
                    job = mgr.jobs[k]
                    row.set_sub(_t("Installing") + (f", {job.percent:.0f}%" if job.percent is not None else ""))
                elif k == "accel" and mgr.restart_needed:
                    row.dot.set_state("ok")
                    row.set_sub(_t("Installed, restart to use it"))
                else:
                    row.dot.set_state(item.state)
                    row.set_sub(subs[k] if item.state == "todo" else item.headline)
            elif k == "ready":
                row.dot.set_state("ok" if mgr.checked() and not mgr.remaining() and not mgr.jobs else "todo")
            else:
                row.dot.set_state("ok" if self._keys.index(k) < self._stack.currentIndex() else "todo")
        self._sync_accel()
        self._sync_models()
        self._sync_gaia()
        self._sync_ready()
        self._back.setVisible(self._stack.currentIndex() > 0)
        last = cur == self._keys[-1]
        self._later.setVisible(not last)
        if last:
            text = _t("Start using SASpro")
        elif cur == "welcome":
            text = _t("Get started")
        elif cur in STEPS and not self._done(cur) and not mgr.is_running(cur):
            text = _t("Skip for now")
        else:
            text = _t("Next")
        self._next.setText(text)
        # One bright button at a time: the step's own while it needs doing.
        look = "sc_plain" if text == _t("Skip for now") else "sc_primary"
        if self._next.objectName() != look:
            self._next.setObjectName(look)
            _restyle(self._next)

    # ---- pages -----------------------------------------------------------
    def _page_welcome(self, lay):
        lay.addWidget(_label(_t("Thank you for choosing Seti Astro Suite Pro"), "sc_h1"))
        lay.addWidget(_label(_t(
            "SASpro is a massive passion project — thousands of hours of development poured into "
            "building something I'm genuinely proud of: a serious, full-featured "
            "astrophotography processing and scientific analysis platform that anyone can use.\n\n"
            "This project exists because of a genuine love for deep-sky imaging and a belief that "
            "powerful tools should be accessible to everyone. Whether you're imaging "
            "from a light-polluted backyard or a dark-sky site, chasing faint nebulae or transiting "
            "exoplanets, SASpro was built to be there with you.\n\n"
            "Thank you for being part of this journey. Clear skies."), "sc_lead"))
        sign = _label(_t("— Franklin Marek (Seti Astro)"), "sc_sign")
        sign.setAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(sign)
        box = QFrame()
        box.setObjectName("sc_panel")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(6)
        bl.addWidget(_label(_t("Three steps get everything working"), "sc_state"))
        bl.addWidget(_label(_t("Each one shows whether it is already done. The downloads run in "
                               "the background, and you can skip any step and come back to it."),
                            "sc_detail"))
        lay.addWidget(box)

    def _page_layout(self, lay):
        s = self._mw.settings
        lay.addWidget(_label(_t("Which layout would you like?"), "sc_h1"))
        lay.addWidget(_label(_t(
            "SASpro now starts with a simpler layout. The one you had before was found and kept. "
            "Click either to see it behind this window."), "sc_lead"))
        facts = dict(workspace_layout.describe_saved_layout(s))
        theme = facts.get("theme", "gray")
        row = QHBoxLayout()
        row.setSpacing(16)
        self._card_new = _ChoiceCard(
            _t("New layout"),
            _t("Two rows of tools and an open workspace. Every panel is one click away on the edges."),
            _layout_picture("new", "gray", False))
        self._card_old = _ChoiceCard(
            _t("My previous layout"), saved_layout_text(s) + ".",
            _layout_picture("old", theme, "second_window" in facts))
        self._card_new.clicked.connect(lambda: self._choose_layout(False))
        self._card_old.clicked.connect(lambda: self._choose_layout(True))
        row.addWidget(self._card_new, 1)
        row.addWidget(self._card_old, 1)
        lay.addLayout(row)
        self._card_new.set_chosen(self._asking)
        self._card_old.set_chosen(False)
        lay.addWidget(_label(_t(
            "You can change your mind at any time under View > Layout."), "sc_small"))
        lay.addWidget(_label(_t("Worth knowing"), "sc_label"))
        tips = QVBoxLayout()
        tips.setSpacing(4)
        native = QKeySequence.SequenceFormat.NativeText
        find = QKeySequence("Ctrl+Shift+P").toString(native)
        docs = getattr(self._mw, "act_docs", None)
        docs = docs.shortcut().toString(native) if docs is not None else "F1"
        for text in (
            _t("Command Search finds any tool by name: press {0} and start typing.").format(find),
            _t("Functions and Tools are grouped by what they do; Stacking and Analysis have their own menus."),
            _t("Analysis now has Blink, Magnitude / Surface Brightness and Object SNR."),
            _t("{0} opens the documentation, with a search box.").format(docs),
        ):
            tips.addWidget(_label("•  " + text, "sc_small"))
        lay.addLayout(tips)

    def _choose_layout(self, old: bool):
        self._card_new.setEnabled(False)
        self._card_old.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        # Let the click show before the window behind is rebuilt.
        QTimer.singleShot(20, lambda: self._apply_layout(old))

    def _apply_layout(self, old: bool):
        try:
            if old:
                self._mw._use_saved_layout()
            else:
                self._mw._use_new_layout()
        finally:
            QApplication.restoreOverrideCursor()
            self._card_new.setEnabled(True)
            self._card_old.setEnabled(True)
            self._card_new.set_chosen(not old)
            self._card_old.set_chosen(old)
            self.raise_()
            self.activateWindow()
            self._sync()

    def _page_accel(self, lay):
        lay.addWidget(_label(_t("Hardware acceleration"), "sc_h1"))
        lay.addWidget(_label(_t(
            "Cosmic Clarity, star removal, denoising, drizzle and much of stacking run through "
            "this package, on your graphics card when you have one. Without it they are slow, "
            "and some won't run at all."), "sc_lead"))
        self._accel = _StatePanel()
        self._accel_btn = _button(_t("Install"), "sc_primary", self._mgr.install_accel)
        self._accel_py = _button(_t("Get Python"), "sc_primary", self._get_python)
        self._accel_again = _button(_t("Check again"), "sc_plain", self._mgr.refresh)
        self._accel_restart = _button(_t("Restart SASpro now"), "sc_primary", self._restart)
        self._accel.add(self._accel_py, self._accel_btn, self._accel_restart, self._accel_again)
        lay.addWidget(self._accel)
        self._accel_note = _label("", "sc_small")
        lay.addWidget(self._accel_note)

    def _sync_accel(self):
        mgr, item = self._mgr, self._mgr.items["accel"]
        if mgr.is_running("accel"):
            self._accel.show_job(_t("Installing hardware acceleration"), mgr.jobs["accel"])
            self._accel_note.setText(_t(
                "It can look stalled while the packages download and unpack; that is normal. "
                "Leave SASpro open until it finishes. You can carry on with the next step meanwhile."))
        elif mgr.restart_needed:
            self._accel.show_item(type(item)("ok", _t("Installed"),
                                             _t("Restart SASpro to start using it.")))
            self._accel_note.setText("")
        else:
            self._accel.show_item(item)
            err = self._last_error.get("accel")
            if err and item.state != "ok":
                self._accel_note.setText(_t("The last install failed: {0}").format(err))
            elif item.state == "todo":
                self._accel_note.setText(_t(
                    "About 2.5 GB to download. You can carry on with the next step while it runs."))
            elif item.state == "warn" and item.extra.get("gpu"):
                self._accel_note.setText(_t(
                    "Installing again often fixes this. If it doesn't, update your graphics "
                    "driver first."))
            else:
                self._accel_note.setText("")
        running, restart = mgr.is_running("accel"), mgr.restart_needed
        blocked = item.state == "blocked" and not running and not restart
        self._accel_py.setVisible(blocked)
        self._accel_py.setText(_t("Get Python {0}").format(item.extra.get("need", "")))
        self._accel_again.setVisible(blocked)
        # A supported Python and something to do: install, or repair.
        fix = item.state in ("todo", "warn", "ok") and "backend" in item.extra
        fix = fix and "have" not in item.extra
        self._accel_btn.setVisible(fix and not running and not restart)
        main = item.state == "todo" or (item.state == "warn" and bool(item.extra.get("gpu")))
        self._accel_btn.setText(_t("Install") if item.state == "todo" else _t("Repair"))
        if self._accel_btn.objectName() != ("sc_primary" if main else "sc_plain"):
            self._accel_btn.setObjectName("sc_primary" if main else "sc_plain")
            _restyle(self._accel_btn)
        self._accel_restart.setVisible(restart and not running and self._can_restart())

    def _get_python(self):
        import sys
        item = self._mgr.items["accel"]
        need, url = item.extra.get("need", ""), item.extra.get("url", "")
        if sys.platform.startswith("win") or not need:
            QDesktopServices.openUrl(QUrl(url or "https://www.python.org/downloads/"))
            return
        how = (_t("In Terminal:\n\n    brew install python@{0}\n\nor download it from python.org.")
               if sys.platform == "darwin" else
               _t("Install python{0} from your distribution's package manager."))
        QMessageBox.information(self._mw, _t("Python {0}").format(need),
                                how.format(need) + "\n\n" + _t("Then choose Check again."))

    def _page_models(self, lay):
        lay.addWidget(_label(_t("Cosmic Clarity AI models"), "sc_h1"))
        lay.addWidget(_label(_t(
            "Sharpening, denoising, aberration correction, Dark Star, super-resolution and "
            "satellite trail removal use these models. They are downloaded separately from SASpro."),
            "sc_lead"))
        self._models = _StatePanel()
        self._models_btn = _button(_t("Download"), "sc_primary", self._mgr.install_models)
        self._models_zip = _button(_t("Install from ZIP…"), "sc_plain", self._pick_zip)
        self._models_stop = _button(_t("Stop"), "sc_plain", self._mgr.cancel_models)
        self._models.add(self._models_btn, self._models_zip, self._models_stop)
        lay.addWidget(self._models)
        self._models_note = _label("", "sc_small")
        lay.addWidget(self._models_note)

    def _sync_models(self):
        mgr, item = self._mgr, self._mgr.items["models"]
        running = mgr.is_running("models")
        if running:
            self._models.show_job(_t("Installing the models"), mgr.jobs["models"])
        else:
            self._models.show_item(item)
        err = self._last_error.get("models")
        if err and not running and item.state != "ok":
            self._models_note.setText(_t("The last download failed: {0}").format(err))
        elif item.state in ("todo", "warn") and not running:
            self._models_note.setText(_t(
                "About 2.8 GB to download. You can carry on while it runs. Install from ZIP "
                "takes a models file you already downloaded."))
        else:
            self._models_note.setText("")
        main = item.state in ("todo", "warn")
        self._models_btn.setVisible(not running and item.state != "checking")
        self._models_btn.setText(_t("Download") if item.state == "todo" else
                                 _t("Update") if item.state == "warn" else _t("Download again"))
        if self._models_btn.objectName() != ("sc_primary" if main else "sc_plain"):
            self._models_btn.setObjectName("sc_primary" if main else "sc_plain")
            _restyle(self._models_btn)
        self._models_zip.setVisible(not running)
        self._models_stop.setVisible(running)

    def _pick_zip(self):
        path, _f = QFileDialog.getOpenFileName(self._mw, _t("Select models ZIP"), "",
                                               "ZIP (*.zip)")
        if path and os.path.exists(path):
            self._last_error.pop("models", None)
            self._mgr.install_models_zip(path)

    def _page_gaia(self, lay):
        lay.addWidget(_label(_t("Gaia star catalogs"), "sc_h1"))
        lay.addWidget(_label(_t(
            "Optional. With Gaia catalogs on your own disk, SASpro calibrates color and "
            "plate-solves without going online. They are large, so you choose which sets to "
            "install in the Gaia manager."), "sc_lead"))
        self._gaia = _StatePanel()
        self._gaia_btn = _button(_t("Open the Gaia manager"), "sc_primary", self._open_gaia)
        self._gaia.add(self._gaia_btn)
        lay.addWidget(self._gaia)
        self._gaia_lines = {}
        for key, title, body in (
            ("spectral", _t("Spectrum Library"),
             _t("For SFCC color calibration, magnitudes and narrowband extraction. "
                "The recommended sets are about 11 GB.")),
            ("astro", _t("Astrometry"),
             _t("For plate solving, and for blind solving once its index is built there. "
                "The recommended sets are about 35 GB.")),
        ):
            box = QFrame()
            box.setObjectName("sc_panel")
            bl = QVBoxLayout(box)
            bl.setContentsMargins(18, 12, 18, 12)
            bl.setSpacing(3)
            head = QHBoxLayout()
            head.addWidget(_label(title, "sc_row_title", wrap=False))
            head.addStretch(1)
            n = _label("", "sc_detail", wrap=False)
            head.addWidget(n)
            bl.addLayout(head)
            bl.addWidget(_label(body, "sc_detail"))
            self._gaia_lines[key] = n
            lay.addWidget(box)

    def _sync_gaia(self):
        item = self._mgr.items["gaia"]
        self._gaia.show_item(item)
        for key, lb in self._gaia_lines.items():
            n = item.extra.get(key)
            lb.setText(_t("{0} of {1} sets").format(*n) if n else "")

    def _open_gaia(self):
        self._mgr.gaia_seen()
        self._mw._open_gaia_database()
        win = getattr(self._mw, "_gaia_db_window", None)
        if win is not None:
            try:
                win.library_changed.connect(self._mgr.refresh)
                win.raise_()
                win.activateWindow()
            except Exception:
                pass

    def _page_ready(self, lay):
        self._ready_h1 = _label("", "sc_h1")
        lay.addWidget(self._ready_h1)
        box = QFrame()
        box.setObjectName("sc_panel")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)
        self._ready_rows = {}
        for k, title in (("accel", _t("Hardware acceleration")), ("models", _t("AI models")),
                         ("gaia", _t("Gaia catalogs"))):
            row = QHBoxLayout()
            row.setSpacing(12)
            dot = _Dot(22)
            text = _label(title, "sc_row_title", wrap=False)
            state = _label("", "sc_detail", wrap=False)
            fix = _button(_t("Set up"), "sc_link", lambda key=k: self._go(key))
            row.addWidget(dot)
            row.addWidget(text)
            row.addStretch(1)
            row.addWidget(state)
            row.addWidget(fix)
            bl.addLayout(row)
            self._ready_rows[k] = (dot, state, fix)
        self._ready_restart = _button(_t("Restart SASpro now"), "sc_plain", self._restart)
        bl.addWidget(self._ready_restart, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(box)

        lay.addWidget(_label(_t("Theme"), "sc_label"))
        row = QHBoxLayout()
        row.setSpacing(10)
        self._theme_btns = {}
        for key, text in (("gray", _t("Gray")), ("dark", _t("Dark")), ("light", _t("Light")),
                          ("system", _t("System"))):
            b = _button(text, "sc_plain", lambda k=key: self._set_theme(k))
            self._theme_btns[key] = b
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addWidget(_label(_t("Find your way around"), "sc_label"))
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(_button(_t("First Steps tour"), "sc_plain", self._tour))
        row.addWidget(_button(_t("Beginner video"), "sc_plain",
                              lambda: QDesktopServices.openUrl(QUrl(TUTORIAL_URL))))
        row.addWidget(_button(_t("Documentation"), "sc_plain", self._docs))
        row.addWidget(_button(_t("Discord"), "sc_plain",
                              lambda: QDesktopServices.openUrl(QUrl(DISCORD_URL))))
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(_label(_t(
            "Using GraXpert, ASTAP, StarNet++ or Astrometry.net? Give SASpro their paths, or your "
            "Astrometry.net key, in Preferences."), "sc_small"))
        lay.addWidget(_button(_t("Open Preferences"), "sc_link", self._prefs), 0,
                      Qt.AlignmentFlag.AlignLeft)

    def _sync_ready(self):
        mgr = self._mgr
        left = [k for k in ("accel", "models") if not self._done(k) and not mgr.is_running(k)]
        if not mgr.checked():
            head = _t("Checking")
        elif mgr.jobs:
            head = _t("Almost there")
        elif left:
            head = _t("You can start now")
        else:
            head = _t("You're ready")
        self._ready_h1.setText(head)
        for k, (dot, state, fix) in self._ready_rows.items():
            item = mgr.items[k]
            if mgr.is_running(k):
                job = mgr.jobs[k]
                dot.set_state("busy")
                state.setText(_t("Installing") + (f", {job.percent:.0f}%" if job.percent is not None else ""))
                fix.hide()
            elif k == "accel" and mgr.restart_needed:
                dot.set_state("ok")
                state.setText(_t("Installed, restart to use it"))
                fix.hide()
            else:
                dot.set_state(item.state)
                state.setText(_t("Optional") if k == "gaia" and item.state == "todo" else item.headline)
                fix.setVisible(item.needs_doing)
        self._ready_restart.setVisible(mgr.restart_needed and not mgr.jobs and self._can_restart())
        theme = (self._mw.settings.value("ui/theme", "system", type=str) or "system").lower()
        for k, b in self._theme_btns.items():
            want = "true" if k == theme else "false"
            if b.property("chosen") != want:
                b.setProperty("chosen", want)
                _restyle(b)

    # ---- actions ---------------------------------------------------------
    def _set_theme(self, key):
        self._mw.settings.setValue("ui/theme", key)
        self._mw._apply_theme_safely()
        self._sync()

    def _tour(self):
        mw = self._mw
        self.close()
        from setiastro.saspro.first_steps_tour import start_first_steps_tour
        start_first_steps_tour(mw)

    def _docs(self):
        act = getattr(self._mw, "act_docs", None)
        if act is not None:
            act.trigger()

    def _prefs(self):
        self._mw._open_settings()
        self._mgr.refresh()

    def _can_restart(self) -> bool:
        from setiastro.saspro.settings_sandbox import active
        return hasattr(self._mw, "restart_app") and not active()

    def _restart(self):
        self._mw.restart_app()

    def _on_job_finished(self, name, ok, msg):
        if ok or not msg:                        # no message: stopped on request
            self._last_error.pop(name, None)
        else:
            self._last_error[name] = msg
        self._sync()

    # ---- window ----------------------------------------------------------
    def event(self, ev):
        # Coming back from the Gaia manager, a browser or a Python installer.
        if ev.type() == QEvent.Type.WindowActivate and getattr(self, "_mgr", None) is not None:
            if not self._mgr.jobs:
                self._mgr.refresh()
        return super().event(ev)

    def done(self, result):                      # every way of closing ends here
        super().done(result)
        self._closed()

    def _closed(self):
        if getattr(self, "_is_closed", False):
            return
        self._is_closed = True
        for sig, slot in ((self._mgr.changed, self._sync),
                          (self._mgr.finished, self._on_job_finished)):
            try:
                sig.disconnect(slot)
            except Exception:
                pass
        if self._saw_gaia:
            self._mw.settings.setValue(GAIA_SEEN_KEY, True)     # no more reminders of it
        _after_setup_closed(self._mw)


# ── opening it ──────────────────────────────────────────────────────────────

def _alive(w) -> bool:
    try:
        from PyQt6 import sip
        return w is not None and not sip.isdeleted(w)
    except Exception:
        return w is not None


def open_setup_center(main_window, start: str | None = None) -> SetupCenter:
    """Show the setup window, on `start` ("accel", "models", "gaia", ...)."""
    dlg = getattr(main_window, "_setup_center", None)
    if _alive(dlg) and dlg.isVisible():
        if start:
            dlg.go(start)
    else:
        dlg = SetupCenter(main_window, start)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        main_window._setup_center = dlg
        dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    pill = getattr(main_window, "_setup_pill", None)
    if _alive(pill):
        pill.sync()
    return dlg


def _after_setup_closed(main_window):
    """Closed, however: the welcome has been seen and the layout question is
    answered by whatever is on screen. Then the tour, for a new install."""
    main_window._setup_center = None
    if main_window.settings.value(REOPEN_KEY, "", type=str):
        return          # restarting in another language: it opens again as it was, in that one
    from setiastro.saspro import first_run_dialog
    was_welcome = first_run_dialog.welcome_due(main_window)
    first_run_dialog.mark_welcome_shown(main_window)
    workspace_layout.choice_made(main_window.settings)
    if was_welcome and not getattr(main_window, "_shutting_down", False):
        try:
            from setiastro.saspro.first_steps_tour import maybe_start_first_steps_tour
            maybe_start_first_steps_tour(main_window)
        except Exception as e:
            print(f"[first-steps] tour failed to start: {e}")
    show_reminder(main_window)


def find_main_window(widget=None):
    """The SASpro main window above `widget`, or any open one."""
    w = widget
    while w is not None:
        if hasattr(w, "docman") or hasattr(w, "doc_manager"):
            return w
        w = w.parent() if hasattr(w, "parent") else None
    for w in QApplication.topLevelWidgets():
        if hasattr(w, "_open_setup_center"):
            return w
    return None


def offer_setup(parent, step: str, title: str, text: str, detail: str = "") -> bool:
    """A tool needs something that isn't installed: say so, with a button that
    opens the setup window on the right step. True if it was opened."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Information)
    box.setWindowTitle(title)
    box.setText(text)
    if detail:
        box.setInformativeText(detail)
    main_window = find_main_window(parent)
    go = None
    if main_window is not None and hasattr(main_window, "_open_setup_center"):
        go = box.addButton(_t("Set it up now"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(_t("Not now"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(go)
    else:
        box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if go is None or box.clickedButton() is not go:
        return False
    QTimer.singleShot(0, lambda: main_window._open_setup_center(step))
    return True


# ── the reminder ────────────────────────────────────────────────────────────

_PILL_STYLE = """
#SetupPill { background-color: #1a1a2e; border: 1px solid #3a3a6a; border-radius: 10px; }
#sp_text { color: #ffffff; font-size: 13px; font-weight: 600; background: transparent; }
#sp_sub { color: #aaaacc; font-size: 12px; background: transparent; }
QPushButton#sp_go { color: #ffffff; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3344bb, stop:1 #5566dd); }
QPushButton#sp_go:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4455cc, stop:1 #6677ee); }
QPushButton#sp_x { background: transparent; border: none; color: #8888bb; font-size: 11px; padding: 2px 4px; }
QPushButton#sp_x:hover { color: #ffffff; }
QProgressBar { background-color: #0f0f1b; border: 1px solid #2a2a4a; border-radius: 4px; max-height: 6px; min-height: 6px; }
QProgressBar::chunk { background-color: #6f86ff; border-radius: 3px; }
"""


class SetupPill(QFrame):
    """Bottom right of the workspace: what is left to set up, or what is
    installing. Hidden while the setup window or the First Steps card is up."""

    def __init__(self, main_window):
        super().__init__(main_window)
        self.setObjectName("SetupPill")
        self.setStyleSheet(_PILL_STYLE)
        self.setFixedWidth(330)
        self._mw = main_window
        self._mgr = setup_manager(main_window)
        self._showing = ""                       # "busy", "restart" or "todo"
        self._closed = set()                     # what the user closed, this session
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 11, 8, 11)
        lay.setSpacing(12)
        self._dot = _Dot(26)
        lay.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(6)
        self._text = _label("", "sp_text")
        self._sub = _label("", "sp_sub")
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        col.addWidget(self._text)
        col.addWidget(self._sub)
        col.addWidget(self._bar)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._go = _button(_t("Continue"), "sp_go", lambda: self._mw._open_setup_center(self._target()))
        self._off = _button(_t("Don't remind me"), "sp_x", self._never)
        row.addWidget(self._go)
        row.addWidget(self._off)
        row.addStretch(1)
        col.addLayout(row)
        lay.addLayout(col, 1)
        x = _button("✕", "sp_x", self._not_now)
        x.setToolTip(_t("Hide until SASpro is started again"))
        lay.addWidget(x, 0, Qt.AlignmentFlag.AlignTop)
        main_window.installEventFilter(self)
        self._mgr.changed.connect(self.sync)
        self._watch = QTimer(self)               # the tour card comes and goes
        self._watch.setInterval(1500)
        self._watch.timeout.connect(self.sync)
        self._watch.start()
        self.hide()
        self.sync()

    def _target(self) -> str:
        busy = next((k for k in STEPS if self._mgr.is_running(k)), None)
        if busy:
            return busy
        left = self._mgr.remaining()
        return left[0] if left else ("ready" if self._mgr.restart_needed else "welcome")

    def sync(self):
        mgr = self._mgr
        dlg = getattr(self._mw, "_setup_center", None)
        tour = getattr(self._mw, "_first_steps_tour", None)
        if (_alive(dlg) and dlg.isVisible()) or (_alive(tour) and tour.isVisible()):
            self.hide()
            return
        names = {"accel": _t("hardware acceleration"), "models": _t("AI models"),
                 "gaia": _t("Gaia catalogs")}
        if not mgr.jobs:
            self._closed.discard("busy")         # closed for that install only
        self._showing = "busy" if mgr.jobs else "restart" if mgr.restart_needed else "todo"
        if self._showing in self._closed:
            self.hide()
            return
        if mgr.jobs:
            job_name = next(iter(mgr.jobs))
            job = mgr.jobs[job_name]
            self._dot.set_state("busy")
            self._text.setText(_t("Installing {0}").format(
                _t(" and ").join(names[k] for k in mgr.jobs)))
            self._sub.setText(_t("{0} elapsed. Leave SASpro open.").format(job.elapsed()))
            pct = [j.percent for j in mgr.jobs.values() if j.percent is not None]
            self._bar.setRange(0, 1000 if len(pct) == len(mgr.jobs) else 0)
            if pct:
                self._bar.setValue(int(min(pct) * 10))
            self._bar.show()
            self._go.setText(_t("Show"))
            self._off.hide()
        elif mgr.restart_needed:
            self._dot.set_state("ok")
            self._text.setText(_t("Hardware acceleration is installed"))
            self._sub.setText(_t("Restart SASpro to start using it."))
            self._bar.hide()
            self._go.setText(_t("Show"))
            self._off.hide()
        else:
            left = mgr.remaining()
            off = self._mw.settings.value(REMINDER_OFF_KEY, False, type=bool)
            if not mgr.checked() or not left or off:
                self.hide()
                return
            self._dot.set_state("todo")
            self._text.setText(_t("Finish setting up SASpro"))
            self._sub.setText(_t("Still to do: {0}").format(", ".join(names[k] for k in left)))
            self._bar.hide()
            self._go.setText(_t("Continue"))
            self._off.show()
        self.adjustSize()
        self._place()
        if not self.isVisible():
            self.show()
        self.raise_()

    def _not_now(self):
        self._closed.add(self._showing)
        self.hide()

    def _never(self):
        self._mw.settings.setValue(REMINDER_OFF_KEY, True)
        self.hide()

    def eventFilter(self, obj, ev):
        if obj is self._mw and ev.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            self._place()
        return super().eventFilter(obj, ev)

    def _place(self):
        mw = self._mw
        try:
            vp = mw.mdi.viewport()
            tl = vp.mapTo(mw, QPoint(0, 0))
            right, bottom = tl.x() + vp.width(), tl.y() + vp.height()
        except Exception:
            right, bottom = mw.width(), mw.height()
        self.move(max(0, right - self.width() - 20), max(0, bottom - self.height() - 20))


def show_reminder(main_window):
    """The reminder, if anything is left to do. Safe to call more than once."""
    if getattr(main_window, "_shutting_down", False):
        return
    if main_window.settings.value(REOPEN_KEY, "", type=str):
        open_setup_center(main_window)          # back from a language restart
        return
    pill = getattr(main_window, "_setup_pill", None)
    if not _alive(pill):
        main_window._setup_pill = SetupPill(main_window)
    else:
        pill.sync()
    mgr = setup_manager(main_window)
    if not mgr.checked():
        mgr.refresh()
