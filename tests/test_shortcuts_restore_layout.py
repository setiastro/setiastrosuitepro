"""Desktop shortcuts come back where the user left them after a restart.

Reported 2026-11-20 (thedwo): shortcuts arranged in two neat columns on the
right came back scattered every time SASpro was reopened. A chip's rectangle is
as wide as its label, so chips the user lines up by eye often overlap; the load
nudged every overlapping chip to a "free" spot, and load_shortcuts() runs
before the main window is sized, so the only free spots on offer were inside
Qt's default 640x480 viewport. A chip saved at x=1785 came back at x=561, and
that spot became its remembered placement.

Uses SASpro's real MdiArea, CanvasController and ShortcutManager, built in the
order main_window builds them.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication, QPoint, QSettings  # noqa: E402
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from setiastro.saspro import shortcuts as sc  # noqa: E402
from setiastro.saspro.mdi_canvas import CanvasController  # noqa: E402
from setiastro.saspro.mdi_widgets import MdiArea  # noqa: E402

LABELS = ["Statistical Stretch", "Background Neutralization", "Remove Green", "Curves",
          "ABE", "GraXpert", "Cosmic Clarity", "Star Removal", "Pixel Math",
          "Plate Solve", "HDR Multiscale", "Clahe"]

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-shortcut-layout-")
    QCoreApplication.setOrganizationName("SASproTestShortcutLayout")
    QCoreApplication.setApplicationName("ShortcutLayout")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _settle(n=30):
    for _ in range(n):
        _APP.processEvents()


class ShortcutLayoutRestartTests(unittest.TestCase):

    def setUp(self):
        QSettings().clear()
        self.windows = []

    def tearDown(self):
        for mw in self.windows:
            mw.hide()
            mw.deleteLater()
        _settle()

    def build(self):
        """Main window pieces in main_window's order; nothing shown yet."""
        mw = QMainWindow()
        mw._log = lambda *_: None
        mw.mdi = MdiArea()
        mw.setCentralWidget(mw.mdi)
        mw.canvas = CanvasController(mw.mdi, mw)
        mw.mdi.attach_canvas(mw.canvas)
        mw.canvas.set_layer_provider(
            lambda: getattr(getattr(mw, "shortcuts", None), "canvas", None))
        mw.shortcuts = sc.ShortcutManager(mw.mdi, mw)
        pm = QPixmap(32, 32)
        pm.fill(QColor("#4a7"))
        for i, lab in enumerate(LABELS):
            mw.shortcuts.register_action(f"cmd{i:02d}", QAction(QIcon(pm), lab, mw))
        self.windows.append(mw)
        return mw

    def arrange_two_columns(self):
        """Session 1: two columns on the right, pitched closer than the widest
        label, so neighbouring chip rectangles overlap -- as by-eye placement
        does. Returns the saved {sid: (x, y)}."""
        mw = self.build()
        mw.resize(1920, 1040)
        mw.show()
        _settle()
        sm = mw.shortcuts
        sids = [sm.add_shortcut(f"cmd{i:02d}", QPoint(40 * i, 30 * i)) for i in range(len(LABELS))]
        _settle()
        x2 = sm.canvas.rect().width() - 140
        for col, x in ((sids[:6], x2 - 80), (sids[6:], x2)):
            y = 10
            for s in col:
                sm._move_widget_clamped(sm.widgets[s], x, y)
                y += sm.widgets[s].height() - 6
        sm.save_shortcuts()
        widest = max(sm.widgets[s].width() for s in sids)
        self.assertGreater(widest, 80, "labels too short for the columns to overlap")
        saved = json.loads(QSettings().value(sc.SET_KEY_V2, "", type=str))
        return {d["id"]: (d["x"], d["y"]) for d in saved}

    def test_restart_restores_overlapping_columns_exactly(self):
        saved = self.arrange_two_columns()
        mw = self.build()
        sm = mw.shortcuts
        sm.load_shortcuts()                 # before the window is sized, as at startup
        mw.resize(1920, 1040)
        mw.show()
        _settle(60)
        for sid, xy in saved.items():
            w = sm.widgets[sid]
            self.assertEqual((w.x(), w.y()), xy, f"{w.text()} moved on restart")
            self.assertEqual((w._desired_pos.x(), w._desired_pos.y()), xy,
                             f"{w.text()} remembers a different placement")

    def test_new_chip_on_an_occupied_spot_is_still_nudged(self):
        self.arrange_two_columns()
        mw = self.build()
        sm = mw.shortcuts
        sm.load_shortcuts()
        mw.resize(1920, 1040)
        mw.show()
        _settle()
        first = next(iter(sm.widgets.values()))
        nw = sm.widgets[sm.add_shortcut("cmd03", QPoint(first.x(), first.y()))]
        self.assertFalse(nw.geometry().intersects(first.geometry()))


if __name__ == "__main__":
    unittest.main()
