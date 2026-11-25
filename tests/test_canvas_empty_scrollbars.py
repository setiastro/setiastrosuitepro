"""No workspace scrollbars when nothing is outside the view.

Reported 2026-11-25 against 2.0.0 (Ricardo Raimundo): scrollbars on an empty
workspace, nothing open and nothing out of view. Two ways in, both in
mdi_canvas.CanvasController, both because the canvas kept the viewport's size
as if content had claimed it:

* Startup. The canvas takes its size from the viewport before the window is
  laid out (Qt's default 640x480), and the first refresh measured the desktop
  shortcuts against that: every chip right of x=640 was "past the edge", so
  the canvas grew by GROW_MARGIN and never shrank.
* Any viewport shrink -- un-maximizing, a panel opening -- left the canvas at
  the old size. Re-maximizing did not clear it: each scrollbar's 12 px then
  kept the other one up.

Growth that content claimed must still stay (the canvas never shrinks on its
own; Tile and Cascade reset it), so that is pinned here too.

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

from PyQt6.QtCore import QCoreApplication, QSettings  # noqa: E402
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget  # noqa: E402

from setiastro.saspro import shortcuts as sc  # noqa: E402
from setiastro.saspro.mdi_canvas import CanvasController  # noqa: E402
from setiastro.saspro.mdi_widgets import MdiArea  # noqa: E402

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-canvas-scrollbars-")
    QCoreApplication.setOrganizationName("SASproTestCanvasScrollbars")
    QCoreApplication.setApplicationName("CanvasScrollbars")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _settle(n=40):
    for _ in range(n):
        _APP.processEvents()


class EmptyWorkspaceScrollbarTests(unittest.TestCase):

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
        for i, lab in enumerate(("BGE", "SRGC", "GHS", "Lighthouse", "HDR Blender")):
            mw.shortcuts.register_action(f"cmd{i}", QAction(QIcon(pm), lab, mw))
        self.windows.append(mw)
        return mw

    def bars(self, mw):
        return mw.mdi._hbar.isVisible(), mw.mdi._vbar.isVisible()

    def vp(self, mw):
        return mw.mdi.viewport().size()

    def resize(self, mw, w, h):
        mw.resize(w, h)
        _settle()

    def test_startup_with_chips_on_the_right_has_no_scrollbars(self):
        # The column of chips down the right edge from the report, inside
        # the window it is shown at but well right of 640.
        QSettings().setValue(sc.SET_KEY_V2, json.dumps([
            {"id": f"s{i}", "command_id": f"cmd{i}", "label": "", "x": 1700, "y": 10 + 60 * i}
            for i in range(5)]))
        mw = self.build()
        mw.shortcuts.load_shortcuts()       # before the window is sized, as at startup
        mw.resize(1920, 1040)
        mw.show()
        _settle(60)
        right = max(w.geometry().right() for w in mw.shortcuts.widgets.values())
        self.assertLess(right, self.vp(mw).width(), "probe chips must be inside the view")
        self.assertEqual(self.bars(mw), (False, False))
        self.assertEqual(mw.canvas.canvas_rect().size(), self.vp(mw))

    def test_empty_workspace_has_no_scrollbars_after_any_resize(self):
        mw = self.build()
        mw.resize(1920, 1040)
        mw.show()
        _settle()
        for w, h in ((1200, 800), (1920, 1040), (1500, 1000), (1920, 1040)):
            self.resize(mw, w, h)
            self.assertEqual(self.bars(mw), (False, False), f"at {w}x{h}")
            self.assertEqual(mw.canvas.canvas_rect().size(), self.vp(mw), f"at {w}x{h}")

    def test_shrinking_past_a_chip_shows_only_the_bar_that_reaches_it(self):
        QSettings().setValue(sc.SET_KEY_V2, json.dumps([
            {"id": "s0", "command_id": "cmd0", "label": "", "x": 1700, "y": 10}]))
        mw = self.build()
        mw.shortcuts.load_shortcuts()
        mw.resize(1920, 1040)
        mw.show()
        _settle(60)
        chip = mw.shortcuts.widgets["s0"]

        self.resize(mw, 1200, 800)
        self.assertEqual(self.bars(mw), (True, False),
                         "the chip is only out of view sideways")
        self.assertEqual(mw.canvas.canvas_rect().width(), chip.geometry().right() + 1,
                         "exactly far enough to reach the chip, no margin")
        self.assertEqual((chip.x(), chip.y()), (1700, 10), "the chip stays where it was put")

        self.resize(mw, 1920, 1040)
        self.assertEqual(self.bars(mw), (False, False))

    def _add_view(self, mw, x, y):
        sw = mw.mdi.addSubWindow(QWidget())
        sw.resize(400, 300)
        sw.show()
        _settle()
        mw.canvas._rewatch()
        sw.move(x, y)
        _settle()
        return sw

    def test_growth_from_a_dragged_view_still_stays_until_a_tidy(self):
        mw = self.build()
        mw.resize(1600, 1000)
        mw.show()
        _settle()
        vp = self.vp(mw)
        sw = self._add_view(mw, vp.width() - 100, 50)          # pushed past the right edge
        grown = mw.canvas.canvas_rect().width()
        self.assertGreaterEqual(grown, sw.geometry().right() + CanvasController.GROW_MARGIN - 1)
        self.assertTrue(self.bars(mw)[0])

        sw.move(50, 50)                                         # dragged back home
        _settle()
        self.assertEqual(mw.canvas.canvas_rect().width(), grown, "the canvas never shrinks on its own")
        self.assertTrue(self.bars(mw)[0])

        self.resize(mw, 1200, 800)                              # and survives a resize
        self.resize(mw, 1600, 1000)
        self.assertEqual(mw.canvas.canvas_rect().width(), grown)

        mw.canvas.reset_bounds()                                # Tile / Cascade
        _settle()
        self.assertEqual(self.bars(mw), (False, False))
        self.assertEqual(mw.canvas.canvas_rect().size(), self.vp(mw))

    def test_room_made_at_the_home_corner_stays(self):
        mw = self.build()
        mw.resize(1600, 1000)
        mw.show()
        _settle()
        self._add_view(mw, -300, 50)                            # dragged left of the corner
        self.assertEqual(mw.canvas.home_shift().x(), 300)
        self.assertEqual(mw.canvas.offset().x(), 300)
        self.resize(mw, 1200, 800)
        self.resize(mw, 1600, 1000)
        self.assertGreaterEqual(mw.canvas.canvas_rect().width(), self.vp(mw).width() + 300)
        self.assertEqual(mw.canvas.offset().x(), 300, "the view did not move under the user")


if __name__ == "__main__":
    unittest.main()
