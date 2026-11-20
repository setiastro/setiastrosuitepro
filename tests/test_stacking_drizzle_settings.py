"""Drizzle drop shrink: fresh installs get 0.65, and the stored 0.0 the old
startup migration wrote is repaired exactly once.

`QSettings.value(key, None, type=float)` returns 0.0 for a missing key in
PyQt6, never None. The getter and the startup migration both relied on the
None, so every fresh install read drop shrink 0.0 (point drops), showed it in
the spinbox, and wrote it back to all three keys.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyQt6.QtCore import QCoreApplication, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402

KEYS = ("stacking/drizzle_pixfrac", "stacking/drizzle_drop", "stacking/drop_shrink")
FLAG = "stacking/drop_shrink_zero_reset_done"

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep the dialog's QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-drizzle-settings-")
    QCoreApplication.setOrganizationName("SASproTestDrizzleSettings")
    QCoreApplication.setApplicationName("DrizzleSettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


class DropShrinkSettingsTests(unittest.TestCase):

    def setUp(self):
        self.dlg = None
        s = QSettings()
        s.clear()
        s.sync()

    def tearDown(self):
        if self.dlg is not None:
            self.dlg.close()
            self.dlg.deleteLater()
            _APP.processEvents()

    def seed(self, value, flag=None):
        s = QSettings()
        for k in KEYS:
            s.setValue(k, value)
        if flag is not None:
            s.setValue(FLAG, flag)
        s.sync()

    def open(self):
        self.dlg = stacking_suite.StackingSuiteDialog()
        return self.dlg

    def stored(self):
        s = QSettings()
        return [s.value(k, -1.0, type=float) for k in KEYS]

    def test_fresh_install_gets_the_default(self):
        dlg = self.open()
        self.assertEqual(dlg._get_drizzle_pixfrac(), 0.65)
        self.assertEqual(dlg.drizzle_drop_shrink_spin.value(), 0.65)
        self.assertEqual(self.stored(), [0.65] * 3)

    def test_stored_zero_is_reset_once(self):
        self.seed(0.0)
        dlg = self.open()
        self.assertEqual(dlg._get_drizzle_pixfrac(), 0.65)
        self.assertEqual(dlg.drizzle_drop_shrink_spin.value(), 0.65)
        self.assertEqual(self.stored(), [0.65] * 3)
        self.assertTrue(QSettings().value(FLAG, False, type=bool))

    def test_zero_chosen_after_the_reset_is_kept(self):
        self.seed(0.0, flag=True)
        dlg = self.open()
        self.assertEqual(dlg._get_drizzle_pixfrac(), 0.0)
        self.assertEqual(self.stored(), [0.0] * 3)

    def test_non_zero_value_is_left_alone(self):
        self.seed(0.8)
        dlg = self.open()
        self.assertAlmostEqual(dlg._get_drizzle_pixfrac(), 0.8)
        self.assertAlmostEqual(dlg.drizzle_drop_shrink_spin.value(), 0.8)
        self.assertTrue(QSettings().value(FLAG, False, type=bool))


if __name__ == "__main__":
    unittest.main()
