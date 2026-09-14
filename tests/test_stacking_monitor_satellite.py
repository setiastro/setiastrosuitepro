from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from setiastro.saspro.stacking_monitor import (  # noqa: E402
    StackingMonitorDialog,
    _classify,
)


class ClassifySatelliteTrailMessagesTests(unittest.TestCase):
    def test_group_calibration_and_close(self):
        self.assertEqual(
            _classify("📷 Calibrating group: S — 600s (9576x6388)"),
            ("Calibration", "S — 600s (9576x6388)", "running"),
        )
        self.assertEqual(
            _classify("📷 All light groups calibrated"),
            ("Calibration", "", "success"),
        )
        self.assertEqual(
            _classify("✅ Calibration Complete!"),
            ("Calibration", "", "success"),
        )

    def test_satellite_phase_messages(self):
        self.assertEqual(
            _classify("🛰️ Satellite trail removal — 1053 frame(s) on a clean GPU…"),
            ("Satellite Trails", "1053 frame(s)", "running"),
        )
        self.assertEqual(
            _classify("🛰️ 42/1053: NGC7822_S_001.fit"),
            ("Satellite Trails", "", "running"),
        )
        self.assertEqual(
            _classify("✅ Satellite trail removal complete"),
            ("Satellite Trails", "", "success"),
        )
        self.assertEqual(
            _classify("✅ Satellite trail removal complete (12 clipped, 1041 unchanged)"),
            ("Satellite Trails", "", "success"),
        )
        self.assertEqual(
            _classify("⏹ Satellite pass cancelled."),
            ("Satellite Trails", "", "warning"),
        )

    def test_carried_mask_is_not_satellite_progress(self):
        self.assertIsNone(_classify("🛰️ Carried satellite mask → foo.fit"))


class MonitorSatelliteRowSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dlg = StackingMonitorDialog()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()

    def _ops(self):
        return [(r.operation, r.group, r.status) for r in self.dlg._rows]

    def test_satellite_is_a_separate_row_after_last_filter(self):
        self.dlg._on_message("📷 Calibrating group: B — 60s (9576x6388)")
        self.dlg._on_message("📷 Calibrating group: S — 600s (9576x6388)")
        self.dlg._on_message("📷 All light groups calibrated")
        self.dlg._on_message(
            "🛰️ Satellite trail removal — 1053 frame(s) on a clean GPU…"
        )
        self.dlg._on_message("🛰️ 42/1053: NGC7822_S_001.fit")
        self.assertEqual(self.dlg._rows[-1].note, "42/1053: NGC7822_S_001.fit")
        self.dlg._on_message("✅ Satellite trail removal complete")
        self.dlg._on_message("✅ Calibration Complete!")

        self.assertEqual(
            self._ops(),
            [
                ("Calibration", "B — 60s (9576x6388)", "success"),
                ("Calibration", "S — 600s (9576x6388)", "success"),
                ("Satellite Trails", "1053 frame(s)", "success"),
            ],
        )
        self.assertEqual(self.dlg._rows[-1].note, "Satellite trail removal complete")
        self.assertIn("Calibrating group: S", self.dlg._rows[1].note)

    def test_calibration_complete_without_satellite_closes_last_group(self):
        self.dlg._on_message("📷 Calibrating group: S — 600s (9576x6388)")
        self.dlg._on_message("✅ Calibration Complete!")
        self.assertEqual(
            self._ops(),
            [("Calibration", "S — 600s (9576x6388)", "success")],
        )

    def test_satellite_start_closes_last_calibration_if_groups_complete_missing(self):
        self.dlg._on_message("📷 Calibrating group: S — 600s (9576x6388)")
        self.dlg._on_message(
            "🛰️ Satellite trail removal — 12 frame(s) on a clean GPU…"
        )

        self.assertEqual(
            self._ops(),
            [
                ("Calibration", "S — 600s (9576x6388)", "success"),
                ("Satellite Trails", "12 frame(s)", "running"),
            ],
        )


class MonitorStopButtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dlg = StackingMonitorDialog()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()

    def test_stop_is_disabled_when_idle(self):
        self.assertFalse(self.dlg.btn_stop.isEnabled())

    def test_stop_enabled_during_run_disabled_when_finished(self):
        self.dlg.start_run()
        self.assertTrue(self.dlg.btn_stop.isEnabled())
        self.dlg.finish_run(True)
        self.assertFalse(self.dlg.btn_stop.isEnabled())
        self.assertEqual(self.dlg.btn_stop.text(), "■ Stop")

    def test_stop_reenabled_if_a_later_phase_starts(self):
        self.dlg.start_run()
        self.dlg.finish_run(True)
        self.dlg._on_message("⚙️ MFDeconv engine")
        self.assertTrue(self.dlg.btn_stop.isEnabled())

    def test_finish_all_and_mark_stopped_disable_stop(self):
        self.dlg.start_run()
        self.dlg.finish_all(True)
        self.assertFalse(self.dlg.btn_stop.isEnabled())
        self.dlg.start_run()
        self.dlg.btn_stop.setEnabled(False)
        self.dlg.btn_stop.setText("Stopping…")
        self.dlg.mark_stopped()
        self.assertFalse(self.dlg.btn_stop.isEnabled())
        self.assertEqual(self.dlg.btn_stop.text(), "■ Stop")


if __name__ == "__main__":
    unittest.main()
