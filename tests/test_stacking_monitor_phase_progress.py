from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from setiastro.saspro.stacking_monitor import (  # noqa: E402
    StackingMonitorDialog,
    _classify,
)


class ClassifyPhaseProgressTests(unittest.TestCase):
    def test_measurements_progress(self):
        self.assertEqual(
            _classify("📦 Measured 273/429 frames"),
            ("Measurements", "", "running"),
        )

    def test_normalization_chunk_progress(self):
        self.assertEqual(
            _classify("🌀 Normalizing chunk 3/12 (32 frames)…"),
            ("Normalization", "", "running"),
        )

    def test_calibration_progress_not_suppressed(self):
        self.assertEqual(
            _classify("📷 Progress: G — 60s (9576x6388) — 40/120 frames"),
            ("Calibration", "", "running"),
        )

    def test_integration_tile_progress(self):
        self.assertEqual(
            _classify("🔧 Tile 42/200 [G — 60s] — 12.3 MPx/s"),
            ("Integration", "", "running"),
        )
        self.assertEqual(
            _classify("🔧 [LowRAM] Tile 7/50 [S — 600s] 256×256 — 8.1 MPx/s"),
            ("Integration", "", "running"),
        )

    def test_registration_align_progress(self):
        self.assertEqual(
            _classify("📐 Aligning stars… (42/400)"),
            ("Registration", "", "running"),
        )
        self.assertEqual(
            _classify("Aligning stars… (42/400)"),
            ("Registration", "", "running"),
        )


class MonitorPhaseProgressNoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dlg = StackingMonitorDialog()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()

    def test_measurements_note_updates_while_running(self):
        self.dlg._on_message("📏 Phase: Measurements starting…")
        self.dlg._on_message("📦 Measured 273/429 frames")
        self.assertEqual(self.dlg._rows[-1].operation, "Measurements")
        self.assertEqual(self.dlg._rows[-1].status, "running")
        self.assertEqual(self.dlg._rows[-1].note, "Measured 273/429 frames")

    def test_normalization_note_updates_while_running(self):
        self.dlg._on_message("📏 Phase: Normalization starting…")
        self.dlg._on_message("🌀 Normalizing chunk 3/12 (32 frames)…")
        self.assertEqual(self.dlg._rows[-1].operation, "Normalization")
        self.assertEqual(self.dlg._rows[-1].status, "running")
        self.assertIn("Normalizing chunk 3/12", self.dlg._rows[-1].note)

    def test_calibration_progress_updates_note(self):
        self.dlg._on_message("📷 Calibrating group: G — 60s (9576x6388)")
        self.dlg._on_message("📷 Progress: G — 60s (9576x6388) — 40/120 frames")
        self.assertEqual(self.dlg._rows[-1].operation, "Calibration")
        self.assertEqual(self.dlg._rows[-1].status, "running")
        self.assertIn("40/120 frames", self.dlg._rows[-1].note)

    def test_integration_tile_updates_note(self):
        self.dlg._on_message("📏 Phase: Integration starting…")
        self.dlg._on_message("📊 Stacking group 'G — 60s' with Winsorized Sigma [GPU]")
        self.dlg._on_message("🔧 Tile 42/200 [G — 60s] — 12.3 MPx/s")
        self.assertEqual(self.dlg._rows[-1].operation, "Integration")
        self.assertEqual(self.dlg._rows[-1].status, "running")
        self.assertIn("Tile 42/200", self.dlg._rows[-1].note)

    def test_registration_align_updates_note(self):
        self.dlg._on_message("📏 Phase: Star alignment starting…")
        self.dlg._on_message("📐 Aligning stars… (42/400)")
        self.assertEqual(self.dlg._rows[-1].operation, "Registration")
        self.assertEqual(self.dlg._rows[-1].status, "running")
        self.assertIn("Aligning stars… (42/400)", self.dlg._rows[-1].note)


if __name__ == "__main__":
    unittest.main()
