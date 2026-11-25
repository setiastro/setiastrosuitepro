# tests/test_curves_bg_worker.py
"""Curves "Precompute full resolution": a finished worker is released only by its
own finished slot. If a newer worker could start in the gap between isRunning()
going False and that slot running, the late slot deleteLater()'d the newer,
still-running worker: qFatal, SASpro gone (user report, 2.0.0, 2026-11-25)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp(tmp_path):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    old = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)       # keep the Curves keys out of the registry
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    opened = []
    yield app, opened
    for d in opened:            # never close(): close releases buffers; just drop them
        d.hide()
        d.deleteLater()
    app.processEvents()
    QSettings.setDefaultFormat(old)


class _Doc:
    def __init__(self, img):
        self.image = img
        self.metadata = {}
        self.masks = {}
        self.active_mask_id = None


def test_late_finished_never_releases_a_newer_worker(qapp):
    from PyQt6.QtCore import QSettings
    from setiastro.saspro.curve_editor_pro import CurvesDialogPro
    app, opened = qapp
    QSettings().setValue("curves/full_resolution_background", True)
    img = np.random.default_rng(1).random((600, 900, 3), dtype=np.float32)
    dlg = CurvesDialogPro(None, _Doc(img))
    opened.append(dlg)
    dlg.show()
    app.processEvents(); app.processEvents()
    assert dlg._full_bg_enabled and dlg._full_img is not None

    dlg._curves_store["R"] = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
    dlg._schedule_full_bg()
    a = dlg._bg_worker
    assert a is not None
    a.wait()                    # A done; its finished is queued but not delivered
    assert not a.isRunning()

    # The curve moves again before A's finished slot runs (a mouse drag on Windows).
    dlg._curves_store["R"] = [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]
    dlg._schedule_full_bg()
    assert dlg._bg_worker is a, "a second worker started before A was released"
    assert dlg._pending_sig is not None

    deadline = time.time() + 30
    while (dlg._bg_worker is not None or dlg._pending_sig is not None) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert dlg._bg_worker is None and dlg._pending_sig is None
    assert dlg._full_cache is not None
    assert dlg._full_cache_sig == dlg._full_result_signature()     # the latest curve, not A's
