from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_normalize_plate_solve_preset_defaults():
    from setiastro.saspro.plate_solver import normalize_plate_solve_preset
    spec = normalize_plate_solve_preset(None)
    assert spec["seed_mode"] == "auto"
    assert spec["solver"] == "both"
    assert spec["radius_mode"] == "auto"
    assert spec["fov_mode"] == "compute"


def test_normalize_plate_solve_preset_aliases():
    from setiastro.saspro.plate_solver import normalize_plate_solve_preset
    spec = normalize_plate_solve_preset({
        "seed_mode": "Manual",
        "solver_pref": "astap_only",
        "ra": "22:32:14",
        "dec": "+40:42:43",
        "scale_arcsec": "1.46",
        "radius_mode": "value",
        "radius_value": 4,
        "fov_mode": "auto",
    })
    assert spec["seed_mode"] == "manual"
    assert spec["solver"] == "astap_only"
    assert spec["ra"] == "22:32:14"
    assert spec["dec"] == "+40:42:43"
    assert spec["scale"] == pytest.approx(1.46)
    assert spec["radius_mode"] == "value"
    assert spec["radius_value"] == pytest.approx(4.0)
    assert spec["fov_mode"] == "auto"


def test_apply_plate_solve_preset_writes_settings(qapp):
    from PyQt6.QtCore import QSettings
    from setiastro.saspro.plate_solver import (
        apply_plate_solve_preset,
        _get_seed_mode,
        _get_manual_ra,
        _get_solver_preference,
    )
    settings = QSettings("SASproTest", "plate_solve_preset")
    settings.clear()
    apply_plate_solve_preset(settings, {
        "seed_mode": "manual",
        "ra": "10.5",
        "dec": "41.2",
        "solver": "gaia_only",
    })
    assert _get_seed_mode(settings) == "manual"
    assert _get_manual_ra(settings) == "10.5"
    assert _get_solver_preference(settings) == "gaia_only"


def test_apply_plate_solve_to_doc_uses_given_document(qapp, monkeypatch):
    from setiastro.saspro.doc_manager import ImageDocument
    from setiastro.saspro import plate_solver

    seen = {}

    def _fake_solve(parent, doc, settings):
        seen["doc"] = doc
        return True, "ok"

    monkeypatch.setattr(plate_solver, "plate_solve_doc_inplace", _fake_solve)
    doc = ImageDocument(np.full((6, 8), 0.2, dtype=np.float32), metadata={"display_name": "target"})
    ok, res = plate_solver.apply_plate_solve_to_doc(None, doc, None, {"solver": "gaia_only"})
    assert ok is True
    assert res == "ok"
    assert seen["doc"] is doc


def test_plate_solve_command_spec_is_registered():
    from setiastro.saspro.ops.commands import get_spec
    spec = get_spec("plate_solve")
    assert spec is not None
    assert spec.headless_method == "_apply_plate_solve_to_doc"
    assert spec.ui_method == "_open_plate_solver"


def test_plate_solve_preset_dialog_roundtrip(qapp):
    from setiastro.saspro.plate_solver import PlateSolverPresetDialog
    dlg = PlateSolverPresetDialog(None, initial={
        "seed_mode": "none",
        "solver": "astrometry_only",
        "radius_mode": "value",
        "radius_value": 3.5,
        "fov_mode": "value",
        "fov_value": 1.8,
    })
    out = dlg.result_dict()
    assert out["seed_mode"] == "none"
    assert out["solver"] == "astrometry_only"
    assert out["radius_mode"] == "value"
    assert out["radius_value"] == pytest.approx(3.5)
    assert out["fov_mode"] == "value"
    assert out["fov_value"] == pytest.approx(1.8)
