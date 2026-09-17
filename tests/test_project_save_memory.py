from __future__ import annotations

import os
import time
import zipfile

import numpy as np
import pytest

from setiastro.saspro.doc_manager import ImageDocument
from setiastro.saspro.project_io import ProjectWriter, _np_load_from_bytes
from setiastro.saspro.swap_manager import get_swap_manager


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wait_for_swap_file(sm, swap_id, timeout=2.0):
    path = sm.get_swap_path(swap_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return path
        time.sleep(0.02)
    return path


def test_swap_load_state_cache_false_does_not_grow_ram_cache():
    sm = get_swap_manager()
    arr = np.full((24, 32), 0.3, dtype=np.float32)
    sid = sm.save_state(arr)
    assert sid
    _wait_for_swap_file(sm, sid)

    with sm._cache_lock:
        old = sm._cache.pop(sid, None)
        if old is not None:
            sm._cache_used -= sm._arr_nbytes(old)

    loaded = sm.load_state(sid, cache=False)
    assert loaded is not None
    assert np.allclose(np.asarray(loaded), arr)
    assert sid not in sm._cache


def test_project_writer_loads_history_without_caching(tmp_path, qapp, monkeypatch):
    caches = []

    class FakeSM:
        def load_state(self, sid, cache=True, **_kw):
            caches.append(bool(cache))
            return np.ones((8, 8), dtype=np.float32)

    monkeypatch.setattr("setiastro.saspro.project_io.get_swap_manager", lambda: FakeSM())

    doc = ImageDocument(
        np.full((8, 8), 0.5, dtype=np.float32),
        metadata={"display_name": "memtest", "bit_depth": "32-bit floating point"},
    )
    doc._undo = [("undo-sid", {"step": 1}, "Stretch")]
    doc._redo = [("redo-sid", {"step": 2}, "Crop")]

    out = tmp_path / "memtest.sas"
    ProjectWriter.write(str(out), docs=[doc], compress=False)

    assert caches == [False, False]
    with zipfile.ZipFile(out, "r") as z:
        names = z.namelist()
        assert any(n.endswith("/current.npy") or n.endswith("/current.npz") for n in names)
        img = _np_load_from_bytes(z.read([n for n in names if "/current." in n][0]))
        assert img.shape == (8, 8)
        assert np.allclose(img, 0.5)
