"""A spawned worker must import the runtime venv's torch and the app's numpy.

runtime_torch._worker_sys_path starts workers with the app's environment ahead
of the runtime venv. If that environment has a torch of its own (a CPU wheel in
a source run), `import torch` in a worker used to find it, and star detection,
the warp and the demosaic quietly left the GPU.

Real torch is not needed: the "environments" here are folders with stub
packages, and the workers are real spawned processes.
"""
import multiprocessing as mp
import os
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor

import pytest

from setiastro.saspro import runtime_torch as rt


def _pkg(site, name, body):
    d = site / name
    d.mkdir(parents=True)
    (d / "__init__.py").write_text(textwrap.dedent(body))


@pytest.fixture
def envs(tmp_path, monkeypatch):
    """(app site, runtime site), both on sys.path the way a worker gets them."""
    app = tmp_path / "app" / "Lib" / "site-packages"
    run = tmp_path / "runtime" / "py312" / "venv" / "Lib" / "site-packages"
    for site, tag in ((app, "app"), (run, "runtime")):
        for name in ("torch", "torchvision", "sas_probe_other"):
            _pkg(site, name, f"""
                WHERE = {tag!r}
                __version__ = {tag!r}
                class device:
                    def __init__(self, kind): self.type = kind
                """)
    # Whatever an earlier test left behind: the real runtime on sys.path (any
    # in-process stack that touched the GPU puts it there) and its finder.
    # With the runtime moved to the stub below, the real one would pass for
    # the app's environment and hand the workers its numpy.
    real = rt._worker_path_key(rt._runtime_base_dir())
    keep = [p for p in sys.path
            if not (isinstance(p, str) and p and rt._worker_path_key(p).startswith(real + os.sep))]
    monkeypatch.setenv("SASPRO_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(sys, "path", [str(app), str(run)] + keep)
    meta, mods = list(sys.meta_path), dict(sys.modules)
    sys.meta_path[:] = [f for f in meta if not isinstance(f, rt._RuntimeTorchFinder)]
    for k in [k for k in sys.modules if k.split(".")[0] in ("torch", "torchvision")]:
        monkeypatch.delitem(sys.modules, k)
    yield app, run
    sys.meta_path[:] = meta
    for k in [k for k in sys.modules if k not in mods]:
        if k.split(".")[0] in ("torch", "torchvision", "sas_probe_other"):
            del sys.modules[k]


def _names(_=None):
    import importlib
    return {n: importlib.import_module(n).WHERE for n in ("torch", "torchvision", "sas_probe_other")}


def test_torch_comes_from_the_runtime_and_nothing_else_does(envs):
    assert rt.prefer_runtime_torch() is True
    assert _names() == {"torch": "runtime", "torchvision": "runtime", "sas_probe_other": "app"}


def test_without_it_the_apps_torch_wins(envs):
    assert _names() == {"torch": "app", "torchvision": "app", "sas_probe_other": "app"}


def test_a_torch_already_loaded_is_never_mixed_with_another(envs):
    import torch
    assert torch.WHERE == "app"
    assert rt.prefer_runtime_torch() is False
    assert _names()["torchvision"] == "app"


def test_no_runtime_on_the_path_changes_nothing(envs):
    _app, run = envs
    sys.path.remove(str(run))
    before = list(sys.meta_path)
    assert rt.prefer_runtime_torch() is False
    assert sys.meta_path == before
    assert _names()["torch"] == "app"


def test_runtime_without_torch_changes_nothing(envs):
    import shutil
    _app, run = envs
    shutil.rmtree(run / "torch")
    assert rt.prefer_runtime_torch() is False
    assert _names() == {"torch": "app", "torchvision": "app", "sas_probe_other": "app"}


def test_installed_once(envs):
    for _ in range(3):
        assert rt.prefer_runtime_torch() is True
    assert sum(isinstance(f, rt._RuntimeTorchFinder) for f in sys.meta_path) == 1


def test_frozen_build_is_left_alone(envs, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    before = list(sys.meta_path)
    assert rt.prefer_runtime_torch() is False
    assert sys.meta_path == before


def _resolve_in_worker(module):
    import importlib
    m = importlib.import_module(f"setiastro.saspro.{module}")
    from setiastro.saspro import runtime_torch

    def no_bootstrap(*a, **k):
        raise AssertionError("the bootstrap was called in a worker")
    runtime_torch.import_torch = no_bootstrap
    runtime_torch.add_runtime_to_sys_path = lambda *a, **k: runtime_torch.prefer_runtime_torch()
    torch, _dev = m._resolve()
    note = m._RESOLVE_NOTE if module == "gpu_star_detect" else m._NOTE
    import numpy
    return getattr(torch, "WHERE", None), note, os.path.normcase(numpy.__file__)


@pytest.mark.parametrize("module", ["gpu_star_detect", "gpu_warp"])
@pytest.mark.parametrize("broken", [False, True])
def test_the_gpu_modules_in_a_spawned_worker(envs, module, broken):
    import numpy
    _app, run = envs
    if broken:
        (run / "torch" / "__init__.py").write_text("raise ImportError('DLL load failed')")
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        where, note, np_file = ex.submit(_resolve_in_worker, module).result(timeout=120)
    assert np_file == os.path.normcase(numpy.__file__)
    if broken:
        assert where is None and "does not load" in note, (where, note)
    else:
        assert where == "runtime", (where, note)
