from __future__ import annotations

import sys
from types import ModuleType

from setiastro.saspro.runtime_torch import _purge_bad_torch_from_sysmodules


def _clear_torch_modules(monkeypatch) -> None:
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_purge_keeps_complete_site_packages_torch(monkeypatch) -> None:
    _clear_torch_modules(monkeypatch)
    torch = ModuleType("torch")
    torch.__file__ = "/runtime/lib/python3.12/site-packages/torch/__init__.py"
    torch_c = ModuleType("torch._C")
    torch._C = torch_c
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._C", torch_c)

    _purge_bad_torch_from_sysmodules()

    assert sys.modules["torch"] is torch
    assert sys.modules["torch._C"] is torch_c


def test_purge_removes_entire_partial_torch_graph(monkeypatch) -> None:
    _clear_torch_modules(monkeypatch)
    torch = ModuleType("torch")
    torch.__file__ = "/runtime/lib/python3.12/site-packages/torch/__init__.py"
    torch._C = ModuleType("torch._C")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", ModuleType("torch.nn"))

    _purge_bad_torch_from_sysmodules()

    assert not any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    )


def test_purge_removes_entire_shadow_torch_graph(monkeypatch) -> None:
    _clear_torch_modules(monkeypatch)
    torch = ModuleType("torch")
    torch.__file__ = "/worktree/torch/__init__.py"
    torch_c = ModuleType("torch._C")
    torch._C = torch_c
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._C", torch_c)

    _purge_bad_torch_from_sysmodules()

    assert not any(
        name == "torch" or name.startswith("torch.") for name in sys.modules
    )
