"""Cosmic Clarity must never hand back a NaN image as a result.

An RX 9070 XT (gfx1201) under ROCm 7.2 returned all-NaN fp16 output from
MIOpen's Winograd convolutions; Dark Star and Denoise showed it as a black
image. fp32, and fp16 with Winograd off, matched the CPU. So runtime_torch turns
Winograd off for MIOpen, and forward_finite catches any NaN that still gets
through: an fp32 re-run, then NonFiniteOutputError, which the engines answer
with their CPU retry. Real torch on the CPU; the net and autocast are fakes.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from setiastro.saspro.cosmicclarity_engines import _backend_guard as bg

torch = pytest.importorskip("torch")


class _Amp:
    """Stands in for autocast: the fake net reads whether it is on."""
    on = False

    def __enter__(self):
        _Amp.on = True

    def __exit__(self, *exc):
        _Amp.on = False


class _Net:
    def __init__(self, nan_amp=False, nan_fp32=False):
        self.nan_amp, self.nan_fp32, self.calls = nan_amp, nan_fp32, []

    def __call__(self, x):
        self.calls.append("amp" if _Amp.on else "fp32")
        bad = self.nan_amp if _Amp.on else self.nan_fp32
        y = x * 2
        return y * float("nan") if bad else y


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    monkeypatch.setattr(bg, "_AMP_UNRELIABLE", False)


def _x():
    return torch.rand(2, 3, 8, 8)


def test_good_fp16_output_is_used_as_is():
    net, x = _Net(), _x()
    y = bg.forward_finite(torch, net, x, _Amp(), label="t")
    assert net.calls == ["amp"] and torch.equal(y, x * 2)
    assert bg._AMP_UNRELIABLE is False


def test_nan_fp16_reruns_in_fp32_and_the_session_stays_fp32():
    net, x = _Net(nan_amp=True), _x()
    y = bg.forward_finite(torch, net, x, _Amp(), label="t")
    assert net.calls == ["amp", "fp32"] and torch.equal(y, x * 2)
    assert bg._AMP_UNRELIABLE is True
    net.calls.clear()
    bg.forward_finite(torch, net, x, _Amp(), label="t")
    assert net.calls == ["fp32"]                     # no second fp16 attempt


def test_nan_in_fp32_too_raises_for_the_cpu_retry():
    net = _Net(nan_amp=True, nan_fp32=True)
    with pytest.raises(bg.NonFiniteOutputError) as ei:
        bg.forward_finite(torch, net, _x(), _Amp(), label="t")
    e = ei.value
    assert bg.is_backend_unusable_error(e)           # the engines retry on CPU
    assert not bg.is_recoverable_batch_error(e)      # a smaller batch won't fix it
    assert not bg.is_device_lost_error(e)            # and DirectML isn't poisoned
    assert bg._AMP_UNRELIABLE is False               # fp16 was not to blame


@pytest.mark.parametrize("amp_unreliable", [False, True])
def test_nan_already_in_the_input_passes_through(monkeypatch, amp_unreliable):
    monkeypatch.setattr(bg, "_AMP_UNRELIABLE", amp_unreliable)
    x = _x()
    x[0, 0, 0, 0] = float("nan")
    y = bg.forward_finite(torch, _Net(), x, _Amp(), label="t")
    assert torch.isnan(y).any()                      # NaN in, NaN out, no error
    assert bg._AMP_UNRELIABLE is amp_unreliable


def test_dark_star_batches_go_through_the_guard(monkeypatch):
    from setiastro.saspro.cosmicclarity_engines import darkstar_engine as de
    monkeypatch.setattr(de, "_autocast_context", lambda torch, dev: _Amp())
    rng = np.random.default_rng(0)
    tiles = [rng.random((16, 16, 3), dtype=np.float32) for _ in range(3)]
    outs, done = de._infer_tiles_rgb_batched(
        tiles, model=_Net(nan_amp=True),
        models=SimpleNamespace(torch=torch, device=torch.device("cpu")),
        padded_h=16, padded_w=16, batch_size=2)
    assert done == 3
    for t, o in zip(tiles, outs):
        np.testing.assert_allclose(o, t * 2, rtol=1e-6)


def _winograd_after_import(extra):
    """MIOPEN_DEBUG_CONV_WINOGRAD in a fresh process once runtime_torch is
    imported, and in a child that process starts."""
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(bg.__file__))))
    env = {k: v for k, v in os.environ.items() if k != "MIOPEN_DEBUG_CONV_WINOGRAD"}
    env.update(extra, PYTHONPATH=src)
    child = "import os; print(os.environ.get('MIOPEN_DEBUG_CONV_WINOGRAD'))"
    code = ("import os, subprocess, sys\n"
            "from setiastro.saspro import runtime_torch\n"
            "print(os.environ.get('MIOPEN_DEBUG_CONV_WINOGRAD'))\n"
            f"print(subprocess.run([sys.executable, '-c', {child!r}],"
            " capture_output=True, text=True).stdout.strip())\n")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, check=True)
    return r.stdout.split()[-2:]


def test_winograd_is_off_for_miopen_and_children_inherit_it():
    assert _winograd_after_import({}) == ["0", "0"]


def test_a_user_setting_wins():
    assert _winograd_after_import({"MIOPEN_DEBUG_CONV_WINOGRAD": "1"}) == ["1", "1"]
