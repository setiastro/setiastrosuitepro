"""The Unhandled Exception box's report: what it carries, and an email link that fits."""
from __future__ import annotations

import os
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("PyQt6")
from setiastro.saspro.widgets import common_utilities as cu  # noqa: E402

HEAD = "AttributeError: 'StatisticalStretchDialog' object has no attribute '_full_timer'"


def _traceback(n_frames: int) -> str:
    home = os.path.expanduser("~")
    frames = "".join(
        f'  File "{home}\\saspro\\mod{i}.py", line {i}, in f{i}\n    step_{i}()\n'
        for i in range(n_frames))
    return "Traceback (most recent call last):\n" + frames + HEAD + "\n"


def _body(url: str) -> str:
    return parse_qs(urlsplit(url).query)["body"][0]


def test_report_carries_versions_thread_and_error():
    rep = cu.build_crash_report(HEAD, _traceback(3), thread="ProbeWorker")
    assert rep.startswith("SASpro crash report\nSASpro: ")
    for line in ("OS: ", "Python: ", "Thread: ProbeWorker"):
        assert line in rep
    assert HEAD in rep and "mod2.py" in rep


def test_report_hides_the_home_folder():
    home = os.path.expanduser("~")
    rep = cu.build_crash_report(f"OSError: can't read '{home}\\x.fits'".replace("\\", "\\\\"),
                                _traceback(2))
    assert home.lower() not in rep.lower()
    assert home.replace("\\", "\\\\").lower() not in rep.lower()
    assert "~\\saspro\\mod1.py" in rep


def test_email_link_goes_to_frank_with_the_whole_short_report():
    url = cu.crash_mailto_url(HEAD, _traceback(3))
    assert url.startswith("mailto:frank@setiastro.com?subject=")
    body = _body(url)
    assert "What were you doing" in body
    assert HEAD in body and "mod0.py" in body and "cut" not in body


def test_long_traceback_is_cut_from_the_top_to_fit():
    url = cu.crash_mailto_url(HEAD, _traceback(80))
    assert len(url) <= cu._MAILTO_MAX
    body = _body(url)
    assert "SASpro: " in body and HEAD in body
    assert "mod79.py" in body           # innermost frame kept
    assert "mod0.py" not in body        # outermost cut
    assert "clipboard" in body          # and the reader is told where the rest is


def test_huge_error_message_still_fits():
    url = cu.crash_mailto_url("ValueError: " + "x" * 6000, _traceback(5))
    assert len(url) <= cu._MAILTO_MAX
    assert "What were you doing" in _body(url)
