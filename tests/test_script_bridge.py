"""setiastrosuitepro --run-script (script_bridge.py, script_bridge_host.py).

The command runs as a real subprocess against a real single-instance listener
in this process, with a stand-in main window, on its own instance bucket.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from setiastro.saspro import script_bridge as sb  # noqa: E402

SRC = str(Path(__file__).resolve().parents[1] / "src")


# ---- the command on its own -------------------------------------------------

def test_flag_parsing():
    assert sb.wants_run_script(["--run-script", "a.py"])
    assert sb.wants_run_script(["--RUN-SCRIPT=a.py"])
    assert not sb.wants_run_script(["a.fits", "--new-instance"])
    assert sb.script_argument(["--run-script", "a b.py"]) == "a b.py"
    assert sb.script_argument(["--run-script=c.py"]) == "c.py"
    assert sb.script_argument(["--run-script"]) is None
    assert sb.script_argument(["--run-script="]) is None


def test_usage_errors(tmp_path, capsys):
    assert sb.client_main(["--run-script"]) == sb.EXIT_USAGE
    assert sb.client_main(["--run-script", "--help"]) == sb.EXIT_OK
    assert "Exit codes" in capsys.readouterr().out
    assert sb.client_main(["--run-script", str(tmp_path / "missing.py")]) == sb.EXIT_USAGE
    other = tmp_path / "notes.txt"
    other.write_text("x")
    assert sb.client_main(["--run-script", str(other)]) == sb.EXIT_USAGE


def test_not_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SASPRO_INSTANCE", f"nobody{uuid.uuid4().hex[:10]}")
    script = tmp_path / "s.py"
    script.write_text("def run(ctx):\n    pass\n")
    assert sb.client_main(["--run-script", str(script)]) == sb.EXIT_NOT_RUNNING
    assert "not open" in capsys.readouterr().err


# ---- the command against a listener -----------------------------------------

@pytest.fixture(scope="module")
def host():
    """A listener on its own bucket, with a stand-in main window."""
    from PyQt6.QtWidgets import QApplication, QMainWindow, QMdiArea
    from setiastro.saspro.script_bridge_host import run_from_command
    from setiastro.saspro.single_instance import install_primary_listener

    app = QApplication.instance() or QApplication([])
    tag = f"pytest{uuid.uuid4().hex[:10]}"
    old = os.environ.get("SASPRO_INSTANCE")
    os.environ["SASPRO_INSTANCE"] = tag

    class Main(QMainWindow):
        def __init__(self):
            super().__init__()
            self.mdi = QMdiArea()
            self.setCentralWidget(self.mdi)
            self.lines = []

        def _log(self, msg):
            self.lines.append(msg)

    win = Main()
    server = install_primary_listener(
        app, lambda body: None, on_script=lambda body, sock: run_from_command(win, body, sock))
    assert server is not None
    yield app, win, tag
    server.close()
    if old is None:
        os.environ.pop("SASPRO_INSTANCE", None)
    else:
        os.environ["SASPRO_INSTANCE"] = old


class Command:
    """setiastrosuitepro --run-script, as a subprocess; stdout lines timed."""

    def __init__(self, tag, script):
        env = dict(os.environ, SASPRO_INSTANCE=tag, PYTHONPATH=SRC, PYTHONUTF8="1")
        code = ("import sys; from setiastro.saspro.script_bridge import client_main; "
                "sys.exit(client_main(sys.argv[1:]))")
        self.p = subprocess.Popen([sys.executable, "-c", code, "--run-script", str(script)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.out = []                            # (seconds since start, line)
        self.err = b""
        self.t0 = time.monotonic()
        self._threads = [threading.Thread(target=self._read_out, daemon=True),
                         threading.Thread(target=self._read_err, daemon=True)]
        for t in self._threads:
            t.start()

    def _read_out(self):
        for line in self.p.stdout:
            # Text-mode stdout: Windows writes CRLF, as for any console program.
            self.out.append((time.monotonic() - self.t0,
                             line.decode("utf-8").replace("\r\n", "\n")))

    def _read_err(self):
        self.err = self.p.stderr.read()

    def wait(self, app, timeout=30.0):
        end = time.monotonic() + timeout
        while self.p.poll() is None:
            assert time.monotonic() < end, "the command did not finish"
            app.processEvents()
            time.sleep(0.005)
        for t in self._threads:
            t.join(5)
        app.processEvents()
        return self.p.returncode

    @property
    def stdout(self):
        return "".join(line for _t, line in self.out)

    @property
    def stderr(self):
        return self.err.decode("utf-8").replace("\r\n", "\n")


def _script(tmp_path, body, name="script.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def run(host, script):
    app, _win, tag = host
    c = Command(tag, script)
    return c.wait(app), c


def test_output_and_log(host, tmp_path):
    s = _script(tmp_path, """
        def run(ctx):
            print("first line")
            ctx.log("from ctx.log")
            print("Ha α and µm")
            print("no newline at the end", end="")
    """)
    rc, c = run(host, s)
    assert rc == 0, c.stderr
    assert c.stdout.splitlines() == ["first line", "from ctx.log", "Ha α and µm",
                                     "no newline at the end"]
    assert any("from ctx.log" in m for m in host[1].lines)


def test_main_is_an_entry_point_too(host, tmp_path):
    rc, c = run(host, _script(tmp_path, "def main(ctx):\n    print('main ran')\n"))
    assert (rc, c.stdout) == (0, "main ran\n")


def test_exception_gives_file_and_line(host, tmp_path):
    s = _script(tmp_path, """
        def helper():
            raise ValueError("boom")

        def run(ctx):
            print("before")
            helper()
    """, name="breaks.py")
    rc, c = run(host, s)
    assert rc == 1
    assert c.stdout == "before\n"
    err = c.stderr
    assert "Traceback" in err and "ValueError: boom" in err
    assert f'File "{s}", line 3, in helper' in err
    assert "script_bridge_host" not in err          # only the script's own frames


def test_syntax_error(host, tmp_path):
    rc, c = run(host, _script(tmp_path, "def run(ctx):\n    print('x'\n"))
    assert rc == 1
    assert "SyntaxError" in c.stderr and "script.py" in c.stderr


def test_sys_exit(host, tmp_path):
    rc, c = run(host, _script(tmp_path, "import sys\ndef run(ctx):\n    sys.exit(3)\n"))
    assert rc == 1 and "sys.exit(3)" in c.stderr
    rc, c = run(host, _script(tmp_path, "import sys\ndef run(ctx):\n    sys.exit(0)\n"))
    assert rc == 0


def test_no_entry_point(host, tmp_path):
    rc, c = run(host, _script(tmp_path, "X = 1\n"))
    assert rc == 2 and "no run(ctx)" in c.stderr


def test_edit_is_picked_up(host, tmp_path):
    s = _script(tmp_path, "def run(ctx):\n    print('one')\n")
    assert run(host, s)[1].stdout == "one\n"
    s.write_text("def run(ctx):\n    print('two')\n")
    assert run(host, s)[1].stdout == "two\n"


def test_output_streams_while_the_gui_thread_is_busy(host, tmp_path):
    s = _script(tmp_path, """
        import time
        def run(ctx):
            print("first")
            time.sleep(1.5)
            print("second")
    """)
    rc, c = run(host, s)
    assert rc == 0
    (t1, a), (t2, b) = c.out
    assert (a, b) == ("first\n", "second\n")
    assert t2 - t1 > 1.0, c.out                  # "first" came before the sleep ended


def test_print_from_a_thread(host, tmp_path):
    s = _script(tmp_path, """
        import threading
        def run(ctx):
            t = threading.Thread(target=lambda: print("from a thread"))
            t.start()
            t.join()
    """)
    rc, c = run(host, s)
    assert (rc, c.stdout) == (0, "from a thread\n")


def test_modal_dialog(host, tmp_path):
    s = _script(tmp_path, """
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QDialog
        def run(ctx):
            d = QDialog(ctx.main_window())
            QTimer.singleShot(200, lambda: (print("applied"), d.accept()))
            print("result", d.exec())
    """)
    rc, c = run(host, s)
    assert (rc, c.stdout) == (0, "applied\nresult 1\n")


def test_shown_dialog_keeps_the_command_until_closed(host, tmp_path):
    s = _script(tmp_path, """
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QDialog
        def run(ctx):
            d = QDialog(ctx.main_window())
            d.show()
            QTimer.singleShot(600, lambda: (print("closing"), d.close()))
            print("run returned")
    """)
    rc, c = run(host, s)
    assert (rc, c.stdout) == (0, "run returned\nclosing\n")
    assert c.out[1][0] - c.out[0][0] > 0.4


def test_exception_from_a_dialog_button(host, tmp_path):
    s = _script(tmp_path, """
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QDialog, QPushButton, QVBoxLayout
        def run(ctx):
            d = QDialog(ctx.main_window())
            b = QPushButton("Apply", d)
            QVBoxLayout(d).addWidget(b)
            def apply():
                raise RuntimeError("apply failed")
            b.clicked.connect(apply)
            d.show()
            QTimer.singleShot(200, b.click)
            QTimer.singleShot(500, d.close)
    """, name="button.py")
    rc, c = run(host, s)
    assert rc == 1
    assert "RuntimeError: apply failed" in c.stderr and "button.py" in c.stderr


def test_second_script_while_one_runs(host, tmp_path):
    app, _win, tag = host
    long = _script(tmp_path, """
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QDialog
        def run(ctx):
            d = QDialog(ctx.main_window())
            d.show()
            QTimer.singleShot(2500, d.close)
    """, name="long.py")
    quick = _script(tmp_path, "def run(ctx):\n    print('should not run')\n", name="quick.py")
    first = Command(tag, long)
    end = time.monotonic() + 15
    from setiastro.saspro import script_bridge_host as h
    while h._current is None:                    # the first one is running
        assert time.monotonic() < end
        app.processEvents()
        time.sleep(0.005)
    second = Command(tag, quick)
    assert second.wait(app) == sb.EXIT_BUSY
    assert "long.py" in second.stderr and second.stdout == ""
    assert first.wait(app) == 0
    # and free again afterwards
    assert run(host, quick)[1].stdout == "should not run\n"


def test_streams_are_put_back(host, tmp_path):
    out, err, hook = sys.stdout, sys.stderr, sys.excepthook
    run(host, _script(tmp_path, "def run(ctx):\n    print('x')\n"))
    run(host, _script(tmp_path, "def run(ctx):\n    1/0\n"))
    assert (sys.stdout, sys.stderr, sys.excepthook) == (out, err, hook)
