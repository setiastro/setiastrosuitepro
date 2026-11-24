# src/setiastro/saspro/setup_manager.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
What is set up and what isn't, and the two installs that set it up.

One SetupManager lives on the main window. It knows the state of hardware
acceleration, the Cosmic Clarity models and the Gaia libraries, and runs the
acceleration install and the model download in the background, so the setup
window can be closed while they work. The setup window (setup_center.py) and
the reminder in the workspace corner both show what it says.

The checks read files and ask torch, and one of them starts a subprocess, so
they run on a thread; nothing here blocks the GUI.
"""
from __future__ import annotations

import os
import platform
import re
import sys
import threading
import time
from dataclasses import dataclass, field

from PyQt6.QtCore import QCoreApplication, QObject, QThread, QTimer, pyqtSignal, pyqtSlot

REMINDER_OFF_KEY = "setup/reminder_off"
GAIA_SEEN_KEY = "setup/gaia_seen"

STEPS = ("accel", "models", "gaia")

# Sizes of the four model downloads, in MB, in the order they are fetched.
MODEL_PARTS_MB = (1811, 860, 68, 68)


def _t(text: str) -> str:
    return QCoreApplication.translate("Setup", text)


@dataclass
class Item:
    """state: checking, ok, warn (works, could be better), todo, blocked."""
    state: str = "checking"
    headline: str = ""
    detail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def needs_doing(self) -> bool:
        return self.state in ("todo", "blocked")


@dataclass
class Job:
    text: str = ""
    percent: float | None = None
    started: float = 0.0
    part: int = 0

    def elapsed(self) -> str:
        s = int(max(0.0, time.monotonic() - self.started))
        h, m = s // 3600, (s % 3600) // 60
        return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m}:{s % 60:02d}"


# ── checks (any thread) ─────────────────────────────────────────────────────

def gpu_names() -> list:
    """The graphics adapters in this machine, as the system names them."""
    try:
        if sys.platform.startswith("win"):
            return _windows_gpu_names()
        if sys.platform == "darwin":
            return ["Apple Silicon"] if platform.machine() == "arm64" else []
        import subprocess
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=10).stdout
        return [ln.split(": ", 1)[-1].strip() for ln in out.splitlines()
                if re.search(r"VGA|3D controller|Display controller", ln)]
    except Exception:
        return []


def _windows_gpu_names() -> list:
    import ctypes
    from ctypes import wintypes

    class DISPLAY_DEVICEW(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                    ("DeviceString", wintypes.WCHAR * 128),
                    ("StateFlags", wintypes.DWORD),
                    ("DeviceID", wintypes.WCHAR * 128),
                    ("DeviceKey", wintypes.WCHAR * 128)]

    names, i = [], 0
    while i < 64:
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not ctypes.windll.user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        name = dd.DeviceString.strip()
        if name and name not in names:
            names.append(name)
        i += 1
    return names


def best_gpu(names) -> str:
    """The adapter hardware acceleration would use, '' if none of them."""
    def rank(n):
        low = n.lower()
        if "apple silicon" in low:
            return 0
        if "nvidia" in low or "geforce" in low:
            return 1
        if "radeon" in low or "amd" in low:
            return 2
        if "intel" in low and ("arc" in low or "iris xe" in low):
            return 3
        return 9
    ranked = sorted((rank(n), i, n) for i, n in enumerate(names))
    return ranked[0][2] if ranked and ranked[0][0] < 9 else ""


def python_problem() -> dict | None:
    """Why acceleration can't be installed yet, or None. A frozen build needs
    a system Python of its own version; pip and source runs use themselves."""
    from setiastro.saspro.runtime_torch import (
        _SUPPORTED_PY_MINORS, _find_system_python_cmd_for_minor, python_download_url,
    )
    host = sys.version_info
    need = f"3.{host.minor}"
    if host.major != 3 or host.minor not in _SUPPORTED_PY_MINORS:
        return {"need": "", "have": f"{host.major}.{host.minor}"}
    if not getattr(sys, "frozen", False):
        return None
    if _find_system_python_cmd_for_minor(host.minor) is not None:
        return None
    return {"need": need, "url": python_download_url(need)}


def check_accel(preferred: str = "auto") -> Item:
    from setiastro.saspro.accel_installer import current_backend
    gpu = best_gpu(gpu_names())
    backend = current_backend()
    extra = {"gpu": gpu, "backend": backend}
    if backend == "Not installed":
        why = python_problem()
        if why and why.get("need"):
            extra.update(why)
            return Item("blocked", _t("Python {0} is needed first").format(why["need"]),
                        _t("This build of SASpro runs on Python {0}, and its "
                           "acceleration package has to be built by the same "
                           "version. Other Python versions can stay installed.")
                        .format(why["need"]), extra)
        if why:
            return Item("warn", _t("Not available"),
                        _t("SASpro is running on Python {0}, which hardware "
                           "acceleration doesn't support.").format(why["have"]), extra)
        return Item("todo", _t("Not installed"),
                    _t("Graphics card found: {0}").format(gpu) if gpu else
                    _t("No supported graphics card was found. The tools will run "
                       "on the CPU, but they still need this package."), extra)
    if backend.startswith("CPU"):
        if preferred == "cpu":
            return Item("ok", _t("Installed, using the CPU"),
                        _t("Preferences is set to CPU only."), extra)
        if gpu:
            why = backend[3:].strip(" ()")
            return Item("warn", _t("Installed, but not using your graphics card"),
                        f"{gpu}. {why}" if why else gpu, extra)
        return Item("ok", _t("Installed, using the CPU"),
                    _t("No supported graphics card was found."), extra)
    return Item("ok", _t("Installed"), backend, extra)


def check_models() -> Item:
    from setiastro.saspro.model_manager import (
        AI4_REQUIRED_MODELS, AI4_SUPPLEMENT_MODELS, model_path,
    )
    need = [f for files in AI4_REQUIRED_MODELS.values() for f in files]
    missing = [f for f in need if not model_path(f).exists()]
    if len(missing) == len(need):
        return Item("todo", _t("Not installed"), "", {"missing": missing})
    if missing:
        return Item("todo", _t("Incomplete"),
                    _t("{0} of {1} core files are missing.").format(len(missing), len(need)),
                    {"missing": missing})
    newer = [name for name, files in AI4_SUPPLEMENT_MODELS.items()
             if not all(model_path(f).exists() for f in files)]
    if newer:
        return Item("warn", _t("Installed, with newer models available"),
                    ", ".join(newer), {"newer": newer})
    return Item("ok", _t("Installed"))


def check_gaia() -> Item:
    from PyQt6.QtCore import QSettings
    from setiastro.saspro.gaia_database import (
        _SETTINGS_ASTRO_LIBRARY_DIR, _SETTINGS_LIBRARY_DIR, ASTRO_GROUP_DEFS,
        GROUP_DEFS, get_astro_library_dir, get_library_dir,
    )
    settings = QSettings()
    lost = []

    def count(defs, key, folder):
        # A saved folder that isn't there (a drive that isn't connected) is not
        # "None installed". Asked before folder(), which falls back to the
        # default one, or creates the saved one, and would say just that.
        saved = settings.value(key, "", type=str)
        if saved and not os.path.isdir(saved):
            lost.append(saved)
            return 0, len(defs), 0
        have = {e.name: e.stat().st_size for e in os.scandir(folder()) if e.is_file()}
        done = sum(1 for g in defs if all(f in have for f in g.filenames))
        size = sum(have.get(f, 0) for g in defs for f in g.filenames)
        return done, len(defs), size

    s_done, s_all, s_size = count(GROUP_DEFS, _SETTINGS_LIBRARY_DIR, get_library_dir)
    a_done, a_all, a_size = count(ASTRO_GROUP_DEFS, _SETTINGS_ASTRO_LIBRARY_DIR,
                                  get_astro_library_dir)
    extra = {"spectral": (s_done, s_all), "astro": (a_done, a_all)}
    found = _t("{0} of {1} sets installed").format(s_done + a_done, s_all + a_all)
    if lost:
        where = _t("Not found: {0}. Connect that drive, or choose the folder in the Gaia "
                   "manager.").format(", ".join(lost))
        return Item("warn", found if s_done or a_done else _t("Gaia folder not found"),
                    where, extra)
    if not s_done and not a_done:
        return Item("todo", _t("None installed"), "", extra)
    return Item("ok", found,
                _t("{0} GB on disk").format(f"{(s_size + a_size) / 1024 ** 3:.1f}"), extra)


# ── progress text ───────────────────────────────────────────────────────────

_BYTES = re.compile(r"\((\d+)/(\d+) bytes\)")
_PART_STARTS = (("Downloading Walking Noise models", 1),
                ("Downloading Aberration Correction model", 2),
                ("Downloading Aberration Correction V2 model", 3))


def _size(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def model_progress(job: Job, msg: str) -> None:
    """Turn one line from the download worker into the job's text and percent."""
    msg = str(msg).strip()
    for head, part in _PART_STARTS:
        if msg.startswith(head):
            job.part = part
    total = sum(MODEL_PARTS_MB)
    before = sum(MODEL_PARTS_MB[:job.part])
    where = _t("part {0} of {1}").format(job.part + 1, len(MODEL_PARTS_MB))
    m = _BYTES.search(msg)
    if m:
        done, size = int(m.group(1)), max(1, int(m.group(2)))
        job.percent = 100.0 * (before + MODEL_PARTS_MB[job.part] * done / size) / total
        job.text = _t("Downloading {0} of {1}, {2}").format(_size(done), _size(size), where)
        return
    if msg.startswith(("Extracting", "Installing to", "Overlaying", "Installed:")):
        job.text = _t("Installing, {0}").format(where)
        job.percent = 100.0 * (before + MODEL_PARTS_MB[job.part]) / total
    elif msg.startswith(("Downloading", "Connecting", "Trying", "Google Drive")):
        job.text = _t("Connecting, {0}").format(where)
        job.percent = 100.0 * before / total
    elif msg and not msg.startswith("Download complete"):
        job.text = msg.splitlines()[0][:110]


# ── the manager (GUI thread) ────────────────────────────────────────────────

class SetupManager(QObject):
    changed = pyqtSignal()                       # an item or a job changed
    finished = pyqtSignal(str, bool, str)        # job, ok, message
    _checked = pyqtSignal(object)                # from the checking thread

    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window
        self.items = {k: Item() for k in STEPS}
        self.jobs: dict = {}
        self.restart_needed = False
        self._threads: dict = {}
        self._cancel_models = False
        self._checking = False
        self._check_again = False
        self._checked.connect(self._on_checked)
        self._tick = QTimer(self)                # elapsed time keeps moving
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self.changed)

    # ---- state -----------------------------------------------------------
    def refresh(self) -> None:
        """Look again, in the background."""
        if self._checking:
            self._check_again = True
            return
        self._checking = True
        pref = (self._mw.settings.value("accel/preferred_backend", "auto", type=str)
                or "auto").lower()
        done = self._checked

        def work():
            out = {}
            for key, fn in (("models", check_models), ("gaia", check_gaia),
                            ("accel", lambda: check_accel(pref))):
                try:
                    out[key] = fn()
                except Exception as e:
                    out[key] = Item("warn", _t("Couldn't check"), str(e)[:200])
            try:
                done.emit(out)
            except RuntimeError:
                pass                             # SASpro closed meanwhile

        threading.Thread(target=work, name="setup-check", daemon=True).start()

    @pyqtSlot(object)
    def _on_checked(self, out):
        self._checking = False
        self.items.update(out)
        self.changed.emit()
        if self._check_again:
            self._check_again = False
            self.refresh()

    def checked(self) -> bool:
        return all(i.state != "checking" for i in self.items.values())

    def remaining(self) -> list:
        """Steps still worth reminding about. Gaia is optional and large, so
        only until the user has looked at it once."""
        out = [k for k in ("accel", "models")
               if self.items[k].needs_doing and not self.is_running(k)
               and not (k == "accel" and self.restart_needed)]
        if (self.items["gaia"].needs_doing
                and not self._mw.settings.value(GAIA_SEEN_KEY, False, type=bool)):
            out.append("gaia")
        return out

    def gaia_seen(self) -> None:
        self._mw.settings.setValue(GAIA_SEEN_KEY, True)
        self.changed.emit()

    # ---- jobs ------------------------------------------------------------
    def is_running(self, name: str) -> bool:
        return name in self.jobs

    def install_accel(self) -> None:
        if self.is_running("accel"):
            return
        from setiastro.saspro.accel_workers import AccelInstallWorker
        pref = (self._mw.settings.value("accel/preferred_backend", "auto", type=str)
                or "auto").lower()
        self._start("accel", AccelInstallWorker(prefer_gpu=True, preferred_backend=pref),
                    self._on_accel_progress, self._on_accel_finished,
                    _t("Preparing"))

    def install_models(self) -> None:
        if self.is_running("models"):
            return
        from setiastro.saspro.model_workers import make_models_download_worker
        self._cancel_models = False
        self._start("models", make_models_download_worker(lambda: self._cancel_models),
                    self._on_models_progress, self._on_models_finished,
                    _t("Connecting, part 1 of 4"))

    def install_models_zip(self, path: str) -> None:
        if self.is_running("models"):
            return
        from setiastro.saspro.model_workers import ModelsInstallZipWorker
        self._start("models", ModelsInstallZipWorker(path),
                    self._on_zip_progress, self._on_models_finished,
                    _t("Reading the ZIP"))

    def cancel_models(self) -> None:
        self._cancel_models = True
        if "models" in self.jobs:
            self.jobs["models"].text = _t("Stopping")
            self.changed.emit()

    def _start(self, name, worker, on_progress, on_finished, text):
        th = QThread()                           # no parent: outlives any window
        worker.moveToThread(th)
        th.started.connect(worker.run)
        # Bound slots of this object, so they run on the GUI thread.
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        self._threads[name] = (th, worker)
        self.jobs[name] = Job(text=text, started=time.monotonic())
        self._tick.start()
        self.changed.emit()
        th.start()

    @pyqtSlot(str)
    def _on_accel_progress(self, msg):
        job = self.jobs.get("accel")
        line = next((ln.strip() for ln in reversed(str(msg).splitlines()) if ln.strip()), "")
        if job is not None and line:
            job.text = line[:110]
            self.changed.emit()

    @pyqtSlot(str)
    def _on_models_progress(self, msg):
        job = self.jobs.get("models")
        if job is not None and not self._cancel_models:
            model_progress(job, msg)
            self.changed.emit()

    @pyqtSlot(str)
    def _on_zip_progress(self, msg):
        job = self.jobs.get("models")
        if job is not None and str(msg).strip():
            job.text = str(msg).strip()[:110]
            self.changed.emit()

    @pyqtSlot(bool, str)
    def _on_accel_finished(self, ok, msg):
        # A fresh install that couldn't be imported into this run still worked
        # (some Python 3.14 builds); Preferences treats it the same way.
        ok = bool(ok) or "Hardware Acceleration installed successfully" in str(msg)
        if ok:
            self.restart_needed = True
        self._end("accel", ok, msg)

    @pyqtSlot(bool, str)
    def _on_models_finished(self, ok, msg):
        # Stopped on request is not a failure: no message.
        self._end("models", bool(ok), "" if self._cancel_models and not ok else msg)

    def _end(self, name, ok, msg):
        self.jobs.pop(name, None)
        th, worker = self._threads.pop(name, (None, None))
        if th is not None:
            th.quit()
            th.wait()
            worker.deleteLater()
            th.deleteLater()
        if not self.jobs:
            self._tick.stop()
        self.refresh()
        self.changed.emit()
        self.finished.emit(name, ok, str(msg or ""))

    # ---- closing SASpro --------------------------------------------------
    def stop_for_exit(self) -> bool:
        """Ask the model download to stop and give it a moment. True if
        nothing is left running (acceleration can't be stopped part way)."""
        if "models" in self._threads:
            self._cancel_models = True
            th, _worker = self._threads["models"]
            th.quit()
            if th.wait(4000):
                self._threads.pop("models", None)
                self.jobs.pop("models", None)
        return not self._threads


def setup_manager(main_window) -> SetupManager:
    """The main window's manager, made on first use."""
    mgr = getattr(main_window, "_setup_manager", None)
    if mgr is None:
        mgr = main_window._setup_manager = SetupManager(main_window)
    return mgr
