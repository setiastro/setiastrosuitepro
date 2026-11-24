# src/setiastro/saspro/settings_sandbox.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
--firstrun: run SASpro against an empty, throwaway settings store, so it comes
up exactly as a brand-new install does -- default layout, default background,
first-run dialog -- without touching the real settings.

    setiastrosuitepro --firstrun
    python -m setiastro.saspro --firstrun
    SetiAstroSuitePro.exe --firstrun

--upgraderun: the same, but the throwaway store starts as a copy of the real
settings (API keys left out) and this launch behaves as the first start of SASpro 2.0 (the saved
layout is set aside and the welcome window offers it back). For trying the
returning-user path as often as needed; the real settings are only read.

How: every default-constructed QSettings() follows QSettings.defaultFormat(),
so pointing the INI format at a fresh temp folder before anything reads a
setting redirects all of them. The folder is deleted when SASpro exits, so
each --firstrun launch starts from nothing again, except for where the Gaia
databases are: that is carried over from the real settings. They are on disk
like the AI models and the acceleration package, which a first run finds too,
and tens of GB that nobody should be told to download twice. --firstrun also implies
--new-instance (a running SASpro would otherwise just be brought forward).

Not covered: a few tools open QSettings("SetiAstro", "<name>") with an
explicit organisation and application, which always uses the native store
(the registry on Windows). Those still read -- and if changed, write -- the
real settings. named_settings() is the sandbox-aware way to open such a
store; the icon pack and language preferences use it, because both change
what a first-run window looks like.

Stdlib-only at import: __main__.entry() imports this before the
single-instance handoff, where PyQt6 is deliberately not loaded yet.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

FIRSTRUN_FLAGS = ("--firstrun", "--first-run")
UPGRADERUN_FLAGS = ("--upgraderun", "--upgrade-run")
_ENV = "SASPRO_SETTINGS_SANDBOX"
_dir: str | None = None


def _has_flag(argv, flags) -> bool:
    return any(str(a).strip().lower() in flags for a in (argv or []))


def wants_sandbox(argv) -> bool:
    """--firstrun or --upgraderun is on the command line."""
    return _has_flag(argv, FIRSTRUN_FLAGS + UPGRADERUN_FLAGS)


def enter_sandbox(argv) -> str:
    """Enter the sandbox the command line asks for. Idempotent; returns the
    folder. Must run before anything constructs a QSettings."""
    if _dir:
        return _dir
    if _has_flag(argv, UPGRADERUN_FLAGS):
        return _enter_upgrade_sandbox()
    return enter_firstrun_sandbox()


# Where the Gaia databases are (gaia_database.py, gaia_quad_index.py).
_GAIA_DIR_KEYS = ("gaia_library/custom_dir", "gaia_astro_library/custom_dir",
                  "gaia_quad_library/custom_dir")


def _store(fmt):
    """The store QSettings() opens, in that format. QSettings() uses the
    organisation domain on macOS, the name elsewhere (gui_entry.py sets both);
    neither is set yet this early."""
    import sys
    from PyQt6.QtCore import QSettings
    org = "setiastrosuite.pro" if sys.platform == "darwin" else "SetiAstro"
    return QSettings(fmt, QSettings.Scope.UserScope, org, "Seti Astro Suite Pro")


def _enter_upgrade_sandbox() -> str:
    from PyQt6.QtCore import QSettings
    from setiastro.saspro import workspace_layout as wl
    real = _store(QSettings.Format.NativeFormat)
    # Not the API keys: this copy is a plain file until SASpro exits.
    values = {k: real.value(k) for k in real.allKeys() if not k.lower().startswith("api/")}
    folder = enter_firstrun_sandbox()
    copy = _store(QSettings.Format.IniFormat)
    for k, v in values.items():
        copy.setValue(k, v)
    # As 1.x left them: nothing of 2.0 has run yet.
    for k in (wl.GENERATION_KEY, wl.PENDING_KEY, wl.SAVED_GROUP,
              "ui/welcome_shown_v2", "setup"):
        copy.remove(k)
    copy.sync()
    os.environ[wl.FORCE_ENV] = "1"
    print(f"[upgraderun] copied {len(values)} settings into the sandbox.", flush=True)
    return folder


def enter_firstrun_sandbox() -> str:
    """Point QSettings() at an empty temp folder. Idempotent; returns the folder.
    Must run before anything constructs a QSettings."""
    global _dir
    if _dir:
        return _dir
    from PyQt6.QtCore import QSettings
    real = _store(QSettings.Format.NativeFormat)
    gaia = {k: real.value(k) for k in _GAIA_DIR_KEYS if real.contains(k)}
    _dir = tempfile.mkdtemp(prefix="saspro-firstrun-")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _dir)
    if gaia:
        box = _store(QSettings.Format.IniFormat)
        for k, v in gaia.items():
            box.setValue(k, v)
        box.sync()
    os.environ[_ENV] = _dir
    atexit.register(shutil.rmtree, _dir, True)
    print(f"[firstrun] first-run sandbox: settings go to {_dir} and are discarded "
          f"on exit; your real settings are not changed.", flush=True)
    return _dir


def active() -> str | None:
    """The sandbox folder when running with --firstrun or --upgraderun, else None."""
    return _dir


def named_settings(organization: str, application: str):
    """QSettings(organization, application) that follows the sandbox.

    Same store as QSettings(organization, application) in a normal run --
    QSettings.defaultFormat() is the native format unless the sandbox changed
    it -- and inside the sandbox under --firstrun."""
    from PyQt6.QtCore import QSettings
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope,
                     organization, application)
