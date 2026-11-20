# src/setiastro/saspro/settings_sandbox.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
--firstrun: run SASpro against an empty, throwaway settings store, so it comes
up exactly as a brand-new install does -- default layout, default background,
first-run dialog -- without touching the real settings.

    setiastrosuitepro --firstrun
    python -m setiastro.saspro --firstrun
    SetiAstroSuitePro.exe --firstrun

How: every default-constructed QSettings() follows QSettings.defaultFormat(),
so pointing the INI format at a fresh temp folder before anything reads a
setting redirects all of them. The folder is deleted when SASpro exits, so
each --firstrun launch starts from nothing again. --firstrun also implies
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
_ENV = "SASPRO_SETTINGS_SANDBOX"
_dir: str | None = None


def wants_firstrun(argv) -> bool:
    return any(str(a).strip().lower() in FIRSTRUN_FLAGS for a in (argv or []))


def enter_firstrun_sandbox() -> str:
    """Point QSettings() at an empty temp folder. Idempotent; returns the folder.
    Must run before anything constructs a QSettings."""
    global _dir
    if _dir:
        return _dir
    from PyQt6.QtCore import QSettings
    _dir = tempfile.mkdtemp(prefix="saspro-firstrun-")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _dir)
    os.environ[_ENV] = _dir
    atexit.register(shutil.rmtree, _dir, True)
    print(f"[firstrun] first-run sandbox: settings go to {_dir} and are discarded "
          f"on exit; your real settings are not read or changed.", flush=True)
    return _dir


def active() -> str | None:
    """The sandbox folder when running with --firstrun, else None."""
    return _dir


def named_settings(organization: str, application: str):
    """QSettings(organization, application) that follows the sandbox.

    Same store as QSettings(organization, application) in a normal run --
    QSettings.defaultFormat() is the native format unless the sandbox changed
    it -- and inside the sandbox under --firstrun."""
    from PyQt6.QtCore import QSettings
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope,
                     organization, application)
