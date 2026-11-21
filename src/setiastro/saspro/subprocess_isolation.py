# src/setiastro/saspro/subprocess_isolation.py
# -*- coding: utf-8 -*-
"""
Keep the frozen build's own plumbing out of the programs SAS Pro launches.

To run at all, the installer build points parts of its process at its own
install folder:

  * the PyInstaller bootloader calls SetDllDirectoryW(_internal), so Windows
    looks for a missing DLL in SAS Pro's _internal right after the program's
    own folder -- ahead of System32 and PATH;
  * PyInstaller's PyQt6 hook sets QT_PLUGIN_PATH and QML2_IMPORT_PATH to the
    bundled Qt plugins and puts _internal at the front of PATH;
  * ssl_bootstrap points SSL_CERT_FILE / REQUESTS_CA_BUNDLE at the bundled
    certifi file, config_bootstrap points MPLCONFIGDIR at <exe dir>/mpl_config,
    and the bootloader exports its _PYI_* bookkeeping variables.

Every child process inherits all of it. The DLL directory too: checked
directly, a SetDllDirectoryW made in a parent is what GetDllDirectoryW reports
in its child. So StarNet, GraXpert, the RC-Astro tools, the Cosmic Clarity
executables, the Inno updater and the runtime-venv Python resolve missing DLLs
from SAS Pro's _internal before the system, and SyQon Studio -- a Qt6 app of
its own -- searches SAS Pro's Qt 6 plugin folder before its own. Which DLL or
plugin a child ends up with then depends on what this SAS Pro release ships.

install() wraps subprocess.Popen so each child starts clean:

  * an environment entry pointing inside this install is dropped -- the
    user's own values, and everything else, pass through untouched;
  * the _PYI_* variables go, and PYINSTALLER_RESET_ENVIRONMENT=1 makes a
    frozen child (GraXpert, a second SAS Pro) start as a fresh application;
  * on Windows the inherited DLL directory is cleared for the moment the
    child is created, then restored. _internal stays registered with
    os.add_dll_directory meanwhile, so this process's own extension imports
    resolve exactly as before during that window.

Frozen builds only: pip and source runs set none of this. multiprocessing
workers are created without Popen and keep what they rely on. QProcess (the
ASTAP launches) is not covered: Qt creates those processes natively.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

_INSTALLED = False
_SPAWN_LOCK = threading.Lock()   # the DLL directory is process-wide; one child at a time
_DLL_DIR_COOKIE = None           # keeps _internal a user DLL directory while the inherited one is cleared
_K32 = None


def _bundle_roots() -> list[str]:
    """Normalised folders that belong to this frozen install."""
    roots: list[str] = []
    mei = getattr(sys, "_MEIPASS", None)
    if mei:
        roots += [mei, os.path.realpath(mei)]
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    # Only folders that are unambiguously ours. The exe folder itself is not one
    # of them: a portable copy unzipped into Downloads would make every PATH
    # entry under Downloads look like part of the install.
    roots.append(os.path.join(exe_dir, "mpl_config"))
    if sys.platform == "darwin" and os.path.basename(exe_dir) == "MacOS":
        # .app bundle: _MEIPASS is Contents/Frameworks, but the Qt hook also
        # points QML2_IMPORT_PATH into Contents/Resources.
        roots.append(os.path.dirname(exe_dir))
    out = []
    for r in roots:
        n = os.path.normcase(os.path.normpath(r))
        if n not in out:
            out.append(n)
    return out


def _inside(entry: str, roots: list[str]) -> bool:
    s = entry.strip().strip('"')
    if not s or not os.path.isabs(s):
        return False
    n = os.path.normcase(os.path.normpath(s))
    return any(n == r or n.startswith(r + os.sep) for r in roots)


def child_env(env) -> dict:
    """A copy of `env` with this frozen install's own entries taken out.

    Every value is read as an os.pathsep list; entries that resolve inside the
    install are removed and the rest is rejoined unchanged. A value that loses
    all its entries is dropped. Values with nothing of ours are left byte for
    byte as they were. Returns a plain copy when not frozen.
    """
    out = dict(env)
    if not getattr(sys, "frozen", False):
        return out
    roots = _bundle_roots()
    for key, val in list(out.items()):
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        if key.upper().startswith("_PYI_"):
            del out[key]
            continue
        if not val:
            continue
        parts = val.split(os.pathsep)
        kept = [p for p in parts if not _inside(p, roots)]
        if len(kept) == len(parts):
            continue
        if any(p.strip() for p in kept):
            out[key] = os.pathsep.join(kept)
        else:
            del out[key]
    out["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return out


def _kernel32():
    # A private handle, so the argtypes set here never leak into the shared
    # ctypes.windll.kernel32 other modules use.
    global _K32
    if _K32 is None:
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        k.GetDllDirectoryW.restype = wintypes.DWORD
        k.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
        k.SetDllDirectoryW.restype = wintypes.BOOL
        _K32 = k
    return _K32


@contextmanager
def _no_inherited_dll_directory():
    """Clear SetDllDirectoryW for the duration of a CreateProcess, then restore it."""
    with _SPAWN_LOCK:
        saved = None
        try:
            import ctypes
            k = _kernel32()
            buf = ctypes.create_unicode_buffer(32768)
            if k.GetDllDirectoryW(len(buf), buf):
                saved = buf.value or None
            if saved:
                k.SetDllDirectoryW(None)
        except Exception:
            saved = None
        try:
            yield
        finally:
            if saved:
                try:
                    _kernel32().SetDllDirectoryW(saved)
                except Exception:
                    pass


def install() -> bool:
    """Wrap subprocess.Popen for the frozen build. Idempotent, never raises.

    Returns True when the wrapper is in place.
    """
    global _INSTALLED, _DLL_DIR_COOKIE
    if _INSTALLED:
        return True
    if not getattr(sys, "frozen", False):
        return False
    try:
        import subprocess

        orig = subprocess.Popen.__init__
        if getattr(orig, "_saspro_isolated", False):
            _INSTALLED = True
            return True

        is_windows = os.name == "nt"
        if is_windows:
            mei = getattr(sys, "_MEIPASS", None)
            if mei and os.path.isdir(mei):
                try:
                    _DLL_DIR_COOKIE = os.add_dll_directory(mei)
                except Exception:
                    pass

        def __init__(self, *args, **kwargs):
            # env is Popen's 11th parameter. It is passed by keyword in
            # practice; a caller that passed it positionally keeps its own.
            if len(args) <= 10:
                base = kwargs.get("env")
                kwargs["env"] = child_env(os.environ if base is None else base)
            if is_windows:
                with _no_inherited_dll_directory():
                    orig(self, *args, **kwargs)
            else:
                orig(self, *args, **kwargs)

        __init__.__wrapped__ = orig
        __init__.__doc__ = orig.__doc__
        __init__._saspro_isolated = True
        subprocess.Popen.__init__ = __init__
        _INSTALLED = True
        return True
    except Exception:
        return False


def describe() -> str:
    """One line for a startup log or a diagnostics panel."""
    if _INSTALLED:
        return "Child processes: started without the bundle's DLL directory and environment"
    if getattr(sys, "frozen", False):
        return "Child processes: isolation NOT installed (inherit the bundle's environment)"
    return "Child processes: not a frozen build, nothing to isolate"
