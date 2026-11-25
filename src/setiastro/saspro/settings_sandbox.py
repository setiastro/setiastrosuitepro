# src/setiastro/saspro/settings_sandbox.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
--firstrun: run SASpro against an empty, throwaway settings store, so it comes
up exactly as a brand-new install does -- default layout, default background,
first-run dialog -- without touching the real settings.

    setiastrosuitepro --firstrun
    python -m setiastro.saspro --firstrun
    SetiAstroSuitePro.exe --firstrun

Either can be followed by a language, to see a first run in it:

    setiastrosuitepro --firstrun spanish
    setiastrosuitepro --firstrun es
    setiastrosuitepro --upgraderun deutsch

(a code, the English name or the language's own name; see i18n.py). The word
is taken off the command line, so it is not mistaken for a file to open.

--firstrun --recent: a new install, but with the real recent images and
recent projects carried in, and the start panel's thumbnails of those images,
to see the start panel with a full list. Nothing else comes across. On its
own, without --firstrun, --recent does nothing.

    setiastrosuitepro --firstrun --recent

A restart from inside SASpro (a language change, a finished install) carries
on in the same sandbox: the new SASpro is started with --firstrun and
--sandbox-resume=<folder>, and the folder is deleted when that one exits.

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
RECENT_FLAGS = ("--recent", "--recents")      # with --firstrun only
_ENV = "SASPRO_SETTINGS_SANDBOX"
_RESUME = "--sandbox-resume="
_PREFIX = "saspro-firstrun-"
_dir: str | None = None
_handed_over = False          # a restarted SASpro took the folder over


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
    resumed = _resume(argv)
    if resumed:
        return resumed
    if _has_flag(argv, UPGRADERUN_FLAGS):
        folder = _enter_upgrade_sandbox()
    else:
        folder = enter_firstrun_sandbox()
        if _has_flag(argv, RECENT_FLAGS):
            _carry_recents(folder)
    _start_in_language(argv)
    return folder


# --firstrun spanish. Codes and each language's own name come from
# i18n.AVAILABLE_LANGUAGES; these are the English names, also used for the
# console messages (ASCII, so a frozen build's console can always print them).
_ENGLISH_NAMES = {
    "en": "English", "it": "Italian", "fr": "French", "es": "Spanish",
    "zh": "Chinese", "de": "German", "pt": "Portuguese", "ja": "Japanese",
    "hi": "Hindi", "sw": "Swahili", "uk": "Ukrainian", "ru": "Russian",
    "ar": "Arabic",
}


def _language_word(argv) -> int | None:
    """Where the word after --firstrun / --upgraderun is, when there is one
    and it is not a file to open (or another flag). --recent may come
    between them: --firstrun --recent spanish."""
    args = [str(a) for a in (argv or [])]
    for i, a in enumerate(args):
        if a.strip().lower() in FIRSTRUN_FLAGS + UPGRADERUN_FLAGS:
            j = i + 1
            while j < len(args) and args[j].strip().lower() in RECENT_FLAGS:
                j += 1
            if j == len(args):
                return None
            word = args[j].strip()
            if word and not word.startswith("-") and not os.path.exists(word.strip('"').strip("'")):
                return j
            return None
    return None


def without_language_word(argv) -> list:
    """The command line without the language after --firstrun, for the code
    that opens the files named on it."""
    args = list(argv or [])
    i = _language_word(args)
    if i is not None:
        del args[i]
    return args


def _fold(text: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text).strip().lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def language_code(word: str) -> str | None:
    """"spanish", "es", "Español" or "espanol" -> "es"; None if SASpro does not
    have that language."""
    from setiastro.saspro.i18n import AVAILABLE_LANGUAGES
    w = _fold(word)
    for code, own in AVAILABLE_LANGUAGES.items():
        if w in (code, _fold(own), _fold(_ENGLISH_NAMES.get(code, code))):
            return code
    return None


def _start_in_language(argv) -> None:
    """Save the language named after the flag into the sandbox, where the
    startup reads it before any window exists. Never stops a launch."""
    try:
        i = _language_word(argv)
        if i is None:
            return
        word = str(list(argv)[i])
        code = language_code(word)
        if code is None:
            from setiastro.saspro.i18n import AVAILABLE_LANGUAGES
            names = ", ".join(_ENGLISH_NAMES.get(c, c) for c in AVAILABLE_LANGUAGES)
            print(f"[firstrun] {ascii(word)} is not a language SASpro has, so this "
                  f"run starts in English. Languages: {names}.", flush=True)
            return
        from setiastro.saspro.i18n import save_language
        save_language(code)
        print(f"[firstrun] starting in {_ENGLISH_NAMES.get(code, code)} ({code}).", flush=True)
    except Exception as e:
        print(f"[firstrun] could not set the language: {e!r}", flush=True)


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
    _dir = tempfile.mkdtemp(prefix=_PREFIX)
    _use_folder(_dir)
    if gaia:
        box = _store(QSettings.Format.IniFormat)
        for k, v in gaia.items():
            box.setValue(k, v)
        box.sync()
    print(f"[firstrun] first-run sandbox: settings go to {_dir} and are discarded "
          f"on exit; your real settings are not changed.", flush=True)
    return _dir


_RECENT_KEYS = ("recent/image_paths", "recent/project_paths")   # file_mixin._load_recent_lists


def _carry_recents(folder: str) -> None:
    """--firstrun --recent: copy the real recent images and projects into the
    sandbox, with the start panel's thumbnails of them, which are made and kept
    for the recent images only. The real ones are only read. Never stops a
    launch: whatever does not copy just is not there."""
    try:
        from PyQt6.QtCore import QSettings, QStandardPaths
        real = _store(QSettings.Format.NativeFormat)
        box = _store(QSettings.Format.IniFormat)
        counts = []
        for k in _RECENT_KEYS:
            v = real.value(k) if real.contains(k) else None
            if v:
                box.setValue(k, v)
            counts.append(len(v) if isinstance(v, list) else int(bool(v)))
        box.sync()
        # start_panel.thumbnails_dir(), outside the sandbox and inside it. Copied
        # with their times, so the panel sees them as up to date; any missing
        # are made from the images, as on a real start.
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
        src = os.path.join(base, "SASpro", "recent_thumbnails")
        dst = os.path.join(folder, "recent_thumbnails")
        thumbs = 0
        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for n in os.listdir(src):
                if n.lower().endswith(".jpg"):
                    try:
                        shutil.copy2(os.path.join(src, n), os.path.join(dst, n))
                        thumbs += 1
                    except OSError:
                        pass
        print(f"[firstrun] --recent: carried in {counts[0]} recent images, {counts[1]} "
              f"recent projects and {thumbs} thumbnails.", flush=True)
    except Exception as e:
        print(f"[firstrun] --recent: could not carry the recent lists in: {e!r}", flush=True)


def _use_folder(folder: str) -> None:
    from PyQt6.QtCore import QSettings
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, folder)
    os.environ[_ENV] = folder
    atexit.register(_delete_folder)


def _delete_folder() -> None:
    if _dir and not _handed_over:
        shutil.rmtree(_dir, True)


def _resume(argv) -> str | None:
    """--sandbox-resume=<folder>: carry on in the sandbox a restart came from.
    Only a sandbox folder in the temp folder is taken, because this process
    deletes it on exit; anything else starts a new sandbox."""
    global _dir
    for a in (argv or []):
        a = str(a).strip()
        if not a.lower().startswith(_RESUME):
            continue
        folder = os.path.realpath(a[len(_RESUME):].strip('"').strip("'"))
        ours = (os.path.dirname(folder) == os.path.realpath(tempfile.gettempdir())
                and os.path.basename(folder).startswith(_PREFIX)
                and os.path.isdir(folder))
        if not ours:
            print(f"[firstrun] cannot carry on in {ascii(folder)}; starting a new sandbox.",
                  flush=True)
            return None
        _dir = folder
        _use_folder(folder)
        print(f"[firstrun] restarted in the same sandbox: {folder}", flush=True)
        return folder
    return None


def restart_args() -> list:
    """What a restart started from inside the sandbox adds to its command
    line, to carry on in the same settings. [] outside the sandbox."""
    return [FIRSTRUN_FLAGS[0], _RESUME + _dir] if _dir else []


def handed_over() -> None:
    """The restarted SASpro started: leave the folder for it to delete."""
    global _handed_over
    if _dir:
        _handed_over = True


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
