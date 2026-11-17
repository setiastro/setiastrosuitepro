#!/usr/bin/env python3
r"""
worker_env.py — decide, once, whether multi-process pools are usable in THIS
runtime, and give a clear reason when they aren't.

Why this exists
---------------
SASpro's measurement / ABE / registration stages spawn worker processes. Each
spawned child re-imports the scientific stack (numpy, scipy, cv2, astropy,
numba). If the running environment is inconsistent — e.g. a NumPy 2.x with an
old SciPy that still references np.long at import, or two environments bleeding
together on Windows (installer runtime vs a dev .venv) — every worker dies on
import with:

    AttributeError: module 'numpy' has no attribute 'long'
    ... -> concurrent.futures.process.BrokenProcessPool

which cascades to "0 succeeded -> no frames to stack; aborting." The app then
looks completely broken to the user, when in fact only the *workers* can't
import — the main process is fine and could do the work single-threaded.

This module runs a one-shot preflight (spawn a single worker, import the stack)
and caches the answer. Every pool call-site should ask process_pools_ok()
first and fall back to its in-process / threaded path when it's False, so a
broken environment yields a slow-but-working run plus one actionable message
instead of a dead app.

It is import-light and PyQt-free so it is safe to import in spawned children.
"""
from __future__ import annotations

import os
import sys

# tri-state cache: None = not checked yet, True/False = decided
_PP_OK: bool | None = None
_PP_REASON: str = ""
_PP_VERSIONS: dict | None = None


def _env_disabled() -> bool:
    v = os.environ.get("SASPRO_NO_PROCESS_POOLS", "").strip().lower()
    return v not in ("", "0", "false", "no", "off")


# --- runs INSIDE a spawned child: import exactly what the real workers import ---
def _probe_worker():
    import numpy as _np
    import scipy            # noqa: F401
    import scipy.sparse     # noqa: F401  (this is the import that fails on np.long)
    import cv2              # noqa: F401
    import astropy.io.fits  # noqa: F401
    try:
        import numba        # noqa: F401
        _numba_v = numba.__version__
    except Exception:
        _numba_v = None
    import scipy as _sp
    return {
        "numpy": _np.__version__, "numpy_file": _np.__file__,
        "scipy": _sp.__version__, "scipy_file": _sp.__file__,
        "numba": _numba_v,
        "executable": sys.executable,
    }


def check_process_pools(timeout: float = 40.0, log_fn=None, force: bool | None = None) -> bool:
    """Decide whether process pools work. Cached after the first call.

    Spawns ONE worker and has it import the scientific stack the real workers
    use. Returns True if the worker imports cleanly, else False. On False, logs
    (via log_fn if given) a message that names the exception and the fix, using
    the *actual* interpreter path so the command is copy-pasteable.
    """
    global _PP_OK, _PP_REASON, _PP_VERSIONS
    if force is not None:
        _PP_OK = bool(force); _PP_REASON = "forced"; return _PP_OK
    if _PP_OK is not None:
        return _PP_OK

    if _env_disabled():
        _PP_OK = False
        _PP_REASON = "disabled via SASPRO_NO_PROCESS_POOLS"
        if log_fn:
            log_fn("Process pools disabled by SASPRO_NO_PROCESS_POOLS — running single-process.")
        return _PP_OK

    try:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = _mp.get_context("spawn")  # matches how the real pools spawn (Qt-safe)
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
            _PP_VERSIONS = ex.submit(_probe_worker).result(timeout=timeout)
        _PP_OK = True
        _PP_REASON = "ok"
        if log_fn:
            v = _PP_VERSIONS
            log_fn(f"Worker preflight OK (numpy {v['numpy']}, scipy {v['scipy']}, "
                   f"numba {v['numba']}).")
    except Exception as e:
        _PP_OK = False
        _PP_REASON = f"{type(e).__name__}: {e}"
        if log_fn:
            log_fn(
                "\u26A0\uFE0F Multi-process workers unavailable "
                f"({type(e).__name__}: {e}). Running single-process (slower). "
                "This is almost always a NumPy/SciPy version mismatch or two mixed "
                "Python environments. Reinstall the locked stack with:\n"
                f"    {fix_command()}"
            )
    return _PP_OK


def process_pools_ok() -> bool:
    """Cached result. Optimistic (True) until check_process_pools() has run, so
    callers that forget to preflight still behave as before."""
    return True if _PP_OK is None else bool(_PP_OK)


def process_pools_reason() -> str:
    return _PP_REASON


def worker_versions() -> dict | None:
    return _PP_VERSIONS


def _requirements_file() -> str | None:
    """requirements.txt of a source checkout (src/setiastro/saspro -> repo root),
    or None for a pip / pipx / installer install, which has no such file."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, "..", "..", "..", "requirements.txt"))
        return cand if os.path.isfile(cand) else None
    except Exception:
        return None


def fix_command() -> str:
    """A copy-pasteable reinstall command that works for THIS install.

    A bare `-r requirements.txt` only works from a source checkout's root; a
    PyPI user has no such file and pip just errors. Give the absolute path when
    the file exists, otherwise reinstall numpy + scipy into this interpreter.
    """
    req = _requirements_file()
    if req:
        return f'"{sys.executable}" -m pip install -r "{req}"'
    return f'"{sys.executable}" -m pip install --force-reinstall --no-deps numpy scipy'


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p)).rstrip("\\/")


def _own_site_roots() -> set[str]:
    """Package roots that belong to THIS interpreter: its site-packages and its
    per-user site (AppData/Roaming/Python/PythonXY/site-packages on Windows,
    ~/.local/lib/pythonX.Y/site-packages elsewhere). Spawned workers run the
    same interpreter with the same rules, so they see the same roots."""
    roots: set[str] = set()
    try:
        import site
        for p in (site.getsitepackages() or []):
            roots.add(_norm(p))
        if site.ENABLE_USER_SITE:
            roots.add(_norm(site.getusersitepackages()))
    except Exception:
        pass
    for p in (sys.prefix, getattr(sys, "base_prefix", sys.prefix)):
        try:
            roots.add(_norm(os.path.join(p, "Lib", "site-packages")))
        except Exception:
            pass
    return roots


# --- lightweight in-process sanity check for the GUI launcher (no spawn) ------
def diagnose_local_env() -> tuple[bool, str]:
    """Cheap, in-process check for the classic failure modes, for a friendly
    startup dialog. Returns (ok, message). Does NOT spawn — that's what
    check_process_pools() is for. Catches: scipy failing to import at all, and
    numpy/scipy living under different environment roots (the mixed-venv bleed).
    """
    try:
        import numpy as _np
    except Exception as e:
        return False, f"NumPy failed to import: {e}"
    _nmaj = int(getattr(_np, "__version__", "0").split(".")[0] or 0)
    _mods = {}
    for _name in ("scipy", "scipy.sparse", "astropy", "astropy.io.fits"):
        try:
            _mods[_name] = __import__(_name)
        except Exception as e:
            _pkg = _name.split(".")[0]
            return (False,
                    f"{_pkg} failed to import against NumPy {_np.__version__} — this is "
                    f"almost always a NumPy/{_pkg} version mismatch (e.g. NumPy 1.x with a "
                    f"NumPy-2 build, or vice-versa). Reinstall the locked set with:\n"
                    f"    {fix_command()}\n\n"
                    f"Details: {type(e).__name__}: {e}")

    # mixed-environment detection: numpy and scipy should share a site-packages root
    def _site_root(mod):
        p = os.path.normcase(os.path.abspath(os.path.dirname(getattr(mod, "__file__", "") or "")))
        # .../site-packages/<pkg> -> .../site-packages
        head, tail = os.path.split(p)
        return head or p
    try:
        import scipy as _sp  # already imported cleanly above; just rebind the name
        nr, sr = _site_root(_np), _site_root(_sp)
        # Different roots are only a bleed when one of them is FOREIGN to this
        # interpreter (PYTHONPATH, a .pth into another venv, ...). The stock
        # Windows Python keeps packages in two roots of its own: site-packages
        # and the per-user site that `pip install --user` (or pip's automatic
        # user fallback) writes to. Both import fine, the spawned workers see
        # both, and the import checks above already passed, so flagging that
        # layout warned on every launch with nothing wrong and no working fix.
        own = _own_site_roots()
        if nr and sr and nr != sr and not (_norm(nr) in own and _norm(sr) in own):
            _pp = os.environ.get("PYTHONPATH", "").strip()
            return (False,
                    "NumPy and SciPy are loaded from two different environments — "
                    "this bleed causes worker crashes.\n"
                    f"  numpy: {_np.__file__}\n  scipy: {_sp.__file__}\n"
                    + (f"  PYTHONPATH is set: {_pp}\n" if _pp else "")
                    + "\nRun SASpro from a single environment and reinstall:\n"
                    f"    {fix_command()}")
    except Exception:
        pass
    return True, f"env ok (numpy {_np.__version__}, scipy {_sp.__version__})"


if __name__ == "__main__":
    ok = check_process_pools(log_fn=print)
    print("process_pools_ok:", ok, "|", process_pools_reason())
    print("local diagnose:", diagnose_local_env())