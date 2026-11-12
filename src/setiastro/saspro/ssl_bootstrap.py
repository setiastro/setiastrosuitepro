# src/setiastro/saspro/ssl_bootstrap.py
# -*- coding: utf-8 -*-
"""
Point OpenSSL at a CA bundle that actually exists.

macOS is the case this is here for. The python.org installers do NOT use the
system keychain: they ship an "Install Certificates.command" that copies
certifi's bundle into the interpreter, and most people never run it. Until
someone does, the interpreter has no trust store at all and EVERY https call
made through urllib fails with

    [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    unable to get local issuer certificate

which is what the Gaia database downloads were hitting. The tell that it is
this and not a network problem: `requests` and `astroquery` work on the very
same machine, because they default to certifi's bundle rather than asking
OpenSSL for a default. The two libraries disagree, so the symptom lands on
whichever half of SASpro uses urllib -- the Gaia downloads, the survey tile
fetches, the minor-body manifest.

Setting SSL_CERT_FILE is a FIX, not a weakening. It gives verification a
trust store to verify against where it currently has none; nothing is
disabled and no check is skipped. On Windows it is additive rather than a
replacement -- `load_default_certs()` loads the Windows store AND the
default verify paths into the same X509_STORE, measured here at 72 roots
before and 180 after -- so a corporate root that only exists in the Windows
store keeps working.

Both variables are set with "only if unset", so an administrator who has
already pointed them at a corporate bundle keeps it.

This module must not import `ssl`, `certifi` aside. It runs early in
gui_entry, where the Windows VC++ preflight has to come before any
C-extension import, and `ssl` pulls in `_ssl`. Setting the environment is
enough: OpenSSL reads these when a context is first created, which happens
later and elsewhere.
"""

from __future__ import annotations

import os

# "" = ran, nothing to do. A path = that bundle was installed into the env.
_RESULT: str | None = None


def apply() -> str:
    """Install certifi's CA bundle as the OpenSSL default, if needed.

    Idempotent and never raises: a trust-store bootstrap must not be the
    reason the app fails to start.
    """
    global _RESULT
    if _RESULT is not None:
        return _RESULT
    _RESULT = ""

    try:
        import certifi
        where = certifi.where()
    except Exception:
        # certifi arrives with requests, which is a hard dependency, so this
        # should not happen -- but a missing bundle is not worth a traceback
        # on startup.
        return _RESULT

    # A frozen build that failed to collect cacert.pem would otherwise point
    # OpenSSL at a path that does not exist, turning a working trust store
    # into a broken one. Verify before pointing anything at it.
    try:
        if not where or not os.path.isfile(where):
            return _RESULT
    except Exception:
        return _RESULT

    changed = []
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if not os.environ.get(var):
            os.environ[var] = where
            changed.append(var)

    _RESULT = where if changed else ""
    return _RESULT


def describe() -> str:
    """One line for a startup log or a diagnostics panel."""
    if _RESULT is None:
        return "TLS trust store: not checked yet"
    if not _RESULT:
        return ("TLS trust store: using the system default "
                "(SSL_CERT_FILE already set, or certifi unavailable)")
    return f"TLS trust store: certifi bundle installed -> {_RESULT}"
