# src/setiastro/saspro/cds_health.py
"""
Fast reachability probing, mirror failover and circuit-breaking for CDS
services (SIMBAD / VizieR / Sesame).

WHY THIS EXISTS
───────────────
SIMBAD is *optional* for SFCC and SSSC. Both tools derive their actual
calibration from Gaia XP spectra held in the local library; SIMBAD only
contributes spectral-type enrichment and Pickles template matching for the
subset of stars Gaia doesn't cover. When CDS goes down, the correct
behaviour is to notice within a couple of seconds and continue without it.

What happened instead: the old code issued ~15 unbounded network calls per
run (10 `add_votable_fields` validation round-trips + 5 `query_region`
attempts), each falling back to the OS TCP connect timeout (~130 s on
Linux). Thirty-plus minutes of frozen UI before the inevitable fallback.

Four things fix that, and this module provides all four:

  1. A preflight probe with a HARD deadline, so we learn CDS is down in
     ~4 s rather than ~130 s. The deadline is enforced by joining a daemon
     thread, because urlopen's `timeout` does not cover DNS resolution —
     a hung resolver can block well past it.

  2. Mirror failover that is actually populated, and verified end-to-end
     rather than by a shallow liveness check (see MIRRORS below).

  3. A session-wide circuit breaker. The first genuine connection failure
     trips it; every later SIMBAD call in that session returns instantly.
     This is what rescues the batched `query_objects` loops, which would
     otherwise rediscover the outage once per 200-star batch.

  4. A user-visible override, so a user whose network can reach one mirror
     and not the other can pin the working one without waiting for a
     release. Settings → CDS/SIMBAD, or set_server_preference() directly.

Nothing here raises on its own. Callers get None/False and carry on.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional, Tuple

__all__ = [
    "MIRRORS",
    "VIZIER_MIRRORS",
    "canonical_server",
    "probe",
    "simbad_available",
    "current_server",
    "vizier_server",
    "set_server_preference",
    "server_preference",
    "probe_details",
    "note_failure",
    "note_success",
    "is_connection_error",
    "trip",
    "reset",
    "status_text",
    "status_detail",
    "prepare_simbad",
    "call_simbad",
    "Budget",
]

# ── Configuration ────────────────────────────────────────────────────────

# Order matters: first entry is preferred. These are exactly the two values
# astroquery's `simbad.conf.server` accepts — it is a ConfigItem built from
# `servers_list`, so assigning anything else raises and (because every
# assignment here is individually guarded) would silently leave astroquery
# pointed at the default. Verified against astroquery 0.4.10/0.4.11/0.4.12:
#
#     servers_list = ['simbad.cds.unistra.fr', 'simbad.harvard.edu']
#
# WHY HARVARD IS BACK
# -------------------
# An earlier version of this module listed only Strasbourg, on the strength
# of a probe against `simbad.cfa.harvard.edu` that answered on :443 but
# returned HTTP 500 with HTML error output from vizier.cfa.harvard.edu for
# /simbad/sim-tap/*. Two things were wrong with that conclusion:
#
#   * `simbad.cfa.harvard.edu` is the web-UI alias. It is the same machine
#     as `simbad.harvard.edu`, but it is NOT a name astroquery will accept
#     for `conf.server`, and its sim-tap path is not the one astroquery
#     builds. The name to probe and to hand astroquery is
#     `simbad.harvard.edu`.
#   * A shallow "did something respond?" check cannot tell the two apart,
#     which is how a broken endpoint looked healthy in the first place.
#
# So the mirror is listed again AND the probe is now end-to-end: a mirror
# is only declared usable once a real TAP query has returned a VOTable from
# it (see _verify_tap). That is the check that should gate any future
# addition to this list — not reachability.
MIRRORS: tuple[str, ...] = (
    "simbad.cds.unistra.fr",
    "simbad.harvard.edu",
)

# Names users and web links use that astroquery will not accept. Mapped so
# a pasted hostname out of a browser still works as a preference.
_SERVER_ALIASES: dict[str, str] = {
    "simbad.cfa.harvard.edu": "simbad.harvard.edu",
    "simbad.u-strasbg.fr":    "simbad.cds.unistra.fr",   # retired by CDS
    "simbad.unistra.fr":      "simbad.cds.unistra.fr",
}

# VizieR is separate infrastructure from the SIMBAD TAP endpoint, but the
# primary is still a Strasbourg host, so a CDS-wide outage takes it down
# too. These are values astroquery's vizier.conf.server accepts.
VIZIER_MIRRORS: tuple[str, ...] = (
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
    "vizier.hia.nrc.ca",
    "vizier.nao.ac.jp",
)

# VOSI standard availability endpoint — tiny XML payload, no query cost.
# Stage 1 only: necessary, not sufficient.
_PROBE_PATH = "/simbad/sim-tap/availability"

# Stage 2: the cheapest real TAP query there is. A mirror that answers this
# with a VOTable can serve query_region/query_object; one that answers the
# availability endpoint but not this cannot, and is exactly the failure
# mode that got Harvard struck off the list before.
_TAP_PATH = "/simbad/sim-tap/sync"
_TAP_SMOKE_QUERY = "SELECT TOP 1 main_id FROM basic"

PROBE_TIMEOUT = 4.0        # per-mirror, per-stage, seconds
VERIFY_TIMEOUT = 8.0       # TAP smoke query gets a little more room
TTL_UP        = 180.0      # re-probe a healthy service this often
TTL_DOWN      = 45.0       # re-probe a dead service this often
TTL_VERIFIED  = 900.0      # how long an end-to-end TAP verification stands
BREAKER_RESET = 120.0      # breaker self-heals after this long
REHOME_AFTER  = 600.0      # while on a fallback, re-try the primary this often

# How long the whole SIMBAD phase of one run may take before we give up
# and continue with Gaia XP only.
DEFAULT_BUDGET = 45.0

_UA = "SetiAstroSuitePro/cds-health (+https://setiastro.com)"

# QSettings keys, read lazily so this module stays importable without Qt.
_QS_SERVER = "CDS/simbad_server"     # "" = automatic, else a hostname
_QS_PIN    = "CDS/pin_simbad_server" # bool: don't fail over off it
_QS_VIZIER = "CDS/vizier_server"     # "" = automatic, else a hostname

# ── State ────────────────────────────────────────────────────────────────

_lock = threading.RLock()
_state: dict = {
    "server":      MIRRORS[0],   # last mirror known good (or default)
    "ok":          None,         # None = never probed
    "checked_at":  0.0,
    "detail":      "",
    "details":     [],           # per-mirror detail from the last full probe
    "verified_at": {},           # server -> monotonic time of last TAP verify
    "homed_at":    0.0,          # last time the primary was tried first
    "tripped":     False,
    "tripped_at":  0.0,
    "trip_reason": "",
    "pref":        None,         # user preference: None = unread, "" = auto
    "pin":         False,
    "vizier_pref": None,
}


def _now() -> float:
    return time.monotonic()


def canonical_server(name: str) -> str:
    """Map a user-typed or web-UI hostname onto the name astroquery takes."""
    n = (name or "").strip().lower()
    n = n.removeprefix("https://").removeprefix("http://").rstrip("/")
    n = n.split("/", 1)[0]
    return _SERVER_ALIASES.get(n, n)


# ── User preference ──────────────────────────────────────────────────────

def _load_preference() -> None:
    """Read the saved preference from QSettings once, if Qt is present."""
    with _lock:
        if _state["pref"] is not None:
            return
        _state["pref"] = ""
        _state["vizier_pref"] = ""
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings()
        pref = canonical_server(s.value(_QS_SERVER, "", type=str) or "")
        pin  = bool(s.value(_QS_PIN, False, type=bool))
        vpref = canonical_server(s.value(_QS_VIZIER, "", type=str) or "")
    except Exception:                                # noqa: BLE001
        return
    with _lock:
        if pref in MIRRORS:
            _state["pref"] = pref
            _state["server"] = pref
            _state["pin"] = pin
        if vpref in VIZIER_MIRRORS:
            _state["vizier_pref"] = vpref


def set_server_preference(server: str = "", *, pin: bool = False,
                          vizier: Optional[str] = None,
                          persist: bool = True) -> str:
    """
    Choose which SIMBAD mirror to try first.

    `server=""` restores automatic order (primary first, mirrors on
    failure). `pin=True` means never fail over off the chosen mirror — for
    a user whose network reaches one host and not the other, failing over
    just costs them a timeout per lookup.

    Returns the canonical server name actually stored ("" for automatic).
    Clears cached health state, so the next call re-probes.
    """
    srv = canonical_server(server)
    if srv and srv not in MIRRORS:
        srv = ""
    viz = canonical_server(vizier or "")
    if viz and viz not in VIZIER_MIRRORS:
        viz = ""

    with _lock:
        _state["pref"] = srv
        _state["pin"] = bool(pin) and bool(srv)
        if vizier is not None:
            _state["vizier_pref"] = viz
        _state.update(server=srv or MIRRORS[0], ok=None, checked_at=0.0,
                      detail="", details=[], verified_at={}, homed_at=0.0,
                      tripped=False, tripped_at=0.0, trip_reason="")

    if persist:
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings()
            s.setValue(_QS_SERVER, srv)
            s.setValue(_QS_PIN, bool(pin) and bool(srv))
            if vizier is not None:
                s.setValue(_QS_VIZIER, viz)
        except Exception:                            # noqa: BLE001
            pass
    return srv


def server_preference() -> Tuple[str, bool]:
    """Return (preferred_server_or_empty, pinned)."""
    _load_preference()
    with _lock:
        return (_state["pref"] or ""), bool(_state["pin"])


def vizier_server() -> str:
    """The VizieR host to use — user preference, else the primary."""
    _load_preference()
    with _lock:
        return _state["vizier_pref"] or VIZIER_MIRRORS[0]


# ── Low-level probe ──────────────────────────────────────────────────────

def _fetch_with_hard_deadline(url: str, timeout: float):
    """
    GET `url`, guaranteeing return within roughly `timeout` seconds.

    urlopen's own timeout covers connect and read but NOT name resolution.
    A wedged resolver can block for a minute or more regardless of what we
    pass. Running the request on a daemon thread and joining with a
    deadline gives us a real upper bound; if the thread is still blocked we
    abandon it (it is a daemon, so it cannot keep the process alive).

    Returns (code, body, error) with code/body None on failure.
    """
    box: dict = {}

    def _work():
        try:
            req = urllib.request.Request(
                url, method="GET", headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                box["code"] = int(resp.getcode() or 0)
                box["body"] = resp.read(8192).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            # The host answered — that is what we care about.
            box["code"] = int(getattr(e, "code", 0) or 0)
            try:
                box["body"] = e.read(8192).decode("utf-8", "replace")
            except Exception:                        # noqa: BLE001
                box["body"] = ""
        except Exception as e:                       # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=_work, daemon=True,
                         name="cds-health-probe")
    t.start()
    t.join(timeout + 1.0)

    if t.is_alive():
        return None, None, TimeoutError(
            f"no response within {timeout:.0f}s (connection still blocked)")
    if "err" in box:
        return None, None, box["err"]
    return box.get("code"), box.get("body", ""), None


def _strip_ns(body: str) -> str:
    """
    Drop XML namespace prefixes so tag tests match either spelling.

    SIMBAD serves `<vosi:available>true</vosi:available>`; the old test
    looked for `<available>` and never matched, so a maintenance window
    advertising itself as unavailable fell through to the permissive
    "2xx-4xx is good enough" branch.
    """
    out = body.replace(" ", "")
    for prefix in ("vosi:", "vod:", "xml:", "ns0:", "ns1:"):
        out = out.replace(f"<{prefix}", "<").replace(f"</{prefix}", "</")
    return out


def _probe_one(server: str, timeout: float) -> Tuple[bool, str]:
    """Stage 1: is this mirror's TAP service answering? (ok, detail)."""
    url = f"https://{server}{_PROBE_PATH}"
    code, body, err = _fetch_with_hard_deadline(url, timeout)

    if err is not None:
        return False, f"{server}: {type(err).__name__}: {err}"

    if code is None:
        return False, f"{server}: no response"

    low = (body or "").lower()
    flat = _strip_ns(low)

    # A 5xx from the TAP service means the service itself is broken, even
    # though the host answered.
    if code >= 500:
        return False, f"{server}: HTTP {code} from TAP service"

    # VOSI availability is an XML document. An HTML body here means we are
    # talking to a web front end or an error page, not a TAP endpoint.
    if "<html" in low or "<!doctypehtml" in flat:
        return False, f"{server}: HTTP {code} but served HTML, not VOSI XML"

    if "<available>false</available>" in flat:
        return False, f"{server}: reports itself unavailable (maintenance)"

    if "<available>true</available>" in flat:
        return True, f"{server}: VOSI reports available"

    # 2xx-4xx with no VOSI markers: older deployments don't serve
    # /availability at all, and a 404 there still beats a dead socket.
    if 200 <= code < 500:
        return True, f"{server}: HTTP {code} (no VOSI document)"

    return False, f"{server}: HTTP {code}"


def _verify_tap(server: str, timeout: float = VERIFY_TIMEOUT
                ) -> Tuple[bool, str]:
    """
    Stage 2: run the smallest possible real TAP query and insist on a
    VOTable coming back.

    This is the check that distinguishes "the host answers" from "the host
    can serve the queries we are about to make". Cached for TTL_VERIFIED
    so it costs one round-trip per mirror per 15 minutes, not per lookup.
    """
    with _lock:
        last = float(_state["verified_at"].get(server, 0.0))
    if last and (_now() - last) < TTL_VERIFIED:
        return True, f"{server}: TAP verified recently"

    qs = urllib.parse.urlencode({
        "request": "doQuery",
        "lang":    "ADQL",
        "format":  "votable",
        "maxrec":  "1",
        "query":   _TAP_SMOKE_QUERY,
    })
    code, body, err = _fetch_with_hard_deadline(
        f"https://{server}{_TAP_PATH}?{qs}", timeout)

    if err is not None:
        return False, f"{server}: TAP query {type(err).__name__}: {err}"
    if code is None:
        return False, f"{server}: TAP query got no response"

    low = (body or "").lower()
    if code >= 500:
        return False, f"{server}: TAP query returned HTTP {code}"
    if "<votable" in low:
        with _lock:
            _state["verified_at"][server] = _now()
        return True, f"{server}: TAP query returned a VOTable"
    if "<html" in low:
        return False, (f"{server}: TAP query returned HTML, not VOTable "
                       f"(HTTP {code}) — this host is not a usable TAP "
                       f"endpoint")
    return False, f"{server}: TAP query returned HTTP {code}, no VOTable"


# ── Public health API ────────────────────────────────────────────────────

def _probe_order() -> list:
    """Mirrors in the order to try, honouring preference and re-homing."""
    _load_preference()
    with _lock:
        pref = _state["pref"] or ""
        pinned = bool(_state["pin"])
        current = _state["server"] or MIRRORS[0]
        homed_age = _now() - float(_state["homed_at"] or 0.0)

    if pinned and pref:
        return [pref]
    if pref:
        return [pref] + [m for m in MIRRORS if m != pref]

    # Automatic: normally stick to the last mirror that worked, so a dead
    # primary isn't re-timed-out every TTL. But every REHOME_AFTER seconds,
    # try the primary first again — otherwise a session that failed over
    # during an outage stays on the fallback for as long as it runs, even
    # after Strasbourg comes back.
    head = MIRRORS[0] if (current != MIRRORS[0] and
                          homed_age > REHOME_AFTER) else current
    return [head] + [m for m in MIRRORS if m != head]


def probe(*, force: bool = False,
          timeout: float = PROBE_TIMEOUT,
          verify: bool = True,
          status_cb: Optional[Callable[[str], None]] = None
          ) -> Tuple[bool, str, str]:
    """
    Determine whether SIMBAD is reachable, trying each mirror in turn.

    Cached: TTL_UP when healthy, TTL_DOWN when not, so calling this before
    every SIMBAD operation is free after the first time.

    `verify=False` skips the TAP smoke query — only for a quick liveness
    display, never for choosing a mirror to query.

    Returns (ok, server, detail).
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731

    with _lock:
        age = _now() - _state["checked_at"]
        if not force and _state["ok"] is not None:
            ttl = TTL_UP if _state["ok"] else TTL_DOWN
            if age < ttl:
                return bool(_state["ok"]), _state["server"], _state["detail"]

    order = _probe_order()
    if order and order[0] == MIRRORS[0]:
        with _lock:
            _state["homed_at"] = _now()

    details: list = []
    for server in order:
        status_cb(f"Checking SIMBAD availability ({server})…")
        ok, detail = _probe_one(server, timeout)

        if ok and verify:
            ok2, detail2 = _verify_tap(server)
            detail = f"{detail}; {detail2}"
            ok = ok2

        details.append(detail)
        if ok:
            with _lock:
                _state.update(
                    server=server, ok=True, checked_at=_now(), detail=detail,
                    details=list(details), tripped=False, tripped_at=0.0,
                    trip_reason="",
                )
            if server != MIRRORS[0]:
                status_cb(f"Primary SIMBAD unreachable — using mirror {server}")
            return True, server, detail

    joined = " | ".join(details)
    with _lock:
        _state.update(ok=False, checked_at=_now(), detail=joined,
                      details=list(details),
                      tripped=True, tripped_at=_now(),
                      trip_reason="preflight probe failed")
    return False, order[0] if order else MIRRORS[0], joined


def simbad_available(status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """True if SIMBAD looks usable right now (breaker closed + probe ok)."""
    with _lock:
        if _state["tripped"]:
            if _now() - _state["tripped_at"] < BREAKER_RESET:
                return False
            # Breaker cooled off — allow one fresh probe.
            _state["tripped"] = False
    ok, _server, _detail = probe(status_cb=status_cb)
    return ok


def current_server() -> str:
    _load_preference()
    with _lock:
        return _state["server"] or MIRRORS[0]


def probe_details() -> list:
    """Per-mirror results of the last full probe, for dialogs and logs."""
    with _lock:
        return list(_state["details"])


def status_text() -> str:
    """One-line human summary, for status bars and logs."""
    with _lock:
        if _state["ok"] is None:
            return "SIMBAD: not yet checked"
        if _state["ok"] and not _state["tripped"]:
            return f"SIMBAD: available via {_state['server']}"
        # Unavailable. Prefer the concrete per-mirror detail over the
        # generic trip reason: "preflight probe failed" told the user
        # nothing they could act on, which is how a mirror-selection bug
        # looked like a dead network.
        detail = _state["detail"] or _state["trip_reason"] or "unknown"
        return f"SIMBAD: unavailable ({detail})"


def status_detail() -> str:
    """Multi-line detail — one line per mirror tried. Safe to put in a box."""
    with _lock:
        lines = list(_state["details"]) or (
            [_state["detail"]] if _state["detail"] else [])
        pref, pin = (_state["pref"] or ""), bool(_state["pin"])
    out = "\n".join(f"  • {ln}" for ln in lines if ln)
    if pref:
        out += (f"\n  • mirror preference: {pref}"
                f"{' (pinned — no failover)' if pin else ''}")
    return out or "  • no probe has run yet"


def is_connection_error(exc: BaseException) -> bool:
    """
    True when `exc` means "the host is not answering" as opposed to
    "the host answered with a problem".

    This is the retry/no-retry decision. Connection failures must NOT be
    retried in a loop — the host is down and every attempt costs a full
    timeout. Server-side errors (5xx, rate-limit) are worth one retry.
    """
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, urllib.error.URLError) and not isinstance(
            exc, urllib.error.HTTPError):
        return True

    # requests / pyvo live underneath astroquery; match by name so this
    # module does not need to import them.
    name = type(exc).__name__.lower()
    if any(k in name for k in (
            "connection", "timeout", "dns", "resolve", "unreachable")):
        return True

    text = f"{exc}".lower()
    return any(k in text for k in (
        "connection refused", "connection reset", "connection aborted",
        "failed to establish", "name or service not known",
        "temporary failure in name resolution", "timed out",
        "max retries exceeded", "network is unreachable",
        "no route to host", "getaddrinfo failed",
    ))


def trip(reason: str) -> None:
    """Open the breaker: skip SIMBAD for the rest of the session."""
    with _lock:
        _state.update(tripped=True, tripped_at=_now(), trip_reason=str(reason),
                      ok=False, checked_at=_now())


def note_failure(exc: BaseException, *, context: str = "") -> bool:
    """
    Record a SIMBAD failure. Trips the breaker for connection-class errors.
    Returns True if the breaker was tripped (caller should stop retrying).
    """
    if is_connection_error(exc):
        trip(f"{context or 'SIMBAD call'}: {type(exc).__name__}: {exc}")
        return True
    return False


def note_success() -> None:
    with _lock:
        _state.update(ok=True, checked_at=_now(), tripped=False,
                      tripped_at=0.0, trip_reason="")


def reset() -> None:
    """Forget health state — used by a manual 'retry CDS' UI action.

    Keeps the user's mirror preference; only cached health is dropped.
    """
    _load_preference()
    with _lock:
        start = _state["pref"] or MIRRORS[0]
        _state.update(server=start, ok=None, checked_at=0.0, detail="",
                      details=[], verified_at={}, homed_at=0.0,
                      tripped=False, tripped_at=0.0, trip_reason="")


# ── Wall-clock budget ────────────────────────────────────────────────────

class Budget:
    """
    A deadline for one run's whole SIMBAD phase.

    Even with per-call timeouts, a partially-degraded CDS (answering slowly
    rather than not at all) can still eat many minutes across dozens of
    batches. The budget caps the total.
    """

    __slots__ = ("_deadline", "_total")

    def __init__(self, seconds: float = DEFAULT_BUDGET):
        self._total = float(seconds)
        self._deadline = _now() + float(seconds)

    def remaining(self) -> float:
        return max(0.0, self._deadline - _now())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def spent(self) -> float:
        return self._total - self.remaining()

    def __repr__(self) -> str:
        return f"<Budget {self.remaining():.1f}s of {self._total:.1f}s left>"


# ── astroquery integration ───────────────────────────────────────────────

def apply_server(server: str, status_cb: Optional[Callable[[str], None]] = None
                 ) -> bool:
    """
    Point astroquery at `server`. Returns True only if it took.

    This is deliberately NOT a swallow-everything helper. astroquery's
    `simbad.conf.server` is a ConfigItem constructed from `servers_list`,
    so assigning a name outside that list raises — and if that raise is
    swallowed (as every assignment in the old inline code was), astroquery
    keeps querying the default host while the UI reports it is using the
    mirror. That is a failover that silently does nothing, and it is
    indistinguishable from the outage it was meant to work around.
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731

    host = canonical_server(server)
    try:
        from astroquery.simbad import Simbad
        import astroquery.simbad as _aqs
    except Exception as e:                           # noqa: BLE001
        status_cb(f"astroquery.simbad unavailable: {e}")
        return False

    allowed = tuple(getattr(_aqs.conf, "servers_list", MIRRORS) or MIRRORS)
    if host not in allowed:
        status_cb(f"astroquery will not accept SIMBAD server {host!r} "
                  f"(it accepts: {', '.join(allowed)})")
        return False

    applied = False
    try:
        _aqs.conf.server = host
        applied = True
    except Exception as e:                           # noqa: BLE001
        status_cb(f"could not set astroquery conf.server={host}: {e}")

    # The module-level `Simbad` is an *instance*; new instances read conf,
    # but an instance already in use needs setting directly.
    try:
        Simbad.server = host
        applied = True
    except Exception:                                # noqa: BLE001
        pass

    return applied


def _bind_timeouts(timeout: float) -> None:
    """
    Bound astroquery's timeouts under every spelling it has used.

    Caveat worth knowing: on astroquery >= 0.4.8 `simbad.conf.timeout` is
    the TAP *async execution* limit (default 1080 s), not a socket timeout,
    so setting it does not bound a hung connect. The hard-deadline preflight
    above is what actually bounds that. These are set anyway because on
    older versions `TIMEOUT` really was the connect timeout.
    """
    try:
        from astroquery.simbad import Simbad
        import astroquery.simbad as _aqs
    except Exception:                                # noqa: BLE001
        return
    for setter in (
        lambda: setattr(_aqs.conf, "timeout", float(timeout)),
        lambda: setattr(Simbad, "timeout", float(timeout)),
        lambda: setattr(Simbad, "TIMEOUT", float(timeout)),
    ):
        try:
            setter()
        except Exception:                            # noqa: BLE001
            pass


def prepare_simbad(status_cb: Optional[Callable[[str], None]] = None,
                   *,
                   fields: Optional[tuple] = None,
                   legacy_fields: Optional[tuple] = None,
                   timeout: float = 30.0,
                   row_limit: int = 10000):
    """
    Probe, select a mirror, bound the timeouts, and add votable fields.

    Returns the configured Simbad class, or None when SIMBAD is unusable —
    in which case the caller continues with Gaia XP only.

    Subtleties this handles that the old inline code did not:

      * The mirror actually gets applied, and a refusal is reported rather
        than swallowed (see apply_server).

      * Timeouts are set BEFORE `add_votable_fields`, not after. On
        astroquery >= 0.4.8 that call validates against the TAP schema over
        the network, so it needs the bound just as much as the query does.

      * `TIMEOUT` was renamed to `timeout` in astroquery 0.4.8. Both are
        set here, so the bound applies on either version.
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731

    if not simbad_available(status_cb=status_cb):
        status_cb("SIMBAD/CDS unreachable — continuing with Gaia XP only.")
        return None

    try:
        from astroquery.simbad import Simbad
        import astroquery.simbad as _aqs
    except Exception as e:                           # noqa: BLE001
        status_cb(f"astroquery.simbad unavailable: {e}")
        trip(f"import failed: {e}")
        return None

    server = current_server()
    if not apply_server(server, status_cb=status_cb):
        status_cb(f"Could not point astroquery at {server} — queries would "
                  f"go to the default host instead. Skipping SIMBAD.")
        trip(f"mirror {server} not accepted by astroquery")
        return None

    _bind_timeouts(timeout)

    try:
        Simbad.ROW_LIMIT = int(row_limit)
    except Exception:
        pass
    try:
        _aqs.conf.row_limit = int(row_limit)
    except Exception:
        pass

    try:
        Simbad.reset_votable_fields()
    except Exception:
        pass

    if fields is None:
        fields = ("sp", "B", "V", "R", "ra", "dec", "main_id")
    if legacy_fields is None:
        legacy_fields = ("sp", "flux(B)", "flux(V)", "flux(R)",
                         "ra(d)", "dec(d)", "main_id")

    # ONE attempt per field spelling, not five. If the modern names fail
    # for a non-network reason we are on an older astroquery, so the legacy
    # spelling is tried once. If either fails on a connection error we stop
    # immediately — the host is down and further attempts only burn
    # timeouts.
    for spelling, names in (("modern", fields), ("legacy", legacy_fields)):
        try:
            Simbad.add_votable_fields(*names)
            note_success()
            return Simbad
        except Exception as e:                       # noqa: BLE001
            if note_failure(e, context=f"add_votable_fields/{spelling}"):
                status_cb(f"SIMBAD went unreachable during setup — "
                          f"continuing with Gaia XP only. ({e})")
                return None
            # Non-connection error: wrong field names for this version.
            continue

    status_cb("SIMBAD field setup failed (unsupported astroquery field "
              "names) — continuing with Gaia XP only.")
    return None


def call_simbad(fn: Callable, *args,
                runner: Optional[Callable] = None,
                status_cb: Optional[Callable[[str], None]] = None,
                budget: Optional[Budget] = None,
                attempts: int = 2,
                retry_delay: float = 1.0,
                sleep: Optional[Callable[[float], None]] = None,
                label: str = "SIMBAD query",
                **kwargs):
    """
    Invoke a SIMBAD call with breaker, budget and sane retry policy.

    `runner` lets the caller route the call off the GUI thread — pass a
    lambda that forwards to run_async_blocking. Defaults to a direct call.

    Returns the result, or None if SIMBAD is unavailable / the call failed.
    Never raises.
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731
    if sleep is None:
        try:
            from setiastro.saspro.main_helpers import non_blocking_sleep as sleep
        except Exception:
            sleep = time.sleep
    if runner is None:
        runner = lambda f, *a, **k: f(*a, **k)       # noqa: E731

    with _lock:
        if _state["tripped"]:
            return None

    if budget is not None and budget.expired():
        status_cb(f"{label}: SIMBAD time budget exhausted — "
                  f"continuing with Gaia XP only.")
        return None

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        if budget is not None and budget.expired():
            break
        try:
            if attempt > 1:
                status_cb(f"{label} (retry {attempt}/{attempts})…")
            result = runner(fn, *args, **kwargs)
            note_success()
            return result
        except Exception as e:                       # noqa: BLE001
            last_exc = e
            # Connection failure => host is down. Stop now; do not spend
            # another full timeout proving it twice.
            if note_failure(e, context=label):
                status_cb(f"{label}: SIMBAD unreachable — "
                          f"continuing with Gaia XP only.")
                return None
            if attempt < attempts:
                try:
                    sleep(retry_delay * attempt)
                except Exception:
                    pass

    if last_exc is not None:
        status_cb(f"{label} failed: {last_exc} — continuing without SIMBAD.")
    return None