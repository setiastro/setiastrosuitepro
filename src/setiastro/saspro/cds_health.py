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

Three things fix that, and this module provides all three:

  1. A preflight probe with a HARD deadline, so we learn CDS is down in
     ~4 s rather than ~130 s. The deadline is enforced by joining a daemon
     thread, because urlopen's `timeout` does not cover DNS resolution —
     a hung resolver can block well past it.

  2. Mirror failover. astroquery ships two servers and CDS keeps the
     mirrors up during Strasbourg maintenance, but nothing in SASpro ever
     tried the second one.

  3. A session-wide circuit breaker. The first genuine connection failure
     trips it; every later SIMBAD call in that session returns instantly.
     This is what rescues the batched `query_objects` loops, which would
     otherwise rediscover the outage once per 200-star batch.

Nothing here raises on its own. Callers get None/False and carry on.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Optional, Tuple

__all__ = [
    "MIRRORS",
    "probe",
    "simbad_available",
    "current_server",
    "note_failure",
    "note_success",
    "is_connection_error",
    "trip",
    "reset",
    "status_text",
    "prepare_simbad",
    "call_simbad",
    "Budget",
]

# ── Configuration ────────────────────────────────────────────────────────

# Order matters: first entry is preferred. Matches astroquery's
# conf.servers_list so whatever we pick is a value astroquery accepts.
MIRRORS: tuple[str, ...] = ("simbad.cds.unistra.fr", "simbad.harvard.edu")

# VOSI standard availability endpoint — tiny XML payload, no query cost,
# and it is the canonical "is this TAP service up" question.
_PROBE_PATH = "/simbad/sim-tap/availability"

PROBE_TIMEOUT = 4.0        # per-mirror, seconds
TTL_UP        = 180.0      # re-probe a healthy service this often
TTL_DOWN      = 45.0       # re-probe a dead service this often
BREAKER_RESET = 120.0      # breaker self-heals after this long

# How long the whole SIMBAD phase of one run may take before we give up
# and continue with Gaia XP only.
DEFAULT_BUDGET = 45.0

_UA = "SetiAstroSuitePro/cds-health (+https://setiastro.com)"

# ── State ────────────────────────────────────────────────────────────────

_lock = threading.RLock()
_state: dict = {
    "server":      MIRRORS[0],   # last mirror known good (or default)
    "ok":          None,         # None = never probed
    "checked_at":  0.0,
    "detail":      "",
    "tripped":     False,
    "tripped_at":  0.0,
    "trip_reason": "",
}


def _now() -> float:
    return time.monotonic()


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
                box["body"] = resp.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            # The host answered — that is what we care about.
            box["code"] = int(getattr(e, "code", 0) or 0)
            box["body"] = ""
        except Exception as e:                      # noqa: BLE001
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


def _probe_one(server: str, timeout: float) -> Tuple[bool, str]:
    """Probe a single mirror. Returns (ok, human-readable detail)."""
    url = f"https://{server}{_PROBE_PATH}"
    code, body, err = _fetch_with_hard_deadline(url, timeout)

    if err is not None:
        return False, f"{server}: {type(err).__name__}: {err}"

    if code is None:
        return False, f"{server}: no response"

    # An explicit <available>false</available> means planned downtime.
    if body and "false" in body.lower() and "available" in body.lower():
        return False, f"{server}: reports itself unavailable (maintenance)"

    # Any HTTP answer at all means the host is alive and routable. Even a
    # 404 here (older deployments without VOSI) beats a dead socket, and a
    # 5xx is transient rather than "CDS is gone".
    if 200 <= code < 500:
        return True, f"{server}: HTTP {code}"

    return False, f"{server}: HTTP {code}"


# ── Public health API ────────────────────────────────────────────────────

def probe(*, force: bool = False,
          timeout: float = PROBE_TIMEOUT,
          status_cb: Optional[Callable[[str], None]] = None
          ) -> Tuple[bool, str, str]:
    """
    Determine whether SIMBAD is reachable, trying each mirror in turn.

    Cached: TTL_UP when healthy, TTL_DOWN when not, so calling this before
    every SIMBAD operation is free after the first time.

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

    # Try last-known-good mirror first, then the rest.
    with _lock:
        preferred = _state["server"] or MIRRORS[0]
    order = [preferred] + [m for m in MIRRORS if m != preferred]

    details: list[str] = []
    for server in order:
        status_cb(f"Checking SIMBAD availability ({server})…")
        ok, detail = _probe_one(server, timeout)
        details.append(detail)
        if ok:
            with _lock:
                _state.update(
                    server=server, ok=True, checked_at=_now(), detail=detail,
                    tripped=False, tripped_at=0.0, trip_reason="",
                )
            if server != MIRRORS[0]:
                status_cb(f"Primary SIMBAD unreachable — using mirror {server}")
            return True, server, detail

    joined = " | ".join(details)
    with _lock:
        _state.update(ok=False, checked_at=_now(), detail=joined,
                      tripped=True, tripped_at=_now(),
                      trip_reason="preflight probe failed")
    return False, order[0], joined


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
    with _lock:
        return _state["server"] or MIRRORS[0]


def status_text() -> str:
    """One-line human summary, for status bars and logs."""
    with _lock:
        if _state["ok"] is None:
            return "SIMBAD: not yet checked"
        if _state["tripped"]:
            return f"SIMBAD: unavailable ({_state['trip_reason']})"
        if _state["ok"]:
            return f"SIMBAD: available via {_state['server']}"
        return f"SIMBAD: unavailable ({_state['detail']})"


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
        trip(f"{context or 'SIMBAD call'}: {type(exc).__name__}")
        return True
    return False


def note_success() -> None:
    with _lock:
        _state.update(ok=True, checked_at=_now(), tripped=False,
                      tripped_at=0.0, trip_reason="")


def reset() -> None:
    """Forget everything — used by a manual 'retry CDS' UI action."""
    with _lock:
        _state.update(server=MIRRORS[0], ok=None, checked_at=0.0, detail="",
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

    Two subtleties this handles that the old inline code did not:

      * Timeouts are set BEFORE `add_votable_fields`, not after. On
        astroquery >= 0.4.8 that call validates against the TAP schema over
        the network, so it needs the bound just as much as the query does.

      * `TIMEOUT` was renamed to `timeout` in astroquery 0.4.8. Both are
        set here, so the bound applies on either version. Setting only the
        old name on a new astroquery silently does nothing, which is how
        an unbounded call sneaks through.
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

    # Point astroquery at whichever mirror answered the probe.
    for setter in (
        lambda: setattr(_aqs.conf, "server", server),
        lambda: setattr(Simbad, "server", server),
    ):
        try:
            setter()
        except Exception:
            pass

    # Bound both the module default and the class, old and new names.
    for setter in (
        lambda: setattr(_aqs.conf, "timeout", float(timeout)),
        lambda: setattr(Simbad, "timeout", float(timeout)),
        lambda: setattr(Simbad, "TIMEOUT", float(timeout)),
    ):
        try:
            setter()
        except Exception:
            pass

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