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
    "VIZIER_HTTP_ONLY",
    "canonical_server",
    "simbad_client",
    "simbad_web_url",
    "simbad_web_host",
    "point_vizier",
    "vizier_client",
    "vizier_preference",
    "query_vizier",
    "sesame_resolve",
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
    "query_tap_direct",
    "vizier_probe",
    "vizier_available",
    "vizier_details",
    "vizier_status_text",
    "vizier_status_detail",
    "apply_vizier_server",
    "prepare_vizier",
    "note_vizier_failure",
    "Budget",
]

# ── Configuration ────────────────────────────────────────────────────────

# Order matters: first entry is preferred.
#
# Measured 2026-11-21, during the CDS cybersecurity incident, each mirror
# end-to-end through astroquery (query_region, query_object, query_objects
# with the sp/B/V/R/otype/ids fields) plus the sim-id / sim-coo web pages:
#
#   simbad.cds.unistra.fr  primary; HTTP 404 for TAP and the web UI
#   simbad.harvard.edu     all of it, ~0.7 s; newest record 2026-10-14
#                          (CDS: "data ingested up to 2026-10-14")
#   simbad.idia.ac.za      all of it, ~1-4 s; newest record 2026-04-22 --
#                          an older snapshot, so it goes last
#
# simbad.cfa.harvard.edu is the Harvard mirror under its web-UI name. It
# serves /simbad/sim-tap now too (an earlier probe got HTTP 500 there), but
# it is mapped onto simbad.harvard.edu so the same machine is never counted
# as two independent mirrors.
#
# ASTROQUERY KNOWS ONLY THE FIRST TWO. `simbad.conf.servers_list` is
# ['simbad.cds.unistra.fr', 'simbad.harvard.edu'] (astroquery 0.4.10-0.4.12):
#
#   * the `SimbadClass.server` setter checks that list, which is a plain
#     mutable list -- _register_with_astroquery() appends the rest, and an
#     INSTANCE then takes them;
#   * `conf.server` is an option() ConfigItem frozen at import and still
#     raises for anything else -- and a fresh `Simbad()` starts from
#     conf.server. So build new instances with simbad_client(), never
#     `Simbad()`, or they go back to Strasbourg while the module-level
#     instance is on the mirror.
#
# A mirror is only declared usable once a real TAP query has returned a
# VOTable from it (see _verify_tap) -- a shallow "did something respond?"
# check is how a broken endpoint once looked healthy. That is the check that
# should gate any future addition to this list, not reachability.
MIRRORS: tuple[str, ...] = (
    "simbad.cds.unistra.fr",
    "simbad.harvard.edu",
    "simbad.idia.ac.za",
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
# too.
#
# Measured 2026-11-21 with a real query to /viz-bin/votable (the endpoint
# astroquery POSTs to) for each catalogue the tools use -- Gaia DR3, APASS,
# VSX, UCAC4, 2MASS and HyperLEDA -- rather than by reachability:
#
#   vizier.cds.unistra.fr   primary; HTTP 404 (CDS incident)
#   vizier.cfa.harvard.edu  all six over https, 0.3-8 s
#   vizier.idia.ac.za       all six over https, 5-12 s
#   vizier.nao.ac.jp        all six over plain HTTP, 0.7-8 s; :443 timed out
#   vizier.china-vo.org     :443 refused; HTTP 500 for a query on :80
#   vizier.hia.nrc.ca       TCP connect timed out (and on 2026-11-14)
#   vizier.iucaa.in         TCP connect timed out (and on 2026-11-14)
#
# CDS's own mirror list now names IDIA, CfA, NAOJ and China-VO, the last two
# with http:// URLs (see VIZIER_HTTP_ONLY). NRC and IUCAA are no longer on it
# but stay at the end: a mirror that is down today may be back tomorrow, and
# _race_mirrors probes the candidates concurrently, so a dead host loses the
# race instead of adding its timeout to the user's wait. Hosts that fail a
# live call are moved to the back of the order for BREAKER_RESET seconds.
#
# vizier.inasan.ru and vizier.ast.cam.ac.uk are deliberately absent: both
# refuse :443 outright and Cambridge's :80 serves a 404, so neither is a
# VizieR endpoint any more.
VIZIER_MIRRORS: tuple[str, ...] = (
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
    "vizier.idia.ac.za",
    "vizier.nao.ac.jp",
    "vizier.china-vo.org",
    "vizier.hia.nrc.ca",
    "vizier.iucaa.in",
)

# Mirrors that serve VizieR over plain HTTP only. astroquery always builds
# "https://" + VIZIER_SERVER (VizierClass._server_to_url), so these need the
# scheme overridden per Vizier object -- point_vizier() does that, and is the
# only thing that should point one at a host. They must never go into
# vizier.conf.server either: NAOJ is one of astroquery's accepted names, and
# every fresh Vizier() would then try it over https and time out.
VIZIER_HTTP_ONLY: frozenset = frozenset({
    "vizier.nao.ac.jp",
    "vizier.china-vo.org",
})

# VOSI standard availability endpoint — tiny XML payload, no query cost.
# Stage 1 only: necessary, not sufficient.
_PROBE_PATH = "/simbad/sim-tap/availability"

# Stage 2: the cheapest real TAP query there is. A mirror that answers this
# with a VOTable can serve query_region/query_object; one that answers the
# availability endpoint but not this cannot, and is exactly the failure
# mode that got Harvard struck off the list before.
_TAP_PATH = "/simbad/sim-tap/sync"
_TAP_SMOKE_QUERY = "SELECT TOP 1 main_id FROM basic"

# VizieR is queried over its own endpoint, not TAP: astroquery POSTs to
# https://<server>/viz-bin/votable (Vizier._server_to_url). So VizieR health
# has to be measured there -- a healthy SIMBAD TAP says nothing about it, and
# gating VizieR on SIMBAD meant a Strasbourg TAP outage disabled every VizieR
# mirror even while Harvard was serving VizieR normally.
_VIZIER_PATH = "/viz-bin/votable"

# The smallest real VizieR query that proves a mirror can serve DATA.
#
# Three things this had to get right, each learned from a mirror that passed
# a weaker check and then failed in use:
#
#   * a VOTable has to come back (a front page or error document does not),
#   * it must not be a VOTable whose only content is an error INFO, and
#   * it must contain an actual row. Mirrors are PARTIAL -- measured on
#     2026-11-14, vizier.cfa.harvard.edu and vizier.idia.ac.za both answer
#     for I/350/gaiaedr3, II/246, I/322A and V/147/sdss12 with either an
#     empty resource or an error, while serving VII/237 and VII/291 fine.
#     A mirror that returns an empty table for everything is not a mirror
#     you want the deep search failing over to.
#
# VII/237 (HyperLEDA) is the probe catalogue because it is small, every
# mirror carries it, and it is one of the catalogues the deep search uses.
# The cone is M31's centre, which is guaranteed to contain a galaxy, so an
# empty result means the mirror, not the sky.
_VIZIER_SMOKE = {
    "-source":  "VII/237",
    "-c":       "10.684708 +41.268750",
    "-c.rm":    "2",
    "-out.max": "2",
    "-out":     "*",
}

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
    # VizieR health, tracked separately from SIMBAD's: different hosts,
    # different endpoint, and one can be down while the other is fine.
    "vz_server":      None,       # last VizieR mirror known good
    "vz_ok":          None,       # None = never probed
    "vz_checked_at":  0.0,
    "vz_detail":      "",
    "vz_details":     [],
    "vz_verified_at": {},         # server -> monotonic time of last verify
    "vz_bad":         {},         # server -> monotonic time it failed a call
    "vz_tripped":     False,
    "vz_tripped_at":  0.0,
    "vz_trip_reason": "",
    # Host for browser links when no full probe has said which is good.
    "web_host":       "",
    "web_at":         0.0,
    "web_ok":         False,
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
                      tripped=False, tripped_at=0.0, trip_reason="",
                      web_host="", web_at=0.0, web_ok=False)

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


def vizier_preference() -> str:
    """The STORED VizieR preference: "" for automatic, else a hostname.

    Not vizier_server(), which resolves automatic to a concrete host -- a
    settings dialog that pre-selects from that turns automatic into a pin
    the first time someone presses OK.
    """
    _load_preference()
    with _lock:
        return _state["vizier_pref"] or ""


def _vizier_scheme(host: str) -> str:
    return "http" if host in VIZIER_HTTP_ONLY else "https"


def point_vizier(v, host: str) -> bool:
    """
    Point one astroquery Vizier object -- a new instance, or the module-level
    `Vizier` -- at `host`, scheme included. True only if the URL it will now
    request is on that host over the right scheme.

    VIZIER_SERVER is a plain attribute and takes any host; the scheme is the
    catch. `_server_to_url()` hard-codes https, so for an http-only mirror an
    instance attribute of the same name shadows the method, and for every
    other host an override left there by an earlier call is removed.
    """
    host = (host or "").strip()
    if not host:
        return False
    for attr in ("VIZIER_SERVER", "server", "_server"):
        try:
            setattr(v, attr, host)
        except Exception:                            # noqa: BLE001
            pass
    try:
        if host in VIZIER_HTTP_ONLY:
            def _http_url(return_type="votable", _host=host):
                if return_type not in ("votable", "asu-tsv", "asu-fits",
                                       "asu-binfits"):
                    raise TypeError(
                        f"VizieR return type {return_type!r} is not supported")
                return f"http://{_host}/viz-bin/{return_type}"
            v._server_to_url = _http_url
        else:
            vars(v).pop("_server_to_url", None)
    except Exception:                                # noqa: BLE001
        return False
    try:
        url = str(v._server_to_url())
    except Exception:                                # noqa: BLE001
        return False
    return url.startswith(f"{_vizier_scheme(host)}://{host}/")


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


def _race_mirrors(servers, check, timeout: float,
                  status_cb: Optional[Callable[[str], None]] = None):
    """
    Try `check(server)` on every candidate at once; return the best winner.

    "Best" means the one earliest in `servers`, not the one that answered
    first, so preference order still decides when several mirrors are
    healthy -- the race is only there to stop a dead mirror costing a full
    timeout before a live one further down the list is even tried.

    The first candidate is tried on its own first, because in the normal
    case it answers in well under a second and firing N queries at CDS for
    every probe would be rude for no benefit. Returns
    (ok, server, detail, all_details).
    """
    servers = [s for s in servers if s]
    if not servers:
        return False, "", "no mirrors configured", []

    head = servers[0]
    if status_cb:
        status_cb(f"Checking {head}…")
    ok, detail = check(head)
    if ok:
        return True, head, detail, [detail]

    rest = servers[1:]
    if not rest:
        return False, head, detail, [detail]

    if status_cb:
        status_cb(f"{head} did not answer — trying "
                  f"{len(rest)} mirror{'s' if len(rest) != 1 else ''}…")

    # Daemon threads, not a ThreadPoolExecutor: the pool's shutdown joins its
    # workers, so a mirror that is going to time out holds the race open for
    # its full deadline even after a good mirror has answered -- and the
    # pool's atexit hook would hold interpreter shutdown open the same way.
    # Daemon threads let the dead ones finish into a result dict nobody is
    # waiting on.
    results: dict = {}
    r_lock = threading.Lock()

    def _runner(server: str) -> None:
        try:
            got = check(server)
        except Exception as e:                       # noqa: BLE001
            got = (False, f"{server}: {type(e).__name__}: {e}")
        with r_lock:
            results[server] = got

    for server in rest:
        threading.Thread(target=_runner, args=(server,), daemon=True,
                         name=f"cds-probe-{server}").start()

    # Settled = some mirror succeeded and every mirror preferred over it has
    # already reported, so a later answer cannot change the choice.
    deadline = _now() + timeout * 2.0
    while True:
        with r_lock:
            snapshot = dict(results)
        for server in rest:
            got = snapshot.get(server)
            if got is None:
                break
            if got[0]:
                break
        else:
            server = None                            # all reported, none ok
        if (server is None or (snapshot.get(server) or (False,))[0]
                or len(snapshot) == len(rest) or _now() >= deadline):
            break
        time.sleep(0.02)

    with r_lock:
        for server in rest:
            results.setdefault(server, (False, f"{server}: no answer before "
                                               f"another mirror settled it"))
        results = dict(results)

    details = [detail] + [results[s][1] for s in rest if s in results]
    for server in rest:                              # preference order wins
        got = results.get(server)
        if got and got[0]:
            return True, server, got[1], details
    return False, head, " | ".join(details), details


def _verify_vizier(server: str, timeout: float = VERIFY_TIMEOUT
                   ) -> Tuple[bool, str]:
    """
    Run the smallest possible real VizieR query and insist on a VOTable.

    Same standard as _verify_tap, for the same reason: several VizieR
    mirrors answer on :443 with a front page or an error document, and a
    reachability check cannot tell those from a mirror that will serve
    query_region. One extra trap here that SIMBAD does not have - a mirror
    can return a syntactically valid VOTable whose only payload is an error
    INFO ("Request has been interrupted"), which is a failure wearing a
    success's clothes.
    """
    with _lock:
        last = float(_state["vz_verified_at"].get(server, 0.0))
    if last and (_now() - last) < TTL_VERIFIED:
        return True, f"{server}: VizieR verified recently"

    url = (f"{_vizier_scheme(server)}://{server}{_VIZIER_PATH}?"
           f"{urllib.parse.urlencode(_VIZIER_SMOKE)}")
    code, body, err = _fetch_with_hard_deadline(url, timeout)

    if err is not None:
        return False, f"{server}: VizieR query {type(err).__name__}: {err}"
    if code is None:
        return False, f"{server}: VizieR query got no response"

    low = (body or "").lower()
    if code >= 500:
        return False, f"{server}: VizieR query returned HTTP {code}"
    if "<votable" in low:
        flat = _strip_ns(low).replace(" ", "")
        if 'value="error"' in flat or "query_status=error" in flat:
            return False, (f"{server}: VizieR returned a VOTable reporting an "
                           f"error, not data")
        if "<tr>" not in flat and "<tr " not in low:
            return False, (f"{server}: VizieR returned an empty table for a "
                           f"cone that has data -- this mirror does not carry "
                           f"the catalogue")
        with _lock:
            _state["vz_verified_at"][server] = _now()
        return True, f"{server}: VizieR query returned data"
    if "<html" in low or "<!doctypehtml" in _strip_ns(low):
        return False, (f"{server}: VizieR query returned HTML, not VOTable "
                       f"(HTTP {code}) - this host is not serving VizieR")
    return False, f"{server}: VizieR query returned HTTP {code}, no VOTable"


def _vizier_order() -> list:
    """VizieR mirrors to try: preference, then last-good, then the rest.

    Hosts that failed a live call recently go to the back rather than being
    dropped - a mirror that timed out once is still a better last resort
    than no VizieR at all.
    """
    _load_preference()
    with _lock:
        pref = _state["vizier_pref"] or ""
        current = _state["vz_server"] or ""
        bad = {h for h, t in _state["vz_bad"].items()
               if (_now() - float(t)) < BREAKER_RESET}

    order: list = []
    for host in (pref, current, *VIZIER_MIRRORS):
        if host and host not in order:
            order.append(host)
    return ([h for h in order if h not in bad] +
            [h for h in order if h in bad])


def vizier_probe(*, force: bool = False,
                 timeout: float = VERIFY_TIMEOUT,
                 status_cb: Optional[Callable[[str], None]] = None
                 ) -> Tuple[bool, str, str]:
    """
    Find a VizieR mirror that actually answers queries.

    Returns (ok, server, detail). Cached like probe(): TTL_UP when healthy,
    TTL_DOWN when not.
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731

    with _lock:
        if (_state["vz_tripped"] and not force and
                (_now() - _state["vz_tripped_at"]) < BREAKER_RESET):
            return (False, _state["vz_server"] or VIZIER_MIRRORS[0],
                    _state["vz_trip_reason"] or _state["vz_detail"])
        age = _now() - float(_state["vz_checked_at"])
        if not force and _state["vz_ok"] is not None:
            ttl = TTL_UP if _state["vz_ok"] else TTL_DOWN
            if age < ttl:
                return (bool(_state["vz_ok"]),
                        _state["vz_server"] or VIZIER_MIRRORS[0],
                        _state["vz_detail"])

    ok, server, detail, details = _race_mirrors(
        _vizier_order(), lambda s: _verify_vizier(s, timeout), timeout,
        lambda m: status_cb(f"VizieR: {m}"))
    if ok:
        with _lock:
            _state.update(vz_server=server, vz_ok=True,
                          vz_checked_at=_now(), vz_detail=detail,
                          vz_details=list(details), vz_tripped=False,
                          vz_tripped_at=0.0, vz_trip_reason="")
            _state["vz_bad"].pop(server, None)
        if server != VIZIER_MIRRORS[0]:
            status_cb(f"Primary VizieR unreachable — using mirror {server}")
        return True, server, detail

    joined = " | ".join(details)
    with _lock:
        _state.update(vz_ok=False, vz_checked_at=_now(), vz_detail=joined,
                      vz_details=list(details), vz_tripped=True,
                      vz_tripped_at=_now(),
                      vz_trip_reason="no VizieR mirror answered a query")
    return False, VIZIER_MIRRORS[0], joined


def vizier_available(status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """True if some VizieR mirror can serve a query right now."""
    ok, _server, _detail = vizier_probe(status_cb=status_cb)
    return ok


def vizier_details() -> list:
    """Per-mirror results of the last VizieR probe."""
    with _lock:
        return list(_state["vz_details"])


def vizier_status_text() -> str:
    """One-line human summary of VizieR health."""
    with _lock:
        if _state["vz_ok"] is None:
            return "VizieR: not yet checked"
        if _state["vz_ok"] and not _state["vz_tripped"]:
            return f"VizieR: available via {_state['vz_server']}"
        detail = ((_state["vz_trip_reason"] if _state["vz_tripped"] else "")
                  or _state["vz_detail"] or "unknown")
        return f"VizieR: unavailable ({detail})"


def vizier_status_detail() -> str:
    """Multi-line VizieR detail - one line per mirror tried."""
    with _lock:
        lines = list(_state["vz_details"]) or (
            [_state["vz_detail"]] if _state["vz_detail"] else [])
        pref = _state["vizier_pref"] or ""
    out = "\n".join(f"  • {ln}" for ln in lines if ln)
    if pref:
        out += f"\n  • mirror preference: {pref}"
    return out or "  • no VizieR probe has run yet"


def apply_vizier_server(server: str,
                        status_cb: Optional[Callable[[str], None]] = None
                        ) -> bool:
    """
    Point astroquery's VizieR at `server`. Returns True only if it took.

    There are two places to set this and they do not accept the same thing:

      * `vizier.conf.server` is an astropy option() restricted to seven
        names - unistra, cfa.harvard, nao.ac.jp, hia.nrc.ca, ast.cam.ac.uk,
        iucaa.ernet.in (the retired domain) and china-vo.org. Anything else
        raises TypeError "the value is unacceptable".
      * `VizierClass.VIZIER_SERVER` is a plain attribute, takes any host,
        and is what `_server_to_url()` actually builds the request URL from.

    So a mirror outside astroquery's hard-coded seven (vizier.idia.ac.za,
    say) is still usable - it just cannot go in conf. Both are set, conf
    failures are tolerated, and the result is decided by asking astroquery
    what URL it would use, not by whether an assignment threw. Guessing from
    the assignment is how a failover reports success while queries keep
    going to the dead host. An http-only mirror (VIZIER_HTTP_ONLY) is kept
    out of conf entirely and gets its scheme from point_vizier().
    """
    if status_cb is None:
        status_cb = lambda _m: None                  # noqa: E731
    host = (server or "").strip()
    if not host:
        return False
    try:
        import astroquery.vizier as _aqv
        from astroquery.vizier import Vizier as _Vizier
    except Exception as e:                           # noqa: BLE001
        status_cb(f"astroquery.vizier unavailable: {e}")
        return False
    if host not in VIZIER_HTTP_ONLY:
        try:
            _aqv.conf.server = host                  # may legitimately raise
        except Exception:                            # noqa: BLE001
            pass

    # The only answer that matters: where will the next request go?
    if not point_vizier(_Vizier, host):
        try:
            url = str(_Vizier._server_to_url())
        except Exception:                            # noqa: BLE001
            url = "an unreadable URL"
        status_cb(f"astroquery still points at {url} after being asked "
                  f"for {host}")
        return False
    return True


def prepare_vizier(status_cb: Optional[Callable[[str], None]] = None,
                   *, timeout: float = 45.0) -> str:
    """
    Probe, pick a working VizieR mirror, and point astroquery at it.

    Returns the hostname in use, or "" when no mirror answered - in which
    case the caller should say so rather than issue queries that are certain
    to fail.
    """
    ok, server, _detail = vizier_probe(status_cb=status_cb)
    if not ok or not apply_vizier_server(server, status_cb):
        return ""
    try:
        import astroquery.vizier as _aqv
        from astroquery.vizier import Vizier as _Vizier
        for setter in (
            lambda: setattr(_aqv.conf, "timeout", float(timeout)),
            lambda: setattr(_Vizier, "TIMEOUT", float(timeout)),
        ):
            try:
                setter()
            except Exception:                        # noqa: BLE001
                pass
    except Exception:                                # noqa: BLE001
        pass
    return server


def note_vizier_failure(exc: BaseException, *, context: str = "",
                        status_cb: Optional[Callable[[str], None]] = None
                        ) -> bool:
    """
    Record a failed VizieR call and move to another mirror if one works.

    Returns True only when VizieR should be given up on entirely (nothing
    left to try), so a caller loop reads it as "stop". False means another
    mirror is now configured - try the same query again.
    """
    if not is_connection_error(exc):
        return False
    with _lock:
        failed = _state["vz_server"] or VIZIER_MIRRORS[0]
        _state["vz_bad"][failed] = _now()
        _state["vz_verified_at"].pop(failed, None)
        _state["vz_ok"] = None            # force a fresh probe
        _state["vz_checked_at"] = 0.0
    if status_cb:
        status_cb(f"VizieR mirror {failed} failed — trying another…")
    ok, server, _detail = vizier_probe(force=True, status_cb=status_cb)
    if ok and apply_vizier_server(server, status_cb):
        return False
    with _lock:
        _state.update(vz_tripped=True, vz_tripped_at=_now(),
                      vz_trip_reason=(f"{context or 'VizieR call'}: "
                                      f"{type(exc).__name__}: {exc}"))
    return True


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

    def _check(server: str):
        ok, detail = _probe_one(server, timeout)
        if ok and verify:
            ok2, detail2 = _verify_tap(server)
            return ok2, f"{detail}; {detail2}"
        return ok, detail

    ok, server, detail, details = _race_mirrors(
        order, _check, max(timeout, VERIFY_TIMEOUT),
        lambda m: status_cb(f"SIMBAD: {m}"))
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
        # trip_reason stays empty on a probe failure: the per-mirror detail
        # IS the reason here and reads far better than a generic label.
        # status_text() prefers trip_reason only when a live call tripped
        # the breaker, which is the case where detail would be stale.
        _state.update(ok=False, checked_at=_now(), detail=joined,
                      details=list(details),
                      tripped=True, tripped_at=_now(),
                      trip_reason="")
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
        # Unavailable -- but say WHY, and say it from the right source.
        #
        # A probe failure leaves a concrete per-mirror detail ("served HTML,
        # not VOSI XML") and a useless trip reason ("preflight probe
        # failed"), so detail wins there. A breaker tripped by a real query
        # is the other way round: trip_reason holds the actual error and
        # `detail` still holds the LAST SUCCESSFUL PROBE, which produced
        # "SIMBAD: unavailable (simbad.harvard.edu: TAP query returned a
        # VOTable)" -- a message that contradicted itself and sent people
        # looking for a network fault that was not there.
        if _state["tripped"] and _state["trip_reason"]:
            return f"SIMBAD: unavailable ({_state['trip_reason']})"
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


def note_failure(exc: BaseException, *, context: str = "",
                 status_cb: Optional[Callable[[str], None]] = None) -> bool:
    """
    Record a SIMBAD failure, moving to another mirror if one still works.

    Returns True if SIMBAD should be given up on (caller stops retrying);
    False means either the error was not connection-class, or a different
    mirror is now configured and the call is worth repeating.

    Failing over here matters because the breaker is session-wide: before,
    one timeout against whichever mirror we happened to be on disabled
    SIMBAD for two minutes even when the other mirror was answering
    normally the whole time.
    """
    if not is_connection_error(exc):
        return False

    with _lock:
        failed = _state["server"] or MIRRORS[0]
        _state["verified_at"].pop(failed, None)
        pinned = bool(_state["pin"]) and bool(_state["pref"])
    alternatives = [m for m in MIRRORS if m != failed]

    if not pinned and alternatives:
        if status_cb:
            status_cb(f"SIMBAD mirror {failed} failed — trying another…")
        for server in alternatives:
            ok, detail = _probe_one(server, PROBE_TIMEOUT)
            if ok:
                ok, detail2 = _verify_tap(server)
                detail = f"{detail}; {detail2}"
            if ok and apply_server(server):
                with _lock:
                    _state.update(server=server, ok=True, checked_at=_now(),
                                  detail=detail, tripped=False,
                                  tripped_at=0.0, trip_reason="")
                if status_cb:
                    status_cb(f"SIMBAD: now using mirror {server}")
                return False

    trip(f"{context or 'SIMBAD call'}: {type(exc).__name__}: {exc}")
    return True


def query_tap_direct(adql: str, *, server: str = "",
                    maxrec: int = 0,
                    timeout: float = 60.0):
    """
    Run an ADQL query straight against SIMBAD's TAP sync endpoint.

    A last resort for when astroquery's own path fails but the service is
    demonstrably alive -- astroquery wraps pyvo, which has its own session,
    redirect and job-submission behaviour, so "astroquery raised" and
    "SIMBAD is down" are not the same statement. The probe in this module
    talks to /simbad/sim-tap/sync directly and keeps working in exactly
    those cases, so the query may as well go the same way.

    Returns an astropy Table, or None if this path could not produce one.
    """
    host = canonical_server(server) or current_server()
    params = {
        "request": "doQuery",
        "lang":    "ADQL",
        "format":  "votable",
        "query":   str(adql),
    }
    if maxrec and int(maxrec) > 0:
        params["maxrec"] = str(int(maxrec))
    code, body, err = _fetch_with_hard_deadline(
        f"https://{host}{_TAP_PATH}?{urllib.parse.urlencode(params)}", timeout)
    if err is not None or code is None or not body:
        return None
    low = body.lower()
    if "<votable" not in low:
        return None
    flat = _strip_ns(low).replace(" ", "")
    if 'value="error"' in flat or "query_status=error" in flat:
        return None
    try:
        import io as _io
        from astropy.io.votable import parse_single_table
        table = parse_single_table(
            _io.BytesIO(body.encode("utf-8", "replace"))).to_table()
    except Exception:                                # noqa: BLE001
        return None
    note_success()
    return table


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
                      tripped=False, tripped_at=0.0, trip_reason="",
                      web_host="", web_at=0.0, web_ok=False)


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

def _register_with_astroquery(_aqs) -> tuple:
    """
    Make astroquery's SIMBAD server check accept every host in MIRRORS.

    The `SimbadClass.server` setter validates against `conf.servers_list`,
    a plain list on the Conf class, so appending to it is all an INSTANCE
    needs to take a mirror astroquery does not ship with. conf.server is a
    separate, frozen option() and is not changed by this. Returns the list
    as it now stands.
    """
    lst = getattr(_aqs.conf, "servers_list", None)
    if isinstance(lst, list):
        for host in MIRRORS:
            if host not in lst:
                lst.append(host)
        return tuple(lst)
    return tuple(lst or MIRRORS)


def _simbad_target(s, _aqs) -> str:
    """The host a Simbad object will actually query.

    From astroquery 0.4.8 the TAP URL is built from the instance's `server`
    property. Before that, requests went to conf.server whatever the
    instance said, so reading back an attribute we just set would confirm
    nothing there.
    """
    if isinstance(getattr(type(s), "server", None), property):
        return str(s.server)
    return str(getattr(_aqs.conf, "server", "") or "")


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

    So the module-level `Simbad` instance is pointed at the host (which
    works for every registered mirror), conf.server is set where astroquery
    allows it, and the answer comes from where the instance will actually
    send its next query. New instances need simbad_client().
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

    allowed = _register_with_astroquery(_aqs)
    if host not in allowed:
        status_cb(f"astroquery will not accept SIMBAD server {host!r} "
                  f"(it accepts: {', '.join(allowed)})")
        return False

    # Raises for a mirror astroquery does not ship with (IDIA): expected,
    # not a failure. It only seeds new Simbad() objects, and those are made
    # with simbad_client(), which sets the instance itself.
    try:
        _aqs.conf.server = host
    except Exception:                                # noqa: BLE001
        pass

    # The module-level `Simbad` is an *instance*, and it is the one
    # Simbad.query_* and prepare_simbad() callers use.
    try:
        Simbad.server = host
    except Exception as e:                           # noqa: BLE001
        status_cb(f"could not point astroquery's Simbad at {host}: {e}")

    target = _simbad_target(Simbad, _aqs)
    if target != host:
        status_cb(f"astroquery's Simbad still queries "
                  f"{target or 'its default host'} after being asked for "
                  f"{host}")
        return False
    return True


def simbad_client(host: str = "",
                  status_cb: Optional[Callable[[str], None]] = None):
    """
    A NEW astroquery Simbad instance on a working mirror. Use this instead
    of `Simbad()`.

    A fresh `Simbad()` starts from conf.server, which can only hold
    astroquery's two built-in hosts and holds whatever the last
    apply_server() left there -- Strasbourg, unless something failed over
    first. So a tool that built its own instance kept querying Strasbourg
    through an outage while the rest of SASpro was on a mirror.

    With no `host` the mirror is the one the health probe picked (cached,
    so this is free after the first call). Raises ConnectionError when no
    mirror is answering (the text is status_text(), which says why), and
    RuntimeError when astroquery cannot be pointed at the host.

    No timeout is set here on purpose: on astroquery >= 0.4.8 the `timeout`
    setter fetches the mirror's /capabilities over the network.
    """
    from astroquery.simbad import Simbad as _Simbad
    import astroquery.simbad as _aqs

    if host:
        host = canonical_server(host)
    else:
        if not simbad_available(status_cb=status_cb):
            raise ConnectionError(status_text())
        host = current_server()
    _register_with_astroquery(_aqs)
    s = _Simbad()
    try:
        s.server = host
    except Exception as e:                           # noqa: BLE001
        raise RuntimeError(
            f"astroquery will not use SIMBAD server {host}: {e}") from e
    target = _simbad_target(s, _aqs)
    if target != host:
        raise RuntimeError(f"a new Simbad instance queries "
                           f"{target or 'its default host'}, not {host}")
    return s


def simbad_web_host(*, timeout: float = 3.0) -> str:
    """
    The SIMBAD host to send a BROWSER to.

    The mirror the health probe last found working, when there is one.
    Otherwise the mirrors are raced on their availability endpoint alone,
    with a hard deadline, and the answer is cached like a probe. That is
    enough for a web page (it is what the user sees, not a query we parse),
    and it is kept out of the probe's own state so it can never stand in
    for the end-to-end TAP check that choosing a query mirror needs.
    """
    _load_preference()
    with _lock:
        if _state["ok"] and not _state["tripped"]:
            return _state["server"] or MIRRORS[0]
        host = _state["web_host"]
        ttl = TTL_UP if _state["web_ok"] else TTL_DOWN
        if host and (_now() - float(_state["web_at"])) < ttl:
            return host
    order = _probe_order()
    ok, server, _detail, _details = _race_mirrors(
        order, lambda s: _probe_one(s, timeout), timeout)
    host = server if ok else (order[0] if order else MIRRORS[0])
    with _lock:
        _state.update(web_host=host, web_at=_now(), web_ok=bool(ok))
    return host


def simbad_web_url(path: str, *, timeout: float = 3.0) -> str:
    """
    Browser URL for a SIMBAD web page on a mirror that is answering.

    `path` is everything after "/simbad/", e.g. "sim-id?Ident=Vega". Every
    mirror serves the same sim-id / sim-coo / sim-basic / sim-sam pages, so
    an "Open in SIMBAD" link during a Strasbourg outage lands on a mirror
    instead of a 404 page.
    """
    return (f"https://{simbad_web_host(timeout=timeout)}/simbad/"
            f"{str(path).lstrip('/')}")


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



# ── VizieR queries with mirror failover ──────────────────────────────────
#
# prepare_vizier() + note_vizier_failure() are the building blocks; these
# wrap them into the loop every caller otherwise rewrites (WIMI's deep
# search does it inline), and close two gaps a caller hits on its own:
#
#   * apply_vizier_server() points the module-level Vizier and conf at the
#     mirror, but `Vizier(columns=..., ...)` builds a NEW instance that reads
#     conf.server -- and conf only accepts astroquery's seven hard-coded
#     hosts. On a mirror outside that list (vizier.idia.ac.za, which was the
#     only mirror serving VSX/APASS/UCAC4 on 2026-11-19) a fresh instance
#     silently goes back to Strasbourg. vizier_client() sets the instance.
#   * note_vizier_failure() fails over on connection-class errors only. The
#     Strasbourg outage presents as HTTP 404 from /viz-bin/votable -- the
#     host answers, it just isn't serving -- so a caller passing that error
#     through retried the same dead host.

def vizier_client(host: str = "", **kwargs):
    """
    A new astroquery Vizier instance pointed at `host` (default: the mirror
    prepare_vizier() last chose, else the preference/primary).

    Raises RuntimeError when astroquery would still send the request
    elsewhere -- decided by asking astroquery for the URL it will use.
    """
    from astroquery.vizier import Vizier as _Vizier
    host = (host or "").strip()
    if not host:
        with _lock:
            host = _state["vz_server"] or ""
        host = host or vizier_server()
    v = _Vizier(**kwargs)
    if not point_vizier(v, host):
        try:
            url = str(v._server_to_url())
        except Exception as e:                       # noqa: BLE001
            raise RuntimeError(f"cannot read the Vizier URL: {e}") from e
        raise RuntimeError(f"Vizier instance points at {url}, not {host}")
    return v


def _is_http_status_error(exc: BaseException) -> bool:
    """True for an HTTP 4xx/5xx answer (requests.HTTPError and kin)."""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if code is None:
        code = getattr(exc, "code", None)
    try:
        if code is not None and int(code) >= 400:
            return True
    except (TypeError, ValueError):
        pass
    text = f"{exc}".lower()
    return (("client error" in text or "server error" in text)
            and "for url" in text)


def _vizier_result_empty(res) -> bool:
    """True for None, an empty TableList, or a TableList/Table with no rows."""
    try:
        if res is None:
            return True
        if hasattr(res, "colnames"):                 # a single Table
            return len(res) == 0
        if len(res) == 0:                            # TableList
            return True
        return len(res[0]) == 0
    except Exception:                                # noqa: BLE001
        return False


# (host, context) -> when that mirror answered `context` with no rows.
# Mirrors are partial and occasionally quirky (on 2026-11-19 CfA returned
# no rows for a UCAC4 cone with one particular column list and the star for
# any other), so a mirror that came back empty for a lookup that should have
# rows is tried last for that lookup for a while -- not marked dead, since
# it is serving everything else.
_vz_empty_for: dict = {}
EMPTY_MEMORY = 1800.0


def query_vizier(run: Callable[[str], object], *, context: str = "",
                 attempts: int = 3, timeout: float = 30.0,
                 require_rows: bool = False,
                 status_cb: Optional[Callable[[str], None]] = None):
    """
    Run one VizieR request on a working mirror, failing over between them.

    `run(host)` performs the request against `host` (build the client with
    vizier_client(host, ...)) and returns its result. Returns
    (result, host). Up to `attempts` mirrors are tried.

      * Connection failures and HTTP errors (a 404 from /viz-bin/votable is
        how the Strasbourg outage presents) move on to another mirror and
        are recorded in the shared health state, so later lookups -- in any
        tool -- start from a mirror that works.
      * require_rows=True: an EMPTY answer also moves on, to another mirror
        that passes the VizieR probe. Use it where an empty answer can only
        mean the mirror (catalogue cones around a real field, model grids);
        not where "nothing there" is a normal answer (VSX at a random star).
        If every mirror tried comes back empty, the empty result is returned.

    Raises the last error when no mirror served the request, or at once for
    an error no mirror would fix (a malformed request).
    """
    first = prepare_vizier(status_cb=status_cb, timeout=timeout)
    key = context or "vizier"
    now = _now()
    with _lock:
        stale = [k for k, t in _vz_empty_for.items() if now - t > EMPTY_MEMORY]
        for k in stale:
            _vz_empty_for.pop(k, None)
        empty_here = {h for (h, c) in _vz_empty_for if c == key}
        bad = {h for h, t in _state["vz_bad"].items()
               if (now - float(t)) < BREAKER_RESET}
    order: list = []
    for h in ([first] if first else []) + _vizier_order():
        if h and h not in order:
            order.append(h)
    usable = [h for h in order if h == first or h not in bad]   # first just passed the race
    order = ([h for h in usable if h not in empty_here]
             + [h for h in usable if h in empty_here])

    last: Optional[BaseException] = None
    empty_result, empty_host = None, ""
    tried = 0
    for host in order:
        if tried >= max(1, int(attempts)):
            break
        if host != first:
            # Not the mirror prepare_vizier just raced: make sure it serves
            # data before spending a query timeout on it (cached when good).
            ok, _detail = _verify_vizier(host)
            if not ok:
                with _lock:
                    _state["vz_bad"][host] = _now()
                continue
        tried += 1
        try:
            res = run(host)
        except Exception as e:                       # noqa: BLE001
            last = e
            if not (is_connection_error(e) or _is_http_status_error(e)):
                raise
            if status_cb:
                status_cb(f"VizieR mirror {host} failed — trying another…")
            with _lock:
                _state["vz_bad"][host] = _now()
                _state["vz_verified_at"].pop(host, None)
                is_current = (host == (_state["vz_server"] or ""))
            if is_current:
                # Re-race so the shared state (and every other tool) moves off it.
                note_vizier_failure(e if is_connection_error(e)
                                    else ConnectionError(f"{host}: {e}"),
                                    context=context, status_cb=status_cb)
            continue
        if require_rows and _vizier_result_empty(res):
            with _lock:
                _vz_empty_for[(host, key)] = _now()
            if empty_result is None:
                empty_result, empty_host = res, host
            if status_cb:
                status_cb(f"{host} returned no rows for {key} — trying another mirror…")
            continue
        return res, host
    if empty_result is not None:
        return empty_result, empty_host
    if last is not None:
        raise last
    raise ConnectionError(vizier_status_text())


# ── Sesame name resolution with mirror failover ──────────────────────────
#
# Name -> coordinates does not need the SIMBAD TAP service. Sesame is its own
# CDS service (it consults SIMBAD, NED and VizieR behind the scenes) and runs
# on cds.unistra.fr -- a different host from simbad.cds.unistra.fr -- and on
# the VizieR mirrors. On 2026-11-19, with both SIMBAD TAP mirrors down,
# Sesame on cds.unistra.fr and vizier.idia.ac.za resolved HAT-P-32,
# V336 Aps, TYC 9256-617-1 and TOI-1130 in 2-5 s each. On 2026-11-21
# cds.unistra.fr's Sesame was a redirect to IDIA (4.5 s) and CfA answered in
# 0.4 s; these three are the Sesame hosts CDS lists.

SESAME_URLS: tuple[str, ...] = (
    "https://cds.unistra.fr/cgi-bin/nph-sesame",
    "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame",
    "https://vizier.idia.ac.za/viz-bin/nph-sesame",
)


def sesame_resolve(name: str, *, timeout: float = 10.0,
                   status_cb: Optional[Callable[[str], None]] = None):
    """
    Resolve an object name to (ra_deg, dec_deg, url) via Sesame, racing the
    mirrors. Returns None when no mirror resolved it; `sesame_resolve.last`
    then says whether that was "not found" (a mirror answered) or "no mirror
    answered".
    """
    q = urllib.parse.quote((name or "").strip())
    sesame_resolve.last = ""
    if not q:
        return None
    order: list = [SESAME_URLS[0]]
    with _lock:
        vz = _state["vz_server"] or ""
    if vz:
        order.append(f"{_vizier_scheme(vz)}://{vz}/viz-bin/nph-sesame")   # known-alive host
    for url in SESAME_URLS[1:]:
        if url not in order:
            order.append(url)
    found: dict = {}
    answered: list = []

    def _check(base: str) -> Tuple[bool, str]:
        code, body, err = _fetch_with_hard_deadline(f"{base}/-ox/SNV?{q}", timeout)
        if err is not None or code is None:
            return False, f"{base}: {type(err).__name__ if err else 'no response'}"
        low = (body or "").lower()
        if code >= 400 or "<html" in low:
            return False, f"{base}: HTTP {code}"
        answered.append(base)
        try:
            ra = float(body.split("<jradeg>", 1)[1].split("</jradeg>", 1)[0])
            dec = float(body.split("<jdedeg>", 1)[1].split("</jdedeg>", 1)[0])
        except (IndexError, ValueError):
            return False, f"{base}: name not resolved"
        found[base] = (ra, dec)
        return True, f"{base}: resolved"

    ok, url, _detail, details = _race_mirrors(order, _check, timeout,
                                              status_cb=status_cb)
    if ok and url in found:
        return found[url][0], found[url][1], url
    sesame_resolve.last = ("not found" if answered else
                           "no Sesame mirror answered: " + " | ".join(details))
    return None


sesame_resolve.last = ""
