#saspro/minorbodycatalog.py
from __future__ import annotations

"""
Minor body (asteroid + comet) catalog helper for SASpro.

Responsibilities:
- Fetch JSON manifest from GitHub (saspro-minorbody-data repo).
- Download / update the SQLite minor body database from the release asset.
- Provide a small API for querying the DB and (optionally) computing
  RA/Dec for a subset of objects via Skyfield.

This module is intentionally independent of Qt; higher-level UI / QSettings
integration should live in the main app code and call into this helper.
"""
import json
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Tuple

import numpy as np

# Astropy imports for local Kepler propagation
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)
import astropy.units as u

# Skyfield imports kept for backward compatibility but no longer used
# for position computation — can be removed entirely if desired
try:
    from skyfield.api import load as sf_load
    from skyfield.data import mpc as sf_mpc
    from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
except Exception:
    sf_load = None
    sf_mpc  = None
    GM_SUN  = None
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw GitHub URL to the manifest in your data repo
MANIFEST_URL = (
    "https://raw.githubusercontent.com/setiastro/"
    "saspro-minorbody-data/main/saspro_minor_bodies_manifest.json"
)
# Human-friendly page users can open in a browser to grab the DB manually
# (works around macOS Python SSL: CERTIFICATE_VERIFY_FAILED issues).
REPO_URL          = "https://github.com/setiastro/saspro-minorbody-data"
RELEASES_PAGE_URL = "https://github.com/setiastro/saspro-minorbody-data/releases/latest"
# Default filenames (as defined by the manifest you showed)
DEFAULT_DB_BASENAME = "saspro_minor_bodies.sqlite"
DEFAULT_MANIFEST_BASENAME = "saspro_minor_bodies_manifest.json"

# Heliocentric gravitational parameter in AU^3 / day^2, from the Gaussian
# gravitational constant k = 0.01720209895 (mu = k^2).
MU_SUN_AU3_PER_DAY2 = 0.01720209895 ** 2

# How close to 1.0 the eccentricity has to be before the elliptical and
# hyperbolic formulations both go numerically soft and Barker's parabolic
# equation is the well-conditioned choice. At e = 1 +/- 1e-8 the semimajor
# axis q/(1-e) exceeds 1e8 AU and n underflows, so neither branch is usable.
_PARABOLIC_TOL = 1e-8


def _solve_kepler_hyperbolic(M: float, e: float,
                             tol: float = 1e-11, max_iter: int = 100) -> float:
    """
    Solve the hyperbolic Kepler equation  M = e*sinh(H) - H  for H.

    Newton from a starting guess that stays sane for large |M|: using
    H0 = M directly diverges badly once sinh(M) overflows the step, so
    seed with an asinh-based estimate instead.
    """
    if M == 0.0:
        return 0.0
    sign = 1.0 if M > 0 else -1.0
    Ma = abs(M)
    # Good seed: for large M, e*sinh(H) ~ M  =>  H ~ asinh(M/e)
    H = np.arcsinh(Ma / e) if Ma > 6.0 else Ma / max(e - 1.0, 1e-6)
    if not np.isfinite(H):
        H = np.arcsinh(Ma / e)
    for _ in range(max_iter):
        f = e * np.sinh(H) - H - Ma
        fp = e * np.cosh(H) - 1.0
        if fp == 0.0 or not np.isfinite(fp):
            break
        dH = f / fp
        H -= dH
        if abs(dH) < tol:
            break
    return sign * H


def _conic_r_nu(dt_days: float, q: float, e: float):
    """
    Heliocentric distance and true anomaly at `dt_days` after perihelion,
    for ANY conic section.

    This exists because the catalog's comet half is dominated by
    near-parabolic long-period orbits, which have no semimajor axis at
    all. The previous elliptical-only path could not express them: fed
    e >= 1 it produced a = q/(1-e) <= 0 and a Newton iteration on the
    elliptical Kepler equation that converges to a plausible-looking but
    meaningless answer. Dispatching on conic type is the only way to get
    a correct position for those objects.

    Returns (r_au, nu_rad).
    """
    mu = MU_SUN_AU3_PER_DAY2

    # ---- Parabolic: Barker's equation -------------------------------
    if abs(e - 1.0) <= _PARABOLIC_TOL:
        # D + D^3/3 = sqrt(mu / (2 q^3)) * dt, solved in closed form by
        # Cardano:  D^3 + 3D - 3M = 0  =>  D = B - 1/B.
        M_bar = np.sqrt(mu / (2.0 * q ** 3)) * dt_days
        A = 1.5 * M_bar
        B = np.cbrt(A + np.sqrt(A * A + 1.0))
        D = B - 1.0 / B
        r = q * (1.0 + D * D)
        nu = 2.0 * np.arctan(D)
        return float(r), float(nu)

    # ---- Hyperbolic -------------------------------------------------
    if e > 1.0:
        a_abs = q / (e - 1.0)                 # |a|
        M = np.sqrt(mu / a_abs ** 3) * dt_days
        H = _solve_kepler_hyperbolic(float(M), float(e))
        r = a_abs * (e * np.cosh(H) - 1.0)
        nu = 2.0 * np.arctan2(
            np.sqrt(e + 1.0) * np.tanh(H / 2.0), np.sqrt(e - 1.0))
        return float(r), float(nu)

    # ---- Elliptical -------------------------------------------------
    a = q / (1.0 - e)
    M = np.sqrt(mu / a ** 3) * dt_days
    M = float(np.mod(M, 2.0 * np.pi))
    E = _solve_kepler(M, float(e))
    r = a * (1.0 - e * np.cos(E))
    nu = np.arctan2(np.sqrt(1.0 - e * e) * np.sin(E), np.cos(E) - e)
    return float(r), float(nu)


def _solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for E (eccentric anomaly).
    Uses Newton-Raphson iteration.
    """
    # Good initial guess
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _ymd_to_jd(year, month, day) -> float:
    """
    Calendar date -> Julian Date at 0h, with fractional-day support.

    Shared by the packed-epoch decoder and the comet perihelion-time
    derivation so both land on exactly the same scale; a half-day
    disagreement between them would show up as a systematic position
    offset that looks like bad orbital elements.
    """
    try:
        di = int(np.floor(float(day)))
        frac = float(day) - di
        y = int(year)
        m = int(month)
        if m <= 2:
            y -= 1
            m += 12
        A = int(y // 100)
        B = 2 - A + int(A // 4)
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + di + B - 1524.5
        return float(jd) + frac
    except Exception:
        return float("nan")


def _decode_packed_epoch(packed: str) -> float:
    """
    Decode MPC packed epoch string (e.g. 'K245N') to Julian date.
    Format: C YY MM DD where C is century letter, MM/DD are base-36 encoded.
    """
    if not packed or len(packed) < 5:
        # fallback to J2000
        return 2451545.0

    def n(c):
        return ord(c) - (48 if c.isdigit() else 55)

    try:
        century = 100 * n(packed[0]) + int(packed[1:3])
        month   = n(packed[3])
        day     = n(packed[4])

        # Julian day from calendar date (same formula as Skyfield uses)
        y = century
        m = month
        d = day
        if m <= 2:
            y -= 1
            m += 12
        A = int(y / 100)
        B = 2 - A + int(A / 4)
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
        return jd
    except Exception:
        return 2451545.0
    
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MinorBodyManifest:
    schema_version: int
    version: str
    generated_utc: str
    download_url: str
    download_filename: str
    counts_asteroids: int
    counts_comets: int
    raw: Dict[str, Any]


# ---------------------------------------------------------------------------
# Network helpers (urllib only, to keep deps minimal)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """Fetch JSON from a URL using urllib."""
    req = urllib.request.Request(url, headers={"User-Agent": "SetiAstroSuitePro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _http_download_binary(url: str, dest: Path, chunk_size: int = 65536, timeout: float = 30.0) -> None:
    """Download a binary file from URL to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "SetiAstroSuitePro/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f_out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
        tmp.replace(dest)


# ---------------------------------------------------------------------------
# Manifest + DB management
# ---------------------------------------------------------------------------

def fetch_remote_manifest(url: str = MANIFEST_URL) -> MinorBodyManifest:
    """Fetch the remote manifest from GitHub and parse it."""
    data = _http_get_json(url)
    # Defensive parsing: tolerate missing bits gracefully
    dl = data.get("download", {})
    counts = data.get("counts", {})

    return MinorBodyManifest(
        schema_version=int(data.get("schema_version", 1)),
        version=str(data.get("version", "unknown")),
        generated_utc=str(data.get("generated_utc", "")),
        download_url=str(dl.get("url", "")),
        download_filename=str(dl.get("filename", DEFAULT_DB_BASENAME)),
        counts_asteroids=int(counts.get("asteroids", 0)),
        counts_comets=int(counts.get("comets", 0)),
        raw=data,
    )


def load_local_manifest(path: Path) -> Optional[MinorBodyManifest]:
    """Load a previously saved local manifest (if it exists)."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dl = data.get("download", {})
        counts = data.get("counts", {})
        return MinorBodyManifest(
            schema_version=int(data.get("schema_version", 1)),
            version=str(data.get("version", "unknown")),
            generated_utc=str(data.get("generated_utc", "")),
            download_url=str(dl.get("url", "")),
            download_filename=str(dl.get("filename", DEFAULT_DB_BASENAME)),
            counts_asteroids=int(counts.get("asteroids", 0)),
            counts_comets=int(counts.get("comets", 0)),
            raw=data,
        )
    except Exception:
        return None


def save_local_manifest(path: Path, manifest: MinorBodyManifest) -> None:
    """Write manifest JSON to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.raw, f, indent=2)


def ensure_minor_body_db(
    data_dir: Path,
    manifest_url: str = MANIFEST_URL,
    force_refresh: bool = False,
) -> Tuple[Path, MinorBodyManifest]:
    """
    Ensure that the minor body SQLite DB exists (and is up to date).

    Parameters
    ----------
    data_dir : Path
        Directory where the DB + local manifest should live.
    manifest_url : str
        URL to the JSON manifest in the saspro-minorbody-data repo.
    force_refresh : bool
        If True, always re-download the DB even if version matches.

    Returns
    -------
    db_path : Path
        Path to the local SQLite DB.
    manifest : MinorBodyManifest
        Parsed manifest object for the local version.

    Notes
    -----
    This function is network-only; any UI / progress should wrap it and
    catch exceptions.
    """
    data_dir = data_dir.resolve()
    local_manifest_path = data_dir / DEFAULT_MANIFEST_BASENAME

    remote = fetch_remote_manifest(manifest_url)
    db_path = data_dir / remote.download_filename

    local = load_local_manifest(local_manifest_path)

    needs_download = force_refresh

    if not needs_download:
        if local is None:
            needs_download = True
        elif local.version != remote.version:
            needs_download = True
        elif not db_path.is_file():
            needs_download = True

    if needs_download:
        if not remote.download_url:
            raise RuntimeError("Manifest does not contain a download URL for the DB.")
        _http_download_binary(remote.download_url, db_path)
        save_local_manifest(local_manifest_path, remote)
        manifest = remote
    else:
        # local manifest + DB are assumed valid
        manifest = local

    return db_path, manifest

def install_local_db_file(
    src_path: Path,
    data_dir: Path,
    manifest_url: str = MANIFEST_URL,
) -> Tuple[Path, MinorBodyManifest]:
    """
    Install a user-provided minor-body SQLite DB into ``data_dir``.

    Intended as an escape hatch when the automatic downloader can't
    reach GitHub — most commonly the macOS Python
    ``SSL: CERTIFICATE_VERIFY_FAILED`` error. The user downloads
    ``saspro_minor_bodies.sqlite`` from the releases page in a browser
    (which uses the system trust store), then points us at the file.

    We validate that the file really is a SQLite DB with the expected
    ``asteroids`` and ``comets`` tables, copy it into ``data_dir``,
    and write a local manifest. If the remote manifest is reachable we
    use its version/counts; otherwise we synthesize a minimal one from
    the DB itself so the UI still has something sensible to show.

    Returns ``(db_path, manifest)``, matching ``ensure_minor_body_db``.
    """
    import shutil

    src = Path(src_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    # ---- Validate: readable SQLite with expected tables ----
    try:
        uri = f"file:{src.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name IN ('asteroids','comets')"
            )
            tables = {row[0] for row in cur.fetchall()}
            missing = {"asteroids", "comets"} - tables
            if missing:
                raise RuntimeError(
                    "Selected file does not look like a minor-body database "
                    f"(missing table(s): {', '.join(sorted(missing))})."
                )
            cur.execute("SELECT COUNT(*) FROM asteroids")
            n_ast = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM comets")
            n_com = int(cur.fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Not a valid SQLite database: {exc}") from exc

    # ---- Best-effort remote manifest (may fail with the same SSL error;
    #      that's fine, we'll synthesize one below) ----
    remote_manifest: Optional[MinorBodyManifest] = None
    try:
        remote_manifest = fetch_remote_manifest(manifest_url)
    except Exception:
        remote_manifest = None

    dest_name = (
        remote_manifest.download_filename
        if (remote_manifest and remote_manifest.download_filename)
        else DEFAULT_DB_BASENAME
    )

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / dest_name

    # ---- Copy into place atomically ----
    tmp = db_path.with_suffix(db_path.suffix + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(db_path)

    # ---- Save (or synthesize) the local manifest ----
    local_manifest_path = data_dir / DEFAULT_MANIFEST_BASENAME
    if remote_manifest is not None:
        save_local_manifest(local_manifest_path, remote_manifest)
        return db_path, remote_manifest

    synthesized_raw = {
        "schema_version": 1,
        "version": "manual-install",
        "generated_utc": "",
        "download": {"url": "", "filename": dest_name},
        "counts": {"asteroids": n_ast, "comets": n_com},
    }
    manifest = MinorBodyManifest(
        schema_version=1,
        version="manual-install",
        generated_utc="",
        download_url="",
        download_filename=dest_name,
        counts_asteroids=n_ast,
        counts_comets=n_com,
        raw=synthesized_raw,
    )
    save_local_manifest(local_manifest_path, manifest)
    return db_path, manifest

# ---------------------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------------------

class MinorBodyCatalog:
    """
    Thin wrapper around the saspro minor body SQLite DB.

    Usage:
        cat = MinorBodyCatalog(db_path)
        print(cat.counts)
        df = cat.get_bright_asteroids(H_max=20.0, limit=50000)
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Minor body DB not found: {self.db_path}")
        # We keep connection lazy; open on demand
        self._conn: Optional[sqlite3.Connection] = None

    # ---- Connection management -------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # Use read-only URI mode when possible
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _has_column(self, table: str, column: str) -> bool:
        """
        Does `table` carry `column`? Used to light up newer schema
        features without breaking an older database that predates them —
        users upgrade the app and the catalog independently, so neither
        can assume the other has moved.
        """
        key = (table, column)
        cache = getattr(self, "_col_cache", None)
        if cache is None:
            cache = self._col_cache = {}
        if key in cache:
            return cache[key]
        try:
            cur = self._get_conn().cursor()
            cur.execute(f"PRAGMA table_info({table})")
            cols = {row[1] for row in cur.fetchall()}
            found = column in cols
        except sqlite3.Error:
            found = False
        cache[key] = found
        return found

    # ---- Introspection ---------------------------------------------------------

    @property
    def counts(self) -> Dict[str, int]:
        """Return row counts for 'asteroids' and 'comets' tables."""
        conn = self._get_conn()
        cur = conn.cursor()
        result = {}
        for table in ("asteroids", "comets"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                result[table] = int(cur.fetchone()[0])
            except sqlite3.Error:
                result[table] = 0
        return result

    # ---- Simple queries --------------------------------------------------------

    def get_bright_asteroids(
        self,
        H_max: float = 20.0,
        limit: Optional[int] = 100000,
    ):
        """
        Return a pandas DataFrame of relatively bright asteroids.

        We treat magnitude_H as numeric even though the column is TEXT in the
        current schema, by casting it to REAL for filtering and ordering.
        """
        import pandas as pd

        conn = self._get_conn()

        # Skip rows the builder already determined can never yield a
        # position. Doing it in SQL means an incomplete row never occupies
        # a slot in the LIMIT — previously a run capped at 50,000 could
        # spend part of its budget on rows guaranteed to be discarded.
        # Guarded by a column check so an older database still works.
        extra = ""
        if self._has_column("asteroids", "two_body_ok"):
            extra = " AND two_body_ok = 1"

        sql = f"""
            SELECT *
            FROM asteroids
            WHERE CAST(magnitude_H AS REAL) <= ?{extra}
            ORDER BY CAST(magnitude_H AS REAL) ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            df = pd.read_sql_query(sql, conn, params=(H_max, limit))
        else:
            df = pd.read_sql_query(sql, conn, params=(H_max,))
        return df

    def get_bright_comets(
        self,
        H_max: float = 15.0,
        limit: Optional[int] = 5000,
    ):
        """
        Bright comets by magnitude.

        Deliberately NOT filtered on two_body_ok. The builder clears that
        flag for e >= 1 because those orbits have no semimajor axis and so
        no mean-anomaly element set — but compute_positions_astropy now
        propagates them from (q, T_perihelion) via Barker's equation and
        the hyperbolic Kepler equation. Filtering here would throw away
        exactly the long-period comets that support was added for.
        """
        import pandas as pd

        conn = self._get_conn()
        # Try to detect magnitude column
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(comets)")
        cols = {row[1] for row in cur.fetchall()}

        mag_col = None
        # Prefer g, then k, then any H-style absolute mag
        for candidate in ("magnitude_g", "magnitude_k", "absolute_magnitude", "magnitude_H", "H"):
            if candidate in cols:
                mag_col = candidate
                break

        if mag_col is None:
            # No magnitude info we recognize – just return a limited subset
            sql = "SELECT * FROM comets"
            if limit is not None:
                sql += " LIMIT ?"
                return pd.read_sql_query(sql, conn, params=(limit,))
            return pd.read_sql_query(sql, conn)

        sql = f"""
            SELECT *
            FROM comets
            WHERE {mag_col} <= ?
            ORDER BY {mag_col} ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            df = pd.read_sql_query(sql, conn, params=(H_max, limit))
        else:
            df = pd.read_sql_query(sql, conn, params=(H_max,))
        return df

    def get_asteroids_by_designation(
        self,
        designations: Iterable[str],
    ):
        """
        Fetch asteroid rows for one or more MPC designations.

        Parameters
        ----------
        designations : iterable of str
            MPC designations like "1 Ceres", "433 Eros", etc.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        designations = list(designations)
        if not designations:
            import pandas as pd  # type: ignore
            return pd.DataFrame()

        conn = self._get_conn()
        placeholders = ",".join("?" for _ in designations)
        sql = f"""
            SELECT *
            FROM asteroids
            WHERE designation IN ({placeholders})
        """
        return pd.read_sql_query(sql, conn, params=designations)

    # -----------------------------------------------------------------------
    # Ephemeris / position scaffold (astropy-based, for small subsets)
    # -----------------------------------------------------------------------
    def compute_positions_astropy(
        self,
        asteroid_rows,
        jd: float,
        ephemeris_path=None,
        topocentric=None,
        progress_cb=None,
        debug: bool = False,
    ):
        """
        Compute RA/Dec using astropy + local orbital elements only.
        No network calls, no Skyfield frame issues.
        """
        import pandas as pd
        import numpy as np
        from astropy.time import Time
        from astropy.coordinates import (
            SkyCoord, GCRS, HeliocentricEclipticIAU76,
            get_body_barycentric_posvel, CartesianRepresentation
        )
        import astropy.units as u

        if isinstance(asteroid_rows, pd.DataFrame):
            df = asteroid_rows.copy()
        else:
            df = pd.DataFrame(list(asteroid_rows))

        # ------------------------------------------------------------------
        # Element validation, conic-aware.
        #
        # Orientation (i, node, argp) plus eccentricity is required for every
        # orbit. Beyond that there are two valid element sets and a row
        # needs only ONE of them:
        #
        #   A) mean-anomaly form   — a, n, M, epoch   (MPCORB asteroids, and
        #                            elliptical comets converted at build time)
        #   B) perihelion form     — q, T_perihelion  (AllCometEls, and the
        #                            only form that can express e >= 1)
        #
        # The old code demanded set A unconditionally, so every parabolic
        # and hyperbolic comet was discarded here before the propagator ever
        # saw it — on top of the fact that comets never carried set A at all.
        # ------------------------------------------------------------------
        orientation = [
            "argument_of_perihelion_degrees",
            "longitude_of_ascending_node_degrees",
            "inclination_degrees",
            "eccentricity",
        ]
        set_a = ["semimajor_axis_au", "mean_anomaly_degrees",
                 "mean_daily_motion_degrees"]
        set_b = ["perihelion_distance_au"]
        maybe_numeric = (orientation + set_a + set_b + ["epoch_jd",
                         "perihelion_year", "perihelion_month", "perihelion_day"])
        for col in maybe_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)

        have_orient = [c for c in orientation if c in df.columns]
        if len(have_orient) < len(orientation):
            missing = set(orientation) - set(have_orient)
            if debug:
                print(f"[MinorBodies] rows lack orientation columns {missing}")
            return []
        ok_orient = df[have_orient].notna().all(axis=1)

        ok_a = pd.Series(False, index=df.index)
        if all(c in df.columns for c in set_a):
            ok_a = df[set_a].notna().all(axis=1)

        # Perihelion time: prefer a stored JD, else rebuild it from the
        # year/month/day triple AllCometEls provides. Deriving it here means
        # an existing database works unchanged — no rebuild needed to get
        # parabolic and hyperbolic comets working.
        if "perihelion_jd" not in df.columns:
            if all(c in df.columns for c in
                   ("perihelion_year", "perihelion_month", "perihelion_day")):
                df["perihelion_jd"] = [
                    _ymd_to_jd(y, m, d)
                    if (pd.notna(y) and pd.notna(m) and pd.notna(d)) else np.nan
                    for y, m, d in zip(df["perihelion_year"],
                                       df["perihelion_month"],
                                       df["perihelion_day"])
                ]
            else:
                df["perihelion_jd"] = np.nan
        else:
            df["perihelion_jd"] = pd.to_numeric(df["perihelion_jd"], errors="coerce")

        ok_b = pd.Series(False, index=df.index)
        if "perihelion_distance_au" in df.columns:
            ok_b = (df["perihelion_distance_au"].notna()
                    & (df["perihelion_distance_au"] > 0)
                    & df["perihelion_jd"].notna())

        keep = ok_orient & (ok_a | ok_b)
        df = df[keep].copy()
        df["_has_set_a"] = ok_a[keep]
        df["_has_set_b"] = ok_b[keep]

        if debug:
            print(f"[MinorBodies] rows after cleaning: {len(df)} (was {before})"
                  f"  [mean-anomaly: {int(df['_has_set_a'].sum())}, "
                  f"perihelion-only: {int((~df['_has_set_a'] & df['_has_set_b']).sum())}]")

        if df.empty:
            return []

        # Observation time
        t_obs = Time(jd, format="jd", scale="tt")
        if debug:
            print(f"[MinorBodies] obs time: {t_obs.iso}  JD={jd:.6f}")

        jd_obs = float(jd)

        # Earth barycentric position at observation time (ICRS, AU)
        from astropy.coordinates import solar_system_ephemeris, EarthLocation, GCRS
        with solar_system_ephemeris.set("builtin"):
            earth_bary, _ = get_body_barycentric_posvel("earth", t_obs)
        earth_xyz = np.array([
            earth_bary.x.to(u.au).value,
            earth_bary.y.to(u.au).value,
            earth_bary.z.to(u.au).value,
        ])

        # Sun barycentric position AND velocity. The velocity is used to
        # shift the Sun to the light-retarded time below; it is small but
        # not negligible (~0.8" at 1.5 AU if ignored).
        with solar_system_ephemeris.set("builtin"):
            sun_bary, sun_bary_v = get_body_barycentric_posvel("sun", t_obs)
        sun_xyz = np.array([
            sun_bary.x.to(u.au).value,
            sun_bary.y.to(u.au).value,
            sun_bary.z.to(u.au).value,
        ])
        try:
            sun_vel_xyz = np.array([
                sun_bary_v.x.to(u.au / u.day).value,
                sun_bary_v.y.to(u.au / u.day).value,
                sun_bary_v.z.to(u.au / u.day).value,
            ])
        except Exception:
            sun_vel_xyz = np.zeros(3)

        # ------------------------------------------------------------------
        # Observer position: TOPOCENTRIC when a site is supplied.
        #
        # Previously this parameter was accepted and silently ignored — every
        # position was computed from the geocentre. The geocentre-to-observer
        # vector is one Earth radius, which subtends ~4.4" on a main-belt
        # asteroid at 2 AU and ~18" at 0.5 AU. For anyone trying to confirm
        # a specific object against a plate solution, that alone is enough
        # to make the marker miss.
        # ------------------------------------------------------------------
        obs_xyz = earth_xyz
        if topocentric is not None:
            try:
                lat_deg, lon_deg, elev_m = (list(topocentric) + [0.0, 0.0, 0.0])[:3]
                loc = EarthLocation(lat=float(lat_deg) * u.deg,
                                    lon=float(lon_deg) * u.deg,
                                    height=float(elev_m) * u.m)
                gc = loc.get_itrs(obstime=t_obs).transform_to(
                    GCRS(obstime=t_obs)).cartesian
                off = np.array([
                    gc.x.to(u.au).value,
                    gc.y.to(u.au).value,
                    gc.z.to(u.au).value,
                ])
                obs_xyz = earth_xyz + off
                if debug:
                    print(f"[MinorBodies] topocentric: lat={lat_deg} lon={lon_deg} "
                          f"elev={elev_m}m, offset={np.linalg.norm(off)*1.495978707e8:.0f} km")
            except Exception as exc:
                print(f"[MinorBodies] topocentric setup failed, using geocentre: {exc!r}")
                obs_xyz = earth_xyz
        elif debug:
            print("[MinorBodies] no site supplied — geocentric positions "
                  "(up to ~18\" parallax error on close objects)")

        # Speed of light, AU per day.
        _C_AU_PER_DAY = 173.1446326846693

        if debug:
            print(f"[MinorBodies] Earth ICRS (AU): {earth_xyz}")

        results = []
        total = len(df)
        ok = 0
        failed = 0

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if progress_cb is not None:
                try:
                    cont = progress_cb(i, total)
                except Exception:
                    cont = True
                if cont is False:
                    if debug:
                        print("[MinorBodies] aborted by progress_cb")
                    break

            try:
                # ── 1) Epoch ──────────────────────────────────────────────────
                # Prefer a precomputed numeric epoch when the database
                # carries one (the builder stores epoch_jd), and only fall
                # back to decoding the packed string. Decoding five
                # characters per row per light-time iteration across tens of
                # thousands of rows is pure overhead when the value is
                # already sitting in the table.
                epoch_jd = row.get("epoch_jd", None)
                try:
                    epoch_jd = float(epoch_jd)
                    if not np.isfinite(epoch_jd):
                        raise ValueError
                except Exception:
                    epoch_jd = _decode_packed_epoch(
                        str(row.get("epoch_packed", "")).strip())

                # ── 2) Orbital elements ───────────────────────────────────────
                e   = float(row["eccentricity"])
                inc = np.radians(float(row["inclination_degrees"]))
                Om  = np.radians(float(row["longitude_of_ascending_node_degrees"]))
                om  = np.radians(float(row["argument_of_perihelion_degrees"]))

                # Pick the propagation form from the elements the row
                # actually carries.
                #
                # Mean-anomaly rows are deliberately NOT reduced to
                # (q, T_perihelion). That conversion needs T = epoch - M/n,
                # and MPC's published mean_daily_motion_degrees differs from
                # sqrt(mu/a^3) in the fourth decimal. Harmless on its own,
                # but M/n is ~800 days for a typical main-belt object, so the
                # mismatch lands as ~0.1 degree of mean anomaly — a few
                # hundred arcsec on sky. Use each row's own n and M directly.
                use_mean_anomaly = bool(row.get("_has_set_a", False)) and e < 1.0
                a = M0_deg = n_deg = None
                q_peri = t_peri = None

                if use_mean_anomaly:
                    a = float(row["semimajor_axis_au"])
                    M0_deg = float(row["mean_anomaly_degrees"])
                    n_deg = float(row["mean_daily_motion_degrees"])
                    if not (np.isfinite(a) and a > 0 and np.isfinite(M0_deg)
                            and np.isfinite(n_deg) and n_deg != 0.0):
                        use_mean_anomaly = False

                if not use_mean_anomaly:
                    q_peri = float(row.get("perihelion_distance_au", np.nan))
                    t_peri = float(row.get("perihelion_jd", np.nan))
                    if not (np.isfinite(q_peri) and q_peri > 0
                            and np.isfinite(t_peri) and np.isfinite(e)):
                        raise ValueError("degenerate conic elements")

                # ── 3) Constant rotation terms for this object ────────────────
                cos_Om = np.cos(Om); sin_Om = np.sin(Om)
                cos_om = np.cos(om); sin_om = np.sin(om)
                cos_i  = np.cos(inc); sin_i = np.sin(inc)

                # Standard rotation matrix: orbital → ecliptic
                Xx =  cos_Om*cos_om - sin_Om*sin_om*cos_i
                Xy = -cos_Om*sin_om - sin_Om*cos_om*cos_i
                Yx =  sin_Om*cos_om + cos_Om*sin_om*cos_i
                Yy = -sin_Om*sin_om + cos_Om*cos_om*cos_i
                Zx =  sin_om*sin_i
                Zy =  cos_om*sin_i

                eps = np.radians(23.439291111)   # obliquity J2000.0
                cos_e = np.cos(eps); sin_e = np.sin(eps)

                if use_mean_anomaly:
                    def _r_nu(jd_at, _a=a, _e=e, _M0=M0_deg,
                              _n=n_deg, _ej=epoch_jd):
                        """Classic MPCORB path — unchanged numerics."""
                        M = np.radians(_M0 + _n * (jd_at - _ej)) % (2.0 * np.pi)
                        E = _solve_kepler(M, _e)
                        cos_E = np.cos(E); sin_E = np.sin(E)
                        return (_a * (1.0 - _e * cos_E),
                                np.arctan2(np.sqrt(1.0 - _e * _e) * sin_E,
                                           cos_E - _e))
                else:
                    def _r_nu(jd_at, _q=q_peri, _e=e, _tp=t_peri):
                        """Perihelion path — any conic, including e >= 1."""
                        return _conic_r_nu(jd_at - _tp, _q, _e)

                def _helio_icrs(jd_at):
                    """Heliocentric ICRS position (AU) at a given JD.

                    Factored out so the light-time loop can evaluate it at
                    the retarded time without duplicating the Kepler maths.
                    """
                    r, nu = _r_nu(jd_at)
                    x_orb = r * np.cos(nu)
                    y_orb = r * np.sin(nu)
                    x_ecl = Xx*x_orb + Xy*y_orb
                    y_ecl = Yx*x_orb + Yy*y_orb
                    z_ecl = Zx*x_orb + Zy*y_orb
                    return np.array([
                        x_ecl,
                        cos_e*y_ecl - sin_e*z_ecl,
                        sin_e*y_ecl + cos_e*z_ecl,
                    ])

                # ── 4) Light-time iteration ───────────────────────────────────
                # We see the object where it WAS when the light left it. For
                # a main-belt asteroid that is 12-17 minutes earlier, during
                # which it moves 6-9 arcsec across the sky. Omitting this was
                # the largest systematic in this routine — it put every
                # prediction consistently ahead of the real object along its
                # direction of motion.
                #
                # Solve  tau = |r_ast(t - tau) - r_obs(t)| / c  by iteration.
                # Converges to well under a milliarcsecond in two passes for
                # anything that isn't a sungrazer.
                tau = 0.0
                d_vec = None
                dist = 0.0
                for _lt in range(3):
                    p_helio = _helio_icrs(jd_obs - tau)
                    # Sun's barycentric position at the retarded time, to
                    # first order in its barycentric velocity.
                    p_bary = p_helio + (sun_xyz - sun_vel_xyz * tau)
                    d_vec = p_bary - obs_xyz
                    dist = float(np.sqrt(d_vec.dot(d_vec)))
                    tau_new = dist / _C_AU_PER_DAY
                    if abs(tau_new - tau) < 1e-10:
                        tau = tau_new
                        break
                    tau = tau_new

                dx, dy, dz = float(d_vec[0]), float(d_vec[1]), float(d_vec[2])

                # ── 5) ICRS RA/Dec — astrometric place ────────────────────────
                # Light-time corrected, NOT aberration corrected. That is the
                # right quantity to compare against a WCS fitted to catalog
                # star positions, which are themselves un-aberrated.
                sc = SkyCoord(
                    x=dx*u.au, y=dy*u.au, z=dz*u.au,
                    representation_type="cartesian",
                    frame="icrs",
                )
                sc_sph = sc.represent_as("unitspherical")

                results.append({
                    "designation": row.get("designation", ""),
                    "ra_deg":      float(sc_sph.lon.deg),
                    "dec_deg":     float(sc_sph.lat.deg),
                    "distance_au": float(dist),
                    "light_time_s": float(tau * 86400.0),
                })
                ok += 1

            except Exception as exc:
                failed += 1
                if debug and failed <= 10:
                    print(f"[MinorBodies] FAILED '{row.get('designation','')}': {repr(exc)}")

        if debug:
            print(f"[MinorBodies] total={total}, ok={ok}, failed={failed}")

        return results